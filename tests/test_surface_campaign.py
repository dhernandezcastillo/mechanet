"""A campaign built on a picked surface: `mechanet generate --on`.

The builder part runs without calculations (RuO2(110): two open metals m0038 and m0046, two
bridging O a0016 and a0019). The workflow part relaxes a coverage campaign with the synthetic
VASP, lets the relaxed surface differ from its start, and builds the mechanism on it: finished
calculations are copied byte for byte and new intermediates start from the relaxed surface.
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from support import FAKE_VASP, write_potcars

from mechanet.builder import Builder
from mechanet.cli import main
from mechanet.families.coverage import count
from mechanet.fileio import file_hash, read_json, write_json
from mechanet.plugins import default_registry
from mechanet.project import coverage_count, generate, load_project, resolve_config, validate_project
from mechanet.runner import run_state
from mechanet.structure import minimum_image, read_poscar, write_poscar
from mechanet.surfaces import pick, relaxed_surface

ROOT = Path(__file__).resolve().parents[1]
RUO2 = ROOT / "examples" / "ruo2_110"
TWO_OH = {"m0038": "OH", "m0046": "OH"}


def run(*args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main([str(a) for a in args])
    return code, out.getvalue(), err.getvalue()


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {"MECHANET_CONFIG_DIR": str(self.root / "config")})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def project_copy(self, name="p", **changes):
        shutil.copytree(RUO2, self.root / name, ignore=shutil.ignore_patterns("campaign"))
        path = self.root / name / "project.yaml"
        data = yaml.safe_load(path.read_text())
        for key, value in changes.items():
            if key == "scope":
                data["scope"].update(value)
            else:
                data[key] = value
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return self.root / name

    def builder(self, project, name, surface=None, **scope):
        loaded = load_project(project / "project.yaml")
        config = resolve_config(loaded, cluster=False)
        config["scope"].update(scope)
        if surface is not None:
            config["scope"]["surface"] = surface
        return Builder(config, self.root / name, base_dir=loaded["_dir"], registry=default_registry(config["plugins"]))


class SurfaceBuilderTests(TempCase):
    def test_families_write_their_intermediates_on_the_surface(self):
        project = self.project_copy()
        plain = self.builder(project, "plain")
        plain_report = plain.finish(["aem"])
        on = self.builder(project, "on", {"name": "two-OH", "occupancy": dict(TWO_OH)})
        report = on.finish(["aem"])
        # the same cycles, each on the surface: a site the family does not name keeps its OH
        self.assertEqual(report["complete_stoichiometric_cycles"], plain_report["complete_stoichiometric_cycles"])
        self.assertEqual(report["symmetry_unique_cycles"], plain_report["symmetry_unique_cycles"])
        family_nodes = [n for n in on.nodes.values() if n["id"] != "ref"]
        for node in family_nodes:
            states = {**on.native, **node["occupancy"]}
            self.assertIn("OH", (states["m0038"], states["m0046"]), node["id"])
        surface = on.nodes[on.surface_node]
        self.assertEqual(surface["occupancy"], TWO_OH)
        self.assertEqual(surface["family"], "surface")
        self.assertIn("aem", surface["families"])  # *OH of the cycle on one site is the surface itself
        self.assertEqual(on.nodes["ref"]["occupancy"], {})
        network = read_json(self.root / "on" / "network.json")
        self.assertEqual(
            network["surface"],
            {
                "name": "two-OH",
                "node_id": on.surface_node,
                "occupancy": TWO_OH,
                "coupling": None,
                "description": "terminal 2 OH | bridge 2 O",
                "source_campaign": None,
            },
        )
        self.assertEqual(report["surface"]["imported_states"], 0)
        self.assertNotIn("surface", read_json(self.root / "plain" / "network.json"))
        self.assertNotIn("surface", plain_report)

    def test_a_surface_that_breaks_a_translation_loses_that_symmetry(self):
        project = self.project_copy()
        symmetric = self.builder(project, "sym", {"name": "s", "occupancy": dict(TWO_OH)})
        self.assertEqual(len(symmetric.translations), len(self.builder(project, "plain").translations))
        lopsided = self.builder(project, "lop", {"name": "l", "occupancy": {"m0038": "OH"}})
        self.assertEqual(lopsided.translations, [])
        report = lopsided.finish(["aem"])
        self.assertEqual(report["symmetry_equivalent_nodes"], 0)
        self.assertEqual(report["candidate_nodes"], report["computed_nodes"])

    def test_a_site_set_back_to_the_state_of_the_slab(self):
        """surface_bridge writes OH, O, OOH, OO and the vacancy on a bridge; on a surface where the
        bridges are OH, its O is the slab's own state again, also in the symmetry images."""
        project = self.project_copy()
        base = {"a0016": "OH", "a0019": "OH"}
        on = self.builder(project, "on", {"name": "bridges-OH", "occupancy": base})
        report = on.finish(["surface_bridge"])
        plain = self.builder(project, "plain").finish(["surface_bridge"])
        self.assertEqual(report["complete_stoichiometric_cycles"], plain["complete_stoichiometric_cycles"])
        one_back = on.node({"a0016": "O"}, "surface_bridge", "a0016")
        self.assertEqual({**on.native, **on.nodes[one_back]["occupancy"]}["a0019"], "OH")
        both_back = on._node({"a0016": "O", "a0019": "O"}, "surface_bridge", "x")
        self.assertEqual(both_back, "ref")
        for node in on.nodes.values():
            if node.get("computed_by"):
                self.assertEqual(node["delta_atoms"], on.nodes[node["computed_by"]]["delta_atoms"])

    def test_an_empty_surface_is_the_slab_as_given(self):
        project = self.project_copy()
        plain = self.builder(project, "plain")
        plain.finish(["aem"])
        same = self.builder(project, "same", {"name": "nothing", "occupancy": {"m0038": "empty"}})
        same.finish(["aem"])
        self.assertEqual(same.surface_node, "ref")  # a surface by name, the slab by content
        self.assertEqual(sorted(plain.nodes), sorted(same.nodes))
        self.assertEqual(sorted(plain.states), sorted(same.states))

    def test_a_site_back_in_the_state_of_the_slab_sits_where_the_slab_has_it(self):
        """NiFe LDH: terminal a0065 is OH in the slab and O on the surface, where its oxygen relaxed
        0.2 A towards the metal. With OH again, the oxygen is back at the place its H belongs to."""
        loaded = load_project(ROOT / "examples" / "nife_ldh" / "project.yaml")
        config = resolve_config(loaded, cluster=False)
        plain = Builder(config, self.root / "plain", base_dir=loaded["_dir"], registry=default_registry())
        site = "a0065"
        metal = plain.sites[site]["metal_neighbors"][0]["id"]
        oxygen = plain.seed.positions[plain.seed_index[site]]
        towards = minimum_image(plain.seed.positions[plain.seed_index[metal]] - oxygen, plain.seed.cell)
        relaxed = oxygen + 0.2 * towards / np.linalg.norm(towards)
        config["scope"]["surface"] = {
            "name": "oxo",
            "occupancy": {site: "O"},
            "coupling": None,
            "given": {"positions": {site: relaxed.tolist()}},
        }
        on = Builder(config, self.root / "on", base_dir=loaded["_dir"], registry=default_registry())
        surface = on.state_structures[on.nodes[on.surface_node]["state_ids"][0]]
        self.assertTrue(np.allclose(surface.positions[surface.ids.index(site)], relaxed))  # the surface as given
        node = on.node({site: "OH"}, "aem", site)  # the slab's own state on that site
        self.assertEqual(node, "ref")
        other = on.node({site: "OH", "a0066": "O"}, "aem", "a0066")
        start = on.state_structures[on.nodes[other]["state_ids"][0]]
        self.assertTrue(np.allclose(start.positions[start.ids.index(site)], oxygen))
        hydrogen = plain.sites[site]["hydrogen_neighbors"][0]["id"]
        bond = start.positions[start.ids.index(hydrogen)] - start.positions[start.ids.index(site)]
        self.assertAlmostEqual(
            float(np.linalg.norm(minimum_image(bond, start.cell))),
            plain.sites[site]["hydrogen_neighbors"][0]["distance_A"],
            places=6,
        )

    def test_bad_surfaces_are_refused(self):
        project = self.project_copy()
        with self.assertRaisesRegex(ValueError, "unknown site a9999"):
            self.builder(project, "a", {"name": "x", "occupancy": {"a9999": "O"}})
        with self.assertRaisesRegex((KeyError, ValueError), "Xx"):
            self.builder(project, "b", {"name": "x", "occupancy": {"m0038": "Xx"}})
        with self.assertRaisesRegex(ValueError, "coupling of the surface"):
            self.builder(project, "c", {"name": "x", "occupancy": {"m0038": "O"}, "coupling": ["m0038"]})

    def test_coverage_around_a_surface(self):
        project = self.project_copy()
        scope = {"coverage": {"sites": ["m0038", "m0046"], "states": {"open_metal": ["empty", "OH", "O"]}}}
        plain = count(self.builder(project, "plain", **scope))
        around = self.builder(project, "around", {"name": "two-OH", "occupancy": dict(TWO_OH)}, **scope)
        found = count(around)
        self.assertEqual(found["surfaces"], plain["surfaces"])  # the same nine patterns, counted from the surface
        self.assertEqual(found["states"], {"m0038": ["OH", "empty", "O"], "m0046": ["OH", "empty", "O"]})
        around.finish(["coverage"])
        occupancies = [{**around.native, **n["occupancy"]} for n in around.nodes.values()]
        pairs = {(o["m0038"], o["m0046"]) for o in occupancies}
        self.assertEqual(len(pairs), 9)
        self.assertEqual(len(around.nodes), 9)  # the slab as given is the pattern empty/empty, the surface OH/OH


class PickTests(TempCase):
    def setUp(self):
        super().setUp()
        scope = {"coverage": {"sites": ["m0038", "m0046"], "states": {"open_metal": ["empty", "OH"]}}}
        self.project = self.project_copy(scope=scope, surfaces={"two-OH": dict(TWO_OH), "as-given": {"m0038": "empty"}})
        self.loaded = load_project(self.project / "project.yaml")
        self.config = resolve_config(self.loaded, cluster=False)
        self.registry = default_registry(self.config["plugins"])

    def pick(self, on, source=None):
        return pick(self.loaded, self.config, self.registry, on, source)

    def test_a_surface_of_the_project_file(self):
        surface = self.pick("two-OH")
        self.assertEqual((surface["name"], surface["occupancy"], surface["coupling"]), ("two-OH", TWO_OH, None))
        self.assertNotIn("source", surface)  # no coverage campaign yet
        self.assertEqual(self.pick("as-given")["occupancy"], {})

    def test_an_intermediate_of_the_coverage_campaign(self):
        generate(self.project / "project.yaml", coverage=True)
        network = read_json(self.project / "coverage" / "network.json")
        both = next(n for n in network["nodes"] if n["occupancy"] == TWO_OH)
        surface = self.pick(both["id"])
        self.assertEqual((surface["name"], surface["occupancy"]), (both["id"], TWO_OH))
        self.assertEqual(Path(surface["source"]["campaign"]), (self.project / "coverage").resolve())
        self.assertNotIn("relaxed", surface)  # nothing was run
        self.assertIn("no finished relaxation of the surface", surface["notes"][0])
        image = next(n for n in network["nodes"] if n.get("computed_by"))
        moved = self.pick(image["id"])
        self.assertEqual(moved["name"], image["computed_by"])
        self.assertIn(f"{image['id']} is the image of {image['computed_by']}", moved["notes"][0])
        declared = self.pick("two-OH")  # the declared surface is found among the intermediates of the source
        self.assertIn("source", declared)
        self.assertNotIn("source", self.pick("two-OH", "none"))

    def test_what_is_not_a_surface(self):
        with self.assertRaisesRegex(
            ValueError, "No surface 'nothing'.*surfaces: as-given, slab, two-OH.*no coverage campaign"
        ):
            self.pick("nothing")
        with self.assertRaisesRegex(FileNotFoundError, "is not a campaign"):
            self.pick("two-OH", self.root / "nowhere")

    def test_another_slab_or_other_inputs_are_not_mixed(self):
        generate(self.project / "project.yaml", coverage=True)
        incar = self.project / "inputs" / "INCAR_final"
        incar.write_text(incar.read_text() + "\nENCUT = 777\n")
        with self.assertRaisesRegex(
            ValueError, "INCAR_final of the project .* is not the one .*coverage was built with"
        ):
            self.pick("two-OH")
        self.assertNotIn("source", self.pick("two-OH", "none"))

    def test_generate_on_a_surface(self):
        report = generate(self.project / "project.yaml", on="two-OH", families=["aem"])
        self.assertEqual(Path(report["campaign"]), self.project / "on_two-OH")
        self.assertEqual(report["surface"]["name"], "two-OH")
        config = read_json(self.project / "on_two-OH" / "config.json")
        self.assertEqual(config["scope"]["surface"], {"name": "two-OH", "occupancy": TWO_OH, "coupling": None})
        with self.assertRaisesRegex(ValueError, "give the surface with --on"):
            generate(self.project / "project.yaml", source=self.project / "coverage")

    def test_coverage_count_around_a_surface(self):
        found = coverage_count(self.project / "project.yaml", on="two-OH")
        self.assertIn("coverage around two-OH: 4 surfaces on 2 sites", found["text"])
        self.assertEqual(found["states"], {"m0038": ["OH", "empty"], "m0046": ["OH", "empty"]})

    def test_surfaces_block_of_the_project_file(self):
        data = yaml.safe_load((self.project / "project.yaml").read_text())
        self.assertEqual(validate_project(data), [])
        for surfaces, message in (
            ([1], "`surfaces` must map a name"),
            ({"a b": {"m0038": "OH"}}, "must be letters, digits"),
            ({"x": {}}, "surfaces.x must map site IDs to states"),
            ({"x": {"m0039": "OH"}}, "unknown sites of surfaces.x: 'm0039' (did you mean 'm0038'?)"),
            ({"x": {"m0038": 1}}, "surfaces.x.m0038 must be a state"),
            ({"x": {"m0038": "O", "coupling": ["m0038"]}}, "surfaces.x.coupling must list two sites"),
        ):
            problems = validate_project({**data, "surfaces": surfaces})
            self.assertTrue(any(message in p for p in problems), (surfaces, problems))


@unittest.skipIf(os.environ.get("MECHANET_SKIP_SLOW"), "slow workflow test disabled by environment")
class SurfaceWorkflowTests(unittest.TestCase):
    """Coverage of the two open metals (empty, OH), relaxed; then the adsorbate evolution on the
    surface with OH on both. The relaxed surface is moved away from its start on purpose."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env = patch.dict(os.environ, {"MECHANET_CONFIG_DIR": str(cls.tmp / "config")})
        cls.env.start()
        project = cls.tmp / "ruo2"
        shutil.copytree(RUO2, project)
        data = yaml.safe_load((project / "project.yaml").read_text())
        fake = cls.tmp / "fake_vasp.py"
        fake.write_text(FAKE_VASP)
        data["scope"]["families"] = ["aem"]
        data["scope"]["coverage"] = {"sites": ["m0038", "m0046"], "states": {"open_metal": ["empty", "OH"]}}
        data["vasp"] = {
            "potcars": write_potcars(cls.tmp, ("Ru", "O", "H")),
            "magmom_by_element": None,
            "command": [sys.executable, "-I", str(fake)],
        }
        data["frequency"] = {"central": "final"}
        (project / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        cls.project, cls.coverage = project, project / "coverage"
        code, out, err = run("--json", "generate", project, "--coverage")
        assert code == 0, err
        network = read_json(cls.coverage / "network.json")
        cls.surface_id = next(n["id"] for n in network["nodes"] if n["occupancy"] == TWO_OH)
        for state in sorted((cls.coverage / "states").iterdir()):
            positions = read_poscar(state / "POSCAR").positions
            (state / "harmonic.json").write_text(
                json.dumps({"x0": positions.reshape(-1).tolist(), "K": (10.0 * np.eye(positions.size)).tolist()})
            )
        assert run("prepare", cls.coverage)[0] == 0
        for state in sorted((cls.coverage / "states").iterdir()):
            assert run_state(state, through="final", execute=True)["status"] == "complete", state.name
        # The synthetic VASP leaves the atoms where they start. The surface is moved by hand, as a
        # relaxation would: a surface Ru sinks and the H of one OH turns (the bonds stay).
        cls.moved = {"a0038": np.array([0.0, 0.0, -0.06]), "h_m0046": np.array([0.12, 0.10, 0.0])}

        def relax_to(state_id, shifts):
            state = cls.coverage / "states" / state_id
            ids = read_json(state / "state.json")["atom_ids"]
            contcar = state / "02_final" / "CONTCAR"
            relaxed = read_poscar(contcar)
            for atom, shift in shifts.items():
                relaxed.positions[ids.index(atom)] += shift
            write_poscar(contcar, relaxed)
            attempt = read_json(state / "02_final" / "attempt.json")
            recorded = attempt["output_hashes"]
            if isinstance(recorded.get("CONTCAR"), dict):
                recorded["CONTCAR"]["sha256"] = file_hash(contcar)
            else:
                recorded["CONTCAR"] = file_hash(contcar)
            write_json(state / "02_final" / "attempt.json", attempt)

        relax_to(f"{cls.surface_id}__v1", cls.moved)
        # The surface with OH on one site only loses its H in every start: its relaxed structures
        # are no longer that surface, so they must not stand for it in another campaign.
        single = next(n for n in network["nodes"] if len(n["occupancy"]) == 1 and n["state_ids"])
        cls.single_id, cls.lost = single["id"], sorted(single["state_ids"])
        (site,) = single["occupancy"]
        for state_id in cls.lost:
            relax_to(state_id, {f"h_{site}": np.array([0.0, 0.0, 3.0])})
        cls.before = {
            str(path.relative_to(cls.coverage)): file_hash(path)
            for path in sorted(cls.coverage.rglob("*"))
            if path.is_file()
        }
        code, out, err = run("--json", "generate", project, "--on", cls.surface_id)
        assert code == 0, err
        cls.report = json.loads(out)
        cls.campaign = project / f"on_{cls.surface_id}"

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_the_relaxed_surface_is_found(self):
        relaxed = relaxed_surface(self.coverage, self.surface_id)
        self.assertEqual(relaxed["state_id"], f"{self.surface_id}__v1")
        self.assertEqual(relaxed["geometry"], "02_final/CONTCAR")
        self.assertIsNone(relaxed_surface(self.coverage, "n_not_there"))
        surface = read_json(self.campaign / "config.json")["scope"]["surface"]
        self.assertEqual(surface["relaxed"], {"state_id": relaxed["state_id"], "geometry": "02_final/CONTCAR"})

    def test_finished_calculations_are_copied_byte_for_byte(self):
        surface = self.report["surface"]
        self.assertEqual(surface["node_id"], self.surface_id)
        self.assertEqual(Path(surface["source_campaign"]), self.coverage.resolve())
        network = read_json(self.campaign / "network.json")
        imported = [n for n in network["nodes"] if n.get("geometry_status") == "imported"]
        # the slab as given and OH on both sites are surfaces of the coverage campaign that kept
        # their structure; OH on one site did not, so it is not copied and starts afresh here
        self.assertEqual(len(imported), 2)
        self.assertEqual(surface["imported_nodes"], 2)
        self.assertEqual(sorted(x["state_id"] for x in surface["not_copied"]), self.lost)
        single = next(n for n in network["nodes"] if n["id"] == self.single_id)
        self.assertEqual(single["geometry_status"], "generated")
        self.assertNotIn("imported_from", single)
        for sid in single["state_ids"]:
            self.assertIn("seeded_from", read_json(self.campaign / "states" / sid / "state.json"))
            self.assertFalse((self.campaign / "states" / sid / "02_final").exists())
        self.assertIn(self.surface_id, [n["id"] for n in imported])
        for node in imported:
            self.assertEqual(Path(node["imported_from"]["campaign"]), self.coverage.resolve())
            for sid in node["state_ids"]:
                for name in ("state.json", "POSCAR", "plan.json", "02_final/OUTCAR", "02_final/CONTCAR"):
                    ours, theirs = self.campaign / "states" / sid / name, self.coverage / "states" / sid / name
                    self.assertEqual(file_hash(ours), file_hash(theirs), (sid, name))
                origin = read_json(self.campaign / "states" / sid / "imported.json")
                self.assertEqual(origin["state_id"], sid)
        after = {
            str(path.relative_to(self.coverage)): file_hash(path)
            for path in sorted(self.coverage.rglob("*"))
            if path.is_file()
        }
        self.assertEqual(after, self.before)  # the coverage campaign is read, never written

    def test_new_intermediates_start_from_the_relaxed_surface(self):
        relaxed = read_poscar(self.coverage / "states" / f"{self.surface_id}__v1" / "02_final" / "CONTCAR")
        relaxed_ids = read_json(self.coverage / "states" / f"{self.surface_id}__v1" / "state.json")["atom_ids"]
        at_rest = dict(zip(relaxed_ids, relaxed.positions))
        start_of_surface = read_poscar(self.coverage / "states" / f"{self.surface_id}__v1" / "POSCAR")
        for atom, shift in self.moved.items():  # the relaxation did move the surface
            k = relaxed_ids.index(atom)
            moved = minimum_image(relaxed.positions[k] - start_of_surface.positions[k], relaxed.cell)
            self.assertTrue(np.allclose(moved, shift, atol=0.02), (atom, moved))
        network = read_json(self.campaign / "network.json")
        new = [n for n in network["nodes"] if n.get("geometry_status") == "generated"]
        self.assertGreaterEqual(len(new), 3)  # *O, *OOH and *OO on one site, OH on the other
        self.assertEqual(self.report["surface"]["nodes_started_from_the_relaxed_surface"], len(new))
        for node in new:
            states = {**{"m0038": "empty", "m0046": "empty"}, **node["occupancy"]}
            changed = [site for site in ("m0038", "m0046") if states[site] != "OH"]
            self.assertEqual(len(changed), 1, node["id"])
            for sid in node["state_ids"]:
                meta = read_json(self.campaign / "states" / sid / "state.json")
                self.assertEqual(meta["seeded_from"]["state_id"], f"{self.surface_id}__v1")
                start = read_poscar(self.campaign / "states" / sid / "POSCAR")
                site = changed[0]
                own = {site, f"h_{site}", f"d_{site}", f"p_{site}", f"w_{site}"}
                for k, atom in enumerate(meta["atom_ids"]):
                    if atom in own:
                        continue
                    apart = np.linalg.norm(minimum_image(start.positions[k] - at_rest[atom], start.cell))
                    self.assertLess(apart, 1e-6, (sid, atom))  # every other atom sits where the surface relaxed
                kept = "m0046" if site == "m0038" else "m0038"
                self.assertIn(f"h_{kept}", meta["atom_ids"])  # the OH that stays is the relaxed one
                if states[site] != "empty" and not meta["variant"]["binding_modes"]:
                    # the new adsorbate is placed on the metal where the metal relaxed: Ru a0038 sank
                    metal = "a" + site[1:]
                    bond = minimum_image(start.positions[meta["atom_ids"].index(site)] - at_rest[metal], start.cell)
                    expected = default_registry().adsorbate(states[site]).metal_distance_A
                    self.assertAlmostEqual(float(np.linalg.norm(bond)), expected, places=6, msg=sid)

    def test_the_campaign_runs_and_is_analysed_like_any_other(self):
        self.assertEqual(run("validate", self.campaign)[0], 0)
        self.assertEqual(run("prepare", self.campaign)[0], 0)  # the copied states keep the plan they ran under
        status = json.loads(run("--json", "status", self.campaign, "--no-slurm", "--no-html")[1])
        self.assertIsInstance(status, dict)
        for state in sorted((self.campaign / "states").iterdir()):
            if (state / "02_final" / "OUTCAR").exists():
                continue
            positions = read_poscar(state / "POSCAR").positions.reshape(-1)
            (state / "harmonic.json").write_text(
                json.dumps({"x0": positions.tolist(), "K": (10.0 * np.eye(len(positions))).tolist()})
            )
            self.assertEqual(run_state(state, through="final", execute=True)["status"], "complete", state.name)
        code, out, _ = run("--json", "review", self.campaign)
        self.assertEqual(code, 0)
        review = json.loads(out)
        states = len(list((self.campaign / "states").iterdir()))
        self.assertEqual((review["finished"], review["accepted"], review["not_accepted"]), (states, states, []))
        references = self.project / "inputs" / "references.json"
        references.write_text(
            json.dumps({"references": {"H2": {"E": -6.77, "Gcorr": 0.0}, "H2O": {"E": -14.22, "Gcorr": 0.0}}})
        )
        code, out, err = run(
            "--json",
            "pourbaix",
            self.campaign,
            "--references",
            references,
            "--screening",
            "--scale",
            "RHE",
            "--window",
            "0.5:1.5",
        )
        self.assertEqual(code, 0, err)
        labels = {row["node_id"]: row["label"] for row in json.loads(out)["candidates"]}
        self.assertEqual(labels.get(self.surface_id, "the surface"), "the surface")
        lines = read_json(self.campaign / "pourbaix" / "pourbaix.json")["lines"]
        by_id = {line["node_id"]: line["label"] for line in lines}
        self.assertEqual(by_id[self.surface_id], "the surface")
        self.assertEqual(by_id["ref"], "m0038:empty m0046:empty")  # the slab as given, seen from the surface

    def test_text_of_generate_says_what_was_reused(self):
        code, out, _ = run("generate", self.project, "--on", self.surface_id, "--out", self.tmp / "again")
        self.assertEqual(code, 0)
        self.assertIn(f"on the surface {self.surface_id}: terminal 2 OH | bridge 2 O", out)
        self.assertIn("finished calculations of 2 intermediates copied from", out)
        self.assertIn(f"{len(self.lost)} finished calculations were not copied", out)
        self.assertIn(f"new intermediates start from the relaxed surface ({self.surface_id}__v1)", out)
        code, out, err = run("generate", self.project, "--on", self.surface_id, "--on", "ref", "--out", self.tmp / "x")
        self.assertEqual(code, 2)
        self.assertIn("--out names one folder", err)


if __name__ == "__main__":
    unittest.main()
