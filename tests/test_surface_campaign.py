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

    def test_the_search_motif_of_a_family_is_on_the_surface(self):
        """oo_oo writes a transition-state search guess with its own call of structure(): on a
        surface it carries the surface, as the intermediates of the family do."""
        project = self.project_copy()
        on = self.builder(project, "on", {"name": "bridges-OH", "occupancy": {"a0016": "OH", "a0019": "OH"}})
        self.assertEqual(on.surface_pattern({"m0038": "OO"}), ({"a0016": "OH", "a0019": "OH", "m0038": "OO"}, None))
        on.finish(["oo_oo"])
        (motif,) = on.motifs
        self.assertIn("h_a0016", motif["atom_ids"])
        self.assertIn("h_a0019", motif["atom_ids"])
        plain = self.builder(project, "plain")
        plain.finish(["oo_oo"])
        self.assertEqual(len(motif["atom_ids"]), len(plain.motifs[0]["atom_ids"]) + 2)

    def test_an_adsorbate_of_the_surface_in_the_way_of_a_new_one(self):
        """The surface has OOH on a bridging O whose outer O lies over the open metal next to it,
        where an adsorbate of that metal would sit. Nothing fits on that metal next to it, and the
        atoms of the surface are not turned: those intermediates start from the slab instead."""
        project = self.project_copy()
        plain = self.builder(project, "plain")
        blocked, free = "m0046", "m0038"
        site = plain.sites[blocked]
        over = plain.seed.positions[plain.seed_index[site["metal_id"]]] + 1.9 * np.asarray(site["direction"])
        given = {"d_a0016": over.tolist(), "p_a0016": (over + 0.98 * np.asarray(site["direction"])).tolist()}
        occupancy = {"a0016": "OOH"}
        surface = {"name": "x", "occupancy": occupancy, "coupling": None, "given": {"positions": given}}
        on = self.builder(project, "on", surface)
        report = on.finish(["aem"])
        self.assertEqual(report["nodes_without_viable_start"], [])
        anew = report["surface"]["started_anew"]
        on_the_blocked_metal = [n["id"] for n in on.nodes.values() if n["occupancy"].get(blocked, "empty") != "empty"]
        self.assertTrue(on_the_blocked_metal)
        self.assertEqual(sorted(anew), sorted(on_the_blocked_metal))
        self.assertTrue(any(n["occupancy"].get(free, "empty") != "empty" for n in on.nodes.values()))
        for node_id in anew:
            node = on.nodes[node_id]
            self.assertEqual({**on.native, **node["occupancy"]}["a0016"], "OOH")
            self.assertTrue(node["state_ids"])
            self.assertFalse(any("seeded_from" in on.states[sid] for sid in node["state_ids"]))
        others = [n for n in on.nodes.values() if n["id"] not in anew and n["id"] != "ref" and n["state_ids"]]
        self.assertTrue(others)
        self.assertTrue(all("seeded_from" in on.states[sid] for n in others for sid in n["state_ids"]))
        self.assertEqual(report["surface"]["nodes_started_from_the_slab_as_given"], len(others))
        # what did not fit next to the atoms of the surface is kept, marked as that attempt
        marked = {r["node_id"] for r in on.rejected if r.get("next_to_the_atoms_of_the_surface")}
        self.assertEqual(marked, set(anew))

    def test_the_coupled_pair_of_the_surface_apart(self):
        """The surface has a peroxo pair between a bridging O and the O on a metal, lifted towards
        each other in its relaxed structure. A family that changes one of the two breaks the pair:
        the other atom is then where the slab has it, not where the bond held it."""
        project = self.project_copy()
        probe = self.builder(project, "probe")
        a, b = next(p for p in probe.nearby_pairs() if p[0].startswith("a") and p[1].startswith("m"))
        other = "m0038" if b != "m0038" else "m0046"
        surface = {"name": "peroxo", "occupancy": {b: "O"}, "coupling": [a, b]}
        paired = self.builder(project, "paired", surface)
        start = paired.state_structures[paired.nodes[paired.surface_node]["state_ids"][0]]
        relaxed = {aid: start.positions[k].tolist() for k, aid in enumerate(start.ids)}
        in_slab = paired.seed.positions[paired.seed_index[a]]
        lifted = float(np.linalg.norm(minimum_image(np.array(relaxed[a]) - in_slab, start.cell)))
        self.assertGreater(lifted, 0.3)  # the bridging O left its place to bond
        on = self.builder(project, "on", {**surface, "given": {"positions": relaxed}})
        beside = on.node({other: "OH"}, "aem", other)  # the pair is not named: it stays as it relaxed
        self.assertEqual(on.nodes[beside]["coupling"], [a, b])
        kept = on.state_structures[on.nodes[beside]["state_ids"][0]]
        self.assertTrue(np.allclose(kept.positions[kept.ids.index(a)], relaxed[a]))
        broken = on.node({b: "OH"}, "aem", b)  # names one site of the pair
        self.assertIsNone(on.nodes[broken]["coupling"])
        apart = on.state_structures[on.nodes[broken]["state_ids"][0]]
        self.assertTrue(np.allclose(apart.positions[apart.ids.index(a)], in_slab))

    def test_sites_the_pattern_makes_equivalent_are_compared_in_the_relaxed_surface(self):
        """Two OH on the two metals, which a translation of the slab maps onto each other. In the
        relaxed surface one H points elsewhere, 0.5 A from where the translation puts the other."""
        project = self.project_copy()
        plain = self.builder(project, "plain")
        self.assertIsNone(plain._surface_images_apart())  # no surface, no relaxed structure
        start = plain.structure(TWO_OH, None)[0]
        index = {aid: k for k, aid in enumerate(start.ids)}
        shift = minimum_image(
            plain.seed.positions[plain.seed_index["a0046"]] - plain.seed.positions[plain.seed_index["a0038"]],
            plain.seed.cell,
        )
        given = {aid: start.positions[k].tolist() for aid, k in index.items()}
        for atom in ("m0038", "h_m0038"):  # the second OH is the first one, translated
            given[atom.replace("38", "46")] = (start.positions[index[atom]] + shift).tolist()
        surface = {"name": "two-OH", "occupancy": dict(TWO_OH), "coupling": None}
        same = self.builder(project, "same", {**surface, "given": {"positions": given}})
        self.assertTrue(same.translations)
        self.assertIsNone(same._surface_images_apart())  # the slab file as given is not a relaxed surface
        same.environment["state_id"] = "relaxed"
        self.assertLess(same._surface_images_apart(), 0.02)  # within what the slab itself keeps of the translation
        turned = dict(given)
        turned["h_m0046"] = (np.array(given["h_m0046"]) + (0.3, 0.0, 0.4)).tolist()
        differ = self.builder(project, "differ", {**surface, "given": {"positions": turned}})
        differ.environment["state_id"] = "relaxed"
        self.assertAlmostEqual(differ._surface_images_apart(), 0.5, places=2)


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

    def test_an_id_without_a_source(self):
        """--from none: the pattern of an ID is still read from the coverage campaign; nothing else is."""
        generate(self.project / "project.yaml", coverage=True)
        network = read_json(self.project / "coverage" / "network.json")
        both = next(n for n in network["nodes"] if n["occupancy"] == TWO_OH)
        surface = self.pick(both["id"], "none")
        self.assertEqual((surface["name"], surface["occupancy"]), (both["id"], TWO_OH))
        self.assertNotIn("source", surface)
        self.assertNotIn("relaxed", surface)
        self.assertIn("none of its calculations is reused", surface["notes"][0])
        # also after an input changed: a source is then refused, and --from none is what the refusal advises
        incar = self.project / "inputs" / "INCAR_final"
        incar.write_text(incar.read_text() + "\nENCUT = 777\n")
        with self.assertRaisesRegex(ValueError, "--from none"):
            self.pick(both["id"])
        self.assertEqual(self.pick(both["id"], "none")["occupancy"], TWO_OH)
        with self.assertRaisesRegex(ValueError, "No surface 'n_nothing'.*an intermediate of"):
            self.pick("n_nothing", "none")

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
        self.assertEqual(Path(report["campaign"]), (self.project / "on_two-OH").resolve())
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
        self.assertTrue(all("no longer the intermediate it was built as" in x["reason"] for x in surface["not_copied"]))
        self.assertIsInstance(surface["started_anew"], list)
        self.assertIsInstance(surface["equivalent_sites_differ_by_A"], float)  # the two OH, compared as relaxed
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
        # (an intermediate with no start next to the relaxed atoms would start from the slab and be listed)
        anew = set(self.report["surface"]["started_anew"])
        new = [n for n in network["nodes"] if n.get("geometry_status") == "generated" and n["id"] not in anew]
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
        # The second pass: frequencies for one intermediate only. The reference surface goes with
        # it (every formation energy is counted from it), and the diagram of free energies is drawn.
        # (Not the surface itself: its relaxed structure was moved by hand above, away from the
        # minimum of the synthetic forces, which the static point before the frequencies checks.)
        network = read_json(self.campaign / "network.json")
        target = next(n["id"] for n in network["nodes"] if n.get("geometry_status") == "generated")
        code, out, err = run("select", self.campaign, "--nodes", target)
        self.assertEqual(code, 0, err)
        self.assertIn("with the reference surface (ref)", out)
        chosen = read_json(self.campaign / "thermo_selection.json")
        self.assertEqual(chosen["criteria"]["nodes"], [target])
        self.assertEqual(chosen["criteria"]["reference_surface"], "ref")
        self.assertEqual(sorted(n["node_id"] for n in chosen["nodes"]), sorted([target, "ref"]))
        self.assertIn("ref", chosen["state_ids"])
        # the reference surface was copied from the coverage campaign and goes on here
        self.assertTrue((self.campaign / "states" / "ref" / "imported.json").is_file())
        for sid in chosen["state_ids"]:
            done = run_state(self.campaign / "states" / sid, through="freq", execute=True)
            self.assertEqual(done["status"], "complete", sid)
        code, out, err = run(
            "--json", "pourbaix", self.campaign, "--references", references, "--scale", "RHE", "--window", "0.5:1.5"
        )
        self.assertEqual(code, 0, err)
        rows = json.loads(out)["candidates"]
        self.assertTrue(rows)
        self.assertTrue(all(row["free_energy"] for row in rows))
        self.assertLessEqual({row["node_id"] for row in rows}, {target, "ref"})

    def test_text_of_generate_says_what_was_reused(self):
        code, out, _ = run("generate", self.project, "--on", self.surface_id, "--out", self.tmp / "again")
        self.assertEqual(code, 0)
        self.assertIn(f"on the surface {self.surface_id}: terminal 2 OH | bridge 2 O", out)
        self.assertIn("finished calculations of 2 intermediates copied from", out)
        lost = len(self.lost)
        self.assertIn(f"{lost} finished {'calculation was' if lost == 1 else 'calculations were'} not copied", out)
        self.assertIn(f"new intermediates start from the relaxed surface ({self.surface_id}__v1)", out)
        code, out, err = run("generate", self.project, "--on", self.surface_id, "--on", "ref", "--out", self.tmp / "x")
        self.assertEqual(code, 2)
        self.assertIn("--out names one folder", err)


@unittest.skipIf(os.environ.get("MECHANET_SKIP_SLOW"), "slow workflow test disabled by environment")
class PartialSourceTests(unittest.TestCase):
    """A coverage campaign under the other protocol (frequency.central: refine), of which one
    start of an intermediate has not finished when the mechanism campaign is built on it."""

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
        data["scope"]["coverage"] = {"sites": ["m0038"], "states": {"open_metal": ["empty", "OOH"]}}
        data["vasp"] = {
            "potcars": write_potcars(cls.tmp, ("Ru", "O", "H")),
            "magmom_by_element": None,
            "command": [sys.executable, "-I", str(fake)],
        }
        data["frequency"] = {"central": "refine"}
        data["thermodynamics"]["references"] = "inputs/references.json"
        (project / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        (project / "inputs" / "references.json").write_text(
            json.dumps({"references": {"H2": {"E": -6.77, "Gcorr": 0.0}, "H2O": {"E": -14.22, "Gcorr": 0.0}}})
        )
        cls.data, cls.project, cls.coverage = data, project, project / "coverage"
        code, _, err = run("--json", "generate", project, "--coverage")
        assert code == 0, err
        network = read_json(cls.coverage / "network.json")
        node = next(n for n in network["nodes"] if n["occupancy"] == {"m0038": "OOH"})
        cls.node_id, cls.variants = node["id"], sorted(node["state_ids"])
        assert len(cls.variants) >= 2, cls.variants
        for state in sorted((cls.coverage / "states").iterdir()):
            positions = read_poscar(state / "POSCAR").positions
            (state / "harmonic.json").write_text(
                json.dumps({"x0": positions.reshape(-1).tolist(), "K": (10.0 * np.eye(positions.size)).tolist()})
            )
        assert run("prepare", cls.coverage)[0] == 0
        for state in sorted((cls.coverage / "states").iterdir()):
            assert run_state(state, through="final", execute=True)["status"] == "complete", state.name

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_from_the_screening_to_a_campaign_with_a_start_left_behind(self):
        # 1. The screening level reads the relaxations, which is all a campaign has at this point:
        #    under this protocol the energies of the analysis are those of 04_static, not run yet.
        code, out, err = run(
            "--json", "pourbaix", self.coverage, "--screening", "--scale", "RHE", "--window", "1.0:1.8"
        )
        self.assertEqual(code, 0, err)
        self.assertTrue(json.loads(out)["candidates"])
        code, _, err = run("pourbaix", self.coverage, "--electronic-only", "--scale", "RHE")
        self.assertEqual(code, 2)
        self.assertIn("no accepted energy yet", err)

        # 2. As if the second start of *OOH was still running when the mechanism campaign is built.
        finished, running = self.variants[0], self.variants[1:]
        for sid in running:
            for name in ("OUTCAR", "CONTCAR", "vasprun.xml", "attempt.json", "input_hashes.json"):
                (self.coverage / "states" / sid / "02_final" / name).unlink(missing_ok=True)
        (self.coverage / "states" / finished / "run_job.sh").write_text("#SBATCH --chdir=/the/coverage/campaign\n")
        code, out, err = run("generate", self.project, "--on", self.node_id)
        self.assertEqual(code, 0, err)
        left = "1 calculation that has" if len(running) == 1 else f"{len(running)} calculations that have"
        self.assertIn(f"{left} not finished there {'was' if len(running) == 1 else 'were'} left behind", out)
        campaign = self.project / f"on_{self.node_id}"
        surface = read_json(campaign / "build_report.json")["surface"]
        self.assertEqual(
            surface["not_copied"],
            [{"state_id": sid, "reason": "its relaxation has not finished in the source campaign"} for sid in running],
        )
        node = next(n for n in read_json(campaign / "network.json")["nodes"] if n["id"] == self.node_id)
        self.assertEqual(node["state_ids"], [finished])
        self.assertFalse(any((campaign / "states" / sid).exists() for sid in running))
        copied = campaign / "states" / finished
        self.assertTrue((copied / "imported.json").is_file())
        self.assertFalse((copied / "run_job.sh").exists())  # it names the folder of the other campaign

        # 3. Here the copied state belongs to the family that uses it, not only to the one that built it.
        for family in ("aem", "coverage"):
            code, out, _ = run("--json", "list", campaign, "--family", family, "--limit", "0")
            self.assertEqual(code, 0)
            self.assertIn(finished, [row["id"] for row in json.loads(out)["states"]], family)
        self.assertEqual(run("prepare", campaign)[0], 0)

        # 4. Other settings than the source was run with: said when the copied state is prepared.
        changed = {**self.data, "vasp": {**self.data["vasp"], "common_overrides": {"ENCUT": 777}}}
        (self.project / "project.yaml").write_text(yaml.safe_dump(changed, sort_keys=False))
        code, _, err = run("generate", self.project, "--on", self.node_id, "--out", self.tmp / "other")
        self.assertEqual(code, 0, err)
        code, _, err = run("prepare", self.tmp / "other")
        self.assertEqual(code, 2)
        self.assertIn(f"{finished} was copied from", err)
        self.assertIn("prepared with other settings", err)
        self.assertIn("--from none", err)


if __name__ == "__main__":
    unittest.main()
