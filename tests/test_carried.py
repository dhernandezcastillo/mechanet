"""Slabs that already carry adsorbates.

A group that is a registered adsorbate bound through one atom is read as the state of a site:
`mechanet init` takes its atoms out of the slab the campaign is referenced to, and
`mechanet generate` builds the campaign on the slab as it was given. A group that cannot be
read this way stays part of the framework, and the inspection says so with the reason.
The slabs are RuO2(110) (two open metals a0038 and a0046, two bridging O) with groups put on.
"""

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from test_robustness import fen4_graphene

from mechanet.builder import Builder
from mechanet.cli import main
from mechanet.fileio import read_json
from mechanet.inspector import inspect_slab, summary
from mechanet.plugins import default_registry
from mechanet.project import generate, init_project, load_project, resolve_config
from mechanet.structure import Structure, minimum_image, read_poscar, read_structure, write_poscar

ROOT = Path(__file__).resolve().parents[1]
RUO2 = ROOT / "examples" / "ruo2_110"
PT = ROOT / "examples" / "pt_111"
PLUGINS = ROOT / "examples" / "plugins"


def quiet(*_args):
    return None


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {"MECHANET_CONFIG_DIR": str(self.root / "config")})
        self.env.start()
        loaded = load_project(RUO2 / "project.yaml")
        config = resolve_config(loaded, cluster=False)
        self.builder = Builder(config, self.root / "build", base_dir=loaded["_dir"], registry=default_registry())

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def slab_with(self, occupancy, coupling=None, binding_modes=None, name="given.vasp"):
        """The example slab with adsorbates placed by the builder, as a file without atom IDs."""
        choices, _ = self.builder._search(occupancy, coupling, binding_modes or {})  # an orientation that fits
        placement = choices[0]["placement"] if choices else None
        structure = self.builder.structure(occupancy, coupling, placement, binding_modes)[0]
        write_poscar(self.root / name, structure)
        return self.root / name

    def slab_plus(self, atoms, name="given.vasp"):
        """The example slab with extra atoms [(element, position)] appended: an element in a second block."""
        seed = read_poscar(RUO2 / "slab" / "RuO2_110.vasp")
        structure = Structure(
            seed.symbols + [element for element, _ in atoms],
            np.vstack([seed.positions, np.array([position for _, position in atoms])]),
            seed.cell,
            np.vstack([seed.mobile, np.ones((len(atoms), 3), dtype=bool)]),
            [f"a{k + 1:04d}" for k in range(len(seed) + len(atoms))],
        )
        first = {}
        for k, element in enumerate(structure.symbols):
            first.setdefault(element, k)
        order = sorted(range(len(structure)), key=lambda k: (first[structure.symbols[k]], k))  # one block per element
        grouped = Structure(
            [structure.symbols[k] for k in order],
            structure.positions[order],
            structure.cell,
            structure.mobile[order],
            [f"a{k + 1:04d}" for k in range(len(order))],
        )
        write_poscar(self.root / name, grouped)
        return self.root / name

    def above(self, metal="a0038", height=2.0, shift=(0.0, 0.0)):
        position = self.builder.seed.positions[self.builder.seed_index[metal]].copy()
        position[:2] += shift
        position[2] += height
        return position


class FindTests(TempCase):
    def test_ooh_on_a_metal_is_a_site_in_the_state_ooh(self):
        report = inspect_slab(self.slab_with({"m0038": "OOH"}))
        self.assertEqual(report["kept_as_framework"], [])
        (group,) = report["carried_adsorbates"]
        self.assertEqual((group["state"], group["on"]), ("OOH", "ligand"))
        self.assertEqual(len(group["atoms"]), 3)
        self.assertEqual(sorted(group["roles"]), ["d", "p"])
        self.assertEqual(len(group["metals"]), 1)
        text = summary(report)
        self.assertIn("adsorbates the slab carries (1): OOH on", text)
        self.assertIn("`mechanet init` reads them as states of sites", text)
        self.assertNotIn("O-O or ligand-ligand bond", text)  # said once, as what it is

    def test_other_adsorbates_of_the_registry(self):
        for occupancy, state, where in (
            ({"m0038": "OO"}, "OO", "ligand"),
            ({"m0038": "Cl"}, "Cl", "metal"),
            ({"m0038": "OCl"}, "OCl", "ligand"),
            ({"a0016": "OOH"}, "OOH", "ligand"),  # on a bridging oxygen
        ):
            report = inspect_slab(self.slab_with(occupancy, name=f"{state}_{where}.vasp"))
            groups = [(g["state"], g["on"]) for g in report["carried_adsorbates"]]
            self.assertEqual(groups, [(state, where)], occupancy)
            self.assertEqual(report["kept_as_framework"], [])

    def test_o_oh_and_water_stay_what_they_were(self):
        for state in ("O", "OH", "H2O"):
            report = inspect_slab(self.slab_with({"m0038": state}, name=f"{state}.vasp"))
            self.assertEqual(report["carried_adsorbates"], [], state)
            self.assertEqual(report["kept_as_framework"], [], state)
            natives = sorted(s["native"] for s in report["proposal"]["sites"] if s["kind"] == "terminal")
            self.assertEqual(natives, [state])
        plain = inspect_slab(RUO2 / "slab" / "RuO2_110.vasp")
        self.assertEqual((plain["carried_adsorbates"], plain["kept_as_framework"]), ([], []))

    def test_groups_that_stay_in_the_framework_say_why(self):
        pair = inspect_slab(self.slab_with({"m0038": "O", "m0046": "O"}, ("m0038", "m0046"), name="pair.vasp"))
        self.assertEqual(pair["carried_adsorbates"], [])
        (kept,) = pair["kept_as_framework"]
        self.assertEqual((kept["formula"], len(kept["atoms"])), ("O2", 2))
        self.assertIn("it binds the surface through 2 atoms", kept["reason"])
        side_on = inspect_slab(self.slab_with({"m0038": "OO"}, binding_modes={"m0038": "side_on"}, name="side.vasp"))
        self.assertEqual(side_on["carried_adsorbates"], [])
        self.assertEqual(len(side_on["kept_as_framework"]), 1)
        self.assertIn("bound side-on", side_on["kept_as_framework"][0]["reason"])
        text = summary(side_on)
        self.assertIn("kept as part of the framework, not reactive: O2", text)

    def test_what_no_registered_adsorbate_describes(self):
        nitrogen = self.above(height=1.9)
        amide = [("N", nitrogen), ("H", nitrogen + (0.8, 0.0, 0.6)), ("H", nitrogen + (-0.8, 0.0, 0.6))]
        report = inspect_slab(self.slab_plus(amide))
        self.assertEqual(report["carried_adsorbates"], [])
        (group,) = report["kept_as_framework"]
        self.assertEqual(group["formula"], "H2N")
        self.assertIn("no registered adsorbate is H2N bound through N", group["reason"])
        self.assertIn("a plugin can register it", group["reason"])
        # with the plugin that registers NH2 the same file carries an adsorbate
        nrr = default_registry([str(PLUGINS / "nrr.py")])
        (found,) = inspect_slab(self.root / "given.vasp", registry=nrr)["carried_adsorbates"]
        self.assertEqual((found["state"], found["on"], sorted(found["roles"])), ("NH2", "metal", ["h", "w"]))

    def test_an_adsorbate_through_another_element_sits_on_one_metal(self):
        from mechanet.carried import find

        # H in a hollow of three metals, and CO on a bridge: no site holds them
        cell, free = 15.0 * np.eye(3), lambda i, ignore=(): True
        metals = np.array([[5.0, 5.0, 5.0], [7.7, 5.0, 5.0], [6.35, 7.3, 5.0]])
        hollow = Structure(
            ["Pt", "Pt", "Pt", "H"],
            np.vstack([metals, [6.35, 5.8, 6.0]]),
            cell,
            np.ones((4, 3), dtype=bool),
            [f"a{k + 1:04d}" for k in range(4)],
        )
        nbrs = {0: [(3, 1.9)], 1: [(3, 1.9)], 2: [(3, 1.9)], 3: [(0, 1.9), (1, 1.9), (2, 1.9)]}
        carried, kept = find(hollow, ["metal"] * 3 + ["H"], nbrs, [False] * 4, free, default_registry())
        self.assertEqual(carried, [])
        self.assertEqual([(group["formula"], group["atoms"]) for group in kept], [("H", ["a0004"])])
        self.assertIn("H binds 3 metals", kept[0]["reason"])
        self.assertIn("written on top of one metal only", kept[0]["reason"])
        bridge = Structure(
            ["Pt", "Pt", "C", "O"],
            np.vstack([metals[:2], [6.35, 5.0, 6.4], [6.35, 5.0, 7.55]]),
            cell,
            np.ones((4, 3), dtype=bool),
            [f"a{k + 1:04d}" for k in range(4)],
        )
        nbrs = {0: [(2, 2.0)], 1: [(2, 2.0)], 2: [(0, 2.0), (1, 2.0), (3, 1.15)], 3: [(2, 1.15)]}
        carried, kept = find(
            bridge, ["metal", "metal", "nonmetal", "ligand"], nbrs, [False] * 4, free, default_registry()
        )
        self.assertEqual(carried, [])
        self.assertEqual([group["formula"] for group in kept], ["CO"])
        self.assertIn("C binds 2 metals", kept[0]["reason"])

    def test_a_lone_anion_between_metals_is_a_ligand_of_the_slab(self):
        # a chloride between two Ru is what the sulfur of a sulfide is: a ligand of the framework.
        # The inspection lists it as before, and it is not called an adsorbate.
        between = [("Cl", self.above(height=1.7, shift=(1.6, 0.0)))]
        report = inspect_slab(self.slab_plus(between))
        self.assertEqual((report["carried_adsorbates"], report["kept_as_framework"]), ([], []))
        self.assertIn("1 Cl ligands are not covered by the built-in adsorbates", summary(report))
        # a slab whose anions are not oxygen: every bridging anion is a ligand, none is "kept"
        seed = read_poscar(RUO2 / "slab" / "RuO2_110.vasp")
        fluoride = Structure(
            ["F" if element == "O" else element for element in seed.symbols],
            seed.positions,
            seed.cell,
            seed.mobile,
            seed.ids,
        )
        report = inspect_slab(fluoride)
        self.assertEqual((report["carried_adsorbates"], report["kept_as_framework"]), ([], []))
        text = summary(report)
        self.assertIn("F ligands are not covered by the built-in adsorbates", text)
        self.assertNotIn("kept as part of the framework", text)

    def test_groups_of_one_kind_are_said_once(self):
        sulfur = [("S", self.above("a0038", height=2.2)), ("S", self.above("a0046", height=2.2))]
        report = inspect_slab(self.slab_plus(sulfur))
        self.assertEqual(report["carried_adsorbates"], [])
        self.assertEqual([group["formula"] for group in report["kept_as_framework"]], ["S", "S"])
        lines = [line for line in summary(report).splitlines() if "kept as part of the framework" in line]
        self.assertEqual(len(lines), 1)
        self.assertIn("not reactive: 2 x S (", lines[0])
        self.assertIn("no registered adsorbate is S bound through S", lines[0])
        self.assertNotIn("S ligands are not covered", summary(report))  # said once, as what it is

    def test_two_adsorbates_on_one_metal(self):
        from mechanet.carried import find

        structure = Structure(
            ["Ru", "Cl", "Cl"],
            np.array([[5.0, 5.0, 5.0], [6.9, 5.0, 6.3], [3.1, 5.0, 6.3]]),
            15.0 * np.eye(3),
            np.ones((3, 3), dtype=bool),
            ["a0001", "a0002", "a0003"],
        )
        nbrs = {0: [(1, 2.3), (2, 2.3)], 1: [(0, 2.3)], 2: [(0, 2.3)]}
        carried, kept = find(
            structure, ["metal", "ligand", "ligand"], nbrs, [False] * 3, lambda i, ignore=(): True, default_registry()
        )
        self.assertEqual(carried, [])
        self.assertEqual([group["atoms"] for group in kept], [["a0002"], ["a0003"]])
        self.assertIn("a0001 holds 2 adsorbates", kept[0]["reason"])
        self.assertIn("a metal has one open site", kept[0]["reason"])
        one = {0: [(1, 2.3)], 1: [(0, 2.3)], 2: []}
        carried, kept = find(
            structure, ["metal", "ligand", "ligand"], one, [False] * 3, lambda i, ignore=(): True, default_registry()
        )
        self.assertEqual([(group["state"], group["on"]) for group in carried], [("Cl", "metal")])
        self.assertEqual(kept, [])

    def test_a_hydride_with_the_plugin_that_knows_it(self):
        seed = read_structure(PT / "slab" / "Pt_111.vasp")
        top = int(np.argmax(seed.positions[:, 2]))
        hydride = seed.positions[top] + (0.0, 0.0, 1.6)
        structure = Structure(
            seed.symbols + ["H"],
            np.vstack([seed.positions, hydride]),
            seed.cell,
            np.vstack([seed.mobile, np.ones((1, 3), dtype=bool)]),
            [f"a{k + 1:04d}" for k in range(len(seed) + 1)],
        )
        her = default_registry([str(PLUGINS / "her.py")])
        (group,) = inspect_slab(structure, registry=her)["carried_adsorbates"]
        self.assertEqual((group["state"], group["on"], group["metals"]), ("H", "metal", [seed.ids[top]]))
        without = inspect_slab(structure)
        self.assertEqual(without["carried_adsorbates"], [])
        self.assertIn("no registered adsorbate is H bound through H", without["kept_as_framework"][0]["reason"])

    def test_a_covalent_sheet_is_the_slab_itself(self):
        report = inspect_slab(fen4_graphene(self.root / "FeN4.vasp"))
        self.assertEqual(report["carried_adsorbates"], [])
        self.assertEqual(report["kept_as_framework"], [])


class InitTests(TempCase):
    def init(self, slab, name="p", **options):
        lines = []
        result = init_project(slab, self.root / name, interactive=False, output=lines.append, **options)
        return result, "\n".join(map(str, lines))

    def test_the_adsorbate_leaves_the_slab_and_the_campaign_is_built_on_the_slab_as_given(self):
        given_file = self.slab_with({"m0038": "OOH"})
        _, said = self.init(given_file)
        self.assertIn("the slab without them (49 atoms) is the reference of the campaign", said)
        project = self.root / "p"
        data = yaml.safe_load((project / "project.yaml").read_text())
        self.assertEqual(data["slab"]["file"], "slab/given_bare.vasp")
        self.assertEqual(data["slab"]["carried"], "slab/as_given.json")
        self.assertTrue((project / "slab" / "given.vasp").is_file())  # the file as given is kept
        self.assertEqual(len(read_poscar(project / "slab" / "given_bare.vasp")), 49)
        carried = read_json(project / "slab" / "as_given.json")
        ((site, state),) = carried["occupancy"].items()
        self.assertEqual(state, "OOH")
        self.assertEqual(sorted(carried["positions"]), [f"d_{site}", f"p_{site}"])
        kinds = {s["id"]: (s["kind"], s["native"]) for s in data["sites"]}
        self.assertEqual(kinds[site], ("terminal", "O"))
        self.assertEqual(sorted(k for k, _ in kinds.values()), ["bridge", "bridge", "open_metal", "terminal"])

        report = generate(project / "project.yaml", families=["aem"])
        campaign = project / "campaign"
        self.assertEqual(report["surface"]["name"], "as-given")
        self.assertEqual(report["surface"]["occupancy"], {site: "OOH"})
        self.assertEqual(report["surface"]["nodes_started_from_the_relaxed_surface"], 0)
        self.assertGreater(report["surface"]["nodes_started_from_the_slab_as_given"], 3)
        self.assertEqual(report["nodes_without_viable_start"], [])
        config = read_json(campaign / "config.json")
        self.assertEqual(config["slab"]["carried"]["occupancy"], {site: "OOH"})  # the campaign carries it along
        given = read_structure(given_file)
        network = read_json(campaign / "network.json")
        surface = next(n for n in network["nodes"] if n["id"] == network["surface"]["node_id"])
        (state_id,) = surface["state_ids"]
        start = read_poscar(campaign / "states" / state_id / "POSCAR")
        self.assertEqual(sorted(start.symbols), sorted(given.symbols))
        for symbol, position in zip(start.symbols, start.positions):  # the start is the file, atom for atom
            apart = min(
                np.linalg.norm(minimum_image(position - other, start.cell))
                for element, other in zip(given.symbols, given.positions)
                if element == symbol
            )
            self.assertLess(apart, 1e-6)
        meta = read_json(campaign / "states" / state_id / "state.json")
        self.assertEqual(meta["seeded_from"]["structure"], "the slab file as given")
        # next to it, the mechanism on the other metal keeps the OOH where the file has it
        anew = set(report["surface"]["started_anew"])  # none here; if one had no start next to the OOH, not it
        other = next(
            n
            for n in network["nodes"]
            if n["occupancy"].get(site) == "OOH" and len(n["occupancy"]) == 2 and n["state_ids"] and n["id"] not in anew
        )
        beside = read_json(campaign / "states" / other["state_ids"][0] / "state.json")
        k = beside["atom_ids"].index(f"d_{site}")
        there = read_poscar(campaign / "states" / other["state_ids"][0] / "POSCAR").positions[k]
        self.assertLess(
            np.linalg.norm(minimum_image(there - np.array(carried["positions"][f"d_{site}"]), start.cell)), 1e-6
        )

        # a coverage campaign of this project probes around the slab as given, and says so
        from mechanet.project import coverage_count

        found = coverage_count(project / "project.yaml")
        self.assertIn("coverage around as-given:", found["text"])
        self.assertEqual(found["states"][site][0], "OOH")
        bare = generate(project / "project.yaml", families=["aem"], on="slab")
        self.assertEqual(Path(bare["campaign"]), (project / "on_slab").resolve())
        self.assertEqual(bare["surface"]["occupancy"], {})
        again = generate(project / "project.yaml", families=["aem"], on="as-given", out=self.root / "again")
        self.assertEqual(again["surface"]["occupancy"], {site: "OOH"})
        self.assertGreater(again["surface"]["nodes_started_from_the_slab_as_given"], 3)

    def test_a_frequency_mask_loses_the_same_atoms(self):
        given_file = self.slab_with({"m0038": "OOH"})
        mask = read_poscar(given_file)
        mask.mobile[:] = (mask.positions[:, 2] > mask.positions[:, 2].max() - 3.0)[:, None]
        (self.root / "other").mkdir()
        for name in ("mask.vasp", "other/given.vasp"):  # a mask of its own name, and one named like the slab
            write_poscar(self.root / name, mask)
            project = self.root / name.replace("/", "_").replace(".vasp", "")
            init_project(given_file, project, interactive=False, output=quiet, frequency_mask=self.root / name)
            data = yaml.safe_load((project / "project.yaml").read_text())
            expected = "slab/mask_bare.vasp" if name == "mask.vasp" else "slab/given_mask_bare.vasp"
            self.assertEqual(data["slab"]["frequency_mask"], expected)
            self.assertEqual(data["slab"]["file"], "slab/given_bare.vasp")
            bare, bare_mask = read_poscar(project / data["slab"]["file"]), read_poscar(project / expected)
            self.assertEqual((len(bare), len(bare_mask)), (49, 49))
            self.assertEqual(bare.symbols, bare_mask.symbols)
            self.assertTrue(np.allclose(bare.positions, bare_mask.positions))
            self.assertTrue(bare_mask.mobile.any() and not bare_mask.mobile.all())  # the region of the mask, kept
            report = generate(project / "project.yaml", families=["aem"])
            self.assertEqual(report["surface"]["name"], "as-given")
            self.assertEqual(report["nodes_without_viable_start"], [])

    def test_a_damaged_carried_entry_says_what_to_do(self):
        self.init(self.slab_with({"m0038": "OOH"}))
        project = self.root / "p"
        entry = project / "slab" / "as_given.json"
        original = entry.read_text()
        carried = read_json(entry)
        ((site, _),) = carried["occupancy"].items()
        text = (project / "project.yaml").read_text(encoding="utf-8")

        # the site of the adsorbate deleted from project.yaml: its atoms are not in the slab file any more
        line = next(row for row in text.splitlines() if f"id: {site}," in row)
        (project / "project.yaml").write_text(text.replace(line + "\n", ""), encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            generate(project / "project.yaml", families=["aem"])
        message = str(caught.exception)
        self.assertIn(f"has OOH on {site}, and `sites` of project.yaml has no such site", message)
        self.assertIn("Restore the site", message)
        # taking the adsorbate out of the file is the way to study the slab without it
        entry.write_text(json.dumps({**carried, "occupancy": {}}))
        report = generate(project / "project.yaml", families=["aem"], out=self.root / "without")
        self.assertEqual(report["surface"]["occupancy"], {})
        self.assertEqual(report["surface"]["nodes_started_from_the_slab_as_given"], 0)
        (project / "project.yaml").write_text(text, encoding="utf-8")

        entry.write_text("not json")
        with self.assertRaises(ValueError) as caught:
            generate(project / "project.yaml", families=["aem"])
        self.assertIn("slab/as_given.json (slab.carried) cannot be read as JSON", str(caught.exception))
        entry.write_text('{"name": "as-given", "occupancy": {}}')
        with self.assertRaises(ValueError) as caught:
            generate(project / "project.yaml", families=["aem"])
        self.assertIn("must hold `name`, `occupancy` {site: state} and `positions`", str(caught.exception))
        entry.write_text(original.replace('"OOH"', '"OOX"'))
        with self.assertRaises(ValueError) as caught:
            generate(project / "project.yaml", families=["aem"])
        self.assertIn("Unknown adsorbate 'OOX'", str(caught.exception))
        entry.unlink()
        with self.assertRaises(FileNotFoundError) as caught:
            generate(project / "project.yaml", families=["aem"])
        self.assertIn("slab.carried names slab/as_given.json, which does not exist", str(caught.exception))
        self.assertIn("remove the entry to build on the slab without them", str(caught.exception))

    def test_a_relaxed_bond_beyond_the_cutoff_of_a_start_is_not_a_fault_of_the_start(self):
        """A surface whose atoms are given, with an OH whose O-H is 1.30 A: bonded for a relaxed
        structure (lost beyond 1.35 A), too long for a start (1.25 A). The surface and the starts
        next to it keep it as it is."""
        structure = self.builder.structure({"m0038": "OH", "m0046": "OH"}, None)[0]
        index = {aid: k for k, aid in enumerate(structure.ids)}
        bond = structure.positions[index["h_m0046"]] - structure.positions[index["m0046"]]
        structure.positions[index["h_m0046"]] = structure.positions[index["m0046"]] + 1.30 * bond / np.linalg.norm(bond)
        loaded = load_project(RUO2 / "project.yaml")
        config = resolve_config(loaded, cluster=False)
        given = {aid: structure.positions[index[aid]].tolist() for aid in ("m0038", "h_m0038", "m0046", "h_m0046")}
        config["scope"]["surface"] = {
            "name": "stretched",
            "occupancy": {"m0038": "OH", "m0046": "OH"},
            "coupling": None,
            "given": {"positions": given},
        }
        on = Builder(config, self.root / "on", base_dir=loaded["_dir"], registry=default_registry())
        surface = on.nodes[on.surface_node]
        self.assertTrue(surface["state_ids"], on.rejected)  # the surface itself, as given
        oxo = on.node({"m0038": "O"}, "aem", "m0038")
        self.assertTrue(on.nodes[oxo]["state_ids"], on.rejected)
        start = on.state_structures[on.nodes[oxo]["state_ids"][0]]
        k_h, k_o = start.ids.index("h_m0046"), start.ids.index("m0046")
        self.assertAlmostEqual(float(np.linalg.norm(start.positions[k_h] - start.positions[k_o])), 1.30, places=6)

    def test_a_group_on_a_metal_gives_an_open_site_with_the_direction_of_the_file(self):
        _, said = self.init(self.slab_with({"m0038": "Cl"}), reaction="cer")
        self.assertNotIn("has no reservoir", said)
        data = yaml.safe_load((self.root / "p" / "project.yaml").read_text())
        site = next(s for s in data["sites"] if s["id"] == "m0038")
        self.assertEqual((site["kind"], site["native"], site["metal"]), ("open_metal", "empty", "a0038"))
        self.assertTrue(np.allclose(site["direction"], self.builder.sites["m0038"]["direction"], atol=1e-4))
        carried = read_json(self.root / "p" / "slab" / "as_given.json")
        self.assertEqual(carried["occupancy"], {"m0038": "Cl"})
        self.assertEqual(sorted(carried["positions"]), ["m0038"])
        report = generate(self.root / "p" / "project.yaml")
        self.assertEqual(report["surface"]["description"], "terminal 1 Cl, 1 empty | bridge 2 O")

    def test_a_reaction_without_a_reservoir_for_the_carried_element_is_said(self):
        _, said = self.init(self.slab_with({"m0038": "Cl"}), reaction="oer")
        self.assertIn("the reaction oer has no reservoir for the Cl the slab carries on m0038", said)

    def test_answering_no_keeps_the_group_in_the_framework(self):
        given_file = self.slab_with({"m0038": "OOH"})
        lines = []

        def answer(prompt):
            return "n" if "carries" in prompt else ""

        init_project(given_file, self.root / "p", interactive=True, input_func=answer, output=lines.append)
        said = "\n".join(map(str, lines))
        self.assertIn("kept as part of the framework: these atoms stay where they are and do not react", said)
        data = yaml.safe_load((self.root / "p" / "project.yaml").read_text())
        self.assertNotIn("carried", data["slab"])
        self.assertEqual(len(read_structure(self.root / "p" / data["slab"]["file"])), 51)
        self.assertEqual(len(data["sites"]), 3)

    def test_the_site_of_a_carried_adsorbate_cannot_be_dropped_at_init(self):
        """The OOH left the slab; only its site can put it back. Leaving that site out of the list
        of sites to keep would silently turn M-OOH into a fixed M-O: the site stays, and init says so."""
        given_file = self.slab_with({"m0038": "OOH"})
        lines = []

        def answer(prompt):
            if "proposed reactive site" in prompt:
                return "n"
            if "Site IDs to keep" in prompt:
                return "a0016,a0019"
            return ""

        init_project(given_file, self.root / "p", interactive=True, input_func=answer, output=lines.append)
        said = "\n".join(map(str, lines))
        carried = read_json(self.root / "p" / "slab" / "as_given.json")
        ((site, state),) = carried["occupancy"].items()
        self.assertEqual(state, "OOH")
        self.assertIn(f"{site} stays a site: it holds the OOH the slab carried", said)
        data = yaml.safe_load((self.root / "p" / "project.yaml").read_text())
        self.assertEqual(sorted(s["id"] for s in data["sites"]), sorted(["a0016", "a0019", site]))
        report = generate(self.root / "p" / "project.yaml", families=["aem"])
        self.assertEqual(report["surface"]["occupancy"], {site: "OOH"})

    def test_a_file_with_the_adsorbate_in_blocks_of_its_own(self):
        oxygen = self.above(height=1.95)
        distal = oxygen + (1.1, 0.0, 0.95)
        hydroperoxo = [("O", oxygen), ("O", distal), ("H", distal + (0.6, 0.0, 0.75))]
        seed = read_poscar(RUO2 / "slab" / "RuO2_110.vasp")
        structure = Structure(
            seed.symbols + [element for element, _ in hydroperoxo],
            np.vstack([seed.positions, np.array([position for _, position in hydroperoxo])]),
            seed.cell,
            np.vstack([seed.mobile, np.ones((3, 3), dtype=bool)]),
            [f"a{k + 1:04d}" for k in range(len(seed) + 3)],
        )
        from ase.io import write

        write(self.root / "blocks.vasp", structure.to_ase(), format="vasp", direct=True, sort=False)
        _, said = self.init(self.root / "blocks.vasp")
        self.assertIn("lists an element in more than one block", said)
        self.assertIn("the slab without them (49 atoms)", said)
        carried = read_json(self.root / "p" / "slab" / "as_given.json")
        self.assertEqual(list(carried["occupancy"].values()), ["OOH"])
        report = generate(self.root / "p" / "project.yaml", families=["aem"])
        self.assertEqual(report["nodes_without_viable_start"], [])

    def test_inspect_and_init_through_the_command_line(self):
        given_file = self.slab_with({"m0038": "OOH"})
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["inspect", str(given_file)])
        self.assertEqual(code, 0, err.getvalue())
        self.assertIn("adsorbates the slab carries (1): OOH on", out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = main(["init", str(given_file), str(self.root / "cli"), "--yes"])
        self.assertEqual(code, 0)
        self.assertTrue((self.root / "cli" / "slab" / "as_given.json").is_file())
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = main(["generate", str(self.root / "cli"), "--families", "aem"])
        self.assertEqual(code, 0)
        self.assertIn("on the surface as-given: terminal 1 OOH, 1 empty | bridge 2 O", out.getvalue())
        self.assertIn("intermediates start from the slab file as given", out.getvalue())
        shutil.rmtree(self.root / "cli")


if __name__ == "__main__":
    unittest.main()
