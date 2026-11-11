"""An adsorbate that leaves the surface together with an atom of the surface.

O2 made of an adsorbed O and a lattice O is the step of a lattice-oxygen route. It gives the
intermediate it started from no energy, like any release, but the surface it leaves behind is a
possible intermediate: the checks name the atom that left and read that surface from the relaxed
structure without the molecule. On RuO2(110): *O on the open Ru site m0038, next to the bridging
O a0019 (a declared site) and to lattice O that is not a site.
"""

import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from mechanet.builder import Builder
from mechanet.discovered import _left_behind_start, node_id
from mechanet.fileio import read_json
from mechanet.prescreen import new_intermediate
from mechanet.project import generate
from mechanet.review import _Campaign, classify, describe_changes, surface_left_lines
from mechanet.structure import Structure, minimum_image, read_poscar, write_poscar
from mechanet.topology import _own_release, _release_kind, topology_check

RUO2 = Path(__file__).resolve().parents[1] / "examples" / "ruo2_110"
SITE, BRIDGE = "m0038", "a0019"


class ReleaseRuleTests(unittest.TestCase):
    """The rule on its own: whose atoms a released molecule holds."""

    SYMBOLS = {"m0038": "O", "d_m0038": "O", "a0019": "O", "a0001": "O", "a0040": "Ru", "a0090": "H"}
    NATIVE = {"m0038": "empty", "a0019": "O"}

    def kind(self, atoms, occupancy, lost=(("a0040", "m0038"),), new=()):
        released = [{"formula": "", "atoms": list(atoms), "from": ["a0040"]}]
        args = (released, [tuple(p) for p in lost], list(new), occupancy, None, self.NATIVE, self.SYMBOLS)
        self.assertEqual(_own_release(*args), _release_kind(*args)[0] == "own")
        return _release_kind(*args)

    def test_the_adsorbate_alone_is_its_own(self):
        self.assertEqual(self.kind(["m0038"], {"m0038": "O"}), ("own", []))
        self.assertEqual(self.kind(["m0038", "d_m0038"], {"m0038": "OO"}), ("own", []))
        # the O of a site the intermediate changed is its own too (the O2 of a bridge-bound OO)
        self.assertEqual(self.kind(["a0019", "d_m0038"], {"a0019": "OO"}, lost=[("a0040", "a0019")]), ("own", []))

    def test_an_atom_of_the_surface_in_the_molecule_is_named(self):
        both = (("a0040", "m0038"), ("a0040", "a0019"))
        self.assertEqual(self.kind(["a0019", "m0038"], {"m0038": "O"}, lost=both), ("with_surface_atom", ["a0019"]))
        self.assertEqual(self.kind(["a0001", "m0038"], {"m0038": "O"}), ("with_surface_atom", ["a0001"]))
        # a native H that came along is not counted: the H of a site's own ligand goes with the site
        self.assertEqual(self.kind(["m0038", "a0090"], {"m0038": "O"}), ("own", []))

    def test_other_changes_at_the_metals_still_wait_for_review(self):
        self.assertEqual(self.kind(["a0019", "m0038"], {"m0038": "O"}, new=[("a0040", "a0001")]), (None, []))
        self.assertEqual(
            self.kind(["a0019", "m0038"], {"m0038": "O"}, lost=[("a0040", "m0038"), ("a0040", "a0001")]), (None, [])
        )
        self.assertEqual(self.kind(["a0019"], {"m0038": "O"}, lost=[("a0040", "a0019")]), (None, []))  # a spectator


class ReportTests(unittest.TestCase):
    """What review and the pre-screening say, from the record the checks write."""

    def micro(self, **left):
        return {"status": "surface_atom_released", "taken_surface_atoms": ["a0019"], "surface_left": left}

    def test_three_kinds_of_surface_left_behind(self):
        known = self.micro(occupancy_changes={"a0019": "empty"}, node_id="n_rep", matched_node_id="n_0019empty")
        unlisted = self.micro(occupancy_changes={"a0019": "empty"}, node_id=None)
        outside = self.micro(occupancy_changes=None, node_id=None, reason="O a0001 is not part of a reactive site")
        self.assertEqual(
            surface_left_lines(known),
            [
                "it took a0019 of the surface along: no bound minimum from this start",
                "the surface left behind is a0019: empty (n_0019empty), an intermediate of this campaign: "
                "a lattice route leads there from this one",
            ],
        )
        self.assertIn("a possible intermediate that this campaign does not list", surface_left_lines(unlisted)[1])
        self.assertIn("the sites of this campaign cannot describe (O a0001 is not", surface_left_lines(outside)[1])
        released = [{"formula": "O₂", "atoms": ["a0019", "m0038"], "from": ["a0040"]}]
        for micro, expected in (
            (known, None),
            (unlisted, "a0019: empty (after O₂ left with a surface atom)"),
            (outside, None),
        ):
            topology = {"status": "changed_connectivity_needs_review", "microstate": micro, "released": released}
            self.assertEqual(new_intermediate(topology), expected)
            # no energy for the intermediate it started from, and no decision to take
            self.assertEqual(classify({"node_id": "n", "final_topology": topology}), ("no_bound_minimum", None))


@unittest.skipIf(os.environ.get("MECHANET_SKIP_SLOW"), "slow build disabled by environment")
class RutileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env = patch.dict(os.environ, {"MECHANET_CONFIG_DIR": str(cls.tmp / "config")})
        cls.env.start()
        project = cls.tmp / "p"
        shutil.copytree(RUO2, project)
        cls.full = cls.tmp / "full"
        generate(project / "project.yaml", cls.full)
        # the same slab and sites with the adsorbate route only: no vacancy of the bridging O
        data = yaml.safe_load((project / "project.yaml").read_text())
        data["scope"]["families"] = ["aem"]
        (project / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        cls.aem = cls.tmp / "aem"
        generate(project / "project.yaml", cls.aem)

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def release(self, campaign, partner=None):
        """*O on the open metal site, with its O 4.5 A up; ``partner`` leaves with it as O2."""
        network = read_json(campaign / "network.json")
        node = next(n for n in network["nodes"] if (n.get("occupancy") or {}) == {SITE: "O"} and not n.get("coupling"))
        state = campaign / "states" / node["state_ids"][0]
        meta, start = read_json(state / "state.json"), read_poscar(state / "POSCAR")
        ids = meta["atom_ids"]
        positions = start.positions.copy()
        positions[ids.index(SITE)] += np.array([0.0, 0.0, 4.5])
        if partner is not None:
            positions[ids.index(partner)] = positions[ids.index(SITE)] + np.array([1.21, 0.0, 0.0])
        path = self.tmp / "moved.vasp"
        write_poscar(path, Structure(start.symbols, positions, start.cell, start.mobile, start.ids))
        config = read_json(campaign / "config.json")
        return topology_check(state, meta, config, geometry=path, use_review=False), meta, network

    def test_the_adsorbate_alone_is_released_as_before(self):
        topology, meta, _network = self.release(self.full)
        self.assertEqual(topology["microstate"]["status"], "adsorbate_released")
        self.assertNotIn("surface_left", topology["microstate"])
        self.assertEqual(describe_changes(topology, meta)[-1], "no bound minimum from this start")

    def test_with_a_bridging_oxygen_the_vacancy_is_named(self):
        topology, meta, network = self.release(self.full, BRIDGE)
        micro = topology["microstate"]
        self.assertFalse(topology["valid_for_assigned_node"])
        self.assertEqual(micro["status"], "surface_atom_released")
        self.assertEqual(micro["taken_surface_atoms"], [BRIDGE])
        self.assertEqual([(m["formula"], m["atoms"]) for m in topology["released"]], [("O₂", [BRIDGE, SITE])])
        self.assertTrue(micro["reason"].endswith(f"taking O {BRIDGE} of the surface along"), micro["reason"])
        vacancy = next(n for n in network["nodes"] if (n.get("occupancy") or {}) == {BRIDGE: "empty"})
        left = micro["surface_left"]
        self.assertEqual((left["occupancy_changes"], left["coupling"]), ({BRIDGE: "empty"}, None))
        self.assertEqual(left["matched_node_id"], vacancy["id"])
        self.assertEqual(left["node_id"], vacancy.get("computed_by") or vacancy["id"])
        self.assertEqual(left["delta_atoms"], vacancy["delta_atoms"])
        camp = _Campaign(self.full)
        lines = describe_changes(topology, meta, camp.occupancy_label)
        self.assertEqual(lines[0], f"O₂ (O {BRIDGE}, O {SITE}) released from Ru a0038 and Ru a0039 and Ru a0045")
        self.assertEqual(lines[-2], f"it took O {BRIDGE} of the surface along: no bound minimum from this start")
        self.assertEqual(
            lines[-1],
            f"the surface left behind is μ-□₁₉ ({vacancy['id']}), an intermediate of this campaign: "
            "a lattice route leads there from this one",
        )
        self.assertIsNone(new_intermediate(topology, camp.occupancy_label))  # nothing new: the campaign has it

    def test_a_campaign_without_the_vacancy_is_told_what_it_lacks(self):
        topology, meta, _network = self.release(self.aem, BRIDGE)
        left = topology["microstate"]["surface_left"]
        self.assertEqual((left["occupancy_changes"], left["node_id"]), ({BRIDGE: "empty"}, None))
        camp = _Campaign(self.aem)
        self.assertEqual(
            describe_changes(topology, meta, camp.occupancy_label)[-1],
            "the surface left behind is μ-□₁₉, a possible intermediate that this campaign does not list: "
            "mechanet review --discover can add it, with a start made from this structure without the molecule",
        )
        self.assertEqual(new_intermediate(topology, camp.occupancy_label), "μ-□₁₉ (after O₂ left with a surface atom)")

    def test_a_lattice_oxygen_that_is_no_site_is_named_for_declaring(self):
        sites = {s["id"] for s in read_json(self.full / "sites.json")["sites"]}
        _topology, meta, network = self.release(self.full)
        node = next(n for n in network["nodes"] if (n.get("occupancy") or {}) == {SITE: "O"} and not n.get("coupling"))
        start = read_poscar(self.full / "states" / node["state_ids"][0] / "POSCAR")
        ids = meta["atom_ids"]
        own = start.positions[ids.index(SITE)]
        lattice = [
            i
            for i, s in enumerate(start.symbols)
            if s == "O" and re.fullmatch(r"a\d{4}", ids[i]) and ids[i] not in sites
        ]
        nearest = min(lattice, key=lambda i: np.linalg.norm(minimum_image(start.positions[i] - own, start.cell)))
        topology, meta, _network = self.release(self.full, ids[nearest])
        micro = topology["microstate"]
        self.assertEqual((micro["status"], micro["taken_surface_atoms"]), ("surface_atom_released", [ids[nearest]]))
        self.assertIsNone(micro["surface_left"]["occupancy_changes"])
        self.assertEqual(
            micro["surface_left"]["reason"],
            f"O {ids[nearest]} is not part of a reactive site: declare it as a site to follow this route",
        )
        self.assertIn("that the sites of this campaign cannot describe", describe_changes(topology, meta)[-1])

    def test_the_surface_left_behind_gets_the_builders_start_with_the_relaxed_positions(self):
        """What review --discover writes for the vacancy: the atoms, order, masks and bonds the
        builder gives that intermediate, at the positions of the relaxed structure without the O2."""
        topology, meta, _network = self.release(self.aem, BRIDGE)
        changes = topology["microstate"]["surface_left"]["occupancy_changes"]
        new_id = node_id(changes, None)
        start = _left_behind_start(self.aem, meta["id"], meta, topology, changes, None, new_id, "μ-□₁₉")
        folder = self.aem / "states" / start
        self.addCleanup(shutil.rmtree, folder, True)
        # the ID the campaign with the bridge families gives that intermediate
        vacancy = next(
            n for n in read_json(self.full / "network.json")["nodes"] if n.get("occupancy") == {BRIDGE: "empty"}
        )
        self.assertEqual(start, f"{vacancy['id']}__v1")
        written, new = read_poscar(folder / "POSCAR"), read_json(folder / "state.json")
        self.assertEqual(set(meta["atom_ids"]) - set(new["atom_ids"]), {BRIDGE, SITE})
        relaxed = dict(zip(meta["atom_ids"], read_poscar(self.tmp / "moved.vasp").positions))
        for atom, position in zip(new["atom_ids"], written.positions):
            self.assertLess(float(np.linalg.norm(position - relaxed[atom])), 1e-8, atom)
        builder = Builder(read_json(self.aem / "config.json"), self.tmp / "again", base_dir=self.aem)
        st, freq, bonds, _info = builder.structure(dict(changes), None)
        self.assertEqual(new["atom_ids"], st.ids)
        self.assertEqual(new["symbols"], st.symbols)
        self.assertEqual(new["frequency_mobile"], freq.tolist())
        self.assertEqual(new["relax_mobile"], st.mobile.tolist())
        self.assertEqual(written.mobile.tolist(), st.mobile.tolist())
        self.assertEqual(new["expected_bonds"], bonds)
        self.assertEqual((new["node_id"], new["delta_atoms"]), (new_id, {"O": -1, "H": 0}))
        self.assertEqual(new["from_relaxed"]["removed_atom_ids"], sorted([BRIDGE, SITE]))
        self.assertEqual(new["from_relaxed"]["released"], ["O₂"])
        # it is not written twice, nor for an intermediate the atoms left behind do not make up
        with self.assertRaisesRegex(ValueError, "exists already"):
            _left_behind_start(self.aem, meta["id"], meta, topology, changes, None, new_id, "μ-□₁₉")
        other = {BRIDGE: "OH"}  # needs the O that left, and an H that is not there
        refused = self.aem / "states" / f"{node_id(other, None)}__v1"
        self.assertFalse(refused.exists())
        with self.assertRaisesRegex(ValueError, "the atoms left behind are not those of"):
            _left_behind_start(self.aem, meta["id"], meta, topology, other, None, node_id(other, None), "μ-OH")
        self.assertFalse(refused.exists())


if __name__ == "__main__":
    unittest.main()
