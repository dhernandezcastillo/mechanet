"""The NiFe LDH example rebuilds its frozen reference build exactly (network, states, coordinates, masks).

Set MECHANET_SKIP_SLOW=1 to skip (about 35 s).
"""

import json
import os
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mechanet.project import generate
from mechanet.structure import minimum_image, read_poscar

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "nife_ldh"


@unittest.skipIf(os.environ.get("MECHANET_SKIP_SLOW"), "slow regression disabled by environment")
class LDHRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        with tarfile.open(EXAMPLE / "reference" / "campaign.tar.xz") as tar:
            tar.extractall(cls.tmp, filter="data")
        cls.reference = cls.tmp / "reference"
        cls.campaign = cls.tmp / "campaign"
        cls.report = generate(EXAMPLE / "project.yaml", cls.campaign)
        cls.old = json.loads((cls.reference / "network.json").read_text())
        cls.new = json.loads((cls.campaign / "network.json").read_text())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_counts(self):
        r = self.report
        self.assertEqual(
            (
                r["candidate_nodes"],
                r["computed_nodes"],
                r["symmetry_equivalent_nodes"],
                r["unique_initial_structures"],
                r["proposed_edges"],
                r["complete_stoichiometric_cycles"],
                r["symmetry_unique_cycles"],
            ),
            (189, 95, 94, 128, 488, 368, 184),
        )

    def test_network_is_identical(self):
        old_nodes = {n["id"]: n for n in self.old["nodes"]}
        new_nodes = {n["id"]: n for n in self.new["nodes"]}
        self.assertEqual(set(old_nodes), set(new_nodes))
        for key, node in old_nodes.items():
            for field in (
                "state_ids",
                "computed_by",
                "occupancy",
                "coupling",
                "family",
                "site",
                "geometry_status",
                "symmetry_multiplicity",
            ):
                self.assertEqual(node.get(field), new_nodes[key].get(field), f"{key} {field}")
            self.assertEqual(sorted(node["families"]), sorted(new_nodes[key]["families"]))
        old_edges = {e["id"]: e for e in self.old["edges"]}
        new_edges = {e["id"]: e for e in self.new["edges"]}
        self.assertEqual(set(old_edges), set(new_edges))
        for key, edge in old_edges.items():
            for field in ("from", "to", "water", "pairs", "oxygen", "family"):
                self.assertEqual(edge[field], new_edges[key][field], f"{key} {field}")
        old_cycles = {c["id"]: c["edge_ids"] for c in self.old["cycles"]}
        self.assertEqual(old_cycles, {c["id"]: c["edge_ids"] for c in self.new["cycles"]})

    def test_starting_structures_are_identical(self):
        old_states = sorted(p.name for p in (self.reference / "states").iterdir())
        self.assertEqual(old_states, sorted(p.name for p in (self.campaign / "states").iterdir()))
        worst = 0.0
        for sid in old_states:
            a = read_poscar(self.reference / "states" / sid / "POSCAR")
            b = read_poscar(self.campaign / "states" / sid / "POSCAR")
            self.assertEqual(a.symbols, b.symbols, sid)
            worst = max(
                worst,
                max(float(np.linalg.norm(minimum_image(x - y, a.cell))) for x, y in zip(a.positions, b.positions)),
            )
            old_meta = json.loads((self.reference / "states" / sid / "state.json").read_text())
            new_meta = json.loads((self.campaign / "states" / sid / "state.json").read_text())
            for field in ("atom_ids", "relax_mobile", "frequency_mobile", "node_id"):
                self.assertEqual(old_meta[field], new_meta[field], f"{sid} {field}")
            self.assertEqual(
                {k: v for k, v in old_meta["delta_atoms"].items() if v},
                {k: v for k, v in new_meta["delta_atoms"].items() if v},
                sid,
            )
        self.assertLess(worst, 1e-6)


if __name__ == "__main__":
    unittest.main()
