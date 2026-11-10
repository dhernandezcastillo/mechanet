"""The NiFe LDH example rebuilds its frozen reference build exactly (network, states, coordinates, masks).

Exactly, on x86-64 processors (Linux and Windows give the same build). Orientations that a
symmetry of the site maps onto each other score the same up to rounding in the last digits, and
which of them the search keeps follows that rounding: other processors (Apple Silicon) keep
another of the tied orientations for a few starts. There the network, the states, their atoms
and masks must still match, and at most a tenth of the starts may differ in coordinates.

Set MECHANET_SKIP_SLOW=1 to skip (about 35 s).
"""

import json
import os
import platform
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mechanet.project import generate
from mechanet.structure import minimum_image, read_poscar

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "nife_ldh"
X86 = platform.machine().lower() in ("x86_64", "amd64", "x86-64", "x64", "i386", "i686")


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
        moved = {}
        for sid in old_states:
            a = read_poscar(self.reference / "states" / sid / "POSCAR")
            b = read_poscar(self.campaign / "states" / sid / "POSCAR")
            self.assertEqual(a.symbols, b.symbols, sid)
            shift = max(float(np.linalg.norm(minimum_image(x - y, a.cell))) for x, y in zip(a.positions, b.positions))
            old_meta = json.loads((self.reference / "states" / sid / "state.json").read_text())
            new_meta = json.loads((self.campaign / "states" / sid / "state.json").read_text())
            if shift >= 1e-6:  # which start moved, and which placement (and score) each build chose
                old_variant, new_variant = old_meta.get("variant", {}), new_meta.get("variant", {})
                moved[sid] = {
                    "largest_shift_A": round(shift, 4),
                    "reference": old_variant.get("placement"),
                    "now": new_variant.get("placement"),
                    "reference_score": repr(old_variant.get("search_score")),
                    "now_score": repr(new_variant.get("search_score")),
                }
            for field in ("atom_ids", "relax_mobile", "frequency_mobile", "node_id"):
                self.assertEqual(old_meta[field], new_meta[field], f"{sid} {field}")
            self.assertEqual(
                {k: v for k, v in old_meta["delta_atoms"].items() if v},
                {k: v for k, v in new_meta["delta_atoms"].items() if v},
                sid,
            )
        report = "starting structures that moved:\n" + json.dumps(moved, indent=1)
        if X86:
            self.assertFalse(moved, report)
        else:  # equally scored orientations may be told apart differently (see the module docstring)
            self.assertLessEqual(len(moved), len(old_states) // 10, report)


if __name__ == "__main__":
    unittest.main()
