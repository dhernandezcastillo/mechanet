"""Surface Pourbaix diagram: exact envelopes and regions, formation lines."""

import math
import unittest

import numpy as np

from mechanet.pourbaix import _coefficients, formation_lines, lower_envelope, node_label, stability_regions
from mechanet.reactions import OER
from mechanet.thermo import Conditions

NERNST = Conditions().nernst


class EnvelopeTests(unittest.TestCase):
    def test_lower_envelope_of_three_lines(self):
        # native (0), *OH from 0.8 V, *O from 1.0 V (RHE-like, no pH dependence)
        a = np.array([0.0, 0.8, 1.8])
        b = np.array([0.0, -1.0, -2.0])
        segments = lower_envelope(a, b, -0.5, 2.5)
        self.assertEqual([k for _, _, k in segments], [0, 1, 2])
        self.assertAlmostEqual(segments[0][1], 0.8)
        self.assertAlmostEqual(segments[1][1], 1.0)
        self.assertEqual(segments[-1][1], 2.5)

    def test_skipped_phase_and_ties(self):
        # *OH is never stable when *O crosses first
        segments = lower_envelope(np.array([0.0, 1.0, 1.6]), np.array([0.0, -1.0, -2.0]), 0.0, 2.0)
        self.assertEqual([k for _, _, k in segments], [0, 2])
        self.assertAlmostEqual(segments[0][1], 0.8)
        # identical lines: the first one wins and nothing flickers
        segments = lower_envelope(np.array([0.0, 0.0]), np.array([-1.0, -1.0]), 0.0, 1.0)
        self.assertEqual(segments, [(0.0, 1.0, 0)])

    def test_regions_tile_the_window(self):
        rng = np.random.default_rng(3)
        A = rng.uniform(-1, 3, 8)
        B = -rng.integers(0, 4, 8).astype(float)
        C = B * NERNST * rng.uniform(0, 1, 8)
        regions = stability_regions(A, B, C, (0.0, 14.0), (-1.0, 3.0))
        total = sum(area for _, _, area, _ in regions)
        self.assertAlmostEqual(total, 14.0 * 4.0, places=6)
        # every region is where its line is lowest (checked at the centroid)
        for index, _, _, (ph, u) in regions:
            values = A + B * u + C * ph
            self.assertEqual(int(np.argmin(values)), index)

    def test_rhe_conversion_removes_pcet_ph_dependence(self):
        lines = [{"A_eV": 0.8, "B_eV_per_V": -1.0, "C_eV_per_pH": -NERNST}]
        _, B, C = _coefficients(lines, "RHE", NERNST)
        self.assertAlmostEqual(B[0], -1.0)
        self.assertAlmostEqual(C[0], 0.0, places=12)


class FormationLineTests(unittest.TestCase):
    def data(self, energies):
        nodes = [
            {"id": "ref", "occupancy": {"s1": "empty"}, "delta_atoms": {}},
            {"id": "oh", "occupancy": {"s1": "OH"}, "delta_atoms": {"O": 1, "H": 1}},
            {"id": "o", "occupancy": {"s1": "O"}, "delta_atoms": {"O": 1}},
            {"id": "o_image", "occupancy": {"s2": "O"}, "delta_atoms": {"O": 1}, "computed_by": "o"},
        ]
        reservoirs = {"valid": True, "errors": [], "mu_H2_eV": -6.8, "mu_H2O_eV": -14.2, "mu_O2_eV": -9.68}
        return {
            "network": {"nodes": nodes, "reference": "ref"},
            "selected": energies,
            "main": OER(),
            "reservoirs": reservoirs,
            "settings": {"temperature_K": 298.15},
            "thermo_cfg": {},
        }

    def test_lines_follow_the_che(self):
        mu_h = 0.5 * -6.8
        # choose node energies so that *OH forms at 0.80 V and *O at 1.00 V vs RHE
        g_ref = -300.0
        g_oh = g_ref + (-14.2) - mu_h + 0.80
        g_o = g_ref + (-14.2) - 2 * mu_h + 0.80 + 1.00
        lines, missing, conditions = formation_lines(self.data({"ref": g_ref, "oh": g_oh, "o": g_o, "o_image": g_o}))
        self.assertEqual(missing, [])
        by_id = {line["node_id"]: line for line in lines}
        self.assertEqual(sorted(by_id), ["o", "oh", "ref"])  # symmetry images are not repeated
        self.assertEqual(by_id["o"]["symmetry_images"], 1)
        self.assertEqual(by_id["oh"]["electrons"], 1)
        self.assertAlmostEqual(by_id["oh"]["A_eV"], 0.80)
        self.assertAlmostEqual(by_id["o"]["A_eV"], 1.80)
        self.assertAlmostEqual(by_id["o"]["C_eV_per_pH"], -2 * conditions.nernst)
        self.assertEqual(node_label({"occupancy": {"s1": "O"}}, {"s1": "empty"}), "s1:O")
        A, B, C = _coefficients(lines, "RHE", conditions.nernst)
        order = [lines[k]["node_id"] for _, _, k in lower_envelope(A, B, 0.0, 2.0)]
        self.assertEqual(order, ["ref", "oh", "o"])

    def test_missing_reference_energy_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "reference surface"):
            formation_lines(self.data({"oh": -1.0}))

    def test_missing_nodes_are_listed(self):
        lines, missing, _ = formation_lines(self.data({"ref": 0.0, "oh": -15.0}))
        self.assertEqual(missing, ["o"])
        self.assertTrue(all(math.isfinite(line["A_eV"]) for line in lines))


if __name__ == "__main__":
    unittest.main()
