"""Surfaces of a campaign: candidates for the mechanism, compositions, hull, names and figures.

The designed case is the adsorbate evolution on one site: the bare site (0 eV), *OH at
0.80 - U, *O at 2.40 - 2U and *OOH at 3.80 - 3U (eV, U vs RHE). The bare site is stable up to
0.80 V, *OH up to 1.50 V, *OOH above; *O is never stable and comes closest, 0.10 eV, at 1.50 V.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mechanet.fileio import write_json
from mechanet.pourbaix import lower_envelope
from mechanet.surfaces import (
    DEFAULT_MARGIN_EV,
    candidates,
    composition_text,
    compositions,
    envelope_gap,
    hull,
    lead,
    level_title,
    names,
)

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def designed():
    lines = [
        {"node_id": "ref", "label": "native", "electrons": 0, "delta_atoms": {}},
        {"node_id": "oh", "label": "m1:OH", "electrons": 1, "delta_atoms": {"O": 1, "H": 1}},
        {"node_id": "o", "label": "m1:O", "electrons": 2, "delta_atoms": {"O": 1}},
        {"node_id": "ooh", "label": "m1:OOH", "electrons": 3, "delta_atoms": {"O": 2, "H": 1}},
    ]
    return lines, np.array([0.0, 0.80, 2.40, 3.80]), np.array([0.0, -1.0, -2.0, -3.0])


class EnvelopeGapTests(unittest.TestCase):
    def test_designed_distances(self):
        _, a, b = designed()
        gap, at = envelope_gap(a, b, (1.0, 1.8))
        self.assertTrue(np.allclose(gap, [0.2, 0.0, 0.1, 0.0]))
        self.assertAlmostEqual(at[0], 1.0)  # the bare site is closest at the low end of the window
        self.assertAlmostEqual(at[2], 1.5)  # *O is closest where *OH hands over to *OOH

    def test_exact_minimum_against_a_fine_scan(self):
        rng = np.random.default_rng(11)
        for _ in range(30):
            n = int(rng.integers(3, 30))
            a = rng.normal(size=n) * 2
            b = -rng.integers(0, 7, size=n).astype(float)
            lo = float(rng.uniform(-1, 1))
            hi = lo + float(rng.uniform(0.2, 2.5))
            gap, at = envelope_gap(a, b, (lo, hi))
            grid = np.linspace(lo, hi, 20001)
            values = a[:, None] + b[:, None] * grid[None, :]
            scanned = (values - values.min(axis=0)[None, :]).min(axis=1)
            self.assertTrue(np.all(gap <= scanned + 1e-12))  # never above what a scan finds
            self.assertTrue(np.allclose(gap, scanned, atol=2e-3))
            self.assertTrue(np.all((at >= lo) & (at <= hi)))

    def test_lead_of_a_stable_surface(self):
        _, a, b = designed()
        distance, where, other = lead(a, b, 1, (1.0, 1.5))
        # *OH is 0.20 eV below the bare site at 1.0 V, 0.40 eV below it and *O at 1.2 V, 0.10 eV below *O at 1.5 V
        self.assertAlmostEqual(distance, 0.4)
        self.assertAlmostEqual(where, 1.2)
        self.assertIn(other, (0, 2))
        self.assertEqual(lead(a[:1], b[:1], 0, (0.0, 1.0)), (None, None, None))


class CandidateTests(unittest.TestCase):
    def test_stable_surfaces_first_then_the_close_ones(self):
        lines, a, b = designed()
        rows = candidates(lines, a, b, (1.0, 1.8), 0.2, (-0.5, 2.5), {"oh": "terminal 1 OH"})
        self.assertEqual([row["node_id"] for row in rows], ["oh", "ooh", "o", "ref"])
        self.assertEqual([row["rank"] for row in rows], [1, 2, 3, 4])
        oh, ooh, o, ref = rows
        self.assertEqual(oh["surface"], "terminal 1 OH")
        self.assertIsNone(ooh["surface"])
        self.assertTrue(oh["stable_in_window"] and ooh["stable_in_window"])
        self.assertEqual((oh["stable_from_V"], oh["stable_to_V"]), (0.8, 1.5))
        self.assertEqual((ooh["stable_from_V"], ooh["stable_to_V"]), (1.5, 2.5))
        self.assertAlmostEqual(oh["lead_eV"], 0.4)
        self.assertAlmostEqual(oh["lead_at_V"], 1.2)
        self.assertFalse(o["stable_in_window"])
        self.assertAlmostEqual(o["above_stable_eV"], 0.1)
        self.assertAlmostEqual(o["closest_at_V"], 1.5)
        self.assertIsNone(o["lead_eV"])
        self.assertIsNone(o["stable_from_V"])
        self.assertAlmostEqual(ref["above_stable_eV"], 0.2)
        # the bare site is stable below the window: its range is reported although it is no stable candidate
        self.assertEqual((ref["stable_from_V"], ref["stable_to_V"]), (-0.5, 0.8))

    def test_the_margin_decides_who_is_carried(self):
        lines, a, b = designed()
        self.assertEqual([r["node_id"] for r in candidates(lines, a, b, (1.0, 1.8), 0.15)], ["oh", "ooh", "o"])
        self.assertEqual([r["node_id"] for r in candidates(lines, a, b, (1.0, 1.8), 0.0)], ["oh", "ooh"])
        self.assertEqual(len(candidates(lines, a, b, (1.0, 1.8), 5.0)), 4)
        self.assertEqual(DEFAULT_MARGIN_EV, 0.2)

    def test_two_lines_that_cross_at_the_start_of_the_window(self):
        """A flat line and a falling one meet 2e-13 V inside the window: the falling one is the
        stable surface of the window, not the one that is lower by 1e-13 eV at its first point."""
        lo, hi = 1.0, 1.8
        a = np.array([0.5000000000005, 1.5000000000007])
        b = np.array([0.0, -1.0])
        at_start = a + b * lo
        self.assertLess(at_start[0], at_start[1])
        self.assertEqual([index for _, _, index in lower_envelope(a, b, lo, hi)], [1])
        lines = [{"node_id": f"s{index}", "label": f"s{index}", "electrons": -b[index]} for index in range(2)]
        rows = {row["node_id"]: row for row in candidates(lines, a, b, (lo, hi), 0.2)}
        self.assertTrue(rows["s1"]["stable_in_window"])
        self.assertFalse(rows["s0"]["stable_in_window"])

    def test_a_window_inside_one_stable_range(self):
        lines, a, b = designed()
        rows = candidates(lines, a, b, (0.9, 1.1), 0.05)
        self.assertEqual([r["node_id"] for r in rows], ["oh"])
        self.assertEqual((rows[0]["stable_from_V"], rows[0]["stable_to_V"]), (0.9, 1.1))  # the window is the range

    def test_bad_input_is_refused(self):
        lines, a, b = designed()
        with self.assertRaisesRegex(ValueError, "lo < hi"):
            candidates(lines, a, b, (1.5, 1.5))
        with self.assertRaisesRegex(ValueError, "margin"):
            candidates(lines, a, b, (1.0, 1.5), -0.1)


class CompositionTests(unittest.TestCase):
    def test_isomers_are_grouped_and_ranked(self):
        lines, a, b = designed()
        lines = lines + [
            {"node_id": "oh_b", "label": "m2:OH", "electrons": 1, "delta_atoms": {"O": 1, "H": 1}},
            {"node_id": "oh_c", "label": "m3:OH", "electrons": 1, "delta_atoms": {"H": 1, "O": 1}},
        ]
        a = np.append(a, [0.95, 0.83])
        b = np.append(b, [-1.0, -1.0])
        groups = compositions(lines, a, b, (1.0, 1.5), {"oh": "first"})
        self.assertEqual([g["composition"] for g in groups], ["as the slab", "+1 H +1 O", "+1 O", "+1 H +2 O"])
        hydroxyl = groups[1]
        self.assertEqual((hydroxyl["n_surfaces"], hydroxyl["lowest"], hydroxyl["lowest_surface"]), (3, "oh", "first"))
        self.assertAlmostEqual(hydroxyl["spread_eV"], 0.15)
        self.assertAlmostEqual(hydroxyl["second_eV"], 0.03)
        self.assertEqual([s["node_id"] for s in hydroxyl["surfaces"]], ["oh", "oh_c", "oh_b"])
        self.assertEqual(hydroxyl["above_stable_eV"], {"1": 0.0, "1.5": 0.0})
        self.assertEqual(groups[2]["above_stable_eV"], {"1": 0.6, "1.5": 0.1})
        self.assertIsNone(groups[0]["second_eV"])

    def test_composition_text(self):
        self.assertEqual(composition_text({}), "as the slab")
        self.assertEqual(composition_text({"H": -2, "O": 0}), "-2 H")
        self.assertEqual(composition_text({"O": 1, "Cl": 1}), "+1 Cl +1 O")


class HullTests(unittest.TestCase):
    def test_designed_hull(self):
        lines, a, b = designed()
        found = hull(lines, a, b, 1.4)
        self.assertEqual([v["node_id"] for v in found["vertices"]], ["ref", "oh", "ooh"])
        self.assertEqual([v["to_next_V"] for v in found["vertices"]], [0.8, 1.5, None])
        by_id = {p["node_id"]: p for p in found["points"]}
        self.assertAlmostEqual(by_id["o"]["deltaG_eV"], -0.4)
        self.assertFalse(by_id["o"]["on_hull"])
        self.assertTrue(by_id["o"]["lowest_for_its_electrons"])

    def test_the_corners_are_the_surfaces_that_are_stable_at_some_potential(self):
        rng = np.random.default_rng(5)
        for _ in range(40):
            n = int(rng.integers(4, 40))
            a = rng.normal(size=n) * 3
            b = -rng.integers(0, 9, size=n).astype(float)
            lines = [{"node_id": f"s{k}", "label": "", "electrons": -b[k], "delta_atoms": {}} for k in range(n)]
            found = hull(lines, a, b, float(rng.uniform(-1, 2)))
            stable = {f"s{index}" for _, _, index in lower_envelope(a, b, -500.0, 500.0)}
            self.assertEqual({v["node_id"] for v in found["vertices"]}, stable)
            transitions = sorted(v["to_next_V"] for v in found["vertices"] if v["to_next_V"] is not None)
            breaks = sorted(round(u_to, 6) for _, u_to, _ in lower_envelope(a, b, -500.0, 500.0)[:-1])
            self.assertTrue(np.allclose(transitions, breaks, atol=1e-5))


class NameTests(unittest.TestCase):
    def test_names_come_from_the_sites_of_the_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            network = {
                "nodes": [
                    {"id": "ref", "occupancy": {}},
                    {"id": "n1", "occupancy": {"m0002": "OOH", "a0001": "OH"}},
                    {"id": "n2", "occupancy": {"m0003": "OOH"}, "computed_by": "n1"},
                ]
            }
            self.assertEqual(names(tmp, network), {})  # a network without sites.json
            sites = [
                {"id": "a0001", "kind": "bridge", "native": "O"},
                {"id": "m0002", "kind": "open_metal", "native": "empty"},
                {"id": "m0003", "kind": "open_metal", "native": "empty"},
            ]
            write_json(Path(tmp) / "sites.json", {"schema_version": 1, "sites": sites})
            self.assertEqual(
                names(tmp, network),
                {"ref": "terminal 2 empty | bridge 1 O", "n1": "terminal 1 OOH, 1 empty | bridge 1 OH"},
            )

    def test_level_titles(self):
        self.assertEqual(level_title("CHE_with_partial_Hessian_free_energy"), "free energy (frequencies)")
        self.assertEqual(
            level_title("CHE_screening_final_E0_plus_tabulated_adsorbate_ZPE_TS"), "energy + tabulated ZPE − TS"
        )
        self.assertEqual(level_title("CHE_screening_final_E0_without_adsorbate_corrections"), "energy only")
        self.assertEqual(level_title("something else"), "something else")


@unittest.skipUnless(HAS_MATPLOTLIB, "figures need matplotlib")
class FigureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.lines, self.a, self.b = designed()
        self.names = {"ref": "terminal 1 empty", "oh": "terminal 1 OH", "o": "terminal 1 O", "ooh": "terminal 1 OOH"}

    def tearDown(self):
        self.tmp.cleanup()

    def written(self, path):
        self.assertTrue(Path(path).is_file())
        self.assertTrue(Path(path).with_suffix(".svg").is_file())
        self.assertGreater(Path(path).stat().st_size, 5000)

    def test_colours_follow_the_surface_and_run_out_into_neutral_ink(self):
        from mechanet.surface_plots import NEUTRAL, SLOTS, surface_colours

        colours = surface_colours(["a", "b", "a", *"cdefghij"])
        self.assertEqual([colours[k] for k in "abcdefgh"], list(SLOTS))
        self.assertEqual((colours["i"], colours["j"]), (NEUTRAL, NEUTRAL))

    def test_lines_hull_margins_isomers(self):
        from mechanet import surface_plots as plots

        self.written(
            plots.plot_lines(
                self.out / "lines.png",
                self.lines,
                self.a,
                self.b,
                (0.4, 2.2),
                "RHE",
                0,
                (1.0, 1.8),
                0.2,
                ["o"],
                self.names,
            )
        )
        self.written(plots.plot_lines(self.out / "plain.png", self.lines, self.a, self.b, (0.0, 2.0), "SHE", 7))
        self.written(
            plots.plot_hull(self.out / "hull.png", hull(self.lines, self.a, self.b, 1.4), "RHE", 0, self.names)
        )
        groups = compositions(self.lines, self.a, self.b, (1.0, 1.4, 1.8), self.names)
        self.written(plots.plot_margins(self.out / "margins.png", groups, (1.0, 1.4, 1.8), 0.2, "RHE", 0, "level"))
        lines = self.lines + [{"node_id": "oh2", "label": "m2:OH", "electrons": 1, "delta_atoms": {"O": 1, "H": 1}}]
        groups = compositions(lines, np.append(self.a, 0.9), np.append(self.b, -1.0), ())
        self.written(plots.plot_isomers(self.out / "isomers.png", groups, {"oh"}))

    def test_levels_with_one_that_is_not_available(self):
        from mechanet import surface_plots as plots

        segments = [
            {"from_V": 0.4, "to_V": 0.8, "node_id": "ref", "label": "native"},
            {"from_V": 0.8, "to_V": 1.5, "node_id": "oh", "label": "m1:OH"},
            {"from_V": 1.5, "to_V": 1.52, "node_id": "o", "label": "m1:O"},  # too narrow for a name or a number
            {"from_V": 1.52, "to_V": 2.2, "node_id": "ooh", "label": "m1:OOH"},
        ]
        levels = [
            {
                "level": "free_energy",
                "title": "free energy (frequencies)",
                "unavailable": "no frequency calculation yet",
            },
            {"level": "screening", "title": "energy + tabulated ZPE − TS", "n_surfaces": 4, "segments": segments},
            {"level": "energy", "title": "energy only", "n_surfaces": 4, "segments": segments[:2]},
        ]
        self.written(plots.plot_levels(self.out / "levels.png", levels, (0.4, 2.2), "RHE", 0, (1.0, 1.8), self.names))


if __name__ == "__main__":
    unittest.main()
