"""Local moments across intermediates: frame relabelling, cycle spin checks, variant solutions."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import SRC  # noqa: E402,F401  (puts src on sys.path)

from mechanet.magnetism import (  # noqa: E402
    FrameMaps,
    MomentTracker,
    magnetic_solutions,
    moments_table,
    short_label,
)
from mechanet.project import validate_project  # noqa: E402

# Slab: site metal Fe a0001 (site O a0005 above it), a near Ni a0002 at 3 A with its own site
# O a0006, a bridging O a0004, and a remote Ni a0003 9 A away. Cell 24 x 12 x 20 A.
SEED = """toy seed
1.0
24.0 0.0 0.0
0.0 12.0 0.0
0.0 0.0 20.0
Fe Ni O
1 2 3
Cartesian
0.0 0.0 5.0
3.0 0.0 5.0
9.0 0.0 5.0
1.5 0.0 5.0
0.0 0.0 6.8
3.0 0.0 6.8
"""
IDS = ["a0001", "a0002", "a0003", "a0004", "a0005", "a0006"]
SYMBOLS = ["Fe", "Ni", "Ni", "O", "O", "O"]
# The image node is the half-cell translation that swaps the two metal/site pairs.
SWAP = {"a0001": "a0002", "a0002": "a0001", "a0005": "a0006", "a0006": "a0005", "a0003": "a0003", "a0004": "a0004"}


def network():
    return {
        "reference": "ref",
        "nodes": [
            {"id": "ref", "occupancy": {}, "state_ids": ["ref"]},
            {"id": "nE", "occupancy": {"a0005": "empty"}, "state_ids": ["nE_v1"]},
            {"id": "nO", "occupancy": {"a0005": "O"}, "state_ids": ["nO_v1", "nO_v2", "nO_v3"]},
            {"id": "nOOH", "occupancy": {"a0005": "OOH"}, "state_ids": ["nOOH_v1"]},
            {
                "id": "nO_img",
                "occupancy": {"a0006": "O"},
                "state_ids": [],
                "computed_by": "nO",
                "symmetry": {"translation_fractional": [0.25, 0.0, 0.0], "site_map": {"a0006": "a0005"}},
            },
        ],
        "edges": [],
        "cycles": [],
        "symmetry": {
            "translations": [
                {
                    "fractional": [0.25, 0.0, 0.0],
                    "site_map": {"a0005": "a0006", "a0006": "a0005"},
                    "atom_id_map": SWAP,
                }
            ]
        },
    }


class MagnetismTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "inputs").mkdir()
        (self.root / "inputs" / "seed.vasp").write_text(SEED)
        sites = [
            {"id": "a0005", "kind": "terminal", "native": "OH", "metal_neighbors": [{"id": "a0001", "element": "Fe"}]},
            {"id": "a0006", "kind": "terminal", "native": "OH", "metal_neighbors": [{"id": "a0002", "element": "Ni"}]},
        ]
        (self.root / "sites.json").write_text(json.dumps({"schema_version": 1, "sites": sites}))
        (self.root / "config.json").write_text(json.dumps({"analysis": {}}))
        self.rows = []

    def tearDown(self):
        self.tmp.cleanup()

    def add_state(self, state_id, node_id, moments, energy=-100.0, extra=(), stage="02_final"):
        """moments: {label: value} for IDS (missing labels: the atom is absent) plus ``extra`` atoms."""
        ids = [i for i in IDS if i in moments] + [label for label, _ in extra]
        symbols = [s for i, s in zip(IDS, SYMBOLS) if i in moments] + [symbol for _, symbol in extra]
        folder = self.root / "states" / state_id
        folder.mkdir(parents=True)
        (folder / "state.json").write_text(json.dumps({"id": state_id, "atom_ids": ids, "symbols": symbols}))
        values = [moments[i] for i in ids]
        self.rows.append(
            {
                "state_id": state_id,
                "node_id": node_id,
                "stages": {stage: {"magmoms": values}},
                "topology": {"microstate": None},
                "final_topology": {"microstate": None},
                "final_valid": True,
                "final_assigned_node_id": node_id,
                "energy_final_e0_eV": energy,
            }
        )

    def standard_cycle(self, remote_ooh=1.6, fe_o=3.7):
        base = {"a0001": 4.1, "a0002": 1.6, "a0003": 1.6, "a0004": 0.02, "a0005": 0.02, "a0006": 0.02}
        self.add_state("ref", "ref", base)
        self.add_state("nE_v1", "nE", {**{k: v for k, v in base.items() if k != "a0005"}, "a0001": 3.9})
        self.add_state("nO_v1", "nO", {**base, "a0001": fe_o, "a0002": 0.9, "a0005": 0.65})
        self.add_state(
            "nOOH_v1",
            "nOOH",
            {**base, "a0001": 3.95, "a0003": remote_ooh, "a0005": 0.12, "d_a0005": 0.08},
            extra=[("d_a0005", "O")],
        )

    def tracker(self, selected=None, config=None):
        if selected is None:
            selected = {"ref": "ref", "nE": "nE_v1", "nO": "nO_v1", "nOOH": "nOOH_v1", "nO_img": "nO_v1"}
        return MomentTracker(self.root, network(), self.rows, selected, "02_final", "topology", config)

    def test_consistent_cycle_tracks_the_active_site(self):
        self.standard_cycle()
        result = self.tracker().cycle(["nE", "ref", "nO", "nOOH"])
        self.assertEqual(result["status"], "consistent", result["flags"])
        atoms = {a["atom_id"]: a for a in result["atoms"]}
        self.assertEqual(atoms["a0001"]["role"], "site metal")
        self.assertEqual(atoms["a0001"]["moments_muB"], [3.9, 4.1, 3.7, 3.95])
        self.assertEqual(atoms["a0005"]["role"], "bound")
        self.assertEqual(atoms["a0005"]["moments_muB"], [None, 0.02, 0.65, 0.12], "absent in the vacancy")
        self.assertEqual(atoms["d_a0005"]["moments_muB"], [None, None, None, 0.08])
        # the near Ni changes by 0.7 muB inside the active-site region: shown, not flagged
        self.assertEqual(atoms["a0002"]["role"], "largest change")
        self.assertEqual([a["role"] for a in result["atoms"]][:3], ["site metal", "bound", "outer"])
        self.assertEqual(result["active_sites"], ["a0005"])
        text = "\n".join(moments_table(result, short_label))
        self.assertIn("Fe a0001 (site metal)", text)
        self.assertIn("+0.65", text)

    def test_change_away_from_the_site_and_sign_flips_are_flagged(self):
        self.standard_cycle(remote_ooh=0.9)
        result = self.tracker().check(["nE", "ref", "nO", "nOOH"])
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(
            [(f["step"], f["atom_id"], f["kind"]) for f in result["flags"]],
            [(3, "a0003", "away_from_site"), (4, "a0003", "away_from_site")],
        )
        self.assertAlmostEqual(result["flags"][0]["distance_to_site_metal_A"], 9.0, places=2)
        # a larger threshold accepts the change
        quiet = self.tracker(config={"analysis": {"moment_change_muB": 0.8}}).check(["nE", "ref", "nO", "nOOH"])
        self.assertEqual(quiet["status"], "consistent")

    def test_sign_flip_at_the_site_metal_is_flagged(self):
        self.standard_cycle(fe_o=-3.7)
        result = self.tracker().check(["nE", "ref", "nO", "nOOH"])
        kinds = {(f["step"], f["atom_id"], f["kind"]) for f in result["flags"]}
        self.assertEqual(kinds, {(2, "a0001", "sign_flip"), (3, "a0001", "sign_flip")})

    def test_cycle_radius_is_configurable(self):
        self.standard_cycle(remote_ooh=0.9)
        # with a 2.0 A radius the near Ni (3 A from Fe) is no longer part of the active site
        result = self.tracker(config={"analysis": {"moment_site_radius_A": 2.0}}).check(["nE", "ref", "nO", "nOOH"])
        flagged = {(f["step"], f["atom_id"]) for f in result["flags"]}
        self.assertIn((2, "a0002"), flagged)
        self.assertIn((3, "a0002"), flagged)

    def test_missing_moments_make_the_check_incomplete(self):
        self.standard_cycle()
        result = self.tracker(selected={"ref": "ref", "nE": "nE_v1", "nO": "nO_v1"}).cycle(["nE", "ref", "nO", "nOOH"])
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["unchecked_steps"], [3, 4])
        self.assertEqual(result["missing"], [{"node_id": "nOOH", "reason": "no structure with an accepted energy"}])
        self.rows[0]["stages"]["02_final"]["magmoms"] = None
        moments, reason = self.tracker().node_moments("ref")
        self.assertIsNone(moments)
        self.assertIn("LORBIT", reason)
        empty = self.tracker(selected={}).cycle(["nE", "ref"])
        self.assertEqual(empty["status"], "not_available")

    def test_symmetry_image_takes_the_representatives_moments_relabelled(self):
        self.standard_cycle()
        moments, reason = self.tracker().node_moments("nO_img")
        self.assertIsNone(reason)
        self.assertEqual(moments["a0006"], ("O", 0.65), "the image's site O carries the representative's value")
        self.assertEqual(moments["a0002"], ("Fe", 3.7))
        self.assertEqual(moments["a0001"], ("Ni", 0.9))
        # a state whose relaxed occupancy is the image's is relabelled into the representative's frame
        self.rows[2]["topology"] = {"microstate": {"status": "same_occupancy", "matched_node_id": "nO_img"}}
        moments, _ = self.tracker().node_moments("nO")
        self.assertEqual(moments["a0006"], ("O", 0.65))

    def test_frame_maps_follow_the_translation_direction(self):
        cycle3 = {"a0001": "a0002", "a0002": "a0003", "a0003": "a0001"}
        net = {
            "nodes": [
                {"id": "rep", "occupancy": {"a0002": "O"}},
                {
                    "id": "img",
                    "occupancy": {"a0001": "O"},
                    "computed_by": "rep",
                    "symmetry": {"translation_fractional": [0.333333, 0.0, 0.0]},
                },
            ],
            "symmetry": {"translations": [{"fractional": [0.333333, 0.0, 0.0], "atom_id_map": cycle3, "site_map": {}}]},
        }
        maps = FrameMaps(net)
        # the image maps onto the representative (a0001 -> a0002), so the representative's
        # atoms read back into the image's frame with the inverse map
        self.assertEqual(maps.relabel(["a0002", "d_a0002", "a0003"], "img"), ["a0001", "d_a0001", "a0002"])
        self.assertEqual(maps.relabel(["a0001"], "rep", matched="img"), ["a0002"])
        self.assertEqual(maps.relabel(["a0001"], "rep"), ["a0001"])
        del net["nodes"][1]["symmetry"]
        self.assertIsNone(FrameMaps(net).relabel(["a0001"], "img"))

    def test_variants_in_different_magnetic_solutions(self):
        self.standard_cycle()
        base = {"a0001": 3.7, "a0002": 0.9, "a0003": 1.6, "a0004": 0.02, "a0005": 0.65, "a0006": 0.02}
        self.add_state("nO_v2", "nO", {**base, "a0001": 1.0}, energy=-99.96)
        self.add_state("nO_v3", "nO", {**base, "a0003": 1.5}, energy=-99.98)
        solutions = self.tracker().variants("nO")
        self.assertEqual([s["state_ids"] for s in solutions], [["nO_v1", "nO_v3"], ["nO_v2"]])
        self.assertAlmostEqual(solutions[1]["relative_eV"], 0.04)
        self.assertEqual(solutions[1]["differs_from_lowest"][0]["atom_id"], "a0001")
        self.assertEqual(solutions[1]["n_atoms_differing"], 1)
        self.assertIsNone(self.tracker().variants("nOOH"), "one variant: nothing to compare")

    def test_magnetic_solutions_group_from_the_lowest_energy(self):
        a = {"a0001": ("Fe", 4.0), "a0002": ("Ni", 1.6), "a0009": ("H", 0.0)}
        flipped = {**a, "a0002": ("Ni", -1.6)}
        low_spin = {**a, "a0001": ("Fe", 1.0)}
        hydrogen_only = {**a, "a0009": ("H", 0.9)}
        groups = magnetic_solutions(
            [("v4", 0.3, low_spin), ("v2", 0.1, flipped), ("v1", 0.0, a), ("v3", 0.2, hydrogen_only)]
        )
        self.assertEqual([g["state_ids"] for g in groups], [["v1", "v3"], ["v2"], ["v4"]])
        self.assertEqual(groups[1]["differs_from_lowest"][0]["this_muB"], -1.6)
        self.assertEqual(groups[0].get("differs_from_lowest"), None)

    def test_short_labels(self):
        self.assertEqual(short_label("ref"), "native")
        self.assertEqual(short_label("n_0065O_36a161"), "0065O")
        self.assertEqual(short_label("n_0061H2O_0065OOH_f4aef6"), "0061H2O_0065OOH")

    def test_project_validation_of_the_thresholds(self):
        errors = validate_project({"analysis": {"moment_change_muB": -1, "moment_site_radius_A": "far"}})
        text = "\n".join(errors)
        self.assertIn("analysis.moment_change_muB must be a positive number", text)
        self.assertIn("analysis.moment_site_radius_A must be a positive number", text)
        self.assertNotIn("unknown analysis keys", text)


if __name__ == "__main__":
    unittest.main()
