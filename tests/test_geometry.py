"""Builder geometry on the real NiFe LDH seed (examples/nife_ldh).

The campaign configuration is resolved from examples/nife_ldh/project.yaml; each test then
restricts its ``sites`` and ``scope.families``.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import EXAMPLE  # noqa: E402

from mechanet.builder import Builder  # noqa: E402
from mechanet.fileio import read_json  # noqa: E402
from mechanet.inspector import inspect_slab  # noqa: E402
from mechanet.project import load_project, resolve_config  # noqa: E402
from mechanet.structure import distance_matrix, minimum_image, read_poscar, slab_normal, write_poscar  # noqa: E402


def ldh_config(families, site_ids, **geometry):
    """Campaign config of the LDH example, restricted to some families and sites."""
    config = resolve_config(load_project(EXAMPLE / "project.yaml"), cluster=False)
    config["scope"]["families"] = list(families)
    config["sites"] = [s for s in config["sites"] if s["id"] in set(site_ids)]
    config["geometry"].update(geometry)
    return config


def same_geometry(builder, first, second, tol=0.05):
    """Two structures coincide up to relabelling identical elements."""
    return (
        builder._label_free_distance(
            list(zip(first.symbols, first.positions)), list(zip(second.symbols, second.positions))
        )
        <= tol
    )


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.seed_path = EXAMPLE / "slab" / "CONTCAR_seed"
        self.freq_path = EXAMPLE / "slab" / "CONTCAR_frequency_mask"

    def test_real_seed_and_mask(self):
        st = read_poscar(self.seed_path)
        self.assertEqual(len(st), 90)
        self.assertEqual(st.symbols.count("Fe"), 4)
        # The Builder's own reading of seed, mask and sites.
        with tempfile.TemporaryDirectory() as tmp:
            builder = Builder(
                ldh_config(["aem"], ["a0061", "a0062", "a0063", "a0064", "a0065", "a0066", "a0067", "a0068"]), tmp
            )
        self.assertEqual(int(builder.freq_mobile.sum()), 42)
        self.assertEqual(int(builder.seed.mobile.any(axis=1).sum()), 74)
        self.assertAlmostEqual(builder.frequency_mask_max_difference_A, 0.02, places=9)
        sites = builder.sites
        self.assertEqual(sites["a0062"]["protonation"], "O")
        self.assertEqual(sites["a0061"]["protonation"], "OH")
        self.assertEqual(sites["a0065"]["kind"], "terminal")
        self.assertEqual(sites["a0061"]["kind"], "bridge")
        self.assertEqual([m["index"] for m in sites["a0065"]["metal_neighbors"]], [3])
        # The slab inspector finds the same sites and the same partial-Hessian region from the seed alone.
        report = inspect_slab(self.seed_path, mask_path=self.freq_path)
        found = {s["id"]: s for s in report["ligand_sites"]}
        self.assertEqual(sorted(found), sorted(sites))
        for aid, site in sites.items():
            self.assertEqual((found[aid]["kind"], found[aid]["native"]), (site["kind"], site["protonation"]))
        self.assertEqual(found["a0065"]["metals"], ["a0003"])
        self.assertEqual(
            (report["comparison_with_mask"]["recall"], report["comparison_with_mask"]["precision"]), (1.0, 1.0)
        )

    def test_poscar_roundtrip_and_skewed_cell(self):
        st = read_poscar(self.seed_path)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "POSCAR"
            write_poscar(p, st)
            reread = read_poscar(p)
            self.assertEqual(st.symbols, reread.symbols)
            np.testing.assert_allclose(st.positions, reread.positions, atol=1e-10)
            np.testing.assert_equal(st.mobile, reread.mobile)
        displacement = st.positions[64] - st.positions[66]
        np.testing.assert_allclose(
            minimum_image(displacement, st.cell),
            minimum_image(displacement + st.cell[1] - st.cell[2], st.cell),
            atol=1e-10,
        )
        n = slab_normal(st.cell)
        self.assertAlmostEqual(n @ st.cell[1], 0)
        self.assertAlmostEqual(n @ st.cell[2], 0)
        self.assertGreater(n @ st.cell[0], 0)

    def test_single_site_cycle_and_constraints(self):
        cfg = ldh_config(["aem"], ["a0065"])
        with tempfile.TemporaryDirectory() as tmp:
            report = Builder(cfg, tmp).finish()
            net = read_json(Path(tmp) / "network.json")
            nodes = {n["id"]: n for n in net["nodes"]}
            edges = {e["id"]: e for e in net["edges"]}
            self.assertEqual(len(net["cycles"]), 2)
            for e in edges.values():
                a, b = nodes[e["from"]]["delta_atoms"], nodes[e["to"]]["delta_atoms"]
                self.assertEqual(b["O"] - a["O"], e["water"] - 2 * e["oxygen"])
                self.assertEqual(b["H"] - a["H"], 2 * e["water"] - e["pairs"])
            for c in net["cycles"]:
                self.assertEqual(
                    [sum(edges[e][k] for e in c["edge_ids"]) for k in ("water", "pairs", "oxygen")], [2, 4, 1]
                )
            seed = read_poscar(self.seed_path)
            for path in (Path(tmp) / "states").glob("*/state.json"):
                meta = read_json(path)
                st = read_poscar(path.parent / "POSCAR")
                self.assertEqual(meta["symbols"], st.symbols)
                self.assertEqual(len(st), len(meta["atom_ids"]))
                for j, aid in enumerate(meta["atom_ids"]):
                    if aid in seed.ids:
                        i = seed.ids.index(aid)
                        np.testing.assert_equal(meta["relax_mobile"][j], seed.mobile[i])
                    else:
                        self.assertEqual(meta["relax_mobile"][j], [True] * 3)
                        self.assertEqual(meta["frequency_mobile"][j], [True] * 3)
                self.assertEqual(st.symbols.count("O") - 48, meta["delta_atoms"]["O"])
                self.assertEqual(st.symbols.count("H") - 22, meta["delta_atoms"]["H"])
                if meta["variant"].get("binding_modes"):
                    ids = meta["atom_ids"]
                    m = st.positions[ids.index("a0003")]
                    op, od = st.positions[ids.index("a0065")], st.positions[ids.index("d_a0065")]
                    self.assertAlmostEqual(np.linalg.norm(minimum_image(op - m, st.cell)), 1.95)
                    self.assertAlmostEqual(np.linalg.norm(minimum_image(od - m, st.cell)), 1.95)
                    self.assertAlmostEqual(np.linalg.norm(minimum_image(od - op, st.cell)), 1.34)
            self.assertTrue(report["no_DFT_performed"])

    def test_hydrated_pair_and_bifunctional_cycles(self):
        cfg = ldh_config(
            ["aem", "surface_bridge", "terminal_bridge_coupling", "bifunctional", "hydrated"], ["a0061", "a0065"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            Builder(cfg, tmp).finish()
            net = read_json(Path(tmp) / "network.json")
            nodes = {n["id"]: n for n in net["nodes"]}
            edges = {e["id"]: e for e in net["edges"]}
            direct_bf = [c for c in net["cycles"] if c["id"].endswith("direct_water_attack")]
            self.assertEqual(len(direct_bf), 1)
            self.assertEqual(
                (
                    edges[direct_bf[0]["edge_ids"][0]]["water"],
                    edges[direct_bf[0]["edge_ids"][0]]["pairs"],
                    edges[direct_bf[0]["edge_ids"][0]]["oxygen"],
                ),
                (1, 1, 1),
            )
            pair_cycles = [c for c in net["cycles"] if c["family"] == "terminal_bridge_coupling"]
            self.assertEqual(len(pair_cycles), 12)
            self.assertTrue(any("via_OOH_OH" in c["id"] for c in net["cycles"]))
            self.assertTrue(any("via_OO_H2O" in c["id"] for c in net["cycles"]))
            for c in net["cycles"]:
                es = [edges[e] for e in c["edge_ids"]]
                self.assertEqual([sum(e[k] for e in es) for k in ("water", "pairs", "oxygen")], [2, 4, 1])
                self.assertEqual(es[0]["from"], es[-1]["to"])
            for e in edges.values():
                x, y = nodes[e["from"]]["delta_atoms"], nodes[e["to"]]["delta_atoms"]
                self.assertEqual(y["O"] - x["O"], e["water"] - 2 * e["oxygen"])
                self.assertEqual(y["H"] - x["H"], 2 * e["water"] - e["pairs"])
                self.assertIn("atom_bookkeeping", e)
            water_nodes = [n for n in nodes.values() if n.get("occupancy") == {"a0065": "H2O"}]
            self.assertEqual(len(water_nodes), 1)
            self.assertEqual(water_nodes[0]["delta_atoms"], {"O": 0, "H": 1})
            meta = read_json(Path(tmp) / "states" / water_nodes[0]["state_ids"][0] / "state.json")
            st = read_poscar(Path(tmp) / "states" / meta["id"] / "POSCAR")
            ids = meta["atom_ids"]
            o, h1, h2 = [st.positions[ids.index(a)] for a in ("a0065", "a0085", "w_a0065")]
            v1, v2 = minimum_image(h1 - o, st.cell), minimum_image(h2 - o, st.cell)
            self.assertAlmostEqual(np.linalg.norm(v1), 0.98)
            self.assertAlmostEqual(np.linalg.norm(v2), 0.98)
            angle = np.degrees(np.arccos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)))
            self.assertAlmostEqual(angle, 104.5)
            # Exner BF-I / BF-II are declared for the terminal donor and its bridge acceptor.
            bf = {c["id"]: c for c in net["cycles"] if c["id"].endswith(("_BF1", "_BF2"))}
            self.assertEqual(sorted(bf), ["bifunctional_a0065_a0061_BF1", "bifunctional_a0065_a0061_BF2"])
            steps = [
                (edges[e]["water"], edges[e]["pairs"], edges[e]["oxygen"])
                for e in bf["bifunctional_a0065_a0061_BF2"]["edge_ids"]
            ]
            self.assertEqual(steps, [(1, 1, 0), (0, 1, 0), (1, 0, 0), (0, 1, 0), (0, 1, 1)])
            steps = [
                (edges[e]["water"], edges[e]["pairs"], edges[e]["oxygen"])
                for e in bf["bifunctional_a0065_a0061_BF1"]["edge_ids"]
            ]
            self.assertEqual(steps, [(1, 1, 0), (0, 1, 0), (1, 1, 0), (0, 1, 1)])
            oo_oh = nodes[edges[bf["bifunctional_a0065_a0061_BF1"]["edge_ids"][2]]["to"]]
            self.assertEqual(
                {k: v for k, v in oo_oh["occupancy"].items() if v != {"a0061": "OH", "a0065": "OH"}[k]}, {"a0065": "OO"}
            )

    def test_wrong_proton_placement_is_not_relabelled_as_intended_node(self):
        cfg = ldh_config(["aem"], ["a0065", "a0068"])
        with tempfile.TemporaryDirectory() as tmp:
            builder = Builder(cfg, tmp)
            # A fixed placement (azimuth 0, tilt atan(0.57/0.82), torsion 180) puts the OOH H on a0068.
            placement = {
                "a0065": {
                    "tilt_deg": float(np.degrees(np.arctan2(0.57, 0.82))),
                    "azimuth_deg": 0.0,
                    "torsion_deg": 180.0,
                }
            }
            st, _, bonds, _ = builder.structure({"a0065": "OOH", "a0068": "O"}, None, placement)
            issues = builder._topology_issues(st, bonds, distance_matrix(st))
            self.assertIsNotNone(issues)
            self.assertIn(["a0068", "p_a0065"], issues["unexpected_short_contacts"])

    def test_translation_twins_are_not_recomputed(self):
        cfg = ldh_config(["surface_bridge"], ["a0061", "a0063"])
        with tempfile.TemporaryDirectory() as tmp:
            report = Builder(cfg, tmp).finish()
            net = read_json(Path(tmp) / "network.json")
            nodes = {n["id"]: n for n in net["nodes"]}
            self.assertEqual(len(net["symmetry"]["translations"]), 1)
            np.testing.assert_allclose(np.abs(net["symmetry"]["translations"][0]["fractional"]), [0, 0.5, 0], atol=1e-4)
            twins = [n for n in nodes.values() if n.get("computed_by")]
            self.assertEqual(len(twins), 4)  # empty, O, OOH, OO on a0063 (native OH is the shared reference)
            for twin in twins:
                self.assertEqual(twin["state_ids"], [])
                rep = nodes[twin["computed_by"]]
                self.assertTrue(rep["state_ids"])
                self.assertEqual(rep["delta_atoms"], twin["delta_atoms"])
                self.assertTrue(all(k.startswith("a0063") or v == "OH" for k, v in twin["occupancy"].items()))
            self.assertEqual(report["symmetry_equivalent_nodes"], 4)
            reps = [c for c in net["cycles"] if c["symmetry_representative"]]
            self.assertEqual(len(reps), len(net["cycles"]) // 2)
            for c in reps:
                self.assertEqual(len(c["symmetry_equivalent_cycles"]), 1)
            states = list((Path(tmp) / "states").glob("*/state.json"))
            self.assertFalse(any("0063" in p.parent.name for p in states))
            translation = net["symmetry"]["translations"][0]
            # The relaxed seed is only approximately symmetric: residuals are reported, not hidden.
            self.assertAlmostEqual(translation["max_residual_A"], 0.001617, places=5)
            self.assertAlmostEqual(translation["rms_residual_A"], 0.000549, places=5)
            self.assertEqual(translation["atom_id_map"]["a0065"], "a0067")
            self.assertEqual(net["symmetry"]["tolerance_A"], 0.005)
            self.assertTrue(all(nodes[t["computed_by"]]["symmetry_multiplicity"] == 2 for t in twins))

    def test_symmetry_tolerance_below_the_seed_residual_disables_reduction(self):
        cfg = ldh_config(["surface_bridge"], ["a0061", "a0063"], symmetry_tolerance_A=0.001)
        with tempfile.TemporaryDirectory() as tmp:
            report = Builder(cfg, tmp).finish()
            self.assertEqual(report["symmetry_equivalent_nodes"], 0)
            self.assertEqual(report["translation_symmetry"]["translations"], [])

    def test_symmetry_reduction_can_be_disabled_and_explicit_magmom_breaks_it(self):
        cfg = ldh_config(["surface_bridge"], ["a0061", "a0063"], symmetry_reduction=False)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(Builder(cfg, tmp).translations, [])
        cfg["geometry"]["symmetry_reduction"] = True
        cfg["vasp"]["magmom_by_id"] = {"a0001": 4.0, "a0003": -4.0}
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(Builder(cfg, tmp).translations, [])

    def test_placements_respect_floors_and_localize_shared_proton(self):
        cfg = ldh_config(["aem", "hydrated"], ["a0065", "a0068"])
        with tempfile.TemporaryDirectory() as tmp:
            report = Builder(cfg, tmp).finish()
            floors = report["clash_floors_A"]
            contacts = report["closest_nonbonded_contacts_over_all_starts_A"]
            self.assertGreaterEqual(contacts["H-H"], floors["H-H"] - 1e-9)
            self.assertGreaterEqual(contacts["H-O"], floors["O-H"] - 1e-9)
            water = [p for p in (Path(tmp) / "states").glob("n_0068H2O_*/state.json")]
            self.assertTrue(water)
            for path in water:
                meta = read_json(path)
                st = read_poscar(path.parent / "POSCAR")
                ids = meta["atom_ids"]
                self.assertIn("a0085", meta["variant"]["localized_shared_protons"])
                d = np.linalg.norm(
                    minimum_image(st.positions[ids.index("a0085")] - st.positions[ids.index("a0065")], st.cell)
                )
                self.assertAlmostEqual(d, 0.98, places=6)
                for h in ("a0088", "w_a0068"):
                    for other in ("a0071", "a0085"):
                        dist = np.linalg.norm(
                            minimum_image(st.positions[ids.index(h)] - st.positions[ids.index(other)], st.cell)
                        )
                        self.assertGreaterEqual(dist, 1.5)
                # Water H atoms keep a Ni-O-H angle of at least 100 degrees (anchor floor).
                o = st.positions[ids.index("a0068")]
                ni = st.positions[ids.index("a0005")]
                for h in ("a0088", "w_a0068"):
                    v1 = minimum_image(ni - o, st.cell)
                    v2 = minimum_image(st.positions[ids.index(h)] - o, st.cell)
                    angle = np.degrees(np.arccos(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2)))
                    self.assertGreaterEqual(angle, 100.0 - 1e-6)

    def test_label_swapped_water_is_the_same_geometry(self):
        cfg = ldh_config(["aem"], ["a0065"])
        with tempfile.TemporaryDirectory() as tmp:
            builder = Builder(cfg, tmp)
            occ = {"a0065": "H2O"}
            a, _, _, _ = builder.structure(
                occ, None, {"a0065": {"tilt_deg": 25.0, "azimuth_deg": 30.0, "roll_deg": 15.0}}
            )
            b, _, _, _ = builder.structure(
                occ, None, {"a0065": {"tilt_deg": 25.0, "azimuth_deg": 30.0, "roll_deg": 195.0}}
            )
            c, _, _, _ = builder.structure(
                occ, None, {"a0065": {"tilt_deg": 25.0, "azimuth_deg": 120.0, "roll_deg": 15.0}}
            )
            self.assertFalse(np.allclose(a.positions, b.positions))
            self.assertTrue(same_geometry(builder, a, b))
            self.assertFalse(same_geometry(builder, a, c))
            # Rolls of 15 and 165 degrees look 1.5 A apart by atom ID but are close without labels.
            d, _, _, _ = builder.structure(
                occ, None, {"a0065": {"tilt_deg": 25.0, "azimuth_deg": 30.0, "roll_deg": 165.0}}
            )
            by_id = max(np.linalg.norm(minimum_image(a.positions[i] - d.positions[i], a.cell)) for i in range(len(a)))
            self.assertGreater(by_id, 1.0)
            self.assertLess(builder._label_free_distance(builder._changed_atoms(a), builder._changed_atoms(d)), 0.75)

    def test_variants_of_a_node_are_distinct_without_labels(self):
        cfg = ldh_config(["pair_coupling", "hydrated"], ["a0065", "a0068"])
        with tempfile.TemporaryDirectory() as tmp:
            builder = Builder(cfg, tmp)
            builder.finish()
            for node in builder.nodes.values():
                structures = [builder.state_structures[s] for s in node["state_ids"]]
                for i in range(len(structures)):
                    for j in range(i):
                        d = builder._label_free_distance(
                            builder._changed_atoms(structures[i]), builder._changed_atoms(structures[j])
                        )
                        self.assertGreaterEqual(d, 0.75, node["id"])
            coupled = [n for n in builder.nodes.values() if n.get("coupling") and not n.get("computed_by")]
            self.assertTrue(coupled)
            self.assertTrue(all(len(n["state_ids"]) <= 2 for n in coupled))

    def test_adsorbate_magmom_without_twin_value_disables_symmetry(self):
        cfg = ldh_config(["surface_bridge"], ["a0061", "a0063"])
        cfg["vasp"]["magmom_by_id"] = {"d_a0061": 1.0}
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(Builder(cfg, tmp).translations, [])
        cfg["vasp"]["magmom_by_id"] = {"d_a0061": 1.0, "d_a0063": 1.0}
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(len(Builder(cfg, tmp).translations), 1)


if __name__ == "__main__":
    unittest.main()
