"""Inspector and builder on the community test systems: RuO2(110), IrO2(110), Pt(111), NiFe LDH."""

import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from mechanet.builder import build
from mechanet.fileio import read_json
from mechanet.inspector import inspect_slab
from mechanet.project import load_project, resolve_config
from mechanet.structure import minimum_image, read_poscar
from mechanet.validation import validate_campaign

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def config_of(example, families=None, **scope):
    config = resolve_config(load_project(EXAMPLES / example / "project.yaml"), cluster=False)
    if families is not None:
        config["scope"]["families"] = list(families)
    config["scope"].update(scope)
    return config


class InspectorTests(unittest.TestCase):
    def test_rutile_110_sites(self):
        for example, slab, metal in (("ruo2_110", "RuO2_110.vasp", "Ru"), ("iro2_110", "IrO2_110.vasp", "Ir")):
            report = inspect_slab(EXAMPLES / example / "slab" / slab)
            kinds = Counter((s["kind"], s["native"]) for s in report["proposal"]["sites"])
            self.assertEqual(kinds, {("bridge", "O"): 2, ("open_metal", "empty"): 2}, example)
            self.assertEqual(report["vacuum"]["axis"], "c")
            self.assertEqual(report["bulk_like_coordination"][metal]["cn"], 6)
            self.assertTrue(all(s["cn"] == 5 for s in report["open_metal_sites"]))
            self.assertEqual(len(report["symmetry"]["site_classes"]), 2)  # one bridge class, one CUS class

    def test_pt111_top_sites(self):
        report = inspect_slab(EXAMPLES / "pt_111" / "slab" / "Pt_111.vasp")
        self.assertEqual(len(report["open_metal_sites"]), 9)
        self.assertEqual(len(report["symmetry"]["site_classes"]), 1)
        self.assertEqual(len(report["symmetry"]["translations"]), 8)
        self.assertTrue(all(abs(s["direction"][2] - 1.0) < 1e-6 for s in report["open_metal_sites"]))

    def test_ldh_sites_match_the_published_mask(self):
        slab = EXAMPLES / "nife_ldh" / "slab"
        report = inspect_slab(slab / "CONTCAR_seed", mask_path=slab / "CONTCAR_frequency_mask")
        self.assertEqual(report["comparison_with_mask"]["recall"], 1.0)
        self.assertEqual(report["comparison_with_mask"]["precision"], 1.0)
        self.assertEqual(report["vacuum"]["axis"], "a")
        self.assertFalse(report["vacuum"]["normal_is_cartesian_z"])
        sites = {s["id"]: (s["kind"], s["native"]) for s in report["proposal"]["sites"]}
        self.assertEqual(
            sites,
            {
                "a0061": ("bridge", "OH"),
                "a0062": ("bridge", "O"),
                "a0063": ("bridge", "OH"),
                "a0064": ("bridge", "O"),
                "a0065": ("terminal", "OH"),
                "a0066": ("terminal", "OH"),
                "a0067": ("terminal", "OH"),
                "a0068": ("terminal", "OH"),
            },
        )
        self.assertEqual(len(report["unbound_molecules"]), 8)
        self.assertEqual({p["H"] for p in report["shared_protons"]}, {"a0085", "a0087"})


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def build(self, name, config, example):
        out = self.tmp / name
        report = build(config, out, base_dir=EXAMPLES / example)
        validation = validate_campaign(out)
        self.assertTrue(validation["valid"], validation["errors"][:5])
        return out, report

    def test_ruo2_oer_core(self):
        out, report = self.build("ruo2", config_of("ruo2_110"), "ruo2_110")
        self.assertEqual(
            (
                report["candidate_nodes"],
                report["computed_nodes"],
                report["proposed_edges"],
                report["complete_stoichiometric_cycles"],
                report["symmetry_unique_cycles"],
            ),
            (63, 35, 146, 100, 50),
        )
        self.assertEqual(report["nodes_without_viable_start"], [])
        # adsorbates on the CUS Ru sit at sensible Ru-O distances
        for sid, expected in (("n_m0038OH_9b2a79__v1", 1.95), ("n_m0038O_d0b786__v1", 1.75)):
            st = read_poscar(out / "states" / sid / "POSCAR")
            meta = read_json(out / "states" / sid / "state.json")
            ru = st.positions[meta["atom_ids"].index("a0038")]
            new = [i for i, aid in enumerate(meta["atom_ids"]) if not aid.startswith("a") and st.symbols[i] == "O"]
            distance = min(np.linalg.norm(minimum_image(st.positions[i] - ru, st.cell)) for i in new)
            self.assertAlmostEqual(distance, expected, delta=0.05)

    def test_ruo2_cer(self):
        _, report = self.build("cer", config_of("ruo2_110_cer"), "ruo2_110_cer")
        network = read_json(self.tmp / "cer" / "network.json")
        self.assertEqual(network["reaction"], "cer")
        self.assertEqual(Counter(c["family"] for c in network["cycles"]), {"cer_volmer_heyrovsky": 2, "cer_ocl": 4})
        for edge in network["edges"]:
            self.assertIn("reservoirs", edge)

    def test_coverage_family_writes_nodes_only(self):
        config = config_of("ruo2_110", ["coverage"], coverage={"sites": ["m0038", "a0016"]})
        _, report = self.build("cov", config, "ruo2_110")
        self.assertEqual(report["complete_stoichiometric_cycles"], 0)
        self.assertGreater(report["candidate_nodes"], 5)

    def test_pt111_aem_uses_symmetry(self):
        _, report = self.build("pt", config_of("pt_111"), "pt_111")
        self.assertEqual(report["computed_nodes"] * 9 - 8, report["candidate_nodes"])  # one class of nine sites
        self.assertEqual(report["symmetry_unique_cycles"], 2)


if __name__ == "__main__":
    unittest.main()
