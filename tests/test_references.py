"""Molecular references: ideal-gas corrections, molecule inputs and reading results back."""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from support import EXAMPLE, FAKE_VASP

from mechanet.references import collect_references, gas_correction, molecule_inputs, write_template


class GasCorrectionTests(unittest.TestCase):
    def test_textbook_values(self):
        # H2 (PBE stretch ~4330 cm-1) and liquid-like water modes at 298.15 K and 1 bar
        h2 = gas_correction("H2", [4330.0, 0, 0, 0, 0, 0])
        h2o = gas_correction("H2O", [3850.0, 3740.0, 1590.0, 0, 0, 0, 0, 0, 0])
        self.assertGreater(h2, -0.06)
        self.assertLess(h2, -0.03)
        self.assertGreater(h2o, 0.04)
        self.assertLess(h2o, 0.10)
        # lower pressure lowers G by kT ln(p)
        self.assertAlmostEqual(
            gas_correction("H2O", [3850.0, 3740.0, 1590.0], pressure_bar=0.035) - h2o,
            8.617333e-5 * 298.15 * -3.352407,
            places=4,
        )


class MoleculeWorkflowTests(unittest.TestCase):
    def test_inputs_run_and_collect(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            fake = tmp / "fake_vasp.py"
            fake.write_text(FAKE_VASP)
            written = molecule_inputs(EXAMPLE / "inputs" / "INCAR_final", tmp / "mol", ("H2", "H2O"))
            self.assertEqual(len(written), 2)
            incar = (tmp / "mol" / "H2O" / "relax" / "INCAR").read_text()
            for tag in ("LDAU", "MAGMOM", "IDIPOL"):
                self.assertNotIn(f"\n{tag} =", incar)
            self.assertIn("ISPIN = 1", incar)
            self.assertIn("IVDW = 12", incar)  # dispersion kept from the slab protocol
            for name in ("H2", "H2O"):
                relax, freq = tmp / "mol" / name / "relax", tmp / "mol" / name / "freq"
                subprocess.run([sys.executable, "-I", str(fake)], cwd=relax, check=True, capture_output=True)
                shutil.copy(relax / "CONTCAR", freq / "POSCAR")
                subprocess.run([sys.executable, "-I", str(fake)], cwd=freq, check=True, capture_output=True)
            refs = tmp / "references.yaml"
            write_template(refs)
            result = collect_references(tmp / "mol", refs)
            data = yaml.safe_load(refs.read_text())
            self.assertEqual(sorted(result["molecules"]), ["H2", "H2O"])
            self.assertAlmostEqual(data["H2"]["energy_e0_eV"], -100.003, places=6)
            self.assertIsInstance(data["H2O"]["free_energy_correction_eV"], float)
            self.assertEqual(data["H2O"]["Gcorr_pressure_bar"], 1.0)
            self.assertEqual(data["oxygen_reference"]["delta_G_2H2O_to_O2_2H2_eV"], 4.92)  # template kept
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
