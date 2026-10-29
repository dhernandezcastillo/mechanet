"""A reaction plugin end to end: examples/plugins/her.py (HER on Pt(111)) through build, VASP
(synthetic), thermo, pathways and Pourbaix, with a new reservoir species and a cathodic reaction.

*H is designed to bind with DeltaG(* -> *H) = +0.09 eV at 0 V vs RHE, so the limiting
potential is -0.09 V, the overpotential 0.09 V, and *H covers the surface below -0.09 V.
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from support import FAKE_VASP, write_potcars

from mechanet.cli import main
from mechanet.fileio import read_json
from mechanet.plugins import default_registry
from mechanet.reactions import SPECIES
from mechanet.runner import run_state
from mechanet.structure import read_poscar

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "examples" / "plugins" / "her.py"
E_H2 = -6.77
DG_H = 0.09


def cli(*args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(["--json", *map(str, args)])
    if code != 0:
        raise AssertionError(f"mechanet {' '.join(map(str, args))} exited {code}: {out.getvalue()}")
    return json.loads(out.getvalue())


@unittest.skipIf(os.environ.get("MECHANET_SKIP_SLOW"), "slow plugin workflow disabled by environment")
class HERPluginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env = patch.dict(os.environ, {"MECHANET_CONFIG_DIR": str(cls.tmp / "config")})
        cls.env.start()
        project = cls.tmp / "pt"
        shutil.copytree(ROOT / "examples" / "pt_111", project)
        data = yaml.safe_load((project / "project.yaml").read_text())
        fake = cls.tmp / "fake_vasp.py"
        fake.write_text(FAKE_VASP)
        data.update(reaction="her", plugins=[str(PLUGIN)])
        data["scope"]["families"] = ["her_volmer_heyrovsky"]
        data["vasp"] = {
            "potcars": write_potcars(cls.tmp, ("Pt", "H")),
            "magmom_by_element": None,
            "command": [sys.executable, "-I", str(fake)],
        }
        data["frequency"] = {"central": "final"}
        data["thermodynamics"] = {"references": "inputs/references.json", "potentials_V": [0.0, -0.1]}
        (project / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        (project / "inputs" / "references.json").write_text(
            json.dumps({"references": {"H2": {"E": E_H2, "Gcorr": 0.0}}})
        )
        cls.campaign = project / "campaign"
        cls.generated = cli("generate", project / "project.yaml")
        for state in sorted((cls.campaign / "states").iterdir()):
            meta = read_json(state / "state.json")
            n_h = meta["delta_atoms"].get("H", 0)
            target = -100.0 + n_h * (0.5 * E_H2 + DG_H - 0.30)  # minus the tabulated ZPE - TS of one H
            (state / "ctl").mkdir(exist_ok=True)
            (state / "ctl" / "eshift_02_final").write_text(f"{target + 100.003:.6f}")
            x0 = read_poscar(state / "POSCAR").positions.reshape(-1)
            (state / "harmonic.json").write_text(
                json.dumps({"x0": x0.tolist(), "K": (10.0 * np.eye(len(x0))).tolist()})
            )
        cli("prepare", cls.campaign)
        for state in sorted((cls.campaign / "states").iterdir()):
            assert run_state(state, through="final", execute=True)["status"] == "complete"

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        shutil.rmtree(cls.tmp)

    def test_plugin_registers_species_reaction_adsorbate_family(self):
        registry = default_registry([PLUGIN])
        self.assertIn("her", registry.reactions)
        self.assertIn("H", registry.adsorbates)
        self.assertEqual(registry.families["her_volmer_heyrovsky"].origin, str(PLUGIN.resolve()))
        self.assertIn("H2", SPECIES)
        self.assertEqual(registry.reaction("her").electrons, -2)
        self.assertEqual(self.generated["complete_stoichiometric_cycles"], 9)
        self.assertEqual(self.generated["nodes_without_viable_start"], [])
        self.assertTrue(cli("validate", self.campaign)["valid"])

    def test_clean_metal_reference_has_no_frequency_region(self):
        ref = self.campaign / "states" / "ref"
        self.assertFalse(np.asarray(read_json(ref / "state.json")["frequency_mobile"]).any())
        (ref / "ctl" / "eshift_04_static").write_text((ref / "ctl" / "eshift_02_final").read_text())
        self.assertEqual(run_state(ref, through="freq", execute=True)["status"], "complete")
        self.assertEqual(read_json(ref / "05_freq" / "frequencies.json")["frequencies_cm1"], [])
        rows = {
            r["state_id"]: r
            for r in cli("collect", self.campaign) and read_json(self.campaign / "analysis" / "states.json")["states"]
        }
        self.assertTrue(rows["ref"]["free_energy_valid"])
        self.assertEqual(rows["ref"]["F_vib_eV"], 0.0)

    def test_cathodic_thermodynamics(self):
        cli("thermo", self.campaign, "--electronic-only")
        thermo = read_json(self.campaign / "analysis" / "thermodynamics.json")
        row = next(
            c for c in thermo["cycles"] if c["cycle_id"] == "her_volmer_heyrovsky_m0010" and c["potential_V"] == 0.0
        )
        self.assertEqual(row["direction"], "cathodic")
        self.assertEqual(
            [round(x, 6) for x in row["step_deltaG_eV"]], [DG_H, -DG_H]
        )  # Volmer, then the releasing Heyrovsky step
        self.assertAlmostEqual(row["limiting_potential_V"], -DG_H, places=6)
        self.assertAlmostEqual(row["thermodynamic_overpotential_V"], DG_H, places=6)

    def test_pathways_and_pourbaix_for_a_cathodic_reaction(self):
        ranked = cli(
            "pathways",
            self.campaign,
            "--rank",
            "--references",
            self.campaign.parent / "inputs" / "references.json",
            "--electronic-only",
            "--potentials",
            "0",
        )
        best = ranked["ranking"]["global_minimum_limiting_potential"]["her"]
        self.assertAlmostEqual(best["limiting_potential_V"], -DG_H, places=6)
        self.assertAlmostEqual(best["thermodynamic_overpotential_V"], DG_H, places=6)
        result = cli(
            "pourbaix", self.campaign, "--electronic-only", "--scale", "RHE", "--at-pH", "0", "--potential=-0.5:0.5"
        )
        steps = [(t["label"], round(t["from_V"], 6)) for t in result["transitions"]]
        self.assertEqual(steps, [("m0010:H", -0.5), ("native", -DG_H)])


if __name__ == "__main__":
    unittest.main()
