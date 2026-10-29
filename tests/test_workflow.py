"""End to end on RuO2(110) with the synthetic VASP: init -> generate -> run -> collect -> thermo ->
pathways --rank -> pourbaix -> plots -> report, through the command line.

Node energies are set through the fake VASP's ``eshift`` marker so that the adsorbate
evolution cycle on the CUS Ru has the steps 0.80, 1.60, 1.40, 1.20 and -0.08 eV at 0 V vs
RHE (screening level: E0 plus the tabulated ZPE - TS corrections). Every number checked below
follows from those steps.
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
from mechanet.runner import run_state
from mechanet.structure import read_poscar

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
E_H2, G_H2, E_H2O, G_H2O = -6.77, 0.0, -14.22, 0.0  # synthetic references
STEPS = {"OH": 0.80, "O": 0.80 + 1.60, "OOH": 0.80 + 1.60 + 1.40, "OO": 0.80 + 1.60 + 1.40 + 1.20}
CORRECTION = {"H": 0.30, "O": 0.05}  # thermodynamics.electronic_mode_corrections


def cli(*args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(["--json", *map(str, args)])
    if code != 0:
        raise AssertionError(f"mechanet {' '.join(map(str, args))} exited {code}: {out.getvalue()}")
    return json.loads(out.getvalue())


@unittest.skipIf(os.environ.get("MECHANET_SKIP_SLOW"), "slow workflow test disabled by environment")
class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env = patch.dict(os.environ, {"MECHANET_CONFIG_DIR": str(cls.tmp / "config")})
        cls.env.start()
        project = cls.tmp / "ruo2"
        shutil.copytree(EXAMPLES / "ruo2_110", project)
        data = yaml.safe_load((project / "project.yaml").read_text())
        fake = cls.tmp / "fake_vasp.py"
        fake.write_text(FAKE_VASP)
        data["scope"]["families"] = ["aem"]
        data["vasp"] = {
            "potcars": write_potcars(cls.tmp, ("Ru", "O", "H")),
            "magmom_by_element": None,
            "command": [sys.executable, "-I", str(fake)],
        }
        data["frequency"] = {"central": "final"}
        data["thermodynamics"]["references"] = "inputs/references.json"
        (project / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        (project / "inputs" / "references.json").write_text(
            json.dumps({"references": {"H2": {"E": E_H2, "Gcorr": G_H2}, "H2O": {"E": E_H2O, "Gcorr": G_H2O}}})
        )
        cls.project = project
        cls.campaign = project / "campaign"
        cls.generated = cli("generate", project / "project.yaml")
        mu_h2o, half_h2 = E_H2O + G_H2O, 0.5 * (E_H2 + G_H2)
        for state in sorted((cls.campaign / "states").iterdir()):
            meta = read_json(state / "state.json")
            delta = {k: v for k, v in meta["delta_atoms"].items() if v}
            status = next((s for s in ("OOH", "OO", "OH", "O") if f"m0038{s}_" in meta["node_id"]), None)
            target = -100.0
            if status:
                n_o, n_h = delta.get("O", 0), delta.get("H", 0)
                # G(node) - G(ref) = n_O mu(H2O) - (2 n_O - n_H) mu(H+ + e-)(0 V RHE) + DeltaG
                target += n_o * mu_h2o - (2 * n_o - n_h) * half_h2 + STEPS[status]
                target -= CORRECTION["O"] * n_o + CORRECTION["H"] * n_h
            (state / "ctl").mkdir(exist_ok=True)
            (state / "ctl" / "eshift_02_final").write_text(f"{target + 100.003:.6f}")
            # harmonic forces around the start: every stage converges at the generated geometry
            positions = read_poscar(state / "POSCAR").positions.reshape(-1)
            (state / "harmonic.json").write_text(
                json.dumps({"x0": positions.tolist(), "K": (10.0 * np.eye(len(positions))).tolist()})
            )
        cli("prepare", cls.campaign)
        for state in sorted((cls.campaign / "states").iterdir()):
            result = run_state(state, through="final", execute=True)
            assert result["status"] == "complete", result

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        shutil.rmtree(cls.tmp)

    def test_generate_plan_validate(self):
        self.assertEqual(self.generated["complete_stoichiometric_cycles"], 4)
        plan = cli("plan", self.campaign)
        self.assertEqual(plan["network"]["cycles_by_family"], {"aem": 4})
        self.assertIsNotNone(plan["cost"]["node_hours_estimate"])  # timings of the finished runs
        self.assertTrue(cli("validate", self.campaign)["valid"])
        self.assertTrue((self.campaign / "method_card.json").is_file())

    def test_thermo_reproduces_the_designed_steps(self):
        result = cli("thermo", self.campaign, "--electronic-only", "--potentials", "0,1.23,1.63")
        self.assertEqual(result["cycles_available"], 4)
        thermo = read_json(self.campaign / "analysis" / "thermodynamics.json")
        row = next(c for c in thermo["cycles"] if c["cycle_id"] == "aem_m0038_AEM" and c["potential_V"] == 0.0)
        self.assertEqual([round(x, 6) for x in row["step_deltaG_eV"]], [0.8, 1.6, 1.4, 1.12])
        self.assertAlmostEqual(row["limiting_potential_V"], 1.6, places=6)
        self.assertAlmostEqual(row["thermodynamic_overpotential_V"], 0.37, places=6)
        bound = next(c for c in thermo["cycles"] if c["cycle_id"] == "aem_m0038_bound_OO" and c["potential_V"] == 0.0)
        self.assertEqual([round(x, 6) for x in bound["step_deltaG_eV"]], [0.8, 1.6, 1.4, 1.2, -0.08])
        # At 1.63 V bound O2 (-1.52 eV) lies lowest, but it gives O2 off without electron transfer
        # (-0.08 eV at every potential): it is transient. *OOH (-1.09 eV) is the resting state; its
        # electrochemical O2 release (1.12 - 1.63 eV) runs downhill, so it is flagged, as in the Pourbaix diagram.
        nodes = {
            tuple(n.get("occupancy", {}).items()): n["id"] for n in read_json(self.campaign / "network.json")["nodes"]
        }
        rest = thermo["network_resting_states"]["1.63"]
        self.assertEqual(rest["node_id"], nodes[(("m0038", "OOH"),)])
        self.assertAlmostEqual(rest["formation_free_energy_eV"], 3.8 - 3 * 1.63, places=5)
        self.assertIn("releases the product downhill", rest["releases_product_downhill"])
        self.assertEqual(list(rest["transient_nodes"]), [nodes[(("m0038", "OO"),)]])
        for cycle in ("aem_m0038_AEM", "aem_m0038_bound_OO"):
            at = next(c for c in thermo["cycles"] if c["cycle_id"] == cycle and c["potential_V"] == 1.63)
            self.assertEqual(at["resting_state_penalty_eV"], 0.0)
        # At 0 V *OOH holds its O2 (release uphill by 1.12 eV); bound O2 desorbs at any potential.
        self.assertEqual(list(thermo["network_resting_states"]["0.0"]["transient_nodes"]), [nodes[(("m0038", "OO"),)]])
        figure = cli("plot-profile", "aem_m0038_AEM", self.campaign, "--potential", "1.23")
        self.assertTrue(Path(figure["figure"]).is_file())

    def test_configured_potentials_keep_their_meaning_on_another_scale(self):
        cli("thermo", self.campaign, "--electronic-only", "--scale", "SHE", "--pH", "14")
        thermo = read_json(self.campaign / "analysis" / "thermodynamics.json")
        nernst = 8.617333262145e-5 * 298.15 * np.log(10.0)
        self.assertEqual(thermo["potentials_V"], [round(u - 14 * nernst, 4) for u in (1.23, 1.43, 1.53, 1.63)])
        self.assertIn("V vs RHE are reported as", thermo["notes"][0])

    def test_pathways_ranking_finds_the_global_minimum(self):
        result = cli(
            "pathways",
            self.campaign,
            "--rank",
            "--references",
            self.project / "inputs" / "references.json",
            "--electronic-only",
            "--potentials",
            "1.23",
        )
        best = result["ranking"]["global_minimum_limiting_potential"]["oer"]
        self.assertAlmostEqual(best["limiting_potential_V"], 1.6, places=6)
        self.assertAlmostEqual(best["thermodynamic_overpotential_V"], 0.37, places=6)
        self.assertEqual(best["cycle_id"], "aem_m0038_AEM")  # the representative, not a translation image
        self.assertIn("steps whose two ends have energies", best["note"])
        ranking = read_json(self.campaign / "pathways" / "ranking.json")
        self.assertGreaterEqual(ranking["n_cycles_ranked"], 4)
        self.assertEqual(list(ranking["best_by_Gmax"]), ["oer"])  # grouped by reaction
        self.assertTrue(all(r["symmetry_representative"] for r in ranking["best_by_Gmax"]["oer"]["1.23"]))
        # --rank alone uses the project's references; the best route is curated, so nothing is adopted
        again = cli("pathways", self.campaign, "--rank", "--electronic-only", "--potentials", "1.23", "--adopt", "best")
        self.assertEqual(again["adopted"], [])
        self.assertEqual(again["ranking"]["global_minimum_limiting_potential"]["oer"]["cycle_id"], "aem_m0038_AEM")

    def test_the_sweep_finds_curated_routes_that_enumeration_did_not_reach(self):
        # --max-length 3 enumerates no OER cycle (each needs four steps); the exact sweep still finds the
        # AEM route, reports it under its curated ID, and adopts nothing.
        before = (self.campaign / "network.json").read_text()
        thermo_before = (
            (self.campaign / "analysis" / "thermodynamics.json").read_text()
            if (self.campaign / "analysis" / "thermodynamics.json").is_file()
            else None
        )
        result = cli(
            "pathways",
            self.campaign,
            "--max-length",
            "3",
            "--rank",
            "--electronic-only",
            "--potentials",
            "1.23",
            "--adopt",
            "best",
        )
        self.assertEqual(result["summary"]["cycles_found"], 0)
        best = result["ranking"]["global_minimum_limiting_potential"]["oer"]
        self.assertEqual(best["cycle_id"], "aem_m0038_AEM")
        self.assertTrue(best["curated"])
        self.assertEqual(result["ranking"]["reactions"]["oer"]["sweep"], "exact")
        self.assertEqual(result["adopted"], [])
        self.assertEqual((self.campaign / "network.json").read_text(), before)
        again = cli(
            "pathways",
            self.campaign,
            "--max-length",
            "3",
            "--rank",
            "--electronic-only",
            "--potentials",
            "1.23",
            "--adopt",
            "best",
        )
        self.assertEqual(again["adopted"], [])  # repeating the command is harmless
        report = read_json(self.campaign / "pathways" / "pathways.json")
        self.assertFalse(any(c["id"].startswith("discovered_") for c in report["cycles"]))
        if thermo_before is not None:  # the ranking writes its own analysis, not over `thermo`'s
            self.assertEqual((self.campaign / "analysis" / "thermodynamics.json").read_text(), thermo_before)

    def test_without_an_exact_sweep_the_best_enumerated_cycle_is_reported(self):
        from unittest.mock import PropertyMock, patch

        from mechanet.reactions import Reaction

        with patch.object(Reaction, "closed_walks_are_turnovers", new_callable=PropertyMock, return_value=False):
            result = cli("pathways", self.campaign, "--rank", "--electronic-only", "--potentials", "1.23")
        info = result["ranking"]["reactions"]["oer"]
        self.assertIn("skipped", info["sweep"])
        best = result["ranking"]["global_minimum_limiting_potential"]["oer"]
        self.assertAlmostEqual(best["limiting_potential_V"], 1.6, places=6)
        self.assertIn("sweep was skipped", best["note"])

    def test_pathways_ranking_refuses_incomplete_references(self):
        broken = self.tmp / "broken_references.json"
        broken.write_text(
            json.dumps(
                {
                    "temperature_K": 298.15,
                    "H2": {"energy_e0_eV": E_H2, "free_energy_correction_eV": G_H2},
                    "H2O": {"energy_e0_eV": E_H2O, "free_energy_correction_eV": G_H2O},
                }
            )
        )
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = main(["pathways", str(self.campaign), "--rank", "--references", str(broken), "--electronic-only"])
        self.assertEqual(code, 2)
        self.assertIn("phase: liquid", err.getvalue())

    def test_pourbaix_transitions(self):
        result = cli("pourbaix", self.campaign, "--electronic-only", "--scale", "RHE", "--at-pH", "0", "--plot")
        steps = [(t["label"], round(t["from_V"], 6)) for t in result["transitions"]]
        # native -> *OH at 0.80 V -> *OOH (3.80 - 3U) at 1.50 V. Bound O2 (5.00 - 4U) would undercut *OH at
        # 1.40 V, but it desorbs downhill (-0.08 eV) at every potential and is left out; *OOH gives O2 off
        # electrochemically above 1.12 V, so its whole region is flagged.
        self.assertEqual(steps, [("native", -0.5), ("m0038:OH", 0.8), ("m0038:OOH", 1.5)])
        self.assertEqual([x["label"] for x in result["left_out"]], ["m0038:OO"])
        self.assertEqual(result["transitions"][-1]["releases_product_downhill"], "throughout")
        self.assertIsNone(result["transitions"][1]["releases_product_downhill"])
        self.assertTrue((self.campaign / "pourbaix" / "pourbaix.png").is_file())
        report = cli("report", self.campaign)
        for heading in ("## Network", "## Calculations", "## Thermodynamics", "## Surface Pourbaix diagram"):
            self.assertIn(heading, report["text"])
        self.assertIn("not shown, because they give the bound product off downhill", report["text"])
        kept = cli("pourbaix", self.campaign, "--electronic-only", "--scale", "RHE", "--at-pH", "0", "--keep-transient")
        steps = [(t["label"], round(t["from_V"], 6)) for t in kept["transitions"]]
        self.assertEqual(steps, [("native", -0.5), ("m0038:OH", 0.8), ("m0038:OO", 1.4)])


if __name__ == "__main__":
    unittest.main()
