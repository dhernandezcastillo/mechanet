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
        # every intermediate ran with the same settings, so the same-settings check stays silent
        self.assertTrue(thermo["protocol_check"]["consistent"])
        self.assertGreater(thermo["protocol_check"]["states_checked"], 0)
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
        # Moments along the cycle on an open metal site: the Ru, the bound O (labelled by the site) and
        # the outer O of *OOH; the synthetic VASP keeps every moment fixed, so the spin check passes.
        spin = thermo["magnetism"]["cycles"]["aem_m0038_AEM"]
        self.assertEqual(spin["status"], "consistent")
        roles = {a["atom_id"]: a["role"] for a in spin["atoms"]}
        self.assertEqual(roles["a0038"], "site metal")
        self.assertEqual(roles["m0038"], "bound")
        self.assertEqual(roles["d_m0038"], "outer")
        self.assertTrue(figure["moments_panel"])

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

    def test_candidates_of_a_window_then_frequencies_for_them_only(self):
        """Lines at 0 V RHE: bare 0, *OH 0.80 - U, *O 2.40 - 2U, *OOH 3.80 - 3U (bound O2 is left out)."""
        result = cli(
            "pourbaix",
            self.campaign,
            "--screening",
            "--scale",
            "RHE",
            "--at-pH",
            "0",
            "--window",
            "1.0:1.8",
            "--figures",
        )
        rows = result["candidates"]
        self.assertEqual([r["label"] for r in rows], ["m0038:OH", "m0038:OOH", "m0038:O", "native"])
        self.assertEqual(
            [r["surface"] for r in rows[:2]],
            ["terminal 1 OH, 1 empty | bridge 2 O", "terminal 1 OOH, 1 empty | bridge 2 O"],
        )
        oh, ooh, o, bare = rows
        self.assertEqual((round(oh["stable_from_V"], 4), round(oh["stable_to_V"], 4)), (0.8, 1.5))
        self.assertAlmostEqual(oh["lead_eV"], 0.4, places=4)  # 0.40 eV below the bare site and *O at 1.20 V
        self.assertAlmostEqual(o["above_stable_eV"], 0.1, places=4)
        self.assertAlmostEqual(o["closest_at_V"], 1.5, places=4)
        self.assertAlmostEqual(bare["above_stable_eV"], 0.2, places=4)
        self.assertEqual(ooh["releases_product_downhill"], "throughout")  # *OOH gives O2 off above 1.12 V
        self.assertIsNone(oh["releases_product_downhill"])
        self.assertFalse(any(r["free_energy"] for r in rows))  # no frequency calculation yet
        self.assertEqual(result["surface_analysis"]["margin_eV"], 0.2)
        folder = self.campaign / "pourbaix"
        stored = read_json(folder / "pourbaix.json")
        self.assertEqual([v["node_id"] for v in stored["hull"]["vertices"]], [r["node_id"] for r in (bare, oh, ooh)])
        self.assertEqual([round(v["to_next_V"], 4) for v in stored["hull"]["vertices"][:2]], [0.8, 1.5])
        levels = {level["level"]: level for level in stored["levels"]}
        self.assertIn("unavailable", levels["free_energy"])
        self.assertEqual([s["label"] for s in levels["screening"]["segments"]], ["native", "m0038:OH", "m0038:OOH"])
        # without the 0.30 eV per H and 0.05 eV per O the steps are other steps: energies alone are another diagram
        self.assertNotEqual(
            [round(s["to_V"], 2) for s in levels["energy"]["segments"]],
            [round(s["to_V"], 2) for s in levels["screening"]["segments"]],
        )
        for name in ("candidates.csv", "compositions.csv", "pourbaix_lines.png", "pourbaix_hull.png"):
            self.assertTrue((folder / name).is_file(), name)
        for name in ("pourbaix_margins.png", "pourbaix_levels.png", "pourbaix_lines.svg"):
            self.assertTrue((folder / name).is_file(), name)
        self.assertFalse((folder / "pourbaix_isomers.png").exists())  # one surface per composition here
        header = (folder / "candidates.csv").read_text().splitlines()[0]
        self.assertTrue(header.startswith("rank,node_id,surface,electrons,stable_in_window"))
        narrow = cli(
            "pourbaix", self.campaign, "--screening", "--scale", "RHE", "--window", "1.0:1.8", "--margin", "0.15"
        )
        self.assertEqual([r["label"] for r in narrow["candidates"]], ["m0038:OH", "m0038:OOH", "m0038:O"])
        selection = cli("select", self.campaign, "--candidates")
        chosen = read_json(self.campaign / "thermo_selection.json")
        wanted = [r["node_id"] for r in narrow["candidates"]]
        self.assertNotIn("ref", wanted)  # the slab as given is no candidate at this margin
        self.assertEqual(chosen["criteria"]["nodes"], sorted(wanted))
        # ... and is selected all the same: every formation energy is counted from it
        self.assertEqual(chosen["criteria"]["reference_surface"], "ref")
        self.assertEqual(sorted(n["node_id"] for n in chosen["nodes"]), sorted([*wanted, "ref"]))
        self.assertIn("ref", chosen["state_ids"])
        self.assertEqual(selection["n_selected"], len(chosen["state_ids"]))
        self.assertTrue(all(any(sid.startswith(node) for node in [*wanted, "ref"]) for sid in chosen["state_ids"]))
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["select", str(self.campaign), "--candidates"]), 0)
        self.assertIn("with the reference surface (ref)", out.getvalue())
        by_hand = cli("select", self.campaign, "--nodes", oh["node_id"])
        self.assertEqual(read_json(self.campaign / "thermo_selection.json")["criteria"]["nodes"], [oh["node_id"]])
        self.assertGreaterEqual(by_hand["n_selected"], 2)  # *OH and the reference surface

    def test_candidates_need_a_window_and_select_needs_the_candidates(self):
        plain = cli("pourbaix", self.campaign, "--electronic-only", "--scale", "RHE")
        self.assertNotIn("candidates", plain)
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = main(["select", str(self.campaign), "--candidates", str(self.tmp / "nothing.csv")])
        self.assertEqual(code, 2)
        self.assertIn("No candidates file", err.getvalue())
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = main(["select", str(self.campaign), "--nodes", "n_not_there"])
        self.assertEqual(code, 2)
        self.assertIn("No intermediates ['n_not_there']", err.getvalue())


if __name__ == "__main__":
    unittest.main()
