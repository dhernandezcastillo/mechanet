"""frequency.central = "final": 04_static and 05_freq at the 02_final geometry, without 03_refine.

The VASP process is the synthetic one from support.py; these tests check the stage
chain, the inputs it writes, and the scheduler/cleanup rules that depend on it.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from support import FINAL_PROTOCOL, POMASS, attempt, calls, make_campaign, marker, run_cli  # noqa: E402
from mechanet.fileio import read_json, write_json  # noqa: E402
from mechanet.runner import _archive_counts, cleanup, prepare, run_state, submission_check  # noqa: E402
from mechanet.slurm import submit, write_job_script  # noqa: E402
from mechanet.structure import read_poscar  # noqa: E402
from mechanet.vasp import (
    FINAL_HAMILTONIAN_FREE_TAGS,
    STAGES,
    STAGE_FINAL,
    STAGE_FREQ,
    STAGE_LOOSE,  # noqa: E402
    STAGE_REFINE,
    STAGE_STATIC,
    force_gate,
    protocol_stages,
    read_incar,
    stage_settings,
)


def stiff_model():
    """Positive-definite K over the 9 coordinates of the Fe-O-H test state."""
    masses = np.repeat([POMASS["Fe"], POMASS["O"], POMASS["H"]], 3)
    return np.diag(25.0 * np.sqrt(masses))


class _Base(unittest.TestCase):
    config_updates = {"frequency": FINAL_PROTOCOL}
    harmonic = stiff_model()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = patch.dict(
            os.environ, {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}, clear=True
        )
        self.env.start()
        self.campaign, self.state, self.meta = make_campaign(self.root, self.config_updates, self.harmonic)

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def listed(self):
        return {
            Path(f["path"]).relative_to(self.state).as_posix()
            for s in cleanup(self.campaign)["states"]
            for f in s["files"]
        }


class FinalProtocolInputTests(_Base):
    def test_plan_has_no_refine_stage_and_static_keeps_the_final_hamiltonian(self):
        prepare(self.campaign)
        plan = read_json(self.state / "plan.json")
        self.assertEqual(plan["central"], "final")
        self.assertEqual(list(plan["stages"]), [STAGE_LOOSE, STAGE_FINAL, STAGE_STATIC, STAGE_FREQ])
        self.assertFalse((self.state / STAGE_REFINE).exists())
        final = read_incar(self.state / STAGE_FINAL / "INCAR")
        static = read_incar(self.state / STAGE_STATIC / "INCAR")
        freq = read_incar(self.state / STAGE_FREQ / "INCAR")
        self.assertEqual((final["LREAL"], final["PREC"], final["EDIFF"]), ("Auto", "Accurate", "1e-5"))
        self.assertEqual(
            {
                k: static[k]
                for k in (
                    "LREAL",
                    "PREC",
                    "EDIFF",
                    "NELM",
                    "ISYM",
                    "ISTART",
                    "ICHARG",
                    "IBRION",
                    "NSW",
                    "LWAVE",
                    "LCHARG",
                    "LMAXMIX",
                )
            },
            {
                "LREAL": "Auto",
                "PREC": "Accurate",
                "EDIFF": "1e-06",
                "NELM": "120",
                "ISYM": "0",
                "ISTART": "0",
                "ICHARG": "1",
                "IBRION": "-1",
                "NSW": "0",
                "LWAVE": ".TRUE.",
                "LCHARG": ".FALSE.",
                "LMAXMIX": "4",
            },
        )
        self.assertEqual(
            {k: freq[k] for k in ("LREAL", "PREC", "EDIFF", "ISYM", "ISTART", "ICHARG", "LWAVE", "LCHARG")},
            {
                "LREAL": "Auto",
                "PREC": "Accurate",
                "EDIFF": "1e-06",
                "ISYM": "0",
                "ISTART": "1",
                "ICHARG": "0",
                "LWAVE": ".FALSE.",
                "LCHARG": ".FALSE.",
            },
        )
        energy_surface = {k: v for k, v in final.items() if k not in FINAL_HAMILTONIAN_FREE_TAGS}
        self.assertIn("ENCUT", energy_surface)
        for values in (static, freq):
            self.assertEqual({k: v for k, v in values.items() if k not in FINAL_HAMILTONIAN_FREE_TAGS}, energy_surface)
        pending = read_json(self.state / STAGE_STATIC / "PENDING.json")
        self.assertEqual(
            (pending["geometry_source"], pending["restart_source"]), ("02_final/CONTCAR", "02_final/CHGCAR")
        )
        script = (self.state / "run_job.sh").read_text()
        self.assertIn("# Stages: 01_loose, 02_final, 04_static, 05_freq (frequency.central = final", script)

    def test_without_the_key_the_refine_protocol_applies(self):
        config = read_json(self.campaign / "config.json")
        config["frequency"].pop("central")
        self.assertEqual(protocol_stages(config), STAGES)
        static = stage_settings(self.campaign, self.meta, STAGE_STATIC, config)
        self.assertEqual(
            (static["values"]["LREAL"], static["geometry_source"], static["restart_source"]),
            (".FALSE.", "03_refine/CONTCAR", "03_refine/CHGCAR"),
        )
        config["frequency"]["central"] = "static"
        with self.assertRaisesRegex(ValueError, "frequency.central must be one of"):
            stage_settings(self.campaign, self.meta, STAGE_FINAL, config)

    def test_force_gate_defaults_follow_the_protocol(self):
        self.assertEqual(force_gate({"frequency": {"central": "final"}}), (0.035, 0.03))
        self.assertEqual(force_gate({"frequency": {}}), (0.03, 0.02))
        self.assertEqual(force_gate({}), (0.03, 0.02))
        self.assertEqual(force_gate({"frequency": {"central": "final", "max_force_eVA": 0.05}}), (0.05, 0.03))

    def test_override_that_changes_the_energy_surface_is_rejected(self):
        config = read_json(self.campaign / "config.json")
        for stage, overrides, tag in (
            (STAGE_STATIC, {STAGE_STATIC: {"LREAL": ".FALSE."}}, "LREAL"),
            (STAGE_FREQ, {STAGE_FREQ: {"ENCUT": 520}}, "ENCUT"),
            (STAGE_STATIC, {STAGE_STATIC: {"PREC": "Normal"}}, "PREC"),
            (STAGE_STATIC, {STAGE_FINAL: {"SIGMA": 0.05}}, "SIGMA"),
        ):
            bad = json.loads(json.dumps(config))
            bad["vasp"]["stage_overrides"] = overrides
            with self.assertRaisesRegex(ValueError, f"differs from 02_final in {tag}"):
                stage_settings(self.campaign, self.meta, stage, bad)
        allowed = json.loads(json.dumps(config))
        allowed["vasp"]["stage_overrides"] = {STAGE_STATIC: {"ALGO": "All", "NELM": 200}, STAGE_FINAL: {"ALGO": "Fast"}}
        allowed["vasp"]["common_overrides"] = {"NCORE": 8, "ENCUT": 520}
        values = stage_settings(self.campaign, self.meta, STAGE_STATIC, allowed)["values"]
        self.assertEqual((values["ALGO"], values["NELM"], values["ENCUT"]), ("All", "200", "520"))
        with self.assertRaisesRegex(ValueError, "not part of"):
            stage_settings(self.campaign, self.meta, STAGE_REFINE, config)
        non_scf = json.loads(json.dumps(config))
        non_scf["vasp"]["stage_overrides"] = {STAGE_FREQ: {"ALGO": "Eigenval"}}
        with self.assertRaisesRegex(ValueError, "self-consistent"):
            stage_settings(self.campaign, self.meta, STAGE_FREQ, non_scf)
        config["vasp"]["stage_overrides"] = {"note": {"text": "not a stage: ignored"}}
        write_json(self.campaign / "config.json", config)
        prepare(self.campaign)
        config["vasp"]["stage_overrides"] = {STAGE_REFINE: {"NSW": 80}}
        write_json(self.campaign / "config.json", config)
        with self.assertRaisesRegex(ValueError, "does not run"):
            prepare(self.campaign)

    def test_switching_the_protocol_after_prepare_is_refused(self):
        prepare(self.campaign)
        config = read_json(self.campaign / "config.json")
        config["frequency"]["central"] = "refine"
        write_json(self.campaign / "config.json", config)
        with self.assertRaisesRegex(ValueError, "prepared with frequency.central = 'final'"):
            prepare(self.campaign)

    def test_through_refine_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "frequency.central = 'final'"):
            run_state(self.state, through="refine")
        with self.assertRaisesRegex(ValueError, "not part of"):
            submit(self.campaign, through="refine")
        with self.assertRaisesRegex(ValueError, "not part of"):
            submit(self.campaign, through={self.state.name: "refine"})
        with self.assertRaisesRegex(ValueError, "not part of"):
            write_job_script(self.state, through="refine")
        out = run_cli(["run-state", str(self.state), "--through", "refine"])
        self.assertEqual(out.returncode, 2)
        self.assertIn("04_static follows 02_final directly", out.stderr)
        self.assertEqual(
            run_state(self.state, through="freq")["stages"], [STAGE_LOOSE, STAGE_FINAL, STAGE_STATIC, STAGE_FREQ]
        )


class FinalProtocolRunTests(_Base):
    def test_static_starts_from_the_final_geometry_and_density(self):
        result = run_state(self.state, through="freq", execute=True)
        self.assertEqual(result["status"], "complete")
        self.assertEqual([s["stage"] for s in result["stages"]], [STAGE_LOOSE, STAGE_FINAL, STAGE_STATIC, STAGE_FREQ])
        self.assertEqual(
            [c for c in calls(self.state) if not c.startswith("05_freq/")], [STAGE_LOOSE, STAGE_FINAL, STAGE_STATIC]
        )
        self.assertEqual(len([c for c in calls(self.state) if c.startswith("05_freq/")]), 12)
        contcar = read_poscar(self.state / STAGE_FINAL / "CONTCAR")
        np.testing.assert_allclose(read_poscar(self.state / STAGE_STATIC / "POSCAR").positions, contcar.positions)
        np.testing.assert_allclose(read_poscar(self.state / STAGE_FREQ / "POSCAR").positions, contcar.positions)
        inputs = read_json(self.state / STAGE_STATIC / "input_hashes.json")
        self.assertEqual(inputs["geometry_source"], str(self.state / STAGE_FINAL / "CONTCAR"))
        self.assertEqual(inputs["restart_files"]["CHGCAR"]["source"], str(self.state / STAGE_FINAL / "CHGCAR"))
        self.assertTrue(attempt(self.state / STAGE_STATIC)["restart_evidence"]["verified"])
        freq_inputs = read_json(self.state / STAGE_FREQ / "input_hashes.json")
        self.assertEqual(freq_inputs["restart_files"]["WAVECAR"]["source"], str(self.state / STAGE_STATIC / "WAVECAR"))
        self.assertTrue(read_json(self.state / STAGE_FREQ / "frequencies.json")["complete"])

    def test_submission_check_goes_from_final_to_static(self):
        self.assertEqual(run_state(self.state, through="final", execute=True)["status"], "complete")
        check = submission_check(self.state, "freq")
        self.assertEqual((check["action"], check["stage"], check["rank"]), ("submit", STAGE_STATIC, 6))
        with patch("subprocess.run", side_effect=AssertionError("preview only")):
            out = submit(self.campaign, through="freq", execute=False)
        self.assertEqual((out["jobs"][0]["status"], out["jobs"][0]["stage"]), ("dry_run", STAGE_STATIC))
        self.assertEqual(out["displacements_remaining"], 12)

    def test_cleanup_keeps_the_final_density_until_static_is_complete(self):
        run_state(self.state, through="final", execute=True)
        self.assertNotIn("02_final/CHGCAR", self.listed(), "04_static still needs it")
        run_state(self.state, through="static", execute=True)
        self.assertEqual(self.listed(), {"02_final/CHGCAR", "04_static/CHGCAR"})
        cleanup(self.campaign, execute=True)
        self.assertTrue((self.state / STAGE_STATIC / "WAVECAR").exists(), "05_freq still needs the static WAVECAR")
        self.assertEqual(run_state(self.state, through="freq", execute=True)["status"], "complete")
        self.assertEqual(self.listed(), {"04_static/WAVECAR"})

    def _static_force(self, force):
        """Move the harmonic minimum so that 04_static sees `force` (eV/A) on O along x."""
        model = np.array(self.harmonic)
        x0 = read_poscar(self.state / "POSCAR").positions.reshape(-1)
        x0[3] += force / model[3, 3]
        write_json(self.state / "harmonic.json", {"x0": x0.tolist(), "K": model.tolist()})

    def test_static_force_in_the_warning_band_passes_the_gate(self):
        config = read_json(self.campaign / "config.json")
        for key in ("max_force_eVA", "warn_force_eVA"):  # the protocol's defaults: 0.035 and 0.03
            config["frequency"].pop(key)
        write_json(self.campaign / "config.json", config)
        self.assertEqual(run_state(self.state, through="final", execute=True)["status"], "complete")
        self._static_force(0.032)  # above warn_force_eVA = 0.03, below max_force_eVA = 0.035
        self.assertEqual(run_state(self.state, through="freq", execute=True)["status"], "complete")
        gate = read_json(self.state / STAGE_FREQ / "input_hashes.json")["static_force_check"]
        self.assertAlmostEqual(gate["max_active_force_eVA"], 0.032, places=4)
        self.assertIn("warning", gate)

    def test_static_force_above_the_gate_blocks_frequencies(self):
        self.assertEqual(run_state(self.state, through="final", execute=True)["status"], "complete")
        self._static_force(0.04)
        with self.assertRaisesRegex(RuntimeError, "exceeds frequency.max_force_eVA = 0.035"):
            run_state(self.state, through="freq", execute=True)
        self.assertEqual(submission_check(self.state, "freq")["status"], "skipped_blocked")
        config = read_json(self.campaign / "config.json")
        config["frequency"]["max_force_eVA"] = 0.045  # the user accepts the larger residual force
        write_json(self.campaign / "config.json", config)
        self.assertEqual(submission_check(self.state, "freq")["action"], "submit")
        self.assertEqual(run_state(self.state, through="freq", execute=True)["status"], "complete")


class FinalProtocolSolutionTests(_Base):
    """04_static must reproduce the 02_final electronic solution before any displacement runs."""

    def test_static_in_another_solution_blocks_before_any_displacement(self):
        self.assertEqual(run_state(self.state, through="final", execute=True)["status"], "complete")
        marker(self.state, "eshift_04_static", "-0.05")
        with self.assertRaisesRegex(RuntimeError, "04_static E0 differs from the 02_final E0 by -49.0 meV"):
            run_state(self.state, through="freq", execute=True)
        self.assertEqual([c for c in calls(self.state) if c.startswith("05_freq")], [], "no displacement ran")
        blocked = read_json(self.state / STAGE_FREQ / "BLOCKED.json")
        self.assertAlmostEqual(blocked["static_minus_final_e0_eV"], -0.049)
        check = submission_check(self.state, "freq")
        self.assertEqual(check["status"], "skipped_blocked")
        self.assertIn("frequency.max_static_e0_shift_eV", check["reason"])
        config = read_json(self.campaign / "config.json")
        config["frequency"]["max_static_e0_shift_eV"] = 0.06  # accepted after inspection
        write_json(self.campaign / "config.json", config)
        self.assertEqual(submission_check(self.state, "freq")["action"], "submit")
        self.assertEqual(run_state(self.state, through="freq", execute=True)["status"], "complete")
        gate = read_json(self.state / STAGE_FREQ / "input_hashes.json")["static_force_check"]
        self.assertAlmostEqual(gate["static_minus_final_e0_eV"], -0.049)

    def test_static_moment_change_blocks_frequencies(self):
        self.assertEqual(run_state(self.state, through="final", execute=True)["status"], "complete")
        marker(self.state, "moment_04_static", "2.0")
        with self.assertRaisesRegex(RuntimeError, "a local moment changed by 2.00 muB from 02_final to 04_static"):
            run_state(self.state, through="freq", execute=True)
        self.assertEqual(submission_check(self.state, "freq")["status"], "skipped_blocked")

    def test_the_refine_protocol_has_no_such_gate(self):
        config = read_json(self.campaign / "config.json")
        config["frequency"]["central"] = "refine"
        write_json(self.campaign / "config.json", config)
        marker(self.state, "moment_04_static", "2.0")
        self.assertEqual(run_state(self.state, through="freq", execute=True)["status"], "complete")
        gate = read_json(self.state / STAGE_FREQ / "input_hashes.json")["static_force_check"]
        self.assertNotIn("static_minus_final_e0_eV", gate)

    def test_failed_static_is_retried_from_the_final_density(self):
        self.assertEqual(run_state(self.state, through="final", execute=True)["status"], "complete")
        marker(self.state, "chg_atomic_04_static")  # VASP reports that it did not read the CHGCAR
        with self.assertRaisesRegex(RuntimeError, "04_static"):
            run_state(self.state, through="static", execute=True)
        self.assertEqual(attempt(self.state / STAGE_STATIC)["status"], "failed")
        self.assertEqual(submission_check(self.state, "freq")["status"], "skipped_needs_retry")
        self.assertNotIn("02_final/CHGCAR", self.listed(), "the retry needs it")
        (self.state / "ctl" / "chg_atomic_04_static").unlink()
        self.assertEqual(run_state(self.state, through="static", execute=True, retry=True)["status"], "complete")
        self.assertEqual(_archive_counts(self.state / STAGE_STATIC)["failed"], 1)
        inputs = read_json(self.state / STAGE_STATIC / "input_hashes.json")
        self.assertEqual(inputs["restart_files"]["CHGCAR"]["source"], str(self.state / STAGE_FINAL / "CHGCAR"))
        self.assertTrue(attempt(self.state / STAGE_STATIC)["restart_evidence"]["verified"])


class FinalProtocolNativeFrequencyTests(_Base):
    config_updates = {"frequency": dict(FINAL_PROTOCOL, method="vasp_ibrion5")}

    def test_ibrion5_reads_the_static_wavecar(self):
        self.assertEqual(run_state(self.state, through="freq", execute=True)["status"], "complete")
        self.assertEqual(calls(self.state), [STAGE_LOOSE, STAGE_FINAL, STAGE_STATIC, STAGE_FREQ])
        inputs = read_json(self.state / STAGE_FREQ / "input_hashes.json")
        self.assertEqual(inputs["restart_files"]["WAVECAR"]["source"], str(self.state / STAGE_STATIC / "WAVECAR"))
        self.assertTrue(attempt(self.state / STAGE_FREQ)["restart_evidence"]["verified"])
        self.assertEqual(read_incar(self.state / STAGE_FREQ / "INCAR")["LREAL"], "Auto")


if __name__ == "__main__":
    unittest.main()
