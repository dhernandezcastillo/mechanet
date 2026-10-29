"""Restartable displacement frequencies (05_freq) and parser additions, with a synthetic VASP."""

import os
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ase  # noqa: E402
import numpy as np  # noqa: E402

from support import FINAL_PROTOCOL, POMASS, attempt, calls, make_campaign, marker, run_cli  # noqa: E402
from mechanet.fileio import read_json, write_json  # noqa: E402
from mechanet.results import (
    FREQUENCY_CM1_PER_SQRT_EV_A2_AMU,
    displacement_error,
    displacement_units,  # noqa: E402
    mass_table,
    parse_run,
)
from mechanet.runner import StopController, TimeLimitReached, run_state  # noqa: E402
from mechanet.structure import read_poscar  # noqa: E402
from mechanet.vasp import STAGE_FINAL, STAGE_FREQ, STAGE_STATIC  # noqa: E402

ASE_OUTCAR = Path(ase.__file__).resolve().parent / "test" / "testdata" / "vasp" / "OUTCAR_example_1"


def harmonic_model(seed=7):
    """K over all 9 coordinates; the active (O, H) block has one negative curvature."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    active = q @ np.diag([30.0, 12.0, 5.0, 2.0, 0.5, -0.3]) @ q.T
    k = np.zeros((9, 9))
    k[3:, 3:] = active
    coupling = rng.normal(scale=0.2, size=(3, 6))  # frozen-active coupling never enters the partial Hessian
    k[:3, 3:], k[3:, :3] = coupling, coupling.T
    k[:3, :3] = np.eye(3) * 4.0
    return k


class _Base(unittest.TestCase):
    config_updates = None
    harmonic = None

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


class DisplacementFrequencyTests(_Base):
    harmonic = harmonic_model()

    def test_frequencies_match_the_analytic_harmonic_model(self):
        self.assertEqual(run_state(self.state, through="freq", execute=True)["status"], "complete")
        report = read_json(self.state / STAGE_FREQ / "frequencies.json")
        self.assertEqual(report["mass_source"], "POTCAR POMASS")
        self.assertEqual(report["masses_amu"], {"Fe": POMASS["Fe"], "O": POMASS["O"], "H": POMASS["H"]})
        k_active = self.harmonic[3:, 3:]
        np.testing.assert_allclose(report["hessian_eV_A2"], k_active, atol=1e-4)
        masses = np.repeat([POMASS["O"], POMASS["H"]], 3)
        eigen = np.linalg.eigvalsh(k_active / np.sqrt(np.outer(masses, masses)))
        expected = sorted(
            (np.sign(eigen) * FREQUENCY_CM1_PER_SQRT_EV_A2_AMU * np.sqrt(np.abs(eigen))).tolist(), reverse=True
        )
        np.testing.assert_allclose(report["frequencies_cm1"], expected, atol=0.5)
        self.assertEqual(sum(f < 0 for f in report["frequencies_cm1"]), 1, "negative curvature -> one imaginary mode")
        parsed = parse_run(self.state / STAGE_FREQ)
        np.testing.assert_allclose(parsed["frequencies_cm1"], report["frequencies_cm1"])
        self.assertTrue(parsed["frequency_converged"])
        self.assertIn("frequencies.json", attempt(self.state / STAGE_FREQ)["output_hashes"])

    def test_every_displaced_poscar_differs_by_exactly_h_on_its_dof(self):
        run_state(self.state, through="freq", execute=True)
        central = read_poscar(self.state / STAGE_FREQ / "POSCAR")
        units = displacement_units(central, 0.015)
        self.assertEqual(len(units), 12)
        for unit in units:
            displaced = read_poscar(self.state / STAGE_FREQ / "units" / unit["id"] / "POSCAR")
            self.assertIsNone(displacement_error(central, displaced, unit))
            delta = displaced.positions - central.positions
            self.assertAlmostEqual(delta[unit["atom"], unit["axis"]], 0.015 * unit["sign"], places=10)
            self.assertEqual(int(np.count_nonzero(np.abs(delta) > 1e-10)), 1)
        wrong = dict(units[0], displacement_A=-0.015)
        self.assertIsNotNone(
            displacement_error(
                central, read_poscar(self.state / STAGE_FREQ / "units" / units[0]["id"] / "POSCAR"), wrong
            )
        )

    def test_interrupted_displacement_set_reruns_only_missing_units(self):
        run_state(self.state, through="static", execute=True)
        marker(self.state, "kill_unit_u004")
        killed = run_cli(["run-state", str(self.state), "--through", "freq", "--execute"])
        self.assertEqual(killed.returncode, -signal.SIGKILL)
        before = [c for c in calls(self.state) if c.startswith("05_freq/")]
        self.assertEqual(len(before), 7)  # u001-u003 (+/-) done, u004 plus killed
        driver = attempt(self.state / STAGE_FREQ)
        self.assertEqual(driver["status"], "running")
        self.assertFalse(parse_run(self.state / STAGE_FREQ)["completed"])
        self.assertEqual(parse_run(self.state / STAGE_FREQ)["n_displacements_done"], 6)
        self.assertEqual(run_state(self.state, through="freq", execute=True)["status"], "complete")  # no --retry
        after = [c for c in calls(self.state) if c.startswith("05_freq/")][len(before) :]
        self.assertEqual(len(after), 6)
        self.assertEqual(after[0], "05_freq/u004_atom003_x_plus")
        self.assertFalse(any(c.split("/")[1].startswith(("u001", "u002", "u003")) for c in after))
        killed_unit = self.state / STAGE_FREQ / "units" / "u004_atom003_x_plus"
        self.assertEqual(read_json(killed_unit / "previous_attempts" / "001" / "archive.json")["kind"], "interrupted")
        self.assertTrue(parse_run(self.state / STAGE_FREQ)["frequency_converged"])

    def test_sigusr1_during_units_finishes_the_current_unit_and_stops(self):
        run_state(self.state, through="static", execute=True)
        marker(self.state, "slow_unit_u003")
        process = run_cli(["run-state", str(self.state), "--through", "freq", "--execute"], wait=False)
        started = self.state / "ctl" / "vasp_started"
        deadline = time.time() + 60
        while not started.exists() and time.time() < deadline:
            time.sleep(0.05)
        self.assertTrue(started.exists())
        process.send_signal(signal.SIGUSR1)
        _, err = process.communicate(timeout=120)
        self.assertEqual(process.returncode, 0, err)
        self.assertEqual(read_json(self.state / "run.json")["status"], "incomplete_time_limit")
        unit = self.state / STAGE_FREQ / "units" / "u003_atom002_z_plus"
        self.assertEqual(attempt(unit)["status"], "complete", "single points are not cut short")
        self.assertFalse((unit / "STOPCAR").exists())
        self.assertEqual(parse_run(self.state / STAGE_FREQ)["n_displacements_done"], 5)
        (self.state / "ctl" / "slow_unit_u003").unlink()
        self.assertEqual(run_state(self.state, through="freq", execute=True)["status"], "complete")
        self.assertEqual(len([c for c in calls(self.state) if c.startswith("05_freq/")]), 12)

    def test_unit_scf_failure_needs_retry_and_other_units_continue(self):
        run_state(self.state, through="static", execute=True)
        marker(self.state, "nelm_u002_atom002_y_plus")
        with self.assertRaisesRegex(RuntimeError, "1 displacement unit"):
            run_state(self.state, through="freq", execute=True)
        self.assertEqual(len([c for c in calls(self.state) if c.startswith("05_freq/")]), 12)
        self.assertEqual(attempt(self.state / STAGE_FREQ)["status"], "failed_units")
        with self.assertRaisesRegex(RuntimeError, "--retry"):
            run_state(self.state, through="freq", execute=True)
        (self.state / "ctl" / "nelm_u002_atom002_y_plus").unlink()
        self.assertEqual(run_state(self.state, through="freq", execute=True, retry=True)["status"], "complete")
        self.assertEqual(len([c for c in calls(self.state) if c.startswith("05_freq/")]), 13)

    def test_accepted_units_are_pruned_and_still_verify(self):
        self.assertEqual(run_state(self.state, through="freq", execute=True)["status"], "complete")
        n_calls = len(calls(self.state))
        bulky = ("PROCAR", "DOSCAR", "vasprun.xml", "EIGENVAL", "CHG", "PCDAT", "XDATCAR", "REPORT", "WAVECAR")
        kept = (
            "OUTCAR",
            "OSZICAR",
            "CONTCAR",
            "POSCAR",
            "INCAR",
            "KPOINTS",
            "POTCAR",
            "attempt.json",
            "input_hashes.json",
            "stdout.log",
            "stderr.log",
        )
        units = sorted((self.state / STAGE_FREQ / "units").iterdir())
        self.assertEqual(len(units), 12)
        for unit in units:
            self.assertFalse([name for name in bulky if (unit / name).exists()], unit.name)
            self.assertEqual([name for name in kept if not (unit / name).exists()], [], unit.name)
            record = attempt(unit)
            self.assertEqual(set(record["output_hashes"]), {"OUTCAR", "CONTCAR"})
            self.assertTrue({"vasprun.xml", "PROCAR"} <= {p["name"] for p in record["pruned_outputs"]})
        self.assertTrue((self.state / STAGE_STATIC / "vasprun.xml").exists(), "only units are pruned")
        parsed = parse_run(self.state / STAGE_FREQ)
        self.assertTrue(parsed["frequency_converged"])
        self.assertEqual(len(parsed["magnetic_moment_history"]), 12)
        np.testing.assert_allclose(
            parsed["frequencies_cm1"], read_json(self.state / STAGE_FREQ / "frequencies.json")["frequencies_cm1"]
        )
        self.assertEqual(run_state(self.state, through="freq", execute=True)["status"], "complete")
        self.assertEqual(len(calls(self.state)), n_calls, "pruned units are not redone")

    def test_cleanup_prunes_units_accepted_without_pruning(self):
        config = read_json(self.campaign / "config.json")
        config["frequency"]["prune_unit_outputs"] = False
        write_json(self.campaign / "config.json", config)
        run_state(self.state, through="freq", execute=True)
        unit = self.state / STAGE_FREQ / "units" / "u001_atom002_x_plus"
        self.assertTrue((unit / "PROCAR").exists() and (unit / "vasprun.xml").exists())
        from mechanet.runner import cleanup

        def unit_files(report):
            return [f for s in report["states"] for f in s["files"] if "/units/" in f["path"]]

        self.assertEqual(unit_files(cleanup(self.campaign)), [], "pruning disabled: cleanup keeps unit outputs")
        config["frequency"]["prune_unit_outputs"] = True
        write_json(self.campaign / "config.json", config)
        dry = unit_files(cleanup(self.campaign))
        self.assertEqual(
            {Path(f["path"]).name for f in dry},
            {"PROCAR", "DOSCAR", "vasprun.xml", "EIGENVAL", "PCDAT", "XDATCAR", "REPORT"},
        )
        self.assertEqual(len(dry), 12 * 7)
        self.assertGreater(sum(f["bytes"] for f in dry), 0)
        self.assertTrue((unit / "PROCAR").exists(), "dry run deletes nothing")
        n_calls = len(calls(self.state))
        cleanup(self.campaign, execute=True)
        self.assertFalse((unit / "PROCAR").exists() or (unit / "vasprun.xml").exists())
        self.assertTrue((unit / "OUTCAR").exists() and (unit / "OSZICAR").exists())
        self.assertEqual(run_state(self.state, through="freq", execute=True)["status"], "complete")
        self.assertEqual(len(calls(self.state)), n_calls)
        self.assertTrue(parse_run(self.state / STAGE_FREQ)["frequency_converged"])

    def test_unsafe_prune_list_is_rejected(self):
        from mechanet.runner import unit_prune_files

        for names in (["OUTCAR"], ["CONTCAR", "PROCAR"], ["../POSCAR"], ["*.xml"], "PROCAR"):
            with self.assertRaises(ValueError):
                unit_prune_files({"frequency": {"unit_prune_files": names}})
        self.assertEqual(
            unit_prune_files({"frequency": {"prune_unit_outputs": False, "unit_prune_files": ["OUTCAR"]}}), ()
        )
        self.assertIn("vasprun.xml", unit_prune_files({}))
        config = read_json(self.campaign / "config.json")
        config["frequency"]["unit_prune_files"] = ["OUTCAR"]
        write_json(self.campaign / "config.json", config)
        with self.assertRaisesRegex(ValueError, "unit_prune_files"):
            run_state(self.state, through="freq", execute=True)
        self.assertFalse((self.state / STAGE_FREQ / "units").exists())

    def test_wavecar_fallback_in_a_unit_is_rejected(self):
        run_state(self.state, through="static", execute=True)
        marker(self.state, "wavefallback")
        with self.assertRaisesRegex(RuntimeError, "need attention"):
            run_state(self.state, through="freq", execute=True)
        unit = attempt(self.state / STAGE_FREQ / "units" / "u001_atom002_x_plus")
        self.assertEqual(unit["status"], "failed")
        self.assertIn("WAVECAR restart fallback", unit["reason"])


class FinalProtocolDisplacementTests(DisplacementFrequencyTests):
    """The same displacement tests with frequency.central = "final" (04_static follows 02_final)."""

    config_updates = {"frequency": FINAL_PROTOCOL}

    def test_no_refine_stage_ran(self):
        self.assertEqual(run_state(self.state, through="freq", execute=True)["status"], "complete")
        self.assertEqual(
            [c for c in calls(self.state) if not c.startswith("05_freq/")], ["01_loose", "02_final", "04_static"]
        )
        self.assertFalse((self.state / "03_refine").exists())


class IBRION5AndParserTests(_Base):
    config_updates = {"frequency": {"method": "vasp_ibrion5"}}

    def test_vasp_ibrion5_method(self):
        self.assertEqual(run_state(self.state, through="freq", execute=True)["status"], "complete")
        self.assertEqual(calls(self.state)[-1], "05_freq")
        parsed = parse_run(self.state / STAGE_FREQ)
        self.assertEqual(parsed["frequency_method"], "vasp_ibrion5")
        self.assertTrue(parsed["frequency_converged"])
        self.assertEqual(parsed["frequencies_cm1"], [100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        self.assertEqual(parsed["n_displacements_expected"], 12)
        self.assertEqual(attempt(self.state / STAGE_FREQ)["restart_evidence"]["verified"], True)

    def test_eentro_and_completed_ionic_steps_for_every_stage(self):
        run_state(self.state, through="static", execute=True)
        final = parse_run(self.state / STAGE_FINAL)
        self.assertEqual(final["n_completed_ionic_steps"], 2)
        self.assertAlmostEqual(final["eentro_eV"], -0.0001 * 2 - 0.00001 * 2)
        static = parse_run(self.state / STAGE_STATIC)
        self.assertEqual(static["n_completed_ionic_steps"], 1)
        self.assertAlmostEqual(static["eentro_eV"], -0.0001 * 3 - 0.00001)
        empty = parse_run(self.root / "nowhere")
        self.assertIsNone(empty["eentro_eV"])
        self.assertEqual(empty["n_completed_ionic_steps"], 0)

    @unittest.skipUnless(ASE_OUTCAR.is_file(), "ASE test data not installed")
    def test_real_vasp_outcar(self):
        folder = self.root / "real"
        folder.mkdir()
        (folder / "OUTCAR").write_bytes(ASE_OUTCAR.read_bytes())
        parsed = parse_run(folder)
        self.assertEqual(parsed["energy_e0_eV"], -68.23102426)
        self.assertEqual(parsed["eentro_eV"], 0.00701681)
        self.assertEqual(parsed["n_completed_ionic_steps"], 1)
        self.assertEqual(parsed["n_scf_iterations"], 63)
        self.assertTrue(parsed["completed"] and parsed["electronic_converged"])


class MassAndDeadlineUnitTests(unittest.TestCase):
    def test_mass_table_falls_back_to_standard_masses_with_a_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            potcar = Path(tmp) / "POTCAR"
            potcar.write_text(
                "TITEL = PAW_PBE O\nVRHFIN =O: s2p4\nEnd of Dataset\n"
                "TITEL = PAW_PBE H\nVRHFIN =H: s1\nPOMASS =    1.000; ZVAL = 1\nEnd of Dataset\n"
            )
            masses, source, warnings = mass_table(potcar, ["O", "H"])
            self.assertEqual(source, "standard atomic masses")
            self.assertAlmostEqual(masses["O"], 15.999)
            self.assertTrue(warnings)
            potcar.write_text(
                "VRHFIN =O: s2p4\n   POMASS =   16.000; ZVAL   =    6.000    mass and valenz\nEnd of Dataset\n"
            )
            self.assertEqual(mass_table(potcar, ["O"])[:2], ({"O": 16.0}, "POTCAR POMASS"))

    def test_deadline_uses_configured_minimum_and_measured_unit_durations(self):
        stopper = StopController({"scheduler": {"min_remaining_s": {"displacement": 100}}}, environ={})
        stopper.end_time = time.time() + 1000
        stopper.start_time = stopper.end_time - 86000  # a 24 h allocation
        stopper.check("displacement")
        stopper.check("displacement", measured_s=700)
        with self.assertRaisesRegex(TimeLimitReached, "needs at least 1250"):
            stopper.check("displacement", measured_s=1000)
        with self.assertRaisesRegex(TimeLimitReached, "single_point"):
            stopper.check("single_point")  # default 5400 s
        stopper.requested = "SIGUSR1"
        with self.assertRaisesRegex(TimeLimitReached, "SIGUSR1"):
            stopper.check("relaxation")
        self.assertIsNone(StopController({}, environ={}).end_time)
        # A measured duration longer than the whole allocation must not block every future job.
        short = StopController({"scheduler": {"min_remaining_s": {"single_point": 100}}}, environ={})
        short.end_time = time.time() + 950
        short.start_time = short.end_time - 1000
        short.check("single_point", measured_s=5000)
        self.assertIn("starting it anyway", short.events[-1]["warning"])
        # Later in the same allocation the capped requirement (900 s) is no longer met:
        # the stage is not started and no "starting it anyway" warning is logged.
        late = StopController({"scheduler": {"min_remaining_s": {"single_point": 100}}}, environ={})
        late.end_time = time.time() + 500
        late.start_time = late.end_time - 1000
        with self.assertRaisesRegex(TimeLimitReached, "needs at least 900"):
            late.check("single_point", measured_s=5000)
        self.assertEqual(late.events, [])
        self.assertEqual(StopController({}, environ={"SLURM_JOB_END_TIME": "1234.5"}).end_time, 1234.5)


if __name__ == "__main__":
    unittest.main()
