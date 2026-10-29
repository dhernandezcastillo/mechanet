"""Execution integration tests use a clearly synthetic VASP process, never VASP.

The fake process emits VASP-formatted convergence markers so these tests exercise
the real parser and stage controller together, including failure boundaries.
"""

import errno
import json
import os
import signal
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from support import attempt, calls, make_campaign, marker, run_cli  # noqa: E402
from mechanet import runner  # noqa: E402
from mechanet.fileio import read_json, write_json  # noqa: E402
from mechanet.results import parse_run  # noqa: E402
from mechanet.runner import (
    LockBusy,
    _archive_counts,
    cleanup,
    exclusive_lock,
    prepare,  # noqa: E402
    run_state,
)
from mechanet.slurm import submit  # noqa: E402
from mechanet.structure import read_poscar, write_poscar  # noqa: E402
from mechanet.vasp import (
    STAGES,
    STAGE_FINAL,
    STAGE_FREQ,
    STAGE_LOOSE,
    STAGE_REFINE,  # noqa: E402
    STAGE_STATIC,
    assemble_potcar,
    stage_settings,
)


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

    def set_config(self, section, **values):
        config = read_json(self.campaign / "config.json")
        config.setdefault(section, {}).update(values)
        write_json(self.campaign / "config.json", config)


class StageProtocolTests(_Base):
    def test_stage_settings_follow_interface_table(self):
        self.assertEqual(STAGES, ("01_loose", "02_final", "03_refine", "04_static", "05_freq"))
        v = {s: stage_settings(self.campaign, self.meta, s)["values"] for s in STAGES}
        self.assertEqual(
            (v[STAGE_LOOSE]["PREC"], v[STAGE_LOOSE]["EDIFF"], v[STAGE_LOOSE]["NSW"]), ("Normal", "1e-4", "40")
        )
        self.assertEqual((v[STAGE_LOOSE]["ISTART"], v[STAGE_LOOSE]["ICHARG"]), ("0", "2"))
        self.assertEqual(v[STAGE_FINAL]["LDAUU"], "2.56 0 0")
        self.assertEqual(v[STAGE_FINAL]["NCORE"], "8")
        self.assertEqual(
            {k: v[STAGE_FINAL][k] for k in ("ISTART", "ICHARG", "LWAVE", "LCHARG", "LMAXMIX", "NSW", "EDIFFG")},
            {
                "ISTART": "0",
                "ICHARG": "2",
                "LWAVE": ".FALSE.",
                "LCHARG": ".TRUE.",
                "LMAXMIX": "4",
                "NSW": "500",
                "EDIFFG": "-0.03",
            },
        )
        self.assertEqual(
            {
                k: v[STAGE_REFINE][k]
                for k in (
                    "ISTART",
                    "ICHARG",
                    "PREC",
                    "LREAL",
                    "ISYM",
                    "EDIFF",
                    "EDIFFG",
                    "IBRION",
                    "NSW",
                    "LWAVE",
                    "LCHARG",
                    "LMAXMIX",
                )
            },
            {
                "ISTART": "0",
                "ICHARG": "1",
                "PREC": "Accurate",
                "LREAL": ".FALSE.",
                "ISYM": "0",
                "EDIFF": "1e-06",
                "EDIFFG": "-0.015",
                "IBRION": "2",
                "NSW": "60",
                "LWAVE": ".FALSE.",
                "LCHARG": ".TRUE.",
                "LMAXMIX": "4",
            },
        )
        self.assertEqual(
            {
                k: v[STAGE_STATIC][k]
                for k in (
                    "ISTART",
                    "ICHARG",
                    "EDIFF",
                    "NELM",
                    "LREAL",
                    "ISYM",
                    "IBRION",
                    "NSW",
                    "LWAVE",
                    "LCHARG",
                    "LMAXMIX",
                )
            },
            {
                "ISTART": "0",
                "ICHARG": "1",
                "EDIFF": "1e-07",
                "NELM": "120",
                "LREAL": ".FALSE.",
                "ISYM": "0",
                "IBRION": "-1",
                "NSW": "0",
                "LWAVE": ".TRUE.",
                "LCHARG": ".FALSE.",
                "LMAXMIX": "4",
            },
        )
        # Displacement units: static settings except ISTART 1, ICHARG 0, LWAVE/LCHARG off.
        freq = stage_settings(self.campaign, self.meta, STAGE_FREQ)
        self.assertEqual(freq["method"], "displacements")
        self.assertEqual(freq["displacement_A"], 0.015)
        self.assertEqual(
            {k: freq["values"][k] for k in ("ISTART", "ICHARG", "IBRION", "NSW", "LWAVE", "LCHARG", "EDIFF")},
            {
                "ISTART": "1",
                "ICHARG": "0",
                "IBRION": "-1",
                "NSW": "0",
                "LWAVE": ".FALSE.",
                "LCHARG": ".FALSE.",
                "EDIFF": "1e-07",
            },
        )
        self.assertNotIn("POTIM", freq["values"])
        self.assertNotIn("MAGMOM", v[STAGE_FINAL])
        config = read_json(self.campaign / "config.json")
        config["vasp"]["magmom_by_element"] = {"Fe": 5, "O": 0, "H": 0}
        self.assertEqual(stage_settings(self.campaign, self.meta, STAGE_FINAL, config)["values"]["MAGMOM"], "5 0 0")
        config["frequency"]["method"] = "vasp_ibrion5"
        values = stage_settings(self.campaign, self.meta, STAGE_FREQ, config)["values"]
        self.assertEqual((values["IBRION"], values["NSW"], values["NFREE"], values["POTIM"]), ("5", "1", "2", "0.015"))

    def test_stage_consistency_rejects_protocol_drift(self):
        config = read_json(self.campaign / "config.json")
        for stage, override, message in (
            (STAGE_REFINE, {"ICHARG": 2}, "ICHARG=1"),
            (STAGE_STATIC, {"LREAL": "Auto"}, "LREAL"),
            (STAGE_STATIC, {"LWAVE": False}, "WAVECAR"),
            (STAGE_FINAL, {"LCHARG": False}, "LCHARG"),
            (STAGE_REFINE, {"EDIFFG": 0.001}, "EDIFFG"),
        ):
            bad = json.loads(json.dumps(config))
            bad["vasp"]["stage_overrides"] = {stage: override}
            with self.assertRaisesRegex(ValueError, message):
                stage_settings(self.campaign, self.meta, stage, bad)
        bad = json.loads(json.dumps(config))
        bad["frequency"]["nfree"] = 4
        with self.assertRaisesRegex(ValueError, "central differences"):
            stage_settings(self.campaign, self.meta, STAGE_FREQ, bad)

    def test_potcar_species_validation(self):
        config = read_json(self.campaign / "config.json")
        bad = Path(config["vasp"]["potcars"]["H"])
        bad.write_text("TITEL = PAW_PBE O\nVRHFIN = O: test\nEnd of Dataset\n")
        with self.assertRaisesRegex(ValueError, "Expected one H"):
            assemble_potcar(self.campaign, self.meta)


class StageChainTests(_Base):
    def test_full_stage_chain_and_resume_use_real_parser(self):
        prepare(self.campaign)
        result = run_state(self.state, through="freq", execute=True)
        self.assertEqual(result["status"], "complete")
        units = [c for c in calls(self.state) if c.startswith("05_freq/")]
        self.assertEqual(calls(self.state)[:4], ["01_loose", "02_final", "03_refine", "04_static"])
        self.assertEqual(len(units), 12)  # 2 active atoms x 3 Cartesian DOF x (+h, -h)
        freq = read_poscar(self.state / STAGE_FREQ / "POSCAR")
        static = read_poscar(self.state / STAGE_STATIC / "POSCAR")
        refined = read_poscar(self.state / STAGE_REFINE / "CONTCAR")
        self.assertTrue(np.allclose(freq.positions, static.positions))
        self.assertTrue(np.allclose(static.positions, refined.positions))
        self.assertEqual(freq.mobile.tolist(), self.meta["frequency_mobile"])
        self.assertFalse((self.state / STAGE_FINAL / "WAVECAR").exists())  # 02_final writes no WAVECAR any more
        refine_inputs = read_json(self.state / STAGE_REFINE / "input_hashes.json")
        self.assertEqual(refine_inputs["restart_files"]["CHGCAR"]["source"], str(self.state / STAGE_FINAL / "CHGCAR"))
        self.assertNotIn("CHGCAR", refine_inputs["files"])
        self.assertTrue((self.state / STAGE_FREQ / "frequencies.json").is_file())
        self.assertEqual(
            set(attempt(self.state / STAGE_FINAL)["output_hashes"]),
            {"OUTCAR", "vasprun.xml", "CONTCAR"} & set(attempt(self.state / STAGE_FINAL)["output_hashes"]),
        )
        self.assertNotIn("CHGCAR", attempt(self.state / STAGE_FINAL)["output_hashes"])
        parsed = parse_run(self.state / STAGE_FREQ)
        for key in ("completed", "electronic_converged", "frequency_converged", "frequency_spectrum_complete"):
            self.assertTrue(parsed[key], key)
        self.assertEqual(
            (
                parsed["frequency_method"],
                parsed["n_displacements_expected"],
                parsed["n_displacements_done"],
                parsed["expected_frequency_modes"],
            ),
            ("displacements", 12, 12, 6),
        )
        self.assertEqual(len(parsed["magnetic_moment_history"]), 12)
        self.assertIn("DO_NOT_USE_FOR_CHE", parsed["energy_context"])
        self.assertTrue((self.state / STAGE_FREQ / "input_hashes.json").is_file())
        self.assertEqual(attempt(self.state / STAGE_FREQ)["status"], "complete")
        for unit in (self.state / STAGE_FREQ / "units").iterdir():
            self.assertFalse((unit / "WAVECAR").exists(), "unit WAVECAR copies are deleted after acceptance")
        run_state(self.state, through="freq", execute=True)
        self.assertEqual(len(calls(self.state)), 16)
        self.assertEqual(read_json(self.state / "run.json")["status"], "complete")

    def test_failed_final_blocks_downstream_and_explicit_retry_archives(self):
        marker(self.state, "fail_02_final")
        with self.assertRaisesRegex(RuntimeError, "Final geometry"):
            run_state(self.state, through="freq", execute=True)
        self.assertFalse((self.state / STAGE_REFINE / "OUTCAR").exists())
        self.assertEqual(read_json(self.state / "run.json")["status"], "failed")
        with self.assertRaisesRegex(RuntimeError, "--retry"):
            run_state(self.state, through="freq", execute=True)
        result = run_state(self.state, through="freq", execute=True, retry=True)
        self.assertEqual(result["status"], "complete")
        archived = self.state / STAGE_FINAL / "previous_attempts" / "001"
        self.assertTrue((archived / "OUTCAR").is_file())
        self.assertEqual(read_json(archived / "archive.json")["kind"], "failed")
        self.assertFalse((archived / "CHGCAR").exists(), "restart-file copies are not archived")

    def test_refine_requires_its_tighter_force_criterion(self):
        marker(self.state, "force_03_refine", "0.02")  # below 02_final's 0.03, above refine's 0.015
        with self.assertRaisesRegex(RuntimeError, "Refined geometry: movable-atom force 0.02 exceeds 0.015"):
            run_state(self.state, through="refine", execute=True)

    def test_static_force_gate_blocks_frequencies_and_warns_between_thresholds(self):
        marker(self.state, "bigforce_04_static")
        with self.assertRaisesRegex(RuntimeError, "static active force 0.2 exceeds 0.03"):
            run_state(self.state, through="freq", execute=True)
        self.assertTrue((self.state / STAGE_STATIC / "OUTCAR").exists())
        self.assertTrue((self.state / STAGE_FREQ / "BLOCKED.json").exists())
        self.assertFalse((self.state / STAGE_FREQ / "units").exists())
        other = self.root / "second"
        campaign, state, _ = make_campaign(other)
        marker(state, "force_04_static", "0.025")  # above warn_force_eVA (0.02), below max_force_eVA (0.03)
        self.assertEqual(run_state(state, through="freq", execute=True)["status"], "complete")
        report = read_json(state / STAGE_FREQ / "frequencies.json")
        self.assertAlmostEqual(report["static_force_check"]["max_active_force_eVA"], 0.025)
        self.assertTrue(any("warn_force_eVA" in w for w in report["warnings"]))

    def test_changed_inputs_cannot_reuse_outputs(self):
        run_state(self.state, through="final", execute=True)
        path = self.state / STAGE_FINAL / "INCAR"
        path.write_text(path.read_text().replace("ENCUT = 500", "ENCUT = 400"))
        with self.assertRaisesRegex(ValueError, "modified after preparation"):
            run_state(self.state, through="static", execute=True)

    def test_edited_final_geometry_cannot_be_used_as_converged_output(self):
        run_state(self.state, through="final", execute=True)
        path = self.state / STAGE_FINAL / "CONTCAR"
        atoms = read_poscar(path)
        atoms.positions[1, 0] += 0.1
        write_poscar(path, atoms)
        with self.assertRaisesRegex(ValueError, "Accepted output changed"):
            run_state(self.state, through="static", execute=True)

    def test_hdf5_species_tokens_are_accepted(self):
        marker(self.state, "hdf5_species")
        self.assertEqual(run_state(self.state, through="static", execute=True)["status"], "complete")
        self.assertIn("Fe/1a2b3c4d", (self.state / STAGE_REFINE / "CONTCAR").read_text())
        self.assertEqual(read_poscar(self.state / STAGE_REFINE / "CONTCAR").symbols, ["Fe", "O", "H"])

    def test_static_atomic_start_line_is_a_restart_fallback(self):
        marker(self.state, "chg_atomic_04_static")
        with self.assertRaisesRegex(RuntimeError, "fallback of required CHGCAR restart .*overlapping atoms"):
            run_state(self.state, through="static", execute=True)
        refine = attempt(self.state / STAGE_REFINE)
        self.assertTrue(refine["restart_evidence"]["verified"])  # "charge-density read from file: ..."
        self.assertFalse(refine["restart_evidence"]["fallback"])


class InterruptionTests(_Base):
    def test_walltime_kill_continues_from_last_ionic_contcar_without_retry(self):
        marker(self.state, "kill_02_final")
        killed = run_cli(["run-state", str(self.state), "--through", "final", "--execute"])
        self.assertEqual(killed.returncode, -signal.SIGKILL)
        self.assertEqual(attempt(self.state / STAGE_FINAL)["status"], "running")
        killed_contcar = read_poscar(self.state / STAGE_FINAL / "CONTCAR")
        start = read_poscar(self.state / STAGE_LOOSE / "CONTCAR")
        self.assertGreater(abs(killed_contcar.positions[1, 0] - start.positions[1, 0]), 0.02)
        result = run_state(self.state, through="final", execute=True)  # no --retry
        self.assertEqual(result["status"], "complete")
        archive = self.state / STAGE_FINAL / "previous_attempts" / "001"
        self.assertEqual(read_json(archive / "archive.json")["kind"], "interrupted")
        inputs = read_json(self.state / STAGE_FINAL / "input_hashes.json")
        self.assertEqual(inputs["geometry_source"], str(archive / "CONTCAR"))
        self.assertEqual(inputs["continuation"]["previous_completed_ionic_steps"], 3)
        self.assertTrue(
            np.allclose(read_poscar(self.state / STAGE_FINAL / "POSCAR").positions, killed_contcar.positions)
        )
        self.assertEqual(
            _archive_counts(self.state / STAGE_FINAL), {"failed": 0, "interrupted": 1, "launcher_error": 0}
        )

    def test_stop_between_archiving_and_launch_keeps_the_continuation_geometry(self):
        marker(self.state, "kill_02_final")
        run_cli(["run-state", str(self.state), "--through", "final", "--execute"])
        killed_contcar = read_poscar(self.state / STAGE_FINAL / "CONTCAR")
        captured = {}
        real_install, real_geometry = runner.StopController.install, runner._stage_geometry

        def install(stopper):
            captured["stopper"] = stopper
            real_install(stopper)

        def geometry(*args, **kwargs):
            source = real_geometry(*args, **kwargs)
            captured["stopper"].requested = "SIGUSR1"  # signal lands after archiving, before launch
            return source

        with patch.object(runner.StopController, "install", install), patch.object(runner, "_stage_geometry", geometry):
            self.assertEqual(run_state(self.state, through="final", execute=True)["status"], "incomplete_time_limit")
        pending = read_json(self.state / STAGE_FINAL / "continuation.json")
        self.assertTrue(pending["geometry"].endswith("previous_attempts/001/CONTCAR"))
        self.assertFalse((self.state / STAGE_FINAL / "attempt.json").exists())
        self.assertEqual(run_state(self.state, through="final", execute=True)["status"], "complete")
        self.assertTrue(
            np.allclose(read_poscar(self.state / STAGE_FINAL / "POSCAR").positions, killed_contcar.positions)
        )
        self.assertFalse((self.state / STAGE_FINAL / "continuation.json").exists())
        self.assertEqual(_archive_counts(self.state / STAGE_FINAL)["interrupted"], 1)

    def test_max_continuations_caps_automatic_continuation(self):
        self.set_config("vasp", max_continuations=1)
        for _ in range(2):
            marker(self.state, "kill_02_final")
            self.assertEqual(
                run_cli(["run-state", str(self.state), "--through", "final", "--execute"]).returncode, -signal.SIGKILL
            )
        with self.assertRaisesRegex(RuntimeError, "max_continuations=1"):
            run_state(self.state, through="final", execute=True)

    def test_sigusr1_writes_stopcar_and_run_ends_incomplete_with_exit_zero(self):
        marker(self.state, "slow_02_final")
        process = run_cli(["run-state", str(self.state), "--through", "final", "--execute"], wait=False)
        started = self.state / "ctl" / "vasp_started"
        deadline = time.time() + 60
        while not started.exists() and time.time() < deadline:
            time.sleep(0.05)
        self.assertTrue(started.exists(), process.stderr.read() if process.poll() is not None else "timeout")
        time.sleep(0.3)
        process.send_signal(signal.SIGUSR1)
        out, err = process.communicate(timeout=120)
        self.assertEqual(process.returncode, 0, err)
        run = read_json(self.state / "run.json")
        self.assertEqual(run["status"], "incomplete_time_limit")
        self.assertEqual(run["signals"][0]["signal"], "SIGUSR1")
        self.assertTrue(run["signals"][0]["stopcar"])
        record = attempt(self.state / STAGE_FINAL)
        self.assertEqual(record["status"], "interrupted")
        self.assertEqual((self.state / STAGE_FINAL / "STOPCAR").read_text().strip(), "LSTOP = .TRUE.")
        self.assertEqual(json.loads(out)["status"], "incomplete_time_limit")
        (self.state / "ctl" / "slow_02_final").unlink()
        self.assertEqual(run_state(self.state, through="final", execute=True)["status"], "complete")
        archive = self.state / STAGE_FINAL / "previous_attempts" / "001"
        self.assertEqual(read_json(archive / "archive.json")["kind"], "interrupted")
        self.assertTrue((archive / "STOPCAR").exists())
        self.assertFalse((self.state / STAGE_FINAL / "STOPCAR").exists())
        self.assertIn("CONTCAR", read_json(self.state / STAGE_FINAL / "input_hashes.json")["geometry_source"])

    def test_deadline_stops_cleanly_before_a_stage(self):
        # A 24 h job with 120 s left (< min_remaining_s.relaxation, 1800 s).
        os.environ["SLURM_JOB_START_TIME"] = str(time.time() - 86280)
        os.environ["SLURM_JOB_END_TIME"] = str(time.time() + 120)
        result = run_state(self.state, through="final", execute=True)
        self.assertEqual(result["status"], "incomplete_time_limit")
        self.assertIn("relaxation", result["reason"])
        self.assertFalse((self.state / STAGE_LOOSE / "attempt.json").exists())
        self.assertEqual(calls(self.state), [])
        self.set_config("scheduler", min_remaining_s={"relaxation": 1, "single_point": 10**9, "displacement": 1})
        result = run_state(self.state, through="static", execute=True)
        self.assertEqual(result["status"], "incomplete_time_limit")
        self.assertEqual(result["stopped_stage"], STAGE_STATIC)
        self.assertEqual(calls(self.state), ["01_loose", "02_final", "03_refine"])
        del os.environ["SLURM_JOB_END_TIME"], os.environ["SLURM_JOB_START_TIME"]
        self.assertEqual(run_state(self.state, through="static", execute=True)["status"], "complete")
        self.assertEqual(calls(self.state)[-1], "04_static")

    def test_half_recorded_attempt_is_finalized(self):
        marker(self.state, "postkill_02_final")
        self.assertEqual(
            run_cli(["run-state", str(self.state), "--through", "final", "--execute"]).returncode, -signal.SIGKILL
        )
        self.assertEqual(attempt(self.state / STAGE_FINAL)["status"], "running")
        result = run_state(self.state, through="final", execute=True)
        self.assertEqual(result["stages"][1]["status"], "finalized")
        record = attempt(self.state / STAGE_FINAL)
        self.assertEqual(record["status"], "complete")
        self.assertIn("bookkeeping finalized after interruption", record["warnings"])
        self.assertEqual(
            set(record["output_hashes"]), {"OUTCAR", "vasprun.xml", "CONTCAR"} & set(record["output_hashes"])
        )
        self.assertIn("OUTCAR", record["output_hashes"])
        self.assertEqual(calls(self.state).count("02_final"), 1)

    def test_launcher_error_is_relaunched_without_retry_or_attempt_use(self):
        config = read_json(self.campaign / "config.json")
        good = config["vasp"]["command"]
        self.set_config("vasp", command=[str(self.root / "missing" / "vasp_std")])
        with self.assertRaisesRegex(RuntimeError, "Could not start the VASP launcher"):
            run_state(self.state, through="final", execute=True)
        self.assertEqual(attempt(self.state / STAGE_LOOSE)["status"], "launcher_error")
        self.set_config("vasp", command=good, max_attempts=1)  # no failure budget at all
        self.assertEqual(run_state(self.state, through="final", execute=True)["status"], "complete")
        self.assertEqual(
            _archive_counts(self.state / STAGE_LOOSE), {"failed": 0, "interrupted": 0, "launcher_error": 1}
        )


class LockAndPreparationTests(_Base):
    def test_prepare_refuses_a_campaign_not_written_by_mechanet(self):
        # mechanet writes build reports with schema 3; a campaign with an older schema is refused.
        write_json(self.campaign / "build_report.json", {"schema_version": 1, "unique_initial_structures": 519})
        with self.assertRaisesRegex(ValueError, "not generated by mechanet"):
            prepare(self.campaign)
        write_json(self.campaign / "build_report.json", {"schema_version": 2})
        with self.assertRaisesRegex(ValueError, "not generated by mechanet"):
            prepare(self.campaign)
        write_json(self.campaign / "build_report.json", {"schema_version": 3})
        self.assertEqual(prepare(self.campaign)["count"], 1)

    def test_measured_duration_ignores_runs_killed_before_completion(self):
        folder = self.state / STAGE_FINAL
        folder.mkdir(exist_ok=True)
        archive = folder / "previous_attempts" / "001"
        archive.mkdir(parents=True)
        # A run that hung after one ionic step and was killed at the walltime.
        write_json(
            archive / "attempt.json",
            {"status": "interrupted", "elapsed_s": 84600, "result": {"completed": False, "n_completed_ionic_steps": 1}},
        )
        self.assertIsNone(runner._measured_seconds(folder, per_ionic_step=True))
        # A cleanly stopped run (footer written) does count.
        write_json(
            folder / "attempt.json",
            {"status": "interrupted", "elapsed_s": 3000, "result": {"completed": True, "n_completed_ionic_steps": 10}},
        )
        self.assertAlmostEqual(runner._measured_seconds(folder, per_ionic_step=True), 300.0)

    def test_enosys_flock_falls_back_to_directory_lock(self):
        base = self.state / ".run.lock"
        unsupported = OSError(errno.ENOSYS, "Function not implemented")
        with patch.object(runner.fcntl, "flock", side_effect=unsupported):
            with exclusive_lock(base, "test") as info:
                self.assertEqual(info["mode"], "mkdir")
                owner = read_json(self.state / ".run.lock.d" / "owner.json")
                self.assertEqual((owner["host"], owner["pid"]), (socket.gethostname(), os.getpid()))
                with self.assertRaises(LockBusy):
                    with exclusive_lock(base, "test"):
                        pass
            self.assertFalse((self.state / ".run.lock.d").exists())
            # A stale owner (dead process on this host) is broken and the lock acquired.
            (self.state / ".run.lock.d").mkdir()
            write_json(
                self.state / ".run.lock.d" / "owner.json",
                {"host": socket.gethostname(), "pid": 2**22 + 12345, "slurm_job_id": None, "token": "old"},
            )
            with exclusive_lock(base, "test") as info:
                self.assertEqual(read_json(self.state / ".run.lock.d" / "owner.json")["pid"], os.getpid())
            # A live owner keeps the lock.
            (self.state / ".run.lock.d").mkdir()
            write_json(
                self.state / ".run.lock.d" / "owner.json",
                {"host": socket.gethostname(), "pid": os.getppid(), "slurm_job_id": None, "token": "live"},
            )
            with self.assertRaises(LockBusy):
                with exclusive_lock(base, "test"):
                    pass
            # run_state itself works on such a file system.
            (self.state / ".run.lock.d" / "owner.json").unlink()
            (self.state / ".run.lock.d").rmdir()
            self.assertEqual(run_state(self.state, through="loose", execute=True)["lock"], "mkdir")

    def test_flock_conflict_is_reported(self):
        with exclusive_lock(self.state / ".run.lock", "test"):
            with self.assertRaisesRegex(RuntimeError, "Another runner holds the lock"):
                run_state(self.state, through="loose", execute=True)

    def test_prepare_does_not_overwrite_job_script_but_submit_writes_requested_scope(self):
        prepare(self.campaign)
        script = self.state / "run_job.sh"
        self.assertIn("--through final", script.read_text())
        script.write_text("#!/bin/bash\n# user edited\n")
        prepare(self.campaign)
        run_state(self.state, through="loose", execute=True)  # run_state also calls prepare()
        self.assertEqual(script.read_text(), "#!/bin/bash\n# user edited\n")
        with patch("subprocess.run", side_effect=AssertionError("No external process allowed")):
            out = submit(self.campaign, through="refine", execute=False)
        self.assertEqual(out["jobs"][0]["status"], "dry_run")
        text = script.read_text()
        self.assertIn("--through refine --execute", text)
        self.assertIn("#SBATCH --signal=B:USR1@1800", text)
        self.assertIn("--mail-type=NONE", text)
        self.assertNotIn("--mail-user", text)
        # mechanet renames the interpreter variable of the job script (OER_PYTHON -> MECHANET_PY).
        self.assertIn('exec "$MECHANET_PY"', text)


class RestartFileTests(_Base):
    def test_deleting_restart_files_after_acceptance_is_harmless_and_cleanup_respects_consumers(self):
        run_state(self.state, through="static", execute=True)
        dry = cleanup(self.campaign)
        listed = {Path(f["path"]).relative_to(self.state).as_posix() for s in dry["states"] for f in s["files"]}
        self.assertEqual(listed, {"02_final/CHGCAR", "03_refine/CHGCAR", "04_static/CHGCAR"})
        self.assertTrue((self.state / STAGE_FINAL / "CHGCAR").exists(), "dry run deletes nothing")
        cleanup(self.campaign, execute=True)
        for name in ("02_final/CHGCAR", "03_refine/CHGCAR", "04_static/CHGCAR"):
            self.assertFalse((self.state / name).exists())
        self.assertTrue((self.state / STAGE_STATIC / "WAVECAR").exists(), "05_freq still needs the static WAVECAR")
        self.assertEqual(run_state(self.state, through="static", execute=True)["status"], "complete")
        with patch("subprocess.run", side_effect=AssertionError("completion is checked before SLURM")):
            out = submit(self.campaign, through="static", execute=True)
        self.assertEqual(out["jobs"][0]["status"], "skipped_complete")
        self.assertEqual(run_state(self.state, through="freq", execute=True)["status"], "complete")
        listed = {
            Path(f["path"]).relative_to(self.state).as_posix()
            for s in cleanup(self.campaign)["states"]
            for f in s["files"]
        }
        self.assertEqual(listed, {"04_static/WAVECAR"})
        cleanup(self.campaign, execute=True)
        self.assertEqual(run_state(self.state, through="freq", execute=True)["status"], "complete")
        self.assertTrue(parse_run(self.state / STAGE_FREQ)["frequency_converged"])
        self.assertTrue(parse_run(self.state / STAGE_STATIC)["input_hashes_valid"])

    def test_cleanup_skips_running_states(self):
        run_state(self.state, through="final", execute=True)
        with exclusive_lock(self.state / ".run.lock", "test"):
            report = cleanup(self.campaign, execute=True)
        self.assertEqual(report["skipped_running"], [self.state.name])

    def test_vasp_deleting_its_wavecar_copy_is_harmless(self):
        marker(self.state, "vasp_deletes_wavecar")
        self.assertEqual(run_state(self.state, through="freq", execute=True)["status"], "complete")
        self.assertTrue((self.state / STAGE_STATIC / "WAVECAR").exists(), "copies, never links, protect the source")


if __name__ == "__main__":
    unittest.main()
