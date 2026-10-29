"""Submission that never re-queues hopeless states, a campaign job cap, refill and ledger repair."""

import contextlib
import io
import os
import shutil
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import attempt, calls, make_campaign, marker, run_cli  # noqa: E402
from test_execution_frequencies import harmonic_model  # noqa: E402
from mechanet import slurm as scheduler  # noqa: E402
from mechanet.cli import main  # noqa: E402
from mechanet.fileio import read_json, write_json  # noqa: E402
from mechanet.runner import (
    LauncherError,
    LockBusy,
    StageFailed,
    _archive_counts,
    run_state,  # noqa: E402
    submission_check,
)
from mechanet.slurm import (
    _refill_decision,
    enable_refill,
    refill_after_run,
    refill_status,  # noqa: E402
    resolve_submission,
    set_refill,
    submit,
)
from mechanet.vasp import STAGE_FINAL, STAGE_FREQ  # noqa: E402

FAKE_SQUEUE = r"""#!/bin/sh
here="$(dirname "$0")"
echo "$@" >> "$here/squeue_calls.txt"
env | grep -c '^SQUEUE_' > "$here/last_env_squeue" || true
fmt=""; user=""; job=""
while [ $# -gt 0 ]; do
  case "$1" in
    --format=*) fmt="${1#--format=}" ;;
    -o) fmt="$2"; shift ;;
    --user=*) user="${1#--user=}" ;;
    --jobs|-j) job="$2"; shift ;;
    --jobs=*) job="${1#--jobs=}" ;;
  esac
  shift
done
if [ "$fmt" = "%e" ]; then echo "2031-05-06T07:08:09"; exit 0; fi
if [ -n "$user" ]; then
  if [ -e "$here/squeue_user_fail" ]; then echo "squeue: error: Unable to contact slurm controller" >&2; exit 1; fi
  if [ -e "$here/squeue_user.txt" ]; then cat "$here/squeue_user.txt"; fi
  exit 0
fi
if [ -n "$job" ]; then
  if [ -e "$here/job_$job.txt" ]; then cat "$here/job_$job.txt"; exit 0; fi
  echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1
fi
exit 0
"""

FAKE_SBATCH = r"""#!/bin/sh
here="$(dirname "$0")"
echo "$@" >> "$here/sbatch_calls.txt"
if [ -e "$here/sbatch_fail" ]; then echo "sbatch: error: Batch job submission failed: Invalid account" >&2; exit 1; fi
n=$(cat "$here/next_id" 2>/dev/null || echo 5000)
n=$((n + 1))
echo "$n" > "$here/next_id"
if [ -e "$here/sbatch_signal" ]; then sig=$(cat "$here/sbatch_signal"); rm -f "$here/sbatch_signal"; kill -s "$sig" "$PPID"; sleep 0.3; fi
if [ -e "$here/sbatch_selfkill" ]; then rm -f "$here/sbatch_selfkill"; kill -s KILL $$; fi
if [ -e "$here/sbatch_timeout" ]; then
  rm -f "$here/sbatch_timeout"
  echo "$n|PENDING|$(dirname "$2")" >> "$here/squeue_user.txt"
  echo "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation" >&2; exit 1
fi
env | grep -c '^SLURM_JOB_ID=' > "$here/last_env_slurm_job_id" || true
echo "$n;cluster"
"""


DYING_VASP = r"""
import os, signal, time
from pathlib import Path
Path("OUTCAR").write_text(" NIONS = 3\n Iteration    1(   1)\n BRMIX: very serious problems\n"
                          "  free energy    TOTEN  =     -100.0 eV\n")
os.kill(os.getppid(), signal.SIGTERM)   # scancel or the hard time limit: the runner marks the attempt interrupted
time.sleep(1.0)
os.kill(os.getppid(), signal.SIGKILL)   # KillWait expires before the runner records its own verdict
"""


class _Base(unittest.TestCase):
    config_updates = None
    harmonic = None

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name, text in (("squeue", FAKE_SQUEUE), ("sbatch", FAKE_SBATCH)):
            (self.bin / name).write_text(text)
            (self.bin / name).chmod(0o755)
        env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
        env["PATH"] = f"{self.bin}{os.pathsep}{env.get('PATH', '')}"
        self.env = patch.dict(os.environ, env, clear=True)
        self.env.start()
        self.campaign, self.state, _ = make_campaign(self.root, self.config_updates, self.harmonic, name="s_a")
        # mechanet's CLI recognizes a campaign by config.json and network.json; give the synthetic one a network.
        write_json(
            self.campaign / "network.json",
            {
                "reaction": "oer",
                "reference": "test",
                "nodes": [{"id": "test", "delta_atoms": {}, "state_ids": ["s_a"]}],
                "edges": [],
                "cycles": [],
            },
        )

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def clone(self, name):
        target = self.campaign / "states" / name
        shutil.copytree(self.state, target, ignore=shutil.ignore_patterns("0*_*", "plan.json", "run_job.sh"))
        meta = read_json(target / "state.json")
        meta["id"] = name
        write_json(target / "state.json", meta)
        return target

    def sbatch_count(self):
        path = self.bin / "sbatch_calls.txt"
        return len(path.read_text().splitlines()) if path.exists() else 0

    def statuses(self, out):
        return {j["state_id"]: j["status"] for j in out["jobs"]}

    def set_config(self, section, **values):
        config = read_json(self.campaign / "config.json")
        config.setdefault(section, {}).update(values)
        write_json(self.campaign / "config.json", config)


class SkipHopelessStatesTests(_Base):
    def test_failed_state_waits_for_retry_and_is_not_resubmitted(self):
        marker(self.state, "fail_02_final")
        with self.assertRaises(StageFailed):
            run_state(self.state, through="final", execute=True)
        out = submit(self.campaign, through="final", execute=True)
        self.assertEqual(self.statuses(out), {"s_a": "skipped_needs_retry"})
        self.assertEqual(out["attention"][0]["stage"], "02_final")
        self.assertEqual(self.sbatch_count(), 0)
        out = submit(self.campaign, through="final", execute=True, retry=True)
        self.assertEqual(self.statuses(out), {"s_a": "submitted"})
        self.assertIn("--retry", (self.state / "run_job.sh").read_text())

    def test_exhausted_budgets_are_skipped_even_with_retry(self):
        self.set_config("vasp", max_attempts=1)
        marker(self.state, "fail_02_final")
        with self.assertRaises(StageFailed):
            run_state(self.state, through="final", execute=True)
        out = submit(self.campaign, through="final", execute=True, retry=True)
        self.assertEqual(self.statuses(out), {"s_a": "skipped_exhausted"})
        self.assertIn("max_attempts=1", out["attention"][0]["reason"])
        self.assertEqual(
            self.statuses(submit(self.campaign, through="final")),
            {"s_a": "skipped_exhausted"},
            "--retry would not run it either, so it is reported as exhausted",
        )
        # continuation budget: an interrupted attempt after max_continuations archived interruptions
        other = self.clone("s_b")
        self.set_config("vasp", max_continuations=1)
        folder = other / "01_loose"
        (folder / "previous_attempts" / "001").mkdir(parents=True)
        write_json(folder / "previous_attempts" / "001" / "archive.json", {"kind": "interrupted"})
        write_json(folder / "attempt.json", {"status": "interrupted"})
        check = submission_check(other, "final")
        self.assertEqual((check["action"], check["status"]), ("skip", "skipped_exhausted"))
        self.assertEqual(self.sbatch_count(), 0)

    def test_force_gate_block_is_skipped_until_the_threshold_allows_it(self):
        marker(self.state, "bigforce_04_static")
        with self.assertRaisesRegex(StageFailed, "static active force"):
            run_state(self.state, through="freq", execute=True)
        self.assertEqual(self.statuses(submit(self.campaign, through="freq", execute=True)), {"s_a": "skipped_blocked"})
        self.set_config("frequency", max_force_eVA=0.5)
        self.assertEqual(self.statuses(submit(self.campaign, through="freq", execute=True)), {"s_a": "submitted"})

    def test_a_dead_runners_attempt_gets_a_verdict_so_it_is_not_queued_again(self):
        marker(self.state, "fail_02_final")
        marker(self.state, "postkill_02_final")  # VASP ends unconverged, then the runner dies
        self.assertEqual(
            run_cli(["run-state", str(self.state), "--through", "final", "--execute"]).returncode, -signal.SIGKILL
        )
        self.assertEqual(attempt(self.state / STAGE_FINAL)["status"], "running")
        self.assertEqual(submission_check(self.state, "final")["action"], "submit")  # only the runner can judge
        with self.assertRaisesRegex(StageFailed, "--retry"):
            run_state(self.state, through="final", execute=True)
        record = attempt(self.state / STAGE_FINAL)
        self.assertEqual(record["status"], "failed")
        self.assertIn("verdict_recorded_at_utc", record)
        check = submission_check(self.state, "final")
        self.assertEqual((check["action"], check["status"]), ("skip", "skipped_needs_retry"))
        self.assertEqual(run_state(self.state, through="final", execute=True, retry=True)["status"], "complete")
        self.assertEqual(_archive_counts(self.state / STAGE_FINAL)["failed"], 1)

    def test_a_dead_runner_after_the_last_continuation_is_recorded_as_exhausted(self):
        self.set_config("vasp", max_continuations=1)
        for _ in range(2):
            marker(self.state, "kill_02_final")
            self.assertEqual(
                run_cli(["run-state", str(self.state), "--through", "final", "--execute"]).returncode, -signal.SIGKILL
            )
        with self.assertRaisesRegex(StageFailed, "max_continuations=1"):
            run_state(self.state, through="final", execute=True)
        self.assertEqual(attempt(self.state / STAGE_FINAL)["status"], "interrupted")
        self.assertEqual(submission_check(self.state, "final")["status"], "skipped_exhausted")

    def test_an_interrupted_attempt_with_fatal_output_gets_a_failed_verdict(self):
        dying = self.root / "vasp_sigterm_then_killed.py"
        dying.write_text(DYING_VASP)
        normal = read_json(self.campaign / "config.json")["vasp"]["command"]
        self.set_config("vasp", command=[sys.executable, "-I", str(dying)])
        self.assertEqual(
            run_cli(["run-state", str(self.state), "--through", "loose", "--execute"]).returncode, -signal.SIGKILL
        )
        self.set_config("vasp", command=normal)
        self.assertEqual(attempt(self.state / "01_loose")["status"], "interrupted")  # from the SIGTERM handler
        with self.assertRaisesRegex(StageFailed, "--retry"):
            run_state(self.state, through="loose", execute=True)
        self.assertEqual(attempt(self.state / "01_loose")["status"], "failed")
        self.assertEqual(submission_check(self.state, "loose")["status"], "skipped_needs_retry")

    def test_states_furthest_along_go_first(self):
        b, c = self.clone("s_b"), self.clone("s_c")
        for folder, status in (
            (b / "01_loose", "complete"),
            (b / "02_final", "interrupted"),
            (c / "01_loose", "complete"),
        ):
            folder.mkdir(exist_ok=True)
            write_json(folder / "attempt.json", {"status": status})
        out = submit(self.campaign, through="final", limit=2)
        self.assertEqual(self.statuses(out), {"s_a": "deferred", "s_b": "dry_run", "s_c": "dry_run"})
        self.assertEqual([j["state_id"] for j in out["jobs"] if j["status"] == "dry_run"], ["s_b", "s_c"])
        self.assertEqual(
            submit(self.campaign, through="final", limit=1)["jobs"][1]["status"],
            "dry_run",
            "an interrupted 02_final (s_b) goes before a fresh 02_final (s_c)",
        )
        self.assertIn("signals", run_state(self.state, through="loose", execute=True))

    def test_limit_counts_submissions_not_the_first_states(self):
        self.clone("s_b")
        run_state(self.state, through="loose", execute=True)
        out = submit(self.campaign, through="loose", execute=True, limit=1)
        self.assertEqual(self.statuses(out), {"s_a": "skipped_complete", "s_b": "submitted"})
        preview = submit(self.campaign, through="final", limit=1)
        self.assertEqual(self.statuses(preview), {"s_a": "dry_run", "s_b": "deferred"})
        self.assertTrue((self.state / "run_job.sh").is_file())


class JobCapTests(_Base):
    def test_cap_counts_active_campaign_jobs_and_prefers_started_states(self):
        b, c = self.clone("s_b"), self.clone("s_c")
        write_json(
            self.campaign / "submissions.json",
            {
                "schema_version": 1,
                "entries": [{"state_id": "s_a", "through": "final", "status": "submitted", "job_id": "4001"}],
            },
        )
        (self.bin / "squeue_user.txt").write_text("4001 RUNNING\n9999 PENDING\n")
        (c / "01_loose").mkdir()
        write_json(c / "01_loose" / "attempt.json", {"status": "interrupted"})  # earlier work: goes first
        out = submit(self.campaign, through="final", execute=True, max_active=2)
        self.assertEqual(self.statuses(out), {"s_a": "skipped_active", "s_b": "deferred", "s_c": "submitted"})
        self.assertEqual((out["active_jobs"], out["max_active"]), (2, 2))
        self.assertEqual(self.sbatch_count(), 1)
        # A configured cap applies without the option; the cap is reached, so nothing more is submitted.
        self.set_config("scheduler", max_active_jobs=2)
        (self.bin / "squeue_user.txt").write_text("4001 RUNNING\n5001 PENDING\n")
        out = submit(self.campaign, through="final", execute=True)
        self.assertEqual(self.statuses(out)["s_b"], "deferred")
        self.assertEqual(self.sbatch_count(), 1)
        self.assertTrue(b.is_dir())

    def test_refill_needs_a_cap_and_neither_a_preview_nor_a_failed_submission_enables_it(self):
        with self.assertRaisesRegex(ValueError, "needs a job cap"):
            submit(self.campaign, through="loose", refill=True)
        out = submit(self.campaign, through="loose", refill=True, max_active=1)
        self.assertIn("not enabled", out["refill"])
        (self.bin / "sbatch_fail").write_text("")
        out = submit(self.campaign, through="loose", execute=True, refill=True, max_active=1)
        self.assertEqual((out["error_kind"], self.statuses(out)["s_a"]), ("sbatch_failed", "submission_failed"))
        self.assertIn("not enabled", out["refill"])
        self.assertFalse((self.campaign / "refill.json").exists())


class _RefillBase(_Base):
    def run_job(self, state, job_id, **kwargs):
        """run-state as SLURM job ``job_id``; the refill's one-line report is kept in self.job_stderr."""
        os.environ["SLURM_JOB_ID"] = job_id
        try:
            with contextlib.redirect_stderr(io.StringIO()) as err:
                return run_state(state, execute=True, **kwargs)
        finally:
            self.job_stderr = err.getvalue()
            del os.environ["SLURM_JOB_ID"]


class RefillTests(_RefillBase):
    def test_finishing_jobs_top_the_campaign_up_and_problems_pause_it(self):
        self.clone("s_b")
        self.clone("s_c")
        self.clone("s_d")
        out = submit(self.campaign, through="final", execute=True, max_active=1, refill=True)
        self.assertEqual(
            self.statuses(out), {"s_a": "submitted", "s_b": "deferred", "s_c": "deferred", "s_d": "deferred"}
        )
        self.assertTrue(read_json(self.campaign / "refill.json")["enabled"])
        job_a = out["jobs"][0]["job_id"]
        self.assertEqual(self.run_job(self.state, job_a, through="final")["status"], "complete")
        ledger = read_json(self.campaign / "submissions.json")["entries"]
        self.assertNotIn("job_finished_at_utc", ledger[0], "a job never marks itself finished (requeue keeps its ID)")
        self.assertEqual((ledger[-1]["state_id"], ledger[-1]["origin"]), ("s_b", "refill"))
        event = read_json(self.campaign / "refill.json")["events"][-1]
        self.assertEqual((event["decision"], event["submitted"]), ("refill", ["s_b"]))
        self.assertEqual(
            (self.bin / "last_env_slurm_job_id").read_text().strip(),
            "0",
            "sbatch from a job must not inherit the parent job's SLURM_* variables",
        )
        # A state that needs a decision frees its slot for the next one.
        marker(self.campaign / "states" / "s_b", "fail_02_final")
        with self.assertRaises(StageFailed):
            self.run_job(self.campaign / "states" / "s_b", ledger[-1]["job_id"], through="final")
        event = read_json(self.campaign / "refill.json")["events"][-1]
        self.assertEqual((event["decision"], event["submitted"], event["attention"]), ("refill", ["s_c"], ["s_b"]))
        # A job that could not start VASP pauses the refill instead of burning through the queue.
        job_c = read_json(self.campaign / "submissions.json")["entries"][-1]["job_id"]
        os.environ["SLURM_JOB_START_TIME"] = str(time.time() - 86395)  # a 24 h job with 5 s left
        os.environ["SLURM_JOB_END_TIME"] = str(time.time() + 5)
        self.assertEqual(
            self.run_job(self.campaign / "states" / "s_c", job_c, through="final")["status"], "incomplete_time_limit"
        )
        del os.environ["SLURM_JOB_END_TIME"], os.environ["SLURM_JOB_START_TIME"]
        status = refill_status(self.campaign)
        self.assertFalse(status["enabled"])
        self.assertIn("min_remaining_s", status["disabled_reason"])
        n_jobs = self.sbatch_count()
        # So does a launcher error, after the user resumes it.
        self.assertTrue(set_refill(self.campaign, enabled=True)["enabled"])
        config = read_json(self.campaign / "config.json")
        config["vasp"]["command"] = [str(self.root / "missing" / "vasp_std")]
        write_json(self.campaign / "config.json", config)
        ledger = read_json(self.campaign / "submissions.json")
        ledger["entries"].append({"state_id": "s_d", "through": "final", "status": "submitted", "job_id": "7777"})
        write_json(self.campaign / "submissions.json", ledger)
        with self.assertRaises(LauncherError):
            self.run_job(self.campaign / "states" / "s_d", "7777", through="final")
        status = refill_status(self.campaign)
        self.assertFalse(status["enabled"])
        self.assertIn("launcher", status["disabled_reason"])
        self.assertEqual(self.sbatch_count(), n_jobs)

    def test_cancelled_and_interactive_runs_submit_nothing(self):
        self.clone("s_b")
        submit(self.campaign, through="loose", execute=True, max_active=1, refill=True)
        self.assertEqual(self.sbatch_count(), 1)
        # run-state outside SLURM (no SLURM_JOB_ID) never submits
        self.assertEqual(run_state(self.state, through="loose", execute=True)["status"], "complete")
        self.assertEqual((self.sbatch_count(), read_json(self.campaign / "refill.json")["events"]), (1, []))
        # a job ended by scancel (SIGTERM) gets no replacement, and the refill stays on
        os.environ["SLURM_JOB_ID"] = "5001"
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                event = refill_after_run(
                    self.campaign,
                    self.state,
                    {
                        "status": "incomplete_time_limit",
                        "vasp_launches": 1,
                        "slurm_job_id": "5001",
                        "signals": [{"signal": "SIGTERM", "at_utc": "2026-10-26T00:00:00+00:00"}],
                    },
                )
        finally:
            del os.environ["SLURM_JOB_ID"]
        self.assertEqual(event["decision"], "ignore")
        self.assertEqual(self.sbatch_count(), 1)
        self.assertTrue(refill_status(self.campaign)["enabled"])
        # preemption with requeue: the same job ID is pending again and still counts
        self.assertNotIn("job_finished_at_utc", read_json(self.campaign / "submissions.json")["entries"][0])
        (self.bin / "squeue_user.txt").write_text(f"5001|PENDING|{self.state}\n")
        out = submit(self.campaign, through="loose", execute=True, max_active=1)
        self.assertEqual((self.statuses(out), out["active_jobs"]), ({"s_a": "skipped_complete", "s_b": "deferred"}, 1))
        # a job this campaign did not submit (an salloc session, your own script) never refills
        events = len(read_json(self.campaign / "refill.json")["events"])
        self.assertEqual(self.run_job(self.campaign / "states" / "s_b", "8888", through="loose")["status"], "complete")
        self.assertEqual((self.sbatch_count(), len(read_json(self.campaign / "refill.json")["events"])), (1, events))

    def test_a_cluster_that_refuses_sbatch_inside_jobs_pauses_the_refill(self):
        self.clone("s_b")
        out = submit(self.campaign, through="loose", execute=True, max_active=1, refill=True)
        (self.bin / "sbatch").unlink()
        if shutil.which("sbatch"):
            self.skipTest("a real sbatch is on PATH")
        self.assertEqual(self.run_job(self.state, out["jobs"][0]["job_id"], through="loose")["status"], "complete")
        status = refill_status(self.campaign)
        self.assertFalse(status["enabled"])
        self.assertIn("sbatch cannot be run here", status["disabled_reason"])
        self.assertIn("--submit-from login", status["disabled_reason"])
        self.assertIn("refill is now paused", self.job_stderr)
        self.assertEqual(read_json(self.campaign / "submissions.json")["entries"][-1]["status"], "submission_failed")


class RefillTargetTests(_Base):
    def test_selections_merge_and_each_state_keeps_its_deepest_final_stage(self):
        b, _ = self.clone("s_b"), self.clone("s_c")
        enable_refill(self.campaign, {"s_a": "final", "s_b": "final"}, 1)
        enable_refill(self.campaign, {"s_b": "static", "s_c": "static"}, 3)
        status = enable_refill(self.campaign, {"s_b": "loose"}, 2)
        targets = read_json(self.campaign / "refill.json")["targets"]
        self.assertEqual(targets, {"s_a": "final", "s_b": "static", "s_c": "static"})
        self.assertEqual(
            (status["max_active"], status["n_states"], status["states_by_final_stage"]),
            (2, 3, {"final": 1, "static": 2}),
        )
        preview = submit(self.campaign, through=targets, max_active=5)
        self.assertEqual({j["state_id"]: j["through"] for j in preview["jobs"]}, targets)
        self.assertIn("--through static", (b / "run_job.sh").read_text())
        self.assertIn("--through final", (self.state / "run_job.sh").read_text())
        with self.assertRaisesRegex(ValueError, "No final stage"):
            submit(self.campaign, selection=["s_a", "s_b"], through={"s_a": "final"})

    def test_cli_shows_pauses_resumes_recaps_and_clears(self):
        enable_refill(self.campaign, {"s_a": "final"}, 2)

        def cli(*args):
            with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
                code = main(["refill", str(self.campaign), *args])
            return code, out.getvalue()

        self.assertEqual(cli("--off")[0], 0)
        self.assertEqual(refill_status(self.campaign)["disabled_reason"], "paused by the user")
        self.assertEqual(cli("--on", "--max-active", "6")[0], 0)
        status = refill_status(self.campaign)
        self.assertEqual((status["enabled"], status["max_active"]), (True, 6))
        self.assertEqual(cli("--off")[0], 0)
        self.assertEqual(cli("--submit-from", "login", "--on")[0], 0)
        status = refill_status(self.campaign)
        self.assertEqual((status["enabled"], status["submit_from"]), (True, "login"))
        self.assertEqual(cli("--clear", "--max-active", "3")[0], 2)
        code, text = cli("--clear")
        self.assertEqual(code, 0)
        self.assertIn("No refill", text)
        self.assertFalse((self.campaign / "refill.json").exists())
        self.assertEqual(cli("--on")[0], 2)


class RefillDecisionTests(unittest.TestCase):
    def test_what_each_way_of_ending_does(self):
        cases = [
            ({"status": "complete"}, None, "refill"),
            ({"status": "complete", "signals": [{"signal": "SIGUSR1"}]}, None, "refill"),
            ({"status": "incomplete_time_limit", "vasp_launches": 2}, None, "refill"),
            ({"status": "incomplete_time_limit", "vasp_launches": 0, "reason": "120 s left"}, None, "disable"),
            (
                {"status": "incomplete_time_limit", "vasp_launches": 3, "signals": [{"signal": "SIGTERM"}]},
                None,
                "ignore",
            ),
            ({"status": "failed", "failure_kind": "stage"}, StageFailed("x"), "refill"),
            ({"status": "failed", "failure_kind": "launcher"}, LauncherError("x"), "disable"),
            ({"status": "failed", "failure_kind": "other"}, ValueError("x"), "disable"),
            (None, LockBusy("busy"), "ignore"),
            (None, ValueError("Prepared protocol changed"), "disable"),
        ]
        for record, failure, expected in cases:
            with self.subTest(record=record, failure=failure):
                self.assertEqual(_refill_decision(record, failure)[0], expected)


class SlurmViewTests(_Base):
    def test_jobs_missing_from_the_listing_are_asked_for_and_squeue_settings_are_ignored(self):
        self.clone("s_b")
        write_json(
            self.campaign / "submissions.json",
            {
                "schema_version": 1,
                "entries": [{"state_id": "s_a", "through": "loose", "status": "submitted", "job_id": "4001"}],
            },
        )
        (self.bin / "squeue_user.txt").write_text("")  # hidden partition, or filtered by SQUEUE_* settings
        (self.bin / "job_4001.txt").write_text("PENDING\n")  # but SLURM still holds the job
        os.environ["SQUEUE_STATES"] = "R"
        out = submit(self.campaign, through="loose", execute=True, max_active=1)
        self.assertEqual(self.statuses(out), {"s_a": "skipped_active", "s_b": "deferred"})
        self.assertEqual((self.bin / "last_env_squeue").read_text().strip(), "0")
        calls = (self.bin / "squeue_calls.txt").read_text()
        self.assertTrue("--all" in calls and "--states=all" in calls)
        (self.bin / "job_4001.txt").unlink()  # purged: finished, and remembered as such
        out = submit(self.campaign, through="loose", execute=True, max_active=1)
        self.assertEqual(self.statuses(out)["s_a"], "submitted")
        self.assertTrue(read_json(self.campaign / "submissions.json")["entries"][0]["job_finished_at_utc"])
        asked = (self.bin / "squeue_calls.txt").read_text().count("--jobs 4001")
        submit(self.campaign, through="loose", execute=True, max_active=1)
        self.assertEqual((self.bin / "squeue_calls.txt").read_text().count("--jobs 4001"), asked)

    def test_a_job_in_a_state_folder_counts_even_when_the_ledger_does_not_know_it(self):
        self.clone("s_b")
        self.clone("s_c")
        (self.bin / "squeue_user.txt").write_text(f"9001|PENDING|{self.state}\n9002|RUNNING|/elsewhere/job\n")
        out = submit(self.campaign, through="loose", execute=True, max_active=2)
        self.assertEqual(self.statuses(out), {"s_a": "skipped_active", "s_b": "submitted", "s_c": "deferred"})
        self.assertEqual((out["jobs"][0]["job_id"], out["jobs"][0]["found_in_folder"]), ("9001", True))
        self.assertEqual(out["active_jobs"], 2)

    def test_an_sbatch_timeout_that_queued_the_job_is_not_submitted_twice(self):
        self.clone("s_b")
        (self.bin / "sbatch_timeout").write_text("")
        out = submit(self.campaign, through="loose", execute=True, max_active=2)
        self.assertEqual(self.statuses(out), {"s_a": "submission_unknown", "s_b": "deferred"})
        self.assertEqual(out["error_kind"], "sbatch_unknown")
        out = submit(self.campaign, through="loose", execute=True, max_active=2)
        self.assertEqual(self.statuses(out), {"s_a": "skipped_submission_unknown", "s_b": "submitted"})
        self.assertEqual(out["active_jobs"], 2, "the queued job of s_a counts toward the cap")
        resolve_submission(self.campaign, "s_a", job_id="5001")
        self.assertEqual(
            self.statuses(submit(self.campaign, through="loose", execute=True, max_active=3))["s_a"], "skipped_active"
        )


class RefillSafetyTests(_RefillBase):
    def end_job(self, state, record):
        """refill_after_run for the state's last ledger job, with a synthetic run record."""
        entries = read_json(self.campaign / "submissions.json")["entries"]
        job = [e for e in entries if e["state_id"] == Path(state).name][-1]["job_id"]
        os.environ["SLURM_JOB_ID"] = job
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                return refill_after_run(self.campaign, state, dict(record, slurm_job_id=job))
        finally:
            del os.environ["SLURM_JOB_ID"]

    def mark(self, state, status):
        folder = Path(state) / "01_loose"
        folder.mkdir(exist_ok=True)
        write_json(folder / "attempt.json", {"status": status, "reason": "synthetic"})

    def test_a_pause_wins_over_a_refill_waiting_for_the_lock(self):
        self.clone("s_b")
        out = submit(self.campaign, through="loose", execute=True, max_active=1, refill=True)
        real = scheduler._sbatch_env

        def pause_meanwhile():
            set_refill(self.campaign, enabled=False)  # `mechanet refill --off` gets the lock first
            return real()

        n = self.sbatch_count()
        with patch.object(scheduler, "_sbatch_env", pause_meanwhile):
            self.mark(self.state, "complete")
            event = self.end_job(self.state, {"status": "complete", "vasp_launches": 1})
        self.assertEqual((self.sbatch_count(), event["submitted"]), (n, []))
        self.assertIn("refill_paused", event)
        self.assertEqual(out["jobs"][0]["status"], "submitted")

    def test_three_failed_states_in_a_row_pause_the_refill(self):
        names = ["s_a", "s_b", "s_c", "s_d", "s_e"]
        for name in names[1:]:
            self.clone(name)
        submit(self.campaign, through="loose", execute=True, max_active=1, refill=True)
        states = {n: self.campaign / "states" / n for n in names}
        failed = {"status": "failed", "failure_kind": "stage", "vasp_launches": 1}
        self.mark(states["s_a"], "failed")
        self.assertEqual(self.end_job(states["s_a"], failed)["submitted"], ["s_b"])
        self.mark(states["s_b"], "complete")
        self.assertEqual(
            self.end_job(states["s_b"], {"status": "complete", "vasp_launches": 1})["submitted"], ["s_c"]
        )  # a success resets the count
        for name, nxt in (("s_c", ["s_d"]), ("s_d", ["s_e"])):
            self.mark(states[name], "failed")
            self.assertEqual(self.end_job(states[name], failed)["submitted"], nxt)
        self.mark(states["s_e"], "failed")
        event = self.end_job(states["s_e"], failed)
        self.assertEqual((event["decision"], event.get("submitted")), ("disable", None))
        self.assertIn("3 jobs in a row", refill_status(self.campaign)["disabled_reason"])

    def test_sbatch_errors_in_three_refills_in_a_row_pause_it(self):
        for name in ("s_b", "s_c"):
            self.clone(name)
        submit(self.campaign, through="loose", execute=True, max_active=1, refill=True)
        self.mark(self.state, "complete")
        ledger = read_json(self.campaign / "submissions.json")
        (self.bin / "sbatch_fail").write_text("")
        for k in range(3):
            ledger["entries"].append(
                {"state_id": "s_a", "through": "loose", "status": "submitted", "job_id": str(7001 + k)}
            )
            write_json(self.campaign / "submissions.json", ledger)
            event = self.end_job(self.state, {"status": "complete", "vasp_launches": 1})
            self.assertIn("Invalid account", event["error"])
            ledger = read_json(self.campaign / "submissions.json")
            self.assertEqual(refill_status(self.campaign)["enabled"], k < 2)
        self.assertIn("sbatch failed in 3 refills in a row", refill_status(self.campaign)["disabled_reason"])

    def test_the_refill_leaves_states_whose_last_job_made_no_progress_to_you(self):
        self.clone("s_b")
        enable_refill(self.campaign, {"s_a": "loose", "s_b": "loose"}, 1)
        write_json(
            self.state / "run.json",
            {"status": "failed", "vasp_launches": 0, "failure_kind": "stage", "reason": "synthetic"},
        )
        past = time.time() - 100
        os.utime(self.campaign / "config.json", (past - 100, past - 100))
        os.utime(self.state / "run.json", (past, past))
        out = submit(self.campaign, through="loose", execute=True, max_active=1, origin="refill")
        self.assertEqual(self.statuses(out), {"s_a": "skipped_stalled", "s_b": "submitted"})
        self.assertEqual([e["state_id"] for e in out["attention"]], ["s_a"])
        self.assertEqual(
            self.statuses(submit(self.campaign, selection=["s_a"], through="loose")),
            {"s_a": "dry_run"},
            "your own submit is not held",
        )
        self.set_config("frequency", max_force_eVA=0.05)  # a change after that job lifts the hold once
        self.assertEqual(
            self.statuses(submit(self.campaign, selection=["s_a"], through="loose", origin="refill")),
            {"s_a": "dry_run"},
        )

    def test_a_late_signal_cannot_kill_a_finished_run_and_sigterm_stops_the_refill(self):
        for name in ("s_b", "s_c"):
            self.clone(name)
        for sig, expected in (("USR1", ["s_b", "s_c"]), ("TERM", ["s_b"])):
            with self.subTest(signal=sig):
                write_json(
                    self.campaign / "submissions.json",
                    {
                        "schema_version": 1,
                        "entries": [{"state_id": "s_a", "through": "loose", "status": "submitted", "job_id": "4001"}],
                    },
                )
                with contextlib.suppress(FileNotFoundError):
                    (self.campaign / "refill.json").unlink()
                enable_refill(self.campaign, {"s_a": "loose", "s_b": "loose", "s_c": "loose"}, 3)
                (self.bin / "sbatch_signal").write_text(sig)  # sbatch signals the runner that called it
                proc = run_cli(
                    ["run-state", str(self.state), "--through", "loose", "--execute"], env={"SLURM_JOB_ID": "4001"}
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                submitted = [
                    e["state_id"]
                    for e in read_json(self.campaign / "submissions.json")["entries"]
                    if e.get("origin") == "refill" and e["status"] == "submitted"
                ]
                self.assertEqual(submitted, expected)


class SecondReviewTests(_RefillBase):
    """Regression tests for the second review of the refill (bursts, login mode, breaker, edge cases)."""

    def finish(self, state, job, record):
        os.environ["SLURM_JOB_ID"] = job
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                return refill_after_run(self.campaign, state, dict(record, slurm_job_id=job))
        finally:
            del os.environ["SLURM_JOB_ID"]

    def done(self, state):
        (state / "01_loose").mkdir(exist_ok=True)
        write_json(state / "01_loose" / "attempt.json", {"status": "complete"})

    def test_jobs_ending_together_refill_every_slot(self):
        names = ["s_a", "s_b", "s_c", "s_d", "s_e", "s_f"]
        states = {"s_a": self.state, **{n: self.clone(n) for n in names[1:]}}
        running = {"s_a": "4001", "s_b": "4002", "s_c": "4003"}
        write_json(
            self.campaign / "submissions.json",
            {
                "schema_version": 1,
                "entries": [
                    {"state_id": sid, "through": "loose", "status": "submitted", "job_id": job}
                    for sid, job in running.items()
                ],
            },
        )
        enable_refill(self.campaign, {n: "loose" for n in names}, 3)
        listing = {job: ["RUNNING", states[sid]] for sid, job in running.items()}
        for sid, job in running.items():
            self.done(states[sid])
            (self.bin / "squeue_user.txt").write_text("".join(f"{j}|{st}|{wd}\n" for j, (st, wd) in listing.items()))
            self.finish(states[sid], job, {"status": "complete", "vasp_launches": 1})
            for e in read_json(self.campaign / "submissions.json")["entries"]:
                if e.get("origin") == "refill" and e["job_id"] not in listing:
                    listing[e["job_id"]] = ["PENDING", states[e["state_id"]]]
            listing[job][0] = "COMPLETING"  # its run-state has exited; the epilog is still running
        replacements = [
            e["state_id"]
            for e in read_json(self.campaign / "submissions.json")["entries"]
            if e.get("origin") == "refill"
        ]
        self.assertEqual(replacements, ["s_d", "s_e", "s_f"])

    def test_a_requeued_job_counts_again_and_clears_its_end_mark_when_it_restarts(self):
        self.clone("s_b")
        write_json(
            self.campaign / "submissions.json",
            {
                "schema_version": 1,
                "entries": [{"state_id": "s_a", "through": "loose", "status": "submitted", "job_id": "4001"}],
            },
        )
        enable_refill(self.campaign, {"s_a": "loose", "s_b": "loose"}, 1)
        self.finish(
            self.state,
            "4001",
            {"status": "incomplete_time_limit", "vasp_launches": 1, "signals": [{"signal": "SIGTERM"}]},
        )
        self.assertTrue(read_json(self.campaign / "submissions.json")["entries"][0]["run_ended_at_utc"])
        (self.bin / "squeue_user.txt").write_text(f"4001|PENDING|{self.state}\n")  # requeued
        self.assertEqual(
            self.statuses(submit(self.campaign, through="loose", execute=True, max_active=1)),
            {"s_a": "skipped_active", "s_b": "deferred"},
        )
        scheduler.note_job_started(self.campaign, "4001")  # it runs again
        self.assertNotIn("run_ended_at_utc", read_json(self.campaign / "submissions.json")["entries"][0])
        (self.bin / "squeue_user.txt").write_text(f"4001|RUNNING|{self.state}\n")
        self.assertEqual(
            self.statuses(submit(self.campaign, through="loose", execute=True, max_active=1)),
            {"s_a": "skipped_active", "s_b": "deferred"},
        )

    def test_login_mode_records_job_ends_and_tick_submits(self):
        for name in ("s_b", "s_c"):
            self.clone(name)
        out = submit(self.campaign, through="loose", execute=True, max_active=1, refill=True)
        set_refill(self.campaign, submit_from="login")
        self.done(self.state)
        n = self.sbatch_count()
        event = self.finish(self.state, out["jobs"][0]["job_id"], {"status": "complete", "vasp_launches": 1})
        self.assertEqual((event["decision"], self.sbatch_count()), ("recorded", n))
        with contextlib.redirect_stdout(io.StringIO()) as text:
            # mechanet prints JSON only with --json
            self.assertEqual(main(["--json", "refill", str(self.campaign), "--tick"]), 0)
        self.assertIn('"tick"', text.getvalue())
        self.assertEqual(self.sbatch_count(), n + 1)
        set_refill(self.campaign, enabled=False)
        self.assertEqual(scheduler.refill_tick(self.campaign)["decision"], "paused")

    def test_the_breaker_counts_only_failures_without_progress(self):
        names = ["s_a", "s_b", "s_c", "s_d", "s_e"]
        states = {"s_a": self.state, **{n: self.clone(n) for n in names[1:]}}
        enable_refill(self.campaign, {n: "loose" for n in names}, 1)
        ledger = {
            "schema_version": 1,
            "entries": [
                {"state_id": n, "through": "loose", "status": "submitted", "job_id": str(4001 + k)}
                for k, n in enumerate(names)
            ],
        }
        write_json(self.campaign / "submissions.json", ledger)

        def failed(state, stage, attempt_record):
            (state / stage).mkdir(exist_ok=True)
            write_json(state / stage / "attempt.json", attempt_record)
            if stage != "01_loose":
                self.done(state)  # its 01_loose target is reached; only the failure matters
            return {"status": "failed", "failure_kind": "stage", "vasp_launches": 3, "failed_stage": stage}

        progress = [
            failed(states["s_a"], "05_freq", {"status": "failed_units", "completed_this_run": 11, "failures": ["x"]}),
            failed(
                states["s_b"],
                "02_final",
                {
                    "status": "failed",
                    "returncode": 0,
                    "result": {"completed": True, "last_electronic_converged": True, "ionic_converged": False},
                },
            ),
            failed(states["s_c"], "05_freq", {"status": "failed_units", "completed_this_run": 2, "failures": ["y"]}),
        ]
        for name, record in zip(names, progress):
            self.finish(states[name], ledger["entries"][names.index(name)]["job_id"], record)
        self.assertTrue(refill_status(self.campaign)["enabled"], "hard cases with progress do not count")
        refine = states["s_b"] / "03_refine"
        refine.mkdir(exist_ok=True)
        write_json(
            refine / "attempt.json",
            {
                "status": "failed",
                "returncode": 0,
                "result": {
                    "completed": True,
                    "last_electronic_converged": True,
                    "restart_evidence": {"fallback": True},
                },
            },
        )
        self.assertTrue(
            scheduler._failure_without_progress(states["s_b"], {"vasp_launches": 1, "failed_stage": "03_refine"}),
            "a restart file that was not read counts: it is often the same for every state",
        )
        crash = {"status": "failed", "returncode": 137, "result": {"completed": False}}
        for name in ("s_d", "s_e", "s_a"):
            record = failed(states[name], "01_loose", crash)
            self.finish(states[name], ledger["entries"][names.index(name)]["job_id"], record)
        self.assertIn("made no progress", refill_status(self.campaign)["disabled_reason"])

    def test_small_cases_signal_killed_sbatch_failed_listing_two_jobs_in_a_folder(self):
        b = self.clone("s_b")
        self.clone("s_c")
        (self.bin / "sbatch_selfkill").write_text("")
        out = submit(self.campaign, selection=["s_a"], through="loose", execute=True, max_active=3)
        self.assertEqual(self.statuses(out), {"s_a": "submission_unknown"})
        (self.bin / "squeue_user_fail").write_text("")
        with self.assertRaisesRegex(RuntimeError, "refusing possible duplicate submission"):
            submit(self.campaign, through="loose", execute=True, max_active=3)
        (self.bin / "squeue_user_fail").unlink()
        resolve_submission(self.campaign, "s_a", not_submitted=True)
        (self.bin / "squeue_user.txt").write_text(f"9001|RUNNING|{b}\n9002|PENDING|{b}\n")
        out = submit(self.campaign, through="loose", execute=True, max_active=3)
        self.assertEqual(self.statuses(out), {"s_a": "submitted", "s_b": "skipped_active", "s_c": "deferred"})
        self.assertEqual(out["active_jobs"], 3, "two jobs in one folder take two slots")

    def test_a_refill_waiting_for_the_lock_uses_a_changed_list_of_states_next_time(self):
        self.clone("s_b")
        out = submit(self.campaign, through="loose", execute=True, max_active=2, refill=True)
        real = scheduler._sbatch_env

        def changed_meanwhile():
            enable_refill(self.campaign, {"s_a": "freq"}, 2)  # s_a now goes through freq
            return real()

        self.done(self.state)
        with patch.object(scheduler, "_sbatch_env", changed_meanwhile):
            event = self.finish(self.state, out["jobs"][0]["job_id"], {"status": "complete", "vasp_launches": 1})
        self.assertIn("refill_changed", event)
        self.assertEqual(event["submitted"], [])

    def test_a_stall_is_lifted_by_a_changed_configuration_not_by_a_touched_file(self):
        self.clone("s_b")
        enable_refill(self.campaign, {"s_a": "loose", "s_b": "loose"}, 2)
        from mechanet.fileio import file_hash

        write_json(
            self.state / "run.json",
            {
                "status": "failed",
                "vasp_launches": 0,
                "reason": "synthetic",
                "config_sha256": file_hash(self.campaign / "config.json"),
            },
        )
        future = time.time() + 3600
        os.utime(self.campaign / "config.json", (future, future))  # copied with a future time stamp
        self.assertEqual(
            self.statuses(submit(self.campaign, through="loose", origin="refill"))["s_a"], "skipped_stalled"
        )
        self.set_config("frequency", max_force_eVA=0.05)
        self.assertEqual(self.statuses(submit(self.campaign, through="loose", origin="refill"))["s_a"], "dry_run")

    def test_a_single_point_cut_off_twice_is_left_to_you(self):
        enable_refill(self.campaign, {"s_a": "static"}, 1)
        for stage in ("01_loose", "02_final", "03_refine"):
            (self.state / stage).mkdir(exist_ok=True)
            write_json(self.state / stage / "attempt.json", {"status": "complete"})
        static = self.state / "04_static"
        (static / "previous_attempts" / "001").mkdir(parents=True)
        write_json(static / "previous_attempts" / "001" / "archive.json", {"kind": "interrupted"})
        write_json(static / "attempt.json", {"status": "interrupted"})
        out = submit(self.campaign, through="static", origin="refill")
        self.assertEqual(self.statuses(out), {"s_a": "skipped_stalled"})
        self.assertIn("cut off twice", out["jobs"][0]["reason"])
        self.assertEqual(self.statuses(submit(self.campaign, through="static")), {"s_a": "dry_run"})


class LedgerRepairTests(_Base):
    def test_unknown_outcome_is_resolved_explicitly(self):
        write_json(
            self.campaign / "submissions.json",
            {"schema_version": 1, "entries": [{"state_id": "s_a", "through": "final", "status": "submission_unknown"}]},
        )
        self.assertEqual(
            self.statuses(submit(self.campaign, through="final", execute=True)), {"s_a": "skipped_submission_unknown"}
        )
        self.assertEqual(
            self.statuses(submit(self.campaign, through="final")),
            {"s_a": "skipped_submission_unknown"},
            "a preview shows it too",
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            resolve_submission(self.campaign, "s_a")
        entry = resolve_submission(self.campaign, "s_a", job_id="4001")
        self.assertEqual((entry["status"], entry["job_id"]), ("submitted", "4001"))
        (self.bin / "squeue_user.txt").write_text("4001 PENDING\n")
        self.assertEqual(self.statuses(submit(self.campaign, through="final", execute=True)), {"s_a": "skipped_active"})
        with self.assertRaisesRegex(ValueError, "nothing to resolve"):
            resolve_submission(self.campaign, "s_a", not_submitted=True)
        # an sbatch reported as failed that SLURM queued after all can be recorded too
        ledger = read_json(self.campaign / "submissions.json")
        ledger["entries"].append({"state_id": "s_a", "through": "final", "status": "submission_failed"})
        write_json(self.campaign / "submissions.json", ledger)
        self.assertEqual(resolve_submission(self.campaign, "s_a", job_id="4002")["job_id"], "4002")


class HeldUnitTests(_Base):
    config_updates = {"frequency": {"max_unit_failures_per_job": 1}}
    harmonic = harmonic_model()

    def test_the_preview_counts_the_displacements_left(self):
        self.assertEqual(submit(self.campaign, through="freq")["displacements_remaining"], 12)  # 2 atoms x 3 x +-h
        self.assertNotIn("displacements_remaining", submit(self.campaign, through="static"))
        self.assertEqual(run_state(self.state, through="freq", execute=True)["status"], "complete")
        self.assertEqual(self.statuses(submit(self.campaign, through="freq")), {"s_a": "skipped_complete"})
        # A set in progress: count only the units that are not accepted yet.
        driver = read_json(self.state / STAGE_FREQ / "attempt.json")
        write_json(self.state / STAGE_FREQ / "attempt.json", dict(driver, status="incomplete"))
        for unit in sorted((self.state / STAGE_FREQ / "units").iterdir())[:3]:
            (unit / "attempt.json").unlink()
        self.assertEqual(submit(self.campaign, through="freq")["displacements_remaining"], 3)

    def test_the_refill_holds_a_displacement_set_that_only_fails(self):
        run_state(self.state, through="static", execute=True)
        marker(self.state, "nelm_u001_atom002_x_plus")
        with self.assertRaises(StageFailed):
            run_state(self.state, through="freq", execute=True)
        driver = read_json(self.state / STAGE_FREQ / "attempt.json")
        self.assertEqual((driver["status"], driver["completed_this_run"]), ("failed_units", 0))
        enable_refill(self.campaign, {"s_a": "freq"}, 1)
        self.assertEqual(
            self.statuses(submit(self.campaign, through="freq", origin="refill")), {"s_a": "skipped_stalled"}
        )
        self.assertEqual(self.statuses(submit(self.campaign, through="freq")), {"s_a": "dry_run"})

    def test_earlier_unit_failures_do_not_use_this_jobs_failure_allowance(self):
        run_state(self.state, through="static", execute=True)
        marker(self.state, "nelm_u001_atom002_x_plus")
        with self.assertRaisesRegex(StageFailed, "1 displacement unit"):
            run_state(self.state, through="freq", execute=True)
        first = [c for c in calls(self.state) if c.startswith("05_freq/")]
        self.assertEqual(len(first), 1, "the allowance of one failure per job stops the first job")
        with self.assertRaisesRegex(StageFailed, "wait for --retry"):
            run_state(self.state, through="freq", execute=True)
        second = [c for c in calls(self.state) if c.startswith("05_freq/")][len(first) :]
        self.assertEqual(len(second), 11, "the held unit is skipped; every other unit runs")
        check = submission_check(self.state, "freq")
        self.assertEqual((check["action"], check["status"]), ("skip", "skipped_needs_retry"))
        self.assertEqual(submission_check(self.state, "freq", retry=True)["action"], "submit")
        (self.state / "ctl" / "nelm_u001_atom002_x_plus").unlink()
        self.assertEqual(run_state(self.state, through="freq", execute=True, retry=True)["status"], "complete")
        self.assertEqual(read_json(self.state / STAGE_FREQ / "attempt.json")["status"], "complete")


if __name__ == "__main__":
    unittest.main()
