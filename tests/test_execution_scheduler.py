"""Scheduler ledger, SLURM job-state handling, CLI options and network-twin validation."""

import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from support import make_campaign  # noqa: E402
from mechanet import cli  # noqa: E402
from mechanet.campaign import selection  # noqa: E402
from mechanet.fileio import read_json, write_json  # noqa: E402
from mechanet.runner import job_end_time, run_state  # noqa: E402
from mechanet.slurm import ACTIVE_JOB_STATES, job_is_active, submit  # noqa: E402
from mechanet.structure import Structure, write_poscar  # noqa: E402
from mechanet.validation import validate_campaign  # noqa: E402

FAKE_SQUEUE = r"""#!/bin/sh
here="$(dirname "$0")"
echo "$@" >> "$here/squeue_calls.txt"
job=""; fmt=""; user=""
while [ $# -gt 0 ]; do
  case "$1" in
    --jobs|-j) job="$2"; shift ;;
    --jobs=*) job="${1#--jobs=}" ;;
    -o) fmt="$2"; shift ;;
    --format=*) fmt="${1#--format=}" ;;
    --user=*) user="${1#--user=}" ;;
  esac
  shift
done
if [ "$fmt" = "%e" ]; then echo "2031-05-06T07:08:09"; exit 0; fi
if [ -n "$user" ]; then
  if [ -e "$here/squeue_user_fail" ]; then echo "squeue: error: Unable to contact slurm controller" >&2; exit 1; fi
  if [ -e "$here/squeue_user.txt" ]; then cat "$here/squeue_user.txt"; exit 0; fi
  printf '222 COMPLETED\n223 TIMEOUT\n333 RUNNING\n334 PENDING\n'
  exit 0
fi
case "$job" in
  111) echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1 ;;
  222) echo "COMPLETED" ;;
  223) echo "TIMEOUT" ;;
  333) echo "RUNNING" ;;
  334) echo "PENDING" ;;
  444) echo "slurm_load_jobs error: Socket timed out on send/recv operation" >&2; exit 1 ;;
esac
exit 0
"""

FAKE_SBATCH = """#!/bin/sh
echo "$@" >> "$(dirname "$0")/sbatch_calls.txt"
echo "4242;cluster"
"""


class SchedulerTests(unittest.TestCase):
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
        self.campaign, self.state, _ = make_campaign(self.root)

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def ledger(self, job_id, status="submitted"):
        write_json(
            self.campaign / "submissions.json",
            {
                "schema_version": 1,
                "entries": [{"state_id": self.state.name, "through": "final", "status": status, "job_id": job_id}],
            },
        )

    def squeue_calls(self):
        path = self.bin / "squeue_calls.txt"
        return path.read_text().splitlines() if path.exists() else []

    def test_job_states(self):
        self.assertFalse(job_is_active("111"))  # purged: "Invalid job id specified"
        self.assertFalse(job_is_active("222"))  # finished, still listed for MinJobAge
        self.assertFalse(job_is_active("223"))
        self.assertTrue(job_is_active("333"))
        self.assertTrue(job_is_active("334"))
        self.assertFalse(job_is_active("999"))  # rc 0 and no output
        with self.assertRaisesRegex(RuntimeError, "Cannot verify SLURM job 444"):
            job_is_active("444")
        self.assertTrue({"PENDING", "RUNNING", "COMPLETING", "REQUEUED", "SUSPENDED"} <= ACTIVE_JOB_STATES)
        self.assertNotIn("COMPLETED", ACTIVE_JOB_STATES)

    def test_purged_and_recently_finished_jobs_are_resubmitted(self):
        for job in ("111", "222"):
            self.ledger(job)
            out = submit(self.campaign, through="final", execute=True)
            self.assertEqual((out["jobs"][0]["status"], out["jobs"][0]["job_id"]), ("submitted", "4242"))
        entries = read_json(self.campaign / "submissions.json")["entries"]
        self.assertEqual(entries[-1]["status"], "submitted")
        self.assertIn("run_job.sh", (self.bin / "sbatch_calls.txt").read_text())

    def test_active_job_is_skipped_and_unknown_squeue_failure_refuses(self):
        self.ledger("333")
        self.assertEqual(submit(self.campaign, through="final", execute=True)["jobs"][0]["status"], "skipped_active")
        self.assertTrue(any("--user=" in c for c in self.squeue_calls()), "one squeue call for all jobs")
        self.ledger("444")
        (self.bin / "squeue_user_fail").write_text("")  # batch query fails: fall back to per-job queries
        with self.assertRaisesRegex(RuntimeError, "refusing possible duplicate submission"):
            submit(self.campaign, through="final", execute=True)
        (self.bin / "squeue_user_fail").unlink()
        self.ledger(None, status="submitting")
        out = submit(self.campaign, through="final", execute=True)
        self.assertEqual(out["jobs"][0]["status"], "skipped_submission_unknown")
        self.assertEqual([e["state_id"] for e in out["attention"]], [self.state.name])
        self.assertFalse((self.bin / "sbatch_calls.txt").exists())

    def test_completed_target_stage_is_skipped_before_any_slurm_query(self):
        run_state(self.state, through="loose", execute=True)
        self.ledger("444")  # squeue would fail for this job
        before = len(self.squeue_calls())
        out = submit(self.campaign, through="loose", execute=True)
        self.assertEqual(out["jobs"][0]["status"], "skipped_complete")
        self.assertEqual(len(self.squeue_calls()), before)
        self.assertFalse((self.bin / "sbatch_calls.txt").exists())

    def test_resubmitting_after_a_time_limited_job_continues_the_state(self):
        os.environ["SLURM_JOB_START_TIME"] = str(time.time() - 86390)  # a 24 h job with 10 s left
        os.environ["SLURM_JOB_END_TIME"] = str(time.time() + 10)
        self.assertEqual(run_state(self.state, through="final", execute=True)["status"], "incomplete_time_limit")
        del os.environ["SLURM_JOB_END_TIME"], os.environ["SLURM_JOB_START_TIME"]
        self.ledger("222")
        out = submit(self.campaign, through="final", execute=True)
        self.assertEqual(out["jobs"][0]["status"], "submitted")
        script = (self.state / "run_job.sh").read_text()
        self.assertIn("--through final --execute", script)
        self.assertNotIn("--retry", script)

    def test_allocation_end_time_sources(self):
        self.assertIsNone(job_end_time({}))
        self.assertEqual(job_end_time({"SLURM_JOB_END_TIME": "1999999999", "SLURM_JOB_ID": "555"}), 1999999999.0)
        expected = time.mktime(time.strptime("2031-05-06T07:08:09", "%Y-%m-%dT%H:%M:%S"))
        self.assertEqual(job_end_time({"SLURM_JOB_ID": "555"}), expected)
        self.assertIn("-h -j 555 -o %e", self.squeue_calls()[-1])


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = patch.dict(
            os.environ, {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}, clear=True
        )
        self.env.start()
        self.campaign, self.state, _ = make_campaign(self.root)
        # mechanet's CLI recognizes a campaign by config.json and network.json; give the synthetic one a network.
        write_json(
            self.campaign / "network.json",
            {
                "reaction": "oer",
                "reference": "test",
                "nodes": [{"id": "test", "delta_atoms": {}, "state_ids": [self.state.name]}],
                "edges": [],
                "cycles": [],
            },
        )

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def main(self, *args):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["--json"] + [str(a) for a in args])  # mechanet prints JSON only with --json
        return code, buffer.getvalue()

    def test_through_choices_and_dry_run(self):
        for name, stage in (("loose", "01_loose"), ("refine", "03_refine"), ("freq", "05_freq")):
            code, out = self.main("run-state", self.state, "--through", name)
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out)["stages"][-1], stage)
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            self.main("run-state", self.state, "--through", "everything")

    def test_selected_restricts_and_intersects(self):
        # mechanet: the selection helper is mechanet.campaign.selection and the hint names `mechanet select`.
        with self.assertRaisesRegex(FileNotFoundError, "mechanet select"):
            selection(self.campaign, selected=True)
        write_json(
            self.campaign / "thermo_selection.json", {"schema_version": 1, "state_ids": [self.state.name, "other"]}
        )
        self.assertEqual(selection(self.campaign, selected=True), [self.state.name, "other"])
        self.assertEqual(selection(self.campaign, states=f"{self.state.name},zzz", selected=True), [self.state.name])
        with self.assertRaisesRegex(ValueError, "no states"):
            selection(self.campaign, states="zzz", selected=True)
        write_json(self.campaign / "thermo_selection.json", {"schema_version": 1, "state_ids": [self.state.name]})
        code, out = self.main("prepare", self.campaign, "--selected")
        self.assertEqual((code, json.loads(out)["count"]), (0, 1))
        code, out = self.main("cleanup", self.campaign, "--selected")
        self.assertEqual(code, 0)
        self.assertIn("Dry run", json.loads(out)["note"])

    # mechanet: the CLI resolves the campaign to a Path and its summary reads the counts that the real
    # select_for_thermo / analyze_thermodynamics always return, so the stand-ins carry those keys too.
    SELECT_RESULT = {
        "schema_version": 1,
        "state_ids": ["x"],
        "n_selected": 1,
        "waiting_nodes": [],
        "unresolved_nodes": [],
    }
    THERMO_RESULT = {
        "analysis_level": "x",
        "energy_source": "final",
        "frequency_central": "final",
        "scale": "RHE",
        "conditions": {"temperature_K": 298.15, "pH": 0.0, "activities": {}},
        "reservoirs": {"valid": True, "errors": []},
        "nodes": [],
        "cycles": [],
        "best_cycles": {},
    }

    def test_select_and_energy_source_are_passed_to_the_science_layer(self):
        with patch("mechanet.analysis.select_for_thermo", create=True, return_value=dict(self.SELECT_RESULT)) as select:
            code, out = self.main("select", self.campaign, "--window", "0.2", "--max-per-node", "3")
        self.assertEqual(code, 0)
        select.assert_called_once_with(
            Path(self.campaign), energy_window_eV=0.2, max_variants_per_node=3, include_partial=False
        )
        with patch("mechanet.analysis.select_for_thermo", create=True, return_value=dict(self.SELECT_RESULT)) as select:
            self.assertEqual(self.main("select", self.campaign, "--include-partial")[0], 0)
        self.assertTrue(select.call_args.kwargs["include_partial"])
        self.assertEqual(json.loads(out)["state_ids"], ["x"])
        with patch("mechanet.analysis.analyze_thermodynamics", return_value=dict(self.THERMO_RESULT)) as thermo:
            code, out = self.main("thermo", self.campaign, "--references", "r.json", "--energy-source", "final")
            self.assertEqual(code, 0)
            self.assertEqual(thermo.call_args.kwargs["energy_source"], "final")
            self.main("thermo", self.campaign, "--references", "r.json")
            # mechanet passes energy_source=None (the protocol default) instead of omitting the keyword.
            self.assertIsNone(thermo.call_args.kwargs.get("energy_source"))

    def test_thermo_summary_reports_energy_source_and_frequency_central(self):
        with patch("mechanet.analysis.analyze_thermodynamics", return_value=dict(self.THERMO_RESULT)):
            code, out = self.main("thermo", self.campaign, "--references", "r.json", "--energy-source", "final")
        self.assertEqual(code, 0)
        self.assertEqual((json.loads(out)["energy_source"], json.loads(out)["frequency_central"]), ("final", "final"))


class ValidationTwinTests(unittest.TestCase):
    def test_computed_by_twins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "inputs").mkdir()
            atoms = Structure(
                ["Fe", "O", "H"],
                np.array([[0, 0, 0], [1.8, 0, 0], [2.5, 0.7, 0]]),
                np.eye(3) * 10,
                np.array([[False] * 3, [True] * 3, [True] * 3]),
                ["a0001", "a0002", "a0003"],
            )
            write_poscar(root / "inputs" / "CONTCAR_seed", atoms)
            # mechanet campaigns always carry config.json; validation reads the seed path from it.
            write_json(root / "config.json", {"reaction": "oer", "inputs": {"seed": "inputs/CONTCAR_seed"}})
            state = root / "states" / "ref"
            state.mkdir(parents=True)
            write_poscar(state / "POSCAR", atoms)
            write_json(
                state / "state.json",
                {
                    "id": "ref",
                    "node_id": "ref",
                    "symbols": atoms.symbols,
                    "atom_ids": atoms.ids,
                    "relax_mobile": atoms.mobile.tolist(),
                    "frequency_mobile": atoms.mobile.tolist(),
                    "delta_atoms": {"O": 0, "H": 0},
                },
            )
            nodes = [
                {"id": "ref", "delta_atoms": {"O": 0, "H": 0}, "state_ids": ["ref"]},
                {"id": "twin", "delta_atoms": {"O": 0, "H": 0}, "state_ids": [], "computed_by": "ref"},
            ]
            write_json(root / "network.json", {"nodes": nodes, "edges": [], "cycles": []})
            report = validate_campaign(root)
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["warnings"], [])
            nodes += [
                {"id": "bad", "delta_atoms": {"O": 1, "H": 0}, "state_ids": [], "computed_by": "ref"},
                {"id": "lost", "delta_atoms": {"O": 0, "H": 0}, "state_ids": [], "computed_by": "missing"},
                {"id": "empty", "delta_atoms": {"O": 0, "H": 0}, "state_ids": []},
            ]
            write_json(root / "network.json", {"nodes": nodes, "edges": [], "cycles": []})
            report = validate_campaign(root)
            self.assertFalse(report["valid"])
            # mechanet words it as a composition (every element, not only O and H).
            self.assertTrue(any("bad: computed_by ref has a different composition" in e for e in report["errors"]))
            self.assertTrue(any("lost: computed_by points to missing node" in e for e in report["errors"]))
            self.assertEqual(report["warnings"], ["empty: no realizable geometry (all proposed placements rejected)"])


if __name__ == "__main__":
    unittest.main()
