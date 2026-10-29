"""SLURM backend: job scripts from a cluster profile, a submission ledger, a job cap and refill.

Nothing is submitted unless ``execute`` is set. One SLURM job runs one state through its
stages; a job that reaches its walltime stops cleanly (SIGUSR1 -> STOPCAR) and the same
``mechanet submit`` command continues it. Job scripts are built from the campaign's
``scheduler`` section (a resolved cluster profile, see ``mechanet.clusters``), so the same
campaign runs on any SLURM cluster once its profile is set.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from .clusters import RESERVED_SBATCH, command_problems, timing_report
from .clusters import stop_lead as _stop_lead
from .fileio import atomic_text, file_hash, read_json, write_json
from .vasp import (
    RELAXATION_STAGES,
    STAGE_FREQ,
    STAGE_STATIC,
    THROUGH_NAMES,
    central_mode,
    find_campaign,
    protocol_stages,
    resolve_through,
)

# squeue %T states of a job that still exists and may run (or is running) VASP.
ACTIVE_JOB_STATES = frozenset(
    {
        "PENDING",
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "SUSPENDED",
        "REQUEUED",
        "REQUEUE_HOLD",
        "REQUEUE_FED",
        "RESIZING",
        "SIGNALING",
        "STAGE_OUT",
        "STOPPED",
        "SPECIAL_EXIT",
        "RESV_DEL_HOLD",
    }
)
# Final states. Any other state squeue reports counts as active (fail safe).
FINAL_JOB_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
)
SQUEUE_TIMEOUT_S = 120
SBATCH_TIMEOUT_S = 180
# sbatch messages after which the job may exist nevertheless (the request reached the controller).
AMBIGUOUS_SBATCH_ERRORS = ("timed out", "zero bytes")
UNKNOWN_SUBMISSION = ("submitting", "submission_unknown")
# States in which a job whose run-state has already ended was requeued (it will run again).
REQUEUE_STATES = frozenset({"PENDING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "SPECIAL_EXIT"})
JOB_PREFIX = "mn_"
# Short keys of the scheduler section and the sbatch option each sets (``sbatch:`` entries win).
SCHEDULER_SHORT_KEYS = {
    "nodes": "nodes",
    "ntasks": "ntasks",
    "walltime": "time",
    "partition": "partition",
    "account": "account",
    "qos": "qos",
    "constraint": "constraint",
    "mem": "mem",
}


def _directive(value):
    value = str(value)
    if "\n" in value or "\r" in value:
        raise ValueError("Newlines are not allowed in SLURM settings")
    return shlex.quote(value)


NO_PROFILE = (
    "This campaign has no cluster profile (scheduler.sbatch in config.json): import a working VASP job "
    "script with `mechanet cluster import JOB_SCRIPT --name NAME`, then apply it to this campaign with "
    "`mechanet cluster use NAME` (and set `cluster: NAME` in project.yaml for future campaigns)"
)
DEFAULT_ENVIRONMENT = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def has_profile(scheduler):
    return bool((scheduler.get("sbatch") or {}).get("time") or scheduler.get("walltime"))


def sbatch_options(scheduler):
    """Ordered sbatch long options of a cluster profile (its ``sbatch`` mapping and the short keys)."""
    if not has_profile(scheduler) and not scheduler.get("sbatch"):
        raise ValueError(NO_PROFILE)
    options = {}
    for key, name in SCHEDULER_SHORT_KEYS.items():
        if scheduler.get(key) not in (None, ""):
            options[name] = scheduler[key]
    for key, value in (scheduler.get("sbatch") or {}).items():
        name = str(key).lstrip("-")
        if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
            raise ValueError(f"Invalid sbatch option name {key!r} in the cluster profile")
        if name in RESERVED_SBATCH:
            raise ValueError(f"The cluster profile may not set --{name}; mechanet sets it for every job")
        options[name] = value
    if "time" not in options:
        raise ValueError("The cluster profile needs a walltime (sbatch: {time: '24:00:00'})")
    return options


def setup_lines(scheduler):
    """Shell lines run before the job starts (module loads, environment activation, exports)."""
    lines = []
    module_use = scheduler.get("module_use") or []
    for path in [module_use] if isinstance(module_use, str) else module_use:
        lines.append("module use " + shlex.quote(str(path)))
    modules = scheduler.get("modules") or []
    if isinstance(modules, str):
        raise ValueError("scheduler.modules must be a list")
    if modules:
        lines.append("module load " + " ".join(shlex.quote(str(m)) for m in modules))
    setup = scheduler.get("setup") or []
    if isinstance(setup, str):
        setup = setup.splitlines()
    lines.extend(str(line) for line in setup if str(line).strip())
    return lines


def _env_value(value):
    """A double-quoted shell word: $VARIABLES expand, command substitution does not."""
    text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$(", "\\$(")
    return f'"{text}"'


def _set_by_setup(name, setup):
    return any(re.search(rf"(^|[\s;])(export\s+)?{re.escape(name)}=", line) for line in setup)


def job_script_text(scheduler, state_dir, through="final", retry=False, stages_comment=None):
    """The text of a job script that runs one state with this ``scheduler`` section."""
    errors, _ = timing_report(scheduler)
    if errors:
        raise ValueError("Cluster profile: " + "; ".join(errors))
    options = sbatch_options(scheduler)
    lead = _stop_lead(scheduler)
    if not 0 < lead <= 65535:
        raise ValueError("scheduler.stop_signal_lead_s must be between 1 and 65535 seconds")
    state_dir = Path(state_dir)
    job_name = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{JOB_PREFIX}{state_dir.name}")[:100]
    lines = [f"#!{scheduler.get('shell') or '/bin/bash'}", f"#SBATCH --job-name={job_name}"]
    lines += [
        f"#SBATCH --{name}" if value is True else f"#SBATCH --{name}={_directive(value)}"
        for name, value in options.items()
        if value is not False and value is not None
    ]
    # SIGUSR1 to the batch shell (the exec'd Python runner) before the time limit:
    # the runner writes STOPCAR so VASP ends its current ionic step cleanly.
    lines += [f"#SBATCH --signal=B:USR1@{lead}"]
    if "mail-type" not in options:
        lines.append("#SBATCH --mail-type=NONE")
    lines += [
        f"#SBATCH --chdir={_directive(state_dir)}",
        "#SBATCH --output=slurm-%j.out",
        "#SBATCH --error=slurm-%j.err",
        "",
        "# Generated by mechanet; regenerate with `mechanet submit` after changing the cluster profile.",
    ]
    if stages_comment:
        lines.append(stages_comment)
    lines.append("# Re-running the same submit command continues an interrupted or time-limited state.")
    setup = setup_lines(scheduler)
    environment = scheduler.get("environment") or {}
    # Pure-MPI defaults first, so that the profile's own setup lines and environment win.
    lines += [
        f"export {key}={_env_value(value)}"
        for key, value in DEFAULT_ENVIRONMENT.items()
        if key not in environment and not _set_by_setup(key, setup)
    ]
    if setup:
        lines.append("# Cluster profile set-up, run as in your own job script (a failing line does not stop the job).")
    lines += setup
    lines += [f"export {key}={_env_value(value)}" for key, value in environment.items()]
    python = scheduler.get("python") or sys.executable
    lines += [
        "set -euo pipefail",
        "cd -- " + shlex.quote(str(state_dir)),
        "# The Python with mechanet (cluster profile `python:`, else the one that ran `mechanet submit`);",
        "# MECHANET_PYTHON overrides it at run time.",
        "MECHANET_PY=" + shlex.quote(str(python)),
        'if [[ -n "${MECHANET_PYTHON:-}" ]]; then MECHANET_PY="$MECHANET_PYTHON"; fi',
        "# exec: the runner itself receives the batch-shell signal (--signal=B:USR1).",
    ]
    args = ["-m", "mechanet", "run-state", str(state_dir), "--through", through, "--execute"]
    if retry:
        args.append("--retry")
    lines.append('exec "$MECHANET_PY" ' + " ".join(shlex.quote(a) for a in args))
    return "\n".join(lines) + "\n"


def write_job_script(state_dir, through="final", retry=False):
    state_dir = Path(state_dir).resolve()
    campaign = find_campaign(state_dir)
    config = read_json(campaign / "config.json")
    if through not in THROUGH_NAMES:
        raise ValueError(f"through must be one of {', '.join(THROUGH_NAMES)}")
    resolve_through(through, config)
    comment = f"# Stages: {', '.join(protocol_stages(config))} (frequency.central = {central_mode(config)})."
    text = job_script_text(config.get("scheduler", {}), state_dir, through, retry, stages_comment=comment)
    path = state_dir / "run_job.sh"
    atomic_text(path, text)
    path.chmod(0o755)
    return path


def _squeue(args):
    """squeue without the user's SQUEUE_* environment settings (state, partition or account filters)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SQUEUE_")}
    return subprocess.run(
        ["squeue", *args], capture_output=True, text=True, check=False, timeout=SQUEUE_TIMEOUT_S, env=env
    )


def _state_word(text):
    words = str(text).split()
    return words[0].upper().rstrip("+") if words else ""


def _job_states(job_id):
    """The states squeue reports for one job; empty when SLURM no longer knows it.

    Any squeue failure other than "Invalid job id specified" raises, because a
    duplicate VASP run would corrupt outputs.
    """
    try:
        proc = _squeue(["--noheader", "--all", "--states=all", "--jobs", str(job_id), "--format=%T"])
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Cannot verify SLURM job {job_id}; refusing possible duplicate submission: {exc}") from exc
    if proc.returncode != 0:
        if "invalid job id" in (proc.stderr + proc.stdout).lower():
            return set()
        raise RuntimeError(
            f"Cannot verify SLURM job {job_id}; refusing possible duplicate submission: {proc.stderr.strip()}"
        )
    return {_state_word(line) for line in proc.stdout.splitlines() if line.strip()}


def job_is_active(job_id):
    """True while SLURM still holds the job in a state that is not final.

    A job purged from the controller ("Invalid job id specified") or in a final state
    (COMPLETED, FAILED, TIMEOUT, CANCELLED, ...) is not active; every other state
    counts as active.
    """
    return any(state not in FINAL_JOB_STATES for state in _job_states(job_id))


REFILL_FILE = "refill.json"
# Skips that need a decision by the user; everything else resolves itself.
ATTENTION_STATUSES = (
    "skipped_needs_retry",
    "skipped_exhausted",
    "skipped_blocked",
    "skipped_submission_unknown",
    "skipped_stalled",
)
MAX_REFILL_EVENTS = 50
REFILL_MAX_CONSECUTIVE_FAILURES = 3  # default of scheduler.refill_max_consecutive_failures
REFILL_MAX_SBATCH_ERRORS = 3


def _utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _current_user():
    try:
        import getpass

        return getpass.getuser()
    except Exception:  # noqa: BLE001 - any lookup failure means "unknown"
        return os.environ.get("USER") or os.environ.get("LOGNAME") or ""


def _check_through(value):
    if value not in THROUGH_NAMES:
        raise ValueError(f"through must be one of {', '.join(THROUGH_NAMES)}")
    return value


def _depth(through):
    return list(THROUGH_NAMES).index(through)


def _user_jobs():
    """{job_id: (state, working directory)} for every job of this user that SLURM still knows.

    --all includes hidden partitions, --states=all includes finished jobs, and the
    user's SQUEUE_* settings are ignored. None when the listing is not available.
    """
    user = _current_user()
    if not user:
        return None
    try:
        proc = _squeue(["--noheader", "--all", "--states=all", f"--user={user}", "--format=%i|%T|%Z"])
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    jobs = {}
    for line in proc.stdout.splitlines():
        parts = line.split("|", 2) if "|" in line else line.split(None, 2)
        if len(parts) >= 2 and parts[0].strip():
            jobs[parts[0].strip()] = (_state_word(parts[1]), parts[2].strip() if len(parts) > 2 else "")
    return jobs


def active_job_ids(job_ids):
    """The subset of ``job_ids`` that SLURM still holds in a state that is not final.

    One user listing covers all of them; a job missing from it, or every job when the
    listing fails, is asked for on its own (job_is_active refuses when SLURM cannot answer).
    """
    wanted = {str(j) for j in job_ids if j}
    if not wanted:
        return set()
    listing = _user_jobs()
    if listing is None:
        return {job for job in sorted(wanted) if job_is_active(job)}
    active = {job for job in wanted if job in listing and listing[job][0] not in FINAL_JOB_STATES}
    return active | {job for job in sorted(wanted - set(listing)) if job_is_active(job)}


def _slurm_view(campaign, entries, latest, excluded):
    """The campaign as SLURM sees it: active and finished ledger jobs, and active jobs per state folder.

    A ledger job missing from the user listing is asked for on its own. A job running
    in a state folder counts even when the ledger does not know it (an sbatch that
    timed out after the job was queued, or a job submitted by hand). A job whose
    run-state has recorded its end no longer counts while SLURM tidies up (epilog,
    COMPLETING), unless SLURM shows it requeued. Raises when SLURM cannot be asked.
    """
    known = {
        str(e["job_id"])
        for e in latest.values()
        if e.get("job_id") and e.get("status") == "submitted" and not e.get("job_finished_at_utc")
    }
    known -= excluded
    ended = {str(e["job_id"]) for e in entries if e.get("job_id") and e.get("run_ended_at_utc")}
    states_root = os.path.realpath(Path(campaign) / "states")
    listing = _user_jobs()
    if listing is None:
        raise RuntimeError(
            "Cannot list your SLURM jobs (squeue --user failed or is missing); refusing possible duplicate submission"
        )
    active, finished, folders = set(), set(), {}
    for job, (state, workdir) in listing.items():
        if job in excluded:
            continue
        if state in FINAL_JOB_STATES:
            if job in known:
                finished.add(job)
            continue
        if job in ended and state not in REQUEUE_STATES:
            continue  # its run-state has ended; SLURM is only tidying up
        if job in known:
            active.add(job)
        if workdir:
            path = os.path.realpath(workdir)
            if os.path.dirname(path) == states_root:
                folders.setdefault(os.path.basename(path), set()).add(job)
    for job in sorted(known - set(listing)):
        live = {state for state in _job_states(job) if state not in FINAL_JOB_STATES}
        if not live:
            finished.add(job)
        elif job not in ended or live & REQUEUE_STATES:
            active.add(job)
    return {"active": active, "finished": finished, "folders": folders}


def _latest_entries(ledger):
    latest = {}
    for entry in ledger.get("entries", []):
        latest[entry["state_id"]] = entry
    return latest


def _sbatch_env():
    """Environment for sbatch run from inside a job: no SLURM_* variables of the parent job."""
    return {k: v for k, v in os.environ.items() if not k.startswith("SLURM_") or k == "SLURM_CONF"}


def _stall_reason(state_dir, campaign, stage):
    """Why the refill leaves this state to you, or None.

    Its last job made no progress (it stopped before starting VASP, or every
    displacement it tried failed), or the stage it would run next is a single point
    that was cut off twice without finishing (it does not fit into one job). A change
    of <campaign>/config.json after that job lifts the hold for one more job.
    """
    from .runner import _archive_counts, _read_attempt, _unit_ids

    state_dir, campaign = Path(state_dir), Path(campaign)
    run = {}
    with contextlib.suppress(OSError, ValueError):
        run = read_json(state_dir / "run.json")
    config = campaign / "config.json"
    with contextlib.suppress(OSError):
        if run.get("config_sha256"):
            if file_hash(config) != run["config_sha256"]:
                return None
        elif (state_dir / "run.json").is_file() and config.stat().st_mtime > (state_dir / "run.json").stat().st_mtime:
            return None
    if run.get("status") in ("failed", "incomplete_time_limit") and run.get("vasp_launches") == 0:
        return f"its last job stopped before starting VASP ({str(run.get('reason'))[:200]})"
    folder = state_dir / stage

    def cut_off(path):
        return _archive_counts(path)["interrupted"] + int(
            _read_attempt(path).get("status") in ("interrupted", "running")
        )

    if stage == STAGE_FREQ:
        driver = _read_attempt(folder)
        if driver.get("status") == "failed_units" and driver.get("completed_this_run") == 0:
            return (
                f"every displacement its last job tried failed ({len(driver.get('failures') or [])} new failures, "
                "none completed)"
            )
        units = _unit_ids(folder)
        if units:
            twice = [u for u in units if cut_off(folder / "units" / u) >= 2]
            if twice:
                return f"displacement {twice[0]} was cut off twice without finishing"
            return None
    if stage == STAGE_STATIC or stage == STAGE_FREQ:
        if cut_off(folder) >= 2:
            return f"the single point {stage} was cut off twice without finishing; it must fit into one job"
    return None


def _last_run_warning(state_dir):
    """A warning when the state's last job ended at its time limit without starting VASP."""
    try:
        run = read_json(Path(state_dir) / "run.json")
    except (OSError, ValueError):
        return None
    if run.get("status") == "incomplete_time_limit" and run.get("vasp_launches") == 0:
        return (
            f"its last job ended without starting VASP ({str(run.get('reason'))[:160]}); check the walltime, "
            "stop_signal_lead_s and min_remaining_s of the cluster profile (`mechanet cluster show NAME`)"
        )
    return None


def _resolve_hint(campaign, sid):
    return (
        f"`mechanet resolve-submission {shlex.quote(sid)} {shlex.quote(str(campaign))} --job-id ID` if SLURM "
        "shows the job, else the same command with --not-submitted"
    )


def _failure_without_progress(state_dir, record):
    """Does a failed run look like a broken setup (a crash, a fatal error, nothing completed)?

    Runs in which VASP ended normally at its NSW limit, and displacement sets that
    completed units in this job, are hard cases rather than a broken setup.
    """
    from .runner import _read_attempt

    if int(record.get("vasp_launches") or 0) == 0:
        return False  # nothing ran: the refill's stall check handles such a state
    stage = str(record.get("failed_stage"))
    attempt = _read_attempt(Path(state_dir) / stage)
    if stage == STAGE_FREQ and "completed_this_run" in attempt:
        return int(attempt.get("completed_this_run") or 0) == 0
    result = attempt.get("result") or {}
    if (result.get("restart_evidence") or attempt.get("restart_evidence") or {}).get("fallback"):
        return True  # a restart file that was not read: often the same for every state
    if (
        stage in RELAXATION_STAGES
        and attempt.get("returncode") in (0, None)
        and result.get("completed")
        and result.get("last_electronic_converged")
        and not result.get("fatal_errors")
    ):
        return False  # VASP ran to its ionic step limit (or missed the force criterion): a hard case
    return True


def submit(
    campaign,
    selection=None,
    through="final",
    execute=False,
    limit=None,
    retry=False,
    max_active=None,
    refill=False,
    exclude_job_ids=(),
    verify_complete=True,
    origin="user",
    env=None,
    stop_check=None,
):
    """Preview or submit one SLURM job per state, with duplicate protection and an optional job cap.

    Nothing is submitted without ``execute``. A state is submitted only if its job can
    make progress now: states whose requested stage is complete, whose job is still
    pending or running, whose last attempt failed (until ``retry``), whose attempt or
    continuation budget is used up, or that are blocked by the frequency force gate
    are skipped and reported. ``limit`` caps the submissions of this call;
    ``max_active`` (or ``scheduler.max_active_jobs``) caps this campaign's pending +
    running jobs. States furthest along the stage sequence go first. ``refill`` adds
    the selection to the campaign's refill, so that every job tops the campaign up to
    the cap when it ends. ``through`` is one final stage for every state, or a mapping
    state ID -> final stage (used by the refill). With ``origin="refill"`` the current
    refill settings are re-read under the lock (a pause wins), states whose last job
    made no progress are left to the user, and ``stop_check()`` (true after a SIGTERM)
    stops further submissions.
    """
    from .runner import (
        exclusive_lock,
        prepare,
        remaining_displacements,
        state_directories,
        submission_check,
        _verify_input_hashes,
        _verify_output_hashes,
    )

    campaign = Path(campaign).resolve()
    if isinstance(through, dict):
        targets = {str(k): _check_through(v) for k, v in through.items()}
        if selection is None:
            selection = sorted(targets)
    else:
        targets = None
        _check_through(through)
    if limit is not None and int(limit) < 1:
        raise ValueError("limit must be positive")
    config = read_json(campaign / "config.json")
    scheduler = config.get("scheduler") or {}
    if not has_profile(scheduler):
        raise ValueError(NO_PROFILE)
    timing_errors, timing_warnings = timing_report(scheduler)
    if timing_errors:
        raise ValueError("Cluster profile: " + "; ".join(timing_errors))
    command = (config.get("vasp") or {}).get("command")
    if not isinstance(command, list) or not command or not all(isinstance(v, str) for v in command):
        raise ValueError(
            f"vasp.command in {campaign / 'config.json'} must be a list of words, e.g. [srun, vasp_std] "
            "(the cluster profile's vasp_command, or vasp.command in project.yaml)"
        )
    problems = command_problems(command)
    if problems:
        raise ValueError("vasp.command: " + "; ".join(problems))
    for goal in set(targets.values()) if targets is not None else {through}:
        resolve_through(goal, config)  # e.g. no "refine" when frequency.central = "final"
    if max_active is None and config.get("scheduler", {}).get("max_active_jobs"):
        max_active = config["scheduler"]["max_active_jobs"]
    if max_active is not None and int(max_active) < 1:
        raise ValueError("max_active must be positive")
    if refill and max_active is None:
        raise ValueError("--refill needs a job cap: --max-active N or scheduler.max_active_jobs")
    paths = state_directories(campaign, selection)
    if targets is not None:
        missing = sorted(p.name for p in paths if p.name not in targets)
        if missing:
            raise ValueError(f"No final stage given for {missing}")

    def target(sid):
        return targets[sid] if targets is not None else through

    prepare(campaign, [p.name for p in paths])
    refill_mode = origin == "refill"
    excluded = {str(j) for j in exclude_job_ids if j}
    registry = campaign / "submissions.json"
    error = error_kind = stopped = None
    held = {}
    with exclusive_lock(campaign / ".submission.lock", "campaign submissions", blocking=True, timeout=300.0):
        ledger = read_json(registry) if registry.exists() else {"schema_version": 1, "entries": []}
        latest = _latest_entries(ledger)
        if refill_mode:
            # The settings may have changed while this job waited for the lock: a pause wins,
            # and a changed list of states waits for the next refill.
            current = {}
            with contextlib.suppress(OSError, ValueError):
                current = read_json(_refill_path(campaign))
            if not current.get("enabled"):
                held["refill_paused"] = current.get("disabled_reason") or "the refill is paused"
            elif targets is not None and current.get("targets") != targets:
                held["refill_changed"] = "the refill's states changed while this job waited; the next refill uses them"
            elif current.get("max_active"):
                max_active = int(current["max_active"])
        view, busy_jobs, busy_unknown = [], set(), set()

        def slurm():
            """SLURM is asked once per call, and only if needed; finished ledger jobs are remembered."""
            if not view:
                view.append(_slurm_view(campaign, ledger["entries"], latest, excluded))
                if view[0]["finished"]:
                    stamp = _utc_now()
                    for e in ledger["entries"]:
                        if str(e.get("job_id")) in view[0]["finished"] and not e.get("job_finished_at_utc"):
                            e["job_finished_at_utc"] = stamp
                    write_json(registry, ledger)
            return view[0]

        def occupied():
            """Jobs of this campaign that are, or may be, pending or running (an unknown outcome counts as one)."""
            v = slurm()
            jobs = set(v["active"]) | {job for ids in v["folders"].values() for job in ids} | busy_jobs
            unknown = {
                sid for sid, e in latest.items() if e.get("status") in UNKNOWN_SUBMISSION and sid not in v["folders"]
            }
            return len(jobs) + len(unknown | busy_unknown)

        jobs, candidates = [], []
        for state_dir in paths:
            sid = state_dir.name
            goal = target(sid)
            entry = {"state_id": sid, "through": goal}
            jobs.append(entry)
            last = latest.get(sid)
            attempt = state_dir / THROUGH_NAMES[goal] / "attempt.json"
            # Completion first: an accepted target stage needs no job and no SLURM query.
            if attempt.exists() and read_json(attempt).get("status") == "complete":
                if verify_complete:
                    _verify_input_hashes(attempt.parent)
                    _verify_output_hashes(attempt.parent)
                entry["status"] = "skipped_complete"
                continue
            if last and last.get("status") in UNKNOWN_SUBMISSION:
                entry.update(
                    status="skipped_submission_unknown",
                    reason=(
                        f"the outcome of an earlier sbatch is unknown ({last.get('status')}); check "
                        f"squeue/sacct for a job named {JOB_PREFIX}{sid}, then run {_resolve_hint(campaign, sid)}"
                    ),
                )
                continue
            if execute:
                v = slurm()
                if last and last.get("job_id") and str(last["job_id"]) in v["active"]:
                    entry.update(status="skipped_active", job_id=last["job_id"])
                    continue
                if sid in v["folders"]:
                    entry.update(status="skipped_active", job_id=sorted(v["folders"][sid])[0], found_in_folder=True)
                    continue
            check = submission_check(state_dir, goal, retry=retry, config=config)
            if check["action"] == "complete":
                entry["status"] = "skipped_complete"
                continue
            if check["action"] == "skip":
                entry.update(status=check["status"], stage=check["stage"], reason=check["reason"])
                continue
            if refill_mode:
                why = _stall_reason(state_dir, campaign, check["stage"])
                if why:
                    entry.update(
                        status="skipped_stalled",
                        stage=check["stage"],
                        reason=f"{why}; the refill leaves this state to you: inspect it, then submit it yourself",
                    )
                    continue
            entry["stage"] = check["stage"]
            warning = _last_run_warning(state_dir)
            if warning:
                entry["warning"] = warning
            if goal == "freq":
                count = remaining_displacements(state_dir, config)
                if count is not None:
                    entry["displacements_remaining"] = count
            if not execute and last:
                entry["previous_submission"] = last
            candidates.append((-check["rank"], len(candidates), state_dir, entry))
        candidates.sort(key=lambda item: item[:2])  # furthest along first, then the sorted order
        slots = len(candidates)
        if held:
            slots = 0
        elif max_active is not None and candidates and execute:
            slots = min(slots, max(0, int(max_active) - occupied()))
        elif max_active is not None:
            slots = min(slots, int(max_active))
        if limit is not None:
            slots = min(slots, int(limit))
        for rank, (_, _, state_dir, entry) in enumerate(candidates):
            if rank >= slots or error or stopped:
                entry["status"] = "deferred"
                continue
            if execute and stop_check is not None and stop_check():
                stopped = "SIGTERM received; no further jobs were submitted"
                entry["status"] = "deferred"
                continue
            script = write_job_script(state_dir, entry["through"], retry)
            command = ["sbatch", "--parsable", str(script)]
            entry["command"] = command
            if not execute:
                entry["status"] = "dry_run"
                continue
            sid = entry["state_id"]
            pending = {
                "state_id": sid,
                "through": entry["through"],
                "status": "submitting",
                "submitted_at_utc": _utc_now(),
                "retry": retry,
                "origin": origin,
            }
            ledger["entries"].append(pending)
            write_json(registry, ledger)

            def record(status, kind, message, pending=pending, entry=entry, **extra):
                pending.update(status=status, **extra)
                write_json(registry, ledger)
                entry.update(status=status, **extra)
                return message, kind

            try:
                proc = subprocess.run(
                    command, capture_output=True, text=True, check=False, env=env, timeout=SBATCH_TIMEOUT_S
                )
            except subprocess.TimeoutExpired:
                busy_unknown.add(sid)
                error, error_kind = record(
                    "submission_unknown",
                    "sbatch_unknown",
                    f"sbatch did not answer within {SBATCH_TIMEOUT_S} s for {sid}; the job may exist: check SLURM, "
                    f"then run {_resolve_hint(campaign, sid)}",
                    error=f"no answer within {SBATCH_TIMEOUT_S} s",
                )
                continue
            except OSError as exc:
                error, error_kind = record(
                    "submission_failed", "sbatch_unavailable", f"sbatch could not be run: {exc}", error=str(exc)
                )
                continue
            if proc.returncode != 0:
                message = proc.stderr.strip() or proc.stdout.strip()
                if proc.returncode < 0:
                    message = f"sbatch was ended by signal {-proc.returncode}" + (f" ({message})" if message else "")
                if proc.returncode < 0 or any(marker in message.lower() for marker in AMBIGUOUS_SBATCH_ERRORS):
                    busy_unknown.add(sid)
                    error, error_kind = record(
                        "submission_unknown",
                        "sbatch_unknown",
                        f"sbatch for {sid} failed in a way that may still have queued the job ({message}); "
                        f"check SLURM, then run {_resolve_hint(campaign, sid)}",
                        error=message,
                    )
                else:
                    error, error_kind = record(
                        "submission_failed", "sbatch_failed", f"sbatch failed for {sid}: {message}", error=message
                    )
                continue
            token = proc.stdout.strip().split(";", 1)[0]
            if not re.fullmatch(r"\d+", token):
                busy_unknown.add(sid)
                error, error_kind = record(
                    "submission_unknown",
                    "sbatch_unknown",
                    f"sbatch succeeded but returned an unrecognized job ID; check SLURM, then run "
                    f"{_resolve_hint(campaign, sid)}",
                    response=proc.stdout.strip(),
                )
                continue
            record("submitted", None, None, job_id=token)
            busy_jobs.add(token)
    summary = {}
    for entry in jobs:
        summary[entry["status"]] = summary.get(entry["status"], 0) + 1
    result = {
        "execute": execute,
        "jobs": jobs,
        "count": len(jobs),
        "summary": summary,
        "active_jobs": occupied() if view else None,
        "max_active": max_active,
        "attention": [e for e in jobs if e["status"] in ATTENTION_STATUSES],
        "warnings": timing_warnings + [f"{e['state_id']}: {e['warning']}" for e in jobs if e.get("warning")],
    }
    counted = [e for e in jobs if "displacements_remaining" in e]
    if counted:
        # The frequency work these jobs (and the deferred ones) still contain: one VASP single point each.
        result["displacements_remaining"] = sum(e["displacements_remaining"] for e in counted)
    if not execute and max_active is not None:
        result["note"] = "Preview: SLURM was not queried, so the cap assumes no active jobs."
    result.update(held)
    if stopped:
        result["stopped"] = stopped
    if error:
        result.update(error=error, error_kind=error_kind)
    if refill:
        if not execute:
            result["refill"] = "not enabled by a preview; repeat with --execute"
        elif error:
            result["refill"] = "not enabled because a submission failed; fix the cause and repeat this command"
        else:
            result["refill"] = enable_refill(campaign, {p.name: target(p.name) for p in paths}, max_active)
    return result


# ---------------------------------------------------------------------------
# Ledger repair
# ---------------------------------------------------------------------------


def resolve_submission(campaign, state_id, job_id=None, not_submitted=False):
    """Settle a submission whose outcome is unknown (or a failed one that SLURM queued), after checking SLURM."""
    from .runner import exclusive_lock

    if (job_id is None) == (not not_submitted):
        raise ValueError("Give exactly one of --job-id or --not-submitted")
    if job_id is not None and not re.fullmatch(r"\d+", str(job_id)):
        raise ValueError("A SLURM job ID consists of digits")
    campaign = Path(campaign).resolve()
    registry = campaign / "submissions.json"
    with exclusive_lock(campaign / ".submission.lock", "campaign submissions", blocking=True, timeout=300.0):
        ledger = read_json(registry) if registry.exists() else {"schema_version": 1, "entries": []}
        entries = [e for e in ledger["entries"] if e["state_id"] == state_id]
        if not entries:
            raise ValueError(f"No submission is recorded for {state_id}")
        last = entries[-1]
        allowed = UNKNOWN_SUBMISSION + (("submission_failed",) if job_id is not None else ())
        if last.get("status") not in allowed:
            raise ValueError(f"The last submission of {state_id} is '{last.get('status')}'; nothing to resolve")
        if job_id is not None:
            last.update(status="submitted", job_id=str(job_id))
        else:
            last["status"] = "submission_failed"
        last.update(resolved_at_utc=_utc_now(), resolved_by="mechanet resolve-submission")
        write_json(registry, ledger)
    return last


# ---------------------------------------------------------------------------
# Refill: each finishing job tops the campaign up to its cap (docs/execution.md)
# ---------------------------------------------------------------------------


def _refill_path(campaign):
    return Path(campaign).resolve() / REFILL_FILE


def _update_refill(campaign, change, create=False):
    """Read-modify-write refill.json under the campaign's submission lock."""
    from .runner import exclusive_lock

    campaign = Path(campaign).resolve()
    path = _refill_path(campaign)
    with exclusive_lock(campaign / ".submission.lock", "campaign submissions", blocking=True, timeout=300.0):
        if not path.is_file() and not create:
            raise FileNotFoundError(
                f"{path} not found; set up refill with `mechanet submit ... --max-active N --refill --execute`"
            )
        record = read_json(path) if path.is_file() else {}
        change(record)
        write_json(path, record)
    return record


def enable_refill(campaign, targets, max_active):
    """Add states (state ID -> final stage) to the refill, set the cap and enable it.

    States already in the refill keep the deeper of their old and new final stage, so
    adding a selection never shortens or drops what an earlier selection asked for.
    """
    targets = {str(k): _check_through(v) for k, v in targets.items()}
    if not targets:
        raise ValueError("The refill needs at least one state")
    if int(max_active) < 1:
        raise ValueError("max_active must be positive")

    def change(record):
        current = dict(record.get("targets") or {})
        for sid, goal in targets.items():
            if sid not in current or _depth(goal) > _depth(current[sid]):
                current[sid] = goal
        stamp = _utc_now()
        record.update(
            schema_version=1,
            enabled=True,
            targets=current,
            max_active=int(max_active),
            enabled_at_utc=stamp,
            changed_at_utc=stamp,
            disabled_reason=None,
            consecutive_failed_states=0,
            consecutive_sbatch_errors=0,
        )
        record.setdefault("events", [])

    _update_refill(campaign, change, create=True)
    return refill_status(campaign)


def set_refill(campaign, enabled=None, reason=None, max_active=None, submit_from=None):
    """Pause (enabled=False) or resume (True) the refill, change its cap, or where it submits from.

    ``submit_from`` is "jobs" (each ending job submits) or "login" (jobs only record
    their end; `mechanet refill <campaign> --tick` submits, e.g. from a login node).
    """
    if max_active is not None and int(max_active) < 1:
        raise ValueError("max_active must be positive")
    if submit_from not in (None, "jobs", "login"):
        raise ValueError("submit_from must be 'jobs' or 'login'")

    def change(record):
        if submit_from is not None:
            record["submit_from"] = submit_from
        if enabled is not None:
            record.update(enabled=bool(enabled), disabled_reason=None if enabled else (reason or "paused by the user"))
            if enabled:
                record.update(consecutive_failed_states=0, consecutive_sbatch_errors=0)
        if max_active is not None:
            record["max_active"] = int(max_active)
        record["changed_at_utc"] = _utc_now()

    _update_refill(campaign, change)
    return refill_status(campaign)


def clear_refill(campaign):
    """Remove the refill (its states, cap and history); running jobs are not affected."""
    from .runner import exclusive_lock

    campaign = Path(campaign).resolve()
    with exclusive_lock(campaign / ".submission.lock", "campaign submissions", blocking=True, timeout=300.0):
        with contextlib.suppress(FileNotFoundError):
            _refill_path(campaign).unlink()
    return refill_status(campaign)


def refill_status(campaign):
    path = _refill_path(campaign)
    if not path.is_file():
        return {"enabled": False, "note": "No refill is set up for this campaign (no refill.json)."}
    record = read_json(path)
    targets = record.pop("targets", None) or {}
    by_stage = {}
    for goal in targets.values():
        by_stage[goal] = by_stage.get(goal, 0) + 1
    record.update(n_states=len(targets), states_by_final_stage=by_stage, events=record.get("events", [])[-10:])
    return record


def _refill_decision(record, failure):
    """('refill' | 'disable' | 'ignore', reason) for a run that just ended in a job of the campaign."""
    from .runner import LockBusy

    if not record:
        if failure is None or isinstance(failure, LockBusy):
            return "ignore", f"no run record from this job ({type(failure).__name__ if failure else 'none'})"
        return "disable", (
            f"run-state stopped before it could record the run: {type(failure).__name__}: {str(failure)[:300]}"
        )
    if any(event.get("signal") == "SIGTERM" for event in record.get("signals") or []):
        return "ignore", (
            "the job received SIGTERM (scancel, preemption or the hard time limit); no replacement is submitted for it"
        )
    status = record.get("status")
    if status == "complete":
        return "refill", "the state finished its requested stages"
    if status == "incomplete_time_limit":
        if int(record.get("vasp_launches") or 0) >= 1:
            return "refill", "the job stopped cleanly before its time limit after making progress"
        return "disable", (
            "the job stopped before starting VASP ("
            + str(record.get("reason"))[:200]
            + "); compare the walltime with scheduler.min_remaining_s and the measured stage "
            "times, then resume with `mechanet refill <campaign> --on`"
        )
    if status == "failed" and record.get("failure_kind") == "stage":
        return "refill", "this state failed; the other states continue"
    return "disable", (
        f"the run failed ({record.get('failure_kind') or 'unknown'}): {str(record.get('reason'))[:300]}"
        "; fix the cause, then resume with `mechanet refill <campaign> --on`"
    )


def _mark_job(campaign, job_id, started):
    """Record in the ledger that this job's run-state started or ended; False if the job is not in it.

    The end mark lets other refills stop counting the job while SLURM tidies up; the
    start mark clears it again when a requeued job runs once more.
    """
    from .runner import exclusive_lock

    registry = Path(campaign) / "submissions.json"
    if not registry.is_file():
        return False
    with exclusive_lock(Path(campaign) / ".submission.lock", "campaign submissions", blocking=True, timeout=120.0):
        ledger = read_json(registry)
        found = [e for e in ledger.get("entries", []) if str(e.get("job_id")) == str(job_id)]
        stamp = _utc_now()
        for entry in found:
            if started:
                entry.pop("run_ended_at_utc", None)
                entry["run_started_at_utc"] = stamp
            else:
                entry["run_ended_at_utc"] = stamp
        if found:
            write_json(registry, ledger)
    return bool(found)


def note_job_started(campaign, job_id):
    """Called when run-state starts inside a job of a campaign that uses the refill."""
    return _mark_job(campaign, job_id, started=True)


def _run_record_of_this_job(state_dir):
    """run.json written by this job (a failed run raises before returning its record)."""
    job_id = os.environ.get("SLURM_JOB_ID")
    with contextlib.suppress(OSError, ValueError):
        record = read_json(Path(state_dir) / "run.json")
        if job_id and str(record.get("slurm_job_id")) == str(job_id) and record.get("finished_at_utc"):
            return record
    return None


def _absorb(event, outcome, result):
    """Copy what a refill submit did into its event; note sbatch trouble for the counters."""
    event["submitted"] = [j["state_id"] for j in result["jobs"] if j["status"] == "submitted"]
    event["active_jobs"] = result["active_jobs"]
    event["attention"] = [j["state_id"] for j in result["attention"]]
    for key in ("refill_paused", "refill_changed", "stopped"):
        if result.get(key):
            event[key] = result[key]
    if result.get("error"):
        event["error"] = result["error"]
        if result.get("error_kind") == "sbatch_unavailable":
            outcome["disable"] = (
                f"sbatch cannot be run here ({result['error']}); if jobs cannot submit on this "
                "cluster, switch to `mechanet refill <campaign> --submit-from login` and run "
                "`mechanet refill <campaign> --tick` from a login node (docs/execution.md)"
            )
        else:
            outcome["sbatch"] = "error"
    else:
        outcome["sbatch"] = "ok"


def _absorb_exception(event, outcome, exc):
    event["error"] = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, (ValueError, KeyError, FileNotFoundError)):
        # A problem in the campaign itself (configuration, plan, selection) repeats at every
        # refill; transient SLURM or lock errors are retried by the next one instead.
        outcome["disable"] = outcome["disable"] or f"refill failed: {event['error'][:300]}"


def _finish(campaign, event, outcome, echo):
    """Record the event, update the sbatch error count, pause if needed, and print one line."""

    def change(current):
        if outcome["sbatch"] == "error":
            n = int(current.get("consecutive_sbatch_errors") or 0) + 1
            current["consecutive_sbatch_errors"] = n
            if n >= REFILL_MAX_SBATCH_ERRORS and not outcome["disable"]:
                outcome["disable"] = (
                    f"sbatch failed in {n} refills in a row (last: {str(event.get('error'))[:200]}); "
                    "if SLURM refuses jobs submitted from compute nodes, use "
                    "`mechanet refill <campaign> --submit-from login` and `--tick` "
                    "(docs/execution.md), otherwise resume with `mechanet refill <campaign> --on`"
                )
        elif outcome["sbatch"] == "ok":
            current["consecutive_sbatch_errors"] = 0
        if outcome["disable"] and current.get("enabled"):
            current.update(enabled=False, disabled_reason=outcome["disable"], changed_at_utc=event["at_utc"])
        current["events"] = (current.get("events", []) + [event])[-MAX_REFILL_EVENTS:]

    with contextlib.suppress(Exception):
        _update_refill(campaign, change)
    if echo:
        with contextlib.suppress(Exception):
            message = f"mechanet refill: {event.get('decision', 'error')}"
            if event.get("why"):
                message += f": {event['why']}"
            if "submitted" in event:
                message += f"; submitted {len(event['submitted'])}: {' '.join(event['submitted'])}"
            if event.get("error"):
                message += f"; error: {event['error']}"
            if outcome["disable"]:
                message += "; refill is now paused"
            print(message, file=sys.stderr)
    return event


def refill_after_run(campaign, state_dir, record, failure=None, stop_check=None):
    """Called by run-state when it ends: record the end and top the campaign up to the cap. Never raises.

    Only jobs recorded in the campaign's submissions.json take part; run-state on a
    login node, in an salloc session or from your own script never submits.
    ``stop_check()`` is true once the job has received SIGTERM. With
    ``submit_from: login`` the job only records its end and the failure count;
    `mechanet refill <campaign> --tick` submits.
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    campaign = Path(campaign).resolve()
    if not _refill_path(campaign).is_file() or not _mark_job(campaign, job_id, started=False):
        return None
    try:
        settings = read_json(_refill_path(campaign))
    except (OSError, ValueError):
        return None
    if not settings.get("enabled"):
        return None
    stop_check = stop_check or (lambda: False)
    event = {"at_utc": _utc_now(), "job_id": job_id, "state_id": Path(state_dir).name}
    outcome = {"disable": None, "sbatch": None}
    try:
        if record is None:
            record = _run_record_of_this_job(state_dir)
        decision, why = _refill_decision(record, failure)
        if decision == "refill" and stop_check():
            decision, why = "ignore", "SIGTERM arrived while the job was ending; no replacement is submitted"
        if decision == "refill":
            # Failed states in a row that look like a broken setup point to a common cause.
            broken = record.get("status") == "failed" and _failure_without_progress(state_dir, record)
            limit = REFILL_MAX_CONSECUTIVE_FAILURES
            with contextlib.suppress(OSError, ValueError, TypeError):
                limit = int(
                    read_json(campaign / "config.json")
                    .get("scheduler", {})
                    .get("refill_max_consecutive_failures", limit)
                )
            box = {}

            def count(current):
                if broken:
                    current["consecutive_failed_states"] = int(current.get("consecutive_failed_states") or 0) + 1
                elif record.get("status") != "failed":
                    current["consecutive_failed_states"] = 0
                n = int(current.get("consecutive_failed_states") or 0)
                box.update(enabled=bool(current.get("enabled")), mode=current.get("submit_from", "jobs"))
                if broken and n >= limit and current.get("enabled"):
                    reason = (
                        f"{n} jobs in a row ended with a failed state that made no progress (a crash, a fatal "
                        "VASP error or nothing completed); look for a common cause (memory, MPI, the VASP "
                        "build, a setting), then resume with `mechanet refill <campaign> --on`"
                    )
                    current.update(enabled=False, disabled_reason=reason, changed_at_utc=_utc_now())
                    box["paused"] = reason

            _update_refill(campaign, count)
            if box.get("paused"):
                decision, why = "disable", box["paused"]
            elif not box.get("enabled"):
                decision, why = "ignore", "the refill was paused meanwhile"
            elif box.get("mode") == "login":
                decision, why = (
                    "recorded",
                    "submissions come from `mechanet refill <campaign> --tick` (submit_from login)",
                )
        event.update(decision=decision, why=why)
        if decision == "disable":
            outcome["disable"] = why
        elif decision == "refill":
            result = submit(
                campaign,
                through=settings["targets"],
                execute=True,
                max_active=settings["max_active"],
                exclude_job_ids={job_id},
                verify_complete=False,
                origin="refill",
                env=_sbatch_env(),
                stop_check=stop_check,
            )
            _absorb(event, outcome, result)
    except Exception as exc:  # noqa: BLE001 - a refill problem must never fail the finished run
        _absorb_exception(event, outcome, exc)
    return _finish(campaign, event, outcome, echo=True)


def refill_tick(campaign):
    """One refill round from wherever this runs, typically a login node (`mechanet refill <campaign> --tick`).

    It submits up to the cap with the refill's rules (stored states and cap, a pause
    wins, states whose last job made no progress are left to you) and records an event.
    """
    campaign = Path(campaign).resolve()
    if not _refill_path(campaign).is_file():
        raise FileNotFoundError(
            f"No refill is set up for {campaign}; set one up with "
            "`mechanet submit ... --max-active N --refill --execute`"
        )
    settings = read_json(_refill_path(campaign))
    if not settings.get("enabled"):
        return {"decision": "paused", "why": settings.get("disabled_reason") or "the refill is paused", "submitted": []}
    event = {
        "at_utc": _utc_now(),
        "job_id": None,
        "state_id": None,
        "decision": "tick",
        "why": "mechanet refill --tick",
    }
    outcome = {"disable": None, "sbatch": None}
    try:
        result = submit(
            campaign,
            through=settings["targets"],
            execute=True,
            max_active=settings["max_active"],
            verify_complete=False,
            origin="refill",
        )
        _absorb(event, outcome, result)
    except Exception as exc:  # noqa: BLE001 - record and report like an in-job refill
        _absorb_exception(event, outcome, exc)
    _finish(campaign, event, outcome, echo=False)
    if outcome["disable"]:
        event["paused"] = outcome["disable"]
    return event
