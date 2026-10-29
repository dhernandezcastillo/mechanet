"""Prepare and execute the staged VASP calculation of one state (docs/execution.md).

Execution is opt-in.  Attempts are archived, never erased (restart-file copies
excepted).  Two kinds of unsuccessful attempts are distinguished:

* interrupted (wall-time kill, SIGUSR1/SIGTERM, STOPCAR): continued
  automatically in the next run, up to ``vasp.max_continuations``;
* failed (completed but unconverged, fatal VASP error, SCF non-convergence):
  needs ``--retry`` and counts toward ``vasp.max_attempts``.

A launcher error (VASP could not be started) is fixed by the user and simply
re-launched; it consumes neither budget.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import errno
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

from .clusters import DEFAULT_MIN_REMAINING_S
from .fileio import atomic_text, file_hash, object_hash, read_json, write_json
from .vasp import (
    STAGES,
    STAGE_LOOSE,
    STAGE_FINAL,
    STAGE_REFINE,
    STAGE_STATIC,
    STAGE_FREQ,
    RELAXATION_STAGES,
    assemble_potcar,
    central_mode,
    find_campaign,
    frequency_gates,
    plan_mode,
    protocol_stages,
    render_incar,
    resolve_through,
    stage_settings,
    stage_sources,
    state_species,
)

OUTPUT_HASH_FILES = ("OUTCAR", "vasprun.xml", "CONTCAR")
# Displacement units are verified from OUTCAR (+CONTCAR) only, so pruning never invalidates them.
UNIT_OUTPUT_HASH_FILES = ("OUTCAR", "CONTCAR")
RESTART_FILES = ("WAVECAR", "CHGCAR")
DRIVER_OUTPUT_FILES = ("frequencies.json",)
DEFAULT_UNIT_PRUNE_FILES = (
    "PROCAR",
    "DOSCAR",
    "vasprun.xml",
    "EIGENVAL",
    "CHG",
    "PCDAT",
    "XDATCAR",
    "REPORT",
    "WAVECAR",
)
PROTECTED_UNIT_FILES = frozenset(
    {
        "OUTCAR",
        "OSZICAR",
        "CONTCAR",
        "POSCAR",
        "INCAR",
        "KPOINTS",
        "POTCAR",
        "attempt.json",
        "input_hashes.json",
        "stage.json",
        "continuation.json",
        "stdout.log",
        "stderr.log",
        "previous_attempts",
    }
)
LOCK_FALLBACK_ERRNOS = frozenset({errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP, errno.ENOLCK, errno.EINVAL})
STALE_OWNERLESS_LOCK_S = 300.0
FINALIZED_NOTE = "bookkeeping finalized after interruption"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class TimeLimitReached(Exception):
    """Stop cleanly: too little allocation time is left, or a stop signal arrived."""


class LockBusy(RuntimeError):
    """Another live process holds the lock."""


class _UnitFailure(Exception):
    """A displacement unit failed in this run in a way that needs --retry; other units may continue."""


class _UnitHeld(Exception):
    """A displacement unit failed in an earlier run and waits for --retry (or its budget is used up)."""


class StageFailed(RuntimeError):
    """This state needs a decision (failed result, --retry, exhausted budget, force gate).

    Other states are unaffected, so a refilling campaign keeps going."""


class LauncherError(RuntimeError):
    """VASP could not be started: an environment problem that affects every state."""


# ---------------------------------------------------------------------------
# State selection and validation
# ---------------------------------------------------------------------------


def state_directories(campaign, selection=None, limit=None):
    campaign = Path(campaign).resolve()
    paths = sorted(p for p in (campaign / "states").iterdir() if (p / "state.json").is_file())
    for path in paths:
        if path.resolve().parent != (campaign / "states").resolve():
            raise ValueError(f"State directory resolves outside this campaign: {path}")
    if selection is not None and not isinstance(selection, str) and len(selection) == 0:
        raise ValueError("The state selection is empty")
    if selection:
        if isinstance(selection, str):
            selection = [s.strip() for s in selection.split(",") if s.strip()]
        selected = set(selection)
        unknown = selected - {p.name for p in paths}
        if unknown:
            raise ValueError(f"Unknown state IDs: {sorted(unknown)}")
        paths = [p for p in paths if p.name in selected]
    if limit is not None:
        if int(limit) < 1:
            raise ValueError("limit must be positive")
        paths = paths[: int(limit)]
    return paths


def _state_validation(state_dir, config=None):
    from .structure import read_poscar

    state_dir = Path(state_dir)
    meta = read_json(state_dir / "state.json")
    if meta.get("id") != state_dir.name:
        raise ValueError(f"State ID does not match its folder: {state_dir}")
    atoms = read_poscar(state_dir / "POSCAR")
    if list(atoms.symbols) != meta["symbols"]:
        raise ValueError(f"POSCAR/state.json atom-order mismatch in {state_dir}")
    n = len(atoms)
    if len(meta["atom_ids"]) != n or len(set(meta["atom_ids"])) != n:
        raise ValueError("State atom IDs must be unique and match POSCAR")
    for key in ("relax_mobile", "frequency_mobile"):
        flags = np.asarray(meta[key], dtype=bool)
        if flags.shape != (n, 3):
            raise ValueError(f"{key} must have shape ({n},3)")
        if np.any(flags.any(axis=1) != flags.all(axis=1)):
            raise ValueError("This partial-Hessian protocol supports whole-atom TTT/FFF masks only")
    if not np.array_equal(atoms.mobile, np.asarray(meta["relax_mobile"], dtype=bool)):
        raise ValueError(f"Initial POSCAR constraints differ from state.json in {state_dir}")
    state_species(meta, config)
    return meta


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def _link_potential(stage_dir, target):
    link = Path(stage_dir) / "POTCAR"
    if link.exists() or link.is_symlink():
        if link.is_symlink() and link.resolve() == Path(target).resolve():
            return
        if link.is_file() and file_hash(link) == file_hash(target):
            return
        raise ValueError(f"Refusing to replace different POTCAR at {link}")
    link.symlink_to(os.path.relpath(target, link.parent))


def _require_current_campaign(campaign):
    """Refuse campaigns whose build report has a schema older than 3 (not written by mechanet).

    Their config.json lacks mechanet's cluster profiles and species handling. Synthetic
    campaigns without a build report are allowed.
    """
    report = Path(campaign) / "build_report.json"
    if report.is_file():
        try:
            schema = int(read_json(report).get("schema_version", 1))
        except (ValueError, TypeError, OSError):
            schema = 1
        if schema < 3:
            raise ValueError(
                f"{campaign} was not generated by mechanet (build_report schema {schema}, mechanet writes 3); "
                "regenerate it with `mechanet generate`"
            )


def prepare(campaign, selection=None, with_potcars=False, limit=None):
    """Write stage inputs and SLURM wrappers; does not submit or run calculations."""
    from .slurm import write_job_script

    campaign = Path(campaign).resolve()
    _require_current_campaign(campaign)
    config = read_json(campaign / "config.json")
    mode = central_mode(config)
    unused = sorted(
        set(config.get("vasp", {}).get("stage_overrides") or {}) & (set(STAGES) - set(protocol_stages(mode)))
    )
    if unused:
        raise ValueError(
            f"vasp.stage_overrides names {unused}, which the frequency.central = {mode!r} protocol "
            "does not run; remove those entries"
        )
    kpoints = campaign / "inputs" / "KPOINTS"
    if not kpoints.is_file():
        raise FileNotFoundError(kpoints)
    prepared = []
    sbatch = config.get("scheduler") or {}
    cluster_ready = bool((sbatch.get("sbatch") or {}).get("time") or sbatch.get("walltime"))
    for state_dir in state_directories(campaign, selection, limit):
        meta = _state_validation(state_dir, config)
        specs = {name: stage_settings(campaign, meta, name, config) for name in protocol_stages(mode)}
        plan = {
            "schema_version": 2,
            "state_id": meta["id"],
            "state_json_sha256": file_hash(state_dir / "state.json"),
            "initial_poscar_sha256": file_hash(state_dir / "POSCAR"),
            "kpoints_sha256": file_hash(kpoints),
            "stages": specs,
        }
        if mode != "refine":
            plan["central"] = mode  # plans of the default protocol carry no "central" key
        plan["sha256"] = object_hash(plan)
        existing = state_dir / "plan.json"
        if existing.exists() and read_json(existing) != plan:
            previous = None
            with contextlib.suppress(OSError, ValueError):
                previous = plan_mode(read_json(existing))
            if previous is not None and previous != mode:
                raise ValueError(
                    f"{state_dir.name} was prepared with frequency.central = {previous!r}, but config.json "
                    f"now says {mode!r}; a prepared state keeps its protocol. Restore the setting, "
                    "or use a new campaign for the other protocol"
                )
            raise ValueError(
                f"Prepared protocol changed for {state_dir.name}; use a new campaign to preserve provenance"
            )
        if not existing.exists():
            write_json(existing, plan)
        potential = provenance = None
        if with_potcars:
            potential, provenance = assemble_potcar(campaign, meta, config)
            write_json(state_dir / "potcar_provenance.json", provenance)
        for name, spec in specs.items():
            folder = state_dir / name
            folder.mkdir(exist_ok=True)
            expected = render_incar(spec["values"])
            incar = folder / "INCAR"
            if incar.exists() and incar.read_text() != expected:
                raise ValueError(f"INCAR was modified after preparation: {incar}; use a new campaign/config")
            if not incar.exists():
                atomic_text(incar, expected)
            kp = folder / "KPOINTS"
            if kp.exists() and file_hash(kp) != file_hash(kpoints):
                raise ValueError(f"KPOINTS changed in {folder}")
            if not kp.exists():
                shutil.copy2(kpoints, kp)
            if not (folder / "stage.json").exists():
                write_json(folder / "stage.json", spec)
            if name == STAGE_LOOSE and not (folder / "POSCAR").exists():
                shutil.copy2(state_dir / "POSCAR", folder / "POSCAR")
            if potential is not None:
                _link_potential(folder, potential)
            if name != STAGE_LOOSE and not (folder / "input_hashes.json").exists():
                write_json(
                    folder / "PENDING.json",
                    {
                        "geometry_source": spec["geometry_source"],
                        "restart_source": spec["restart_source"],
                        "note": "POSCAR is created by run-state from the successful preceding stage, "
                        "not this placeholder.",
                    },
                )
        if cluster_ready and not (state_dir / "run_job.sh").exists():
            write_job_script(state_dir, through="final")
        prepared.append(
            {
                "state_id": meta["id"],
                "path": str(state_dir),
                "plan_sha256": plan["sha256"],
                "potcar_ready": potential is not None,
            }
        )
    result = {"prepared": prepared, "count": len(prepared), "submitted": False}
    if not cluster_ready:
        result["note"] = (
            "No cluster profile in config.json, so no job scripts were written; set one with "
            "`mechanet cluster use NAME` (submit writes the job scripts)."
        )
    return result


# ---------------------------------------------------------------------------
# Locks (flock, with an atomic-mkdir fallback for file systems without flock)
# ---------------------------------------------------------------------------


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError, TypeError, OverflowError):
        return True
    return True


def _read_owner(lock_dir):
    try:
        return json.loads((Path(lock_dir) / "owner.json").read_text())
    except (OSError, ValueError):
        return None


def _owner_is_stale(owner, lock_dir):
    """A mkdir lock is stale only when its owner is demonstrably gone."""
    if not owner:
        try:
            age = time.time() - Path(lock_dir).stat().st_mtime
        except FileNotFoundError:
            return False
        return age > STALE_OWNERLESS_LOCK_S
    same_host = owner.get("host") == socket.gethostname()
    job = owner.get("slurm_job_id")
    if job and str(job) == os.environ.get("SLURM_JOB_ID"):
        # An earlier incarnation (requeue) or an earlier step of this very job.
        return (not _pid_alive(owner.get("pid"))) if same_host else True
    if job:
        from .slurm import job_is_active

        try:
            return not job_is_active(job)
        except (RuntimeError, OSError):
            return False
    if same_host:
        return not _pid_alive(owner.get("pid"))
    return False


def _break_stale_lock(lock_dir, stale_owner):
    lock_dir = Path(lock_dir)
    moved = lock_dir.with_name(f"{lock_dir.name}.stale.{uuid.uuid4().hex}")
    try:
        os.rename(lock_dir, moved)
    except OSError:
        return
    current = _read_owner(moved)
    if current and stale_owner and current.get("token") != stale_owner.get("token"):
        # Another process acquired the lock in between: give it back.
        with contextlib.suppress(OSError):
            os.rename(moved, lock_dir)
            return
    shutil.rmtree(moved, ignore_errors=True)


def _mkdir_lock(lock_dir, description, blocking, timeout):
    lock_dir = Path(lock_dir)
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.mkdir(lock_dir)
        except FileExistsError:
            owner = _read_owner(lock_dir)
            if _owner_is_stale(owner, lock_dir):
                _break_stale_lock(lock_dir, owner)
                continue
            if not blocking or time.monotonic() >= deadline:
                raise LockBusy(
                    f"Another runner holds the lock for {description} "
                    f"(directory lock owned by {owner or 'an unknown process'})"
                ) from None
            time.sleep(0.5)
            continue
        owner = {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "started_at_utc": now(),
            "token": uuid.uuid4().hex,
        }
        write_json(lock_dir / "owner.json", owner)
        return owner["token"]


def _release_mkdir_lock(lock_dir, token):
    lock_dir = Path(lock_dir)
    owner = _read_owner(lock_dir)
    if not owner or owner.get("token") != token:
        return
    moved = lock_dir.with_name(f"{lock_dir.name}.released.{uuid.uuid4().hex}")
    with contextlib.suppress(OSError):
        os.rename(lock_dir, moved)
        shutil.rmtree(moved, ignore_errors=True)


def _try_flock(handle, description, blocking, timeout):
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError as exc:
            if not blocking or time.monotonic() >= deadline:
                raise LockBusy(f"Another runner holds the lock for {description}") from exc
            time.sleep(0.25)
        except OSError as exc:
            if exc.errno in LOCK_FALLBACK_ERRNOS:
                return False
            raise


@contextlib.contextmanager
def exclusive_lock(base, description, blocking=False, timeout=300.0):
    """flock(LOCK_EX|LOCK_NB) on ``base``; atomic ``base.d`` directory lock where flock is unsupported."""
    base = Path(base)
    handle = base.open("a+")
    try:
        has_flock = _try_flock(handle, description, blocking, timeout)
    except BaseException:
        handle.close()
        raise
    if has_flock:
        try:
            yield {"mode": "flock", "path": str(base)}
        finally:
            handle.close()
        return
    handle.close()
    lock_dir = base.with_name(base.name + ".d")
    token = _mkdir_lock(lock_dir, description, blocking, timeout)
    try:
        yield {"mode": "mkdir", "path": str(lock_dir)}
    finally:
        _release_mkdir_lock(lock_dir, token)


def _state_lock(state_dir):
    state_dir = Path(state_dir)
    return exclusive_lock(state_dir / ".run.lock", state_dir.name)


# ---------------------------------------------------------------------------
# Time limit and stop signals
# ---------------------------------------------------------------------------


def job_end_time(environ=None):
    """Allocation end (epoch s): SLURM_JOB_END_TIME, else `squeue -h -j $SLURM_JOB_ID -o %e`, else None."""
    env = os.environ if environ is None else environ
    raw = env.get("SLURM_JOB_END_TIME")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    job = env.get("SLURM_JOB_ID")
    if not job:
        return None
    try:
        proc = subprocess.run(
            ["squeue", "-h", "-j", str(job), "-o", "%e"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env={k: v for k, v in os.environ.items() if not k.startswith("SQUEUE_")},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or not lines:
        return None
    try:
        return time.mktime(time.strptime(lines[0], "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return None


class StopController:
    """Signal-driven graceful stop and allocation-deadline checks for one run."""

    def __init__(self, config, environ=None):
        configured = config.get("scheduler", {}).get("min_remaining_s") or {}
        if not isinstance(configured, dict):
            raise ValueError(
                "scheduler.min_remaining_s must be a mapping such as {relaxation: 1800, single_point: 5400, "
                "displacement: 2700}"
            )
        self.min_remaining = dict(DEFAULT_MIN_REMAINING_S)
        self.min_remaining.update({k: float(v) for k, v in configured.items() if k in DEFAULT_MIN_REMAINING_S})
        self.end_time = job_end_time(environ)
        env = os.environ if environ is None else environ
        try:
            self.start_time = float(env.get("SLURM_JOB_START_TIME")) if env.get("SLURM_JOB_START_TIME") else time.time()
        except ValueError:
            self.start_time = time.time()
        self.requested = None
        self.active = None
        self.events = []
        self._previous = {}

    def install(self):
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            try:
                self._previous[sig] = signal.signal(sig, self._handle)
            except (ValueError, OSError):  # not the main thread
                pass

    def uninstall(self):
        for sig, handler in self._previous.items():
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, handler)
        self._previous = {}

    def _handle(self, signum, frame):
        name = signal.Signals(signum).name
        self.requested = self.requested or name
        event = {"signal": name, "at_utc": now()}
        active = self.active
        if active is not None:
            event["folder"] = str(active["folder"])
            if active["kind"] == "relaxation":
                with contextlib.suppress(OSError):
                    atomic_text(active["folder"] / "STOPCAR", "LSTOP = .TRUE.\n")
                event["stopcar"] = True
            record = active["attempt"]
            record.update(stop_requested_at_utc=event["at_utc"], stop_signal=name)
            if signum == signal.SIGTERM:
                record.update(
                    status="interrupted", reason="SIGTERM while VASP was running (time limit or cancellation)"
                )
            with contextlib.suppress(OSError, ValueError):
                write_json(active["attempt_path"], record)
        self.events.append(event)

    def remaining(self):
        return None if self.end_time is None else self.end_time - time.time()

    def check(self, kind, measured_s=None, label="the next unit"):
        if self.requested:
            raise TimeLimitReached(f"stop requested by {self.requested} before {label}")
        remaining = self.remaining()
        if remaining is None:
            return
        required = self.min_remaining[kind]
        source = f"min_remaining_s.{kind}"
        if measured_s and 1.25 * float(measured_s) > required:
            required, source = 1.25 * float(measured_s), f"measured duration of {label}, with margin"
        warning = None
        # Never demand more than an allocation can provide: that would stop every future
        # job before launch (exit 0) without anyone noticing. A requirement capped at 90%
        # of the allocation is met at the start of a fresh job, so such a stage runs one per job.
        allocation = self.end_time - self.start_time if self.end_time is not None else None
        if allocation and allocation > 0 and required > 0.9 * allocation:
            warning = (
                f"{label} needs about {required:.0f} s ({source}), more than 90% of this allocation "
                f"({allocation:.0f} s); starting it anyway"
            )
            required = 0.9 * allocation
        if remaining < required:
            raise TimeLimitReached(
                f"{max(remaining, 0):.0f} s left in the allocation; {label} needs at least {required:.0f} s ({kind})"
            )
        if warning:
            self.events.append({"warning": warning, "at_utc": now()})


# ---------------------------------------------------------------------------
# Stage evidence helpers
# ---------------------------------------------------------------------------


def _max_force(forces, mobile):
    if forces is None:
        return None
    f = np.asarray(forces, dtype=float)
    m = np.asarray(mobile, dtype=bool)
    if f.shape != m.shape or not np.isfinite(f).all():
        return None
    return float(np.max(np.linalg.norm(np.where(m, f, 0.0), axis=1)))


def _geometry_valid(path, meta):
    from .structure import read_poscar

    try:
        atoms = read_poscar(path)
        return (
            atoms.symbols == meta["symbols"]
            and np.isfinite(atoms.positions).all()
            and abs(np.linalg.det(atoms.cell)) > 1e-6
        )
    except (ValueError, OSError, IndexError, TypeError):
        return False


def _restart_evidence(folder, stage):
    """Backward-compatible wrapper: stage name -> required restart file."""
    from .results import restart_evidence

    required = {STAGE_REFINE: "CHGCAR", STAGE_STATIC: "CHGCAR", STAGE_FREQ: "WAVECAR"}.get(stage, stage)
    return restart_evidence(folder, required if required in ("CHGCAR", "WAVECAR") else None)


def _read_attempt(folder):
    path = Path(folder) / "attempt.json"
    try:
        return read_json(path) if path.exists() else {}
    except (OSError, ValueError):
        return {"status": "running", "reason": "unreadable attempt.json"}


def _acceptable(stage, result, folder, meta, spec, attempt=None):
    folder = Path(folder)
    if attempt is None:
        attempt = _read_attempt(folder)
    if attempt.get("status") == "failed" and attempt.get("returncode") not in (None, 0):
        return False, f"Recorded launcher failure {attempt['returncode']} cannot be accepted as success"
    if not result.get("completed"):
        return False, "VASP did not write a normal completion footer"
    if result.get("input_hashes_valid") is False:
        return False, "Input hash mismatch: outputs do not belong to the current stage inputs"
    if stage in RELAXATION_STAGES:
        if not result.get("last_electronic_converged", False):
            return False, "Required electronic calculation(s) did not converge"
        if not _geometry_valid(folder / "CONTCAR", meta):
            return False, "Missing or incompatible final CONTCAR"
        stopped = (result.get("stopcar_present") or attempt.get("stop_requested_at_utc")) and attempt.get(
            "status"
        ) != "complete"
        if stopped and not result.get("ionic_converged"):
            # Even the loose stage must finish its own NSW/criterion; a STOPCAR stop is an interruption.
            return False, "Relaxation was stopped via STOPCAR before reaching its stopping criterion"
    elif not result.get("electronic_converged", False):
        return False, "Required electronic calculation(s) did not converge"
    if stage in (STAGE_FINAL, STAGE_REFINE):
        label = "Final geometry" if stage == STAGE_FINAL else "Refined geometry"
        if not result.get("ionic_converged"):
            return False, f"{label} did not reach its ionic convergence criterion"
        threshold = -float(spec["values"].get("EDIFFG", "0"))
        if threshold <= 0:
            return False, f"{stage} must use an explicit negative EDIFFG force threshold"
        max_force = _max_force(result.get("forces_eVA"), meta["relax_mobile"])
        if max_force is None or max_force > threshold + 1e-5:
            return False, f"{label}: movable-atom force {max_force} exceeds {threshold} eV/Angstrom"
    if stage == STAGE_FREQ and not result.get("frequency_converged", False):
        return False, "Frequency calculation lacks a complete, SCF-converged partial Hessian"
    evidence = result.get("restart_evidence") or {}
    if evidence.get("fallback"):
        return False, (
            f"VASP reported failure/fallback of required {evidence.get('required')} restart "
            f"({', '.join(evidence.get('markers', []))})"
        )
    return True, "accepted"


def _classify(relaxation, result, folder, attempt, live=False, stop_signal=None, returncode=None):
    """Interrupted (continue automatically), launcher_error (re-launch) or failed (needs --retry)."""
    folder = Path(folder)
    status = attempt.get("status")
    if status == "launcher_error":
        return "launcher_error"
    fatal = bool(result.get("fatal_errors"))
    footer = bool(result.get("completed"))
    stopcar = (folder / "STOPCAR").exists() or bool(attempt.get("stop_requested_at_utc"))
    stopped_relaxation = relaxation and stopcar and footer and not result.get("ionic_converged")
    if status == "interrupted" and not fatal:
        return "interrupted"
    if live:
        if stop_signal and not fatal and (not footer or stopped_relaxation):
            return "interrupted"
        outcar = folder / "OUTCAR"
        if returncode not in (None, 0) and (not outcar.exists() or outcar.stat().st_size == 0):
            return "launcher_error"
        if stopped_relaxation and not fatal:
            return "interrupted"
        return "failed"
    if status in ("running", None):
        # The process that recorded this attempt is gone (we hold the state lock).
        if stopped_relaxation and not fatal:
            return "interrupted"
        if not footer and not fatal and result.get("input_hashes_valid") is not False:
            return "interrupted"
    return "failed"


def _archive_counts(folder):
    counts = {"failed": 0, "interrupted": 0, "launcher_error": 0}
    archives = Path(folder) / "previous_attempts"
    if archives.is_dir():
        for path in archives.iterdir():
            if not path.is_dir() or not path.name.isdigit():
                continue
            info = {}
            with contextlib.suppress(OSError, ValueError):
                info = read_json(path / "archive.json")
            kind = info.get("kind", "failed")  # an archive without a kind is an explicit --retry failure
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def _archive_attempt(folder, kind="failed", reason=None):
    """Move one attempt into previous_attempts/NNN; restart-file copies are deleted, not archived."""
    folder = Path(folder)
    archives = folder / "previous_attempts"
    archives.mkdir(exist_ok=True)
    numbers = [int(p.name) for p in archives.iterdir() if p.is_dir() and p.name.isdigit()]
    target = archives / f"{max(numbers, default=0) + 1:03d}"
    target.mkdir()
    status = _read_attempt(folder).get("status")
    removed = []
    for name in RESTART_FILES:
        path = folder / name
        if path.is_file() or path.is_symlink():
            removed.append({"name": name, "bytes": path.stat().st_size if path.exists() else 0})
            path.unlink()
    keep = {"previous_attempts", "INCAR", "KPOINTS", "POTCAR", "stage.json", "PENDING.json", "units"}
    for path in list(folder.iterdir()):
        if path.name in keep:
            if path.name in ("INCAR", "KPOINTS", "stage.json"):
                shutil.copy2(path, target / path.name)
            continue
        shutil.move(str(path), str(target / path.name))
    write_json(
        target / "archive.json",
        {
            "kind": kind,
            "reason": reason,
            "attempt_status": status,
            "archived_at_utc": now(),
            "removed_restart_files": removed,
        },
    )
    return target


def _verify_input_hashes(folder):
    folder = Path(folder)
    path = folder / "input_hashes.json"
    if not path.is_file():
        raise ValueError(
            f"Existing output without mechanet input provenance: {folder}; import separately or start a new state"
        )
    for name, expected in read_json(path)["files"].items():
        if name in RESTART_FILES:
            continue  # restart-file copies are verified before launch only
        current = folder / name
        if not current.is_file() or file_hash(current) != expected:
            raise ValueError(f"Input changed after this attempt: {current}; cannot reuse its output")


def unit_prune_files(config):
    """Files deleted from an accepted displacement unit (empty when pruning is disabled)."""
    freq = config.get("frequency", {})
    if not freq.get("prune_unit_outputs", True):
        return ()
    names = freq.get("unit_prune_files", list(DEFAULT_UNIT_PRUNE_FILES))
    if isinstance(names, str) or not all(isinstance(n, str) for n in names):
        raise ValueError("frequency.unit_prune_files must be a list of file names")
    bad = [n for n in names if n in PROTECTED_UNIT_FILES or not n or n in (".", "..") or any(c in n for c in "/\\*?[")]
    if bad:
        raise ValueError(
            f"frequency.unit_prune_files may not name {bad}: units are verified from OUTCAR, "
            "CONTCAR, POSCAR and their records"
        )
    return tuple(dict.fromkeys(names))


def _prune_unit(unit_dir, names):
    removed = []
    for name in names:
        path = Path(unit_dir) / name
        if path.is_file() and not path.is_symlink():
            removed.append({"name": name, "bytes": path.stat().st_size})
            path.unlink()
    return removed


def _verify_output_hashes(folder, names=OUTPUT_HASH_FILES + DRIVER_OUTPUT_FILES):
    folder = Path(folder)
    attempt = folder / "attempt.json"
    if not attempt.exists():
        return
    for name, expected in read_json(attempt).get("output_hashes", {}).items():
        if name not in names:
            continue  # restart files (and pruned unit outputs) may be deleted after acceptance
        if isinstance(expected, dict):
            expected = expected.get("sha256")
        path = folder / name
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Accepted output changed after its calculation: {path}; cannot silently reuse it")


def _output_hashes(folder, names=OUTPUT_HASH_FILES):
    return {name: file_hash(Path(folder) / name) for name in names if (Path(folder) / name).is_file()}


def _finalize(folder, attempt, result, names=OUTPUT_HASH_FILES):
    """An attempt recorded as running/interrupted whose VASP run completed acceptably."""
    record = dict(attempt)
    record.update(
        status="complete",
        reason="accepted",
        result=result,
        finished_at_utc=record.get("finished_at_utc") or now(),
        output_hashes=_output_hashes(folder, names),
    )
    record["warnings"] = list(record.get("warnings", [])) + [FINALIZED_NOTE]
    write_json(Path(folder) / "attempt.json", record)
    return record


def _stage_geometry(state_dir, stage, meta, continuation=None, plan=None):
    """Write the stage POSCAR from its geometry source (plan.json; e.g. 04_static reads 03_refine/CONTCAR,
    or 02_final/CONTCAR with frequency.central = "final"). The frequency reference is always the
    04_static central geometry, never a displaced one."""
    from .structure import read_poscar, write_poscar

    relative = ((plan or {}).get("stages", {}).get(stage) or {}).get("geometry_source") or stage_sources(
        plan_mode(plan)
    )[0][stage]
    source = Path(continuation) if continuation is not None else state_dir / relative
    atoms = read_poscar(source)
    if list(atoms.symbols) != meta["symbols"]:
        raise ValueError(f"Stage geometry atom ordering changed at {source}")
    atoms.ids = list(meta["atom_ids"])
    atoms.mobile = np.asarray(meta["frequency_mobile"] if stage == STAGE_FREQ else meta["relax_mobile"], dtype=bool)
    write_poscar(state_dir / stage / "POSCAR", atoms)
    return str(source)


def _copy_restart(source, dest, known_sha=None):
    """Copy (never link) a restart file and verify the copy before launch."""
    source, dest = Path(source), Path(dest)
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"A nonempty {source} is required to continue the electronic solution")
    digest = known_sha or file_hash(source)
    if dest.is_symlink():
        dest.unlink()
    shutil.copy2(source, dest)
    if file_hash(dest) != digest:
        raise ValueError(f"Copied restart file {dest} differs from {source}")
    return {"source": str(source), "bytes": source.stat().st_size, "sha256": digest}


def _continuation_geometry(relaxation, archived, result, meta, kind):
    if not relaxation:
        return None, "stage default geometry source"
    contcar, poscar = archived / "CONTCAR", archived / "POSCAR"
    steps = int(result.get("n_completed_ionic_steps") or 0)
    usable = steps >= 1 and _geometry_valid(contcar, meta)
    if kind == "failed":
        usable = usable and bool(result.get("last_electronic_converged"))
    if usable:
        return contcar, f"CONTCAR of {archived.name} after {steps} completed ionic step(s)"
    if _geometry_valid(poscar, meta):
        return poscar, f"input POSCAR of {archived.name} (no usable completed ionic step)"
    return None, "stage default geometry source"


# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------


class _Run:
    """Per-run context shared by the stage functions."""

    def __init__(self, state_dir, campaign, config, plan, meta, retry, stopper, command):
        self.state_dir, self.campaign, self.config = state_dir, campaign, config
        self.plan, self.meta, self.retry, self.stopper, self.command = plan, meta, retry, stopper, command
        vasp = config.get("vasp", {})
        self.max_attempts = int(vasp.get("max_attempts", 2))
        self.max_continuations = int(vasp.get("max_continuations", 8))
        self.notes = []
        self.launches = 0
        self._hash_cache = {}

    def source_hash(self, path):
        path = Path(path)
        key = (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
        if key not in self._hash_cache:
            self._hash_cache[key] = file_hash(path)
        return self._hash_cache[key]

    def attempt_record(self):
        return {
            "status": "running",
            "command": self.command,
            "started_at_utc": now(),
            "config_sha256": file_hash(self.campaign / "config.json"),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }

    def launch(self, folder, kind, attempt):
        """Run the VASP command; returns (returncode, OSError or None, elapsed seconds)."""
        self.launches += 1
        self.stopper.active = {
            "folder": Path(folder),
            "kind": kind,
            "attempt": attempt,
            "attempt_path": Path(folder) / "attempt.json",
        }
        started = time.time()
        try:
            with (Path(folder) / "stdout.log").open("w") as out, (Path(folder) / "stderr.log").open("w") as err:
                proc = subprocess.run(self.command, cwd=folder, stdout=out, stderr=err, check=False)
            return proc.returncode, None, time.time() - started
        except OSError as exc:
            return None, exc, time.time() - started
        finally:
            self.stopper.active = None


def _measured_seconds(folder, per_ionic_step):
    """Measured wall time from earlier attempts of this stage.

    Relaxations stop cleanly after their current ionic step, so the longest time per
    completed ionic step is what must fit; a single point cannot be cut short, so
    the longest earlier run is used. Only attempts in which VASP ended normally
    (timing footer written, including STOPCAR stops) count: a run killed at the
    walltime, possibly after hanging, would otherwise inflate the estimate until no
    allocation is ever long enough.
    """
    values = []
    archives = Path(folder) / "previous_attempts"
    paths = [p / "attempt.json" for p in archives.iterdir()] if archives.is_dir() else []
    paths.append(Path(folder) / "attempt.json")  # the attempt about to be archived, if any
    for path in paths:
        with contextlib.suppress(OSError, ValueError, TypeError, ZeroDivisionError, AttributeError):
            record = read_json(path)
            if not record.get("elapsed_s") or record.get("status") == "launcher_error":
                continue
            if not (record.get("result") or {}).get("completed"):
                continue
            if per_ionic_step:
                steps = int(record.get("result", {}).get("n_completed_ionic_steps") or 0)
                if steps:
                    values.append(float(record["elapsed_s"]) / steps)
            else:
                values.append(float(record["elapsed_s"]))
    return max(values) if values else None


def _gate_failures(record, gates, final):
    """Why 05_freq may not start, from 04_static measurements (a fresh check or a BLOCKED.json).

    Every protocol: the largest force on a frequency-active atom. frequency.central = "final"
    in addition: 04_static repeats the converged 02_final point, so its E0 and local moments must
    reproduce 02_final, or the Hessian would belong to another electronic solution than the energy.
    """
    reasons = []
    force, limit = record.get("max_active_force_eVA"), gates["max_force_eVA"]
    if force is None or float(force) > limit:
        reasons.append(
            f"static active force {force} exceeds " + (f"frequency.max_force_eVA = {limit}" if final else f"{limit}")
        )
    if final:
        if "static_minus_final_e0_eV" in record:
            shift = record["static_minus_final_e0_eV"]
            if shift is None or abs(float(shift)) > gates["max_static_e0_shift_eV"]:
                amount = f"{float(shift) * 1000:+.1f} meV" if shift is not None else "an unknown amount (E0 missing)"
                reasons.append(
                    f"04_static E0 differs from the 02_final E0 by {amount} at the same geometry "
                    f"(frequency.max_static_e0_shift_eV = {gates['max_static_e0_shift_eV']})"
                )
        change = record.get("max_moment_change_final_to_static_muB")
        if change is not None and float(change) > gates["max_moment_change_muB"]:
            reasons.append(
                f"a local moment changed by {float(change):.2f} muB from 02_final to 04_static "
                f"(frequency.max_moment_change_muB = {gates['max_moment_change_muB']})"
            )
    return reasons


def _static_force_gate(run):
    """Check 04_static before any frequency single point runs (see _gate_failures)."""
    from .results import parse_run, same_solution

    gates = frequency_gates(run.config)
    final = plan_mode(run.plan) == "final"
    static_result = parse_run(run.state_dir / STAGE_STATIC)
    max_force = _max_force(static_result.get("forces_eVA"), run.meta["frequency_mobile"])
    gate = {
        "max_active_force_eVA": max_force,
        "threshold_eVA": gates["max_force_eVA"],
        "warn_threshold_eVA": gates["warn_force_eVA"],
    }
    if final:
        gate.update(
            same_solution(parse_run(run.state_dir / STAGE_FINAL), static_result),
            max_static_e0_shift_eV=gates["max_static_e0_shift_eV"],
            max_moment_change_muB=gates["max_moment_change_muB"],
        )
    failures = _gate_failures(gate, gates, final)
    if failures:
        write_json(run.state_dir / STAGE_FREQ / "BLOCKED.json", dict(gate, reason="; ".join(failures)))
        if final:
            raise StageFailed(
                f"{run.meta['id']}: " + "; ".join(failures) + ". 04_static repeats the converged 02_final point "
                "with the same settings, so compare the forces, energies and local moments of both stages "
                "(another SCF solution?). If the difference is acceptable, raise the threshold named above and "
                "run again; otherwise leave this state out of the free energies"
            )
        raise StageFailed(f"{run.meta['id']}: {failures[0]}; inspect/reoptimize before frequencies")
    if max_force > gates["warn_force_eVA"]:
        gate["warning"] = (
            f"Static force on a frequency-active atom ({max_force:.4f} eV/A) exceeds "
            f"frequency.warn_force_eVA={gates['warn_force_eVA']}; low modes may be contaminated."
        )
        run.notes.append(gate["warning"])
    return gate


def _assess_existing(run, label, folder, relaxation, evaluate, output_names=OUTPUT_HASH_FILES):
    """Decide what to do with an existing attempt, without archiving anything yet.

    ``evaluate(attempt)`` returns (result, accepted, reason).  Returns
    ("already_complete" | "finalized", None) or ("relaunch", info); raises when the
    attempt needs --retry or a budget is exhausted.
    """
    _verify_input_hashes(folder)
    _verify_output_hashes(folder, tuple(output_names) + DRIVER_OUTPUT_FILES)
    attempt = _read_attempt(folder)
    label = f"{run.meta['id']} {label}"
    if attempt.get("status") == "launcher_error":
        return "relaunch", {"kind": "launcher_error", "reason": attempt.get("reason"), "result": {}}
    result, accepted, reason = evaluate(attempt)
    if accepted:
        if attempt.get("status") == "complete":
            return "already_complete", None
        _finalize(folder, attempt, result, output_names)
        return "finalized", None
    kind = _classify(relaxation, result, folder, attempt)
    counts = _archive_counts(folder)
    if kind == "interrupted":
        if counts["interrupted"] >= run.max_continuations:
            _record_verdict(folder, attempt, "interrupted", reason)
            raise StageFailed(
                f"{label}: {counts['interrupted']} automatic continuations already used "
                f"(vasp.max_continuations={run.max_continuations}); inspect walltime and "
                "convergence before continuing in a new campaign or with a larger cap"
            )
    else:
        if not run.retry:
            _record_verdict(folder, attempt, "failed", reason)
            raise StageFailed(f"{label}: {reason}. Use --retry for a recorded same-protocol attempt.")
        if counts["failed"] + 1 >= run.max_attempts:
            _record_verdict(folder, attempt, "failed", reason)
            raise StageFailed(f"Maximum recorded attempts reached for {label}")
    return "relaunch", {"kind": kind, "reason": reason, "result": result}


def _record_verdict(folder, attempt, status, reason):
    """Store the verdict on an attempt before refusing to continue it.

    An attempt left "running" by a runner that died (or without attempt.json), or one
    marked "interrupted" by a SIGTERM whose outputs then turn out to be a failure (a
    fatal VASP message), would otherwise look continuable to `submit`, and every new
    job would stop at once. Complete and failed attempts are left untouched.
    """
    stored = attempt.get("status")
    if stored == status or stored not in ("running", None, "interrupted"):
        return
    folder = Path(folder)
    path = folder / "attempt.json"
    if attempt.get("reason") == "unreadable attempt.json" and path.exists():
        os.replace(path, folder / "attempt.unreadable.json")  # keep the evidence
        attempt = {}
    write_json(
        path,
        dict(
            attempt,
            status=status,
            reason=reason,
            verdict_recorded_at_utc=now(),
            verdict_note="the runner that started this attempt had stopped; status recorded by a later run-state",
        ),
    )


def _archive_for_relaunch(run, folder, relaxation, info):
    """Archive the assessed attempt and persist where the next attempt starts from.

    continuation.json survives a stop between archiving and launching; it is removed
    once the next attempt is recorded.
    """
    archived = _archive_attempt(folder, info["kind"], info["reason"])
    if info["kind"] == "launcher_error":
        poscar = archived / "POSCAR"
        geometry = poscar if relaxation and _geometry_valid(poscar, run.meta) else None
        why = "re-launch after a launcher error"
    else:
        geometry, why = _continuation_geometry(relaxation, archived, info["result"], run.meta, info["kind"])
    continuation = {
        "kind": info["kind"],
        "archive": archived.name,
        "geometry": str(geometry) if geometry else None,
        "why": why,
        "previous_completed_ionic_steps": info["result"].get("n_completed_ionic_steps"),
    }
    write_json(Path(folder) / "continuation.json", continuation)
    return continuation


def _pending_continuation(folder):
    path = Path(folder) / "continuation.json"
    if not path.is_file():
        return None
    continuation = read_json(path)
    if continuation.get("geometry") and not Path(continuation["geometry"]).is_file():
        continuation["geometry"] = None
    return continuation


def _run_vasp_stage(run, stage):
    from .results import parse_run

    state_dir, meta = run.state_dir, run.meta
    folder = state_dir / stage
    spec = run.plan["stages"][stage]
    relaxation = stage in RELAXATION_STAGES
    kind = "relaxation" if relaxation else "single_point"

    def evaluate(attempt):
        result = parse_run(folder)
        return (result,) + _acceptable(stage, result, folder, meta, spec, attempt)

    info = None
    if any((folder / f).exists() for f in ("OUTCAR", "vasprun.xml", "stdout.log", "attempt.json")):
        action, info = _assess_existing(run, stage, folder, relaxation, evaluate)
        if action != "relaunch":
            return action
    run.stopper.check(kind, _measured_seconds(folder, per_ionic_step=relaxation), label=stage)
    gate = _static_force_gate(run) if stage == STAGE_FREQ else None
    continuation = (
        _archive_for_relaunch(run, folder, relaxation, info) if info is not None else _pending_continuation(folder)
    )
    source = _stage_geometry(state_dir, stage, meta, continuation["geometry"] if continuation else None, plan=run.plan)
    restart_files = {}
    # Electronic restart source from plan.json: 03_refine reads 02_final/CHGCAR; 04_static reads
    # 03_refine/CHGCAR (02_final/CHGCAR with frequency.central = "final"); 05_freq reads 04_static/WAVECAR.
    restart = spec.get("restart_source", stage_sources(plan_mode(run.plan))[1][stage])
    if restart and Path(restart).name == "CHGCAR":
        if (folder / "WAVECAR").exists():
            raise ValueError(f"{stage} must start from CHGCAR only; unexpected WAVECAR")
        restart_files["CHGCAR"] = _copy_restart(state_dir / restart, folder / "CHGCAR")
    elif restart and Path(restart).name == "WAVECAR":
        wave = state_dir / restart
        restart_files["WAVECAR"] = _copy_restart(
            wave, folder / "WAVECAR", run.source_hash(wave) if wave.is_file() else None
        )
    else:
        # Fresh stages must never accidentally inherit an old electronic restart.
        for name in RESTART_FILES:
            if (folder / name).exists():
                raise ValueError(f"Unexpected restart file in a fresh-start stage: {folder / name}")
    with contextlib.suppress(FileNotFoundError):
        (folder / "STOPCAR").unlink()
    input_record = {
        "files": {f: file_hash(folder / f) for f in ("INCAR", "POSCAR", "KPOINTS", "POTCAR")},
        "restart_files": restart_files,
        "state_json_sha256": file_hash(state_dir / "state.json"),
        "plan_sha256": run.plan["sha256"],
        "geometry_source": source,
        "continuation": continuation,
        "restart_wavecar": restart_files.get("WAVECAR"),
        "restart_chgcar": restart_files.get("CHGCAR"),
        "static_force_check": gate,
        "started_at_utc": now(),
    }
    write_json(folder / "input_hashes.json", input_record)
    for pending in (folder / "PENDING.json", folder / "BLOCKED.json"):
        with contextlib.suppress(FileNotFoundError):
            pending.unlink()
    if run.stopper.requested:
        raise TimeLimitReached(f"stop requested by {run.stopper.requested} before launching {stage}")
    attempt = run.attempt_record()
    write_json(folder / "attempt.json", attempt)
    with contextlib.suppress(FileNotFoundError):
        (folder / "continuation.json").unlink()
    returncode, launch_error, elapsed = run.launch(folder, kind, attempt)
    attempt = dict(attempt)
    result = parse_run(folder)
    accepted, reason = _acceptable(stage, result, folder, meta, spec, attempt)
    warnings = list(run.notes) if stage == STAGE_FREQ else []
    evidence = result.get("restart_evidence", {})
    if evidence.get("required") and evidence.get("verified") is None and not evidence.get("fallback"):
        warnings.append(
            "Restart input was provided, but this VASP build's successful-read message was not recognized; "
            "inspect startup output."
        )
    if launch_error is not None:
        status, reason = "launcher_error", f"Could not start the VASP launcher {run.command[0]!r}: {launch_error}"
    else:
        if returncode != 0 and accepted:
            accepted, reason = False, f"VASP launcher returned {returncode}"
        if accepted:
            status = "complete"
        else:
            status = _classify(
                relaxation, result, folder, attempt, live=True, stop_signal=run.stopper.requested, returncode=returncode
            )
            if status == "launcher_error":
                reason = f"VASP did not start (launcher returned {returncode}, no OUTCAR); inspect stderr.log"
            elif status == "interrupted":
                reason = f"interrupted ({run.stopper.requested or 'STOPCAR'}): {reason}"
    attempt.update(
        status=status,
        returncode=returncode,
        finished_at_utc=now(),
        elapsed_s=round(elapsed, 3),
        reason=reason,
        result=result,
        restart_evidence=evidence,
        warnings=warnings,
    )
    if status == "complete":
        attempt["output_hashes"] = _output_hashes(folder)
    write_json(folder / "attempt.json", attempt)
    if status == "complete":
        return "complete"
    if status == "interrupted":
        raise TimeLimitReached(f"{meta['id']} {stage} {reason}; the next run continues automatically")
    if status == "launcher_error":
        raise LauncherError(
            f"{meta['id']} {stage}: {reason}. Fix the environment; the next run re-launches "
            "this stage without --retry and without using an attempt."
        )
    raise StageFailed(f"{meta['id']} {stage}: {reason}; subsequent stages were not started")


# ---------------------------------------------------------------------------
# 05_freq, method "displacements"
# ---------------------------------------------------------------------------


def _unit_ids(freq_dir):
    """IDs of the displacement units of a started 05_freq (from units.json); [] when none."""
    with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
        return [unit["id"] for unit in read_json(Path(freq_dir) / "units.json").get("units", [])]
    return []


def _unit_durations(units_dir):
    values = []
    if units_dir.is_dir():
        for path in units_dir.iterdir():
            with contextlib.suppress(OSError, ValueError, TypeError):
                record = read_json(path / "attempt.json")
                if record.get("status") == "complete" and record.get("elapsed_s"):
                    values.append(float(record["elapsed_s"]))
    return values


def _run_unit(run, freq_dir, unit, central, wavecar, durations):
    from .results import displaced_structure, displacement_error, unit_status
    from .structure import read_poscar, write_poscar

    unit_dir = freq_dir / "units" / unit["id"]
    label = f"{run.meta['id']} 05_freq unit {unit['id']}"
    delete_wavecar = run.config.get("frequency", {}).get("delete_wavecar_after_use", True)
    prune = unit_prune_files(run.config)

    def tidy_accepted():
        if delete_wavecar:
            with contextlib.suppress(FileNotFoundError):
                (unit_dir / "WAVECAR").unlink()
        return _prune_unit(unit_dir, prune)

    def evaluate(attempt):
        status = unit_status(unit_dir, central, unit)
        if attempt.get("status") == "failed" and attempt.get("returncode") not in (None, 0):
            return status["result"] or {}, False, f"Recorded launcher failure {attempt['returncode']}"
        return status["result"] or {}, status["done"], status["reason"]

    info = None
    if any((unit_dir / f).exists() for f in ("OUTCAR", "vasprun.xml", "stdout.log", "attempt.json")):
        try:
            action, info = _assess_existing(
                run, f"05_freq unit {unit['id']}", unit_dir, False, evaluate, output_names=UNIT_OUTPUT_HASH_FILES
            )
        except StageFailed as exc:
            # An earlier failure waiting for --retry (or an exhausted budget): skip this unit
            # without counting it against this job's failure allowance.
            raise _UnitHeld(str(exc)) from exc
        if action != "relaunch":
            tidy_accepted()
            return action
    run.stopper.check("displacement", max(durations) if durations else None, label=f"unit {unit['id']}")
    continuation = _archive_for_relaunch(run, unit_dir, False, info) if info is not None else None
    unit_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(freq_dir / "INCAR", unit_dir / "INCAR")
    shutil.copy2(freq_dir / "KPOINTS", unit_dir / "KPOINTS")
    _link_potential(unit_dir, (freq_dir / "POTCAR").resolve())
    displaced = displaced_structure(central, unit)
    write_poscar(unit_dir / "POSCAR", displaced)
    error = displacement_error(central, read_poscar(unit_dir / "POSCAR"), unit)
    if error:
        raise ValueError(f"{label}: displaced POSCAR failed verification: {error}")
    restart = _copy_restart(wavecar, unit_dir / "WAVECAR", run.source_hash(wavecar))
    with contextlib.suppress(FileNotFoundError):
        (unit_dir / "STOPCAR").unlink()
    write_json(
        unit_dir / "input_hashes.json",
        {
            "files": {f: file_hash(unit_dir / f) for f in ("INCAR", "POSCAR", "KPOINTS", "POTCAR")},
            "restart_files": {"WAVECAR": restart},
            "restart_wavecar": restart,
            "central_poscar_sha256": file_hash(freq_dir / "POSCAR"),
            "unit": unit,
            "state_json_sha256": file_hash(run.state_dir / "state.json"),
            "plan_sha256": run.plan["sha256"],
            "continuation": continuation or _pending_continuation(unit_dir),
            "started_at_utc": now(),
        },
    )
    if run.stopper.requested:
        raise TimeLimitReached(f"stop requested by {run.stopper.requested} before unit {unit['id']}")
    attempt = run.attempt_record()
    attempt["unit"] = unit
    write_json(unit_dir / "attempt.json", attempt)
    with contextlib.suppress(FileNotFoundError):
        (unit_dir / "continuation.json").unlink()
    returncode, launch_error, elapsed = run.launch(unit_dir, "single_point", attempt)
    attempt = dict(attempt)
    status_info = unit_status(unit_dir, central, unit)
    result = status_info["result"] or {}
    accepted, reason = status_info["done"], status_info["reason"]
    warnings = []
    evidence = result.get("restart_evidence", {})
    if evidence.get("required") and evidence.get("verified") is None and not evidence.get("fallback"):
        warnings.append("WAVECAR restart was provided, but the successful-read message was not recognized.")
    if launch_error is not None:
        status, reason = "launcher_error", f"Could not start the VASP launcher {run.command[0]!r}: {launch_error}"
    else:
        if returncode != 0 and accepted:
            accepted, reason = False, f"VASP launcher returned {returncode}"
        if accepted:
            status = "complete"
        else:
            status = _classify(
                False, result, unit_dir, attempt, live=True, stop_signal=run.stopper.requested, returncode=returncode
            )
            if status == "launcher_error":
                reason = f"VASP did not start (launcher returned {returncode}, no OUTCAR)"
    attempt.update(
        status=status,
        returncode=returncode,
        finished_at_utc=now(),
        elapsed_s=round(elapsed, 3),
        reason=reason,
        result=result,
        restart_evidence=evidence,
        warnings=warnings,
    )
    if status == "complete":
        attempt["output_hashes"] = _output_hashes(unit_dir, UNIT_OUTPUT_HASH_FILES)
        attempt["pruned_outputs"] = tidy_accepted()
    write_json(unit_dir / "attempt.json", attempt)
    if status == "complete":
        durations.append(elapsed)
        return "complete"
    if status == "interrupted":
        raise TimeLimitReached(f"{label} interrupted ({run.stopper.requested}); the next run continues it")
    if status == "launcher_error":
        raise LauncherError(f"{label}: {reason}. Fix the environment; the next run re-launches without --retry.")
    raise _UnitFailure(f"{label}: {reason}. Use --retry for a recorded same-protocol attempt.")


def _run_freq_displacements(run):
    from .results import compute_displacement_frequencies, displacement_units, parse_run
    from .structure import read_poscar

    state_dir, meta = run.state_dir, run.meta
    folder = state_dir / STAGE_FREQ
    spec = run.plan["stages"][STAGE_FREQ]
    unit_prune_files(run.config)  # reject an unsafe prune list before any unit runs
    driver = _read_attempt(folder)
    if driver.get("status") == "complete":
        _verify_input_hashes(folder)
        _verify_output_hashes(folder)
        result = parse_run(folder)
        if result["frequency_converged"]:
            return "already_complete"
        raise RuntimeError(
            f"{meta['id']} 05_freq is recorded complete but its displacement set no longer "
            f"verifies: {result['warnings'][:3]}"
        )
    wavecar = state_dir / STAGE_STATIC / "WAVECAR"
    if (folder / "input_hashes.json").exists():
        _verify_input_hashes(folder)
        record = read_json(folder / "input_hashes.json")
        static_poscar = state_dir / STAGE_STATIC / "POSCAR"
        if record.get("central_source_sha256") and file_hash(static_poscar) != record["central_source_sha256"]:
            raise ValueError(f"{meta['id']}: 04_static/POSCAR changed after the displacement set was started")
    else:
        run.stopper.check("displacement", None, label="05_freq set-up")
        gate = _static_force_gate(run)
        source = _stage_geometry(state_dir, STAGE_FREQ, meta, plan=run.plan)
        if not wavecar.is_file() or wavecar.stat().st_size == 0:
            raise FileNotFoundError(f"A nonempty {wavecar} is required to continue the electronic solution")
        write_json(
            folder / "input_hashes.json",
            {
                "files": {f: file_hash(folder / f) for f in ("INCAR", "POSCAR", "KPOINTS", "POTCAR")},
                "restart_files": {
                    "WAVECAR": {
                        "source": str(wavecar),
                        "bytes": wavecar.stat().st_size,
                        "sha256": run.source_hash(wavecar),
                    }
                },
                "central_source_sha256": file_hash(state_dir / STAGE_STATIC / "POSCAR"),
                "state_json_sha256": file_hash(state_dir / "state.json"),
                "plan_sha256": run.plan["sha256"],
                "geometry_source": source,
                "static_force_check": gate,
                "method": "displacements",
                "displacement_A": spec["displacement_A"],
                "started_at_utc": now(),
            },
        )
        for pending in (folder / "PENDING.json", folder / "BLOCKED.json"):
            with contextlib.suppress(FileNotFoundError):
                pending.unlink()
    central = read_poscar(folder / "POSCAR")
    units = displacement_units(central, spec["displacement_A"])
    plan_units = {"schema_version": 1, "displacement_A": spec["displacement_A"], "units": units}
    if (folder / "units.json").exists() and read_json(folder / "units.json") != plan_units:
        raise ValueError(f"{meta['id']}: 05_freq/units.json does not match the central geometry and mask")
    if not (folder / "units.json").exists():
        write_json(folder / "units.json", plan_units)
    driver = {
        "status": "running",
        "method": "displacements",
        "started_at_utc": driver.get("started_at_utc") or now(),
        "n_units": len(units),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    durations = _unit_durations(folder / "units")
    failures, held, completed = [], [], []
    max_failures = int(run.config.get("frequency", {}).get("max_unit_failures_per_job", 3))

    def progress(status, **extra):
        done = sum(1 for u in units if _read_attempt(folder / "units" / u["id"]).get("status") == "complete")
        driver.update(status=status, n_units_complete=done, updated_at_utc=now(), **extra)
        write_json(folder / "attempt.json", driver)

    progress("running")
    pending = [u["id"] for u in units if _read_attempt(folder / "units" / u["id"]).get("status") != "complete"]
    if pending and (not wavecar.is_file() or wavecar.stat().st_size == 0):
        progress("incomplete", error=f"{wavecar} missing")
        raise FileNotFoundError(f"A nonempty {wavecar} is required for the {len(pending)} remaining displacement units")
    try:
        for unit in units:
            try:
                if _run_unit(run, folder, unit, central, wavecar, durations) in ("complete", "finalized"):
                    completed.append(unit["id"])
            except _UnitHeld as exc:
                held.append(str(exc))
            except _UnitFailure as exc:
                failures.append(str(exc))
                if len(failures) >= max_failures:
                    break
    except TimeLimitReached:
        progress("incomplete", failures=failures, held=held, completed_this_run=len(completed))
        raise
    except BaseException as exc:
        progress("incomplete", failures=failures, held=held, completed_this_run=len(completed), error=str(exc))
        raise
    if failures or held:
        progress("failed_units", failures=failures, held=held, completed_this_run=len(completed))
        waiting = f" and {len(held)} earlier unit failure(s) wait for --retry" if held else ""
        raise StageFailed(
            f"{meta['id']} 05_freq: {len(failures)} displacement unit(s) failed in this run{waiting}; "
            "they need attention: " + " | ".join((failures + held)[:3])
        )
    report, _ = compute_displacement_frequencies(folder)
    if not report["complete"]:
        progress("incomplete", error="units accepted but Hessian incomplete")
        raise RuntimeError(f"{meta['id']} 05_freq: Hessian incomplete after all units: {report['warnings'][:3]}")
    gate = read_json(folder / "input_hashes.json").get("static_force_check") or {}
    report["static_force_check"] = gate
    if gate.get("warning"):
        report["warnings"].append(gate["warning"])
    write_json(folder / "frequencies.json", report)
    result = parse_run(folder)
    if not result["frequency_converged"]:
        progress("incomplete", error="frequency set did not verify")
        raise RuntimeError(f"{meta['id']} 05_freq: displacement frequencies did not verify: {result['warnings'][:3]}")
    driver.update(
        status="complete",
        finished_at_utc=now(),
        reason="accepted",
        output_hashes=_output_hashes(folder, DRIVER_OUTPUT_FILES),
        result={
            "frequencies_cm1": result["frequencies_cm1"],
            "n_displacements_done": result["n_displacements_done"],
            "mass_source": result["mass_source"],
            "warnings": result["warnings"],
        },
    )
    driver["n_units_complete"] = len(units)
    write_json(folder / "attempt.json", driver)
    return "complete"


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_state(state_dir, through="final", execute=False, retry=False):
    """Run selected state stages synchronously inside a compute allocation.

    Interrupted stages continue automatically; --retry archives and retries an
    unsuccessful stage using unchanged settings, at most vasp.max_attempts total.
    It never invents new convergence parameters.  A stop signal or an exhausted
    allocation ends the run with status "incomplete_time_limit" (not an error).
    """
    state_dir = Path(state_dir).resolve()
    campaign = find_campaign(state_dir)
    config = read_json(campaign / "config.json")
    through = resolve_through(through, config)
    stages = protocol_stages(config)
    wanted = stages[: stages.index(through) + 1]
    command = config.get("vasp", {}).get("command")
    if not command:
        raise ValueError(
            "No VASP launch command: set vasp_command in the cluster profile (e.g. "
            "[mpirun, -n, '128', vasp_std]) and regenerate or edit <campaign>/config.json"
        )
    if not isinstance(command, list) or not all(isinstance(v, str) for v in command):
        raise ValueError("vasp.command must be an explicit list of executable/arguments, not shell text")
    if execute:
        from .clusters import expand_command

        command = expand_command(command)
    if not execute:
        return {
            "state_id": state_dir.name,
            "execute": False,
            "stages": list(wanted),
            "command": command,
            "note": "No VASP process started; run inside a compute allocation with --execute.",
        }
    stopper = StopController(config)
    stopper.install()  # early: SIGUSR1's default action would terminate the runner
    if os.environ.get("SLURM_JOB_ID") and (campaign / "refill.json").is_file():
        with contextlib.suppress(Exception):
            from .slurm import note_job_started

            note_job_started(campaign, os.environ["SLURM_JOB_ID"])
    record = failure = None
    try:
        record = _run_locked(state_dir, campaign, config, through, wanted, retry, command, stopper)
        return record
    except BaseException as exc:
        failure = exc
        raise
    finally:
        # Top the campaign up to its job cap (docs/execution.md, "Job cap and refill"). The stop handlers stay
        # installed meanwhile, so a late SIGUSR1/SIGTERM cannot kill a finished run, and a
        # SIGTERM stops further submissions. Nothing here may replace the run's own result.
        with contextlib.suppress(Exception):
            if (campaign / "refill.json").is_file():
                from .slurm import refill_after_run

                refill_after_run(
                    campaign,
                    state_dir,
                    record,
                    failure,
                    stop_check=lambda: any(e.get("signal") == "SIGTERM" for e in stopper.events),
                )
        stopper.uninstall()


def _run_locked(state_dir, campaign, config, through, wanted, retry, command, stopper):
    with _state_lock(state_dir) as lock:
        prepare(campaign, [state_dir.name], with_potcars=True)
        meta = _state_validation(state_dir, config)
        plan = read_json(state_dir / "plan.json")
        record = {
            "state_id": meta["id"],
            "through": through,
            "started_at_utc": now(),
            "stages": [],
            "status": "running",
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "config_sha256": file_hash(campaign / "config.json"),
            "lock": lock["mode"],
            "allocation_end_utc": (
                dt.datetime.fromtimestamp(stopper.end_time, dt.timezone.utc).isoformat() if stopper.end_time else None
            ),
        }
        write_json(state_dir / "run.json", record)
        run = _Run(state_dir, campaign, config, plan, meta, retry, stopper, command)
        current = None
        try:
            for stage in wanted:
                current = stage
                if stage == STAGE_FREQ and plan["stages"][stage].get("method") == "displacements":
                    outcome = _run_freq_displacements(run)
                else:
                    outcome = _run_vasp_stage(run, stage)
                record["stages"].append({"stage": stage, "status": outcome})
                write_json(state_dir / "run.json", record)
            record.update(status="complete", finished_at_utc=now(), vasp_launches=run.launches, signals=stopper.events)
        except TimeLimitReached as exc:
            record.update(
                status="incomplete_time_limit",
                stopped_stage=current,
                reason=str(exc),
                finished_at_utc=now(),
                signals=stopper.events,
                vasp_launches=run.launches,
            )
        except BaseException as exc:
            kind = (
                "stage" if isinstance(exc, StageFailed) else "launcher" if isinstance(exc, LauncherError) else "other"
            )
            record.update(
                status="failed",
                failed_stage=current,
                reason=str(exc),
                finished_at_utc=now(),
                signals=stopper.events,
                vasp_launches=run.launches,
                failure_kind=kind,
            )
            write_json(state_dir / "run.json", record)
            raise
        write_json(state_dir / "run.json", record)
        return record


def _stage_complete(folder):
    return _read_attempt(folder).get("status") == "complete"


def _attempt_verdict(folder, label, retry, max_attempts, max_continuations):
    """(status, reason) when an existing attempt cannot continue without the user, else (None, None)."""
    attempt = _read_attempt(folder)
    status = attempt.get("status")
    if status not in ("failed", "interrupted"):
        return None, None  # absent, launcher_error, running (dead runner: the runner decides) or complete
    counts = _archive_counts(folder)
    if status == "interrupted":
        if counts["interrupted"] >= max_continuations:
            return "skipped_exhausted", (
                f"{label}: {counts['interrupted']} automatic continuations used "
                f"(vasp.max_continuations={max_continuations})"
            )
        return None, None
    if counts["failed"] + 1 >= max_attempts:
        return "skipped_exhausted", f"{label}: all vasp.max_attempts={max_attempts} attempts used"
    if not retry:
        return "skipped_needs_retry", f"{label} failed: {attempt.get('reason') or 'see attempt.json'}; use --retry"
    return None, None


def submission_check(state_dir, through, retry=False, config=None):
    """Would a job for this state make progress now?  Read-only mirror of the runner's rules.

    Nothing is run, archived, finalized or hashed. Returns a dict with ``action``
    ("submit", "skip" or "complete"), ``status`` (for a skip: skipped_needs_retry,
    skipped_exhausted or skipped_blocked), ``stage``, ``reason``, ``progress`` (the
    state has attempts in any stage) and ``rank`` (2 x index of the stage a job would
    run next, +1 when that stage has already started): `submit` runs the highest first.
    """
    state_dir = Path(state_dir)
    if config is None:
        config = read_json(find_campaign(state_dir) / "config.json")
    vasp = config.get("vasp", {})
    max_attempts = int(vasp.get("max_attempts", 2))
    max_continuations = int(vasp.get("max_continuations", 8))
    through = resolve_through(through, config)
    stages = protocol_stages(config)
    started = any((state_dir / stage / "attempt.json").is_file() for stage in STAGES)

    def answer(action, status, stage, reason=None, progress=started, begun=False):
        return {
            "action": action,
            "status": status,
            "stage": stage,
            "reason": reason,
            "progress": progress,
            "rank": 2 * STAGES.index(stage) + int(bool(begun)),
        }

    for stage in stages[: stages.index(through) + 1]:
        folder = state_dir / stage
        if _stage_complete(folder):
            continue
        if stage == STAGE_FREQ:
            plan = {}
            with contextlib.suppress(OSError, ValueError):
                plan = read_json(state_dir / "plan.json")
            blocked = folder / "BLOCKED.json"
            if blocked.is_file():
                info = {}
                with contextlib.suppress(OSError, ValueError):
                    info = read_json(blocked)
                # The recorded 04_static measurements against the current thresholds (raising one releases it).
                final = (plan_mode(plan) if plan else central_mode(config)) == "final"
                failures = _gate_failures(info, frequency_gates(config), final)
                if failures:
                    return answer("skip", "skipped_blocked", stage, "; ".join(failures))
            method = (plan.get("stages", {}).get(STAGE_FREQ) or {}).get(
                "method", config.get("frequency", {}).get("method", "displacements")
            )
            if method == "displacements":
                units = []
                with contextlib.suppress(OSError, ValueError):
                    units = read_json(folder / "units.json").get("units", [])
                runnable, waiting = 0, []
                for unit in units:
                    unit_dir = folder / "units" / unit["id"]
                    if _stage_complete(unit_dir):
                        continue
                    status, reason = _attempt_verdict(
                        unit_dir, f"unit {unit['id']}", retry, max_attempts, max_continuations
                    )
                    if status is None:
                        runnable += 1
                    else:
                        waiting.append((status, reason))
                if units and not runnable and waiting:
                    status = (
                        "skipped_needs_retry"
                        if any(s == "skipped_needs_retry" for s, _ in waiting)
                        else "skipped_exhausted"
                    )
                    return answer(
                        "skip",
                        status,
                        stage,
                        f"{len(waiting)} displacement unit(s) wait: " + "; ".join(r for _, r in waiting[:3]),
                        True,
                    )
                return answer(
                    "submit",
                    "runnable",
                    stage,
                    progress=started or bool(units),
                    begun=bool(units) or (folder / "attempt.json").is_file(),
                )
        status, reason = _attempt_verdict(folder, stage, retry, max_attempts, max_continuations)
        if status:
            return answer("skip", status, stage, reason, True)
        return answer(
            "submit",
            "runnable",
            stage,
            begun=(folder / "attempt.json").is_file() or (folder / "continuation.json").is_file(),
        )
    return answer("complete", "complete", through, progress=True)


def remaining_displacements(state_dir, config=None):
    """Displacement single points 05_freq still has to run (method "displacements"); None otherwise."""
    state_dir = Path(state_dir)
    if config is None:
        config = read_json(find_campaign(state_dir) / "config.json")
    folder = state_dir / STAGE_FREQ
    plan = {}
    with contextlib.suppress(OSError, ValueError):
        plan = read_json(state_dir / "plan.json")
    method = (plan.get("stages", {}).get(STAGE_FREQ) or {}).get(
        "method", config.get("frequency", {}).get("method", "displacements")
    )
    if method != "displacements":
        return None
    if _stage_complete(folder):
        return 0
    units = []
    with contextlib.suppress(OSError, ValueError):
        units = read_json(folder / "units.json").get("units", [])
    if units:
        return sum(1 for unit in units if not _stage_complete(folder / "units" / unit["id"]))
    meta = read_json(state_dir / "state.json")
    return 2 * sum(1 for row in meta["frequency_mobile"] for flag in row if flag)  # +h and -h per coordinate


def cleanup(campaign, selection=None, execute=False):
    """List (dry run) or delete WAVECAR/CHGCAR files that no pending downstream stage needs,
    and the bulky outputs of accepted displacement units (frequency.prune_unit_outputs)."""
    campaign = Path(campaign).resolve()
    config = read_json(campaign / "config.json")
    prune, mode = unit_prune_files(config), central_mode(config)
    report = {"execute": execute, "states": [], "skipped_running": [], "total_bytes": 0, "n_files": 0}
    for state_dir in state_directories(campaign, selection):
        try:
            with _state_lock(state_dir):
                files = _cleanup_candidates(state_dir, prune, mode)
                for item in files:
                    if execute:
                        with contextlib.suppress(FileNotFoundError):
                            Path(item["path"]).unlink()
                if files:
                    size = sum(item["bytes"] for item in files)
                    report["states"].append({"state_id": state_dir.name, "files": files, "bytes": size})
                    report["total_bytes"] += size
                    report["n_files"] += len(files)
        except LockBusy:
            report["skipped_running"].append(state_dir.name)
    return report


def _restart_consumers(mode):
    """{(stage, restart file): the stage that still needs it}, from the protocol's restart sources.

    A restart file is needed until the stage that reads it is complete; the copy inside the
    reading stage is needed until that stage itself is complete.
    """
    needed = {}
    for stage, source in stage_sources(mode)[1].items():
        if source:
            upstream, name = source.split("/")
            needed[(upstream, name)] = stage
            needed.setdefault((stage, name), stage)
    return needed


def _cleanup_candidates(state_dir, prune=(), mode="refine"):
    done = {stage: _stage_complete(state_dir / stage) for stage in STAGES}
    plan = {}
    with contextlib.suppress(OSError, ValueError):
        plan = read_json(state_dir / "plan.json")
    # refine: 02_final/CHGCAR until 03_refine, 03_refine/CHGCAR until 04_static; final: 02_final/CHGCAR
    # until 04_static; 04_static/WAVECAR until 05_freq. A prepared state follows its plan.json.
    needed_until = _restart_consumers(plan_mode(plan) if plan else mode)
    candidates, seen = [], set()

    def add(path, why):
        if path.is_file() and not path.is_symlink() and str(path) not in seen:
            seen.add(str(path))
            candidates.append({"path": str(path), "bytes": path.stat().st_size, "reason": why})

    for stage in STAGES:
        folder = state_dir / stage
        if not folder.is_dir():
            continue
        for name in RESTART_FILES:
            consumer = needed_until.get((stage, name))
            if consumer is None:
                add(folder / name, "no stage reads this file")
            elif done[consumer]:
                add(folder / name, f"{consumer} is complete")
        for archived in folder.glob("previous_attempts/*"):
            for name in RESTART_FILES:
                add(archived / name, "archived attempt")
        units = folder / "units"
        if stage == STAGE_FREQ and units.is_dir():
            static_wave = (state_dir / STAGE_STATIC / "WAVECAR").is_file()
            for unit_dir in sorted(units.iterdir()):
                accepted = _stage_complete(unit_dir)
                for name in RESTART_FILES:
                    if name == "WAVECAR" and not (accepted or static_wave):
                        continue
                    add(unit_dir / name, "unit complete" if accepted else "redundant copy of 04_static/WAVECAR")
                if accepted:
                    for name in prune:
                        add(unit_dir / name, "bulky output of an accepted unit (frequency.unit_prune_files)")
                for archived in unit_dir.glob("previous_attempts/*"):
                    for name in RESTART_FILES:
                        add(archived / name, "archived attempt")
    return candidates
