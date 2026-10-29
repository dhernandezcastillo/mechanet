"""Cluster profiles: how jobs are submitted and VASP is launched on one SLURM machine.

A profile is a small YAML file in ``~/.config/mechanet/clusters/<name>.yaml`` (the folder
follows ``$XDG_CONFIG_HOME``; ``$MECHANET_CONFIG_DIR`` overrides it). It is written once per
machine, usually by importing a job script that already works there::

    mechanet cluster import my_vasp_job.sh --name hpc1

and a project refers to it by name (``cluster: hpc1`` in project.yaml). Campaigns carry the
resolved profile in ``config.json`` (``scheduler`` and ``vasp.command``), so jobs keep running
even if the profile changes; ``mechanet cluster use NAME`` applies a new or edited profile
to an existing campaign.

Profile keys::

    mechanet_cluster: 1
    name: hpc1
    description: free text
    sbatch:                      # any sbatch long option, except those mechanet sets itself
      nodes: 1                   # (job-name, signal, chdir, output, error)
      ntasks: 112
      time: "23:59:00"           # required
      partition: standard
      account: my_project
    shell: /bin/bash             # job script interpreter (bash; e.g. "/bin/bash -l" for a login shell)
    setup:                       # shell lines run before VASP, verbatim and in this order
      - module load vasp/6.4.1
      - export OMP_NUM_THREADS=1
      - source ~/venvs/mechanet/bin/activate
    module_use: []               # optional shorthands, emitted before `setup`:
    modules: []                  #   `module use PATH` and `module load NAME ...`
    environment: {MKL_NUM_THREADS: "1"}   # exported after `setup`; values may use $VARIABLES
    vasp_command: [mpirun, -n, "$SLURM_NTASKS", vasp_std]   # $VAR, ${VAR}, ${VAR:-default}
    python: null                 # interpreter with mechanet on the compute nodes (default: the submitting one)
    max_active_jobs: 20          # job cap per campaign (submit --max-active overrides it)
    stop_signal_lead_s: 1800     # SIGUSR1 this long before the time limit -> STOPCAR
                                 #   (default: 1800 s, or a quarter of a walltime shorter than 2 h)
    min_remaining_s: {relaxation: 1800, single_point: 5400, displacement: 2700}
    refill_max_consecutive_failures: 3
    potcar_dir: /path/to/potpaw_PBE   # licensed PAW library on this machine (optional)

The VASP command is run by the Python runner inside the job, not by a shell: it may use
``$VAR``, ``${VAR}`` and ``${VAR:-default}`` of the job's environment (exported variables), but
no command substitution, pipes or redirections (the runner writes ``vasp.out`` itself).
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

from .fileio import object_hash, read_yaml, write_yaml

SCHEMA = 1
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
KNOWN_KEYS = {
    "mechanet_cluster",
    "name",
    "description",
    "sbatch",
    "shell",
    "module_use",
    "modules",
    "setup",
    "environment",
    "vasp_command",
    "python",
    "mpi_ranks",
    "max_active_jobs",
    "stop_signal_lead_s",
    "min_remaining_s",
    "refill_max_consecutive_failures",
    "potcar_dir",
    "imported_from",
    "ignored_lines",
}
# Set by mechanet for every job, or incompatible with its submission (sbatch --parsable).
RESERVED_SBATCH = frozenset({"job-name", "signal", "chdir", "output", "error", "parsable", "wrap", "wait", "test-only"})
# Options that identify a person or a single submission; never copied from an imported script.
PERSONAL_SBATCH = {
    "mail-user",
    "mail-type",
    "job-name",
    "output",
    "error",
    "chdir",
    "signal",
    "array",
    "dependency",
    "comment",
    "begin",
    "hold",
    "deadline",
    "kill-on-invalid-dep",
}
TIME_OPTIONS = {"time", "time-min", "begin", "deadline"}
DEFAULT_SHELL = "/bin/bash"
DEFAULT_STOP_LEAD_S = 1800
# Time a stage needs to be started (cluster profile: min_remaining_s), by kind of VASP run.
DEFAULT_MIN_REMAINING_S = {"relaxation": 1800.0, "single_point": 5400.0, "displacement": 2700.0}
SHORT_SBATCH = {
    "N": "nodes",
    "n": "ntasks",
    "t": "time",
    "p": "partition",
    "A": "account",
    "q": "qos",
    "C": "constraint",
    "c": "cpus-per-task",
    "J": "job-name",
    "o": "output",
    "e": "error",
    "D": "chdir",
    "w": "nodelist",
    "x": "exclude",
    "a": "array",
    "d": "dependency",
    "G": "gpus",
    "M": "clusters",
    "L": "licenses",
    "S": "core-spec",
    "s": "oversubscribe",
}
SBATCH_FLAGS = {
    "exclusive",
    "requeue",
    "no-requeue",
    "oversubscribe",
    "contiguous",
    "overcommit",
    "s",
    "hold",
    "wait",
    "test-only",
    "parsable",
    "kill-on-invalid-dep",
}
LAUNCHERS = ("mpirun", "mpiexec", "srun", "ibrun", "aprun", "mpiexec.hydra", "orterun", "jsrun")
# Lines of an imported script that prepare the job's environment; kept verbatim, in order.
SETUP_PATTERNS = (
    r"module(\s|$)",
    r"ml(\s|$)",
    r"source\s",
    r"\.\s",
    r"export\s",
    r"ulimit\s",
    r"unset\s",
    r"umask\s",
    r"eval\s",
    r"(conda|mamba|micromamba)\s",
    r"spack\s",
    r"set\s+[-+]",
)
WRAPPERS = ("time", "exec", "command", "nohup", "builtin")
VASP_RE = re.compile(r"(^|/)vasp[\w.+-]*$", re.IGNORECASE)
# Programs that a launcher may start in a job script before VASP (never the VASP launch line).
UTILITIES = {
    "mkdir",
    "cp",
    "mv",
    "rm",
    "ln",
    "cat",
    "echo",
    "printf",
    "tar",
    "ls",
    "cd",
    "touch",
    "chmod",
    "hostname",
    "sleep",
    "date",
    "rsync",
    "find",
    "sed",
    "awk",
    "grep",
    "env",
    "true",
    "false",
    "test",
}
SHELL_KEYWORDS = {"if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done", "case", "esac", "select"}
BLOCK_OPENERS = ("if", "for", "while", "until", "case", "select")
BLOCK_CLOSERS = ("fi", "done", "esac")
COUNT_OPTIONS = {
    "nodes",
    "ntasks",
    "ntasks-per-node",
    "ntasks-per-core",
    "ntasks-per-socket",
    "ntasks-per-gpu",
    "cpus-per-task",
    "cpus-per-gpu",
    "gpus",
    "gpus-per-node",
    "gpus-per-task",
    "threads-per-core",
}
# Variables a job has without any setup line.
RUNTIME_VARIABLE_PREFIXES = ("SLURM_", "SLURMD_", "PMI_", "PMIX_", "OMPI_", "I_MPI_")
STANDARD_VARIABLES = {"HOME", "USER", "LOGNAME", "PATH", "PWD", "HOSTNAME", "TMPDIR", "SHELL", "LANG"}
ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
REDIRECT_RE = re.compile(
    r"""(?:(?<=\s)|^)(?:\d+|&)?(?:>>|>\||>&|<&|<>|>|<)\s*(?:&?-|&?\d+(?=\s|$)|"[^"]*"|'[^']*'|[^\s;&|<>]+)"""
)
CONTROL_TOKENS = {"&&", "||", ";", "|", "&", "|&", ";;", "(", ")"}
# Expansions the runner performs in vasp_command (it runs no shell): $VAR, ${VAR}, ${VAR:-x}, ${VAR-x}.
COMMAND_VAR_RE = re.compile(r"\$(?:([A-Za-z_]\w*)|\{([A-Za-z_]\w*)(?:(:?-)((?:\$\{[^}]*\}|[^}])*))?\})")


def config_dir():
    """mechanet's per-user configuration folder."""
    if os.environ.get("MECHANET_CONFIG_DIR"):
        return Path(os.environ["MECHANET_CONFIG_DIR"]).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base).expanduser() / "mechanet"


def clusters_dir():
    return config_dir() / "clusters"


def profile_path(name):
    if not NAME_RE.match(str(name)):
        raise ValueError(f"Invalid cluster profile name {name!r} (letters, digits, '_', '.', '-')")
    return clusters_dir() / f"{name}.yaml"


def list_profiles():
    """Every profile in the config folder with its status: ok, invalid (with the problem) or unreadable."""
    folder = clusters_dir()
    out = []
    for path in sorted(folder.glob("*.yaml")) if folder.is_dir() else []:
        row = {"name": path.stem, "path": str(path)}
        try:
            raw = read_yaml(path) or {}
        except Exception as exc:  # noqa: BLE001 - one broken file must not hide the others
            out.append({**row, "status": "unreadable", "problem": _first_line(exc)})
            continue
        sbatch = raw.get("sbatch") if isinstance(raw, dict) and isinstance(raw.get("sbatch"), dict) else {}
        try:
            command = " ".join(_command_list(raw.get("vasp_command") or [])) if isinstance(raw, dict) else ""
        except (TypeError, ValueError):
            command = str(raw.get("vasp_command"))
        row.update(
            description=str(raw.get("description", "")) if isinstance(raw, dict) else "",
            walltime=sbatch.get("time"),
            vasp_command=command,
        )
        try:
            validate_profile(raw, where=f"cluster profile {path}")
            row["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - one broken file must not hide the others
            problem = str(exc).split(": ", 1)[-1] if isinstance(exc, ValueError) else _first_line(exc)
            row.update(status="invalid", problem=problem)
        out.append(row)
    return out


def _first_line(exc):
    text = str(exc).splitlines()[0] if str(exc) else ""
    return text if isinstance(exc, ValueError) else f"{type(exc).__name__}: {text}"


def _command_list(value):
    if isinstance(value, str):
        return shlex.split(value)
    return [str(v) for v in value]


def walltime_seconds(value):
    """Seconds of a SLURM --time value (MM, MM:SS, HH:MM:SS, D-HH, D-HH:MM, D-HH:MM:SS); None if unlimited."""
    text = str(value).strip()
    if text.upper() in ("UNLIMITED", "INFINITE", "-1", "0"):
        return None  # SLURM: a time limit of 0 means none
    match = re.fullmatch(r"(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?", text)
    if not match:
        raise ValueError(
            f"walltime {value!r} is not a SLURM time (minutes, MM:SS, HH:MM:SS, D-HH, D-HH:MM or D-HH:MM:SS)"
        )
    days, first, second, third = (int(g) if g is not None else None for g in match.groups())
    if days is not None:
        return days * 86400 + first * 3600 + (second or 0) * 60 + (third or 0)
    if second is None:
        return first * 60
    if third is None:
        return first * 60 + second
    return first * 3600 + second * 60 + third


def _scheduler_time(scheduler):
    return (scheduler.get("sbatch") or {}).get("time") or scheduler.get("walltime")


def stop_lead(scheduler):
    """Seconds between the stop signal (SIGUSR1) and the time limit of a job."""
    if scheduler.get("stop_signal_lead_s"):
        return int(scheduler["stop_signal_lead_s"])
    try:
        walltime = walltime_seconds(_scheduler_time(scheduler)) if _scheduler_time(scheduler) else None
    except ValueError:
        walltime = None
    if walltime:
        return max(60, min(DEFAULT_STOP_LEAD_S, walltime // 4))
    return DEFAULT_STOP_LEAD_S


def _duration(seconds):
    seconds = int(seconds)
    if seconds % 3600 == 0:
        return f"{seconds // 3600} h"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds // 60} min" if seconds % 60 == 0 else f"{seconds} s"


def timing_report(scheduler):
    """(errors, warnings) about the walltime, the stop signal and the minimum times of stages."""
    errors, warnings = [], []
    raw = _scheduler_time(scheduler)
    if raw is None:
        return errors, warnings
    try:
        walltime = walltime_seconds(raw)
    except ValueError as exc:
        return [str(exc)], warnings
    if walltime is None:
        return errors, warnings
    lead = stop_lead(scheduler)
    if lead >= walltime:
        errors.append(
            f"stop_signal_lead_s ({lead} s) is not shorter than the walltime ({_duration(walltime)}): every job "
            "would be told to stop as it starts; lower stop_signal_lead_s or raise sbatch.time"
        )
        return errors, warnings
    if lead > walltime / 2:
        warnings.append(
            f"the stop signal comes {lead} s before the end of a {_duration(walltime)} job, so relaxations "
            "get less than half of each job; consider a lower stop_signal_lead_s"
        )
    minimum = dict(DEFAULT_MIN_REMAINING_S)
    configured = scheduler.get("min_remaining_s")
    if isinstance(configured, dict):
        minimum.update({k: float(v) for k, v in configured.items() if k in minimum and _is_number(v)})
    long = [kind for kind, value in minimum.items() if value > 0.9 * walltime]
    if long:
        warnings.append(
            f"with a walltime of {_duration(walltime)}, {', '.join(long)} stages (min_remaining_s: "
            + ", ".join(f"{minimum[k]:.0f} s" for k in long)
            + ") start only at the beginning of a job, one per job; a single point must finish within one job"
        )
    return errors, warnings


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _valid_shell(text):
    words = str(text).split()
    if not words or not words[0].startswith("/"):
        return False
    program = words[0].rsplit("/", 1)[-1]
    return program == "bash" or (program == "env" and "bash" in words[1:])


def command_problems(command):
    """Problems of a vasp_command (the runner executes it without a shell)."""
    problems = []
    for arg in command:
        text = str(arg)
        if "$(" in text or "`" in text:
            problems.append(f"{text!r} uses command substitution; use an exported variable or a literal value")
            continue
        rest = COMMAND_VAR_RE.sub("", text)
        if "$" in rest:
            problems.append(f"{text!r}: only $VAR, ${{VAR}} and ${{VAR:-default}} are expanded")
        if text in CONTROL_TOKENS or re.fullmatch(r"\d*(>>?|<|>&|&>)\S*", text):
            problems.append(f"{text!r} is a shell operator or redirection; the runner runs no shell")
    return problems


def validate_profile(profile, where="cluster profile"):
    """Normalized copy of a profile; ValueError lists every problem at once."""
    if not isinstance(profile, dict):
        raise ValueError(f"The {where} must be a mapping")
    errors = []
    expected = {
        "sbatch": dict,
        "environment": dict,
        "min_remaining_s": dict,
        "vasp_command": (str, list),
        "setup": (str, list),
        "module_use": (str, list),
        "modules": (str, list),
        "shell": str,
        "python": str,
        "potcar_dir": str,
        "description": str,
        "ignored_lines": list,
    }
    names = {dict: "a mapping", list: "a list", str: "text"}
    wrong = [
        f"{key} must be " + " or ".join(names[t] for t in (kinds if isinstance(kinds, tuple) else (kinds,)))
        for key, kinds in expected.items()
        if profile.get(key) is not None and not isinstance(profile[key], kinds)
    ]
    if wrong:
        raise ValueError(f"Invalid {where}: " + "; ".join(wrong))
    if isinstance(profile.get("vasp_command"), list) and not all(
        isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in profile["vasp_command"]
    ):
        raise ValueError(f"Invalid {where}: vasp_command entries must be words (text or numbers)")
    unknown = sorted(set(profile) - KNOWN_KEYS)
    if unknown:
        errors.append(f"unknown keys {unknown}")
    out = {k: v for k, v in profile.items() if k in KNOWN_KEYS}
    out["mechanet_cluster"] = int(profile.get("mechanet_cluster", SCHEMA))
    sbatch = {}
    for key, value in (profile.get("sbatch") or {}).items():
        name = str(key).lstrip("-")
        if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
            errors.append(f"invalid sbatch option {key!r}")
        elif name in RESERVED_SBATCH:
            errors.append(f"sbatch option --{name} is set by mechanet for every job; remove it")
        else:
            sbatch[name] = value
    if "time" not in sbatch:
        errors.append("sbatch.time (the walltime, e.g. '24:00:00') is required")
    elif _is_number(sbatch["time"]):
        errors.append(
            "sbatch.time must be a quoted string such as '23:59:00' (YAML reads an unquoted 23:59:00 "
            "as a number of seconds)"
        )
    else:
        try:
            walltime_seconds(sbatch["time"])
        except ValueError as exc:
            errors.append(f"sbatch.time: {exc}")
    out["sbatch"] = sbatch
    if profile.get("shell") is not None and not _valid_shell(profile["shell"]):
        errors.append(f"shell {profile['shell']!r} must run bash, e.g. /bin/bash or '/bin/bash -l'")
    for key in ("module_use", "modules", "setup"):
        value = profile.get(key)
        if value is None:
            out[key] = []
        elif isinstance(value, str):
            out[key] = value.splitlines() if key == "setup" else [value]
        elif isinstance(value, list):
            out[key] = [str(v) for v in value]
        else:
            errors.append(f"{key} must be a list")
    command = profile.get("vasp_command")
    if not command:
        errors.append("vasp_command is required, e.g. [srun, vasp_std] or 'mpirun -n $SLURM_NTASKS vasp_std'")
    else:
        try:
            out["vasp_command"] = _command_list(command)
            errors += [f"vasp_command: {p}" for p in command_problems(out["vasp_command"])]
        except ValueError as exc:
            errors.append(f"vasp_command: {exc}")
    minimum = profile.get("min_remaining_s")
    if minimum is not None:
        if not isinstance(minimum, dict):
            errors.append(
                "min_remaining_s must be a mapping such as {relaxation: 1800, single_point: 5400, displacement: 2700}"
            )
        else:
            unknown_kinds = sorted(set(minimum) - set(DEFAULT_MIN_REMAINING_S))
            if unknown_kinds:
                errors.append(
                    f"min_remaining_s: unknown kinds {unknown_kinds} (known: {', '.join(DEFAULT_MIN_REMAINING_S)})"
                )
            bad = [k for k, v in minimum.items() if not _is_number(v) or v < 0]
            if bad:
                errors.append(f"min_remaining_s: {bad} must be numbers of seconds >= 0")
    env = profile.get("environment")
    if env is not None and not isinstance(env, dict):
        errors.append("environment must be a mapping of variable -> value")
    elif env is not None:
        bad = [k for k in env if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(k))]
        if bad:
            errors.append(f"invalid environment variable names {bad}")
        out["environment"] = {str(k): str(v) for k, v in env.items()}
    for key in ("max_active_jobs", "stop_signal_lead_s", "mpi_ranks", "refill_max_consecutive_failures"):
        if profile.get(key) is not None:
            try:
                value = int(profile[key])
                if value <= 0:
                    raise ValueError
                out[key] = value
            except (TypeError, ValueError):
                errors.append(f"{key} must be a positive integer")
                out.pop(key, None)
    if out.get("stop_signal_lead_s", 1) > 65535:
        errors.append("stop_signal_lead_s must be at most 65535 s")
    for key, value in out.items():
        for text in value if isinstance(value, list) else [value] if isinstance(value, str) else []:
            if "\n" in str(text) and key != "description":
                errors.append(f"{key}: newlines are not allowed inside one entry")
    if not errors:
        timing_errors, timing_warnings = timing_report(out)
        errors += timing_errors
        out["_warnings"] = timing_warnings
    if errors:
        raise ValueError(f"Invalid {where}: " + "; ".join(errors))
    return out


def load_profile(name_or_path):
    """A profile by name (from the config folder) or by file path."""
    text = str(name_or_path)
    path = Path(text).expanduser()
    if "/" in text or path.suffix in (".yaml", ".yml"):
        if not path.is_file():
            raise FileNotFoundError(f"Cluster profile file {path} not found")
    else:
        path = profile_path(text)
        if not path.is_file():
            rows = list_profiles()
            known = [p["name"] for p in rows if p.get("status") == "ok"]
            broken = len(rows) - len(known)
            raise FileNotFoundError(
                f"No cluster profile {text!r} in {clusters_dir()} (usable: {', '.join(known) or 'none'}"
                + (f"; {broken} invalid, see `mechanet cluster list`" if broken else "")
                + "); create one with `mechanet cluster import JOB_SCRIPT --name NAME`"
            )
    profile = validate_profile(read_yaml(path) or {}, where=f"cluster profile {path}")
    profile.setdefault("name", path.stem)
    profile["_path"] = str(path)
    return profile


def save_profile(profile, name, overwrite=False):
    path = profile_path(name)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Cluster profile {path} exists; pass --force to replace it")
    clean = validate_profile({**profile, "name": name})
    order = [
        "mechanet_cluster",
        "name",
        "description",
        "sbatch",
        "shell",
        "module_use",
        "modules",
        "setup",
        "environment",
        "vasp_command",
        "python",
        "mpi_ranks",
        "max_active_jobs",
        "stop_signal_lead_s",
        "min_remaining_s",
        "refill_max_consecutive_failures",
        "potcar_dir",
        "imported_from",
        "ignored_lines",
    ]
    clean = {k: clean[k] for k in order if k in clean}
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# mechanet cluster profile. Edit freely; `mechanet cluster show NAME` checks it and prints\n"
        "# the job script it produces. `setup` lines run verbatim before VASP; vasp_command runs\n"
        "# inside the job without a shell, with $VAR, ${VAR} and ${VAR:-default} expanded there.\n"
    )
    write_yaml(path, {k: v for k, v in clean.items() if v not in (None, [], {})}, header=header)
    return path


# ---------------------------------------------------------------------------
# Importing an existing job script
# ---------------------------------------------------------------------------


def _sbatch_tokens(directive):
    """(name, value) pairs of one '#SBATCH ...' line (a ' # ...' tail is a comment)."""
    try:
        tokens = shlex.split(directive, comments=False)
    except ValueError:
        # An unbalanced quote in a trailing comment ("# don't change"): drop the comment.
        tokens = shlex.split(re.split(r"\s#", directive, maxsplit=1)[0], comments=False)
    out, i = [], 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("#"):
            break
        if token.startswith("--"):
            name, eq, value = token[2:].partition("=")
            if not eq:
                if name in SBATCH_FLAGS or i + 1 >= len(tokens) or tokens[i + 1].startswith("-"):
                    value = True
                else:
                    i += 1
                    value = tokens[i]
            out.append((name, value))
        elif token.startswith("-") and len(token) >= 2:
            key = token[1]
            name = SHORT_SBATCH.get(key, key)
            value = token[2:]
            if not value:
                if name in SBATCH_FLAGS or i + 1 >= len(tokens) or tokens[i + 1].startswith("-"):
                    value = True
                else:
                    i += 1
                    value = tokens[i]
            out.append((name, value.lstrip("=") if isinstance(value, str) else value))
        i += 1
    return out


def _option_value(name, value):
    """Counts become integers (nodes: 2); times, accounts and everything else stay strings."""
    if name in COUNT_OPTIONS and isinstance(value, str) and re.fullmatch(r"[1-9]\d*", value):
        return int(value)
    return value


def _logical_lines(text):
    """(first line number, text) of a script, with backslash continuations joined."""
    out, buffer, start = [], "", None
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        start = number if start is None else start
        if line.endswith("\\") and not line.lstrip().startswith("#"):
            buffer += line[:-1] + " "
            continue
        out.append((start, buffer + line))
        buffer, start = "", None
    if buffer:
        out.append((start, buffer))
    return out


def _resolve(word, variables):
    return re.sub(r"\$\{?([A-Za-z_]\w*)\}?", lambda m: variables.get(m.group(1), m.group(0)), word)


def _tokens(text):
    lexer = shlex.shlex(text, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    return list(lexer)


def _probe(line):
    """The line with command substitutions replaced by one opaque word (for classification only)."""
    return re.sub(r"\$\([^)]*\)|`[^`]*`", "SUBSTITUTION", line)


def _split_command(line):
    """(words, cut-off parts) of a command line: redirections are removed and the line ends at the
    first control operator (&&, ||, ;, |, &). Words are None if the line cannot be parsed."""
    dropped = []
    text = REDIRECT_RE.sub(lambda m: dropped.append(m.group(0).strip()) or " ", line.strip())
    try:
        tokens = _tokens(text)
    except ValueError:
        return None, dropped
    for index, token in enumerate(tokens):
        if token in CONTROL_TOKENS or set(token) <= set("();<>|&"):
            dropped.append(" ".join(tokens[index:]))
            tokens = tokens[:index]
            break
    return tokens, dropped


def _segments(line):
    """The simple commands of a line (words between control operators, shell keywords left out)."""
    try:
        tokens = _tokens(REDIRECT_RE.sub(" ", _probe(line)))
    except ValueError:
        return []
    segments, current = [], []
    for token in tokens:
        if token in CONTROL_TOKENS or set(token) <= set("();<>|&"):
            if current:
                segments.append(current)
            current = []
        elif current or token not in SHELL_KEYWORDS:
            current.append(token)
    if current:
        segments.append(current)
    return segments


def _block_change(line):
    """Net number of shell compound commands (if/for/while/until/case) a line opens."""
    change, command_position = 0, True
    for words in _segments_with_keywords(line):
        for token in words:
            if command_position and token in BLOCK_OPENERS:
                change += 1
            elif command_position and token in BLOCK_CLOSERS:
                change -= 1
            command_position = token in ("then", "do", "else", "elif", "in", "!", "{")
        command_position = True
    return change


def _segments_with_keywords(line):
    try:
        tokens = _tokens(REDIRECT_RE.sub(" ", _probe(line)))
    except ValueError:
        return [line.split()]
    segments, current = [], []
    for token in tokens:
        if token in CONTROL_TOKENS or set(token) <= set("();<>|&"):
            segments.append(current)
            current = []
        else:
            current.append(token)
    segments.append(current)
    return segments


def _units(lines):
    """Lines and compound blocks (if ... fi, for ... done, ...) of a script, in order."""
    units, block, depth = [], [], 0
    for number, line in lines:
        text = line.strip()
        if not block and (not text or text.startswith("#") or (text.split()[0] not in BLOCK_OPENERS)):
            units.append(("line", [(number, line)]))
            continue
        block.append((number, line))
        if text and not text.startswith("#"):
            depth += _block_change(text)
        if depth <= 0:
            units.append(("block", block))
            block, depth = [], 0
    if block:
        units.append(("block", block))
    return units


def _launch_command(words, variables):
    """(VASP command, leading VAR=value assignments, dropped wrappers) if these words launch VASP.

    The binary may be named directly (vasp_std, /opt/vasp/bin/vasp_gam) or through a
    variable set earlier in the script ($VASP_BIN), or a variable whose name says vasp. A
    launcher that starts a utility (srun mkdir /scratch/vasp_tmp) is not a launch line.
    """
    words, assignments, wrappers = list(words), [], []
    while words:
        if words[0] in WRAPPERS:
            wrappers.append(words.pop(0))
        elif words[0] == "timeout":
            dropped = [words.pop(0)]
            while words and words[0].startswith("-"):
                option = words.pop(0)
                dropped.append(option)
                if option in ("-s", "-k") and words:
                    dropped.append(words.pop(0))
            if words:
                dropped.append(words.pop(0))  # the duration
            wrappers.append(" ".join(dropped))
        elif ASSIGNMENT_RE.match(words[0]):
            assignments.append(words.pop(0))
        else:
            break
    if not words:
        return None

    def is_vasp(word):
        return any(VASP_RE.search(part) for part in _resolve(word, variables).split()) or bool(
            re.fullmatch(r"\$\{?\w*vasp\w*\}?", word, re.IGNORECASE)
        )

    if not any(is_vasp(w) for w in words):
        return None
    first = _resolve(words[0], variables).split() or [words[0]]
    if first[0].rsplit("/", 1)[-1] in LAUNCHERS:
        # the program the launcher starts: the first word that is no option, number, assignment
        # or variable (an option's value such as -n $SLURM_NNODES)
        rest = [
            w
            for w in first[1:] + words[1:]
            if not w.startswith(("-", "$")) and not re.fullmatch(r"\d+", w) and not ASSIGNMENT_RE.match(w)
        ]
        if rest and rest[0].rsplit("/", 1)[-1] in UTILITIES:
            return None
        return words, assignments, wrappers
    if is_vasp(words[0]):
        return words, assignments, wrappers
    return None


def _skip_reason(name):
    if name in ("job-name", "output", "error", "chdir", "signal"):
        return "set by mechanet for each job"
    if name in ("mail-user", "mail-type", "comment"):
        return "personal"
    return "per submission"


def _is_setup(line):
    """Does a line prepare the job's environment (kept in the profile's setup)?"""
    text = line.strip()
    if any(re.match(pattern, text) for pattern in SETUP_PATTERNS):
        return True
    return any(segment and all(ASSIGNMENT_RE.match(w) for w in segment) for segment in _segments(text)) or any(
        segment and any(re.match(pattern, " ".join(segment) + " ") for pattern in SETUP_PATTERNS)
        for segment in _segments(text)
    )


def _remember(words, variables):
    for word in words:
        if ASSIGNMENT_RE.match(word):
            name, _, value = word.partition("=")
            variables[name] = value


def parse_job_script(text):
    """Profile fields from an sbatch script that runs VASP, the lines left out, and notes on changes."""
    state = {
        "sbatch": {},
        "setup": [],
        "ignored": [],
        "notes": [],
        "variables": {},
        "vasp_command": None,
        "launch_line": None,
    }
    setup, ignored, notes, variables = state["setup"], state["ignored"], state["notes"], state["variables"]
    shell = None
    lines = _logical_lines(text)
    if lines and lines[0][1].startswith("#!"):
        interpreter = lines[0][1][2:].strip()
        if not _valid_shell(interpreter):
            notes.append(
                f"the script ran under {interpreter!r}; mechanet job scripts run {DEFAULT_SHELL}, "
                "so check that the setup lines are bash"
            )
        elif interpreter != DEFAULT_SHELL:
            shell = interpreter
    for kind, unit in _units(lines):
        if kind == "block":
            first, last = unit[0][0], unit[-1][0]
            span = f"line {first}" if first == last else f"lines {first}-{last}"
            for _, line in unit:
                for segment in _segments(line):
                    if _launch_command(segment, variables):
                        raise ValueError(
                            f"{span}: the VASP launch line {line.strip()!r} is inside {_a_block(unit)}; mechanet "
                            "runs one launch command, so write the one you want as a plain line and import the "
                            "script again"
                        )
            body = [line.rstrip() for _, line in unit if line.strip()]
            if any(_is_setup(line) for line in body):
                setup.extend(body)
                for line in body:
                    for segment in _segments(line):
                        _remember(segment[1:] if segment[:1] == ["export"] else segment, variables)
                notes.append(f"{span}: {_a_block(unit)} with environment lines was kept as written")
            else:
                ignored.extend(line.strip() for line in body)
            continue
        number, raw = unit[0]
        text = raw.strip()
        for line in [text] if text.startswith("#") else _split_semicolons(text):
            _parse_line(number, line, state)
    sbatch, vasp_command, launch_line = state["sbatch"], state["vasp_command"], state["launch_line"]
    if vasp_command:
        vasp_command = _split_known_words(vasp_command, variables, launch_line, notes)
        _note_unset_variables(vasp_command, variables, ignored, notes)
    profile = {
        "mechanet_cluster": SCHEMA,
        "sbatch": sbatch,
        "shell": shell,
        "setup": setup,
        "vasp_command": vasp_command,
    }
    return profile, ignored, notes


def _a_block(unit):
    word = unit[0][1].split()[0]
    return f"{'an' if word[0] in 'aeiou' else 'a'} `{word}` block"


def _split_semicolons(line):
    """The commands of a line separated by ';' outside quotes and $( ) (';;' of case blocks stays)."""
    parts, current, quote, depth, i = [], "", None, 0, 0
    while i < len(line):
        char = line[i]
        if quote:
            current += char
            if char == "\\" and quote == '"' and i + 1 < len(line):
                current += line[i + 1]
                i += 1
            elif char == quote:
                quote = None
        elif char in "'\"":
            quote, current = char, current + char
        elif char == "\\" and i + 1 < len(line):
            current += char + line[i + 1]
            i += 1
        elif char == "#" and (not current or current[-1].isspace()):
            current += line[i:]
            break
        elif char == "(":
            depth, current = depth + 1, current + char
        elif char == ")":
            depth, current = max(0, depth - 1), current + char
        elif char == ";" and depth == 0 and line[i + 1 : i + 2] != ";" and line[i - 1 : i] != ";":
            parts.append(current)
            current = ""
        else:
            current += char
        i += 1
    parts.append(current)
    return [part.strip() for part in parts if part.strip()]


def _parse_line(number, line, state):
    """Classify one command of a job script: an #SBATCH line, the VASP launch line, a setup line."""
    setup, ignored, notes, variables = state["setup"], state["ignored"], state["notes"], state["variables"]
    if not line or line.startswith("#!"):
        return
    if line.startswith("#SBATCH"):
        try:
            pairs = _sbatch_tokens(line[len("#SBATCH") :])
        except ValueError as exc:
            raise ValueError(f"line {number}: cannot read {line!r} ({exc})") from None
        for name, value in pairs:
            if name in PERSONAL_SBATCH or name in RESERVED_SBATCH:
                ignored.append(f"#SBATCH --{name}  ({_skip_reason(name)})")
            else:
                state["sbatch"][name] = _option_value(name, value)
        return
    if line.startswith("#"):
        return
    # Command substitutions are classified as one opaque word (and kept verbatim in setup lines).
    probe = _probe(line)
    words, dropped = _split_command(probe)
    launch = _launch_command(words, variables) if words else None
    if launch is not None:
        if probe != line:
            raise ValueError(
                f"line {number}: the VASP launch line {line!r} uses command substitution, which the "
                "runner cannot evaluate (it runs no shell); set an exported variable before it instead"
            )
        if state["vasp_command"] is not None:
            ignored.append(f"{line}  (second VASP launch line)")
            return
        command, assignments, wrappers = launch
        state["vasp_command"], state["launch_line"] = command, number
        setup.extend(f"export {assignment}" for assignment in assignments)
        left_out = [w for w in wrappers if w not in ("exec", "command", "builtin")] + dropped
        if left_out:
            notes.append(
                f"line {number}: left out {', '.join(repr(d) for d in left_out)} of the VASP launch line "
                "(the runner writes vasp.out itself, runs no shell, and the job's time limit bounds VASP)"
            )
        return
    if words and not dropped and all(ASSIGNMENT_RE.match(w) for w in words):
        # Plain assignments are exported, so that the runner and vasp_command see them.
        _remember(words, variables)
        setup.append(f"export {line}")
        return
    if any(re.match(pattern, line) for pattern in SETUP_PATTERNS):
        if words and words[0] == "export":
            _remember(words[1:], variables)
        setup.append(line)
        return
    ignored.append(line)


def _split_known_words(command, variables, number, notes):
    """Replace a bare $VAR whose value (set in the script) has several words by those words, as the
    shell's word splitting does (VASP_CMD="mpirun -np 4 vasp_std"; $VASP_CMD)."""
    out = []
    for word in command:
        match = re.fullmatch(r"\$\{?([A-Za-z_]\w*)\}?", word)
        value = variables.get(match.group(1)) if match else None
        if value is not None and len(value.split()) > 1:
            out += value.split()
            notes.append(f"line {number}: {word} stands for {value!r} in the VASP command")
        else:
            out.append(word)
    return out


def _note_unset_variables(command, variables, ignored, notes):
    """Warn about $VARIABLES of the VASP command that no kept line of the script sets."""
    used = [m for m in COMMAND_VAR_RE.finditer(" ".join(command)) if m.group(3) is None]  # no default
    for name in dict.fromkeys(m.group(1) or m.group(2) for m in used):
        if name in variables or name in STANDARD_VARIABLES or name.startswith(RUNTIME_VARIABLE_PREFIXES):
            continue
        setter = next((line for line in ignored if re.search(rf"(^|[\s;&|]){re.escape(name)}=", line)), None)
        where = f"the line that sets it was not copied ({setter!r})" if setter else "nothing in the script sets it"
        notes.append(
            f"warning: the VASP command uses ${name}, but {where}; add a setup line that exports it, or the "
            "job fails at launch unless a module sets it"
        )


def profile_from_job_script(path, description=None):
    """(profile, notes) from a working job script, checked like a saved profile; nothing is written."""
    path = Path(path)
    profile, ignored, notes = parse_job_script(path.read_text())
    if not profile["vasp_command"]:
        raise ValueError(
            f"No VASP launch line found in {path}: mechanet looks for a line that runs a VASP binary "
            "(vasp_std, vasp_gam, ..., or $VASP_BIN set earlier in the script), possibly through "
            "mpirun/srun; make that line explicit in the script and import it again"
        )
    if "time" not in profile["sbatch"]:
        raise ValueError(f"{path} sets no walltime (#SBATCH --time); mechanet needs it to stop VASP cleanly")
    profile["description"] = description or f"Imported from {path.name}"
    profile["imported_from"] = path.name
    if ignored:
        profile["ignored_lines"] = ignored
        notes.append(
            f"{len(ignored)} line(s) were not copied (ignored_lines in the profile): "
            + "; ".join(ignored[:3])
            + (" ..." if len(ignored) > 3 else "")
        )
    if not profile["setup"]:
        notes.append("No module or environment lines found; add `setup:` lines if the job needs them.")
    notes.append(
        "Jobs run mechanet with the Python that runs `mechanet submit` (by its absolute path) unless the "
        "profile sets `python:`; it must be reachable from the compute nodes."
    )
    checked = validate_profile({k: v for k, v in profile.items() if v is not None}, where=f"profile from {path}")
    return profile, notes + [f"warning: {w}" for w in checked.get("_warnings", [])]


def import_job_script(path, name, overwrite=False, description=None):
    """Write a profile from a working job script; returns (profile path, profile, notes)."""
    if profile_path(name).exists() and not overwrite:
        raise FileExistsError(f"Cluster profile {profile_path(name)} exists; pass --force to replace it")
    profile, notes = profile_from_job_script(path, description)
    written = save_profile(profile, name, overwrite)
    return written, load_profile(written), notes


# ---------------------------------------------------------------------------
# Applying a profile to a campaign configuration
# ---------------------------------------------------------------------------


def scheduler_section(profile):
    """The ``scheduler`` section of a campaign config.json for this profile."""
    section = {
        "profile": profile.get("name"),
        "profile_sha256": object_hash({k: v for k, v in profile.items() if not k.startswith("_")}),
        "sbatch": dict(profile["sbatch"]),
        "module_use": list(profile.get("module_use") or []),
        "modules": list(profile.get("modules") or []),
        "setup": list(profile.get("setup") or []),
    }
    for key in (
        "shell",
        "environment",
        "python",
        "mpi_ranks",
        "max_active_jobs",
        "stop_signal_lead_s",
        "min_remaining_s",
        "refill_max_consecutive_failures",
    ):
        if profile.get(key) is not None:
            section[key] = profile[key]
    return section


def apply_profile(config, profile, keep_command=False):
    """Set ``scheduler`` and ``vasp.command`` (and ``vasp.potcar_dir`` if unset) from a profile.

    With ``keep_command`` an existing ``vasp.command`` (set in project.yaml) is kept.
    """
    previous = config.get("scheduler") or {}
    config["scheduler"] = {
        **{k: v for k, v in previous.items() if k == "min_remaining_s"},
        **scheduler_section(profile),
    }
    vasp = config.setdefault("vasp", {})
    if not (keep_command and vasp.get("command")):
        vasp["command"] = list(profile["vasp_command"])
    if profile.get("potcar_dir") and not vasp.get("potcars") and not vasp.get("potcar_dir"):
        vasp["potcar_dir"] = profile["potcar_dir"]
    return config


def expand_command(command, environ=None):
    """Expand $VAR, ${VAR} and ${VAR:-default} in a VASP command inside the job.

    Unset variables are an error. An argument that is a single variable reference is split
    into words like an unquoted shell word ($VASP_CMD = "mpirun -np 4 vasp_std"); an argument
    that expands to nothing is dropped.
    """
    environ = os.environ if environ is None else environ
    out = []
    for arg in command:
        text = str(arg)

        def value(match, arg=text):
            name = match.group(1) or match.group(2)
            current = environ.get(name)
            if match.group(3) == ":-" and not current:
                return COMMAND_VAR_RE.sub(value, match.group(4))
            if match.group(3) == "-" and current is None:
                return COMMAND_VAR_RE.sub(value, match.group(4))
            if current is None:
                raise ValueError(f"vasp.command argument {arg!r} refers to the unset environment variable {name}")
            return current

        expanded = COMMAND_VAR_RE.sub(value, text)
        if COMMAND_VAR_RE.fullmatch(text):
            out += expanded.split()
        elif expanded or not text:
            out.append(expanded)
    return out
