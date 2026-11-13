"""Model files of the ML-potential presets: where mechanet finds them, and a check before the jobs.

Each preset needs one model file: ``mace-mh1`` MACE-MH-1, ``uma`` UMA ``uma-s-1p2p1`` and
``sevennet`` SevenNet-Omni. A preset takes it from the first of:

1. the path given as ``--option model=PATH``;
2. the models folder (:func:`models_folder`): ``$MECHANET_MODELS``, else
   ``$XDG_DATA_HOME/mechanet/models`` (``~/.local/share/mechanet/models``), where the file sits
   under its official name, directly or in a folder up to two levels down;
3. the copy that the package running the model downloads on the first run and keeps
   (mace-torch, fairchem, sevenn): this needs the internet, and for UMA a Hugging Face login
   with access to the weights.

A file found in 1 or 2 is read from disk and nothing is downloaded. On a cluster whose compute
nodes have no internet, fill the models folder once, for instance from a browser on any
computer (docs/model-files.md), and every job finds the files there.

``mechanet models`` reports, for each preset, the file it would use, whether that file is
complete, whether its package is installed, and so whether a job can run without internet; it
loads and downloads nothing. ``--load`` also loads each model found on this computer, as a job
would, and computes one small structure. mechanet never ships or mirrors model files: each
comes from its official page, under its own licence (:data:`MODEL_FILES`).
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

MODELS_ENV = "MECHANET_MODELS"


@dataclass(frozen=True)
class ModelFile:
    """The model file a preset loads by default, and where its official copy is."""

    calculator: str  # the preset that loads it
    model: str  # the preset's default model name
    title: str
    filename: str  # the official file name
    size: int  # bytes of the file mechanet was tested with
    url: str  # direct download link
    page: str  # the page that describes the model (for UMA, where access is asked for)
    licence: str
    access: str  # what downloading needs besides the link; "" when anyone can
    package: str  # the distribution that runs the model
    requirement: str  # and its pip requirement


MODEL_FILES = (
    ModelFile(
        calculator="mace-mh1",
        model="mh-1",
        title="MACE-MH-1",
        filename="mace-mh-1.model",
        size=59_208_139,
        url="https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model",
        page="https://github.com/ACEsuit/mace-foundations/releases/tag/mace_mh_1",
        licence="Academic Software License (ASL): academic, non-commercial use only",
        access="",
        package="mace-torch",
        requirement="mace-torch>=0.3.14",
    ),
    ModelFile(
        calculator="uma",
        model="uma-s-1p2p1",
        title="UMA uma-s-1p2p1",
        filename="uma-s-1p2p1.pt",
        size=2_333_393_615,
        url="https://huggingface.co/facebook/UMA/resolve/main/checkpoints/uma-s-1p2p1.pt",
        page="https://huggingface.co/facebook/UMA",
        licence="FAIR Chemistry License: acknowledge UMA in publications; pass copies on only with the licence",
        access="a Hugging Face account granted access to facebook/UMA (requests are reviewed)",
        package="fairchem-core",
        requirement="fairchem-core>=2.24",
    ),
    ModelFile(
        calculator="sevennet",
        model="7net-omni",
        title="SevenNet-Omni",
        filename="checkpoint_sevennet_omni.pth",
        size=103_162_838,
        url="https://github.com/MDIL-SNU/SevenNet/releases/download/v0.12.0.cp/checkpoint_sevennet_omni.pth",
        page="https://github.com/MDIL-SNU/SevenNet",
        licence="MIT (the licence of the SevenNet repository that publishes it)",
        access="",
        package="sevenn",
        requirement="sevenn>=0.13",
    ),
)
# Official names of other models the presets take. fairchem names every UMA checkpoint <name>.pt.
OTHER_FILENAMES = {("mace-mh1", "mh-0"): "mace-mh-0.model"}
# Where sevenn keeps a downloaded checkpoint, below its package folder or ~/.cache/sevennet.
_SEVENN_FOLDERS = {"7net-omni": "SevenNet_omni"}
SMOKE_STRUCTURE = "Pt(111) 2x2x3 with O"


def models_folder(environ=None):
    """The models folder: ``$MECHANET_MODELS``, else ``$XDG_DATA_HOME/mechanet/models``, else
    ``~/.local/share/mechanet/models``. It need not exist."""
    env = os.environ if environ is None else environ
    if env.get(MODELS_ENV):
        return Path(env[MODELS_ENV]).expanduser()
    base = Path(env["XDG_DATA_HOME"]).expanduser() if env.get("XDG_DATA_HOME") else Path.home() / ".local" / "share"
    return base / "mechanet" / "models"


def _folder_origin(env):
    if env.get(MODELS_ENV):
        return MODELS_ENV
    return "XDG_DATA_HOME" if env.get("XDG_DATA_HOME") else "default"


def model_file(calculator):
    """The :class:`ModelFile` of a preset (``mace-mh1``, ``uma``, ``sevennet``)."""
    for entry in MODEL_FILES:
        if entry.calculator == calculator:
            return entry
    raise ValueError(
        f"no model file is known for {calculator!r}; known: {', '.join(e.calculator for e in MODEL_FILES)}"
    )


def official_filename(calculator, model):
    """The official file name of a model a preset takes, or None when mechanet does not know it."""
    for entry in MODEL_FILES:
        if (entry.calculator, entry.model) == (calculator, str(model)):
            return entry.filename
    if (calculator, str(model)) in OTHER_FILENAMES:
        return OTHER_FILENAMES[(calculator, str(model))]
    if calculator == "uma" and re.fullmatch(r"uma-[\w.-]+", str(model)):
        return f"{model}.pt"
    return None


def find_model_file(filename, folder=None):
    """``filename`` in the models folder, else in a folder inside it (up to two levels down);
    None when it is in neither. Two copies inside are an error, since which one a job used
    could not be told afterwards."""
    folder = models_folder() if folder is None else Path(folder)
    direct = folder / filename
    if direct.is_file():
        return direct
    if not folder.is_dir():
        return None
    found = sorted({p for pattern in (f"*/{filename}", f"*/*/{filename}") for p in folder.glob(pattern) if p.is_file()})
    if len(found) > 1:
        raise ValueError(
            f"{len(found)} files named {filename} in the models folder {folder}: "
            f"{', '.join(str(p) for p in found)}; keep one"
        )
    return found[0] if found else None


def model_path(calculator, model, filename=None, folder=None):
    """(path, where) of the model file a preset loads without downloading anything: the file
    ``model`` names (where ``"path"``), else the copy of the model's official file
    (``filename``, by default :func:`official_filename`) in the models folder (where
    ``"models folder"``). (None, None) when neither exists and the package provides the model."""
    given = Path(str(model)).expanduser()
    if given.is_file():
        return given, "path"
    filename = filename or official_filename(calculator, model)
    if filename:
        found = find_model_file(filename, folder)
        if found is not None:
            return found, "models folder"
    return None, None


def not_on_this_computer(calculator, model, exc):
    """The error for a model that is neither on this computer nor downloadable (no internet)."""
    filename = official_filename(calculator, model)
    remedy = (
        f"put {filename} in the models folder {models_folder()}"
        if filename
        else "give the path of the model file (--option model=PATH)"
    )
    return RuntimeError(
        f"{calculator}: the model {model!r} is not on this computer and could not be downloaded "
        f"({type(exc).__name__}: {exc}); {remedy}. `mechanet models` shows what is where"
    )


def _cache_home(env):
    return Path(env["XDG_CACHE_HOME"]).expanduser() if env.get("XDG_CACHE_HOME") else Path.home() / ".cache"


def package_copy(entry, environ=None):
    """The copy of ``entry``'s file that its package downloaded earlier, or None. Looked up where
    mace-torch, fairchem and sevenn keep their downloads, without importing them; only for the
    report of ``mechanet models`` (loading leaves this to the package)."""
    env = os.environ if environ is None else environ
    candidates = []
    if entry.package == "mace-torch":
        folder = _cache_home(env) / "mace"
        # mace-torch keeps only the letters, digits and _ of the name: mace-mh-1.model -> macemh1model
        squeezed = "".join(c for c in entry.filename if c.isalnum() or c == "_")
        candidates = [folder / squeezed, folder / entry.filename]
    elif entry.package == "fairchem-core":
        cache = env.get("FAIRCHEM_CACHE_DIR")
        repo = (Path(cache).expanduser() if cache else Path.home() / ".cache" / "fairchem") / "models--facebook--UMA"
        snapshots = sorted((repo / "snapshots").glob("*")) if (repo / "snapshots").is_dir() else []
        ref = repo / "refs" / "main"
        if ref.is_file():  # the snapshot Hugging Face's main branch pointed to at the download first
            snapshots.insert(0, repo / "snapshots" / ref.read_text(encoding="utf-8").strip())
        candidates = [s / "checkpoints" / entry.filename for s in snapshots]
    elif entry.package == "sevenn" and entry.model in _SEVENN_FOLDERS:
        sub = Path(_SEVENN_FOLDERS[entry.model]) / entry.filename
        try:
            spec = importlib.util.find_spec("sevenn")  # finds the package without running it
        except (ImportError, ValueError):
            spec = None
        for location in (spec.submodule_search_locations or []) if spec else []:
            candidates.append(Path(location) / "pretrained_potentials" / sub)
        candidates.append(Path.home() / ".cache" / "sevennet" / sub)
    return next((p for p in candidates if p.is_file()), None)


def check_file(path, full=False):
    """Is a model file complete? Returns size, complete, problem (None when complete) and, with
    ``full``, sha256.

    PyTorch saves a model as a zip archive whose table of contents comes last: a download cut
    short has none, and an error page saved under the model's name is no archive at all.
    ``full`` also checks every member against its checksum and computes the SHA-256 that the
    results record, reading the whole file."""
    path = Path(path)
    report = {"size": path.stat().st_size, "complete": False, "problem": None}
    try:
        with zipfile.ZipFile(path) as archive:
            if not archive.infolist():
                report["problem"] = "an empty archive, not a model"
            elif full:
                damaged = archive.testzip()
                if damaged is not None:
                    report["problem"] = f"damaged: {damaged} fails its checksum"
    except zipfile.BadZipFile:
        report["problem"] = "not a complete model file: a download cut short, or another file saved under its name"
    except OSError as exc:
        report["problem"] = f"cannot be read ({exc})"
    report["complete"] = report["problem"] is None
    if full and report["complete"]:
        from .calculators import file_sha256

        report["sha256"] = file_sha256(path)
    return report


def _installed(distribution):
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _version_tuple(text):
    return tuple(int(n) for n in re.findall(r"\d+", str(text))[:3])


def check(calculators=None, load=False, device="auto", environ=None):
    """A report for each preset (every one in :data:`MODEL_FILES` when ``calculators`` is None):
    the file it would use and where it is, whether the file is complete, whether its package is
    installed, and whether a job can run without internet (``ready``).

    Nothing is downloaded. With ``load``, each preset whose file and package are here is loaded
    as a job would load it and computes one small structure; this sets ``HF_HUB_OFFLINE=1`` for
    the process (unless it is set), so that fairchem has to find everything on this computer.
    """
    env = os.environ if environ is None else environ
    entries = [model_file(name) for name in calculators] if calculators else list(MODEL_FILES)
    folder = models_folder(env)
    if load:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return {
        "folder": str(folder),
        "folder_set_by": _folder_origin(env),
        "folder_exists": folder.is_dir(),
        "models": [_report(entry, folder, env, load, device) for entry in entries],
    }


def _report(entry, folder, env, load, device):
    version = _installed(entry.package)
    minimum = entry.requirement.partition(">=")[2]
    report = {
        "calculator": entry.calculator,
        "model": entry.model,
        "title": entry.title,
        "filename": entry.filename,
        "file": None,
        "found_in": None,
        "size": None,
        "complete": None,
        "problem": None,
        "package": entry.package,
        "package_version": version,
        "package_ok": version is not None and _version_tuple(version) >= _version_tuple(minimum),
        "requirement": entry.requirement,
        "url": entry.url,
        "page": entry.page,
        "licence": entry.licence,
        "access": entry.access,
        "notes": [],
    }
    try:
        path, found_in = find_model_file(entry.filename, folder), "models folder"
    except ValueError as exc:
        path, found_in = None, None
        report["problem"] = str(exc)
    if path is None and report["problem"] is None:
        path, found_in = package_copy(entry, env), f"{entry.package}'s own copy"
    if path is not None:
        report.update(file=str(path), found_in=found_in, **check_file(path, full=load))
        if report["complete"] and report["size"] != entry.size:
            report["notes"].append(
                f"{report['size']:,} bytes, not the {entry.size:,} of the file mechanet was tested with: "
                "another version of the model"
            )
        if report["complete"] and entry.package == "fairchem-core" and found_in != "models folder":
            report["notes"].append(
                "jobs on nodes without internet need HF_HUB_OFFLINE=1, or fairchem first tries to reach Hugging Face"
            )
    ready = path is not None and bool(report["complete"]) and report["package_ok"]
    if load:
        report["test"] = _load(entry, device) if ready else None
        ready = ready and report["test"].get("error") is None
    report["ready"] = ready
    return report


def _load(entry, device):
    """Load ``entry``'s preset as a job would and compute one small structure (energy and forces)."""
    import numpy as np
    from ase.build import add_adsorbate, fcc111

    from .calculators import load_calculator

    try:
        start = time.perf_counter()
        # Third-party potentials print banners on stdout; keep stdout for the result (--json).
        with contextlib.redirect_stdout(sys.stderr):
            calculator = load_calculator(entry.calculator, {"device": device})
            loaded = time.perf_counter()
            atoms = fcc111("Pt", size=(2, 2, 3), vacuum=6.0)
            add_adsorbate(atoms, "O", 1.2, "fcc")
            atoms.calc = calculator.calc
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces())
        done = time.perf_counter()
    except Exception as exc:  # noqa: BLE001 - whatever stops the model is the answer the check gives
        return {"error": f"{type(exc).__name__}: {exc}"}
    model = calculator.describe().get("model") or {}
    return {
        "error": None,
        "device": model.get("device"),
        "load_s": round(loaded - start, 1),
        "compute_s": round(done - loaded, 2),
        "structure": SMOKE_STRUCTURE,
        "energy_eV": round(energy, 4),
        "max_force_eVA": round(float(np.linalg.norm(forces, axis=1).max()), 3),
        "sha256": model.get("sha256"),
    }


def check_text(result):
    """The report of :func:`check`, for people."""
    origin = {
        MODELS_ENV: f"set by {MODELS_ENV}",
        "XDG_DATA_HOME": f"from XDG_DATA_HOME; {MODELS_ENV} sets another",
        "default": f"the default; {MODELS_ENV} sets another",
    }[result["folder_set_by"]]
    lines = [f"Models folder: {result['folder']} ({origin})"]
    if not result["folder_exists"]:
        lines.append("  (it does not exist yet)")
    for r in result["models"]:
        lines += ["", f"{r['calculator']}: {r['title']}, file {r['filename']}"]
        if r["file"]:
            state = "complete" if r["complete"] else f"NOT USABLE: {r['problem']}"
            lines += [f"  file     {r['file']} ({r['found_in']})", f"           {r['size']:,} bytes, {state}"]
        elif r["problem"]:
            lines.append(f"  file     {r['problem']}")
        else:
            needs = "the internet and a Hugging Face login with access" if r["access"] else "the internet"
            lines.append(f"  file     not on this computer: the first run downloads it, which needs {needs}")
        if not r["file"] or not r["complete"]:
            lines.append(f"  get it   {r['url']}")
            if r["access"]:
                lines.append(f"           needs {r['access']}; ask on {r['page']}")
            lines.append("           then put it in the models folder above")
        if r["package_version"] is None:
            lines.append(f'  package  {r["package"]} is not installed here (pip install "{r["requirement"]}")')
        elif not r["package_ok"]:
            lines.append(f"  package  {r['package']} {r['package_version']} is too old: needs {r['requirement']}")
        else:
            lines.append(f"  package  {r['package']} {r['package_version']}")
        lines.append(f"  licence  {r['licence']}")
        for note in r["notes"]:
            lines.append(f"  note     {note}")
        if "test" in r:
            test = r["test"]
            if test is None:
                lines.append("  test     not run: the model cannot load here")
            elif test["error"]:
                lines.append(f"  test     FAILED: {test['error']}")
            else:
                lines.append(
                    f"  test     loaded on {test['device']} in {test['load_s']} s; {test['structure']}: "
                    f"{test['energy_eV']} eV, largest force {test['max_force_eVA']} eV/A ({test['compute_s']} s)"
                )
        lines.append(f"  ready    {'yes, without internet' if r['ready'] else 'no: ' + '; '.join(_missing(r))}")
    lines += ["", "Nothing was downloaded. Filling the models folder: 'Model files on a cluster' in the documentation."]
    return "\n".join(lines)


def _missing(report):
    reasons = []
    if not report["file"]:
        reasons.append(
            "the model file is not on this computer"
            if not report["problem"]
            else "more than one copy of the file in the models folder"
        )
    elif not report["complete"]:
        reasons.append("the model file is not usable")
    if not report["package_ok"]:
        reasons.append(f"{report['requirement']} is not installed here")
    test = report.get("test")
    if isinstance(test, dict) and test.get("error"):
        reasons.append("the test calculation failed")
    return reasons


def register_command(subparsers, registry):
    p = subparsers.add_parser(
        "models",
        help="Where the model files of the ML potentials are, and whether jobs can run without internet",
        description=__doc__.split("\n\n")[0],
    )
    p.add_argument(
        "calculators",
        nargs="*",
        metavar="CALCULATOR",
        help="mace-mh1, uma or sevennet (default: all three); one named here that is not ready makes "
        "the exit status 1, for job scripts",
    )
    p.add_argument(
        "--load",
        action="store_true",
        help="Also load each model found on this computer, as a job would, and compute one small structure "
        "(nothing is downloaded)",
    )
    p.add_argument("--device", default="auto", help="Device for --load: auto (default), cpu or cuda")
    p.set_defaults(handler=_cli)


def _cli(args, registry):
    result = check(args.calculators or None, load=args.load, device=args.device)
    failed = bool(args.calculators) and not all(r["ready"] for r in result["models"])
    return result, check_text(result), 1 if failed else 0
