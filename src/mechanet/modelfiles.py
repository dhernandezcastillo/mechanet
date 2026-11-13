"""Model files of the ML-potential presets: where mechanet finds them, and a check before the jobs.

Each preset needs one model file: ``mace-mh1`` MACE-MH-1, ``uma`` UMA ``uma-s-1p2p1`` and
``sevennet`` SevenNet-Omni. A preset takes it from the first of:

1. the path given as ``--option model=PATH``;
2. the models folder (:func:`models_folder`): ``$MECHANET_MODELS``, else
   ``$XDG_DATA_HOME/mechanet/models`` (``~/.local/share/mechanet/models``), where the file sits
   under its official name, directly or in a folder up to two levels down;
3. the copy that the package running the model downloads on the first run and keeps
   (mace-torch, fairchem, sevenn): this needs the internet, and for UMA a Hugging Face token
   with access to the weights, set up as UMA's documentation says (:func:`hugging_face_token`).

A file found in 1 or 2 is read from disk and nothing is downloaded. On a cluster whose compute
nodes have no internet, fill the models folder once, from a browser on any computer or with
``mechanet models NAME --download`` on the login node (docs/model-files.md), and every job
finds the files there.

``mechanet models`` reports, for each preset, the file it would use, whether that file is
complete, whether its package is installed, and so whether a job can run without internet.
``--load`` also loads each model found on this computer, as a job would, and computes one
small structure. ``--download`` first downloads the named models into the models folder
(:func:`download`): UMA from Hugging Face with the token huggingface_hub finds, the others with
no account. mechanet never ships or mirrors model files: each comes from its official page,
under its own licence (:data:`MODEL_FILES`). Of a token, mechanet only checks where it is set
up: it never shows, stores or records one.
"""

from __future__ import annotations

import contextlib
import http.client
import importlib.util
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
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
# A Hugging Face token as UMA's documentation asks for it: made on TOKEN_PAGE, with the permission
# TOKEN_PERMISSION (a token of type Read has it; a fine-grained one needs it ticked).
TOKEN_PAGE = "https://huggingface.co/settings/tokens"
TOKEN_PERMISSION = "Read access to contents of all public gated repos you can access"
# Files of the model's page that --download saves next to the model: UMA's licence lets copies be
# passed on only together with it, and its acceptable-use policy is part of the licence.
LICENCE_FILES = {"uma": ("LICENSE", "USE_POLICY.md")}


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


def _is_true(value):
    """On, as huggingface_hub reads its switches: 1, ON, YES or TRUE."""
    return str(value or "").strip().upper() in {"1", "ON", "YES", "TRUE"}


def _token_file(env):
    """Where ``hf auth login`` saves the token: ``$HF_TOKEN_PATH``, else ``$HF_HOME/token``, else
    ``~/.cache/huggingface/token`` (``$XDG_CACHE_HOME/huggingface/token``)."""
    if env.get("HF_TOKEN_PATH"):
        return Path(os.path.expandvars(env["HF_TOKEN_PATH"])).expanduser()
    if env.get("HF_HOME"):
        return Path(os.path.expandvars(env["HF_HOME"])).expanduser() / "token"
    return _cache_home(env) / "huggingface" / "token"


def hugging_face_token(environ=None):
    """Where huggingface_hub finds a Hugging Face token, set up as UMA's documentation says:
    ``"HF_TOKEN"`` (or the older ``"HUGGING_FACE_HUB_TOKEN"``) when that variable is set, else the
    path of the file ``hf auth login`` saved it in, else None. Only where it is: the token is
    never shown or recorded, and whether it gives access to UMA only Hugging Face can tell."""
    env = os.environ if environ is None else environ
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if env.get(name, "").strip():
            return name
    path = _token_file(env)
    try:
        return str(path) if path.is_file() and path.stat().st_size > 0 else None
    except OSError:
        return None


def _token_words(where):
    if where in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        return f"the token in {where}"
    return f"the token that `hf auth login` saved ({where})"


def hub_access_hint(exc, environ=None):
    """What to do when Hugging Face refused UMA's weights, for an error from huggingface_hub: no
    token here, a token it did not accept (401), or no access yet (403). None for other errors."""
    text = " ".join(f"{type(exc).__name__}: {exc}".split())
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status not in (401, 403):
        number = re.search(r"\b(401|403)\b", text)
        status = int(number.group(1)) if number else None
    if status is None and not re.search(r"gated|authenticat", text, re.IGNORECASE):
        return None
    page = model_file("uma").page
    where = hugging_face_token(environ)
    if where is None:
        return (
            "No Hugging Face token was found on this computer. As UMA's documentation says: request access at "
            f'{page}, create a token at {TOKEN_PAGE} with "{TOKEN_PERMISSION}" (a token of type Read has it), '
            "then run `hf auth login` or set HF_TOKEN"
        )
    who = _token_words(where)
    if status == 401:
        return (
            f"Hugging Face did not accept {who}: mistyped, expired or deleted? Create a new one at {TOKEN_PAGE}, "
            "then log in again with it (`hf auth login`; recent versions need --force to replace a saved token) "
            "or set HF_TOKEN"
        )
    return (
        f"{who[:1].upper()}{who[1:]} gives no access to facebook/UMA yet: access is granted after a review "
        f'(request access at {page}, then wait for the e-mail), and a fine-grained token needs "{TOKEN_PERMISSION}"'
    )


def _size_text(size):
    return f"{size / 1e9:.1f} GB" if size >= 1e9 else f"{size / 1e6:.0f} MB"


def _hub_location(url):
    """(repository, revision, file in the repository) of a Hugging Face download link, else None."""
    found = re.fullmatch(r"https://huggingface\.co/([\w.-]+/[\w.-]+)/resolve/([\w.-]+)/(.+)", url)
    return found.groups() if found else None


def download(calculator, environ=None, log=None):
    """Download a preset's model file into the models folder, from its official page.

    UMA comes from Hugging Face through huggingface_hub (installed with fairchem), with the token
    it finds (:func:`hugging_face_token`), as UMA's documentation describes; its licence files are
    saved next to it (:data:`LICENCE_FILES`). MACE-MH-1 and SevenNet-Omni need no account. A
    complete copy already in the folder is kept, and a broken one is left for you to delete.
    Progress goes to ``log`` (stderr by default). Returns path, status (``"downloaded"`` or
    ``"already in the models folder"``) and licence_files; raises RuntimeError saying what to do.
    """
    env = os.environ if environ is None else environ
    log = log or (lambda message: print(message, file=sys.stderr, flush=True))
    entry = model_file(calculator)
    folder = models_folder(env)
    existing = find_model_file(entry.filename, folder)
    if existing is not None:
        problem = check_file(existing)["problem"]
        if problem:
            raise RuntimeError(
                f"{existing}, in the models folder, is not usable ({problem}): delete it, then try again"
            )
        return {"path": str(existing), "status": "already in the models folder", "licence_files": []}
    hub = _hub_location(entry.url)
    if hub:
        path, licence_files = _download_from_hub(entry, hub, folder, env, log)
    else:
        path, licence_files = _download_link(entry, folder, log), []
    return {"path": str(path), "status": "downloaded", "licence_files": [str(p) for p in licence_files]}


def _download_from_hub(entry, hub, folder, env, log):
    repo, revision, name = hub
    if _is_true(env.get("HF_HUB_OFFLINE")):
        raise RuntimeError(
            "HF_HUB_OFFLINE is set in this shell, so nothing can come from Hugging Face: run `unset HF_HUB_OFFLINE` "
            "for the download (jobs may keep it)"
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            f"downloading {entry.title} needs huggingface_hub, which comes with {entry.package}: run this in the "
            f"environment where {entry.requirement} is installed"
        ) from exc
    where = hugging_face_token(env)
    target = folder / repo.split("/")[-1]
    target.mkdir(parents=True, exist_ok=True)
    log(
        f"{entry.calculator}: downloading {entry.filename} ({_size_text(entry.size)}) from huggingface.co/{repo} "
        f"with {_token_words(where) if where else 'no Hugging Face token (none found here)'}; "
        f"licence: {entry.licence}"
    )
    try:
        path = Path(hf_hub_download(repo_id=repo, filename=name, revision=revision, local_dir=str(target)))
    except Exception as exc:  # noqa: BLE001 - every failure is told in words, with what to do
        hint = hub_access_hint(exc, env)
        detail = " ".join(f"{type(exc).__name__}: {exc}".split())
        reason = f"{hint} ({type(exc).__name__})" if hint else detail
        raise RuntimeError(f"{entry.title} could not be downloaded: {reason}") from exc
    licence_files = []
    for extra in LICENCE_FILES.get(entry.calculator, ()):
        try:
            licence_files.append(
                Path(hf_hub_download(repo_id=repo, filename=extra, revision=revision, local_dir=str(target)))
            )
        except Exception as exc:  # noqa: BLE001 - the model is here: say what is missing
            log(f"{entry.calculator}: {extra} could not be downloaded ({type(exc).__name__}); it is on {entry.page}")
    return path, licence_files


def _download_link(entry, folder, log):
    """Download a file anyone may download into ``folder``, through a .part file that is renamed
    only once the download is complete and is a model file."""
    target = folder / entry.filename
    part = target.with_name(target.name + ".part")
    host = urllib.parse.urlsplit(entry.url).netloc
    log(
        f"{entry.calculator}: downloading {entry.filename} ({_size_text(entry.size)}) from {host}; "
        f"licence: {entry.licence}"
    )
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(entry.url, timeout=60) as response, open(part, "wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            done = shown = 0
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                done += len(chunk)
                if total and done * 10 // total > shown:
                    shown = done * 10 // total
                    log(f"{entry.calculator}: {done / 1e6:.0f} of {total / 1e6:.0f} MB")
        if total and done != total:
            raise RuntimeError(f"the download from {host} stopped after {done:,} of {total:,} bytes: try again")
        problem = check_file(part)["problem"]
        if problem:
            raise RuntimeError(f"what {host} sent is not usable: {problem}")
        os.replace(part, target)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{entry.url} answered {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"no connection to {host} ({exc.reason}): on a computer without internet, download it on one that "
            "has it and copy it into the models folder"
        ) from exc
    except (http.client.HTTPException, TimeoutError, ConnectionError) as exc:
        raise RuntimeError(f"the download from {host} broke off ({type(exc).__name__}): try again") from exc
    except OSError as exc:
        raise RuntimeError(
            f"{entry.filename} could not be saved in {folder} ({exc}); with too little room there, set "
            f"{MODELS_ENV} to a folder with more, such as project storage"
        ) from exc
    finally:
        part.unlink(missing_ok=True)
    return target


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


def check(calculators=None, load=False, device="auto", environ=None, download=False):
    """A report for each preset (every one in :data:`MODEL_FILES` when ``calculators`` is None):
    the file it would use and where it is, whether the file is complete, whether its package is
    installed, and whether a job can run without internet (``ready``); and where a Hugging Face
    token is set up (``hugging_face_token``, never the token).

    Nothing is downloaded unless ``download``: each preset is then first downloaded into the
    models folder (:func:`download`), and what happened is in its report under ``download``.
    With ``load``, each preset whose file and package are here is loaded as a job would load it
    and computes one small structure; this sets ``HF_HUB_OFFLINE=1`` for the process (unless it
    is set), after any download, so that fairchem has to find everything on this computer.
    """
    env = os.environ if environ is None else environ
    entries = [model_file(name) for name in calculators] if calculators else list(MODEL_FILES)
    fetched = {entry.calculator: _download_report(entry, env) for entry in entries} if download else {}
    folder = models_folder(env)
    if load:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    reports = [_report(entry, folder, env, load, device) for entry in entries]
    for report in reports:
        if report["calculator"] in fetched:
            report["download"] = fetched[report["calculator"]]
    return {
        "folder": str(folder),
        "folder_set_by": _folder_origin(env),
        "folder_exists": folder.is_dir(),
        "hugging_face_token": hugging_face_token(env),
        "models": reports,
    }


def _download_report(entry, env):
    try:
        return {**download(entry.calculator, env), "error": None}
    except Exception as exc:  # noqa: BLE001 - one model failing leaves the others to download
        message = str(exc) if isinstance(exc, (RuntimeError, ValueError)) else f"{type(exc).__name__}: {exc}"
        return {"path": None, "status": "failed", "licence_files": [], "error": message}


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
    downloaded = False
    for r in result["models"]:
        lines += ["", f"{r['calculator']}: {r['title']}, file {r['filename']}"]
        if "download" in r:
            downloaded = True
            lines.append(f"  download {_download_words(r['download'])}")
        if r["file"]:
            state = "complete" if r["complete"] else f"NOT USABLE: {r['problem']}"
            lines += [f"  file     {r['file']} ({r['found_in']})", f"           {r['size']:,} bytes, {state}"]
        elif r["problem"]:
            lines.append(f"  file     {r['problem']}")
        else:
            needs = "the internet and a Hugging Face token with access" if r["access"] else "the internet"
            lines.append(f"  file     not on this computer: the first run downloads it, which needs {needs}")
        if (r["file"] and not r["complete"]) or not (r["file"] or r["problem"]):
            first = "delete it, then: " if r["found_in"] == "models folder" else ""
            token = " (with your Hugging Face token)" if r["access"] else ""
            lines.append(f"  get it   {first}mechanet models {r['calculator']} --download{token}")
            lines.append(f"           or in a browser: {r['url']}, then put it in the models folder above")
            if r["access"]:
                lines.append(f"           needs {r['access']}; ask on {r['page']}")
                lines.append(f"  token    {_token_line(result.get('hugging_face_token'))}")
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
    done = "" if downloaded else "Nothing was downloaded. "
    lines += ["", f"{done}Filling the models folder: 'Model files on a cluster' in the documentation."]
    return "\n".join(lines)


def _download_words(outcome):
    if outcome["status"] == "failed":
        return f"FAILED: {outcome['error']}"
    if outcome["status"] != "downloaded":
        return f"{outcome['status']}: nothing downloaded"
    beside = [Path(p).name for p in outcome["licence_files"]]
    return f"saved {outcome['path']}" + (f", with {' and '.join(beside)} next to it" if beside else "")


def _token_line(where):
    if where is None:
        return (
            "none found here (no HF_TOKEN, no `hf auth login`): set one up as UMA's documentation says "
            "('Model files on a cluster' in the documentation)"
        )
    found = (
        f"found in {where}"
        if where in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
        else f"found: saved by `hf auth login` in {where}"
    )
    return f"{found}; whether it gives access, Hugging Face tells when downloading"


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
    p.add_argument(
        "--download",
        action="store_true",
        help="First download the named models into the models folder, from their official pages: UMA from "
        "Hugging Face with your token (HF_TOKEN, or hf auth login, as UMA's documentation says), the others "
        "with no account. Each comes under its own licence, shown as it downloads",
    )
    p.set_defaults(handler=_cli)


def _cli(args, registry):
    if args.download and not args.calculators:
        raise ValueError(
            "name the models to download, e.g. `mechanet models uma --download`: each comes under its own licence"
        )
    result = check(args.calculators or None, load=args.load, device=args.device, download=args.download)
    failed = bool(args.calculators) and not all(r["ready"] for r in result["models"])
    return result, check_text(result), 1 if failed else 0
