"""In-process calculators: relaxations that run inside Python, without a batch queue.

DFT runs go through the staged runner (:mod:`mechanet.runner`): restartable SLURM jobs whose
inputs and outputs are hashed and checked. A *calculator* here is anything that gives energies
and forces inside the Python process: a machine-learned interatomic potential (MACE, UMA, ...),
or ASE's toy EMT potential for tests. ``mechanet prescreen`` (:mod:`mechanet.prescreen`) uses
one to relax the generated starts before any DFT and judges the relaxed structures with the
same checks as DFT results.

A calculator is named by a spec:

* a name the registry knows: ``emt``, ``mace-mp``, ``mace-mh1``, ``uma`` and ``sevennet`` are
  built in, a plugin adds others with ``registry.add_calculator(name, factory, description)``
  (or the ``mechanet.calculators`` entry-point group);
* ``module:callable``: a factory importable from Python, such as ``mace.calculators:mace_mp``.

The factory is called with the options (``--option key=value``) and returns either an ASE
calculator, which :class:`ASECalculator` wraps, or an object with ``describe()`` and
``relax(structure, fmax, steps)`` (the :class:`Calculator` protocol).

Three built-ins are presets of foundation potentials trained on surfaces with adsorbates, each
with the settings it is meant to be used with: ``mace-mh1`` (MACE-MH-1, head ``omat_pbe``),
``uma`` (UMA through fairchem, ``uma-s-1p2p1`` with task ``oc22``) and ``sevennet``
(SevenNet-Omni, modal ``oc22``). A factory may recommend relaxation settings
(:data:`RECOMMENDED_ML_RELAXATION` for the three presets): ``prescreen`` uses them unless
``--optimizer``, ``--fmax`` or ``--steps`` is given. Every preset records its model file and
SHA-256 in ``describe()``, so a changed model file is not mistaken for the one an earlier
result was relaxed with. A preset takes its model file from the path given as ``model``, else
from mechanet's models folder, else from its package, which downloads it on the first run
(:mod:`mechanet.modelfiles`; ``mechanet models`` checks what a job would find).
docs/prescreening.md explains the choice of models and settings.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from .modelfiles import hub_access_hint, model_path, models_folder, not_on_this_computer, official_filename

OPTIMIZERS = ("BFGS", "LBFGS", "FIRE")
DEFAULT_OPTIMIZER = "BFGS"
DEFAULT_FMAX_EVA = 0.05
DEFAULT_STEPS = 300
RELAX_SETTINGS = ("optimizer", "fmax_eVA", "steps")
# How the foundation-potential presets relax. FIRE, because on a NiFe LDH edge (256 coverage
# states, MACE-MH-1) BFGS formed O-O bonds in 18 states where FIRE formed none, with the same
# diagram otherwise; 0.03 eV/A, the force criterion of the DFT relaxations (EDIFFG = -0.03 in
# the pbe-d3 preset); 1500 steps, room for FIRE's slow tail (up to 457 steps there, 87 on
# average from the generated starts).
RECOMMENDED_ML_RELAXATION = {"optimizer": "FIRE", "fmax_eVA": 0.03, "steps": 1500}


@dataclass(frozen=True)
class Relaxation:
    """The outcome of one relaxation: positions in the input's atom order, Cartesian Angstrom."""

    energy_eV: float
    positions: np.ndarray
    max_force_eVA: float  # largest force on an atom the relaxation may move
    converged: bool
    steps: int


@runtime_checkable
class Calculator(Protocol):
    def describe(self) -> dict:
        """What was used: class, package versions, model and options (stored with every result)."""

    def relax(self, structure, fmax: float = DEFAULT_FMAX_EVA, steps: int = DEFAULT_STEPS) -> Relaxation:
        """Relax ``structure`` keeping fixed what its selective-dynamics flags fix."""


@dataclass(frozen=True)
class CalculatorFactory:
    name: str
    make: Callable
    description: str = ""
    origin: str = "builtin"
    # recommended optimizer, fmax_eVA and steps (any of them); empty: mechanet's defaults
    relax_settings: dict = field(default_factory=dict, hash=False)


def check_relax_settings(settings, owner="calculator"):
    """Recommended relaxation settings, checked: a dict with any of optimizer, fmax_eVA, steps."""
    if settings is None:
        return {}
    if not isinstance(settings, dict):
        raise ValueError(f"{owner}: relax_settings must be a dict of {', '.join(RELAX_SETTINGS)}, not {settings!r}")
    unknown = sorted(set(settings) - set(RELAX_SETTINGS))
    if unknown:
        raise ValueError(f"{owner}: unknown relax_settings {unknown}; known: {', '.join(RELAX_SETTINGS)}")
    checked = {}
    if "optimizer" in settings:
        if settings["optimizer"] not in OPTIMIZERS:
            raise ValueError(f"{owner}: optimizer {settings['optimizer']!r} is not one of {', '.join(OPTIMIZERS)}")
        checked["optimizer"] = settings["optimizer"]
    if "fmax_eVA" in settings:
        value = settings["fmax_eVA"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
            raise ValueError(f"{owner}: fmax_eVA must be a positive number, not {value!r}")
        checked["fmax_eVA"] = float(value)
    if "steps" in settings:
        value = settings["steps"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{owner}: steps must be a nonnegative integer, not {value!r}")
        checked["steps"] = int(value)
    return checked


def relax_settings_text(settings):
    """``FIRE, 0.03 eV/A, at most 1500 steps`` for recommended settings; ``-`` without any."""
    parts = []
    if "optimizer" in settings:
        parts.append(settings["optimizer"])
    if "fmax_eVA" in settings:
        parts.append(f"{settings['fmax_eVA']:g} eV/A")
    if "steps" in settings:
        parts.append(f"at most {settings['steps']} steps")
    return ", ".join(parts) or "-"


def to_ase(structure):
    """ASE Atoms of a :class:`mechanet.structure.Structure`, periodic in 3D as in VASP.

    Atoms fixed in all directions get one FixAtoms; atoms fixed in some directions get
    FixCartesian (True means fixed there), so selective dynamics carries over exactly.
    """
    from ase import Atoms
    from ase.constraints import FixAtoms, FixCartesian

    atoms = Atoms(structure.symbols, positions=structure.positions, cell=structure.cell, pbc=True)
    fixed = ~np.asarray(structure.mobile, dtype=bool)
    constraints = []
    whole = np.flatnonzero(fixed.all(axis=1))
    if len(whole):
        constraints.append(FixAtoms(indices=whole))
    partial = np.flatnonzero(fixed.any(axis=1) & ~fixed.all(axis=1))
    for mask in sorted({tuple(bool(x) for x in fixed[i]) for i in partial}):
        constraints.append(FixCartesian([int(i) for i in partial if tuple(fixed[i]) == mask], mask=mask))
    atoms.set_constraint(constraints)
    return atoms


@functools.lru_cache(maxsize=None)
def _distributions_by_package():
    """Top-level package -> distributions that install it. Scanning every installed distribution
    takes seconds in a large environment, and nothing installs or removes one mid-run: once per process."""
    from importlib.metadata import packages_distributions

    return packages_distributions()


def _distribution(module_name):
    """(distribution, version) that provides a module, or (top-level name, None)."""
    from importlib.metadata import PackageNotFoundError, version

    top = str(module_name).split(".")[0]
    for dist in _distributions_by_package().get(top, [top]):
        try:
            return dist, version(dist)
        except PackageNotFoundError:
            continue
    return top, None


class ASECalculator:
    """Relaxes with any ASE calculator: an ML potential, EMT, or a code ASE drives.

    ``recommended``: the relaxation settings the calculator's factory recommends (fmax_eVA,
    steps), which ``prescreen`` uses when none are given. A calculator that carries a
    ``mechanet_model`` dict (the presets do: model file, SHA-256, head or task, versions) has it
    stored under ``model`` in :meth:`describe`.
    """

    def __init__(self, calc, label, optimizer="BFGS", options=None, recommended=None):
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer {optimizer!r}; choose one of {', '.join(OPTIMIZERS)}")
        self.calc = calc
        self.label = str(label)
        self.optimizer = optimizer
        self.options = dict(options or {})
        self.recommended = {
            k: v for k, v in check_relax_settings(recommended, f"calculator {label!r}").items() if k != "optimizer"
        }

    def describe(self):
        import ase

        cls = type(self.calc)
        package, package_version = _distribution(cls.__module__)
        description = {
            "calculator": self.label,
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "package": package,
            "package_version": package_version,
            "ase_version": ase.__version__,
            "options": self.options,
            "optimizer": self.optimizer,
        }
        model = getattr(self.calc, "mechanet_model", None)
        if model:
            description["model"] = dict(model)
        return description

    def relax(self, structure, fmax=DEFAULT_FMAX_EVA, steps=DEFAULT_STEPS):
        import ase.optimize

        atoms = to_ase(structure)
        atoms.calc = self.calc
        taken = 0
        if steps > 0 and np.asarray(structure.mobile).any():
            optimizer = getattr(ase.optimize, self.optimizer)(atoms, logfile=None)
            optimizer.run(fmax=fmax, steps=steps)
            taken = int(getattr(optimizer, "nsteps", steps))
        energy = float(atoms.get_potential_energy())
        forces = np.where(np.asarray(structure.mobile, dtype=bool), atoms.get_forces(), 0.0)
        largest = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0
        return Relaxation(energy, atoms.get_positions(), largest, largest <= fmax, taken)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_emt(**options):
    """ASE's effective-medium toy potential."""
    from ase.calculators.emt import EMT

    return EMT(**options)


def make_mace_mp(model="medium", device="cpu", default_dtype="float64", **options):
    """MACE-MP foundation potential; float64 by default, as MACE recommends for relaxations."""
    try:
        from mace.calculators import mace_mp
    except ImportError as exc:
        raise RuntimeError("the mace-mp calculator needs mace-torch (pip install mace-torch)") from exc
    return mace_mp(model=model, device=device, default_dtype=default_dtype, **options)


_SHA256 = {}  # (resolved path, size, mtime) -> digest: a model file is read once per process


def file_sha256(path):
    """SHA-256 of a (model) file, computed once per process for each path, size and modification time."""
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _SHA256:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        _SHA256[key] = digest.hexdigest()
    return _SHA256[key]


def resolve_device(device="auto"):
    """``auto`` (or empty): ``cuda`` when torch sees a GPU, else ``cpu``; any other value as given."""
    if device in (None, "", "auto"):
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


def _version(distribution):
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _file_record(path, name=None):
    """{"file": name, "sha256"} of a model file, or {} when its path is not known. The name and the
    digest identify the file; its folder does not, so a moved cache relaxes nothing again.
    ``name``: the file's official name, recorded instead of the name on disk, so that a model
    mace-torch saved under another name and its copy in the models folder are the same file."""
    if path is None:
        return {}
    path = Path(path)
    shown = name or path.name
    return {"file": shown, "sha256": file_sha256(path)} if path.is_file() else {"file": shown}


def _announce(calculator, path, where):
    """One line on stderr: the model file a preset loads and where it was found (job logs show it)."""
    print(f"{calculator}: model file {path} ({'the path given' if where == 'path' else where})", file=sys.stderr)


def _with_d3(calc, device, cutoff_A=None):
    """``calc`` plus the D3 dispersion correction with Becke-Johnson damping and PBE parameters,
    as VASP's IVDW = 12, through torch-dftd (the route mace_mp takes for its ``dispersion``).

    The pair cutoff is torch-dftd's 95 bohr (50.3 A, VASP's VDW_RADIUS is 50.2 A) unless
    ``cutoff_A`` is given. Returns (calculator, description of the correction).
    """
    try:
        import torch
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    except ImportError as exc:
        raise RuntimeError(
            "dispersion=true needs torch-dftd in the calculator's environment (pip install torch-dftd)"
        ) from exc
    from ase.calculators.mixing import SumCalculator
    from ase.units import Bohr

    options = {"device": device, "damping": "bj", "xc": "pbe", "dtype": torch.float64}
    cutoff = 95.0 * Bohr if cutoff_A is None else float(cutoff_A)
    options["cutoff"] = cutoff
    record = {"method": "D3(BJ)", "xc": "pbe", "cutoff_A": round(cutoff, 4), "torch-dftd": _version("torch-dftd")}
    return SumCalculator([calc, TorchDFTD3Calculator(**options)]), record


def make_mace_mh1(
    model="mh-1",
    head="omat_pbe",
    device="auto",
    default_dtype="float64",
    dispersion=False,
    dispersion_cutoff_A=None,
    **options,
):
    """MACE-MH-1 (Batatia et al., arXiv:2510.25380), head omat_pbe, in float64.

    The model card recommends the OMAT head (PBE/PBE+U) for surfaces and advises adding D3 for
    systems where dispersion matters (``dispersion=true``, off by default). Of seven MACE
    foundation models scored zero-shot on the hydrogen coverage of a NiFe LDH edge it drew the
    closest Pourbaix diagram, and its few lowest states held every DFT-stable one.

    ``model``: a name mace-torch knows (``mh-1``, also ``mh-0``) or the path of a model file, for
    instance an MH-1 fine-tuned on your own data. ``head``: checked against the file, since
    mace-torch replaces an unknown head by the model's last one with only a warning.
    """
    try:
        from mace.calculators import foundations_models, mace_mp
    except ImportError as exc:
        raise RuntimeError(
            "the mace-mh1 calculator needs mace-torch 0.3.14 or later (pip install 'mace-torch>=0.3.14')"
        ) from exc
    dev = resolve_device(device)
    urls = getattr(foundations_models, "mace_mp_urls", None) or {}
    official = official_filename("mace-mh1", model) or (
        str(urls[str(model)]).rsplit("/", 1)[-1] if str(model) in urls else None
    )
    path, where = model_path("mace-mh1", model, filename=official)
    if path is None:
        known = getattr(foundations_models, "mace_mp_names", None)
        if known is not None and str(model) not in known:
            raise ValueError(f"MACE model {model!r} is neither a file nor a name mace-torch knows ({', '.join(known)})")
        try:
            path, where = Path(foundations_models.download_mace_mp_checkpoint(str(model))), "mace-torch's copy"
        except OSError as exc:  # no internet here: say where a copy would be found
            raise not_on_this_computer("mace-mh1", model, exc) from exc
    _announce("mace-mh1", path, where)
    calc = mace_mp(model=str(path), device=dev, default_dtype=default_dtype, head=head, **options)
    heads = [str(h) for h in getattr(calc, "available_heads", None) or ()]
    used = getattr(calc, "head", None)
    if head is not None and used != head:
        raise ValueError(
            f"MACE model {model!r} has no head {head!r}; it has {', '.join(heads) or 'none listed'} "
            f"(mace-torch would have used {used!r} instead)"
        )
    record = {
        "name": "MACE-MH-1" if str(model) == "mh-1" else str(model),
        "model": str(model),
        **_file_record(path, None if where == "path" else official),
        "head": used,
        "heads": heads,
        "dtype": default_dtype,
        "device": dev,
        "dispersion": None,
        "versions": {"mace-torch": _version("mace-torch")},
    }
    if dispersion:
        calc, record["dispersion"] = _with_d3(calc, dev, dispersion_cutoff_A)
    calc.mechanet_model = record
    return calc


def _uma_checkpoint(pretrained_mlip, model):
    """Local path of a UMA checkpoint: the file itself, or the cached download of a registered name."""
    if Path(str(model)).expanduser().is_file():
        return Path(str(model)).expanduser()
    locate = getattr(pretrained_mlip, "pretrained_checkpoint_path_from_name", None)
    if locate is None:
        return None
    try:
        return Path(locate(str(model)))
    except Exception:  # noqa: BLE001 - provenance only: no path is recorded then
        return None


def make_uma(
    model="uma-s-1p2p1",
    task="oc22",
    device="auto",
    inference_settings="high-precision",
    dispersion=False,
    dispersion_cutoff_A=None,
    **options,
):
    """UMA (Wood et al., arXiv:2506.23971) through fairchem, model uma-s-1p2p1, task oc22.

    ``task`` picks the training data whose level of theory the model reproduces: ``oc22`` oxide
    slabs with adsorbates (spin-polarised PBE+U, the closest to OER and CER on oxides; UMA 1.2
    models only), ``oc20`` metal slabs with adsorbates (RPBE, no oxides), ``omat`` bulk materials
    (PBE/PBE+U, Materials Project settings), ``oc25`` slabs with explicit water (RPBE+D3).
    ``inference_settings``: ``high-precision`` (TF32 off), not fairchem's default, which trades
    accuracy for speed. Neither OC22 nor OMat24 has dispersion: ``dispersion=true`` adds D3(BJ).
    The weights are gated on Hugging Face: set up a token as UMA's documentation says (request
    access to facebook/UMA; a token with read access to gated repositories, in HF_TOKEN or saved
    by ``hf auth login``), or put the file in mechanet's models folder
    (:mod:`mechanet.modelfiles`), which is read without the internet, as is the file ``model``
    names when it is a path; a refusal says which is missing. Otherwise fairchem downloads
    the model on the first run into its cache, ``~/.cache/fairchem`` or the folder in
    ``FAIRCHEM_CACHE_DIR`` (``get_predict_unit``'s ``cache_dir`` moves only two small
    reference-energy files). A file loaded by path goes without those two files: fairchem uses
    them only for an isolated single atom and in its formation-energy wrapper, while the
    energies and forces of slabs and molecules come from the model file, which carries its own
    element references. ``options`` go to ``pretrained_mlip.get_predict_unit`` (``overrides``,
    ``workers``, ...).
    """
    try:
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
    except ImportError as exc:
        raise RuntimeError(
            "the uma calculator needs fairchem-core 2.24 or later (pip install 'fairchem-core>=2.24'), "
            "in an environment of its own: mace-torch needs another e3nn version"
        ) from exc
    dev = resolve_device(device)
    path, where = model_path("uma", model)
    try:
        unit = pretrained_mlip.get_predict_unit(
            str(model if path is None else path), inference_settings=inference_settings, device=dev, **options
        )
    except Exception as exc:  # noqa: BLE001 - a refused download gets a hint, anything else passes unchanged
        hint = hub_access_hint(exc)
        if hint is not None:
            text = " ".join(f"{type(exc).__name__}: {exc}".split())
            filename = official_filename("uma", model) or "the model file"
            raise RuntimeError(
                f"UMA {model!r} could not be loaded ({text}). {hint}. Or put {filename} in mechanet's models "
                f"folder ({models_folder()}): `mechanet models uma` says what is where"
            ) from exc
        if path is None and isinstance(exc, OSError):  # no internet here: say where a copy would be found
            raise not_on_this_computer("uma", model, exc) from exc
        raise
    if path is None:
        path, where = _uma_checkpoint(pretrained_mlip, model), "fairchem's copy"
    _announce("uma", model if path is None else path, where)
    tasks = sorted(str(t) for t in (getattr(unit, "dataset_to_tasks", None) or ()))
    if tasks and str(task) not in tasks:
        raise ValueError(
            f"UMA {model!r} has no task {task!r}; it has {', '.join(tasks)} "
            "(oc22 needs a UMA 1.2 model, such as uma-s-1p2p1)"
        )
    calc = FAIRChemCalculator(unit, task_name=str(task))
    record = {
        "name": "UMA",
        "model": str(model),
        **_file_record(path, None if where == "path" else official_filename("uma", model)),
        "task": str(task),
        "tasks": tasks,
        "inference_settings": str(inference_settings),
        "device": dev,
        "dispersion": None,
        "versions": {"fairchem-core": _version("fairchem-core")},
    }
    if dispersion:
        calc, record["dispersion"] = _with_d3(calc, dev, dispersion_cutoff_A)
    calc.mechanet_model = record
    return calc


def _sevennet_checkpoint(model):
    """Local path of a SevenNet checkpoint: the file itself, or the file a pretrained name resolves to."""
    if Path(str(model)).expanduser().is_file():
        return Path(str(model)).expanduser()
    try:
        from sevenn.util import pretrained_name_to_path

        return Path(pretrained_name_to_path(str(model)))
    except Exception:  # noqa: BLE001 - provenance only: no path is recorded then
        return None


def make_sevennet(
    model="7net-omni", modal="oc22", device="auto", dispersion=False, dispersion_cutoff_A=None, **options
):
    """SevenNet-Omni (Kim et al., arXiv:2510.11241), modal oc22.

    One model trained on fifteen datasets, each with an output channel of its own (a *modal*);
    ``oc22`` reproduces the OC22 oxide slabs with adsorbates (PBE+U), ``mpa`` (the authors'
    default), ``omat24``, ``oc20`` and others exist. The modal is matched exactly: write it in
    lower case. ``dispersion=true`` adds D3(BJ) through torch-dftd (SevenNet's own D3 needs CUDA).
    ``options`` go to ``SevenNetCalculator`` (``enable_cueq``, ``enable_flash``, ...).
    """
    try:
        from sevenn.calculator import SevenNetCalculator
    except ImportError as exc:
        raise RuntimeError(
            "the sevennet calculator needs sevenn 0.13 or later (pip install 'sevenn>=0.13'), "
            "in an environment of its own: mace-torch needs another e3nn version"
        ) from exc
    dev = resolve_device(device)
    path, where = model_path("sevennet", model)
    try:
        calc = SevenNetCalculator(model=str(model if path is None else path), modal=modal, device=dev, **options)
    except OSError as exc:
        if path is None:  # no internet here: say where a copy would be found
            raise not_on_this_computer("sevennet", model, exc) from exc
        raise
    if path is None:
        path, where = _sevennet_checkpoint(model), "sevenn's copy"
    _announce("sevennet", model if path is None else path, where)
    record = {
        "name": "SevenNet-Omni" if str(model) == "7net-omni" else str(model),
        "model": str(model),
        **_file_record(path, None if where == "path" else official_filename("sevennet", model)),
        "modal": modal,
        "device": dev,
        "dispersion": None,
        "versions": {"sevenn": _version("sevenn")},
    }
    if dispersion:
        calc, record["dispersion"] = _with_d3(calc, dev, dispersion_cutoff_A)
    calc.mechanet_model = record
    return calc


for _preset in (make_mace_mh1, make_uma, make_sevennet):
    # also visible to whoever loads the factory through its entry point
    _preset.relax_settings = dict(RECOMMENDED_ML_RELAXATION)


BUILTIN_CALCULATORS = (
    CalculatorFactory(
        "emt",
        make_emt,
        "ASE's effective-medium toy potential (Al Cu Ag Au Ni Pd Pt with H C N O): for tests, not for chemistry",
    ),
    CalculatorFactory(
        "mace-mp",
        make_mace_mp,
        "MACE-MP foundation potential (needs mace-torch); options model (small, medium, large or a file), "
        "device, default_dtype",
    ),
    CalculatorFactory(
        "mace-mh1",
        make_mace_mh1,
        "MACE-MH-1, head omat_pbe, float64 (needs mace-torch>=0.3.14); options head, device, dispersion, model",
        relax_settings=dict(RECOMMENDED_ML_RELAXATION),
    ),
    CalculatorFactory(
        "uma",
        make_uma,
        "UMA through fairchem (needs fairchem-core>=2.24 and access to the gated weights): uma-s-1p2p1, "
        "task oc22 (oxide slabs with adsorbates); options model, task, device, inference_settings, dispersion",
        relax_settings=dict(RECOMMENDED_ML_RELAXATION),
    ),
    CalculatorFactory(
        "sevennet",
        make_sevennet,
        "SevenNet-Omni (needs sevenn>=0.13), modal oc22 (oxide slabs with adsorbates); options model, modal, "
        "device, dispersion",
        relax_settings=dict(RECOMMENDED_ML_RELAXATION),
    ),
)


def register(registry):
    for entry in BUILTIN_CALCULATORS:
        registry.add_calculator(entry.name, entry.make, entry.description, relax_settings=entry.relax_settings)


def parse_options(pairs):
    """``["model=medium", "dispersion=true"]`` -> {"model": "medium", "dispersion": True}; values read as YAML."""
    import yaml

    options = {}
    for pair in pairs or ():
        key, sep, value = str(pair).partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--option needs key=value, not {pair!r}")
        options[key.strip()] = yaml.safe_load(value) if value.strip() else ""
    return options


def _import(spec):
    module_name, _, attribute = spec.partition(":")
    try:
        obj = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"calculator {spec!r}: cannot import {module_name!r} ({exc})") from exc
    for part in attribute.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            raise ValueError(f"calculator {spec!r}: {module_name!r} has no {attribute!r}") from None
    return obj


def load_calculator(spec, options=None, registry=None, optimizer=None):
    """The :class:`Calculator` a spec names, made with ``options`` (see the module docstring).

    ``optimizer``: None takes the one the factory recommends, else BFGS. The returned
    :class:`ASECalculator` carries the factory's recommended fmax and steps
    (``recommended``) and says where its optimizer came from (``optimizer_source``: given,
    calculator or default).
    """
    options = dict(options or {})
    known = dict(getattr(registry, "calculators", None) or {})
    if not known:
        known = {entry.name: entry for entry in BUILTIN_CALCULATORS}
    spec = str(spec).strip()
    if spec in known:
        factory = known[spec].make
        recommended = check_relax_settings(getattr(known[spec], "relax_settings", None), f"calculator {spec!r}")
    elif ":" in spec:
        factory = _import(spec)
        recommended = check_relax_settings(getattr(factory, "relax_settings", None), f"calculator {spec!r}")
    else:
        raise ValueError(f"Unknown calculator {spec!r}: give one of {', '.join(sorted(known))} or module:callable")
    if not callable(factory):
        raise ValueError(f"calculator {spec!r} is not callable")
    if optimizer is not None and optimizer not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {optimizer!r}; choose one of {', '.join(OPTIMIZERS)}")
    chosen = optimizer or recommended.get("optimizer") or DEFAULT_OPTIMIZER
    made = factory(**options)
    if isinstance(made, Calculator):
        return made
    if callable(getattr(made, "get_potential_energy", None)) or callable(getattr(made, "calculate", None)):
        calculator = ASECalculator(made, spec, chosen, options, recommended)
        calculator.optimizer_source = (
            "given" if optimizer is not None else ("calculator" if "optimizer" in recommended else "default")
        )
        return calculator
    raise ValueError(
        f"calculator {spec!r} returned {type(made).__name__}, which is neither an ASE calculator "
        "nor an object with describe() and relax()"
    )


def fingerprint(description, settings):
    """Short hash of a calculator description and the relaxation settings (reuse of results)."""
    text = json.dumps({"calculator": description, "settings": settings}, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]
