"""In-process calculators: relaxations that run inside Python, without a batch queue.

DFT runs go through the staged runner (:mod:`mechanet.runner`): restartable SLURM jobs whose
inputs and outputs are hashed and checked. A *calculator* here is anything that gives energies
and forces inside the Python process: a machine-learned interatomic potential (MACE, UMA, ...),
or ASE's toy EMT potential for tests. ``mechanet prescreen`` (:mod:`mechanet.prescreen`) uses
one to relax the generated starts before any DFT and judges the relaxed structures with the
same checks as DFT results.

A calculator is named by a spec:

* a name the registry knows: ``emt`` and ``mace-mp`` are built in, a plugin adds others with
  ``registry.add_calculator(name, factory, description)`` (or the ``mechanet.calculators``
  entry-point group);
* ``module:callable``: a factory importable from Python, such as ``mace.calculators:mace_mp``.

The factory is called with the options (``--option key=value``) and returns either an ASE
calculator, which :class:`ASECalculator` wraps, or an object with ``describe()`` and
``relax(structure, fmax, steps)`` (the :class:`Calculator` protocol).
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np

OPTIMIZERS = ("BFGS", "LBFGS", "FIRE")
DEFAULT_FMAX_EVA = 0.05
DEFAULT_STEPS = 300


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


def _distribution(module_name):
    """(distribution, version) that provides a module, or (top-level name, None)."""
    from importlib.metadata import PackageNotFoundError, packages_distributions, version

    top = str(module_name).split(".")[0]
    for dist in packages_distributions().get(top, [top]):
        try:
            return dist, version(dist)
        except PackageNotFoundError:
            continue
    return top, None


class ASECalculator:
    """Relaxes with any ASE calculator: an ML potential, EMT, or a code ASE drives."""

    def __init__(self, calc, label, optimizer="BFGS", options=None):
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer {optimizer!r}; choose one of {', '.join(OPTIMIZERS)}")
        self.calc = calc
        self.label = str(label)
        self.optimizer = optimizer
        self.options = dict(options or {})

    def describe(self):
        import ase

        cls = type(self.calc)
        package, package_version = _distribution(cls.__module__)
        return {
            "calculator": self.label,
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "package": package,
            "package_version": package_version,
            "ase_version": ase.__version__,
            "options": self.options,
            "optimizer": self.optimizer,
        }

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
)


def register(registry):
    for entry in BUILTIN_CALCULATORS:
        registry.add_calculator(entry.name, entry.make, entry.description)


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


def load_calculator(spec, options=None, registry=None, optimizer="BFGS"):
    """The :class:`Calculator` a spec names, made with ``options`` (see the module docstring)."""
    options = dict(options or {})
    known = dict(getattr(registry, "calculators", None) or {})
    if not known:
        known = {entry.name: entry for entry in BUILTIN_CALCULATORS}
    spec = str(spec).strip()
    if spec in known:
        factory = known[spec].make
    elif ":" in spec:
        factory = _import(spec)
    else:
        raise ValueError(f"Unknown calculator {spec!r}: give one of {', '.join(sorted(known))} or module:callable")
    if not callable(factory):
        raise ValueError(f"calculator {spec!r} is not callable")
    made = factory(**options)
    if isinstance(made, Calculator):
        return made
    if callable(getattr(made, "get_potential_energy", None)) or callable(getattr(made, "calculate", None)):
        return ASECalculator(made, spec, optimizer, options)
    raise ValueError(
        f"calculator {spec!r} returned {type(made).__name__}, which is neither an ASE calculator "
        "nor an object with describe() and relax()"
    )


def fingerprint(description, settings):
    """Short hash of a calculator description and the relaxation settings (reuse of results)."""
    text = json.dumps({"calculator": description, "settings": settings}, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]
