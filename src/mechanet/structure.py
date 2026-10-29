"""Atomic structures with persistent atom IDs and VASP selective-dynamics flags.

Positions are Cartesian Angstrom, lattice vectors are rows, and ``mobile`` holds the
selective-dynamics booleans (True means movable). Every atom carries a stable ID
(``a0001`` ... for the slab as read; adsorbate atoms get role IDs such as ``h_a0061``),
so masks, magnetic moments and topology checks follow atoms through reordering.

POSCAR files are read and written by this module (VASP 5+ format with explicit species,
including VASP 6 "Fe/1a2b3c4d" species tokens). Any other format ASE understands is read
through :func:`read_structure`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

ELEMENTS = frozenset(
    """
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb
Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf
Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og""".split()
)


def species_symbol(token):
    """Element symbol from a POSCAR/CONTCAR species token.

    VASP 6 builds with HDF5 write POTCAR labels with a hash ("Fe/1a2b3c4d",
    "K_sv/5e6f7a8b"); POTCAR variants ("K_sv", "O_h", "H.75") name one element.
    """
    base = str(token).split("/", 1)[0].split("_", 1)[0]
    match = re.match(r"([A-Z][a-z]?)", base)
    symbol = match.group(1) if match else ""
    if symbol not in ELEMENTS or not re.fullmatch(r"[A-Z][a-z]?(?:[.0-9]+)?", base):
        raise ValueError(f"Unrecognized species token {token!r} on the POSCAR species line")
    return symbol


@dataclass
class Structure:
    symbols: list
    positions: np.ndarray
    cell: np.ndarray
    mobile: np.ndarray
    ids: list

    def __post_init__(self):
        self.symbols = list(self.symbols)
        self.ids = list(self.ids)
        self.positions = np.asarray(self.positions, dtype=float)
        self.cell = np.asarray(self.cell, dtype=float)
        self.mobile = np.asarray(self.mobile, dtype=bool)
        n = len(self.symbols)
        if self.positions.shape != (n, 3) or self.mobile.shape != (n, 3):
            raise ValueError("Positions and selective flags must have shape (N,3)")
        if self.cell.shape != (3, 3) or abs(np.linalg.det(self.cell)) < 1e-8:
            raise ValueError("A nonsingular 3x3 cell is required")
        if len(self.ids) != n or len(set(self.ids)) != n:
            raise ValueError("Atom IDs must be unique and match the atom count")
        if not np.isfinite(self.positions).all() or not np.isfinite(self.cell).all():
            raise ValueError("Non-finite coordinates are not permitted")

    def __len__(self):
        return len(self.symbols)

    def copy(self):
        return Structure(
            self.symbols.copy(), self.positions.copy(), self.cell.copy(), self.mobile.copy(), self.ids.copy()
        )

    def index(self, atom_id):
        return self.ids.index(atom_id)

    @property
    def species_order(self):
        """Elements in order of first appearance."""
        return list(dict.fromkeys(self.symbols))

    def to_ase(self):
        from ase import Atoms
        from ase.constraints import FixAtoms

        atoms = Atoms(self.symbols, positions=self.positions, cell=self.cell, pbc=True)
        fixed = [i for i, flags in enumerate(self.mobile) if not flags.any()]
        if fixed:
            atoms.set_constraint(FixAtoms(indices=fixed))
        return atoms


def default_ids(n):
    return [f"a{i + 1:04d}" for i in range(n)]


def read_poscar(path):
    lines = Path(path).read_text().splitlines()
    if len(lines) < 8:
        raise ValueError(f"Not a complete POSCAR: {path}")
    scalars = [float(x) for x in lines[1].split()]
    raw_cell = np.array([[float(x) for x in line.split()[:3]] for line in lines[2:5]])
    if len(scalars) != 1:
        raise ValueError("This reader requires the usual single POSCAR scale factor")
    scale = scalars[0]
    if scale < 0:
        scale = (abs(scale) / abs(np.linalg.det(raw_cell))) ** (1 / 3)
    if scale == 0:
        raise ValueError("POSCAR scale must be nonzero")
    cell = scale * raw_cell
    tokens = lines[5].split()
    if not tokens or all(re.fullmatch(r"[0-9]+", t) for t in tokens):
        raise ValueError("A VASP5+ POSCAR with explicit element symbols is required")
    elements = [species_symbol(token) for token in tokens]
    try:
        counts = [int(x) for x in lines[6].split()]
    except ValueError as exc:
        raise ValueError("A VASP5+ POSCAR with explicit element symbols is required") from exc
    if len(elements) != len(counts) or any(c < 0 for c in counts):
        raise ValueError("Species and atom-count lines disagree")
    symbols = [symbol for symbol, count in zip(elements, counts) for _ in range(count)]
    cursor = 7
    selective = lines[cursor].strip().lower().startswith("s")
    if selective:
        cursor += 1
    direct = lines[cursor].strip().lower().startswith("d")
    if not direct and not lines[cursor].strip().lower().startswith(("c", "k")):
        raise ValueError("Unrecognized POSCAR coordinate mode")
    cursor += 1
    rows = [line.split() for line in lines[cursor : cursor + len(symbols)]]
    if len(rows) != len(symbols) or any(len(row) < 3 for row in rows):
        raise ValueError("Incomplete POSCAR atomic coordinates")
    pos = np.array([[float(x) for x in row[:3]] for row in rows])
    mobile = np.ones((len(symbols), 3), dtype=bool)
    if selective:
        if any(len(row) < 6 or any(x.upper() not in ("T", "F") for x in row[3:6]) for row in rows):
            raise ValueError("Invalid selective dynamics flags")
        mobile = np.array([[x.upper() == "T" for x in row[3:6]] for row in rows])
    positions = pos @ cell if direct else pos * scale
    return Structure(symbols, positions, cell, mobile, default_ids(len(symbols)))


def from_ase(atoms):
    """Structure from an ASE Atoms object; FixAtoms/FixScaled become selective-dynamics flags."""
    n = len(atoms)
    mobile = np.ones((n, 3), dtype=bool)
    for constraint in atoms.constraints:
        name = type(constraint).__name__
        if name == "FixAtoms":
            mobile[constraint.get_indices()] = False
        elif name == "FixScaled":
            index = np.atleast_1d(constraint.index)
            mask = np.asarray(constraint.mask, dtype=bool).reshape(-1, 3)
            mobile[index] &= ~mask
        elif name == "FixCartesian":
            index = np.atleast_1d(constraint.index)
            mask = np.asarray(constraint.mask, dtype=bool).reshape(-1, 3)
            mobile[index] &= ~mask
    if not atoms.cell.rank == 3:
        raise ValueError("The structure needs a full 3D cell (slab plus vacuum)")
    return Structure(atoms.get_chemical_symbols(), atoms.positions, atoms.cell.array, mobile, default_ids(n))


def read_structure(path):
    """Read POSCAR/CONTCAR with this module, anything else (CIF, extxyz, ...) through ASE."""
    path = Path(path)
    name = path.name.upper()
    if name.startswith(("POSCAR", "CONTCAR")) or path.suffix.lower() in (".vasp", ".poscar"):
        return read_poscar(path)
    try:
        return read_poscar(path)
    except (ValueError, IndexError):
        from ase.io import read

        return from_ase(read(path))


def species_grouped(symbols):
    """True when every element occupies one contiguous block, as a POSCAR requires."""
    seen = []
    for sym in symbols:
        if seen and seen[-1] == sym:
            continue
        if sym in seen:
            return False
        seen.append(sym)
    return True


def grouped_copy(structure):
    """The structure with its atoms grouped by element (first-appearance order, stable within each).

    Atom IDs are renumbered in the new order (a0001, ...), as reading the written POSCAR back gives.
    Returns (structure, permutation): new atom k is old atom permutation[k].
    """
    permutation = species_permutation(structure.symbols, structure.species_order)
    return (
        Structure(
            [structure.symbols[i] for i in permutation],
            structure.positions[permutation],
            structure.cell.copy(),
            structure.mobile[permutation],
            default_ids(len(structure)),
        ),
        permutation,
    )


def species_blocks(symbols):
    """Return contiguous species/count blocks; reject repeated blocks."""
    names, counts = [], []
    for sym in symbols:
        if not names or sym != names[-1]:
            if sym in names:
                raise ValueError("POSCAR writer requires grouped species; use species_permutation")
            names.append(sym)
            counts.append(1)
        else:
            counts[-1] += 1
    return names, counts


def species_permutation(symbols, order):
    """Permutation that groups atoms by ``order`` (elements not in it follow alphabetically)."""
    order = list(order) + sorted(set(symbols) - set(order))
    return np.array([i for symbol in order for i, s in enumerate(symbols) if s == symbol], dtype=int)


def reorder(structure, permutation):
    p = np.asarray(permutation, dtype=int)
    return Structure(
        [structure.symbols[i] for i in p],
        structure.positions[p],
        structure.cell.copy(),
        structure.mobile[p],
        [structure.ids[i] for i in p],
    )


def write_poscar(path, structure, comment="mechanet structure; atom IDs in state.json"):
    """Write without sorting: callers apply one permutation to ALL metadata."""
    names, counts = species_blocks(structure.symbols)
    frac = structure.positions @ np.linalg.inv(structure.cell)
    lines = [comment, "1.0"]
    lines += [" ".join(f"{v:21.15f}" for v in row) for row in structure.cell]
    lines += [" ".join(names), " ".join(map(str, counts)), "Selective dynamics", "Direct"]
    for f, flags in zip(frac, structure.mobile):
        lines.append(" ".join(f"{x:20.14f}" for x in f) + "   " + " ".join("T" if x else "F" for x in flags))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def write_xyz(path, structure):
    """Extended XYZ for independent inspection (constraints also in POSCAR)."""
    lattice = " ".join(f"{x:.12f}" for x in structure.cell.reshape(-1))
    lines = [str(len(structure)), f'Lattice="{lattice}" Properties=species:S:1:pos:R:3:id:S:1 pbc="T T T"']
    lines += [
        f"{s:2s} " + " ".join(f"{x:.10f}" for x in p) + f" {aid}"
        for s, p, aid in zip(structure.symbols, structure.positions, structure.ids)
    ]
    Path(path).write_text("\n".join(lines) + "\n")


OFFSETS27 = np.array(list(product((-1, 0, 1), repeat=3)), dtype=float)


def minimum_image(vector, cell, periodic=(True, True, True)):
    """Nearest image, checking adjacent cells after fractional reduction (skewed cells included)."""
    vector = np.asarray(vector, dtype=float)
    cell = np.asarray(cell, dtype=float)
    frac = vector @ np.linalg.inv(cell)
    frac = frac - np.where(periodic, np.rint(frac), 0)
    offsets = np.array(list(product(*[(-1, 0, 1) if p else (0,) for p in periodic])))
    trial = (frac[None, :] + offsets) @ cell
    return trial[np.argmin(np.einsum("ij,ij->i", trial, trial))]


def distance(structure, i, j):
    return float(np.linalg.norm(minimum_image(structure.positions[j] - structure.positions[i], structure.cell)))


def pbc_distances(points, others, cell, inv_cell=None):
    """Minimum-image distance matrix between two point sets (skew-safe)."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    others = np.asarray(others, dtype=float).reshape(-1, 3)
    if not len(points) or not len(others):
        return np.full((len(points), len(others)), np.inf)
    inv_cell = np.linalg.inv(cell) if inv_cell is None else inv_cell
    frac = (points[:, None, :] - others[None, :, :]) @ inv_cell
    frac -= np.rint(frac)
    best = np.full(frac.shape[:2], np.inf)
    for offset in OFFSETS27:
        cart = (frac + offset) @ cell
        best = np.minimum(best, np.einsum("pqk,pqk->pq", cart, cart))
    return np.sqrt(best)


def distance_matrix(structure):
    vec = structure.positions[:, None, :] - structure.positions[None, :, :]
    frac = vec @ np.linalg.inv(structure.cell)
    frac -= np.rint(frac)
    squared = np.full((len(structure), len(structure)), np.inf)
    for shift in product((-1, 0, 1), repeat=3):
        trial = (frac + np.array(shift)) @ structure.cell
        squared = np.minimum(squared, np.einsum("ijk,ijk->ij", trial, trial))
    return np.sqrt(squared)


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        raise ValueError("Cannot normalize a zero vector")
    return vector / norm


def slab_normal(cell, vacuum_axis=0, sign=1):
    """Unit normal of the periodic plane, pointing along +vacuum axis times ``sign``."""
    axes = [i for i in range(3) if i != vacuum_axis]
    normal = unit(np.cross(cell[axes[0]], cell[axes[1]]))
    if normal @ cell[vacuum_axis] < 0:
        normal = -normal
    return float(sign) * normal
