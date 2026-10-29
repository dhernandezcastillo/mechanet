"""Element roles and per-element data.

A role says how an element takes part in surface chemistry:

* ``metal``      atoms that carry ligands and form active sites (transition metals, Al, Ga, ...)
* ``ligand``     anionic framework or adsorbate atoms bound to metals (O, N, S, halides, ...)
* ``H``          hydrogen
* ``spectator``  alkali and alkaline-earth cations (K+ in an LDH interlayer, Sr in SrIrO3, ...)
* ``nonmetal``   covalent framework atoms that are neither ligands nor metals (C, B, Si, P, ...)
* ``other``      noble gases and anything without a rule

Roles are proposals: a project file can override them per element (for example
framework Mg in an MgAl LDH, or La treated as a spectator A-site).
"""

from __future__ import annotations

from ase.data import atomic_masses, atomic_numbers, covalent_radii

LIGANDS = frozenset({"O", "N", "S", "Se", "Te", "F", "Cl", "Br", "I"})
HALOGENS = frozenset({"F", "Cl", "Br", "I"})
SPECTATORS = frozenset({"Li", "Na", "K", "Rb", "Cs", "Fr", "Be", "Mg", "Ca", "Sr", "Ba", "Ra"})
NONMETALS = frozenset({"B", "C", "Si", "P", "Ge", "As", "Sb"})
NOBLE = frozenset({"He", "Ne", "Ar", "Kr", "Xe", "Rn"})
ROLES = ("metal", "ligand", "H", "spectator", "nonmetal", "other")

# Floors (A) below which a metal-ligand contact counts as a bond even when the scaled
# covalent radii give a shorter cutoff; 2.45 A for M-O follows long Jahn-Teller and
# hydroxide M-O bonds in 3d oxyhydroxides.
METAL_LIGAND_FLOOR_A = {"O": 2.45, "N": 2.5, "F": 2.3, "S": 2.8, "Se": 2.9, "Te": 3.1, "Cl": 2.9, "Br": 3.0, "I": 3.2}
BOND_SCALE = 1.20
OH_BOND_MAX_A = 1.25


def default_role(symbol: str) -> str:
    if symbol == "H":
        return "H"
    if symbol in LIGANDS:
        return "ligand"
    if symbol in SPECTATORS:
        return "spectator"
    if symbol in NONMETALS:
        return "nonmetal"
    if symbol in NOBLE or symbol not in atomic_numbers:
        return "other"
    return "metal"


def roles_for(symbols, overrides=None) -> dict:
    """Element -> role for the elements present, with optional per-element overrides."""
    overrides = dict(overrides or {})
    bad = {k: v for k, v in overrides.items() if v not in ROLES}
    if bad:
        raise ValueError(f"Unknown element roles {bad}; choose from {ROLES}")
    return {s: overrides.get(s, default_role(s)) for s in dict.fromkeys(symbols)}


def covalent_radius(symbol: str) -> float:
    return float(covalent_radii[atomic_numbers[symbol]])


def standard_mass(symbol: str) -> float:
    return float(atomic_masses[atomic_numbers[symbol]])


def bond_cutoff(a: str, b: str, roles: dict, scale: float = BOND_SCALE) -> float:
    """Distance (A) below which two atoms count as bonded in the slab graph."""
    if {a, b} == {"O", "H"}:
        return OH_BOND_MAX_A
    cutoff = scale * (covalent_radius(a) + covalent_radius(b))
    role_a, role_b = roles.get(a, default_role(a)), roles.get(b, default_role(b))
    if role_a == "metal" and role_b == "ligand":
        cutoff = max(cutoff, METAL_LIGAND_FLOOR_A.get(b, 0.0))
    elif role_b == "metal" and role_a == "ligand":
        cutoff = max(cutoff, METAL_LIGAND_FLOOR_A.get(a, 0.0))
    return cutoff


def metal_ligand_cutoff(metal: str, ligand: str, scale: float = BOND_SCALE) -> float:
    return max(scale * (covalent_radius(metal) + covalent_radius(ligand)), METAL_LIGAND_FLOOR_A.get(ligand, 0.0))


# Formal charges used only for the electron-count diagnostic of the inspector.
ANION_CHARGE = {"O": -2, "N": -3, "S": -2, "Se": -2, "Te": -2, "F": -1, "Cl": -1, "Br": -1, "I": -1}
CATION_CHARGE = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1, "Be": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2}

# Starting moments offered by `mechanet init` as the "high-spin" choice (muB per atom).
# They are initial guesses for the SCF, not a statement about the converged spin state.
HIGH_SPIN_MAGMOM = {
    "Ti": 1.0,
    "V": 3.0,
    "Cr": 3.0,
    "Mn": 5.0,
    "Fe": 5.0,
    "Co": 3.0,
    "Ni": 2.0,
    "Cu": 1.0,
    "Mo": 1.0,
    "Ru": 1.0,
    "Rh": 1.0,
    "Ir": 1.0,
    "Ce": 1.0,
}
