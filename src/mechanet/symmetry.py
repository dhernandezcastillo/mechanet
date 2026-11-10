"""Approximate lattice translations of a slab, used to merge equivalent sites and nodes.

A relaxed slab is never exactly symmetric. A pure lattice translation is accepted when
every atom lies within ``tol`` (A) of a same-element image after a least-squares shift,
and every supplied per-atom mask (selective dynamics, frequency region) and optional
per-atom value (an explicit MAGMOM map) is preserved. The residuals are reported, so
the user sees how far from exact each translation is.

Translations never flip the slab or rotate adsorbate orientations, so nodes related by
one can be treated as the same calculation without any change of the starting geometry
other than the shift. Rotations and mirrors are reported by the inspector (spglib) but
not used for merging nodes. :func:`surface_images` gives them for comparing relaxed
structures, where two minima related by one are the same minimum.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

DEFAULT_SYMMETRY_TOLERANCE_A = 0.005


def lattice_translations(structure, masks=(), atom_values=None, tol=DEFAULT_SYMMETRY_TOLERANCE_A):
    """Approximate pure lattice translations mapping the structure (and masks) onto itself.

    A trial shift from one pair of atoms of the least common element gives the atom
    permutation; the translation is then the least-squares shift over all atoms.
    Returns [{"fractional", "permutation", "max_residual_A", "rms_residual_A"}].
    """
    symbols = structure.symbols
    counts = Counter(symbols)
    ref = min(counts, key=lambda s: (counts[s], s))
    ref_atoms = [i for i, s in enumerate(symbols) if s == ref]
    inv = np.linalg.inv(structure.cell)
    frac = structure.positions @ inv
    by_element = {s: [i for i, x in enumerate(symbols) if x == s] for s in counts}
    found = []
    for j in ref_atoms[1:]:
        shift = frac[j] - frac[ref_atoms[0]]
        shift -= np.rint(shift)
        perm, ok = [], True
        for i, sym in enumerate(symbols):
            candidates = by_element[sym]
            d = frac[candidates] - (frac[i] + shift)
            d -= np.rint(d)
            lengths = np.linalg.norm(d @ structure.cell, axis=1)
            k = int(np.argmin(lengths))
            if lengths[k] > 2.0 * tol:  # the trial shift carries one atom's own deviation
                ok = False
                break
            perm.append(candidates[k])
        if not ok or len(set(perm)) != len(perm):
            continue
        delta = frac[perm] - frac - shift
        delta -= np.rint(delta)
        shift = shift + delta.mean(axis=0)
        residual = frac[perm] - frac - shift
        residual -= np.rint(residual)
        lengths = np.linalg.norm(residual @ structure.cell, axis=1)
        if float(lengths.max()) > tol:
            continue
        if any(not np.array_equal(np.asarray(m)[i], np.asarray(m)[perm[i]]) for m in masks for i in range(len(perm))):
            continue
        if atom_values is not None and any(atom_values[i] != atom_values[perm[i]] for i in range(len(perm))):
            continue
        found.append(
            {
                "fractional": shift.tolist(),
                "permutation": perm,
                "max_residual_A": float(lengths.max()),
                "rms_residual_A": float(np.sqrt(np.mean(lengths**2))),
            }
        )
    return found


def spglib_available():
    """True when spglib can be imported (rotations and mirrors are found with it)."""
    try:
        import spglib  # noqa: F401
    except ImportError:
        return False
    return True


def describe_operation(rotation, normal):
    """'turned by 120°', 'mirrored' or 'shifted' for a Cartesian operation that keeps the surface normal."""
    if np.linalg.det(rotation) < 0:
        return "mirrored"
    if np.allclose(rotation, np.eye(3)):
        return "shifted"
    normal = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
    trial = np.eye(3)[int(np.argmin(np.abs(normal)))]
    u = trial - (trial @ normal) * normal
    u /= np.linalg.norm(u)
    v = rotation @ u
    angle = np.degrees(np.arctan2(normal @ np.cross(u, v), u @ v)) % 360
    return f"turned by {angle:.0f}°"


def _dataset(structure, types, symprec):
    """spglib's symmetry dataset, atoms told apart by the integer ``types``; None when spglib fails."""
    try:
        import spglib
    except ImportError:  # pragma: no cover
        return None
    import warnings

    frac = structure.positions @ np.linalg.inv(structure.cell)
    try:
        with warnings.catch_warnings():  # spglib 2.x warns about its legacy error handling
            warnings.simplefilter("ignore", DeprecationWarning)
            return spglib.get_symmetry_dataset((structure.cell, frac, np.asarray(types)), symprec=symprec)
    except Exception:  # noqa: BLE001 - spglib raises on degenerate input in some versions
        return None


def surface_operations(structure, normal, fixed=None, symprec=0.01):
    """Number of spglib operations that keep the surface normal and the fixed/movable mask.

    Diagnostic only (reported by the inspector); returns None when spglib fails.
    """
    from ase.data import atomic_numbers

    numbers = np.array([atomic_numbers[s] for s in structure.symbols])
    if fixed is not None:
        numbers = numbers + 200 * np.asarray(fixed, dtype=int)
    dataset = _dataset(structure, numbers, symprec)
    if dataset is None:
        return None
    lattice = structure.cell.T  # columns are lattice vectors
    inverse = np.linalg.inv(lattice)
    kept = 0
    for rotation in dataset.rotations:
        cartesian = lattice @ rotation @ inverse
        if np.allclose(cartesian @ normal, normal, atol=1e-3):
            kept += 1
    return {
        "symprec_A": symprec,
        "operations_total": int(len(dataset.rotations)),
        "operations_keeping_surface": kept,
        "spacegroup": dataset.international,
    }


def surface_images(structure, normal, labels=None, symprec=DEFAULT_SYMMETRY_TOLERANCE_A):
    """Cartesian operations ``x -> R @ x + t`` that map the slab onto itself and keep its outward normal.

    Atoms are told apart by element and, when given, by ``labels`` (one hashable per atom, such as
    whether it is fixed and its initial magnetic moment), so no operation exchanges a fixed atom
    with a movable one or a spin-up metal with a spin-down one. The identity comes first; it is
    the only operation when spglib fails. Two relaxed structures related by one of these are the
    same minimum seen from another place or angle (``mechanet prescreen`` merges them).
    """
    import json

    identity = (np.eye(3), np.zeros(3))
    # JSON text, so that a label may hold lists (a non-collinear MAGMOM, say)
    keys = [
        json.dumps([symbol, None if labels is None else labels[i]], sort_keys=True, default=str)
        for i, symbol in enumerate(structure.symbols)
    ]
    kinds = {key: k + 1 for k, key in enumerate(dict.fromkeys(keys))}
    dataset = _dataset(structure, [kinds[key] for key in keys], symprec)
    if dataset is None:
        return [identity]
    lattice = structure.cell.T  # columns are lattice vectors
    inverse = np.linalg.inv(lattice)
    normal = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
    images = [identity]
    for rotation, translation in zip(dataset.rotations, dataset.translations):
        cartesian = lattice @ rotation @ inverse
        if not np.allclose(cartesian @ normal, normal, atol=1e-3):
            continue
        if np.array_equal(rotation, np.eye(3, dtype=int)) and np.allclose(translation - np.rint(translation), 0):
            continue
        images.append((cartesian, lattice @ translation))
    return images
