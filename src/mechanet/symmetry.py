"""Approximate lattice translations of a slab, used to merge equivalent sites and nodes.

A relaxed slab is never exactly symmetric. A pure lattice translation is accepted when
every atom lies within ``tol`` (A) of a same-element image after a least-squares shift,
and every supplied per-atom mask (selective dynamics, frequency region) and optional
per-atom value (an explicit MAGMOM map) is preserved. The residuals are reported, so
the user sees how far from exact each translation is.

Translations never flip the slab or rotate adsorbate orientations, so nodes related by
one can be treated as the same calculation without any change of the starting geometry
other than the shift. Rotations and mirrors are reported by the inspector (spglib) but
not used for merging.
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


def surface_operations(structure, normal, fixed=None, symprec=0.01):
    """Number of spglib operations that keep the surface normal and the fixed/movable mask.

    Diagnostic only (reported by the inspector); returns None when spglib fails.
    """
    try:
        import spglib
    except ImportError:  # pragma: no cover
        return None
    from ase.data import atomic_numbers

    numbers = np.array([atomic_numbers[s] for s in structure.symbols])
    if fixed is not None:
        numbers = numbers + 200 * np.asarray(fixed, dtype=int)
    frac = structure.positions @ np.linalg.inv(structure.cell)
    import warnings

    try:
        with warnings.catch_warnings():  # spglib 2.x warns about its legacy error handling
            warnings.simplefilter("ignore", DeprecationWarning)
            dataset = spglib.get_symmetry_dataset((structure.cell, frac, numbers), symprec=symprec)
    except Exception:  # noqa: BLE001 - spglib raises on degenerate input in some versions
        return None
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
