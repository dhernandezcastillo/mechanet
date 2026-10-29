"""Slab inspection: what can be inferred about an electrocatalysis slab from the file alone.

:func:`inspect_slab` returns a JSON-serializable report with the detected slab frame
(vacuum axis, surface normal, active side), element roles, unbound molecules, layers,
ligand groups, open metal sites, shared protons, symmetry, site pairs, a formal electron
count and a *proposal*: the reactive sites, the frequency (partial-Hessian) region and
the fixed region that `mechanet init` writes into project.yaml for confirmation.

Every threshold is a parameter and is echoed in the report, so a proposal can always be
traced back to the rule that produced it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import elements
from .structure import Structure, read_structure, slab_normal, unit, minimum_image, pbc_distances, species_grouped
from .symmetry import lattice_translations, surface_operations, DEFAULT_SYMMETRY_TOLERANCE_A

DEFAULTS = {
    "bond_scale": elements.BOND_SCALE,
    "OH_max_A": elements.OH_BOND_MAX_A,
    "vacuum_min_A": 5.0,
    "layer_gap_A": 0.6,
    "exposure_radius_A": 1.7,
    "exposure_dz_A": 0.3,
    "pair_max_A": 3.3,
    "shared_proton_max_OH_A": 1.6,
    "molecule_max_atoms": 12,
    "symmetry_tolerance_A": DEFAULT_SYMMETRY_TOLERANCE_A,
    "open_site_O_distance_A": 1.75,
    "open_site_clearance_A": 1.2,
    "open_site_probe_radius_A": 2.2,
    "interior_min_thickness_A": 2.5,
}

LIGAND_SITE_ELEMENTS = ("O",)  # ligand sites the built-in adsorbates can transform
SITE_KINDS = ("terminal", "bridge", "mu3", "framework", "open_metal")


def open_site_id(metal_id):
    """Site ID of the open coordination site of metal ``a0012`` -> ``m0012``."""
    return "m" + metal_id[1:]


def _neighbor_graph(structure, roles, scale):
    from ase.neighborlist import neighbor_list

    atoms = structure.to_ase()
    atoms.set_constraint()
    symbols = structure.symbols
    elements_present = sorted(set(symbols))
    cut = {(a, b): elements.bond_cutoff(a, b, roles, scale) for a in elements_present for b in elements_present}
    i_list, j_list, d_list = neighbor_list("ijd", atoms, max(cut.values()))
    nbrs = defaultdict(list)
    for i, j, d in zip(i_list, j_list, d_list):
        if d <= cut[(symbols[i], symbols[j])]:
            nbrs[int(i)].append((int(j), float(d)))
    for i in nbrs:
        nbrs[i].sort(key=lambda x: (x[0], x[1]))
    return nbrs, cut


def find_vacuum(structure, idx, vacuum_min):
    """Largest periodic gap along each lattice vector, using only atoms ``idx``."""
    cell = structure.cell
    frac = (structure.positions @ np.linalg.inv(cell)) % 1.0
    frac = frac[idx]
    vol = abs(np.linalg.det(cell))
    rows = []
    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        spacing = vol / np.linalg.norm(np.cross(cell[j], cell[k]))
        f = np.sort(frac[:, i])
        gaps = np.diff(np.concatenate([f, [f[0] + 1.0]]))
        g = int(np.argmax(gaps))
        rows.append(
            {
                "axis": "abc"[i],
                "index": i,
                "largest_gap_A": float(gaps[g] * spacing),
                "gap_center_frac": float((f[g] + gaps[g] / 2.0) % 1.0),
                "spacing_A": float(spacing),
            }
        )
    best = max(rows, key=lambda r: r["largest_gap_A"])
    others = sorted(r["largest_gap_A"] for r in rows if r is not best)
    if best["largest_gap_A"] < vacuum_min:
        confidence = f"none: no gap above {vacuum_min:.1f} A (bulk input or wrong cell?)"
    elif others[-1] > vacuum_min:
        confidence = "ambiguous: two axes have large gaps"
    else:
        confidence = "high"
    return best, rows, confidence


def slab_heights(structure, axis, gap_center):
    """Height of every atom along the slab normal with the slab made contiguous."""
    cell = structure.cell
    j, k = [x for x in range(3) if x != axis]
    frac = (structure.positions @ np.linalg.inv(cell)) % 1.0
    shifted = (frac[:, axis] - gap_center) % 1.0
    spacing = abs(np.linalg.det(cell)) / np.linalg.norm(np.cross(cell[j], cell[k]))
    ufrac = frac.copy()
    ufrac[:, axis] = shifted
    return shifted * spacing, ufrac


def lateral_distances(cell, ufrac, normal, i, others, axis):
    """In-plane distances from atom i to ``others`` (skewed cells included)."""
    d = ufrac[others] - ufrac[i]
    jk = [m for m in range(3) if m != axis]
    for m in jk:
        d[:, m] -= np.rint(d[:, m])
    best = np.full(len(others), np.inf)
    for s1 in (-1, 0, 1):
        for s2 in (-1, 0, 1):
            shift = np.zeros(3)
            shift[jk[0]], shift[jk[1]] = s1, s2
            cart = (d + shift) @ cell
            lateral = cart - np.outer(cart @ normal, normal)
            best = np.minimum(best, np.linalg.norm(lateral, axis=1))
    return best


def cluster_layers(h, idx, gap):
    order = list(np.asarray(idx)[np.argsort(h[idx])])
    layers, current = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if h[b] - h[a] > gap:
            layers.append(current)
            current = []
        current.append(b)
    layers.append(current)
    return layers


def vacancy_direction(structure, metal, ligand_neighbors, normal, min_norm=0.25):
    """Direction of the missing ligand of an undercoordinated metal (blended toward the vacuum).

    The surface normal is used when the metal has no ligand, and when its ligand rays nearly
    cancel: four ligands in a plane (Fe-N4 in graphene, a square-planar site) leave a sum
    shorter than ``min_norm`` whose direction is numerical noise of the structure. A direction
    less than about 75 degrees from the normal is blended with it.
    """
    normal = np.asarray(normal, dtype=float)
    rays = [
        unit(minimum_image(structure.positions[j] - structure.positions[metal], structure.cell))
        for j in ligand_neighbors
    ]
    total = -np.sum(rays, axis=0) if rays else np.zeros(3)
    if np.linalg.norm(total) < min_norm:
        return normal.copy()
    direction = unit(total)
    if direction @ normal < 0.25:
        blended = direction + normal
        direction = unit(blended) if np.linalg.norm(blended) >= min_norm else normal.copy()
    return direction


def inspect_slab(
    path_or_structure, mask_path=None, options=None, role_overrides=None, vacuum_axis=None, surface_sign=None
):
    """Inspect a slab and propose sites, frequency region and fixed region (see module docstring)."""
    opts = {**DEFAULTS, **(options or {})}
    if isinstance(path_or_structure, Structure):
        structure, source = path_or_structure, None
    else:
        structure, source = read_structure(path_or_structure), str(path_or_structure)
    symbols = structure.symbols
    n_atoms = len(structure)
    roles = elements.roles_for(symbols, role_overrides)
    role = [roles[s] for s in symbols]
    report = {
        "schema_version": 1,
        "file": source,
        "n_atoms": n_atoms,
        "formula": dict(Counter(symbols)),
        "species_grouped": species_grouped(symbols),
        "roles": roles,
        "thresholds": opts,
    }

    nbrs, cutoffs = _neighbor_graph(structure, roles, opts["bond_scale"])
    report["bond_cutoffs_A"] = {f"{a}-{b}": round(v, 3) for (a, b), v in cutoffs.items() if a <= b}

    # Unbound molecules: small components without metal or spectator (contacts to spectator
    # cations ignored). They neither define the vacuum, count as layers, nor block sites.
    parent = list(range(n_atoms))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n_atoms):
        for j, _ in nbrs[i]:
            if "spectator" not in (role[i], role[j]):
                parent[find(i)] = find(j)
    components = defaultdict(list)
    for i in range(n_atoms):
        components[find(i)].append(i)
    molecular = np.zeros(n_atoms, dtype=bool)
    molecules = []
    for members in components.values():
        if len(members) <= opts["molecule_max_atoms"] and not any(role[i] in ("metal", "spectator") for i in members):
            molecular[members] = True
            molecules.append(sorted(members))
    slab_idx = np.array([i for i in range(n_atoms) if not molecular[i]])
    if not len(slab_idx):
        raise ValueError("No slab atoms found (every atom belongs to a small unbound molecule)")

    # Slab frame from slab atoms only.
    best, rows, confidence = find_vacuum(structure, slab_idx, opts["vacuum_min_A"])
    axis = best["index"] if vacuum_axis is None else int(vacuum_axis)
    if vacuum_axis is not None:
        best = rows[axis]
        confidence = "given"
    h, ufrac = slab_heights(structure, axis, best["gap_center_frac"])
    heavy = np.array([i for i in slab_idx if symbols[i] != "H"])
    fixed = ~structure.mobile.any(axis=1)
    mid = 0.5 * (h[heavy].max() + h[heavy].min())
    if surface_sign is not None:
        sign, side_conf = int(surface_sign), "given"
    elif fixed.any() and not fixed.all():
        sign = 1 if h[fixed].mean() < mid else -1
        side_conf = "high: the fixed atoms sit on the other side"
    else:
        sign, side_conf = 1, "low: no constraints in the file; the +axis side is assumed"
    normal = slab_normal(structure.cell, axis, sign)
    hs = sign * h
    top = float(hs[heavy].max())
    report["vacuum"] = {
        "axis": "abc"[axis],
        "axis_index": axis,
        "gap_A": round(best["largest_gap_A"], 2),
        "confidence": confidence,
        "all_axes": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows],
        "surface_normal": [round(float(x), 6) + 0.0 for x in normal],
        "normal_is_cartesian_z": bool(abs(abs(normal[2]) - 1) < 1e-3),
    }
    report["active_side"] = {"sign": sign, "confidence": side_conf}
    report["slab_thickness_A"] = round(float(hs[heavy].max() - hs[heavy].min()), 3)
    report["unbound_molecules"] = [
        {
            "atoms": [f"a{i + 1:04d}" for i in m],
            "formula": "".join(f"{e}{c if c > 1 else ''}" for e, c in sorted(Counter(symbols[i] for i in m).items())),
            "depth_below_top_A": round(top - float(max(hs[i] for i in m)), 2),
        }
        for m in molecules
    ]

    # Layers of heavy slab atoms, counted from the active surface down.
    layers = cluster_layers(hs, heavy, opts["layer_gap_A"])[::-1]
    report["layers"] = {
        "count": len(layers),
        "composition": [dict(Counter(symbols[i] for i in layer)) for layer in layers],
    }
    if fixed.any():
        report["fixed_region"] = {"source": "file", "n_fixed": int(fixed.sum()), "n_movable": int((~fixed).sum())}
    else:
        n_fix = len(layers) // 2
        ids = sorted(f"a{i + 1:04d}" for layer in layers[len(layers) - n_fix :] for i in layer)
        # H bound to a fixed heavy atom is fixed with it.
        fixed_set = {int(x[1:]) - 1 for x in ids}
        for i in range(n_atoms):
            if symbols[i] == "H" and not molecular[i] and any(j in fixed_set for j, _ in nbrs[i]):
                ids.append(f"a{i + 1:04d}")
        report["fixed_region"] = {
            "source": "proposal",
            "fix_bottom_layers": n_fix,
            "n_layers": len(layers),
            "fixed_ids": sorted(set(ids)),
            "note": "No selective dynamics in the file: the bottom half of the layers is proposed as fixed.",
        }

    metal_idx = [i for i in range(n_atoms) if role[i] == "metal"]

    def ligand_cn(i):
        return sum(1 for j, _ in nbrs[i] if role[j] == "ligand" and not molecular[j])

    def metal_cn(i):
        return sum(1 for j, _ in nbrs[i] if role[j] == "metal")

    lo, hi = hs[heavy].min(), hs[heavy].max()
    # A thin slab of one or two layers (a 2D sheet, a single-atom catalyst in graphene, flat or
    # corrugated) has no interior: all its atoms are surface atoms and cannot serve as the
    # bulk-like reference. Stepped and kinked surfaces, whose atomic planes lie closer than the
    # layer gap and merge into one "layer", keep their interior through their thickness.
    has_interior = len(layers) >= 3 or hi - lo > opts["interior_min_thickness_A"]
    inner = [i for i in metal_idx if lo + 0.25 * (hi - lo) <= hs[i] <= hi - 0.25 * (hi - lo)] if has_interior else []
    metallic = bool(metal_idx) and not any(ligand_cn(i) for i in metal_idx)
    cn_of = metal_cn if metallic else ligand_cn
    bulk_cn = {}
    for el in sorted({symbols[i] for i in metal_idx}):
        values = [cn_of(i) for i in inner if symbols[i] == el]
        origin = "interior atoms"
        if not values:
            values = [cn_of(i) for i in inner]
            origin = "host lattice: no interior atom of this element"
        if not values:
            values = [cn_of(i) for i in metal_idx if symbols[i] == el]
            origin = "most common over the slab: no interior metal"
        reference = Counter(values).most_common(1)[0][0] if values else 0
        bulk_cn[el] = {"cn": int(reference), "from": origin + ("; metal-metal, no ligands" if metallic else "")}
    report["bulk_like_coordination"] = bulk_cn

    def exposed(i):
        candidates = np.array([j for j in heavy if j != i and hs[j] - hs[i] > opts["exposure_dz_A"]])
        if len(candidates) == 0:
            return True
        lateral = lateral_distances(structure.cell, ufrac, normal, i, candidates, axis)
        return not np.any(lateral < opts["exposure_radius_A"])

    exposure = {i: exposed(i) for i in range(n_atoms) if symbols[i] != "H" and not molecular[i]}

    def ids(indices):
        return [structure.ids[j] for j in indices]

    ligand_sites, lattice_ligands = [], []
    for i in range(n_atoms):
        if role[i] != "ligand" or molecular[i] or not exposure.get(i):
            continue
        metals = sorted(j for j, _ in nbrs[i] if role[j] == "metal")
        hydrogens = sorted(j for j, _ in nbrs[i] if symbols[j] == "H")
        ligands = sorted(j for j, _ in nbrs[i] if role[j] == "ligand")
        kind = {0: None, 1: "terminal", 2: "bridge"}.get(len(metals), "mu3")
        native = {0: "O", 1: "OH", 2: "H2O"}.get(len(hydrogens)) if symbols[i] == "O" else None
        record = {
            "id": structure.ids[i],
            "index_1based": i + 1,
            "element": symbols[i],
            "kind": kind,
            "native": native,
            "metals": ids(metals),
            "hydrogens": ids(hydrogens),
            "depth_below_top_A": round(top - float(hs[i]), 2),
        }
        if ligands:
            record["ligand_neighbors"] = ids(ligands)
        if symbols[i] not in LIGAND_SITE_ELEMENTS or kind is None or native is None or ligands:
            record["reason"] = (
                "O-O or ligand-ligand bond"
                if ligands
                else "not bound to a metal"
                if kind is None
                else f"{symbols[i]} ligands are not covered by the built-in adsorbates"
                if symbols[i] not in LIGAND_SITE_ELEMENTS
                else "more than two H"
            )
            lattice_ligands.append(record)
        elif kind == "mu3":
            record["reason"] = "lattice-like (three or more metals); add it explicitly to make it reactive"
            lattice_ligands.append(record)
        else:
            ligand_sites.append(record)

    def axially_free(i, position):
        """The probe position above metal ``i`` is free and on the surface, not inside it.

        No slab atom within the clearance of the probe, and no heavy slab atom near it laterally
        (``open_site_probe_radius_A``) higher than the probe less 0.2 A: a probe above a buried
        atom lies among the atoms of the layer above (a dopant under a metal surface).
        """
        others = [j for j in slab_idx if j != i]
        if not others:
            return True
        d = pbc_distances(position, structure.positions[others], structure.cell)
        if d.min() < opts["open_site_clearance_A"]:
            return False
        vectors = np.array([minimum_image(structure.positions[j] - position, structure.cell) for j in others])
        up = vectors @ normal
        lateral = np.linalg.norm(vectors - np.outer(up, normal), axis=1)
        heavy_other = np.array([symbols[j] != "H" for j in others])
        return not bool(np.any(heavy_other & (lateral < opts["open_site_probe_radius_A"]) & (up > -0.2)))

    open_metals, covered = [], []
    for i in metal_idx:
        if not exposure.get(i):
            continue
        cn = cn_of(i)
        reference = bulk_cn[symbols[i]]["cn"]
        ligand_neighbors = [j for j, _ in nbrs[i] if role[j] == "ligand" and not molecular[j]]
        direction = vacancy_direction(structure, i, ligand_neighbors, normal)
        position = structure.positions[i] + opts["open_site_O_distance_A"] * direction
        if cn < reference:
            rule = "below the bulk-like coordination"
        elif not bulk_cn[symbols[i]]["from"].startswith("interior") and axially_free(i, position):
            # Nothing to compare with (the only atoms of this element are surface atoms, e.g.
            # a single-atom catalyst): an exposed metal whose axial position is free and on the
            # surface is open. A buried atom never is: its probe lies among the atoms above it.
            rule = "free axial position (no interior atom of this element to compare with)"
        else:
            covered.append({"id": structure.ids[i], "element": symbols[i], "cn": cn})
            continue
        open_metals.append(
            {
                "id": open_site_id(structure.ids[i]),
                "kind": "open_metal",
                "native": "empty",
                "metal_id": structure.ids[i],
                "index_1based": i + 1,
                "element": symbols[i],
                "cn": cn,
                "bulk_like_cn": reference,
                "rule": rule,
                "direction": [round(float(x), 6) for x in direction],
                "position_A": [round(float(x), 6) for x in position],
            }
        )
    report["ligand_sites"] = ligand_sites
    report["lattice_or_other_ligands"] = lattice_ligands
    report["open_metal_sites"] = open_metals
    report["exposed_fully_coordinated_metals"] = covered

    # Protons shared between two O (strong H-bonds): candidates for proton hops.
    shared = []
    o_atoms = [j for j in range(n_atoms) if symbols[j] == "O"]
    for i in range(n_atoms):
        if symbols[i] != "H":
            continue
        close = sorted(
            (float(np.linalg.norm(minimum_image(structure.positions[j] - structure.positions[i], structure.cell))), j)
            for j in o_atoms
        )
        close = [c for c in close if c[0] <= opts["shared_proton_max_OH_A"]]
        if len(close) >= 2 and close[0][0] <= opts["OH_max_A"]:
            shared.append(
                {
                    "H": structure.ids[i],
                    "donor": structure.ids[close[0][1]],
                    "acceptor": structure.ids[close[1][1]],
                    "donor_distance_A": round(close[0][0], 4),
                    "acceptor_distance_A": round(close[1][0], 4),
                }
            )
    report["shared_protons"] = shared

    # Translations (used by the builder to merge sites) and spglib operations (diagnostic).
    sites_all = ligand_sites + open_metals
    frequency_ids = proposed_frequency_atoms(ligand_sites)
    freq_mask = np.array([[aid in frequency_ids] * 3 for aid in structure.ids])
    translations = lattice_translations(
        structure, masks=(structure.mobile, freq_mask), tol=opts["symmetry_tolerance_A"]
    )
    site_classes = equivalent_sites(structure, sites_all, translations)
    report["symmetry"] = {
        "translations": [
            {"fractional": [round(x, 6) for x in t["fractional"]], "max_residual_A": round(t["max_residual_A"], 6)}
            for t in translations
        ],
        "tolerance_A": opts["symmetry_tolerance_A"],
        "site_classes": site_classes,
        "spglib": surface_operations(structure, normal, fixed),
    }

    # Two-site candidates.
    positions = {
        s["id"]: (
            structure.positions[s["index_1based"] - 1] if s["kind"] != "open_metal" else np.array(s["position_A"])
        )
        for s in sites_all
    }
    pairs = []
    keys = [s["id"] for s in sites_all]
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            d = float(np.linalg.norm(minimum_image(positions[keys[b]] - positions[keys[a]], structure.cell)))
            if d <= opts["pair_max_A"]:
                pairs.append([keys[a], keys[b], round(d, 3)])
    report["site_pairs"] = pairs

    # Formal electron count (diagnostic; not a magnetic model).
    counts = Counter(symbols)
    total = (
        sum(elements.ANION_CHARGE.get(s, 0) * c for s, c in counts.items())
        + counts.get("H", 0)
        + sum(elements.CATION_CHARGE.get(s, 0) * c for s, c in counts.items())
    )

    def region(i):
        if hs[i] >= hs[heavy].max() - 1.0:
            return "active_face"
        if hs[i] <= hs[heavy].min() + 1.0:
            return "opposite_face"
        return "interior"

    bare = Counter(
        region(i)
        for i in range(n_atoms)
        if symbols[i] == "O" and not molecular[i] and not any(symbols[j] in ("H", "O") for j, _ in nbrs[i])
    )
    average = round(-total / len(metal_idx), 3) if metal_idx else None
    note = "Assumes O2-, H+ and alkali/alkaline-earth cations at group valence; a diagnostic, not a spin model."
    # A covalent framework (C, B, Si, P ... bonded to the ligands, as N in N-doped graphene)
    # makes the ionic count meaningless when it holds most of the ligands; a few such bonds
    # (a carbonate on an oxide) do not.
    ligand_atoms = [i for i in slab_idx if role[i] == "ligand"]
    framework_bound = [
        i
        for i in ligand_atoms
        if any(role[j] == "nonmetal" and symbols[j] != "H" and not molecular[j] for j, _ in nbrs[i])
    ]
    framework = sorted(
        {symbols[j] for i in framework_bound for j, _ in nbrs[i] if role[j] == "nonmetal" and symbols[j] != "H"}
    )
    if not metal_idx:
        note = "Not computed: the slab has no metal atom."
    elif average is not None and ligand_atoms and 2 * len(framework_bound) > len(ligand_atoms):
        average = None
        note = (
            f"Not computed: a covalent framework ({', '.join(framework)}) holds most of the ligands "
            f"({len(framework_bound)} of {len(ligand_atoms)}), so an ionic count does not apply."
        )
    elif average is not None and abs(average) > 8:
        note = f"Not meaningful: the ionic count gives {average}, outside any oxidation state; it does not apply here."
        average = None
    report["formal_count"] = {
        "average_metal_oxidation_state": average,
        "bare_O_by_region": {k: bare.get(k, 0) for k in ("active_face", "interior", "opposite_face")},
        "note": note,
    }

    report["proposal"] = {
        "vacuum_axis": axis,
        "surface_sign": sign,
        "sites": [_site_entry(s) for s in sites_all],
        "frequency_atoms": sorted(frequency_ids),
        "spectator_molecules": [m["formula"] for m in report["unbound_molecules"]],
    }
    if mask_path:
        report["comparison_with_mask"] = compare_with_mask(structure, ligand_sites, mask_path)
    return report


def _site_entry(site):
    entry = {"id": site["id"], "kind": site["kind"], "native": site["native"]}
    if site["kind"] == "open_metal":
        entry["metal"] = site["metal_id"]
    return entry


def proposed_frequency_atoms(ligand_sites):
    """Reactive ligand atoms and their H: the partial-Hessian region kept in every node."""
    atoms = set()
    for site in ligand_sites:
        atoms.add(site["id"])
        atoms.update(site["hydrogens"])
    return atoms


def equivalent_sites(structure, sites, translations):
    """Group site IDs that lattice translations map onto each other."""
    index = {aid: i for i, aid in enumerate(structure.ids)}
    by_id = {s["id"]: s for s in sites}
    parent = {s: s for s in by_id}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for t in translations:
        perm = t["permutation"]
        for sid, site in by_id.items():
            anchor = site.get("metal_id") if site["kind"] == "open_metal" else sid
            image_atom = structure.ids[perm[index[anchor]]]
            image = open_site_id(image_atom) if site["kind"] == "open_metal" else image_atom
            if image in by_id:
                a, b = find(sid), find(image)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for sid in by_id:
        groups[find(sid)].append(sid)
    return [sorted(g) for g in sorted(groups.values())]


def compare_with_mask(structure, ligand_sites, mask_path):
    """Recall and precision of the proposed reactive O against a user's frequency-mask file (or structure)."""
    reference = mask_path if isinstance(mask_path, Structure) else read_structure(mask_path)
    if reference.symbols != structure.symbols:
        raise ValueError("The mask file must have the same atoms in the same order as the slab")
    marked = {
        structure.ids[i] for i in range(len(structure)) if structure.symbols[i] == "O" and reference.mobile[i].any()
    }
    found = {s["id"] for s in ligand_sites if s["element"] == "O"}
    return {
        "marked_O": sorted(marked),
        "proposed_O": sorted(found),
        "recall": round(len(marked & found) / len(marked), 3) if marked else None,
        "precision": round(len(marked & found) / len(found), 3) if found else None,
        "missed": sorted(marked - found),
        "extra": sorted(found - marked),
    }


def summary(report):
    """Human-readable multi-line summary of an inspection report."""
    v = report["vacuum"]
    out = [
        f"{report['file'] or 'structure'}: "
        + " ".join(f"{k}{n}" for k, n in report["formula"].items())
        + f" ({report['n_atoms']} atoms)"
    ]
    if not report.get("species_grouped", True):
        out.append(
            "  note: an element appears in more than one block of the file; `mechanet init` writes a copy "
            "grouped by element, and atom IDs follow that copy"
        )
    out.append(
        f"  vacuum along {v['axis']} (gap {v['gap_A']} A, {v['confidence']}); normal {v['surface_normal']}"
        + ("" if v["normal_is_cartesian_z"] else "  (not Cartesian z)")
    )
    side = "+" if report["active_side"]["sign"] > 0 else "-"
    out.append(
        f"  active side: {side}{v['axis']} [{report['active_side']['confidence']}]; "
        f"thickness {report['slab_thickness_A']} A, {report['layers']['count']} "
        + ("layer" if report["layers"]["count"] == 1 else "layers")
    )
    fr = report["fixed_region"]
    out.append(
        "  fixed region: "
        + (
            f"{fr['n_fixed']} fixed / {fr['n_movable']} movable atoms (from file)"
            if fr["source"] == "file"
            else f"proposal: bottom {fr['fix_bottom_layers']} of {fr['n_layers']} layers "
            f"({len(fr['fixed_ids'])} atoms) - confirm"
        )
    )
    roles = defaultdict(list)
    for element, role in report["roles"].items():
        roles[role].append(element)
    out.append("  roles: " + "; ".join(f"{r} {','.join(sorted(e))}" for r, e in sorted(roles.items())))
    out.append(
        "  bulk-like CN: "
        + ", ".join(f"{k}={x['cn']} ({x['from']})" for k, x in report["bulk_like_coordination"].items())
    )
    sites = report["proposal"]["sites"]
    kinds = Counter((s["kind"], s["native"]) for s in sites)
    out.append(
        f"  proposed reactive sites ({len(sites)}): "
        + (", ".join(f"{n}x {kind} {native}" for (kind, native), n in sorted(kinds.items())) or "none")
    )
    for s in sites:
        extra = f" on {s['metal']}" if s.get("metal") else ""
        out.append(f"    {s['id']:>8s}  {s['kind']:<10s} native {s['native']}{extra}")
    if report["lattice_or_other_ligands"]:
        reasons = Counter(x["reason"].split(";")[0] for x in report["lattice_or_other_ligands"])
        out.append(
            "  exposed ligands not proposed: " + ", ".join(f"{n} {reason}" for reason, n in reasons.most_common())
        )
    for s in report.get("open_metal_sites", []):
        if s.get("rule", "").startswith("free axial"):
            out.append(f"    {s['id']}: {s['rule']}")
    sym = report["symmetry"]
    out.append(
        f"  symmetry: {len(sym['translations'])} lattice translation(s) within {sym['tolerance_A']} A -> "
        f"{len(sym['site_classes'])} unique site class(es)"
        + (f"; spglib {sym['spglib']['operations_keeping_surface']} surface operation(s)" if sym.get("spglib") else "")
    )
    if report["shared_protons"]:
        out.append(
            "  shared protons: "
            + ", ".join(f"{s['H']} ({s['donor']} -> {s['acceptor']})" for s in report["shared_protons"])
        )
    out.append(f"  site pairs within {report['thresholds']['pair_max_A']} A: {len(report['site_pairs'])}")
    if report["unbound_molecules"]:

        def where(depth):
            return f"{depth} A below top" if depth >= 0 else f"{-depth} A above top"

        out.append(
            "  unbound molecules (spectators unless you decide otherwise): "
            + ", ".join(
                f"{m['formula']} ({m['atoms'][0]}, {where(m['depth_below_top_A'])})"
                for m in report["unbound_molecules"]
            )
        )
    out.append(
        f"  frequency region proposal: {len(report['proposal']['frequency_atoms'])} atoms "
        "(reactive ligand atoms and their H; every added atom is active too)"
    )
    fc = report["formal_count"]
    bare = fc["bare_O_by_region"]
    bare_text = (
        f"bare O: {bare['active_face']} on the active face, {bare['interior']} inside, "
        f"{bare['opposite_face']} on the opposite face"
        if "O" in report.get("formula", {"O": 1})
        else ""
    )
    if fc["average_metal_oxidation_state"] is None:
        out.append(f"  formal count: {fc['note'][0].lower()}{fc['note'][1:]} {bare_text}".rstrip())
    else:
        average = f"average metal oxidation state {fc['average_metal_oxidation_state']}"
        out.append(f"  formal count: {average}" + (f"; {bare_text}" if bare_text else ""))
    if "comparison_with_mask" in report:
        c = report["comparison_with_mask"]
        out.append(
            f"  vs mask file: recall {c['recall']}, precision {c['precision']}; "
            f"missed {', '.join(c['missed']) or 'none'}; extra {', '.join(c['extra']) or 'none'}"
        )
    return "\n".join(out)


def load_report(path):
    import json

    return json.loads(Path(path).read_text())
