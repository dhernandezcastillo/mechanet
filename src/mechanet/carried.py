"""Adsorbates a slab file already carries.

A slab can come with adsorbates on it: OOH or O2 on an oxygen, a chloride or a hydride on a
metal, CO. The sites of a campaign are oxygen ligands with zero to two H (their states O, OH
and H2O) and bare metals; anything else used to be treated as part of the framework without
a word: no site there, and no chemistry.

This module reads such a group as what it is: the state of a site.

* A group that hangs on the surface through one atom and is a registered adsorbate is
  **carried**: its atoms are taken out of the slab the campaign is referenced to, the site
  it sat on is a site of that slab, and the slab as given becomes a surface (a pattern of
  the sites, see :mod:`mechanet.surfaces`) that the campaign is built on, with the atoms
  where the file has them.

  - On an oxygen bound to one or two metals (OOH, O2, OCl): the oxygen stays, as a
    terminal or bridging site in the state O; the atoms beyond it are taken out.
  - On top of one metal through another element (Cl, H, CO, ...): the whole group is taken
    out, and the metal is an open metal site.

* A group that is no registered adsorbate, binds through several atoms (side-on O2, a
  bond between two sites) or sits between metals through an element other than oxygen (CO
  on a bridge, H in a hollow) is **kept as part of the framework**, and said so with its
  atoms and the reason.

* A lone anion between two or more metals (the sulfur of a sulfide, the fluorine of a
  fluoride, a bridging chloride) is a ligand of the slab, as it always was: the inspection
  lists it with the other ligands and this module says nothing about it.

Nothing is guessed: a group is carried only when its composition and its bonds are those of
exactly one way of writing a registered adsorbate.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .structure import Structure, minimum_image

SURFACE_NAME = "as-given"
# Role labels of the atoms the built-in adsorbate shapes place beyond the proximal atom.
ROLE_HELP = "h: H on the proximal atom; w: second H of water; d: distal atom; p: H on the distal atom"


def _formula(symbols):
    counts = Counter(symbols)
    return "".join(f"{element}{n if n > 1 else ''}" for element, n in sorted(counts.items()))


def _tree_roles(adsorbate, proximal, members, bonded, symbols):
    """Template atoms matched to the atoms of a group: {label: atom index}, or None.

    The template is a tree rooted at the proximal atom; each of its atoms is given an atom of
    its element that is bonded to the atom its parent got (backtracking over the choices).
    """
    order = list(adsorbate.atoms)

    def place(k, taken):
        if k == len(order):
            return {} if len(taken) == len(members) - 1 else None
        label, element, parent, _ = order[k]
        anchor = taken[parent] if parent else proximal
        for j in sorted(bonded[anchor]):
            if j in members and j != proximal and j not in taken.values() and symbols[j] == element:
                rest = place(k + 1, {**taken, label: j})
                if rest is not None:
                    return {label: j, **rest}
        return None

    return place(0, {})


def assign_roles(adsorbate, proximal, members, bonded, symbols):
    """The atoms of a group beyond its proximal atom, by the role labels the builder gives them.

    {label: atom index} when the group is bonded the way the adsorbate is written, else None.
    """
    extra = sorted(members - {proximal})
    on_proximal = [j for j in extra if j in bonded[proximal]]
    geometry = adsorbate.geometry
    if geometry == "atom":
        return {} if not extra else None
    if geometry in ("XH", "water"):
        wanted = 1 if geometry == "XH" else 2
        if len(extra) != wanted or on_proximal != extra or any(symbols[j] != "H" for j in extra):
            return None
        return dict(zip(("h", "w"), extra))
    if geometry in ("XY", "XYH"):
        heavy = [j for j in on_proximal if symbols[j] != "H"]
        if len(heavy) != 1 or len(on_proximal) != 1 or symbols[heavy[0]] != adsorbate.distal:
            return None
        distal = heavy[0]
        beyond = sorted(j for j in bonded[distal] & members if j != proximal)
        if geometry == "XY":
            return {"d": distal} if not beyond and len(extra) == 1 else None
        if len(beyond) != 1 or symbols[beyond[0]] != "H" or len(extra) != 2:
            return None
        return {"d": distal, "p": beyond[0]}
    if geometry == "template":
        return _tree_roles(adsorbate, proximal, members, bonded, symbols)
    return None


def find(structure, role, nbrs, molecular, exposed, registry, max_atoms=12):
    """(carried, kept): the adsorbates a slab carries and the surface groups left in the framework.

    ``role`` is the role of each atom (metal, ligand, ...), ``nbrs`` the bond graph of the
    inspection, ``molecular`` marks atoms of unbound molecules and ``exposed(i, ignore)`` says
    whether nothing but the atoms ``ignore`` sits above atom i (the atoms of a group do not hide
    the atom it binds through). A group is an atom bound to the surface metals together with every atom
    reached from it through atoms that are bound to no metal; one of more than ``max_atoms``
    atoms is the slab itself (the nitrogen of an Fe-N4 site leads into the carbon sheet) and
    is left alone.

    ``carried``: {"state", "proximal", "metals", "atoms", "roles" (label -> atom index),
    "on": "ligand" (the proximal O stays as a site) or "metal" (the group is taken out
    whole)}. ``kept``: {"atoms", "formula", "reason"}. Indices are positions in ``structure``.
    """
    symbols = structure.symbols
    n = len(structure)
    bonded = {i: {j for j, _ in nbrs[i]} for i in range(n)}
    on_metal = {i: sorted(j for j in bonded[i] if role[j] == "metal") for i in range(n)}
    for i in range(n):
        # An H that belongs to another atom (the H of an OH or OOH that leans towards a metal) does
        # not bind the surface; only a hydride does.
        if symbols[i] == "H" and any(role[j] != "metal" for j in bonded[i]):
            on_metal[i] = []
    carried, kept, seen = [], [], set()
    for i in range(n):
        if role[i] in ("metal", "spectator") or molecular[i] or i in seen or not on_metal[i]:
            continue
        members, frontier, links = {i}, [i], set()
        while frontier and len(members) <= max_atoms:
            current = frontier.pop()
            for j in bonded[current]:
                if role[j] in ("metal", "spectator") or j in members:
                    continue
                if not on_metal[j]:
                    members.add(j)
                    frontier.append(j)
                elif symbols[current] != "H":
                    links.add(j)  # another atom that binds the surface (reached through an H: a shared proton)
        if len(members) > max_atoms:
            continue  # not a group on the surface: the framework itself
        if not exposed(i, members):
            continue  # covered by the slab: not on the surface
        extra = members - {i}
        if symbols[i] == "O" and not links and len(extra) <= 2 and all(symbols[j] == "H" for j in extra):
            continue  # an oxygen ligand with zero to two H: a site in the state O, OH or H2O
        if role[i] == "ligand" and not extra and not links and len(on_metal[i]) >= 2:
            continue  # a lone anion between metals is a ligand of the slab: listed by the inspection itself
        if links:
            # PENDING(two-site-adsorbates): a group that binds through several atoms is not the state
            # of one site; it stays in the framework and is said (docs/pending.md).
            # The whole group: everything bonded together, the other atoms on the surface included.
            whole, frontier = set(members), list(links)
            while frontier and len(whole) <= max_atoms:
                current = frontier.pop()
                if current in whole:
                    continue
                whole.add(current)
                frontier += [j for j in bonded[current] if role[j] not in ("metal", "spectator") and j not in whole]
            if len(whole) > max_atoms:
                continue
            seen |= whole
            bound = [structure.ids[j] for j in sorted(whole) if on_metal[j]]
            kept.append(
                {
                    "atoms": [structure.ids[j] for j in sorted(whole)],
                    "formula": _formula(symbols[j] for j in whole),
                    "reason": f"it binds the surface through {len(bound)} atoms ({', '.join(bound)}): a pair between "
                    "two sites or a molecule bound side-on is not the state of one site. Give the slab without it "
                    "to make these sites reactive",
                }
            )
            continue
        seen |= members
        ids = [structure.ids[j] for j in sorted(members)]
        formula = _formula(symbols[j] for j in members)
        if symbols[i] != "O" and len(on_metal[i]) != 1:
            kept.append(
                {
                    "atoms": ids,
                    "formula": formula,
                    "reason": f"{symbols[i]} binds {len(on_metal[i])} metals: an adsorbate bound through an element "
                    "other than oxygen is written on top of one metal only",
                }
            )
            continue
        composition = dict(Counter(symbols[j] for j in members))
        found = []
        for adsorbate in sorted(registry.adsorbates.values(), key=lambda a: a.name):
            if adsorbate.proximal != symbols[i] or {k: v for k, v in adsorbate.composition.items() if v} != composition:
                continue
            roles = assign_roles(adsorbate, i, members, bonded, symbols)
            if roles is not None:
                found.append((adsorbate, roles))
        if len(found) != 1:
            known = sorted(a.name for a in registry.adsorbates.values() if a.proximal == symbols[i])
            why = (
                f"it can be read as {' or '.join(a.name for a, _ in found)}"
                if found
                else f"no registered adsorbate is {formula} bound through {symbols[i]}"
                + (f" (bound through {symbols[i]}: {', '.join(known)})" if known else "")
                + "; a plugin can register it"
            )
            kept.append({"atoms": ids, "formula": formula, "reason": why})
            continue
        adsorbate, roles = found[0]
        carried.append(
            {
                "state": adsorbate.name,
                "proximal": i,
                "metals": on_metal[i],
                "atoms": sorted(members),
                "roles": roles,
                "on": "ligand" if symbols[i] == "O" else "metal",
            }
        )
    # A metal has one open site: two adsorbates on one metal cannot both be written.
    per_metal = Counter(group["metals"][0] for group in carried if group["on"] == "metal")
    for group in [g for g in carried if g["on"] == "metal" and per_metal[g["metals"][0]] > 1]:
        carried.remove(group)
        kept.append(
            {
                "atoms": [structure.ids[j] for j in group["atoms"]],
                "formula": _formula(symbols[j] for j in group["atoms"]),
                "reason": f"{structure.ids[group['metals'][0]]} holds {per_metal[group['metals'][0]]} adsorbates bound "
                "through an element other than oxygen, and a metal has one open site",
            }
        )
    return carried, kept


def from_report(structure, described):
    """The carried groups of an inspection report (:func:`describe`) as indices into ``structure``."""
    index = {aid: k for k, aid in enumerate(structure.ids)}
    return [
        {
            "state": group["state"],
            "on": group["on"],
            "proximal": index[group["proximal"]],
            "metals": [index[x] for x in group["metals"]],
            "atoms": [index[x] for x in group["atoms"]],
            "roles": {label: index[x] for label, x in group["roles"].items()},
        }
        for group in described
    ]


def describe(structure, carried):
    """The carried groups for a report: IDs instead of indices, one record per group."""
    out = []
    for group in carried:
        out.append(
            {
                "state": group["state"],
                "on": group["on"],
                "proximal": structure.ids[group["proximal"]],
                "metals": [structure.ids[j] for j in group["metals"]],
                "atoms": [structure.ids[j] for j in group["atoms"]],
                "roles": {label: structure.ids[j] for label, j in sorted(group["roles"].items())},
            }
        )
    return out


def strip(structure, carried):
    """(bare, index_map): the slab without the atoms the carried groups add to their sites.

    The proximal oxygen of a group on a ligand stays; everything else of the groups goes.
    ``bare`` has the atoms in the order of ``structure`` and fresh IDs (a0001 ...), as a file
    written from it is read; ``index_map`` gives the new index of every atom that stays.
    """
    removed = set()
    for group in carried:
        removed |= set(group["atoms"]) - ({group["proximal"]} if group["on"] == "ligand" else set())
    keep = [i for i in range(len(structure)) if i not in removed]
    bare = Structure(
        [structure.symbols[i] for i in keep],
        structure.positions[keep].copy(),
        structure.cell.copy(),
        structure.mobile[keep].copy(),
        [f"a{k + 1:04d}" for k in range(len(keep))],
    )
    return bare, {old: new for new, old in enumerate(keep)}


def as_given(structure, carried, index_map, bare):
    """The surface the slab file shows, in terms of the bare slab: sites, their states, the atoms.

    Returns {"occupancy": {site ID: state}, "positions": {atom ID a campaign gives the atom:
    [x, y, z]}, "sites": [site entries the project needs for these groups], "groups": [...]}.
    A group on a ligand names the site by its oxygen; a group on a metal names the open metal
    site, with the direction in which the file has the adsorbate.
    """
    occupancy, positions, sites, groups = {}, {}, [], []
    for group in carried:
        proximal = group["proximal"]
        if group["on"] == "ligand":
            site_id = bare.ids[index_map[proximal]]
            kind = {1: "terminal", 2: "bridge"}.get(len(group["metals"]), "mu3")
            sites.append({"id": site_id, "kind": kind, "native": "O"})
        else:
            metal = group["metals"][0]
            metal_id = bare.ids[index_map[metal]]
            site_id = "m" + metal_id[1:]
            towards = minimum_image(structure.positions[proximal] - structure.positions[metal], structure.cell)
            direction = towards / np.linalg.norm(towards)
            sites.append(
                {
                    "id": site_id,
                    "kind": "open_metal",
                    "native": "empty",
                    "metal": metal_id,
                    "direction": [round(float(x), 6) for x in direction],
                }
            )
            positions[site_id] = [float(x) for x in structure.positions[proximal]]
        occupancy[site_id] = group["state"]
        for label, j in sorted(group["roles"].items()):
            positions[f"{label}_{site_id}"] = [float(x) for x in structure.positions[j]]
        groups.append(
            {
                "site": site_id,
                "state": group["state"],
                "atoms_in_the_file": [structure.ids[j] for j in group["atoms"]],
                "removed_from_the_slab": [
                    structure.ids[j] for j in group["atoms"] if not (group["on"] == "ligand" and j == proximal)
                ],
            }
        )
    return {"occupancy": occupancy, "positions": positions, "sites": sites, "groups": groups}
