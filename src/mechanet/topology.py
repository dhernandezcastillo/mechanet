"""Connectivity of relaxed structures: which bonds changed, and which intermediate a structure became.

Bonds are judged by distance with hysteresis (:data:`DEFAULT_HYSTERESIS`; elements that other
adsorbates bring use multiples of the covalent radii). :func:`topology_check` compares a
relaxed structure with its start; when bonds changed, :func:`classify_microstate` reads the
occupancy of every reactive site from the atoms and the structure is assigned to the
intermediate with that occupancy, kept for its own (another tautomer, another binding mode),
reported as released, or left for review. Used by ``collect``, ``review``, ``select`` and
``prescreen``. Part of :mod:`mechanet.analysis` up to 0.1.4, which still exports every name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from . import elements
from .adsorbates import CORE_ELEMENTS, binds_through_hydrogen, extra_elements, tree_key
from .fileio import _load, read_json
from .plugins import default_registry
from .structure import distance_matrix, minimum_image, read_poscar
from .tautomers import background_protons, normalized_owners, pair_hydrogens, pair_side, tautomer_record
from .vasp import STAGE_FINAL, STAGE_REFINE, STAGE_STATIC

DEFAULT_HYSTERESIS = {
    "O-H": (1.20, 1.35),
    "O-O": (1.75, 1.95),
    "M-O": (2.20, 2.50),
    "O-Cl": (1.95, 2.25),
    "M-Cl": (2.60, 2.95),
}
BOND_CUTOFF = {"O-H": 1.25, "O-O": 1.85, "M-O": 2.35, "O-Cl": 2.05, "M-Cl": 2.75}


# Bonds of the elements other adsorbates bring (C, N, ...): lower hysteresis edge, initial-bond
# cutoff and upper edge as multiples of the covalent radii; a metal counts with a typical radius.
GENERIC_BOND_EDGES = (1.15, 1.25, 1.35)

TYPICAL_METAL_RADIUS_A = 1.40
# Two H atoms, watched when an adsorbate binds through H: molecular hydrogen below the lower edge
# (0.74 A when free, up to about 1.0 A on a metal), two separate H beyond the upper one; the
# "elongated" pairs in between are reported as ambiguous. Lower edge, initial bond, upper edge.
HYDROGEN_PAIR_EDGES = (1.00, 1.15, 1.30)
_NATIVE_ID = re.compile(r"a\d{4}$")

_CONTEXT_CACHE = {}


def _metals(config, symbols):
    roles = elements.roles_for(symbols, (config or {}).get("slab", {}).get("roles"))
    return {e for e, r in roles.items() if r == "metal"}


def _pair_type(a, b, metals, extra=frozenset(), metal_h=False):
    """Bond kind of two elements: O-H, O-O, O-Cl, M-O, M-Cl, or, for the ``extra`` elements that
    registered adsorbates bring (C, N, ...), M-X and X-Y (sorted, e.g. C-O, C-H). ``metal_h``:
    an adsorbate binds through H (*H of hydrogen evolution), so M-H and H-H are kinds too.
    None: not watched."""
    pair = {a, b}
    if pair == {"O", "H"}:
        return "O-H"
    if pair == {"O"}:
        return "O-O"
    if pair == {"O", "Cl"}:
        return "O-Cl"
    if "O" in pair and pair & metals:
        return "M-O"
    if "Cl" in pair and pair & metals:
        return "M-Cl"
    if metal_h and "H" in pair and pair & metals:
        return "M-H"
    if metal_h and pair == {"H"}:
        return "H-H"
    if not pair & extra:
        return None
    if pair & metals:
        ligands = pair - metals
        return f"M-{next(iter(ligands))}" if ligands else None
    if not pair <= extra | CORE_ELEMENTS:
        return None
    return "-".join(sorted((a, b)))


def _limits(kind, hysteresis):
    """(lower hysteresis edge, initial-bond cutoff, upper edge) of a bond kind, in A."""
    if kind in hysteresis:
        lower, upper = hysteresis[kind]
        return lower, BOND_CUTOFF.get(kind, 0.5 * (lower + upper)), upper
    if kind == "H-H":
        return HYDROGEN_PAIR_EDGES
    total = sum(TYPICAL_METAL_RADIUS_A if e == "M" else elements.covalent_radius(e) for e in kind.split("-"))
    return tuple(scale * total for scale in GENERIC_BOND_EDGES)


def _final_geometry(state_dir, geometry):
    """The structure to judge: a stage's (or the latest) geometry of the state, or a file given as a Path."""
    if isinstance(geometry, Path):
        return geometry if geometry.is_file() else None
    candidates = {
        "latest": [(STAGE_STATIC, "POSCAR"), (STAGE_REFINE, "CONTCAR"), (STAGE_FINAL, "CONTCAR")],
        STAGE_FINAL: [(STAGE_FINAL, "CONTCAR")],
        STAGE_REFINE: [(STAGE_REFINE, "CONTCAR")],
        STAGE_STATIC: [(STAGE_STATIC, "POSCAR")],
    }[geometry]
    for stage, name in candidates:
        path = Path(state_dir) / stage / name
        if path.exists():
            return path
    return None


def _shared_protons(state_dir, ids, symbols, d_initial, shared_max, meta=None, initial=None):
    """Native low-barrier protons of the seed that are still at their seed position.

    {H index: {O index, O index}}. A transfer of such a proton between two reactive sites that
    the intermediate leaves native gives the same intermediate in another tautomer (see
    :mod:`mechanet.tautomers`); protons the builder localized next to a changed site are not
    listed (they are part of the intermediate's chemistry).
    """
    meta = meta or {}
    variant = meta.get("variant") if isinstance(meta.get("variant"), dict) else {}
    repositioned = set(variant.get("repositioned_native_ids", []))
    index = {aid: k for k, aid in enumerate(ids)}
    campaign = Path(state_dir).resolve().parent.parent
    record = _load(campaign / "surface_analysis.json")
    seed_positions = {}
    seed_path = campaign / "inputs" / "seed.vasp"
    if initial is not None and seed_path.exists():
        try:
            seed = read_poscar(seed_path)
            seed_positions = dict(zip(seed.ids, seed.positions))
        except (OSError, ValueError, IndexError):
            seed_positions = {}

    def at_seed_position(h):
        if ids[h] in repositioned:
            return False
        if initial is None or ids[h] not in seed_positions:
            return True
        return float(np.linalg.norm(minimum_image(initial.positions[h] - seed_positions[ids[h]], initial.cell))) <= 0.15

    shared = {}
    if record.get("shared_protons") is not None:
        for item in record["shared_protons"]:
            h, donor, acceptor = item.get("H"), item.get("donor"), item.get("acceptor")
            if h in index and donor in index and acceptor in index and at_seed_position(index[h]):
                shared[index[h]] = {index[donor], index[acceptor]}
        return shared
    oxygen = [i for i, s in enumerate(symbols) if s == "O" and _NATIVE_ID.match(ids[i])]
    for i, s in enumerate(symbols):
        if s == "H" and _NATIVE_ID.match(ids[i]) and at_seed_position(i):
            partners = [j for j in oxygen if d_initial[i, j] <= shared_max]
            if len(partners) >= 2:
                shared[i] = set(partners)
    return shared


def _occupancy_key(changes, coupling):
    return json.dumps(
        {"occupancy": {k: changes[k] for k in sorted(changes)}, "coupling": sorted(coupling) if coupling else None},
        sort_keys=True,
    )


def _campaign_context(campaign):
    """Nodes of a campaign's network, plus reactive sites and an occupancy lookup.

    Returns None without network.json. Without sites.json ``native`` is None and no
    automatic microstate reassignment is attempted, but explicit review assignments are
    still validated against the network.
    """
    campaign = Path(campaign)
    network_path, sites_path = campaign / "network.json", campaign / "sites.json"
    if not network_path.is_file():
        return None
    key = (
        str(campaign.resolve()),
        network_path.stat().st_mtime_ns,
        sites_path.stat().st_mtime_ns if sites_path.is_file() else None,
    )
    if key in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[key]
    network = json.loads(network_path.read_text())
    nodes = {node["id"]: node for node in network.get("nodes", [])}
    native, lookup = None, {}
    if sites_path.is_file():
        native = {site["id"]: site["native"] for site in read_json(sites_path)["sites"]}
        for node in nodes.values():
            changes = {k: v for k, v in (node.get("occupancy") or {}).items() if native.get(k) != v}
            lookup[_occupancy_key(changes, node.get("coupling"))] = node["id"]
    context = {"native": native, "nodes": nodes, "lookup": lookup}
    _CONTEXT_CACHE.clear()
    _CONTEXT_CACHE[key] = context
    return context


def _representative_node(context, node_id):
    node = context["nodes"].get(node_id) or {}
    return node.get("computed_by") or node_id


def _signatures(registry):
    return {ads.signature: ads.name for ads in registry.adsorbates.values() if ads.proximal}


def classify_microstate(
    structure,
    ids,
    distances,
    context,
    hysteresis,
    initial_distances=None,
    registry=None,
    tautomers=None,
    moved=None,
):
    """Reactive-site occupancy of a relaxed structure.

    Every H must sit on exactly one O (closest O below the lower O-H hysteresis edge and no
    second O that close); every O-O and O-Cl contact must be clearly bonded or clearly not.
    A site atom with one bonded non-site heavy atom is an OO/OOH/OCl-type group; otherwise
    its element and H count give O/OH/H2O/Cl. Two bonded site atoms form a coupled pair when
    both adsorbates allow it (``Adsorbate.couples``: two O for the OER, two *CO through C).
    Protons on framework oxygens must be unchanged relative to the initial structure.
    ``tautomers`` ({h: (seed_o, partner_o)}, see :mod:`mechanet.tautomers`) are background
    shared protons: when the H around their two oxygens are in the transferred or centred
    arrangement (counted, not by label), they are read as in the seed, and ``h`` is appended
    to ``moved``. This needs ``initial_distances``.

    Elements that registered adsorbates bring beyond O, H and Cl (C, N, ...) take part like O:
    they can hold H and bond to other heavy atoms (with bond lengths from covalent radii), and a
    site whose atoms form a larger tree is matched against the template adsorbates by the
    canonical tree rooted at the site atom (``Adsorbate.graph_key``).
    Returns (changes, coupling, reason), or (None, None, reason) outside the site model.
    """
    registry = registry or default_registry()
    signatures = _signatures(registry)
    graph_shapes = any(key[0] == "graph" for key in signatures)
    extra = extra_elements(registry.adsorbates.values())
    symbols = structure.symbols
    index = {aid: k for k, aid in enumerate(ids)}
    lo_oh, hi_oh = hysteresis["O-H"]
    heavy = [i for i, s in enumerate(symbols) if s in ("O", "Cl") or s in extra]
    oxygen = [i for i, s in enumerate(symbols) if s == "O"]
    hosts = [i for i, s in enumerate(symbols) if s == "O" or s in extra]  # atoms that can hold an H
    hydrogen = [i for i, s in enumerate(symbols) if s == "H"]

    def h_kind(i):
        return _pair_type(symbols[i], "H", set(), extra)

    def h_edges(i):
        return (lo_oh, hi_oh) if symbols[i] == "O" else _limits(h_kind(i), hysteresis)[0::2]

    handled = {}
    if tautomers and initial_distances is not None:
        for h, (seed, partner) in tautomers.items():
            start = pair_hydrogens(seed, partner, hydrogen, oxygen, initial_distances, lo_oh, hi_oh)
            end = pair_hydrogens(seed, partner, hydrogen, oxygen, distances, lo_oh, hi_oh)
            if pair_side(start, end) in ("partner", "centred"):
                for o, members in normalized_owners(seed, partner, start, end, distances).items():
                    handled.update({x: o for x in members})
                if moved is not None:
                    moved.append(h)
    hosts_text = "oxygen" if not extra else "atom of " + ", ".join(sorted({"O", *extra}))
    # The H of a site that binds through H (*H on a metal) sits on no O: topology_check judges its
    # metal contacts. On an O (or another host) it has left the site model.
    site_h = {index[s] for s in context["native"] if s in index and symbols[index[s]] == "H"}
    owner = {}
    for h in hydrogen:
        if h in handled:
            owner.setdefault(handled[h], []).append(h)
            continue
        near = sorted((float(distances[h, x]), x) for x in hosts)
        bonded = [(d, x) for d, x in near if d < h_edges(x)[0]]
        if not bonded:
            if near and near[0][0] <= h_edges(near[0][1])[1]:
                return (
                    None,
                    None,
                    (
                        f"{ids[h]} lies {near[0][0]:.2f} A from {ids[near[0][1]]}, inside the "
                        f"{h_kind(near[0][1])} hysteresis band"
                    ),
                )
            if h in site_h:
                continue
            return None, None, f"{ids[h]} is not bonded to any {hosts_text}"
        if h in site_h:
            return None, None, f"{ids[h]}, the H of a site, is bonded to {ids[bonded[0][1]]}"
        if len(bonded) > 1:
            return None, None, f"{ids[h]} is bonded to two {'oxygens' if not extra else 'atoms'}"
        owner.setdefault(bonded[0][1], []).append(h)
    # The H of a site with another H at bonding distance is molecular hydrogen, not *H.
    lo_hh, _, hi_hh = _limits("H-H", hysteresis)
    for h in sorted(site_h):
        for other in hydrogen:
            d = float(distances[h, other])
            if other == h or d > hi_hh:
                continue
            if d < lo_hh:
                return None, None, f"{ids[h]}, the H of a site, is bonded to {ids[other]}: molecular hydrogen"
            return None, None, f"H-H distance {ids[h]}-{ids[other]} = {d:.2f} A is inside the hysteresis band"
    partners = {o: [] for o in heavy}
    for a_pos, a in enumerate(heavy):
        for b in heavy[a_pos + 1 :]:
            kind = _pair_type(symbols[a], symbols[b], set(), extra)
            if kind is None or kind.startswith("M-"):
                continue
            lo, _, hi = _limits(kind, hysteresis)
            d = float(distances[a, b])
            if d < lo:
                partners[a].append(b)
                partners[b].append(a)
            elif d <= hi:
                return None, None, f"{kind} distance {ids[a]}-{ids[b]} = {d:.2f} A is inside the hysteresis band"
    sites = set(context["native"])
    site_index = {index[s]: s for s in sites if s in index}
    occupancy, coupling, claimed = {}, [], set()

    def fragment(root, skip=()):
        """(canonical tree, heavy atoms) of the new atoms bonded to a site atom, or (None, set()) when
        they reach another site or the framework, or close a ring. ``skip``: the atom of the site
        this one is coupled to, which is not part of its adsorbate."""
        atoms = set()

        def walk(atom, parent):
            children = []
            for p in partners.get(atom, []):
                if p == parent or (atom == root and p in skip):
                    continue
                if p in atoms or p == root or p in site_index or _NATIVE_ID.match(ids[p]):
                    raise LookupError
                atoms.add(p)
                children.append(walk(p, atom))
            return tree_key(symbols[atom], children + ["H"] * len(owner.get(atom, [])))

        try:
            return walk(root, None), atoms
        except LookupError:
            return None, set()

    def initial_partners(a):
        found = set()
        for b in heavy:
            kind = None if b == a else _pair_type(symbols[a], symbols[b], set(), extra)
            if kind is not None and not kind.startswith("M-"):
                if float(initial_distances[a, b]) <= _limits(kind, hysteresis)[1]:
                    found.add(b)
        return found

    for s in sorted(sites):
        if s not in index:
            occupancy[s] = "empty"
            continue
        o = index[s]
        h_count = len(owner.get(o, []))
        site_partners = [p for p in partners.get(o, []) if p in site_index]
        other = [p for p in partners.get(o, []) if p not in site_index]
        # A distal atom bonded to the site atom alone: the fixed shapes (OO, OOH, OCl, ...).
        distal, distal_h, simple = None, 0, True
        if other:
            d = other[0]
            if len(other) > 1 or _NATIVE_ID.match(ids[d]) or len(partners[d]) != 1:
                simple = False
            else:
                distal, distal_h = symbols[d], len(owner.get(d, []))
        status = signatures.get((symbols[o], distal, h_count, distal_h)) if simple else None
        if status is not None and distal:
            claimed.add(other[0])
        if status is None and graph_shapes:  # a larger tree: the template adsorbates
            key, atoms = fragment(o, site_partners)
            status = signatures.get(("graph", key)) if key else None
            if status is not None:
                claimed |= atoms
        if site_partners:
            # Bonded to the atom of another site: a coupled pair when it is one bond between two
            # adsorbates that allow it (Adsorbate.couples): two O for the OER, two *CO through C.
            if len(site_partners) > 1 or status is None or not registry.adsorbates[status].couples:
                pair = "O-O" if not extra else "-".join(sorted((symbols[o], symbols[site_partners[0]])))
                return None, None, f"{s} is part of an {pair} arrangement outside the site model"
            coupling.append(s)
        if status is None:
            if not simple:
                return None, None, f"{s} is bonded to a heavy-atom pattern outside the site model"
            return (
                None,
                None,
                (
                    f"{s} holds {symbols[o]}"
                    + (f"-{distal}" if distal else "")
                    + f" with {h_count}+{distal_h} H, which matches no registered adsorbate"
                ),
            )
        occupancy[s] = status
    if len(coupling) not in (0, 2):
        return None, None, "inconsistent site-site O-O coupling" if not extra else "inconsistent coupling between sites"
    for o in heavy:
        if o in site_index or o in claimed:
            continue
        if not _NATIVE_ID.match(ids[o]):
            return None, None, f"adsorbate atom {ids[o]} is not bonded to a reactive site (released or migrated)"
        if partners.get(o):
            if symbols[o] in ("O", "Cl") and any(symbols[p] in ("O", "Cl") for p in partners[o]):
                return None, None, f"framework atom {ids[o]} forms an O-O/O-Cl bond"
            # a framework of the other elements (the C of a carbon support) keeps its own bonds
            if initial_distances is not None and set(partners[o]) != initial_partners(o):
                return None, None, f"framework atom {ids[o]} changed its bonds"
    if initial_distances is not None:
        hydrogen = [h for h, sym in enumerate(symbols) if sym == "H"]
        for o in hosts:
            if o in site_index or o in claimed:
                continue
            cutoff = BOND_CUTOFF["O-H"] if symbols[o] == "O" else _limits(h_kind(o), hysteresis)[1]
            before = sum(1 for h in hydrogen if initial_distances[h, o] <= cutoff)
            if before != len(owner.get(o, [])):
                what = "oxygen" if symbols[o] == "O" else "atom"
                return None, None, f"the proton count on framework {what} {ids[o]} changed"
    changes = {s: v for s, v in occupancy.items() if v != context["native"][s]}
    return changes, (sorted(coupling) or None), None


FRAGMENT_NAMES = {
    (("O", 2),): "O₂",
    (("H", 1), ("O", 2)): "OOH",
    (("H", 2), ("O", 1)): "H₂O",
    (("H", 1), ("O", 1)): "OH",
    (("H", 2), ("O", 2)): "H₂O₂",
    (("O", 1),): "O",
    (("Cl", 2),): "Cl₂",
    (("Cl", 1), ("O", 1)): "ClO",
    (("Cl", 1), ("H", 1), ("O", 1)): "HOCl",
    (("Cl", 1), ("H", 1)): "HCl",
    (("H", 3), ("N", 1)): "NH₃",
}
# Bonds within a released molecule: the lower hysteresis edges where they exist.
_FRAGMENT_BONDS = {"H-H": 0.90, "Cl-H": 1.45, "Cl-Cl": 2.30}
_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def fragment_formula(symbols):
    """'O₂', 'OOH', 'H₂O', ... for a released molecule; a Hill formula otherwise."""
    counts = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    key = tuple(sorted(counts.items()))
    if key in FRAGMENT_NAMES:
        return FRAGMENT_NAMES[key]
    if "C" in counts:
        order = ["C"] + (["H"] if "H" in counts else []) + sorted(e for e in counts if e not in ("C", "H"))
    else:
        order = sorted(counts, key=lambda e: (e != "H", e))
    return "".join(e + (str(counts[e]).translate(_SUBSCRIPT) if counts[e] > 1 else "") for e in order)


def _named(atom_id, ids, symbols):
    """'Fe a0003': an atom by element and ID."""
    return f"{symbols[ids.index(atom_id)]} {atom_id}" if atom_id in ids else str(atom_id)


def _own_release(released, lost_metal, new_metal, occupancy, coupling, native, symbol_of, heavy=("O", "Cl")):
    """True when the intermediate's own adsorbate left the surface and nothing else changed at the metals.

    Every released molecule must hold a heavy atom (O or Cl, or an element another adsorbate
    brings) of a site the intermediate changes (or one it added: an H it added does not make a
    spectator water its own); a molecule of H alone (H or H2, from the *H of hydrogen evolution)
    must consist of such sites' own atoms only, so an H2 that took the H of another ligand along
    is not the intermediate's own. Every lost metal contact must belong to such a molecule, and
    no new contact formed: a spectator ligand that left, or one that moved to another metal,
    still needs review.
    """
    if not released or new_metal:
        return False
    own = {s for s, v in (occupancy or {}).items() if (native or {}).get(s) != v} | set(coupling or ())

    def owned(atoms):
        if atoms and all(symbol_of.get(a) == "H" for a in atoms):
            return all(a in own for a in atoms)
        return any(symbol_of.get(a) in heavy and (a in own or not _NATIVE_ID.match(a)) for a in atoms)

    if not all(owned(item["atoms"]) for item in released):
        return False
    gone = {a for item in released for a in item["atoms"]}
    return all(gone.intersection(pair) for pair in lost_metal)


def released_text(released, ids, symbols):
    """'O₂ (O a0065, O d_a0065) left Fe a0003' for every released molecule (its heavy atoms named)."""
    parts = []
    for item in released:
        atoms = [a for a in item["atoms"] if a in ids]
        heavy = [a for a in atoms if symbols[ids.index(a)] != "H"] or atoms  # H or H2: its own atoms
        metals = ", ".join(_named(m, ids, symbols) for m in item["from"])
        parts.append(f"{item['formula']} ({', '.join(_named(a, ids, symbols) for a in heavy)}) left {metals}")
    return "; ".join(parts)


def released_fragments(ids, symbols, distances, metals, lost_metal, hysteresis, extra=frozenset(), metal_h=False):
    """Molecules that left the metal during the relaxation: no bound minimum from this start.

    From every ligand that lost its contact with an active metal, the molecule is the set of
    O, H and Cl atoms (and atoms of the ``extra`` elements other adsorbates bring, C or N) bonded
    to it in the relaxed structure (below the lower hysteresis edges); it is released when none
    of its heavy atoms lies within the upper M-O, M-Cl (M-X) edge of any metal atom. With
    ``metal_h`` (an adsorbate binds through H) a molecule of H alone, H or H2, is judged by its
    H atoms and the M-H edge, and two H count as H2 below the H-H edge of the checks. Returns
    [{"formula", "atoms", "from"}], ``from`` being the metals it lost contact with.
    """
    if not lost_metal:
        return []
    index = {aid: k for k, aid in enumerate(ids)}
    light = [i for i, s in enumerate(symbols) if s in ("O", "H", "Cl") or s in extra]
    metal_atoms = [i for i, s in enumerate(symbols) if s in metals]
    reach = {"O": hysteresis["M-O"][1], "Cl": hysteresis.get("M-Cl", DEFAULT_HYSTERESIS["M-Cl"])[1]}
    reach.update({e: _limits(f"M-{e}", hysteresis)[2] for e in extra})
    if metal_h:
        reach["H"] = _limits("M-H", hysteresis)[2]

    def bonded(i, j):
        kind = _pair_type(symbols[i], symbols[j], set(), extra) or "-".join(sorted((symbols[i], symbols[j])))
        if kind in hysteresis:
            limit = hysteresis[kind][0]
        elif metal_h and kind == "H-H":
            limit = HYDROGEN_PAIR_EDGES[0]
        elif kind in _FRAGMENT_BONDS:
            limit = _FRAGMENT_BONDS[kind]
        else:
            limit = _limits(kind, hysteresis)[0] if set(kind.split("-")) & extra else None
        return limit is not None and float(distances[i, j]) < limit

    lost_from = {}
    for pair in lost_metal:
        a, b = (index.get(x) for x in pair[:2])
        if a is None or b is None:
            continue
        ligand, metal = (b, a) if symbols[a] in metals else (a, b)
        lost_from.setdefault(ligand, set()).add(ids[metal])
    out, seen = [], set()
    for start in sorted(lost_from):
        if start in seen:
            continue
        group, stack = {start}, [start]
        while stack:
            i = stack.pop()
            for j in light:
                if j not in group and bonded(i, j):
                    group.add(j)
                    stack.append(j)
        seen |= group
        probes = [i for i in group if symbols[i] != "H"]
        if not probes and metal_h:
            probes = sorted(group)  # H or H2: still bound when an H is within the M-H edge of a metal
        if any(
            float(distances[i, m]) <= reach.get(symbols[i], hysteresis["M-O"][1]) for i in probes for m in metal_atoms
        ):
            continue
        atoms = sorted(group, key=lambda i: (symbols[i] == "H", ids[i]))
        out.append(
            {
                "formula": fragment_formula([symbols[i] for i in atoms]),
                "atoms": [ids[i] for i in atoms],
                "from": sorted(set().union(*(lost_from.get(i, set()) for i in group))),
            }
        )
    return out


def _binding_mode_only(metal_changes, ids, symbols, distances, hysteresis):
    """True if every changed M-X contact belongs to an adsorbate (distal) O or Cl atom whose
    bonded site oxygen is attached to that same metal: an end-on/side-on change."""
    index = {aid: k for k, aid in enumerate(ids)}
    lo_oo = hysteresis["O-O"][0]
    upper_mo = hysteresis["M-O"][1]
    for a, b in metal_changes:
        o_id, m_id = (a, b) if symbols[index[a]] in ("O", "Cl") else (b, a)
        if _NATIVE_ID.match(o_id) or symbols[index[o_id]] not in ("O", "Cl"):
            return False
        o, m = index[o_id], index[m_id]
        anchors = [p for p, sym in enumerate(symbols) if sym == "O" and p != o and distances[o, p] < lo_oo]
        if not any(distances[p, m] <= upper_mo for p in anchors):
            return False
    return True


def _assign_microstate(meta, changes, coupling, reason, context, registry):
    """Map a relaxed occupancy onto the campaign's nodes."""
    original = meta.get("node_id")
    if changes is None:
        return {"status": "outside_site_model", "assigned_node_id": None, "reason": reason}
    delta = {"O": 0, "H": 0}
    for site, status in changes.items():
        new = registry.adsorbate(status).composition
        old = registry.adsorbate(context["native"][site]).composition
        for element in set(new) | set(old):
            delta[element] = delta.get(element, 0) + new.get(element, 0) - old.get(element, 0)
    delta = {k: v for k, v in delta.items() if k in ("O", "H") or v}
    record = {"occupancy_changes": changes, "coupling": coupling, "delta_atoms": delta}
    expected = meta.get("delta_atoms")
    if expected is not None and {k: int(v) for k, v in expected.items() if k in ("O", "H") or v} != delta:
        return dict(
            record,
            status="outside_site_model",
            assigned_node_id=None,
            reason="relaxed site occupancy does not account for every added or removed atom",
        )
    target = context["lookup"].get(_occupancy_key(changes, coupling))
    if target is None:
        return dict(
            record,
            status="unlisted_microstate",
            assigned_node_id=None,
            reason="relaxed occupancy matches no node of this campaign",
        )
    representative = _representative_node(context, target)
    own = {original, *meta.get("node_aliases", [])}
    own |= {_representative_node(context, n) for n in own if n}
    if target in own or representative in own:
        return dict(record, status="same_occupancy", assigned_node_id=original, matched_node_id=target)
    return dict(
        record,
        status="reassigned",
        assigned_node_id=representative,
        matched_node_id=target,
        via_symmetry_image=target if representative != target else None,
    )


def topology_check(state_dir, meta, config=None, geometry="latest", registry=None, use_review=True):
    """Distance-based connectivity of the relaxed structure versus the initial guess.

    Bonds use fixed cutoffs (O-H 1.25, O-O 1.85, O-Cl 2.05, active M-O 2.35, M-Cl 2.75 A)
    on the initial structure, with hysteresis on the final one: an initial bond is lost only
    beyond the upper edge, a new bond forms only below the lower edge of
    ``topology.hysteresis_A``; crossings inside the band are reported as soft changes. A
    relaxed structure whose site occupancy matches another node is assigned to that node.

    ``geometry`` names a stage ("latest", "02_final", ...) or is a Path to any structure with the
    state's atoms in its order (``mechanet prescreen`` passes a relaxation by an ML potential).
    ``use_review=False`` ignores the decisions of the state's review.json, which were made for
    its DFT result.
    """
    config = config or {}
    registry = registry or default_registry()
    extra = extra_elements(registry.adsorbates.values())
    metal_h = binds_through_hydrogen(registry.adsorbates.values())
    topo = config.get("topology", {})
    hysteresis = {k: tuple(v) for k, v in {**DEFAULT_HYSTERESIS, **topo.get("hysteresis_A", {})}.items()}
    shared_max = float(topo.get("shared_proton_max_OH_A", 1.6))
    final_path = _final_geometry(state_dir, geometry)
    if final_path is None:
        return {
            "status": "not_available",
            "valid_for_assigned_node": False,
            "reason": "No final geometry is available.",
            "geometry": str(geometry),
        }
    try:
        initial = read_poscar(Path(state_dir) / "POSCAR")
        final = read_poscar(final_path)
        ids = meta["atom_ids"]
        if initial.symbols != final.symbols or len(ids) != len(final):
            return {"status": "atom_mapping_mismatch", "valid_for_assigned_node": False}
        if not np.allclose(initial.cell, final.cell, atol=1e-7):
            return {
                "status": "cell_changed",
                "valid_for_assigned_node": False,
                "reason": "This campaign compares fixed-cell surface models.",
            }
        metals = _metals(config, initial.symbols)
        active = {i for i, flags in enumerate(meta["frequency_mobile"]) if any(flags)}
        d_initial, d_final = distance_matrix(initial), distance_matrix(final)
        symbols = initial.symbols
        shared = _shared_protons(state_dir, ids, symbols, d_initial, shared_max, meta, initial)
        lost, formed, soft, shifts = [], [], [], []
        lost_metal, new_metal = [], []
        ambiguous = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                kind = _pair_type(symbols[i], symbols[j], metals, extra, metal_h)
                if kind is None:
                    continue
                to_metal = kind.startswith("M-")
                if to_metal:
                    ligand_index = j if symbols[i] in metals else i
                    if ligand_index not in active:
                        continue
                lower, cutoff, upper = _limits(kind, hysteresis)
                bonded = d_initial[i, j] <= cutoff
                d = float(d_final[i, j])
                pair = tuple(sorted((ids[i], ids[j])))
                if bonded and d > upper:
                    (lost_metal if to_metal else lost).append(pair)
                elif not bonded and d < lower:
                    (new_metal if to_metal else formed).append(pair)
                elif bonded != (d <= cutoff):
                    soft.append(
                        {
                            "pair": list(pair),
                            "type": kind,
                            "initial_A": float(d_initial[i, j]),
                            "final_A": d,
                            "band_A": [lower, upper],
                        }
                    )
                if lower <= d <= upper and (i in active or j in active):
                    ambiguous.append({"pair": list(pair), "type": kind, "distance_A": d})
        lower_oh = hysteresis["O-H"][0]
        for h, partner_set in shared.items():
            start = min(partner_set, key=lambda o: d_initial[h, o])
            end = min(partner_set, key=lambda o: d_final[h, o])
            if end != start and d_final[h, end] < lower_oh:
                shifts.append(
                    {
                        "H": ids[h],
                        "from": ids[start],
                        "to": ids[end],
                        "final_distances_A": {ids[o]: float(d_final[h, o]) for o in sorted(partner_set)},
                        "note": "Transfer of a native shared proton. The state is assigned to the node with "
                        "the resulting occupancy when one exists; otherwise, when both oxygens are sites "
                        "this intermediate leaves native (background), it counts for its own node as "
                        "another tautomer; otherwise it needs review.",
                    }
                )
        required = {tuple(sorted(b[:2])) for b in meta.get("expected_bonds", []) if isinstance(b, list) and len(b) >= 2}
        index = {aid: k for k, aid in enumerate(ids)}
        missing_required = []
        for a, b in sorted(required):
            if a not in index or b not in index:
                missing_required.append((a, b))
                continue
            ia, ib = index[a], index[b]
            kind = _pair_type(symbols[ia], symbols[ib], metals, extra, metal_h)
            upper = _limits(kind, hysteresis)[2] if kind else 1.25
            if d_final[ia, ib] > upper:
                missing_required.append((a, b))
        changed = bool(missing_required or lost or formed or lost_metal or new_metal)
        review = _load(Path(state_dir) / "review.json") if use_review else {}
        override = bool(review.get("accept_relaxed_topology") and review.get("comment"))
        context = _campaign_context(Path(state_dir).resolve().parent.parent)
        native = context["native"] if context is not None else None
        index_of = {aid: k for k, aid in enumerate(ids)}
        own_occupancy, own_coupling = meta.get("occupancy"), meta.get("coupling")
        if own_occupancy is None and context is not None and meta.get("node_id") in context["nodes"]:
            node = context["nodes"][meta["node_id"]]
            own_occupancy, own_coupling = node.get("occupancy"), node.get("coupling")
        released = released_fragments(ids, symbols, d_final, metals, lost_metal, hysteresis, extra, metal_h)
        gone = {index_of[a] for item in released for a in item["atoms"] if a in index_of}
        background = background_protons(shared, ids, d_initial, native, own_occupancy, own_coupling, gone)
        tautomer = tautomer_record(shared, ids, symbols, d_initial, d_final, native, background, hysteresis, gone)
        for shift in shifts:
            shift["background"] = bool((tautomer.get(shift["H"]) or {}).get("background"))
        own_node = meta.get("node_id")
        microstate = None
        if review.get("assign_to_node") and review.get("comment"):
            target = str(review["assign_to_node"])
            problem = None
            if context is not None:
                node = context["nodes"].get(target)
                if node is None:
                    problem = f"review.json assigns unknown node {target!r}"
                elif meta.get("delta_atoms") is not None and node.get("delta_atoms") != meta.get("delta_atoms"):
                    problem = f"review.json assigns {target!r}, whose composition differs from this state"
                else:
                    target = _representative_node(context, target)
            status = "assignment_by_review_rejected" if problem else "assigned_by_explicit_review"
            microstate = {"status": status, "assigned_node_id": None if problem else target, "reason": problem}
        elif changed and not override and context is not None and context["native"] is not None:
            changes, coupling, reason = classify_microstate(
                final, ids, d_final, context, hysteresis, d_initial, registry
            )
            microstate = _assign_microstate(meta, changes, coupling, reason, context, registry)
            if background and microstate["status"] in ("unlisted_microstate", "outside_site_model"):
                # Background shared protons may sit on either of their oxygens: the same
                # intermediate in another tautomer (see mechanet.tautomers).
                moved = []
                alt_changes, alt_coupling, alt_reason = classify_microstate(
                    final, ids, d_final, context, hysteresis, d_initial, registry, tautomers=background, moved=moved
                )
                # The reading holds only if the protons stay background for the occupancy it gives.
                pair_sites = {ids[o] for h in moved for o in background[h]}
                if moved and alt_changes is not None and not pair_sites & (set(alt_changes) | set(alt_coupling or ())):
                    alternative = _assign_microstate(meta, alt_changes, alt_coupling, alt_reason, context, registry)
                    if alternative["status"] != "outside_site_model":
                        microstate = dict(
                            alternative,
                            tautomer_moved=sorted(ids[h] for h in moved),
                            exact_occupancy={
                                "status": microstate["status"],
                                "occupancy_changes": microstate.get("occupancy_changes"),
                                "reason": microstate.get("reason"),
                            },
                        )
            metal_changes = lost_metal + new_metal
            if metal_changes and _binding_mode_only(metal_changes, ids, symbols, d_final, hysteresis):
                microstate = dict(microstate, binding_mode_changed=True)
            elif metal_changes and _own_release(
                released,
                lost_metal,
                new_metal,
                own_occupancy,
                own_coupling,
                native,
                dict(zip(ids, symbols)),
                heavy=("O", "Cl", *sorted(extra)),
            ):
                microstate = dict(
                    microstate,
                    status="adsorbate_released",
                    assigned_node_id=None,
                    reason=released_text(released, ids, symbols) + ": no bound minimum from this start",
                )
            elif metal_changes:
                microstate = dict(
                    microstate,
                    status="metal_attachment_changed",
                    assigned_node_id=None,
                    reason="active metal-ligand contacts changed; assign explicitly in review.json",
                )
        if microstate is not None:
            assigned = microstate.get("assigned_node_id")
            valid = assigned is not None and assigned == own_node
            status = {
                "same_occupancy": (
                    "same_occupancy_binding_mode_changed"
                    if microstate.get("binding_mode_changed")
                    else "same_node_other_tautomer"
                    if microstate.get("tautomer_moved")
                    else "same_occupancy_proton_labels_permuted"
                ),
                "reassigned": "reassigned_to_existing_node",
                "assigned_by_explicit_review": "assigned_by_explicit_review",
            }.get(microstate["status"], "changed_connectivity_needs_review")
        else:
            valid = not changed or override
            assigned = own_node if valid else None
            status = (
                "changed_connectivity_needs_review"
                if changed and not override
                else "accepted_by_explicit_review"
                if changed
                else "connectivity_retained"
            )
        geometry_label = (
            str(final_path.relative_to(state_dir)) if Path(final_path).is_relative_to(state_dir) else str(final_path)
        )
        return {
            "status": status,
            "valid_for_assigned_node": valid,
            "assigned_node_id": assigned,
            "microstate": microstate,
            "geometry": geometry_label,
            "missing_expected_bonds": sorted(set(missing_required)),
            "lost_bonds": sorted(set(lost)),
            "new_bonds": sorted(set(formed)),
            "lost_active_metal_contacts": sorted(set(lost_metal)),
            "new_active_metal_contacts": sorted(set(new_metal)),
            "shared_proton_transfers": shifts,
            "tautomer": tautomer,
            "released": released,
            "soft_changes_within_hysteresis": soft,
            "threshold_ambiguities": ambiguous,
            "hysteresis_A": {k: list(v) for k, v in hysteresis.items()},
            "review": review or None,
            "note": (
                "Distance-based connectivity diagnostic with hysteresis. A relaxed structure whose site "
                "occupancy matches another node is assigned to that node; one that matches no node, or "
                "whose metal attachment changed, needs review (review.json: accept_relaxed_topology or "
                "assign_to_node, with a comment). No oxidation states are inferred."
            ),
        }
    except (OSError, ValueError, KeyError, IndexError) as exc:
        return {"status": "classification_failed", "valid_for_assigned_node": False, "reason": str(exc)}
