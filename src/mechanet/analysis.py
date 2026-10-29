"""Collect calculation evidence, audit relaxed structures and evaluate complete cycles.

Collection is read-only with respect to calculation folders. Reports are regenerated from
the VASP outputs and input hashes, not from a stale cache. Unknown, failed, altered and
imaginary-mode results stay in the report but are excluded from free energies.

Stage sequence (frequency.central = "refine", the default): 01_loose -> 02_final ->
03_refine -> 04_static -> 05_freq; the node energy is the 04_static E0. With
frequency.central = "final" there is no 03_refine and the node energy is the 02_final E0;
04_static seeds the 05_freq displacements at that geometry.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from . import elements
from .fileio import file_hash, read_json
from .plugins import campaign_registry, default_registry
from .reactions import SPECIES, clean, electrons as vector_electrons
from .results import FREQUENCY_CM1_PER_SQRT_EV_A2_AMU, parse_run, read_incar, same_solution
from .structure import distance_matrix, minimum_image, read_poscar
from .thermo import (
    DOWNHILL_TOLERANCE_EV,
    Conditions,
    cycle_metrics,
    formation_free_energy,
    release_last_rotation,
    reservoir_chemical_potentials,
    validate_network,
    vector_free_energy,
    vibrational_thermochemistry,
)
from .vasp import (
    STAGES,
    STAGE_FINAL,
    STAGE_REFINE,
    STAGE_STATIC,
    STAGE_FREQ,
    central_mode,
    frequency_gates,
    plan_mode,
    protocol_stages,
)

DEFAULT_HYSTERESIS = {
    "O-H": (1.20, 1.35),
    "O-O": (1.75, 1.95),
    "M-O": (2.20, 2.50),
    "O-Cl": (1.95, 2.25),
    "M-Cl": (2.60, 2.95),
}
BOND_CUTOFF = {"O-H": 1.25, "O-O": 1.85, "M-O": 2.35, "O-Cl": 2.05, "M-Cl": 2.75}
ENERGY_SOURCE_TEXT = {
    "refine": f"{STAGE_STATIC} E0 at the {STAGE_REFINE} geometry",
    "final": f"{STAGE_FINAL} E0 at the {STAGE_FINAL} geometry; 05_freq at the same geometry "
    f"with the {STAGE_FINAL} Hamiltonian (frequency.central = final)",
}
_NATIVE_ID = re.compile(r"a\d{4}$")
_CONTEXT_CACHE = {}


def _load(path, default=None):
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text())


def _write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + "\n")


def _csv(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                    if key in fields
                }
            )


def _settings(campaign, temperature_K, low_frequency_cutoff_cm1, imaginary_tolerance_cm1):
    config = _load(Path(campaign) / "config.json")
    thermo = config.get("thermodynamics", {})
    return config, {
        "temperature_K": float(temperature_K if temperature_K is not None else thermo.get("temperature_K", 298.15)),
        "low_frequency_cutoff_cm1": float(
            low_frequency_cutoff_cm1
            if low_frequency_cutoff_cm1 is not None
            else thermo.get("low_frequency_cutoff_cm1", 100.0)
        ),
        "imaginary_tolerance_cm1": float(
            imaginary_tolerance_cm1
            if imaginary_tolerance_cm1 is not None
            else thermo.get("imaginary_tolerance_cm1", 80.0)
        ),
    }


def _metals(config, symbols):
    roles = elements.roles_for(symbols, (config or {}).get("slab", {}).get("roles"))
    return {e for e, r in roles.items() if r == "metal"}


def _force_max(forces, mask):
    if forces is None:
        return None
    values = np.asarray(forces, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape or not np.isfinite(values).all():
        return None
    return float(np.max(np.linalg.norm(np.where(mask, values, 0.0), axis=1), initial=0.0))


def _geometry_match(first_path, second_path, tolerance_A=1e-5):
    try:
        first, second = read_poscar(first_path), read_poscar(second_path)
        if first.symbols != second.symbols or not np.allclose(first.cell, second.cell, atol=1e-8):
            return False
        distances = [
            float(np.linalg.norm(minimum_image(b - a, first.cell))) for a, b in zip(first.positions, second.positions)
        ]
        return max(distances, default=0.0) <= tolerance_A
    except (OSError, ValueError, IndexError):
        return False


def _pair_type(a, b, metals):
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
    return None


def _final_geometry(state_dir, geometry):
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

    Used only to annotate a transfer of such a proton between its two oxygens; a transfer
    still counts as a change of the intermediate (a different tautomer).
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


def classify_microstate(structure, ids, distances, context, hysteresis, initial_distances=None, registry=None):
    """Reactive-site occupancy of a relaxed structure.

    Every H must sit on exactly one O (closest O below the lower O-H hysteresis edge and no
    second O that close); every O-O and O-Cl contact must be clearly bonded or clearly not.
    A site atom with one bonded non-site heavy atom is an OO/OOH/OCl-type group; otherwise
    its element and H count give O/OH/H2O/Cl. Two bonded site oxygens form a coupled pair.
    Protons on framework oxygens must be unchanged relative to the initial structure.
    Returns (changes, coupling, reason), or (None, None, reason) outside the site model.
    """
    registry = registry or default_registry()
    signatures = _signatures(registry)
    symbols = structure.symbols
    index = {aid: k for k, aid in enumerate(ids)}
    lo_oh, hi_oh = hysteresis["O-H"]
    heavy = [i for i, s in enumerate(symbols) if s in ("O", "Cl")]
    oxygen = [i for i, s in enumerate(symbols) if s == "O"]
    owner = {}
    for h in (i for i, s in enumerate(symbols) if s == "H"):
        near = sorted((float(distances[h, o]), o) for o in oxygen)
        if not near or near[0][0] >= lo_oh:
            if near and near[0][0] <= hi_oh:
                return (
                    None,
                    None,
                    (f"{ids[h]} lies {near[0][0]:.2f} A from {ids[near[0][1]]}, inside the O-H hysteresis band"),
                )
            return None, None, f"{ids[h]} is not bonded to any oxygen"
        if len(near) > 1 and near[1][0] < lo_oh:
            return None, None, f"{ids[h]} is bonded to two oxygens"
        owner.setdefault(near[0][1], []).append(h)
    partners = {o: [] for o in heavy}
    for a_pos, a in enumerate(heavy):
        for b in heavy[a_pos + 1 :]:
            kind = _pair_type(symbols[a], symbols[b], set())
            if kind not in ("O-O", "O-Cl"):
                continue
            lo, hi = hysteresis[kind]
            d = float(distances[a, b])
            if d < lo:
                partners[a].append(b)
                partners[b].append(a)
            elif d <= hi:
                return None, None, f"{kind} distance {ids[a]}-{ids[b]} = {d:.2f} A is inside the hysteresis band"
    sites = set(context["native"])
    site_index = {index[s]: s for s in sites if s in index}
    occupancy, coupling, claimed = {}, [], set()
    for s in sorted(sites):
        if s not in index:
            occupancy[s] = "empty"
            continue
        o = index[s]
        h_count = len(owner.get(o, []))
        site_partners = [p for p in partners.get(o, []) if p in site_index]
        other = [p for p in partners.get(o, []) if p not in site_index]
        if site_partners:
            if len(site_partners) > 1 or other or h_count or symbols[o] != "O":
                return None, None, f"{s} is part of an O-O arrangement outside the site model"
            coupling.append(s)
            occupancy[s] = "O"
            continue
        distal, distal_h = None, 0
        if other:
            d = other[0]
            if len(other) > 1 or _NATIVE_ID.match(ids[d]) or len(partners[d]) != 1:
                return None, None, f"{s} is bonded to a heavy-atom pattern outside the site model"
            distal, distal_h = symbols[d], len(owner.get(d, []))
            claimed.add(d)
        status = signatures.get((symbols[o], distal, h_count, distal_h))
        if status is None:
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
        return None, None, "inconsistent site-site O-O coupling"
    for o in heavy:
        if o in site_index or o in claimed:
            continue
        if not _NATIVE_ID.match(ids[o]):
            return None, None, f"adsorbate atom {ids[o]} is not bonded to a reactive site (released or migrated)"
        if partners.get(o):
            return None, None, f"framework atom {ids[o]} forms an O-O/O-Cl bond"
    if initial_distances is not None:
        hydrogen = [h for h, sym in enumerate(symbols) if sym == "H"]
        for o in oxygen:
            if o in site_index or o in claimed:
                continue
            before = sum(1 for h in hydrogen if initial_distances[h, o] <= BOND_CUTOFF["O-H"])
            if before != len(owner.get(o, [])):
                return None, None, f"the proton count on framework oxygen {ids[o]} changed"
    changes = {s: v for s, v in occupancy.items() if v != context["native"][s]}
    return changes, (sorted(coupling) or None), None


def _binding_mode_only(metal_changes, ids, symbols, distances, hysteresis):
    """True if every changed M-X contact belongs to an adsorbate (distal) atom whose bonded
    site atom is attached to that same metal: an end-on/side-on change."""
    index = {aid: k for k, aid in enumerate(ids)}
    lo_oo = hysteresis["O-O"][0]
    upper_mo = hysteresis["M-O"][1]
    for a, b in metal_changes:
        o_id, m_id = (a, b) if symbols[index[a]] in ("O", "Cl") else (b, a)
        if _NATIVE_ID.match(o_id):
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


def topology_check(state_dir, meta, config=None, geometry="latest", registry=None):
    """Distance-based connectivity of the relaxed structure versus the initial guess.

    Bonds use fixed cutoffs (O-H 1.25, O-O 1.85, O-Cl 2.05, active M-O 2.35, M-Cl 2.75 A)
    on the initial structure, with hysteresis on the final one: an initial bond is lost only
    beyond the upper edge, a new bond forms only below the lower edge of
    ``topology.hysteresis_A``; crossings inside the band are reported as soft changes. A
    relaxed structure whose site occupancy matches another node is assigned to that node.
    """
    config = config or {}
    registry = registry or default_registry()
    topo = config.get("topology", {})
    hysteresis = {k: tuple(v) for k, v in {**DEFAULT_HYSTERESIS, **topo.get("hysteresis_A", {})}.items()}
    shared_max = float(topo.get("shared_proton_max_OH_A", 1.6))
    final_path = _final_geometry(state_dir, geometry)
    if final_path is None:
        return {
            "status": "not_available",
            "valid_for_assigned_node": False,
            "reason": "No final geometry is available.",
            "geometry": geometry,
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
                kind = _pair_type(symbols[i], symbols[j], metals)
                if kind is None:
                    continue
                if kind in ("M-O", "M-Cl"):
                    ligand_index = i if symbols[i] in ("O", "Cl") else j
                    if ligand_index not in active:
                        continue
                lower, upper = hysteresis[kind]
                bonded = d_initial[i, j] <= BOND_CUTOFF[kind]
                d = float(d_final[i, j])
                pair = tuple(sorted((ids[i], ids[j])))
                if bonded and d > upper:
                    (lost_metal if kind in ("M-O", "M-Cl") else lost).append(pair)
                elif not bonded and d < lower:
                    (new_metal if kind in ("M-O", "M-Cl") else formed).append(pair)
                elif bonded != (d <= BOND_CUTOFF[kind]):
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
                        "note": "Transfer of a native low-barrier proton: a different tautomer. The state is "
                        "assigned to the node with the resulting occupancy when one exists; otherwise "
                        "it needs review.",
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
            kind = _pair_type(symbols[ia], symbols[ib], metals)
            upper = hysteresis.get(kind, (0, BOND_CUTOFF.get(kind, 1.25)))[1]
            if d_final[ia, ib] > upper:
                missing_required.append((a, b))
        changed = bool(missing_required or lost or formed or lost_metal or new_metal)
        review = _load(Path(state_dir) / "review.json")
        override = bool(review.get("accept_relaxed_topology") and review.get("comment"))
        context = _campaign_context(Path(state_dir).resolve().parent.parent)
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
            metal_changes = lost_metal + new_metal
            if metal_changes and _binding_mode_only(metal_changes, ids, symbols, d_final, hysteresis):
                microstate = dict(microstate, binding_mode_changed=True)
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


def _provenance_gate(state_dir, stage, result, require_hashes):
    reasons = []
    if result.get("input_hashes_valid") is False:
        reasons.append(stage + ": input hashes mismatch")
    if require_hashes and result.get("input_hashes_valid") is None:
        reasons.append(stage + ": no verified input hashes")
    record = _load(Path(state_dir) / stage / "input_hashes.json")
    expected = record.get("state_json_sha256")
    if expected and expected != file_hash(Path(state_dir) / "state.json"):
        reasons.append(stage + ": state metadata changed after calculation")
    expected_plan = record.get("plan_sha256")
    plan_path = Path(state_dir) / "plan.json"
    if expected_plan and plan_path.exists() and expected_plan != _load(plan_path).get("sha256"):
        reasons.append(stage + ": calculation plan changed")
    attempt = _load(Path(state_dir) / stage / "attempt.json")
    if attempt.get("returncode") not in (None, 0):
        reasons.append(stage + ": VASP launcher returned nonzero")
    if attempt and attempt.get("status") != "complete":
        reasons.append(f"{stage}: the runner has not accepted this attempt (status {attempt.get('status')!r})")
    if (result.get("restart_evidence") or {}).get("fallback"):
        reasons.append(stage + ": VASP reported a fallback of the required CHGCAR/WAVECAR restart")
    for name, expected_hash in attempt.get("output_hashes", {}).items():
        if name in ("WAVECAR", "CHGCAR"):
            continue
        output_path = Path(state_dir) / stage / name
        if isinstance(expected_hash, dict):
            expected_hash = expected_hash.get("sha256")
        if not expected_hash:
            continue
        if not output_path.is_file() or file_hash(output_path) != expected_hash:
            reasons.append(stage + ": accepted output changed or disappeared: " + name)
    return reasons


def _ediffg(result):
    try:
        return float(result.get("incar", {}).get("EDIFFG", "0"))
    except (TypeError, ValueError):
        return 0.0


def _relaxation_reasons(state_dir, stage, result, meta, require_hashes, require_ionic=True):
    reasons = []
    if not result["completed"]:
        reasons.append(stage + ": incomplete")
    if not result["last_electronic_converged"]:
        reasons.append(stage + ": electronic convergence unresolved")
    reasons.extend(_provenance_gate(state_dir, stage, result, require_hashes))
    force = _force_max(result.get("forces_eVA"), meta["relax_mobile"])
    if require_ionic:
        if not result["ionic_converged"]:
            reasons.append(stage + ": ionic convergence unresolved")
        ediffg = _ediffg(result)
        if ediffg >= 0:
            reasons.append(stage + ": no explicit negative EDIFFG force criterion")
        elif force is None or force > -ediffg + 1e-5:
            reasons.append(stage + ": final movable-atom force exceeds criterion or is missing")
    return reasons, force


def _usable(topology):
    """The relaxed structure represents some node: its own, or the one it relaxed into."""
    return bool(topology.get("valid_for_assigned_node") or topology.get("assigned_node_id"))


def _final_validity(state_dir, meta, final, require_hashes, config, registry=None):
    reasons, force = _relaxation_reasons(state_dir, STAGE_FINAL, final, meta, require_hashes)
    topology = topology_check(state_dir, meta, config, geometry=STAGE_FINAL, registry=registry)
    if not _usable(topology):
        reasons.append(STAGE_FINAL + ": relaxed connectivity is unavailable or needs explicit review")
    return reasons, force, topology


def imaginary_mode_character(freq_dir, h_fraction_min=0.5, stretch_min=0.5):
    """Displacement character of every imaginary mode of a displacements-method Hessian.

    H-dominated motion along O-H bonds is proton-transfer-like (the structure is not a
    minimum along a proton coordinate); H motion across the bond is a libration/rotation.
    Returns None when no stored Hessian is available.
    """
    freq_dir = Path(freq_dir)
    report = _load(freq_dir / "frequencies.json")
    hessian, masses, units = report.get("hessian_eV_A2"), report.get("masses_amu"), report.get("units")
    if not hessian or not masses or not units or not (freq_dir / "POSCAR").is_file():
        return None
    central = read_poscar(freq_dir / "POSCAR")
    active = [(int(u["atom_index_1based"]) - 1, "xyz".index(u["axis"])) for u in units[::2]]
    m = np.array([float(masses[central.symbols[a]]) for a, _ in active])
    dynamical = np.asarray(hessian, dtype=float) / np.sqrt(np.outer(m, m))
    eigenvalues, vectors = np.linalg.eigh(0.5 * (dynamical + dynamical.T))
    distances = distance_matrix(central)
    oxygen = [i for i, sym in enumerate(central.symbols) if sym == "O"]
    modes = []
    for k, value in enumerate(eigenvalues):
        if value >= 0:
            break
        pattern = vectors[:, k] / np.sqrt(m)
        disp = {}
        for (atom, axis), amplitude in zip(active, pattern):
            disp.setdefault(atom, np.zeros(3))[axis] = amplitude
        total = sum(float(u @ u) for u in disp.values()) or 1.0
        weights = {atom: float(u @ u) / total for atom, u in disp.items()}
        h_fraction = sum(w for atom, w in weights.items() if central.symbols[atom] == "H")
        along = across = 0.0
        for atom, u in disp.items():
            if central.symbols[atom] != "H" or not oxygen:
                continue
            o = min(oxygen, key=lambda j: distances[atom, j])
            if distances[atom, o] > BOND_CUTOFF["O-H"]:
                continue
            bond = minimum_image(central.positions[atom] - central.positions[o], central.cell)
            unit_bond = bond / np.linalg.norm(bond)
            relative = u - disp.get(o, np.zeros(3))
            along += float(relative @ unit_bond) ** 2
            across += float(relative @ relative)
        stretch = along / across if across > 0 else 0.0
        character = (
            "proton_transfer_like"
            if h_fraction >= h_fraction_min and stretch >= stretch_min
            else "H_libration_or_rotation"
            if h_fraction >= h_fraction_min
            else "heavy_atom_dominated"
        )
        top = sorted(weights.items(), key=lambda x: -x[1])[:3]
        modes.append(
            {
                "cm1": -FREQUENCY_CM1_PER_SQRT_EV_A2_AMU * float(np.sqrt(-value)),
                "character": character,
                "H_fraction": round(h_fraction, 3),
                "OH_stretch_fraction": round(stretch, 3),
                "dominant_atoms": [
                    {"poscar_index_1based": atom + 1, "element": central.symbols[atom], "weight": round(w, 3)}
                    for atom, w in top
                ],
            }
        )
    return modes


def _smearing_warnings(stages, limit_eV=0.02):
    """Flag stages whose smearing entropy term is large enough to matter for reaction energies."""
    warnings = []
    for stage in (STAGE_FINAL, STAGE_REFINE, STAGE_STATIC):
        value = (stages.get(stage) or {}).get("eentro_eV")
        if value is None:
            continue
        magnitude = abs(float(value))
        if magnitude > limit_eV:
            warnings.append(
                f"{stage}: smearing entropy |EENTRO| = {magnitude:.3f} eV per cell exceeds {limit_eV:g} eV; "
                "partial occupancies near the Fermi level make E(sigma->0) depend on SIGMA "
                "(VASP default 0.2 eV when SIGMA is not set); check a smaller SIGMA for this state."
            )
    return warnings


def collect(
    campaign,
    out_dir=None,
    temperature_K=None,
    low_frequency_cutoff_cm1=None,
    imaginary_tolerance_cm1=None,
    registry=None,
):
    """Write state, moment, frequency and QC tables; no DFT is run."""
    campaign = Path(campaign).resolve()
    out_dir = Path(out_dir) if out_dir else campaign / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    config, settings = _settings(campaign, temperature_K, low_frequency_cutoff_cm1, imaginary_tolerance_cm1)
    registry = registry or campaign_registry(campaign, config=config)
    require_hashes = bool(config.get("analysis", {}).get("require_input_hashes", True))
    eentro_limit = float(config.get("analysis", {}).get("max_eentro_eV", 0.02))
    gates = frequency_gates(config)
    max_frequency_force, warn_frequency_force = gates["max_force_eVA"], gates["warn_force_eVA"]
    central = central_mode(config)
    state_rows, moments, frequency_rows = [], [], []
    protocol_notes = set()
    for state_dir in sorted((campaign / "states").glob("*")):
        if not (state_dir / "state.json").is_file():
            continue
        meta = _load(state_dir / "state.json")
        stages = {stage: parse_run(state_dir / stage) for stage in protocol_stages(central)}
        final, static, freq = (stages[s] for s in (STAGE_FINAL, STAGE_STATIC, STAGE_FREQ))
        refine = stages.get(STAGE_REFINE)
        warnings = []
        written = read_incar(state_dir / STAGE_FINAL / "INCAR")
        if written and "SIGMA" not in written and str(written.get("ISMEAR", "1")).strip() in ("0", "-1"):
            protocol_notes.add(
                "SIGMA is not set in the stage INCARs: VASP uses its default SIGMA = 0.2 eV with ISMEAR = "
                + str(written.get("ISMEAR"))
                + ". Keep it for consistency with earlier data, but check EENTRO."
            )
        prepared = _load(state_dir / "plan.json")
        state_mode = plan_mode(prepared) if prepared else central
        mixed = (
            (
                f"plan.json: prepared with frequency.central = {state_mode!r}, but config.json says {central!r}; "
                "its later stages follow the other protocol"
            )
            if state_mode != central
            else None
        )
        final_reasons, final_force, final_topology = _final_validity(
            state_dir, meta, final, require_hashes, config, registry
        )
        final_valid = not final_reasons
        reasons = list(final_reasons)
        frequency_reasons = []
        refine_force = None
        if central == "refine":
            refine_reasons, refine_force = _relaxation_reasons(state_dir, STAGE_REFINE, refine, meta, require_hashes)
            reasons.extend(refine_reasons)
            if mixed:
                reasons.append(mixed)
            static_reasons, central_contcar, central_name = reasons, state_dir / STAGE_REFINE / "CONTCAR", "refined"
        else:
            if final["energy_e0_eV"] is None:
                reasons.append(STAGE_FINAL + ": electronic energy is missing")
            if mixed:
                frequency_reasons.append(mixed)
            static_reasons, central_contcar, central_name = (
                frequency_reasons,
                state_dir / STAGE_FINAL / "CONTCAR",
                STAGE_FINAL,
            )
        if not static["completed"]:
            static_reasons.append(STAGE_STATIC + ": incomplete")
        if not static["electronic_converged"]:
            static_reasons.append(STAGE_STATIC + ": electronic convergence unresolved")
        static_reasons.extend(_provenance_gate(state_dir, STAGE_STATIC, static, require_hashes))
        if not _geometry_match(central_contcar, state_dir / STAGE_STATIC / "POSCAR"):
            static_reasons.append(f"{STAGE_STATIC}: geometry does not match the {central_name} structure")
        topology = topology_check(state_dir, meta, config, registry=registry)
        if not _usable(topology):
            reasons.append("Relaxed connectivity is unavailable or needs explicit review")
        elif topology["assigned_node_id"] != meta["node_id"]:
            warnings.append(
                f"Relaxed into the occupancy of node {topology['assigned_node_id']}"
                + (
                    f" (symmetry image {topology['microstate'].get('via_symmetry_image')})"
                    if (topology.get("microstate") or {}).get("via_symmetry_image")
                    else ""
                )
                + f"; its energy is used for that node, not for {meta['node_id']}."
            )
        if static["energy_e0_eV"] is None:
            static_reasons.append(STAGE_STATIC + ": central electronic energy is missing")
        static_shift = None
        if central == "final":
            same = same_solution(final, static)
            static_shift, moment_change = (
                same["static_minus_final_e0_eV"],
                same["max_moment_change_final_to_static_muB"],
            )
            if static_shift is not None and abs(static_shift) > gates["max_static_e0_shift_eV"]:
                frequency_reasons.append(
                    f"{STAGE_STATIC}: E0 differs from the {STAGE_FINAL} E0 by {static_shift * 1000:+.1f} meV at "
                    "the same geometry and Hamiltonian "
                    f"(frequency.max_static_e0_shift_eV = {gates['max_static_e0_shift_eV']:g}); "
                    f"04_static reached another electronic solution, so the frequencies do not belong to the "
                    f"{STAGE_FINAL} energy"
                )
            if moment_change is not None and moment_change > gates["max_moment_change_muB"]:
                frequency_reasons.append(
                    f"{STAGE_STATIC}: a local moment changed by {moment_change:.2f} muB from {STAGE_FINAL} "
                    f"(frequency.max_moment_change_muB = {gates['max_moment_change_muB']:g}); 04_static reached "
                    f"another electronic solution, so the frequencies do not belong to the {STAGE_FINAL} energy"
                )
        electronic_valid = not reasons
        if not freq["frequency_converged"]:
            frequency_reasons.append(STAGE_FREQ + ": incomplete spectrum or unconverged displacement calculations")
        frequency_reasons.extend(_provenance_gate(state_dir, STAGE_FREQ, freq, require_hashes))
        if not _geometry_match(state_dir / STAGE_STATIC / "POSCAR", state_dir / STAGE_FREQ / "POSCAR"):
            frequency_reasons.append(
                STAGE_FREQ + ": displaced Hessian reference does not match the central static geometry"
            )
        try:
            frequency_atoms = read_poscar(state_dir / STAGE_FREQ / "POSCAR")
            if not np.array_equal(frequency_atoms.mobile, np.asarray(meta["frequency_mobile"], dtype=bool)):
                frequency_reasons.append(
                    STAGE_FREQ + ": active-coordinate mask differs from the recorded atom-ID-based region"
                )
        except (OSError, ValueError, IndexError):
            frequency_reasons.append(STAGE_FREQ + ": cannot verify the active-coordinate mask")
        active_force = _force_max(static.get("forces_eVA"), meta["frequency_mobile"])
        if active_force is None or active_force > max_frequency_force + 1e-5:
            frequency_reasons.append(
                STAGE_STATIC + ": force on a frequency-active atom exceeds the configured threshold or is missing"
            )
        elif active_force > warn_frequency_force:
            warnings.append(
                f"{STAGE_STATIC}: frequency-active force {active_force:.3f} eV/A is within the warning band "
                f"({warn_frequency_force:g}-{max_frequency_force:g} eV/A); soft modes may be less reliable."
            )
        expected_modes = sum(sum(row) for row in meta["frequency_mobile"])
        vibration = vibrational_thermochemistry(freq["frequencies_cm1"], expected_modes=expected_modes, **settings)
        frequency_reasons.extend(vibration["errors"])
        warnings.extend(vibration["warnings"])
        mode_character = None
        if vibration["imaginary_modes"]:
            try:
                mode_character = imaginary_mode_character(state_dir / STAGE_FREQ)
            except (OSError, ValueError, KeyError, IndexError) as exc:
                warnings.append(f"{STAGE_FREQ}: imaginary-mode character unavailable ({exc}).")
            if mode_character is None:
                warnings.append(
                    f"{STAGE_FREQ}: no stored Hessian, so imaginary modes were judged by magnitude only; "
                    "inspect their eigenvectors in OUTCAR."
                )
            else:
                for mode in mode_character:
                    if (
                        mode["character"] == "proton_transfer_like"
                        and abs(mode["cm1"]) <= settings["imaginary_tolerance_cm1"]
                    ):
                        frequency_reasons.append(
                            f"{STAGE_FREQ}: imaginary mode {mode['cm1']:.0f} cm^-1 moves H along its O-H bond "
                            "(proton-transfer-like); the structure is not a minimum along that coordinate, "
                            "so the mode is not regularized"
                        )
                    elif mode["character"] == "heavy_atom_dominated":
                        warnings.append(
                            f"{STAGE_FREQ}: imaginary mode {mode['cm1']:.0f} cm^-1 is dominated by heavy-atom "
                            "motion; check that it is numerical before accepting the floor."
                        )
        magnetic_continuity = {"status": "not_available", "max_change_muB": None}
        history = freq.get("magnetic_moment_history", [])
        reference_moments = static.get("magmoms")
        if history and reference_moments is not None:
            differences = np.asarray(history) - np.asarray(reference_moments)[None, :]
            maximum = float(np.max(np.abs(differences)))
            threshold_muB = gates["max_moment_change_muB"]
            magnetic_continuity = {
                "status": "within_diagnostic_threshold" if maximum <= threshold_muB else "needs_review",
                "max_change_muB": maximum,
                "threshold_muB": threshold_muB,
                "n_moment_snapshots": len(history),
                "note": "Projected-moment continuity diagnostic; it cannot prove identical electronic "
                "occupation patterns.",
            }
            if maximum > threshold_muB:
                frequency_reasons.append(
                    STAGE_FREQ + ": projected local moments changed beyond the configured continuity "
                    "threshold; inspect electronic branch consistency"
                )
        warnings.extend(_smearing_warnings(stages, eentro_limit))
        free_energy_valid = electronic_valid and not frequency_reasons and vibration["valid"]
        has_outputs = any(
            (state_dir / stage / "OUTCAR").exists() or (state_dir / stage / "vasprun.xml").exists() for stage in STAGES
        )
        status = (
            "free_energy_ready"
            if free_energy_valid
            else "electronic_energy_ready"
            if electronic_valid
            else "final_energy_ready"
            if final_valid
            else "needs_review"
            if has_outputs
            else "not_run"
        )
        energy = final["energy_e0_eV"] if central == "final" else static["energy_e0_eV"]
        row = {
            "state_id": meta["id"],
            "node_id": meta["node_id"],
            "assigned_node_id": topology.get("assigned_node_id"),
            "final_assigned_node_id": final_topology.get("assigned_node_id"),
            "microstate_status": topology["status"],
            "node_aliases": meta.get("node_aliases", [meta["node_id"]]),
            "family": meta.get("family"),
            "site": meta.get("site"),
            "variant": meta.get("variant"),
            "status": status,
            "n_atoms": len(meta["atom_ids"]),
            "n_frequency_dof": expected_modes,
            "delta_atoms": meta.get("delta_atoms"),
            "final_valid": final_valid,
            "electronic_valid": electronic_valid,
            "free_energy_valid": free_energy_valid,
            "energy_final_e0_eV": final["energy_e0_eV"],
            "energy_e0_eV": energy,
            "energy_static_e0_eV": static["energy_e0_eV"],
            "static_minus_final_e0_eV": static_shift,
            "F_vib_eV": vibration["F_vib_eV"] if not frequency_reasons else None,
            "G_eV": energy + vibration["F_vib_eV"] if free_energy_valid else None,
            "max_final_mobile_force_eVA": final_force,
            "max_refine_mobile_force_eVA": refine_force,
            "max_static_frequency_force_eVA": active_force,
            "final_exclusion_reasons": list(dict.fromkeys(final_reasons)),
            "electronic_exclusion_reasons": list(dict.fromkeys(reasons)),
            "frequency_exclusion_reasons": list(dict.fromkeys(frequency_reasons)),
            "warnings": list(dict.fromkeys(warnings)),
            "topology": topology,
            "final_topology": final_topology,
            "vibrational": vibration,
            "imaginary_mode_character": mode_character,
            "frequency_magnetic_continuity": magnetic_continuity,
            "stages": stages,
        }
        for stage in (STAGE_FINAL, STAGE_REFINE, STAGE_STATIC):
            values = (stages.get(stage) or {}).get("magmoms")
            if values is None or len(values) != len(meta["atom_ids"]):
                continue
            for index, (atom_id, symbol, value) in enumerate(zip(meta["atom_ids"], meta["symbols"], values)):
                moments.append(
                    {
                        "state_id": meta["id"],
                        "stage": stage,
                        "poscar_index_1based": index + 1,
                        "atom_id": atom_id,
                        "element": symbol,
                        "local_moment_muB": value,
                    }
                )
        if final.get("magmoms") is not None and static.get("magmoms") is not None:
            delta = np.asarray(static["magmoms"]) - np.asarray(final["magmoms"])
            row["max_local_moment_change_final_to_static_muB"] = float(np.max(np.abs(delta), initial=0.0))
            if central == "refine" and row["max_local_moment_change_final_to_static_muB"] > 0.5:
                row["warnings"].append(
                    "A local moment changed by >0.5 muB between final and static stages; inspect "
                    "electronic-state continuity."
                )
        for i, value in enumerate(freq["frequencies_cm1"]):
            frequency_rows.append(
                {
                    "state_id": meta["id"],
                    "mode": i + 1,
                    "frequency_cm1": value,
                    "imaginary": value < 0,
                    "floor_cm1": settings["low_frequency_cutoff_cm1"],
                }
            )
        state_rows.append(row)
    report = {
        "schema_version": 3,
        "campaign": str(campaign),
        "frequency_central": central,
        "settings": settings,
        "energy_source": ENERGY_SOURCE_TEXT[central],
        "vibrational_model": "same inherited active region plus new reactive atoms; partial Hessian",
        "protocol_notes": sorted(protocol_notes),
        "states": state_rows,
    }
    _write(out_dir / "states.json", report)
    _csv(
        out_dir / "states.csv",
        state_rows,
        [
            "state_id",
            "node_id",
            "assigned_node_id",
            "microstate_status",
            "family",
            "site",
            "variant",
            "status",
            "n_atoms",
            "n_frequency_dof",
            "energy_final_e0_eV",
            "energy_e0_eV",
            "energy_static_e0_eV",
            "static_minus_final_e0_eV",
            "F_vib_eV",
            "G_eV",
            "final_valid",
            "electronic_valid",
            "free_energy_valid",
            "max_final_mobile_force_eVA",
            "max_refine_mobile_force_eVA",
            "max_static_frequency_force_eVA",
            "final_exclusion_reasons",
            "electronic_exclusion_reasons",
            "frequency_exclusion_reasons",
            "warnings",
        ],
    )
    _csv(
        out_dir / "magnetic_moments.csv",
        moments,
        ["state_id", "stage", "poscar_index_1based", "atom_id", "element", "local_moment_muB"],
    )
    _csv(out_dir / "frequencies.csv", frequency_rows, ["state_id", "mode", "frequency_cm1", "imaginary", "floor_cm1"])
    qc = {
        "n_states": len(state_rows),
        "n_not_run": sum(r["status"] == "not_run" for r in state_rows),
        "n_reassigned_to_other_node": sum(r["assigned_node_id"] not in (None, r["node_id"]) for r in state_rows),
        "reassigned_states": {
            r["state_id"]: r["assigned_node_id"]
            for r in state_rows
            if r["assigned_node_id"] not in (None, r["node_id"])
        },
        "n_final_valid": sum(r["final_valid"] for r in state_rows),
        "n_electronic_valid": sum(r["electronic_valid"] for r in state_rows),
        "n_free_energy_valid": sum(r["free_energy_valid"] for r in state_rows),
        "settings": settings,
        "protocol_notes": sorted(protocol_notes),
        "frequency_central": central,
        "energy_source": ENERGY_SOURCE_TEXT[central],
        "limitations": [
            "Partial Hessians do not prove stability in frozen degrees of freedom.",
            "Electronic continuation does not prove a global magnetic ground state.",
            "Connectivity diagnostics require review for transformed products.",
        ],
    }
    _write(out_dir / "qc.json", qc)
    return {"out_dir": str(out_dir), "summary": qc, "states": state_rows}


def _cycle_nodes(network, cycle_ids):
    """Computed nodes of the given cycles plus the reference surface (symmetry images mapped)."""
    cycles = {c["id"]: c for c in network.get("cycles", [])}
    unknown = [c for c in cycle_ids if c not in cycles]
    if unknown:
        raise ValueError(f"Unknown cycle IDs: {unknown}")
    edges = {e["id"]: e for e in network.get("edges", [])}
    nodes = {n["id"]: n for n in network.get("nodes", [])}
    wanted = {network.get("reference", "ref")}
    for cycle_id in cycle_ids:
        for edge_id in cycles[cycle_id]["edge_ids"]:
            for node_id in (edges[edge_id]["from"], edges[edge_id]["to"]):
                wanted.add((nodes.get(node_id) or {}).get("computed_by") or node_id)
    return wanted


def select_for_thermo(
    campaign,
    energy_window_eV=None,
    max_variants_per_node=None,
    include_partial=False,
    out_path=None,
    cycles=None,
    registry=None,
):
    """Choose which 02_final variants proceed to the later stages.

    For every computed node (symmetry twins excluded), the accepted 02_final variants within
    ``energy_window_eV`` of the node minimum are kept, up to ``max_variants_per_node``. A
    node with variants whose 02_final has not completed is left waiting unless
    ``include_partial`` is set. ``cycles`` restricts the selection to the nodes of those
    cycles (every structure is still evaluated, so one that relaxed into such a node
    competes there).
    """
    campaign = Path(campaign).resolve()
    config = _load(campaign / "config.json")
    registry = registry or campaign_registry(campaign, config=config)
    selection_cfg = config.get("thermo_selection", {})
    window = float(energy_window_eV if energy_window_eV is not None else selection_cfg.get("energy_window_eV", 0.10))
    limit = int(
        max_variants_per_node if max_variants_per_node is not None else selection_cfg.get("max_variants_per_node", 2)
    )
    if window < 0 or limit < 1:
        raise ValueError("energy window must be >= 0 and max variants per node >= 1")
    require_hashes = bool(config.get("analysis", {}).get("require_input_hashes", True))
    network = _load(campaign / "network.json")
    computed = [n for n in network.get("nodes", []) if not n.get("computed_by")]
    wanted = _cycle_nodes(network, list(cycles)) if cycles else None
    selected, per_node, waiting, unresolved = [], [], [], []
    evaluated, own_variants = {}, {}
    for node in computed:
        own_variants[node["id"]] = []
        for sid in node.get("state_ids", []):
            if sid not in evaluated:
                state_dir = campaign / "states" / sid
                meta = _load(state_dir / "state.json")
                final = parse_run(state_dir / STAGE_FINAL)
                runner_status = _load(state_dir / STAGE_FINAL / "attempt.json").get("status")
                if not final["completed"]:
                    if runner_status == "failed":
                        record = {
                            "state_id": sid,
                            "status": "rejected",
                            "energy_final_e0_eV": None,
                            "runner_status": runner_status,
                            "reasons": [f"{STAGE_FINAL}: failed attempt (use --retry to rerun it)"],
                        }
                    else:
                        record = {
                            "state_id": sid,
                            "status": "pending",
                            "energy_final_e0_eV": None,
                            "runner_status": runner_status
                            or ("not_started" if not (state_dir / STAGE_FINAL / "OUTCAR").exists() else None),
                        }
                else:
                    reasons, force, topology = _final_validity(state_dir, meta, final, require_hashes, config, registry)
                    record = {
                        "state_id": sid,
                        "status": "accepted" if not reasons else "rejected",
                        "energy_final_e0_eV": final["energy_e0_eV"],
                        "reasons": reasons,
                        "assigned_node_id": topology.get("assigned_node_id"),
                        "built_for_node_id": meta.get("node_id"),
                    }
                    if not reasons and (
                        (state_dir / STAGE_REFINE / "CONTCAR").is_file()
                        or (state_dir / STAGE_STATIC / "POSCAR").is_file()
                    ):
                        latest = topology_check(state_dir, meta, config, registry=registry)
                        record["latest_assigned_node_id"] = latest.get("assigned_node_id") if _usable(latest) else None
                evaluated[sid] = record
            own_variants[node["id"]].append(evaluated[sid])
    for node in computed:
        if wanted is not None and node["id"] not in wanted:
            continue
        pending = any(v["status"] == "pending" for v in own_variants[node["id"]])
        variants = [v for v in own_variants[node["id"]] if v.get("assigned_node_id") in (None, node["id"])]
        moved_out = {
            v["state_id"]: v["assigned_node_id"]
            for v in own_variants[node["id"]]
            if v.get("assigned_node_id") not in (None, node["id"])
        }
        moved_in = [
            v
            for v in evaluated.values()
            if v.get("assigned_node_id") == node["id"] and v not in own_variants[node["id"]]
        ]
        variants = variants + moved_in
        accepted = sorted(
            (
                v
                for v in variants
                if v["status"] == "accepted"
                and v["energy_final_e0_eV"] is not None
                and v.get("assigned_node_id") == node["id"]
                and v.get("latest_assigned_node_id", node["id"]) == node["id"]
            ),
            key=lambda v: v["energy_final_e0_eV"],
        )
        moved_later = {
            v["state_id"]: v["latest_assigned_node_id"]
            for v in variants
            if v.get("assigned_node_id") == node["id"]
            and "latest_assigned_node_id" in v
            and v["latest_assigned_node_id"] != node["id"]
        }
        extra = {"reassigned_out": moved_out} if moved_out else {}
        if moved_later:
            extra["left_node_after_refinement"] = moved_later
        if pending and not include_partial:
            blocking = [v["state_id"] for v in variants if v["status"] == "pending"]
            waiting.append({"node_id": node["id"], "blocking_states": blocking})
            per_node.append(
                {
                    "node_id": node["id"],
                    "status": "waiting_for_02_final",
                    "blocking_states": blocking,
                    "variants": variants,
                    **extra,
                }
            )
            continue
        if not accepted:
            unresolved.append(node["id"])
            per_node.append({"node_id": node["id"], "status": "no_accepted_variant", "variants": variants, **extra})
            continue
        minimum = accepted[0]["energy_final_e0_eV"]
        chosen = [v for v in accepted if v["energy_final_e0_eV"] - minimum <= window + 1e-12][:limit]
        for v in accepted:
            v["relative_eV"] = v["energy_final_e0_eV"] - minimum
        selected.extend(v["state_id"] for v in chosen)
        per_node.append(
            {
                "node_id": node["id"],
                "status": "selected",
                "selected": [v["state_id"] for v in chosen],
                "variants": variants,
                **extra,
            }
        )
    result = {
        "schema_version": 1,
        "state_ids": sorted(set(selected)),
        "criteria": {
            "energy_window_eV": window,
            "max_variants_per_node": limit,
            "energy": f"{STAGE_FINAL} E0 (same composition within a node)",
            "grouping": "by the node whose site occupancy the relaxed 02_final structure has",
            "include_partial": bool(include_partial),
            "cycles": sorted(cycles) if cycles else None,
        },
        "n_selected": len(set(selected)),
        "waiting_nodes": waiting,
        "unresolved_nodes": unresolved,
        "nodes": per_node,
        "note": (
            "Re-run after more 02_final results arrive (or use --include-partial to select from the variants "
            "finished so far); selection never removes states that already ran later stages."
        ),
    }
    previous = _load(campaign / "thermo_selection.json")
    for sid in previous.get("state_ids", []):
        if sid not in result["state_ids"] and any(
            (campaign / "states" / sid / s / "OUTCAR").exists() for s in (STAGE_REFINE, STAGE_STATIC, STAGE_FREQ)
        ):
            result["state_ids"].append(sid)
    result["state_ids"] = sorted(set(result["state_ids"]))
    _write(Path(out_path) if out_path else campaign / "thermo_selection.json", result)
    return result


# ---------------------------------------------------------------------------
# Thermodynamics of nodes, edges and cycles
# ---------------------------------------------------------------------------


def _electronic_correction(delta_atoms, corrections):
    if not corrections:
        return 0.0
    return sum(
        float(corrections.get(f"per_{element}_eV", 0.0)) * float(count)
        for element, count in (delta_atoms or {}).items()
    )


def edge_vector(edge):
    if edge.get("reservoirs") is not None:
        return clean(edge["reservoirs"])
    return clean({"H2O": -edge["water"], "H+": edge["pairs"], "O2": edge["oxygen"]})


def node_energies(campaign, network, collected, include_vibrations=True, energy_source="static", corrections=None):
    """Lowest accepted energy (or free energy) per computed node; symmetry images inherit it."""
    state_map = {s["state_id"]: s for s in collected["states"]}
    if include_vibrations:

        def value_of(s):
            return s["G_eV"] if s["free_energy_valid"] else None
    elif energy_source == "static":

        def value_of(s):
            return (
                s["energy_e0_eV"] + _electronic_correction(s["delta_atoms"], corrections)
                if s["electronic_valid"] and s["energy_e0_eV"] is not None
                else None
            )
    else:

        def value_of(s):
            return (
                s["energy_final_e0_eV"] + _electronic_correction(s["delta_atoms"], corrections)
                if s["final_valid"] and s["energy_final_e0_eV"] is not None
                else None
            )

    assignment_key = (
        "final_assigned_node_id" if energy_source == "final" and not include_vibrations else "assigned_node_id"
    )

    def valid_nodes(row):
        assigned = row.get(assignment_key)
        if assigned is None:
            return set()
        if assigned == row["node_id"]:
            return {assigned, *row.get("node_aliases", [])}
        return {assigned}

    listed = {sid for n in network.get("nodes", []) for sid in n.get("state_ids", [])}
    node_rows, selected, chosen_state = [], {}, {}
    for node in network.get("nodes", []):
        if node.get("computed_by"):
            continue
        own = [state_map[s] for s in node.get("state_ids", []) if s in state_map]
        incoming = [
            s
            for s in collected["states"]
            if s["state_id"] in listed
            and s["state_id"] not in node.get("state_ids", [])
            and s.get(assignment_key) == node["id"]
            and s.get(assignment_key) != s["node_id"]
        ]
        usable = [s for s in own + incoming if node["id"] in valid_nodes(s)]
        accepted = [(value_of(s), s["state_id"]) for s in usable if value_of(s) is not None]
        if accepted:
            value, sid = min(accepted)
            selected[node["id"]], chosen_state[node["id"]] = value, sid
        node_rows.append(
            {
                "node_id": node["id"],
                "family": node.get("family"),
                "site": node.get("site"),
                "computed_by": None,
                "n_variants": len(own),
                "n_accepted_variants": len(accepted),
                "reassigned_in": [s["state_id"] for s in incoming],
                "reassigned_out": [s["state_id"] for s in own if s.get(assignment_key) not in (None, node["id"])],
                "selected_state_id": chosen_state.get(node["id"]),
                "node_energy_eV": selected.get(node["id"]),
                "status": "available" if node["id"] in selected else "unresolved",
            }
        )
    for node in network.get("nodes", []):
        rep = node.get("computed_by")
        if not rep:
            continue
        if rep in selected:
            selected[node["id"]], chosen_state[node["id"]] = selected[rep], chosen_state[rep]
        node_rows.append(
            {
                "node_id": node["id"],
                "family": node.get("family"),
                "site": node.get("site"),
                "computed_by": rep,
                "n_variants": 0,
                "n_accepted_variants": 0,
                "reassigned_in": [],
                "reassigned_out": [],
                "selected_state_id": chosen_state.get(rep),
                "node_energy_eV": selected.get(node["id"]),
                "status": "available_by_symmetry" if node["id"] in selected else "unresolved",
            }
        )
    return selected, node_rows, assignment_key


RESERVOIR_MOLECULES = {"H2O": {"H2O"}, "H+": {"H2"}, "O2": {"H2", "H2O"}, "Cl-": {"Cl2"}, "Cl2": {"Cl2"}}


def is_chemical(vector):
    """No electron and no charged species: the step's free energy depends on neither U nor pH."""
    return vector_electrons(vector) == 0 and all(SPECIES[name].charge == 0 for name in vector)


def release_free_energies(steps, selected, reservoirs, conditions, scale):
    """DeltaG at 0 V (on ``scale``), electrons and kind of each product-release step whose two
    ends have energies; ``steps`` from :func:`mechanet.pathways.product_release_steps`."""
    out = []
    if not reservoirs.get("valid"):
        return out
    for step in steps:
        if step["from"] in selected and step["to"] in selected:
            vector = step["vector"]
            out.append(
                {
                    **step,
                    "deltaG0_eV": selected[step["to"]]
                    - selected[step["from"]]
                    + vector_free_energy(vector, reservoirs, conditions, conditions.to_she(0.0, scale)),
                    "electrons": vector_electrons(vector),
                    "chemical": is_chemical(vector),
                }
            )
    return out


def _by_representative(nodes):
    members = {}
    for nid, node in nodes.items():
        members.setdefault(node.get("computed_by") or nid, []).append(nid)
    return members


def transient_nodes(node_ids, nodes, steps, tolerance=DOWNHILL_TOLERANCE_EV):
    """Nodes that give a reaction's product off downhill without electron transfer: {node_id: reason}.

    The formation free energy counts H2O and H+ only, not the product, so a node that holds
    the product bound (bound O2) can look like the most stable termination although it gives
    the product off at every potential and pH. Such a node (or a symmetry image of it) with a
    downhill *chemical* release step is transient: an intermediate under turnover, not a
    sink. ``steps`` from :func:`release_free_energies`; the same rule as the Pourbaix diagram.
    """
    members, out = _by_representative(nodes), {}
    for nid in node_ids:
        for member in members.get(nid, [nid]):
            step = next(
                (s for s in steps if s["from"] == member and s["chemical"] and s["deltaG0_eV"] <= tolerance), None
            )
            if step is not None:
                out[nid] = (
                    f"gives {step['species']} off downhill without electron transfer ({step['edge_id']}"
                    + (", a step no family wrote" if step["proposed"] else "")
                    + f", {step['deltaG0_eV']:+.3f} eV at every U)"
                )
                break
    return out


def releasing_nodes(node_ids, nodes, steps, potential, tolerance=DOWNHILL_TOLERANCE_EV):
    """Nodes whose electrochemical product release runs downhill at ``potential``: {node_id: reason}.

    They stay candidate resting states (their formation free energy counts the electrons), and
    are flagged as intermediates under turnover, as in the Pourbaix diagram.
    """
    members, out = _by_representative(nodes), {}
    for nid in node_ids:
        for member in members.get(nid, [nid]):
            for step in steps:
                if step["from"] != member or step["chemical"]:
                    continue
                value = step["deltaG0_eV"] - step["electrons"] * float(potential)
                if value <= tolerance:
                    out[nid] = f"releases the product downhill by {step['edge_id']} ({value:+.3f} eV)"
                    break
            if nid in out:
                break
    return out


def energy_inputs(
    campaign,
    references_path,
    include_vibrations=True,
    temperature_K=None,
    low_frequency_cutoff_cm1=None,
    imaginary_tolerance_cm1=None,
    energy_source=None,
    out_dir=None,
    registry=None,
    species=None,
):
    """Node (free) energies and reservoir chemical potentials of a campaign, validated.

    Shared by ``thermo``, ``pourbaix`` and ``pathways --rank``. ``species`` adds reservoir
    species whose reference molecules are required (default: those of every edge and the
    main reaction's formation species). Returns a dict of everything the analyses use.
    """
    campaign = Path(campaign).resolve()
    config, settings = _settings(campaign, temperature_K, low_frequency_cutoff_cm1, imaginary_tolerance_cm1)
    registry = registry or campaign_registry(campaign, config=config)
    central = central_mode(config)
    if energy_source is None:
        energy_source = "final" if central == "final" else "static"
    if energy_source not in ("static", "final"):
        raise ValueError("energy_source must be 'static' or 'final'")
    if central == "final" and energy_source != "final":
        raise ValueError(
            "frequency.central = 'final': node energies are the 02_final E0 (04_static only seeds "
            "the frequency displacements); use energy_source='final'."
        )
    if include_vibrations and energy_source != "static" and central == "refine":
        raise ValueError(
            "Vibrational free energies are referenced to the 04_static geometry; use energy_source='static'."
        )
    out_dir = Path(out_dir) if out_dir else campaign / "analysis"
    thermo_cfg = config.get("thermodynamics", {})
    collected = collect(campaign, out_dir, registry=registry, **settings)
    network = _load(campaign / "network.json")
    validation = validate_network(network, registry)
    if not validation["valid"]:
        _write(out_dir / "network_validation.json", validation)
        raise ValueError("Network balance/connectivity validation failed: " + "; ".join(validation["errors"][:5]))
    main = registry.reaction(network.get("reaction") or config.get("reaction", "oer"))
    vectors = {edge["id"]: edge_vector(edge) for edge in network.get("edges", [])}
    wanted = set(species or ())
    for vector in vectors.values():
        wanted |= set(vector)
    wanted |= set(main.formation_species)
    needed = set()
    for name in wanted:
        needed |= RESERVOIR_MOLECULES.get(name) or set(SPECIES[name].references or (name,))
    refs_path = Path(references_path)
    refs = read_reference_file(refs_path)
    reservoirs = reservoir_chemical_potentials(
        refs,
        settings["temperature_K"],
        float(thermo_cfg.get("water_equilibrium_pressure_bar", 0.035)),
        needed=tuple(sorted(needed)),
    )
    corrections = (
        None
        if include_vibrations
        else thermo_cfg.get("electronic_mode_corrections", {"per_H_eV": 0.30, "per_O_eV": 0.05})
    )
    if corrections and abs(float(settings["temperature_K"]) - 298.15) > 0.5:
        reservoirs.setdefault("warnings", []).append(
            f"The screening corrections (thermodynamics.electronic_mode_corrections: {corrections}) are "
            f"tabulated ZPE - TS values for 298.15 K; at {settings['temperature_K']:g} K the TS part differs. Set "
            "values for this temperature, or compute 05_freq."
        )
    selected, node_rows, assignment_key = node_energies(
        campaign, network, collected, include_vibrations, energy_source, corrections
    )
    level = (
        "CHE_with_partial_Hessian_free_energy"
        if include_vibrations
        else f"CHE_screening_{energy_source}_E0_plus_tabulated_adsorbate_ZPE_TS"
        if corrections
        else f"CHE_screening_{energy_source}_E0_without_adsorbate_corrections"
    )
    return {
        "campaign": campaign,
        "config": config,
        "settings": settings,
        "registry": registry,
        "central": central,
        "energy_source": energy_source,
        "out_dir": out_dir,
        "thermo_cfg": thermo_cfg,
        "collected": collected,
        "network": network,
        "validation": validation,
        "main": main,
        "vectors": vectors,
        "references_path": refs_path,
        "references": refs,
        "reservoirs": reservoirs,
        "corrections": corrections,
        "selected": selected,
        "node_rows": node_rows,
        "assignment_key": assignment_key,
        "level": level,
    }


def analyze_thermodynamics(
    campaign,
    references_path,
    out_dir=None,
    potentials_V=None,
    include_vibrations=True,
    temperature_K=None,
    low_frequency_cutoff_cm1=None,
    imaginary_tolerance_cm1=None,
    energy_source=None,
    scale=None,
    pH=None,
    activities=None,
    registry=None,
):
    """Free energies of edges and cycles where all inputs are valid; never substitutes missing data.

    ``include_vibrations=True`` uses G = E0 + F_vib(05_freq). Without vibrations the analysis is
    a screening estimate: G = E0 + tabulated adsorbate ZPE - TS per added atom
    (``thermodynamics.electronic_mode_corrections``). Potentials are on ``scale`` (default:
    the campaign reaction's scale, RHE for OER and SHE for CER); ``pH`` and ``activities``
    (e.g. {"Cl-": 1.0}) are analysis settings that need no new calculation.
    """
    data = energy_inputs(
        campaign,
        references_path,
        include_vibrations,
        temperature_K,
        low_frequency_cutoff_cm1,
        imaginary_tolerance_cm1,
        energy_source,
        out_dir,
        registry,
    )
    campaign, settings, registry = data["campaign"], data["settings"], data["registry"]
    central, energy_source, out_dir = data["central"], data["energy_source"], data["out_dir"]
    thermo_cfg, collected, network = data["thermo_cfg"], data["collected"], data["network"]
    validation, main, vectors = data["validation"], data["main"], data["vectors"]
    refs_path, refs, reservoirs = data["references_path"], data["references"], data["reservoirs"]
    corrections, selected, node_rows = data["corrections"], data["selected"], data["node_rows"]
    assignment_key, level = data["assignment_key"], data["level"]
    scale = (scale or thermo_cfg.get("potential_scale") or main.potential_scale).upper()
    conditions = Conditions(
        settings["temperature_K"],
        pH if pH is not None else thermo_cfg.get("pH", 0.0),
        {**(thermo_cfg.get("activities") or {}), **(activities or {})},
    )
    edge_map = {e["id"]: e for e in network.get("edges", [])}
    notes = []
    if potentials_V is not None:
        potentials, given_scale = potentials_V, scale
    elif thermo_cfg.get("potentials_V"):
        potentials = thermo_cfg["potentials_V"]
        given_scale = str(thermo_cfg.get("potential_scale") or main.potential_scale).upper()
    else:
        potentials, given_scale = list(main.default_potentials), main.potential_scale.upper()
    potentials = [float(x) for x in potentials]
    if not all(np.isfinite(potentials)):
        raise ValueError("Potentials must be finite numbers.")
    if given_scale != scale:
        # the configured potentials keep their physical meaning on another scale
        converted = [round(conditions.from_she(conditions.to_she(u, given_scale), scale), 4) for u in potentials]
        notes.append(
            f"The configured potentials {potentials} V vs {given_scale} are reported as {converted} V vs {scale} "
            f"(pH {conditions.pH:g}); pass --potentials to choose them on the {scale} scale."
        )
        potentials = converted
    nodes = {n["id"]: n for n in network.get("nodes", [])}
    # Edge free energies at 0 V on the analysis scale, and electrons per edge.
    delta_zero, edge_rows = {}, []
    for key, edge in edge_map.items():
        vector = vectors[key]
        available = reservoirs["valid"] and edge["from"] in selected and edge["to"] in selected
        if available:
            delta_zero[key] = (
                selected[edge["to"]]
                - selected[edge["from"]]
                + vector_free_energy(vector, reservoirs, conditions, conditions.to_she(0.0, scale))
            )
        n_e = vector_electrons(vector)
        for potential in potentials:
            edge_rows.append(
                {
                    "edge_id": key,
                    "from": edge["from"],
                    "to": edge["to"],
                    "family": edge.get("family"),
                    "reservoirs": vector,
                    "electrons": n_e,
                    "potential_V": potential,
                    "scale": scale,
                    "deltaG_eV": delta_zero[key] - n_e * potential if available else None,
                    "status": "thermodynamic_endpoints_available_path_unvalidated" if available else "unresolved",
                }
            )
    # Resting states: formation free energy of every computed node relative to the reference,
    # among the nodes that do not give the product off chemically downhill (transient_nodes).
    reference = network.get("reference", "ref")
    resting_rows, resting = [], {}
    network_reactions = [
        registry.reaction(name)
        for name in sorted({main.name} | {c.get("reaction") or main.name for c in network.get("cycles", [])})
    ]
    from .pathways import product_release_steps

    steps = release_free_energies(
        product_release_steps(campaign, network, registry, network_reactions), selected, reservoirs, conditions, scale
    )
    if reservoirs["valid"] and reference in selected:
        for potential in potentials:
            formation = {}
            for nid in selected:
                if nodes[nid].get("computed_by"):
                    continue
                vector = main.formation_vector(nodes[nid]["delta_atoms"])
                formation[nid] = formation_free_energy(
                    selected[nid], selected[reference], vector, reservoirs, potential, conditions, scale
                )
            transient = transient_nodes(formation, nodes, steps)
            releasing = releasing_nodes(formation, nodes, steps, potential)
            candidates = {nid: value for nid, value in formation.items() if nid not in transient} or dict(formation)
            best_id = min(candidates, key=lambda nid: (candidates[nid], nid))
            resting[potential] = {
                "node_id": best_id,
                "formation_free_energy_eV": candidates[best_id],
                "formation": formation,
                "candidates": set(candidates),
                "transient": transient,
                "releasing": releasing,
            }
            rank = 0
            for nid, value in sorted(formation.items(), key=lambda x: (x[1], x[0])):
                rank += nid in candidates
                resting_rows.append(
                    {
                        "potential_V": potential,
                        "scale": scale,
                        "rank": rank if nid in candidates else None,
                        "node_id": nid,
                        "formation_free_energy_eV": value,
                        "transient": transient.get(nid),
                        "releases_product_downhill": releasing.get(nid),
                        "delta_atoms": nodes[nid]["delta_atoms"],
                    }
                )
    cycle_rows, cycles_full = [], []
    for cycle in network.get("cycles", []):
        rx = registry.reaction(cycle.get("reaction") or main.name)
        declared = cycle["edge_ids"]
        shift = (
            release_last_rotation([{"reservoirs": vectors[k]} for k in declared], rx.release)
            if all(k in edge_map for k in declared)
            else 0
        )
        edge_ids = declared[shift:] + declared[:shift]
        node_ids = [edge_map[key]["from"] for key in edge_ids]
        complete = reservoirs["valid"] and all(key in delta_zero for key in edge_ids)
        missing = [key for key in edge_ids if key not in delta_zero]
        net_zero = (
            vector_free_energy(rx.net, reservoirs, conditions, conditions.to_she(0.0, scale))
            if reservoirs["valid"]
            else None
        )
        equilibrium = net_zero / rx.electrons if net_zero is not None else None
        for potential in potentials:
            row = {
                "cycle_id": cycle["id"],
                "family": cycle.get("family"),
                "reaction": rx.name,
                "potential_V": potential,
                "scale": scale,
                "status": "available" if complete else "unresolved",
                "symmetry_representative": cycle.get("symmetry_representative", True),
                "symmetry_equivalent_cycles": cycle.get("symmetry_equivalent_cycles", []),
                "discovered": bool(cycle.get("discovered")),
                "node_ids": node_ids,
                "missing_edges": missing,
            }
            if complete:
                metrics = cycle_metrics(
                    [{"reservoirs": vectors[k]} for k in edge_ids],
                    [delta_zero[k] for k in edge_ids],
                    potential,
                    [vector_electrons(vectors[k]) for k in edge_ids],
                    equilibrium,
                    release=rx.release,
                    scale=scale,
                )
                if metrics["rotation"] != 0:
                    raise AssertionError("cycle rotation must already be applied")
                expected_net = net_zero - rx.electrons * potential
                if abs(metrics["net_deltaG_eV"] - expected_net) > 1e-6:
                    raise ValueError(f"Cycle {cycle['id']} violates thermodynamic closure.")
                row.update(metrics)
                row["lowest_intermediate_node"] = node_ids[metrics["resting_state_index"]]
                if potential in resting:
                    network_rest = resting[potential]
                    formation = network_rest["formation"]
                    canonical = [nodes[n].get("computed_by") or n for n in node_ids]
                    in_cycle = [formation[n] for n in canonical if n in network_rest["candidates"]] or [
                        formation[n] for n in canonical if n in formation
                    ]
                    row["network_resting_state_node"] = network_rest["node_id"]
                    row["cycle_contains_network_resting_state"] = network_rest["node_id"] in canonical
                    row["resting_state_penalty_eV"] = (
                        max(0.0, min(in_cycle) - network_rest["formation_free_energy_eV"]) if in_cycle else None
                    )
            row["declared_rotation"] = shift
            cycle_rows.append(row)
            cycles_full.append(dict(row, edge_ids=edge_ids, declared_edge_ids=declared))
    _csv(
        out_dir / "nodes.csv",
        node_rows,
        [
            "node_id",
            "family",
            "site",
            "computed_by",
            "n_variants",
            "n_accepted_variants",
            "selected_state_id",
            "node_energy_eV",
            "status",
            "reassigned_in",
            "reassigned_out",
        ],
    )
    _csv(
        out_dir / "edges.csv",
        edge_rows,
        ["edge_id", "from", "to", "family", "reservoirs", "electrons", "potential_V", "scale", "deltaG_eV", "status"],
    )
    _csv(
        out_dir / "cycles.csv",
        cycle_rows,
        [
            "cycle_id",
            "family",
            "reaction",
            "discovered",
            "symmetry_representative",
            "potential_V",
            "scale",
            "status",
            "largest_step_deltaG_eV",
            "Gmax_U_eV",
            "cyclic_span_eV",
            "lowest_intermediate_node",
            "network_resting_state_node",
            "cycle_contains_network_resting_state",
            "resting_state_penalty_eV",
            "net_deltaG_eV",
            "limiting_potential_V",
            "thermodynamic_overpotential_V",
            "potential_independent_uphill_step",
            "contains_condensed_multi_electron_step",
            "symmetry_equivalent_cycles",
            "missing_edges",
        ],
    )
    _csv(
        out_dir / "resting_states.csv",
        resting_rows,
        [
            "potential_V",
            "scale",
            "rank",
            "node_id",
            "formation_free_energy_eV",
            "transient",
            "releases_product_downhill",
            "delta_atoms",
        ],
    )
    best = best_cycles(cycle_rows, potentials)
    result = {
        "schema_version": 4,
        "settings": settings,
        "potentials_V": potentials,
        "scale": scale,
        "conditions": conditions.describe(),
        "analysis_level": level,
        "energy_source": energy_source,
        "frequency_central": central,
        "electronic_mode_corrections": corrections,
        "reference_file_sha256": file_hash(refs_path) if refs_path.exists() else None,
        "references_supplied": refs,
        "reservoirs": reservoirs,
        "warnings": list(reservoirs.get("warnings") or []),
        "notes": notes + list(reservoirs.get("notes") or []),
        "network_validation": validation,
        "nodes": node_rows,
        "node_energies_eV": selected,
        "edges": edge_rows,
        "cycles": cycles_full,
        "best_cycles": best,
        "network_resting_states": {
            str(k): {
                "node_id": v["node_id"],
                "formation_free_energy_eV": v["formation_free_energy_eV"],
                "releases_product_downhill": v["releasing"].get(v["node_id"]),
                "transient_nodes": v["transient"],
            }
            for k, v in resting.items()
        },
        "resting_state_interpretation": (
            "Thermodynamically preferred computed intermediate at each potential (CHE formation free energy "
            "relative to the native surface). A node that gives the reaction's product off downhill without "
            "electron transfer (bound O2 desorbing) is transient at every potential and listed in "
            "transient_nodes instead; a node whose electrochemical release runs downhill stays a candidate "
            "and is flagged (releases_product_downhill), as in the Pourbaix diagram. It identifies off-cycle "
            "sinks within the sampled space; it is not a kinetically populated resting state under turnover."
        ),
        "microstates_needing_review": [
            {
                "state_id": s["state_id"],
                "built_for_node_id": s["node_id"],
                "status": s["microstate_status"],
                "microstate": (s["topology"] or {}).get("microstate"),
                "energy_e0_eV": s["energy_e0_eV"],
                "energy_final_e0_eV": s["energy_final_e0_eV"],
            }
            for s in collected["states"]
            if s.get(assignment_key) is None and (s["energy_e0_eV"] is not None or s["energy_final_e0_eV"] is not None)
        ],
        "variant_selection": "lowest accepted energy among the calculated structures whose relaxed site occupancy "
        "matches each node (including structures built for another node that relaxed into "
        "it); symmetry twins take their representative's value; a thermodynamic envelope, "
        "not a kinetic accessibility claim",
        "limitations": [
            "All reaction connections remain proposed unless independently investigated.",
            "A favorable thermodynamic descriptor does not establish a rate or dominant mechanism.",
            "Node variants with unresolved connectivity, convergence or imaginary modes are excluded.",
            "A partial Hessian neglects vibration changes in the frozen region.",
            "The resting-state check covers computed intermediates only and is thermodynamic.",
            "Reference standard states and DFT compatibility remain the user's responsibility.",
        ],
    }
    _write(out_dir / "thermodynamics.json", result)
    _write(out_dir / "network_validation.json", validation)
    return result


def best_cycles(cycle_rows, potentials):
    """Per potential and reaction: the available cycles with the lowest Gmax(U) and eta_TD.

    Ties (a cycle and its symmetry images share every value) go to the symmetry representative.
    """
    out = {}

    def key(field):
        return lambda r: (r[field], not r.get("symmetry_representative", True), r["cycle_id"])

    for potential in potentials:
        rows = [r for r in cycle_rows if r["status"] == "available" and r["potential_V"] == potential]
        for reaction in sorted({r["reaction"] for r in rows}):
            subset = [r for r in rows if r["reaction"] == reaction]
            by_gmax = min(subset, key=key("Gmax_U_eV"))
            with_eta = [r for r in subset if r.get("thermodynamic_overpotential_V") is not None]
            by_eta = min(with_eta, key=key("thermodynamic_overpotential_V")) if with_eta else None
            out.setdefault(str(potential), {})[reaction] = {
                "lowest_Gmax": {
                    "cycle_id": by_gmax["cycle_id"],
                    "Gmax_U_eV": by_gmax["Gmax_U_eV"],
                    "discovered": by_gmax["discovered"],
                },
                "lowest_overpotential": (
                    {
                        "cycle_id": by_eta["cycle_id"],
                        "thermodynamic_overpotential_V": by_eta["thermodynamic_overpotential_V"],
                        "discovered": by_eta["discovered"],
                    }
                    if by_eta
                    else None
                ),
            }
    return out


def read_reference_file(path):
    """References as JSON or YAML (molecule -> E and Gcorr, plus conventions)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"References file not found: {path} (write a template with `mechanet references`)")
    if path.suffix.lower() in (".yaml", ".yml"):
        from .fileio import read_yaml

        return read_yaml(path)
    return json.loads(path.read_text())
