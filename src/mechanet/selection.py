"""Choose the variants of each intermediate that continue to the single point and the frequencies.

:func:`select_for_thermo` keeps, per computed intermediate, the accepted ``02_final`` variants
within an energy window of the lowest, after merging variants that relaxed into the same
minimum (:func:`structure_difference`). Part of :mod:`mechanet.analysis` up to 0.1.4, which
still exports every name.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .collection import _final_validity, _usable
from .fileio import _load, _write
from .magnetism import magnetic_solutions, MomentTracker
from .plugins import campaign_registry
from .results import parse_run
from .structure import read_poscar
from .tautomers import NOTE as TAUTOMER_NOTE, TautomerCheck
from .topology import _campaign_context, topology_check
from .vasp import STAGE_FINAL, STAGE_FREQ, STAGE_REFINE, STAGE_STATIC


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


DUPLICATE_ENERGY_EV = 0.005
DUPLICATE_MAX_SHIFT_A = 0.10


def _assignment(cost):
    """Pairs (i, j) of a minimum-cost one-to-one matching (Hungarian with scipy, greedy without)."""
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        return list(zip(rows.tolist(), cols.tolist()))
    except ImportError:
        pairs, used_i, used_j = [], set(), set()
        for flat in np.argsort(cost, axis=None):
            i, j = (int(x) for x in np.unravel_index(flat, cost.shape))
            if i not in used_i and j not in used_j:
                pairs.append((i, j))
                used_i.add(i)
                used_j.add(j)
        return pairs


def structure_difference(first, second, first_moments=None, second_moments=None):
    """How far apart two relaxed structures of one composition are, ignoring atom labels.

    Atoms of each element are matched one to one to their nearest partner (minimum image), so
    two structures whose H atoms swapped places still match. Returns {max_shift_A, rms_shift_A,
    max_moment_change_muB (None without moments)}, or None for different compositions.
    """
    if sorted(first.symbols) != sorted(second.symbols):
        return None
    from .structure import pbc_distances

    shifts, gap = [], None
    have = (
        first_moments is not None
        and second_moments is not None
        and len(first_moments) == len(first.symbols)
        and len(second_moments) == len(second.symbols)
    )
    inv = np.linalg.inv(first.cell)
    for element in sorted(set(first.symbols)):
        i = [k for k, s in enumerate(first.symbols) if s == element]
        j = [k for k, s in enumerate(second.symbols) if s == element]
        cost = pbc_distances(first.positions[i], second.positions[j], first.cell, inv)
        for p, q in _assignment(cost):
            shifts.append(float(cost[p, q]))
            if have:
                change = abs(float(first_moments[i[p]]) - float(second_moments[j[q]]))
                gap = change if gap is None else max(gap, change)
    shifts = np.asarray(shifts)
    return {
        "max_shift_A": round(float(shifts.max()), 4) if len(shifts) else 0.0,
        "rms_shift_A": round(float(np.sqrt((shifts**2).mean())), 4) if len(shifts) else 0.0,
        "max_moment_change_muB": None if gap is None else round(gap, 3),
    }


def _merge_duplicates(campaign, variants, magmoms, energy_eV, shift_A, moment_muB):
    """(kept, duplicates): variants (lowest energy first) that relaxed into the same minimum as a
    lower one are set aside, each with the variant it repeats and the differences found."""
    structures, kept, duplicates = {}, [], []

    def structure(sid):
        if sid not in structures:
            try:
                structures[sid] = read_poscar(campaign / "states" / sid / STAGE_FINAL / "CONTCAR")
            except (OSError, ValueError):
                structures[sid] = None
        return structures[sid]

    for variant in variants:
        sid = variant["state_id"]
        match = None
        for lower in kept:
            gap = abs(variant["energy_final_e0_eV"] - lower["energy_final_e0_eV"])
            if gap >= energy_eV or structure(sid) is None or structure(lower["state_id"]) is None:
                continue
            diff = structure_difference(
                structure(lower["state_id"]), structure(sid), magmoms.get(lower["state_id"]), magmoms.get(sid)
            )
            if diff is None or diff["max_shift_A"] >= shift_A:
                continue
            if diff["max_moment_change_muB"] is not None and diff["max_moment_change_muB"] > moment_muB:
                continue
            match = {"state_id": sid, "same_as": lower["state_id"], "delta_E_meV": round(1000 * gap, 2), **diff}
            break
        if match:
            duplicates.append(match)
        else:
            kept.append(variant)
    return kept, duplicates


def select_for_thermo(
    campaign,
    energy_window_eV=None,
    max_variants_per_node=None,
    include_partial=False,
    out_path=None,
    cycles=None,
    registry=None,
    families=None,
    keep_duplicates=False,
    nodes=None,
):
    """Choose which 02_final variants proceed to the later stages.

    For every computed node (symmetry twins excluded), the accepted 02_final variants within
    ``energy_window_eV`` of the node minimum are kept, up to ``max_variants_per_node``.
    Variants that relaxed into the same minimum as a lower one (energies within
    ``thermo_selection.duplicate_energy_eV``, every atom within
    ``thermo_selection.duplicate_max_shift_A`` of a partner of its element, local moments
    within ``analysis.moment_change_muB``) are merged first, keeping the lowest, unless
    ``keep_duplicates``. A node with variants whose 02_final has not completed is left
    waiting unless ``include_partial`` is set. ``cycles`` and ``families`` restrict the
    selection to the nodes of those cycles, or of the cycles of those families (every
    structure is still evaluated, so one that relaxed into such a node competes there).
    ``nodes`` restricts it to these node IDs as well (the candidates of a surface Pourbaix
    diagram, so that frequencies are run for them only).
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
    same_energy = float(selection_cfg.get("duplicate_energy_eV", DUPLICATE_ENERGY_EV))
    same_shift = float(selection_cfg.get("duplicate_max_shift_A", DUPLICATE_MAX_SHIFT_A))
    if same_energy < 0 or same_shift < 0:
        raise ValueError("thermo_selection.duplicate_energy_eV and duplicate_max_shift_A must be >= 0")
    require_hashes = bool(config.get("analysis", {}).get("require_input_hashes", True))
    network = _load(campaign / "network.json")
    computed = [n for n in network.get("nodes", []) if not n.get("computed_by")]
    restrict = list(cycles or [])
    if families:
        present = sorted({c.get("family") for c in network.get("cycles", []) if c.get("family")})
        unknown = sorted(set(families) - set(present))
        if unknown:
            raise ValueError(f"No cycles of the families {unknown} in this campaign; it has {', '.join(present)}")
        restrict += [c["id"] for c in network["cycles"] if c.get("family") in set(families) and c["id"] not in restrict]
    wanted = _cycle_nodes(network, restrict) if (cycles or families) else None
    if nodes:
        known = {n["id"] for n in network.get("nodes", [])}
        unknown = sorted(set(nodes) - known)
        if unknown:
            raise ValueError(f"No intermediates {unknown[:5]} in this campaign")
        images = {n["id"]: n["computed_by"] for n in network.get("nodes", []) if n.get("computed_by")}
        wanted = (wanted or set()) | {images.get(node_id, node_id) for node_id in nodes}
    selected, per_node, waiting, unresolved = [], [], [], []
    evaluated, own_variants, moment_rows, magmoms = {}, {}, [], {}
    merged_total, tautomer_rows = [], []
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
                    moment_rows.append(
                        {
                            "state_id": sid,
                            "stages": {STAGE_FINAL: {"magmoms": final.get("magmoms")}},
                            "final_topology": topology,
                        }
                    )
                    tautomer_rows.append(
                        {
                            "state_id": sid,
                            "final_valid": not reasons,
                            "final_assigned_node_id": topology.get("assigned_node_id"),
                            "final_topology": topology,
                        }
                    )
                    magmoms[sid] = final.get("magmoms")
                evaluated[sid] = record
            own_variants[node["id"]].append(evaluated[sid])
    moments = MomentTracker(campaign, network, moment_rows, {}, STAGE_FINAL, "final_topology", config)
    context = _campaign_context(campaign)
    tautomers = TautomerCheck(network, (context or {}).get("native"), tautomer_rows, {}, "final_topology")
    several_solutions, several_tautomers = [], []
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
        candidates, duplicates = accepted, []
        if not keep_duplicates and len(accepted) > 1:
            near = [v for v in accepted if v["energy_final_e0_eV"] - minimum <= window + same_energy + 1e-12]
            kept, duplicates = _merge_duplicates(campaign, near, magmoms, same_energy, same_shift, moments.threshold)
            candidates = kept + [v for v in accepted if v not in near]
        if duplicates:
            extra["duplicates"] = duplicates
            merged_total += duplicates
        chosen = [v for v in candidates if v["energy_final_e0_eV"] - minimum <= window + 1e-12][:limit]
        for v in accepted:
            v["relative_eV"] = v["energy_final_e0_eV"] - minimum
        selected.extend(v["state_id"] for v in chosen)
        if len(accepted) > 1:
            solutions = magnetic_solutions(
                [
                    (v["state_id"], v["energy_final_e0_eV"], moments.state_moments(v["state_id"], node["id"])[0])
                    for v in accepted
                ],
                moments.threshold,
            )
            if len(solutions) > 1:
                extra["magnetic_solutions"] = solutions
                several_solutions.append(node["id"])
            energy = {v["state_id"]: v["energy_final_e0_eV"] for v in accepted}
            groups = []
            for group in tautomers.variants(node["id"]):
                sids = sorted((sid for sid in group["state_ids"] if sid in energy), key=lambda sid: energy[sid])
                if sids:
                    groups.append({"on": group["on"], "state_ids": sids, "relative_eV": energy[sids[0]] - minimum})
            if len(groups) > 1:
                extra["tautomers"] = sorted(groups, key=lambda g: g["relative_eV"])
                several_tautomers.append(node["id"])
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
            "families": sorted(families) if families else None,
            **({"nodes": sorted(nodes)} if nodes else {}),
            "duplicates": (
                "kept (--keep-duplicates)"
                if keep_duplicates
                else {
                    "energy_eV": same_energy,
                    "max_shift_A": same_shift,
                    "moment_change_muB": moments.threshold,
                    "rule": "a variant within these limits of a lower one (atoms matched per element, labels "
                    "ignored) is the same minimum; the lowest is kept",
                }
            ),
        },
        "n_selected": len(set(selected)),
        "duplicates_merged": merged_total,
        "waiting_nodes": waiting,
        "unresolved_nodes": unresolved,
        "nodes_with_several_magnetic_solutions": several_solutions,
        "nodes_with_several_tautomers": several_tautomers,
        "nodes": per_node,
        "note": (
            "Re-run after more 02_final results arrive (or use --include-partial to select from the variants "
            "finished so far); selection never removes states that already ran later stages."
        ),
        "magnetic_solutions_note": (
            "Variants whose 02_final local moments differ (a moment of at least analysis.moment_change_muB "
            f"(here {moments.threshold:g} muB) changing sign, or any heavy atom differing by more than that) "
            "converged to different magnetic solutions; nodes.magnetic_solutions lists them, lowest first. "
            "Selection is by energy only."
        ),
        "tautomers_note": TAUTOMER_NOTE,
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
