"""Collect calculation evidence, audit relaxed structures and evaluate complete cycles.

Collection is read-only with respect to calculation folders. Reports are regenerated from
the VASP outputs and input hashes, not from a stale cache. Unknown, failed, altered and
imaginary-mode results stay in the report but are excluded from free energies.

Stage sequence (frequency.central = "refine", the default): 01_loose -> 02_final ->
03_refine -> 04_static -> 05_freq; the node energy is the 04_static E0. With
frequency.central = "final" there is no 03_refine and the node energy is the 02_final E0;
04_static seeds the 05_freq displacements at that geometry.

The checks, the collection, the selection and the node energies live in
:mod:`mechanet.topology`, :mod:`mechanet.collection`, :mod:`mechanet.selection` and
:mod:`mechanet.energies`; every name they define is imported here, so code written for
``mechanet.analysis`` keeps working.
"""

from __future__ import annotations

# The imports of this module up to 0.1.4, kept whole: code that took a helper from here still finds it.
from pathlib import Path  # noqa: F401

import numpy as np

from . import elements  # noqa: F401
from .adsorbates import CORE_ELEMENTS, extra_elements, tree_key  # noqa: F401
from .fileio import file_hash, read_json  # noqa: F401
from .magnetism import NOTE as MAGNETISM_NOTE, MomentTracker, magnetic_solutions  # noqa: F401
from .plugins import campaign_registry, default_registry  # noqa: F401
from .reactions import SPECIES, clean, electrons as vector_electrons  # noqa: F401
from .results import FREQUENCY_CM1_PER_SQRT_EV_A2_AMU, parse_run, read_incar, same_solution  # noqa: F401
from .structure import distance_matrix, minimum_image, read_poscar  # noqa: F401
from .tautomers import (  # noqa: F401
    NOTE as TAUTOMER_NOTE,
    TautomerCheck,
    background_protons,
    normalized_owners,
    pair_hydrogens,
    pair_side,
    tautomer_record,
)
from .thermo import (  # noqa: F401
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
from .vasp import (  # noqa: F401
    STAGES,
    STAGE_FINAL,
    STAGE_REFINE,
    STAGE_STATIC,
    STAGE_FREQ,
    central_mode,
    energy_stage,
    expand_values,
    frequency_gates,
    plan_mode,
    protocol_stages,
)

# Every name below was defined in this module up to 0.1.4 and is still importable from it.
from .fileio import (  # noqa: F401
    _csv,
    _load,
    _write,
)
from .topology import (  # noqa: F401
    BOND_CUTOFF,
    DEFAULT_HYSTERESIS,
    FRAGMENT_NAMES,
    GENERIC_BOND_EDGES,
    TYPICAL_METAL_RADIUS_A,
    _CONTEXT_CACHE,
    _FRAGMENT_BONDS,
    _NATIVE_ID,
    _SUBSCRIPT,
    _assign_microstate,
    _binding_mode_only,
    _campaign_context,
    _final_geometry,
    _limits,
    _metals,
    _named,
    _occupancy_key,
    _own_release,
    _pair_type,
    _representative_node,
    _shared_protons,
    _signatures,
    classify_microstate,
    fragment_formula,
    released_fragments,
    released_text,
    topology_check,
)
from .collection import (  # noqa: F401
    ENERGY_SOURCE_TEXT,
    _ediffg,
    _final_validity,
    _force_max,
    _geometry_match,
    _provenance_gate,
    _relaxation_reasons,
    _settings,
    _smearing_warnings,
    _usable,
    collect,
    imaginary_mode_character,
)
from .selection import (  # noqa: F401
    DUPLICATE_ENERGY_EV,
    DUPLICATE_MAX_SHIFT_A,
    _assignment,
    _cycle_nodes,
    _merge_duplicates,
    select_for_thermo,
    structure_difference,
)
from .energies import (  # noqa: F401
    RESERVOIR_MOLECULES,
    _by_representative,
    _electronic_correction,
    _run_settings,
    _some,
    edge_vector,
    energy_inputs,
    is_chemical,
    node_energies,
    protocol_consistency,
    read_reference_file,
    release_free_energies,
    releasing_nodes,
    transient_nodes,
)


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
    tracker = MomentTracker(
        campaign,
        network,
        collected["states"],
        {row["node_id"]: row.get("selected_state_id") for row in node_rows},
        STAGE_FINAL if energy_source == "final" else STAGE_STATIC,
        "final_topology" if assignment_key == "final_assigned_node_id" else "topology",
        data["config"],
    )
    variant_solutions = {}
    for row in node_rows:
        solutions = tracker.variants(row["node_id"]) if row["computed_by"] is None else None
        row["magnetic_solutions"] = len(solutions) if solutions else None
        if solutions and len(solutions) > 1:
            variant_solutions[row["node_id"]] = {"selected_state_id": row["selected_state_id"], "solutions": solutions}
    tautomer_check = TautomerCheck(
        network,
        tracker.native,
        collected["states"],
        {row["node_id"]: row.get("selected_state_id") for row in node_rows},
        tracker.topology_key,
    )
    node_tautomers = {}
    for row in node_rows:
        groups = tautomer_check.variants(row["node_id"]) if row["computed_by"] is None else []
        row["tautomers"] = len(groups) if len(groups) > 1 else None
        if len(groups) > 1:
            node_tautomers[row["node_id"]] = {"selected_state_id": row["selected_state_id"], "tautomers": groups}
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
    cycle_rows, cycles_full, cycle_moments, cycle_tautomers = [], [], {}, {}
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
        moments = tracker.cycle(node_ids)
        if moments["status"] != "not_available":
            cycle_moments[cycle["id"]] = moments
        tautomers = tautomer_check.cycle(node_ids)
        if tautomers["status"] != "not_available":
            cycle_tautomers[cycle["id"]] = tautomers
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
                "spin_check": moments["status"],
                "spin_flags": len(moments["flags"]),
                "tautomer_check": tautomers["status"],
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
            "magnetic_solutions",
            "tautomers",
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
            "spin_check",
            "spin_flags",
            "tautomer_check",
            "symmetry_equivalent_cycles",
            "missing_edges",
        ],
    )
    _csv(
        out_dir / "cycle_moments.csv",
        [
            {
                "cycle_id": cycle_id,
                "position": k,
                "node_id": entry["node_ids"][k],
                "state_id": entry["state_ids"][k],
                "atom_id": atom["atom_id"],
                "element": atom["element"],
                "role": atom["role"],
                "local_moment_muB": value,
            }
            for cycle_id, entry in cycle_moments.items()
            for atom in entry["atoms"]
            for k, value in enumerate(atom["moments_muB"])
            if value is not None
        ],
        ["cycle_id", "position", "node_id", "state_id", "atom_id", "element", "role", "local_moment_muB"],
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
        "protocol_check": data["protocol_check"],
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
        "magnetism": {
            "stage": tracker.stage,
            "topology_key": tracker.topology_key,
            "settings": tracker.settings,
            "cycles_needing_review": sorted(
                cycle_id for cycle_id, entry in cycle_moments.items() if entry["status"] == "needs_review"
            ),
            "nodes_with_several_solutions": sorted(variant_solutions),
            "cycles": cycle_moments,
            "variants": variant_solutions,
            "note": MAGNETISM_NOTE,
        },
        "tautomers": {
            "topology_key": tautomer_check.topology_key,
            "cycles_needing_review": sorted(
                cycle_id for cycle_id, entry in cycle_tautomers.items() if entry["status"] == "needs_review"
            ),
            "nodes_with_several_tautomers": sorted(node_tautomers),
            "cycles": cycle_tautomers,
            "nodes": node_tautomers,
            "note": TAUTOMER_NOTE,
        },
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
