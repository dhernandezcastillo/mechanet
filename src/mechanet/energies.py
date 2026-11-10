"""Energies of intermediates and steps: the inputs of every thermodynamic analysis.

:func:`node_energies` gives the electronic and free energy of each intermediate from the
collected results; :func:`energy_inputs` adds the references, the reservoir chemical potentials
and the same-settings check (:func:`protocol_consistency`) that ``thermo``, ``pathways`` and
``pourbaix`` share. Part of :mod:`mechanet.analysis` up to 0.1.4, which still exports every name.
"""

from __future__ import annotations

import json
from pathlib import Path

from .collection import collect, _settings
from .fileio import _load, _write
from .plugins import campaign_registry
from .reactions import BUILTIN_SPECIES, clean, electrons as vector_electrons, SPECIES
from .results import read_incar
from .thermo import DOWNHILL_TOLERANCE_EV, reservoir_chemical_potentials, validate_network, vector_free_energy
from .vasp import central_mode, energy_stage, expand_values

# The table of 0.1.4, kept for code that reads it: the reference molecules each built-in
# species needs. They are declared by the species now (Species.references).
RESERVOIR_MOLECULES = {name: set(SPECIES[name].references) for name in sorted(BUILTIN_SPECIES)}


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


def _some(ids, limit=3):
    shown = ", ".join(ids[:limit]) + (", ..." if len(ids) > limit else "")
    return f"{len(ids)} state{'' if len(ids) == 1 else 's'}: {shown}"


def _run_settings(state_dir, stage):
    """What defines the Hamiltonian of one run: protocol tags, PAW dataset and DFT+U of each element.

    The tags are those `references --collect` compares for the molecules (functional, hybrid
    mixing, dispersion, cutoff, LASPH, solvation), read from the INCAR the stage ran with.
    PAW datasets and the species order come from the state's potcar_provenance.json (else the
    run's POTCAR or OUTCAR). Returns None when the stage has no INCAR.
    """
    from .references import CONDITIONAL_TAGS, PROTOCOL_TAGS, _paw_element, _paw_titles, _tag_value

    folder = Path(state_dir) / stage
    if not (folder / "INCAR").is_file():
        return None
    values = read_incar(folder / "INCAR")
    entries = {}
    for tag in PROTOCOL_TAGS:
        switch = CONDITIONAL_TAGS.get(tag)
        if switch and _tag_value(values.get(switch), switch) == "FALSE":
            continue
        entries[tag] = _tag_value(values.get(tag), tag)
    provenance = _load(Path(state_dir) / "potcar_provenance.json", {}) or {}
    datasets = {
        record["element"]: " ".join(str(record.get("titel", "")).split())
        for record in provenance.get("datasets", [])
        if record.get("element")
    }
    if not datasets:
        datasets = {_paw_element(t): t for t in _paw_titles(folder) if _paw_element(t)}
    entries.update({f"PAW {element}": title for element, title in datasets.items()})
    ldau = str(values.get("LDAU", "")).strip().strip(".").upper().startswith("T")
    entries["DFT+U"] = "on" if ldau else "off"
    species = provenance.get("species_order") or list(datasets)
    try:
        arrays = [expand_values(values.get(k, "")) for k in ("LDAUL", "LDAUU", "LDAUJ")] if ldau else []
    except ValueError:
        arrays = []
    if arrays and all(len(a) == len(species) for a in arrays):
        for element, l_value, u_value, j_value in zip(species, *arrays):
            entries[f"U {element}"] = "none" if l_value < 0 else f"L={l_value:g} U={u_value:g} J={j_value:g}"
    return entries


def protocol_consistency(campaign, state_ids, stage):
    """Whether the runs whose energies an analysis compares share one Hamiltonian.

    Compares, between the given states, what :func:`_run_settings` reads for ``stage``. Starting
    moments, parallelisation and convergence tags are not compared: they may differ between
    states without changing the Hamiltonian. A difference is reported, not refused, since a
    converged change of cutoff can be legitimate; the warning says what differs and where.
    Returns {"stage", "states_checked", "unchecked", "consistent", "differences", "warning"}.
    """
    records, unchecked = {}, []
    for sid in sorted(set(state_ids)):
        entries = _run_settings(Path(campaign) / "states" / sid, stage)
        if entries is None:
            unchecked.append(sid)
        else:
            records[sid] = entries
    differences = {}
    for key in sorted({k for entries in records.values() for k in entries}):
        seen = {}
        for sid, entries in records.items():
            if key in entries:
                seen.setdefault(entries[key], []).append(sid)
        if len(seen) > 1:
            differences[key] = dict(sorted(seen.items()))
    warning = None
    if differences:
        parts = [
            f"{key} " + " vs ".join(f"{value} ({_some(sids)})" for value, sids in groups.items())
            for key, groups in differences.items()
        ]
        warning = (
            f"The energies compared here come from {stage} runs with different settings: "
            + "; ".join(parts)
            + ". A free-energy difference between such runs mixes two Hamiltonians; recompute the minority "
            "with one protocol, or confirm that the difference is converged."
        )
    return {
        "stage": stage,
        "states_checked": len(records),
        "unchecked": unchecked,
        "consistent": not differences,
        "differences": differences,
        "warning": warning,
    }


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
        needed |= set(SPECIES[name].references or (name,))
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
    protocol_check = protocol_consistency(
        campaign,
        [row["selected_state_id"] for row in node_rows if row["selected_state_id"] and not row["computed_by"]],
        energy_stage(config, energy_source),
    )
    if protocol_check["warning"]:
        reservoirs.setdefault("warnings", []).append(protocol_check["warning"])
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
        "protocol_check": protocol_check,
        "level": level,
    }


def read_reference_file(path):
    """References as JSON or YAML (molecule -> E and Gcorr, plus conventions)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"References file not found: {path} (write a template with `mechanet references`)")
    if path.suffix.lower() in (".yaml", ".yml"):
        from .fileio import read_yaml

        return read_yaml(path)
    return json.loads(path.read_text())
