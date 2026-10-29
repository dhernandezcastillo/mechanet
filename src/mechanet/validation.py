"""Independent checks of a generated campaign: state identities, compositions and reaction bookkeeping."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .fileio import read_json, write_json
from .structure import read_poscar
from .thermo import validate_network


def validate_campaign(campaign, registry=None):
    """Check every state against its node and the seed, and every edge and cycle balance.

    Writes ``validation.json``. The scope is bookkeeping: no DFT result or pathway is validated.
    """
    from .plugins import campaign_registry

    root = Path(campaign)
    errors, warnings = [], []
    config = read_json(root / "config.json")
    registry = registry or campaign_registry(root, config=config)
    net = read_json(root / "network.json")
    nodes = {n["id"]: n for n in net["nodes"]}
    seed_path = root / config.get("inputs", {}).get("seed", "inputs/seed.vasp")
    seed = read_poscar(seed_path)
    base = Counter(seed.symbols)
    states = {}
    for path in sorted((root / "states").glob("*/state.json")):
        meta = read_json(path)
        sid = meta["id"]
        if sid in states:
            errors.append(f"Repeated state ID {sid}")
        states[sid] = meta
        structure = read_poscar(path.parent / "POSCAR")
        if structure.symbols != meta["symbols"] or len(structure) != len(meta["atom_ids"]):
            errors.append(f"{sid}: POSCAR/metadata atom ordering mismatch")
        if len(set(meta["atom_ids"])) != len(structure):
            errors.append(f"{sid}: atom IDs are not unique")
        for key in ("relax_mobile", "frequency_mobile"):
            array = np.asarray(meta[key])
            if array.shape != (len(structure), 3) or array.dtype != bool:
                errors.append(f"{sid}: invalid {key}")
        if not np.array_equal(structure.mobile, meta["relax_mobile"]):
            errors.append(f"{sid}: relaxation flags in POSCAR differ from state.json")
        composition = Counter(structure.symbols)
        delta = meta.get("delta_atoms", {})
        for element in set(base) | set(composition) | set(delta):
            if composition[element] - base[element] != int(delta.get(element, 0)):
                errors.append(
                    f"{sid}: {element} count {composition[element]} is not seed {base[element]} "
                    f"{int(delta.get(element, 0)):+d}"
                )
        if not np.allclose(structure.cell, seed.cell, atol=1e-10):
            errors.append(f"{sid}: cell differs from the seed")
        if meta["node_id"] not in nodes:
            errors.append(f"{sid}: node {meta['node_id']} missing from network.json")
        for bond in meta.get("expected_bonds", []):
            if len(bond) != 2 or any(i not in meta["atom_ids"] for i in bond):
                errors.append(f"{sid}: expected bond {bond} names a missing atom")
    for nid, node in nodes.items():
        representative = node.get("computed_by")
        if representative is not None:
            if node.get("state_ids"):
                errors.append(f"{nid}: has computed_by={representative} but also its own state_ids")
            seen, target = {nid}, representative
            while target in nodes and nodes[target].get("computed_by") is not None and target not in seen:
                seen.add(target)
                target = nodes[target]["computed_by"]
            if representative not in nodes:
                errors.append(f"{nid}: computed_by points to missing node {representative}")
            elif target in seen:
                errors.append(f"{nid}: computed_by chain is cyclic")
            elif target not in nodes:
                errors.append(f"{nid}: computed_by chain ends at missing node {target}")
            elif (
                nodes[representative]["delta_atoms"] != node["delta_atoms"]
                or nodes[target]["delta_atoms"] != node["delta_atoms"]
            ):
                errors.append(f"{nid}: computed_by {representative} has a different composition")
            elif not nodes[target].get("state_ids"):
                warnings.append(f"{nid}: representative {target} has no realizable geometry")
            continue
        if not node.get("state_ids"):
            warnings.append(f"{nid}: no realizable geometry (all proposed placements rejected)")
        for sid in node.get("state_ids", []):
            if sid not in states:
                errors.append(f"{nid}: missing state {sid}")
            elif {k: v for k, v in states[sid]["delta_atoms"].items() if v} != {
                k: v for k, v in node["delta_atoms"].items() if v
            }:
                errors.append(f"{nid}: variant {sid} has a different composition")
    reference = states.get("ref")
    if reference:
        masks = {
            aid: {key: reference[key][i] for key in ("relax_mobile", "frequency_mobile")}
            for i, aid in enumerate(reference["atom_ids"])
        }
        for sid, meta in states.items():
            for i, aid in enumerate(meta["atom_ids"]):
                for key in ("relax_mobile", "frequency_mobile"):
                    if aid in masks and meta[key][i] != masks[aid][key]:
                        errors.append(f"{sid}: inherited {key} changed for {aid}")
                    elif aid not in masks and not all(meta[key][i]):
                        errors.append(f"{sid}: new atom {aid} is not fully movable in {key}")
    balance = validate_network(net, registry)
    errors += balance["errors"]
    report = {
        "valid": not errors,
        "n_states": len(states),
        "n_nodes": len(nodes),
        "n_edges": len(net["edges"]),
        "n_cycles": len(net["cycles"]),
        "errors": errors,
        "warnings": warnings,
        "scope": "Input bookkeeping and identity checks; no DFT results or pathway validation implied",
    }
    write_json(root / "validation.json", report)
    return report
