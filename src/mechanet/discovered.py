"""Intermediates found by a relaxation: registering a new site occupancy as a node.

A finished structure whose relaxed site occupancy matches no intermediate of the campaign
waits for review (``mechanet review``). When it is a chemically sensible intermediate the
network did not list (the H of *OOH handed to a neighbouring bridge oxygen: *OO₆₅ · μ-OH₆₄,
say), ``mechanet review --discover STATE "why"`` adds that occupancy to network.json as a
node of the campaign, with the same definition as a generated one: the state of every
reactive site, the composition it implies, and its lattice-translation images as symmetry
twins. Nothing is registered silently: the user decides, and the reason is kept.

The structure then counts for the new node at the next collect, exactly as a structure
that relaxed into an existing intermediate; its label comes from the occupancy (the same
notation as every other intermediate), and ``mechanet pathways`` proposes the balanced
steps and cycles through it. The occupancy is written after shared protons between two
sites the intermediate leaves native are put back on their seed oxygens (see
:mod:`mechanet.tautomers`), so another tautomer of the background never becomes a node.

Caveats, reported with every registration: the node has one structure, made from a start
built for another intermediate, with no orientation search and no spin variants of its own;
like every other structure it still needs ``select``, the static check and frequencies.
``--undiscover NODE`` removes it again, with its twins and any adopted step or cycle that
uses them.
"""

from __future__ import annotations

from pathlib import Path

from .fileio import dump_json, read_json, short_hash, utc_now

DISCOVERED = "discovered"
BACKUP = "network.before_discover.json"
CAVEATS = (
    "one structure, from a start built for another intermediate: no orientation search and no spin variants "
    "of its own; like every structure it still needs select, the static check and frequencies"
)


def occupancy_key(changes, coupling):
    """The identity of an intermediate: its site changes and coupled pair (as the builder writes it)."""
    import json

    return json.dumps({"occupancy": changes, "coupling": sorted(coupling) if coupling else None}, sort_keys=True)


def node_id(changes, coupling):
    """Node ID by the builder's convention: n_<site><state>..._<hash of the occupancy key>."""
    tag = "_".join(f"{x[1:] if x.startswith('a') else x}{v}" for x, v in sorted(changes.items()))
    if coupling:
        tag += "_coupled"
    return "n_" + (tag or "pair") + "_" + short_hash(occupancy_key(changes, coupling), 6)


def _ordered(changes, first):
    """The sites of the original intermediate first (in its order), then the others by ID."""
    order = [s for s in first if s in changes] + sorted(s for s in changes if s not in first)
    return {s: changes[s] for s in order}


def is_discovered(node):
    return bool((node or {}).get("discovered_from"))


def images(network, changes, coupling, known=()):
    """Lattice-translation images of an occupancy that are not the occupancy itself or a known node.

    Returns [(image changes, image coupling, back)], ``back`` being the recorded translation that
    takes the image onto the occupancy: a symmetry twin stores the translation from itself to
    its representative, as the builder writes it (``mechanet.magnetism.FrameMaps``). An image
    without such a recorded inverse is left out.
    """
    translations = (network.get("symmetry") or {}).get("translations") or []
    sites = list(changes) + list(coupling or ())
    own = occupancy_key(changes, coupling)
    seen, out = {own}, []
    for translation in translations:
        site_map = translation.get("site_map") or {}
        if not all(s in site_map for s in sites):
            continue
        mapped = {site_map[s]: v for s, v in changes.items()}
        mapped_coupling = sorted(site_map[s] for s in coupling) if coupling else None
        key = occupancy_key(mapped, mapped_coupling)
        if key in seen or key in known:
            continue
        back = next(
            (t for t in translations if all((t.get("site_map") or {}).get(site_map[s]) == s for s in sites)),
            None,
        )
        if back is None:
            continue
        seen.add(key)
        out.append((mapped, mapped_coupling, back))
    return out


def _verdict(campaign, state_id):
    """(meta, topology, other reasons) of a finished structure's 02_final, as collect judges it."""
    from .analysis import _final_validity
    from .plugins import campaign_registry
    from .results import parse_run
    from .review import CONNECTIVITY_REASONS, run_text, stage_status
    from .vasp import STAGE_FINAL

    state_dir = campaign / "states" / state_id
    if not (state_dir / "state.json").is_file():
        raise ValueError(f"No state {state_id!r} in {campaign / 'states'}")
    if stage_status(state_dir) != "complete":
        raise ValueError(f"{state_id}: 02_final has not finished ({run_text(state_dir)})")
    config = read_json(campaign / "config.json")
    registry = campaign_registry(campaign, config=config)
    meta = read_json(state_dir / "state.json")
    require_hashes = bool(config.get("analysis", {}).get("require_input_hashes", True))
    reasons, _force, topology = _final_validity(
        state_dir, meta, parse_run(state_dir / STAGE_FINAL), require_hashes, config, registry
    )
    others = [r for r in reasons if not any(c in r.lower() for c in CONNECTIVITY_REASONS)]
    return meta, topology, others


def discover(campaign, state_id, comment):
    """Register the relaxed site occupancy of ``state_id`` as a node of the campaign.

    Only a finished structure that collect leaves unassigned because its occupancy matches
    no node qualifies. Returns the node, its twins and the new verdict of the structure.
    """
    from .analysis import _CONTEXT_CACHE
    from .review import REVIEW_FILE, _Campaign, state_verdict

    campaign = Path(campaign)
    comment = str(comment or "").strip()
    if not comment:
        raise ValueError("Give a reason: it is kept in network.json with the new intermediate")
    if (campaign / "states" / state_id / REVIEW_FILE).is_file():
        raise ValueError(
            f"{state_id} is settled by its review.json; remove that decision first (mechanet review --clear {state_id})"
        )
    meta, topology, others = _verdict(campaign, state_id)
    micro = topology.get("microstate") or {}
    camp = _Campaign(campaign)
    status = micro.get("status")
    if topology.get("valid_for_assigned_node"):
        raise ValueError(f"{state_id} is accepted for its own intermediate; there is nothing to discover")
    if status == "reassigned":
        target = micro.get("assigned_node_id")
        raise ValueError(f"{state_id} relaxed into {camp.label(target)} ({target}), an intermediate of this campaign")
    if status == "adsorbate_released":
        raise ValueError(f"{state_id}: {micro.get('reason')}; a released molecule is not an intermediate")
    if status == "surface_atom_released":
        from .review import surface_left_label

        left = micro.get("surface_left") or {}
        label = surface_left_label(micro, camp.occupancy_label)
        if label is None:
            raise ValueError(f"{state_id}: {micro.get('reason')}; the surface left behind: {left.get('reason')}")
        if left.get("node_id"):
            raise ValueError(
                f"{state_id}: {micro.get('reason')}; the surface left behind, {label} ({left['node_id']}), "
                "is already an intermediate of this campaign"
            )
        # PENDING(surface-left-structure): it could be registered with a start made from this
        # structure without the released molecule (docs/pending.md).
        raise ValueError(
            f"{state_id}: {micro.get('reason')}; the surface left behind, {label}, is a possible intermediate, but "
            "this structure still holds the released molecule and cannot stand for it: add that intermediate "
            "through the families of the project and generate again"
        )
    if status == "metal_attachment_changed":
        raise ValueError(
            f"{state_id}: a bond to the metal changed, which the site occupancy does not describe; "
            "accept it (--accept) or assign it (--assign) if it is still an intermediate of the campaign"
        )
    if status != "unlisted_microstate":
        reason = micro.get("reason") or topology.get("reason") or topology.get("status")
        raise ValueError(f"{state_id}: its relaxed structure cannot be written as an occupancy of the sites ({reason})")
    if others:
        raise ValueError(f"{state_id} is not accepted for other reasons too: {'; '.join(others)}")
    changes, coupling = micro["occupancy_changes"], micro.get("coupling")
    network = read_json(campaign / "network.json")
    nodes = {n["id"]: n for n in network["nodes"]}
    original = nodes.get(meta["node_id"]) or {}
    changes = _ordered(changes, list((original.get("occupancy") or {})))
    known = set()
    sites = read_json(campaign / "sites.json")["sites"]
    native = {s["id"]: s["native"] for s in sites}
    for node in nodes.values():
        node_changes = {k: v for k, v in (node.get("occupancy") or {}).items() if native.get(k) != v}
        known.add(occupancy_key(node_changes, node.get("coupling")))
    if occupancy_key(changes, coupling) in known:
        raise ValueError(f"{state_id}: its occupancy is already an intermediate of this campaign; run mechanet collect")
    new_id = node_id(changes, coupling)
    if new_id in nodes:
        raise ValueError(f"{new_id} is already in network.json")
    origin = {
        "state_id": state_id,
        "built_for_node_id": meta["node_id"],
        "comment": comment,
        "written_by": "mechanet review --discover",
        "written_utc": utc_now(),
    }
    if micro.get("tautomer_moved"):
        origin["background_tautomer_moved"] = micro["tautomer_moved"]
    delta = micro.get("delta_atoms") or meta.get("delta_atoms")
    node = {
        "id": new_id,
        "state_ids": [],
        "family": DISCOVERED,
        "families": [DISCOVERED],
        "site": "+".join(changes) + ("+coupled" if coupling else ""),
        "occupancy": changes,
        "coupling": list(coupling) if coupling else None,
        "source_ids": [],
        "aliases": [],
        "delta_atoms": delta,
        "geometry_status": "discovered",
        "discovered_from": origin,
    }
    twins = []
    for mapped, mapped_coupling, back in images(network, changes, coupling, known):
        twin_id = node_id(mapped, mapped_coupling)
        if twin_id in nodes or twin_id == new_id:
            continue
        site_map = back.get("site_map") or {}
        twins.append(
            {
                "id": twin_id,
                "state_ids": [],
                "family": DISCOVERED,
                "families": [DISCOVERED],
                "site": "+".join(mapped) + ("+coupled" if mapped_coupling else ""),
                "occupancy": mapped,
                "coupling": mapped_coupling,
                "source_ids": [],
                "aliases": [],
                "delta_atoms": delta,
                "computed_by": new_id,
                "geometry_status": "symmetry_equivalent",
                "symmetry": {
                    "translation_fractional": back.get("fractional"),
                    "max_residual_A": back.get("max_residual_A"),
                    "site_map": {s: site_map[s] for s in sorted(mapped) + sorted(mapped_coupling or [])},
                    "note": "Lattice-translation image of a discovered intermediate; treated as the same "
                    "calculation, not recomputed.",
                },
                "discovered_from": origin,
            }
        )
    if twins:
        node["symmetry_twins"] = [t["id"] for t in twins]
    node["symmetry_multiplicity"] = 1 + len(twins)
    backup = campaign / BACKUP
    if not backup.exists():
        dump_json(backup, network)
    network["nodes"] += [node, *twins]
    network.setdefault("discovered_nodes", []).append(
        {"node_id": new_id, "symmetry_twins": [t["id"] for t in twins], **origin}
    )
    dump_json(campaign / "network.json", network)
    _CONTEXT_CACHE.clear()
    problem = _collect(campaign)
    camp = _Campaign(campaign)
    return {
        "collect_error": problem,
        "node": node,
        "label": camp.label(new_id),
        "twins": [t["id"] for t in twins],
        "built_for": {"node_id": meta["node_id"], "label": camp.label(meta["node_id"])},
        "verdict": state_verdict(campaign, state_id),
        "caveats": CAVEATS,
    }


def _collect(campaign):
    """Collect again, so the analysis tables (and the status page) judge the structures with the new network.

    Returns None, or why collect failed (the network change stands; the next collect redoes it).
    """
    from .analysis import collect

    try:
        collect(campaign)
    except Exception as exc:  # noqa: BLE001 - reported to the user, never hidden
        return f"{type(exc).__name__}: {exc}"
    return None


def undiscover(campaign, node_id_):
    """Remove a discovered node, its symmetry twins, and the adopted steps and cycles that use them.

    Proposed steps the removed cycles brought in and no other cycle uses go too, and so do the
    cycles and steps of pathways/pathways.json that pass through the removed nodes.
    """
    from .analysis import _CONTEXT_CACHE

    campaign = Path(campaign)
    network = read_json(campaign / "network.json")
    nodes = {n["id"]: n for n in network["nodes"]}
    node = nodes.get(node_id_)
    if node is None:
        raise ValueError(f"Unknown intermediate {node_id_!r}")
    if not is_discovered(node):
        raise ValueError(f"{node_id_} was generated with the campaign, not discovered; only discovered ones can go")
    if node.get("computed_by"):
        node_id_, node = node["computed_by"], nodes[node["computed_by"]]
    gone = {node_id_, *node.get("symmetry_twins", [])}
    edge_ids = {e["id"] for e in network["edges"] if e["from"] in gone or e["to"] in gone}
    cycles = [c for c in network["cycles"] if edge_ids.intersection(c["edge_ids"])]
    kept = [c for c in network["cycles"] if c not in cycles]
    still_used = {e for c in kept for e in c["edge_ids"]}
    brought = {e for c in cycles for e in c["edge_ids"]} - still_used
    edge_ids |= {e["id"] for e in network["edges"] if e["id"] in brought and e.get("family") == DISCOVERED}
    network["nodes"] = [n for n in network["nodes"] if n["id"] not in gone]
    network["edges"] = [e for e in network["edges"] if e["id"] not in edge_ids]
    network["cycles"] = kept
    network["discovered_nodes"] = [d for d in network.get("discovered_nodes", []) if d.get("node_id") != node_id_]
    if not network["discovered_nodes"]:
        del network["discovered_nodes"]
    dump_json(campaign / "network.json", network)
    report = campaign / "pathways" / "pathways.json"
    if report.is_file():
        data = read_json(report)
        stale = {e["id"] for e in data.get("new_edges", []) if e["from"] in gone or e["to"] in gone}
        data["new_edges"] = [e for e in data.get("new_edges", []) if e["id"] not in stale]
        data["cycles"] = [
            c
            for c in data.get("cycles", [])
            if not gone.intersection(c.get("node_ids", [])) and not stale & set(c["edge_ids"])
        ]
        dump_json(report, data)
    _CONTEXT_CACHE.clear()
    problem = _collect(campaign)
    return {
        "collect_error": problem,
        "node_id": node_id_,
        "removed_nodes": sorted(gone),
        "removed_edges": sorted(edge_ids),
        "removed_cycles": [c["id"] for c in cycles],
        "state_id": (node.get("discovered_from") or {}).get("state_id"),
    }


def discover_text(result):
    node = result["node"]
    origin = node["discovered_from"]
    lines = [
        f"added {result['label']} ({node['id']}) to network.json as an intermediate of this campaign",
        f"  from {origin['state_id']}, built for {result['built_for']['label']} ({result['built_for']['node_id']})",
    ]
    if result["twins"]:
        lines.append(f"  symmetry images: {', '.join(result['twins'])}")
    verdict = result["verdict"]
    if verdict["kind"] in ("reassigned", "accepted", "reviewed") and verdict["counts_for"] == node["id"]:
        lines.append(f"  {origin['state_id']} now counts for it")
    else:
        lines.append(f"  but {origin['state_id']} does not count for it yet: {verdict['summary']}")
    if result.get("collect_error"):
        lines.append(f"  collect failed ({result['collect_error']}); run mechanet collect")
    lines += [
        f"  caveats: {result['caveats']}",
        "next: mechanet pathways (steps and cycles through it; --adopt ID copies a cycle into network.json), "
        "then mechanet thermo",
    ]
    return "\n".join(lines)


def undiscover_text(result):
    lines = [f"removed {', '.join(result['removed_nodes'])} from network.json"]
    if result["removed_edges"] or result["removed_cycles"]:
        lines.append(
            f"  with {len(result['removed_edges'])} adopted steps and {len(result['removed_cycles'])} cycles that "
            f"used it{': ' + ', '.join(result['removed_cycles']) if result['removed_cycles'] else ''}"
        )
    if result.get("state_id"):
        lines.append(f"  {result['state_id']} needs review again")
    if result.get("collect_error"):
        lines.append(f"  collect failed ({result['collect_error']}); run mechanet collect")
    return "\n".join(lines)
