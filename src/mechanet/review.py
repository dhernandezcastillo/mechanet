"""Finished structures that collect did not accept: what happened to them, and your decisions.

A structure whose 02_final finished (the runner accepted the run) still has to pass
``collect``: its files must be the ones the runner accepted, and its relaxed connectivity
must still describe the intermediate it was built for. A structure that relaxed into another
intermediate of the campaign counts for that one. One that matches no intermediate, or whose
bond to the metal changed, waits for a decision in ``states/<state>/review.json``: accept it
as its intermediate, or count it for another intermediate of the same composition.

``mechanet review`` lists these structures in plain words (which H moved where, which bond
broke, which atom left the metal), the cycles they hold up and what to do; ``--accept`` and
``--assign`` write the decisions, ``--clear`` removes one. ``mechanet status`` marks the
structures the last collect did not accept.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from pathlib import Path

from .elements import default_role
from .fileio import read_json, utc_now, write_json
from .vasp import STAGE_FINAL

REVIEW_FILE = "review.json"
DECIDED = ("accepted_by_explicit_review", "assigned_by_explicit_review")
NOT_ACCEPTED = ("needs_review", "rejected", "reassigned", "no_bound_minimum")
# Reasons the change list already explains.
CONNECTIVITY_REASONS = ("relaxed connectivity is unavailable or needs explicit review",)
MICROSTATE_TEXT = {
    "unlisted_microstate": "the new arrangement of the sites is not an intermediate of this campaign",
    "metal_attachment_changed": "a bond between the metal and an adsorbate or ligand changed",
    "adsorbate_released": "no bound minimum from this start",
    "cell_changed": "the cell changed during the relaxation",
    "atom_mapping_mismatch": "the atoms of the relaxed structure do not match the starting structure",
}


def _read(path, default=None):
    try:
        return read_json(path)
    except (OSError, ValueError):
        return default


def stage_status(state_dir, stage=STAGE_FINAL):
    """The runner's status of one stage (attempt.json), or None when it never ran."""
    return (_read(Path(state_dir) / stage / "attempt.json", {}) or {}).get("status")


def run_text(state_dir):
    """How far a state got: finished, running, failed, ran part of the way or not run yet."""
    final, loose = stage_status(state_dir), stage_status(state_dir, "01_loose")
    if final == "complete":
        return "finished"
    if "running" in (final, loose):
        return "running"
    if "failed" in (final, loose):
        return "failed (see mechanet status)"
    return "ran part of the way" if final or loose else "not run yet"


# ---------------------------------------------------------------------------
# What happened to one structure
# ---------------------------------------------------------------------------


def _pairs(values):
    return [tuple(str(x) for x in pair[:2]) for pair in values or [] if len(pair) >= 2]


def describe_changes(topology, meta, occupancy_label=None):
    """Plain sentences for the changes collect found in a relaxed structure.

    ``topology`` is a ``topology_check`` result (``states.json``: ``final_topology``) and
    ``meta`` the state's state.json. Atoms are named by element and ID, e.g. "H a0087 moved
    from O a0067 to O a0066". ``occupancy_label(changes, coupling)`` names a relaxed site
    occupancy that is no intermediate of the campaign.
    """
    topology = topology or {}
    symbols = dict(zip(meta.get("atom_ids") or [], meta.get("symbols") or []))

    def atom(aid):
        return f"{symbols[aid]} {aid}" if aid in symbols else aid

    lost, new = _pairs(topology.get("lost_bonds")), _pairs(topology.get("new_bonds"))
    lines, explained = [], set()
    released = topology.get("released") or []
    gone = {a for item in released for a in item.get("atoms", [])}
    for item in released:
        heavy = [a for a in item.get("atoms", []) if symbols.get(a) != "H"] or list(item.get("atoms", []))
        lines.append(
            f"{item['formula']} ({', '.join(map(atom, heavy))}) released from {' and '.join(map(atom, item['from']))}"
        )
    tautomer = {
        h: entry
        for h, entry in (topology.get("tautomer") or {}).items()
        if entry.get("background") and entry.get("moved")
    }
    tautomer_note = ": a shared H between two sites this intermediate leaves native, another tautomer"
    moves, loose_ends = {}, []
    told = set()  # H-H pairs already named from their other atom
    for h in sorted({a for pair in lost + new for a in pair if symbols.get(a) == "H"}):
        before = [x for pair in lost if h in pair for x in pair if x != h]
        after = [x for pair in new if h in pair for x in pair if x != h]
        if before and after:
            moves[h] = (before, after)
        elif before:
            rest = [x for x in before if frozenset((h, x)) not in told]
            if rest:
                loose_ends.append(f"H {h} is no longer bonded to {' and '.join(map(atom, rest))}")
        else:
            rest = [x for x in after if frozenset((h, x)) not in told]
            if rest:
                loose_ends.append(f"H {h} is now bonded to {' and '.join(map(atom, rest))}")
        told |= {frozenset((h, x)) for x in before + after if symbols.get(x) == "H"}
        explained.add(h)
    # A relay reads in order: the H that starts it first, then the one that takes over.
    while moves:
        targets = {x for _before, after in moves.values() for x in after}
        h = next((h for h, (before, _a) in moves.items() if not targets.intersection(before)), next(iter(moves)))
        before, after = moves.pop(h)
        lines.append(
            f"H {h} moved from {' and '.join(map(atom, before))} to {' and '.join(map(atom, after))}"
            + (tautomer_note if h in tautomer else "")
        )
    lines += loose_ends
    for pair in lost:
        if not explained.intersection(pair) and not gone.intersection(pair):
            lines.append(f"the {atom(pair[0])}–{atom(pair[1])} bond broke")
    for pair in new:
        if not explained.intersection(pair):
            lines.append(f"a {atom(pair[0])}–{atom(pair[1])} bond formed")
    for shift in topology.get("shared_proton_transfers") or []:
        if shift.get("H") not in explained:
            start, end = atom(shift.get("from")), atom(shift.get("to"))
            lines.append(
                f"the shared H {shift.get('H')} moved from {start} to {end}"
                + (tautomer_note if shift.get("H") in tautomer else "")
            )
            explained.add(shift.get("H"))
    missing = [p for p in _pairs(topology.get("missing_expected_bonds")) if p not in lost and p[::-1] not in lost]
    lines += [f"the expected {atom(a)}–{atom(b)} bond is missing" for a, b in missing]

    def metal_first(pair):
        a, b = pair
        return (a, b) if default_role(symbols.get(a, "")) == "metal" else (b, a)

    for pair in _pairs(topology.get("lost_active_metal_contacts")):
        metal, ligand = metal_first(pair)
        if ligand not in gone:
            lines.append(f"{atom(ligand)} is no longer bound to {atom(metal)}")
    for pair in _pairs(topology.get("new_active_metal_contacts")):
        metal, ligand = metal_first(pair)
        lines.append(f"{atom(ligand)} is now bound to {atom(metal)}")
    micro = topology.get("microstate") or {}
    status, reason = micro.get("status") or topology.get("status"), micro.get("reason") or topology.get("reason")
    if status == "unlisted_microstate" and occupancy_label is not None and micro.get("occupancy_changes"):
        try:
            label = occupancy_label(micro["occupancy_changes"], micro.get("coupling"))
        except Exception:  # noqa: BLE001 - the plain sentence below still says it
            label = None
        if label:
            lines.append(f"the sites are now {label}, which is not an intermediate of this campaign")
            return lines
    if status in MICROSTATE_TEXT:
        lines.append(MICROSTATE_TEXT[status])
    elif status == "outside_site_model" and reason:
        lines.append(f"outside the site model: {reason}")
    elif reason:
        lines.append(str(reason))
    return lines


def classify(row):
    """(kind, counts_for) of one collect row (states.json).

    kind: ``accepted`` (for its own intermediate), ``reviewed`` (settled by review.json),
    ``reassigned`` (relaxed into another intermediate, which its energy counts for),
    ``no_bound_minimum`` (a molecule left the surface), ``needs_review`` (the connectivity
    changed into something that needs a decision) or ``rejected`` (other reasons: files,
    convergence, forces). ``counts_for`` is the intermediate the structure's energy counts
    for, or None.
    """
    topology = row.get("final_topology") or row.get("topology") or {}
    assigned = row.get("final_assigned_node_id", row.get("assigned_node_id"))
    decided = topology.get("status") in DECIDED
    if row.get("final_valid"):
        if assigned in (None, row["node_id"]):
            return ("reviewed" if decided else "accepted"), row["node_id"]
        return ("reviewed" if decided else "reassigned"), assigned
    if (topology.get("microstate") or {}).get("status") == "adsorbate_released":
        return "no_bound_minimum", None
    usable = bool(topology.get("valid_for_assigned_node") or topology.get("assigned_node_id"))
    return ("rejected" if usable else "needs_review"), None


def other_reasons(row):
    """The exclusion reasons of a row that the change list does not already explain."""
    reasons = row.get("final_exclusion_reasons") or []
    return [r for r in reasons if not any(c in r.lower() for c in CONNECTIVITY_REASONS)]


def tautomer_lines(topology):
    """'the shared H a0087 moved from O a0067 to O a0066' for the background shared protons that moved."""
    from .tautomers import moved_lines

    return moved_lines((topology or {}).get("tautomer"))


def _summary(kind, changes, reasons, counts_for_label):
    if kind == "reassigned":
        return f"relaxed into {counts_for_label}" if counts_for_label else "relaxed into another intermediate"
    if changes:
        return changes[0]
    return reasons[0] if reasons else kind.replace("_", " ")


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


class _Campaign:
    """Network, labels and state folders of one campaign."""

    def __init__(self, campaign, registry=None, network=None, notation=None):
        from .notation import Notation

        self.root = Path(campaign)
        self.network = network or read_json(self.root / "network.json")
        self.nodes = {n["id"]: n for n in self.network.get("nodes", [])}
        if notation is None:
            if registry is None:
                try:
                    from .plugins import campaign_registry

                    registry = campaign_registry(self.root)
                except Exception:  # noqa: BLE001 - labels fall back to the family names
                    registry = None
            notation = Notation.from_campaign(self.root, registry=registry, network=self.network)
        self.notation = notation

    def owner(self, node_id):
        return (self.nodes.get(node_id) or {}).get("computed_by") or node_id

    def label(self, node_id):
        if not node_id:
            return None
        try:
            return self.notation.label(node_id)
        except Exception:  # noqa: BLE001 - an unknown node keeps its ID
            return node_id

    def occupancy_label(self, changes, coupling=None):
        return self.notation.occupancy_label(changes, coupling)

    def state_dir(self, state_id):
        return self.root / "states" / state_id


def _item(camp, row):
    sid = row["state_id"]
    state_dir = camp.state_dir(sid)
    kind, counts_for = classify(row)
    meta = _read(state_dir / "state.json", {}) or {}
    topology = row.get("final_topology") or row.get("topology") or {}
    described = ("needs_review", "reassigned", "reviewed", "no_bound_minimum")
    changes = describe_changes(topology, meta, camp.occupancy_label) if kind in described else []
    reasons = other_reasons(row) if kind in ("needs_review", "rejected", "no_bound_minimum") else []
    counts_label = camp.label(counts_for) if counts_for and counts_for != row["node_id"] else None
    micro = topology.get("microstate") or {}
    image = micro.get("via_symmetry_image")
    if counts_label and image and image != counts_for:
        counts_label = f"{camp.label(image)}, image of {counts_label}"
    return {
        "state_id": sid,
        "node_id": row["node_id"],
        "label": camp.label(row["node_id"]),
        "kind": kind,
        "counts_for": counts_for,
        "counts_for_label": counts_label,
        "changes": changes,
        "reasons": reasons,
        "summary": _summary(kind, changes, reasons, counts_label),
        "review": _read(state_dir / REVIEW_FILE),
        "geometry": str(Path("states") / sid / STAGE_FINAL / "CONTCAR"),
        "discoverable": kind == "needs_review" and micro.get("status") == "unlisted_microstate" and not reasons,
        "released": [item["formula"] for item in topology.get("released") or []],
    }


def review_report(campaign, rows=None, registry=None):
    """Finished structures that are not accepted for their own intermediate, and the cycles they hold up.

    ``rows`` are collect's state rows; without them, ``collect`` runs first (and writes the
    analysis tables, as ``mechanet collect`` does).
    """
    if rows is None:
        from .analysis import collect

        rows = collect(campaign, registry=registry)["states"]
    camp = _Campaign(campaign, registry=registry)
    by_node = defaultdict(list)
    for row in rows:
        by_node[row["node_id"]].append(row["state_id"])
    accepted, finished_nodes = set(), set()
    providers = defaultdict(list)  # intermediate -> the structures whose energy it gets
    items, decided, tautomers = [], [], []
    finished = 0
    kinds, released = {}, {}
    for row in rows:
        kind, counts_for = classify(row)
        if row.get("final_valid") and counts_for:
            accepted.add(camp.owner(counts_for))
            providers[camp.owner(counts_for)].append(row["state_id"])
        if stage_status(camp.state_dir(row["state_id"])) != "complete":
            continue
        finished += 1
        finished_nodes.add(row["node_id"])
        kinds[row["state_id"]] = kind
        topology = row.get("final_topology") or row.get("topology") or {}
        released[row["state_id"]] = [item["formula"] for item in topology.get("released") or []]
        if kind == "accepted":
            moved = tautomer_lines(topology)
            if moved:
                tautomers.append(
                    {
                        "state_id": row["state_id"],
                        "node_id": row["node_id"],
                        "label": camp.label(row["node_id"]),
                        "moved": moved,
                    }
                )
            continue
        item = _item(camp, row)
        item["other_starts"] = [
            {"state_id": s, "run": run_text(camp.state_dir(s))} for s in by_node[row["node_id"]] if s != row["state_id"]
        ]
        (decided if kind == "reviewed" else items).append(item)
    for item in items + decided:
        item["energy_from"] = [s for s in providers.get(camp.owner(item["node_id"]), []) if s != item["state_id"]]

    def unbound(node_id):
        """The finished starts of an intermediate, when every one of them lost its adsorbate; else 0."""
        done = [s for s in by_node[node_id] if s in kinds]
        return len(done) if done and all(kinds[s] == "no_bound_minimum" for s in done) else 0

    def why(node_id):
        """Why one intermediate has no accepted structure, in a few words."""
        words = []
        n_unbound = unbound(node_id)
        if n_unbound:
            formulas = sorted({f for s in by_node[node_id] for f in released.get(s, [])})
            words.append(
                f"no bound minimum found from {n_unbound} {'start' if n_unbound == 1 else 'starts'}: "
                f"{' or '.join(formulas) or 'the adsorbate'} left the surface (the starts differ in geometry, "
                "not in their starting magnetic moments)"
            )
        for kind, text in (
            ("needs_review", "needs review"),
            ("reassigned", "relaxed into another intermediate"),
            ("no_bound_minimum", "no bound minimum"),
            ("rejected", "rejected"),
        ):
            if kind == "no_bound_minimum" and n_unbound:
                continue
            n = sum(1 for s in by_node[node_id] if kinds.get(s) == kind)
            if n:
                words.append(text if n == 1 else f"{text} ({n} structures)")
        waiting = [s for s in by_node[node_id] if s not in kinds]
        if waiting:
            words.append(f"{len(waiting)} more {'start' if len(waiting) == 1 else 'starts'} not finished")
        return "; ".join(words) or "no finished structure"

    held, unbound_cycles, available, total = [], [], 0, 0
    for cycle in camp.network.get("cycles", []):
        if not cycle.get("symmetry_representative", True):
            continue
        total += 1
        view = camp.notation.cycle(cycle)
        owners = [camp.owner(n) for n in view["node_ids"]]
        if all(o in accepted for o in owners):
            available += 1
            continue
        if not all(o in finished_nodes for o in owners):
            continue
        missing = [
            {"node_id": node_id, "label": label, "why": why(owner), "no_bound_minimum": bool(unbound(owner))}
            for node_id, owner, label in zip(view["node_ids"], owners, view["labels"])
            if owner not in accepted
        ]
        entry = {
            "cycle_id": cycle["id"],
            "title": view["title"] + (f" ({view['variant']})" if view.get("variant") else ""),
            "site_key": view["site_key"],
            "missing": missing,
        }
        (unbound_cycles if all(m["no_bound_minimum"] for m in missing) else held).append(entry)
    unbound_nodes = sorted({camp.owner(n) for n in by_node if unbound(n) and camp.owner(n) not in accepted})
    return {
        "campaign": str(camp.root),
        "finished": finished,
        "accepted": finished - len(items) - len(decided),
        "not_accepted": items,
        "decided": decided,
        "tautomers": tautomers,
        "no_bound_minimum": [
            {"node_id": n, "label": camp.label(n), "starts": unbound(n), "why": why(n)} for n in unbound_nodes
        ],
        "cycles": {
            "symmetry_unique": total,
            "with_every_intermediate_accepted": available,
            "held_up": held,
            "no_bound_minimum": unbound_cycles,
        },
    }


def review_text(report, ascii=False):
    """The terminal text of ``mechanet review``."""
    from .notation import plain

    text = plain if ascii else str
    items = report["not_accepted"]
    lines = [
        f"{report['finished']} finished {'structure' if report['finished'] == 1 else 'structures'}: "
        f"{report['accepted']} accepted for their intermediate, {len(items)} not"
        + (f", {len(report['decided'])} settled by review.json" if report["decided"] else "")
    ]
    sections = (
        ("needs_review", "Needs your review: the relaxed structure changed"),
        ("no_bound_minimum", "No bound minimum from this start: a molecule left the surface"),
        ("reassigned", "Relaxed into another intermediate: the energy counts there"),
        ("rejected", "Not accepted for other reasons"),
    )
    for kind, heading in sections:
        group = [i for i in items if i["kind"] == kind]
        if not group:
            continue
        lines += ["", heading]
        for item in group:
            head = f"  {item['state_id']}  {item['label']}"
            if kind == "reassigned":
                head += f"  ->  {item['counts_for_label']} ({item['counts_for']})"
            lines.append(head)
            lines += [f"      {line}" for line in item["changes"]]
            lines += [f"      - {reason}" for reason in item["reasons"]]
            if item["energy_from"]:
                lines.append(f"      {item['label']} still has an energy, from {', '.join(item['energy_from'])}")
            for other in item["other_starts"]:
                if other["run"] != "finished":
                    lines.append(f"      another start of {item['label']}: {other['state_id']} ({other['run']})")
    if report.get("tautomers"):
        lines += ["", "Accepted in another tautomer: a shared H between two sites the intermediate leaves native moved"]
        for item in report["tautomers"]:
            lines.append(f"  {item['state_id']}  {item['label']}")
            lines += [f"      {line}" for line in item["moved"]]
    if report["decided"]:
        lines += ["", "Settled by review.json"]
        for item in report["decided"]:
            review = item.get("review") or {}
            target = (
                f"counts for {item['counts_for_label']} ({item['counts_for']})"
                if item["counts_for"] != item["node_id"]
                else "accepted as its intermediate"
            )
            lines.append(f'  {item["state_id"]}  {item["label"]}: {target}; "{review.get("comment", "")}"')
    cycles = report["cycles"]
    lines += [
        "",
        f"Cycles ({cycles['symmetry_unique']} symmetry-unique): {cycles['with_every_intermediate_accepted']} with an "
        "accepted structure for every intermediate",
    ]
    if cycles["held_up"]:
        lines.append("Held up (every intermediate finished, some without an accepted structure):")
        for cycle in cycles["held_up"]:
            lines.append(f"  {cycle['cycle_id']}  {cycle['title']}")
            lines += [f"      {m['label']}: {m['why']}" for m in cycle["missing"]]
    if cycles.get("no_bound_minimum"):
        lines.append("With an intermediate that has no bound minimum (every finished start lost its adsorbate):")
        for cycle in cycles["no_bound_minimum"]:
            lines.append(f"  {cycle['cycle_id']}  {cycle['title']}")
            lines += [f"      {m['label']}: {m['why']}" for m in cycle["missing"]]
    if any(i["kind"] != "no_bound_minimum" for i in items):
        lines += [
            "",
            "What to do: look at each structure (states/STATE/02_final/CONTCAR), then",
            '  mechanet review --accept STATE "why"         it is still its intermediate',
            '  mechanet review --assign STATE NODE "why"    it is another intermediate of the same composition',
        ]
        if any(i.get("discoverable") for i in items):
            lines.append(
                '  mechanet review --discover STATE "why"       it is a new intermediate: add it to the campaign'
            )
        lines += [
            "  mechanet submit --states STATE --execute     run another start of the intermediate",
            "and run mechanet thermo again: it reads the decisions.",
        ]
    if any(i["kind"] == "no_bound_minimum" for i in items):
        lines += [
            "",
            "A structure without a bound minimum needs no decision: it gives its intermediate no energy. Every start "
            "of an intermediate begins from the same magnetic moments (vasp.magmom_*), so a start from other moments "
            "could still find a bound one.",
        ]
    return text("\n".join(lines))


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def state_verdict(campaign, state_id, registry=None):
    """What collect would now say about one finished structure (its 02_final), without collecting the rest."""
    from .analysis import _final_validity
    from .plugins import campaign_registry
    from .results import parse_run

    campaign = Path(campaign)
    state_dir = campaign / "states" / state_id
    config = read_json(campaign / "config.json")
    registry = registry or campaign_registry(campaign, config=config)
    meta = read_json(state_dir / "state.json")
    require_hashes = bool(config.get("analysis", {}).get("require_input_hashes", True))
    reasons, _force, topology = _final_validity(
        state_dir, meta, parse_run(state_dir / STAGE_FINAL), require_hashes, config, registry
    )
    row = {
        "state_id": state_id,
        "node_id": meta["node_id"],
        "final_valid": not reasons,
        "final_topology": topology,
        "final_assigned_node_id": topology.get("assigned_node_id"),
        "final_exclusion_reasons": reasons,
    }
    return _item(_Campaign(campaign, registry=registry), row)


def _composition(delta):
    """{'O': 1, 'H': -1} -> 'H-1 O+1'; the native composition when nothing changes."""
    parts = [f"{el}{int(n):+d}" for el, n in sorted((delta or {}).items()) if n]
    return " ".join(parts) or "the native composition"


def decide(campaign, state_id, accept=None, assign=None, clear=False, registry=None):
    """Write (or with ``clear`` remove) ``states/<state_id>/review.json``, then judge the state again.

    ``accept`` is the comment for accepting the relaxed structure as its own intermediate;
    ``assign`` is (node_id, comment) for counting it for another intermediate of the same
    composition. Returns the record written and the new verdict.
    """
    campaign = Path(campaign)
    state_dir = campaign / "states" / state_id
    if not (state_dir / "state.json").is_file():
        raise ValueError(f"No state {state_id!r} in {campaign / 'states'}")
    path = state_dir / REVIEW_FILE
    if clear:
        existed = path.is_file()
        path.unlink(missing_ok=True)
        return {"state_id": state_id, "cleared": existed, "path": str(path)}
    if stage_status(state_dir) != "complete":
        raise ValueError(f"{state_id}: 02_final has not finished ({run_text(state_dir)}); review it once it has")
    comment = str(accept if assign is None else assign[1]).strip()
    if not comment:
        raise ValueError("Give a reason: the comment is kept in review.json, next to the decision")
    meta = read_json(state_dir / "state.json")
    if assign is not None:
        node_id = str(assign[0])
        nodes = {n["id"]: n for n in read_json(campaign / "network.json").get("nodes", [])}
        node = nodes.get(node_id)
        if node is None:
            close = sorted(n for n in nodes if n.split("_")[:2] == node_id.split("_")[:2])[:5]
            raise ValueError(
                f"Unknown intermediate {node_id!r}"
                + (f"; similar: {', '.join(close)}" if close else "")
                + " (`mechanet show-cycle ID` lists the intermediates of a cycle under its steps)"
            )
        if meta.get("delta_atoms") is not None and node.get("delta_atoms") != meta.get("delta_atoms"):
            raise ValueError(
                f"{node_id} has another composition than {state_id} ({_composition(node.get('delta_atoms'))} vs "
                f"{_composition(meta.get('delta_atoms'))}, relative to the native surface); a structure can only "
                "count for an intermediate with the same atoms"
            )
        record = {"assign_to_node": node_id, "comment": comment}
    else:
        record = {"accept_relaxed_topology": True, "comment": comment}
    record.update(written_by="mechanet review", written_utc=utc_now())
    write_json(path, record)
    return {"state_id": state_id, "review": record, "path": str(path), "verdict": state_verdict(campaign, state_id)}


def decision_text(result):
    """One or two lines for ``mechanet review --accept/--assign/--clear``."""
    sid = result["state_id"]
    if "cleared" in result:
        return (
            f"removed {result['path']}; the next collect judges {sid} again"
            if result["cleared"]
            else f"{sid} had no review.json; nothing to remove"
        )
    verdict = result["verdict"]
    if verdict["kind"] in ("accepted", "reviewed"):
        target = (
            f"{verdict['label']} ({verdict['node_id']})"
            if verdict["counts_for"] == verdict["node_id"]
            else f"{verdict['counts_for_label']} ({verdict['counts_for']})"
        )
        return f"wrote {result['path']}: {sid} now counts for {target}\nrun mechanet thermo again to use it"
    lines = [f"wrote {result['path']}, but {sid} is still not accepted:"]
    lines += [f"  {line}" for line in verdict["changes"][-1:]]
    lines += [f"  - {reason}" for reason in verdict["reasons"]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The last collect, for status
# ---------------------------------------------------------------------------


def last_collect(campaign, network=None, notation=None):
    """The verdicts of the last ``collect`` (analysis/states.json) for the finished structures.

    A structure whose run or review.json changed after that collect counts as ``unknown``.
    Returns None when the campaign has not been collected. ``accepted_nodes`` are the
    intermediates (computed ones) with an accepted structure.
    """
    campaign = Path(campaign)
    path = campaign / "analysis" / "states.json"
    try:
        when = path.stat().st_mtime
        data = read_json(path)
    except (OSError, ValueError):
        return None
    rows = data.get("states", []) if isinstance(data, dict) else data
    camp = _Campaign(campaign, network=network, notation=notation)
    verdicts, accepted = {}, set()
    for row in rows:
        if not isinstance(row, dict) or "state_id" not in row:
            continue
        state_dir = camp.state_dir(row["state_id"])
        attempt = state_dir / STAGE_FINAL / "attempt.json"
        review = state_dir / REVIEW_FILE
        try:
            stale = attempt.stat().st_mtime > when or (review.is_file() and review.stat().st_mtime > when)
        except OSError:
            continue
        kind, counts_for = classify(row)
        if row.get("final_valid") and counts_for and not stale:
            accepted.add(camp.owner(counts_for))
        if stage_status(state_dir) != "complete":
            continue
        if stale:
            verdicts[row["state_id"]] = {"kind": "unknown", "counts_for": None, "summary": "changed since the collect"}
        else:
            item = _item(camp, row)
            verdicts[row["state_id"]] = {k: item[k] for k in ("kind", "counts_for", "counts_for_label", "summary")}
    return {
        "collected_utc": dt.datetime.fromtimestamp(when, dt.timezone.utc).isoformat(),
        "verdicts": verdicts,
        "accepted_nodes": accepted,
    }
