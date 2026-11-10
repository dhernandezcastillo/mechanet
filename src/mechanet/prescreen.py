"""Relax the generated starts with a fast calculator before any DFT, and judge them like DFT results.

``mechanet prescreen`` relaxes the starts (states) of a campaign with an in-process calculator,
usually a machine-learned potential such as MACE or UMA (see :mod:`mechanet.calculators`),
moving the atoms the DFT relaxation would move. Each relaxed structure then goes through the
topology check of ``collect``: did the start keep its intermediate, relax into another
intermediate of the network, give its adsorbate off, or change in a way that would need review?

The prescreen only removes redundancy. For each intermediate, the starts that kept it are
ranked by the calculator's energy, starts that relaxed into the same structure (or a symmetry
image of it) are merged, and those within a window of the lowest (at most N) are suggested for
DFT. Every start the calculator changed stays in the list: one that relaxed into another
intermediate, gave its adsorbate off or rearranged its sites is where DFT, through
``mechanet review --discover``, finds new intermediates, and the potential can be wrong either
way. When the new arrangement of the sites reads as an intermediate the network does not list
(what ``review --discover`` accepts), the summary names it as a possible new intermediate.
Starts the calculator failed on are kept too.

The idea of relaxing many placements with an ML potential and computing the best with DFT
follows AdsorbML (Lan et al., npj Comput. Mater. 2023, doi:10.1038/s41524-023-01121-5), which
discards relaxations where the adsorbate desorbs, dissociates or changes the surface; in a
reaction network those are kept, because they may be new intermediates.

The energies only compare starts of one intermediate, which have the same atoms. They are not
used for reaction energies: adsorption energies from universal ML potentials carry errors of
the order of 0.1 eV or more, the size of the differences that decide which mechanism wins.

Files, in ``<campaign>/prescreen``: ``STATE/CONTCAR`` and ``STATE/result.json`` for every start,
``prescreen.json`` and ``prescreen.csv`` for the run, and ``suggested_states.txt`` for
``mechanet submit --states @FILE``. A start whose POSCAR, calculator and settings are unchanged
is not relaxed again (``--redo`` forces it).
"""

from __future__ import annotations

import contextlib
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .calculators import (
    DEFAULT_FMAX_EVA,
    DEFAULT_STEPS,
    OPTIMIZERS,
    fingerprint,
    load_calculator,
    parse_options,
)
from .fileio import dump_json, file_hash, read_json, utc_now, write_json
from .structure import Structure, read_poscar, write_poscar

PRESCREEN_DIR = "prescreen"
DEFAULT_WINDOW_EV = 0.20  # wider than `select` (0.10 eV): ML energies are less certain than DFT ones
DEFAULT_MAX_PER_NODE = 3
LIST_LIMIT = 30  # lines per list in the readable summary
OUTCOMES = {
    "kept": "kept its intermediate",
    "became_other": "relaxed into another intermediate",
    "released": "gave its adsorbate off",
    "changed": "changed bonds (would need review)",
    "unchecked": "could not be checked",
    "failed": "calculator failed",
}
CSV_FIELDS = (
    "state_id",
    "node_id",
    "outcome",
    "topology_status",
    "assigned_node_id",
    "new_intermediate",
    "energy_eV",
    "relative_eV",
    "max_force_eVA",
    "converged",
    "steps",
    "suggested",
    "why_not",
    "same_as",
    "reused",
    "error",
)


def _outcome(topology):
    if topology.get("valid_for_assigned_node"):
        return "kept"
    if topology.get("status") == "reassigned_to_existing_node":
        return "became_other"
    if (topology.get("microstate") or {}).get("status") == "adsorbate_released":
        return "released"
    if topology.get("status") == "changed_connectivity_needs_review":
        return "changed"
    return "unchecked"


def _topology_summary(topology):
    micro = topology.get("microstate") or {}
    return {
        "status": topology.get("status"),
        "assigned_node_id": topology.get("assigned_node_id"),
        "microstate_status": micro.get("status"),
        "reason": micro.get("reason") or topology.get("reason"),
        "lost_bonds": topology.get("lost_bonds") or [],
        "new_bonds": topology.get("new_bonds") or [],
        "lost_active_metal_contacts": topology.get("lost_active_metal_contacts") or [],
        "new_active_metal_contacts": topology.get("new_active_metal_contacts") or [],
        "released": topology.get("released") or [],
    }


def new_intermediate(topology, occupancy_label=None):
    """The label of the arrangement a relaxed structure reached when it is an intermediate the
    network does not list (what ``mechanet review --discover`` accepts), else None."""
    micro = topology.get("microstate") or {}
    if micro.get("status") == "surface_atom_released":
        # the adsorbate left with an atom of the surface: what it leaves, when the campaign lacks it
        from .review import surface_left_label

        left = micro.get("surface_left") or {}
        label = surface_left_label(micro, occupancy_label)
        if label is None or left.get("node_id"):
            return None
        formulas = " + ".join(item["formula"] for item in topology.get("released") or [])
        return f"{label} (after {formulas or 'a molecule'} left with a surface atom)"
    changes = micro.get("occupancy_changes")
    if micro.get("status") != "unlisted_microstate" or not changes:
        return None
    if occupancy_label is not None:
        try:
            return occupancy_label(changes, micro.get("coupling"))
        except Exception:  # noqa: BLE001 - the sites themselves still name it
            pass
    return ", ".join(f"{site}: {status}" for site, status in changes.items())


def _start(state_dir, meta):
    """The state's POSCAR with the selective dynamics of its DFT relaxation."""
    structure = read_poscar(Path(state_dir) / "POSCAR")
    mobile = meta.get("relax_mobile")
    if mobile is not None and np.shape(mobile) == (len(structure), 3):
        structure.mobile = np.asarray(mobile, dtype=bool)
    return structure


def _relax_state(
    campaign, state_dir, meta, calculator, description, key, settings, config, registry, redo, occupancy_label=None
):
    from .analysis import topology_check
    from .review import describe_changes

    sid = state_dir.name
    out = Path(campaign) / PRESCREEN_DIR / sid
    record_path, contcar = out / "result.json", out / "CONTCAR"
    input_hash = file_hash(state_dir / "POSCAR")
    if not redo and record_path.is_file() and contcar.is_file():
        try:
            old = read_json(record_path)
        except (OSError, ValueError):
            old = {}
        if old.get("fingerprint") == key and old.get("input_sha256") == input_hash and old.get("outcome") != "failed":
            return dict(old, reused=True)
    record = {
        "state_id": sid,
        "node_id": meta.get("node_id"),
        "fingerprint": key,
        "calculator": description,
        "settings": settings,
        "input_sha256": input_hash,
        "created": utc_now(),
    }
    try:
        structure = _start(state_dir, meta)
        relaxation = calculator.relax(structure, fmax=settings["fmax_eVA"], steps=settings["steps"])
        positions = np.asarray(relaxation.positions, dtype=float)
        energy = float(relaxation.energy_eV)
        if (
            positions.shape != structure.positions.shape
            or not np.isfinite(positions).all()
            or not math.isfinite(energy)
        ):
            raise ValueError("the calculator returned no finite energy and positions for every atom")
    except Exception as exc:  # noqa: BLE001 - one start the calculator cannot handle must not stop the others
        record.update(outcome="failed", error=f"{type(exc).__name__}: {exc}")
        out.mkdir(parents=True, exist_ok=True)
        contcar.unlink(missing_ok=True)
        write_json(record_path, record)
        return dict(record, reused=False)
    relaxed = Structure(structure.symbols, positions, structure.cell, structure.mobile, structure.ids)
    write_poscar(
        contcar, relaxed, comment=f"mechanet prescreen ({description.get('calculator')}); atom IDs in state.json"
    )
    try:
        topology = topology_check(state_dir, meta, config, geometry=contcar, registry=registry, use_review=False)
        outcome = _outcome(topology)
        changes = [] if outcome == "kept" else describe_changes(topology, meta, occupancy_label)
        found = new_intermediate(topology, occupancy_label)
    except Exception as exc:  # noqa: BLE001 - a structure that cannot be checked gets no verdict, the run goes on
        topology = {"status": "classification_failed", "reason": f"{type(exc).__name__}: {exc}"}
        outcome, changes, found = "unchecked", [], None
    record.update(
        outcome=outcome,
        energy_eV=round(energy, 6),
        max_force_eVA=round(float(relaxation.max_force_eVA), 4),
        converged=bool(relaxation.converged),
        steps=int(relaxation.steps),
        topology=_topology_summary(topology),
        changes=changes,
        new_intermediate=found,
    )
    write_json(record_path, record)
    return dict(record, reused=False)


def _images(campaign, config):
    """Symmetry operations of the reference surface (:func:`mechanet.symmetry.surface_images`),
    atoms told apart by their relaxation and frequency regions and their MAGMOM; identity only
    when the reference is unknown."""
    from .symmetry import DEFAULT_SYMMETRY_TOLERANCE_A, surface_images

    try:
        network = read_json(campaign / "network.json")
        reference = network.get("reference", "ref")
        node = next(n for n in network["nodes"] if n["id"] == reference)
        state_dir = campaign / "states" / node["state_ids"][0]
        meta = read_json(state_dir / "state.json")
        structure = _start(state_dir, meta)
        normal = read_json(campaign / "surface_analysis.json")["surface_normal"]
    except (OSError, ValueError, KeyError, IndexError, StopIteration):
        return [(np.eye(3), np.zeros(3))]
    ids = meta.get("atom_ids") or structure.ids
    if len(ids) != len(structure):
        ids = structure.ids
    magmom = (config.get("vasp") or {}).get("magmom_by_id") or {}
    frequency = np.asarray(meta.get("frequency_mobile") or [], dtype=bool)
    if frequency.shape != (len(structure), 3):
        frequency = np.zeros((len(structure), 3), dtype=bool)
    labels = [(bool(structure.mobile[i].any()), bool(frequency[i].any()), magmom.get(aid)) for i, aid in enumerate(ids)]
    tolerance = float((config.get("geometry") or {}).get("symmetry_tolerance_A", DEFAULT_SYMMETRY_TOLERANCE_A))
    return surface_images(structure, normal, labels, symprec=tolerance)


def _same_structure(campaign, first, second, images):
    """How two relaxed starts differ when they are the same minimum (directly or by a symmetry
    operation of the slab), else None."""
    from .analysis import DUPLICATE_ENERGY_EV, DUPLICATE_MAX_SHIFT_A, structure_difference

    if abs(first["energy_eV"] - second["energy_eV"]) >= DUPLICATE_ENERGY_EV:
        return None
    folder = Path(campaign) / PRESCREEN_DIR
    try:
        a = read_poscar(folder / first["state_id"] / "CONTCAR")
        b = read_poscar(folder / second["state_id"] / "CONTCAR")
    except (OSError, ValueError):
        return None
    for k, (rotation, shift) in enumerate(images):
        moved = b if k == 0 else Structure(b.symbols, b.positions @ rotation.T + shift, b.cell, b.mobile, b.ids)
        diff = structure_difference(a, moved)
        if diff is not None and diff["max_shift_A"] < DUPLICATE_MAX_SHIFT_A:
            return dict(diff, by_symmetry=k > 0)
    return None


def _suggest(campaign, rows, window_eV, max_per_node, images):
    """Per intermediate: which starts to compute with DFT, and why each other start is left out.

    Only redundancy is removed: copies, and starts above the window or beyond the cap among the
    starts that kept their intermediate. A start the calculator changed (into another
    intermediate, without its adsorbate, into an unlisted arrangement) or gave no verdict on is
    always suggested: DFT decides, and a change may be a new intermediate.
    """
    by_node = defaultdict(list)
    for row in rows:
        by_node[row["node_id"]].append(row)
    nodes = []
    for node_id, group in sorted(by_node.items(), key=lambda item: str(item[0])):
        kept = sorted((r for r in group if r["outcome"] == "kept"), key=lambda r: (r["energy_eV"], r["state_id"]))
        record = {
            "node_id": node_id,
            "starts": len(group),
            "outcomes": dict(Counter(r["outcome"] for r in group)),
        }
        for row in group:
            if row["outcome"] != "kept":
                row["suggested"] = True
        if not kept:
            if all(r["outcome"] in ("failed", "unchecked") for r in group):
                record["note"] = "the calculator gave no verdict on any start"
            else:
                record["note"] = "no start kept this intermediate with the calculator"
        else:
            lowest = kept[0]
            unique = []
            for row in kept:
                row["relative_eV"] = round(row["energy_eV"] - lowest["energy_eV"], 4)
                twin, match = None, None
                for other in unique:
                    match = _same_structure(campaign, other, row, images)
                    if match is not None:
                        twin = other
                        break
                if twin is not None:
                    how = "a symmetry image of" if match["by_symmetry"] else "the same structure as"
                    row.update(suggested=False, same_as=twin["state_id"], why_not=f"{how} {twin['state_id']}")
                    continue
                unique.append(row)
                if row["relative_eV"] > window_eV:
                    row.update(suggested=False, why_not=f"{row['relative_eV']:.2f} eV above {lowest['state_id']}")
                elif sum(1 for u in unique[:-1] if u.get("suggested")) >= max_per_node:
                    row.update(suggested=False, why_not=f"beyond the {max_per_node} lowest starts")
                else:
                    row["suggested"] = True
            record["lowest_state_id"] = lowest["state_id"]
            unconverged = [r["state_id"] for r in kept if not r.get("converged")]
            if unconverged:
                record["unconverged"] = unconverged
        record["suggested"] = sorted(r["state_id"] for r in group if r.get("suggested"))
        nodes.append(record)
    return nodes


def _labels(campaign, registry):
    """(label of a node, label of a site occupancy) in the campaign's notation; IDs when it fails."""
    try:
        from .notation import Notation

        notation = Notation.from_campaign(campaign, registry=registry)
    except Exception:  # noqa: BLE001 - IDs stand in for labels
        return (lambda node_id: node_id), None

    def label(node_id):
        try:
            return notation.label(node_id) if node_id else node_id
        except Exception:  # noqa: BLE001 - an unknown node keeps its ID
            return node_id

    return label, notation.occupancy_label


def prescreen(
    campaign,
    calculator,
    states=None,
    fmax_eVA=DEFAULT_FMAX_EVA,
    steps=DEFAULT_STEPS,
    window_eV=DEFAULT_WINDOW_EV,
    max_per_node=DEFAULT_MAX_PER_NODE,
    redo=False,
    registry=None,
    progress=None,
):
    """Relax the starts with ``calculator`` (a :class:`mechanet.calculators.Calculator`), check them
    and suggest which to compute with DFT; see the module docstring. ``states``: None for all."""
    from .plugins import campaign_registry

    campaign = Path(campaign)
    if not fmax_eVA > 0 or steps < 0 or window_eV < 0 or max_per_node < 1:
        raise ValueError("Need fmax > 0, steps >= 0, window >= 0 and at least one start per intermediate")
    registry = registry or campaign_registry(campaign)
    config = read_json(campaign / "config.json")
    available = {p.parent.name: p.parent for p in (campaign / "states").glob("*/state.json")}
    chosen = sorted(available) if states is None else list(dict.fromkeys(states))
    unknown = [s for s in chosen if s not in available]
    if unknown:
        raise ValueError(f"Unknown states: {', '.join(unknown[:5])}{' ...' if len(unknown) > 5 else ''}")
    if not chosen:
        raise ValueError(f"{campaign} has no states; run `mechanet generate` first")
    description = calculator.describe()
    settings = {"fmax_eVA": float(fmax_eVA), "steps": int(steps)}
    key = fingerprint(description, settings)
    label, occupancy_label = _labels(campaign, registry)
    rows = []
    for k, sid in enumerate(chosen, 1):
        meta = read_json(available[sid] / "state.json")
        row = _relax_state(
            campaign,
            available[sid],
            meta,
            calculator,
            description,
            key,
            settings,
            config,
            registry,
            redo,
            occupancy_label,
        )
        rows.append(row)
        if progress is not None:
            progress(k, len(chosen), row)
    images = _images(campaign, config)
    nodes = _suggest(campaign, rows, window_eV, max_per_node, images)
    suggested = sorted(r["state_id"] for r in rows if r.get("suggested"))
    folder = campaign / PRESCREEN_DIR
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "suggested_states.txt").write_text("".join(f"{sid}\n" for sid in suggested))
    states_out = [
        {
            "state_id": r["state_id"],
            "node_id": r.get("node_id"),
            "outcome": r["outcome"],
            "topology_status": (r.get("topology") or {}).get("status"),
            "assigned_node_id": (r.get("topology") or {}).get("assigned_node_id"),
            "assigned_label": label((r.get("topology") or {}).get("assigned_node_id")),
            "new_intermediate": r.get("new_intermediate"),
            "energy_eV": r.get("energy_eV"),
            "relative_eV": r.get("relative_eV"),
            "max_force_eVA": r.get("max_force_eVA"),
            "converged": r.get("converged"),
            "steps": r.get("steps"),
            "suggested": bool(r.get("suggested")),
            "why_not": r.get("why_not"),
            "same_as": r.get("same_as"),
            "changes": r.get("changes") or [],
            "error": r.get("error"),
            "reused": bool(r.get("reused")),
        }
        for r in rows
    ]
    with (folder / "prescreen.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(states_out)
    warnings = []
    unconverged = [r["state_id"] for r in rows if r.get("outcome") not in ("failed",) and r.get("converged") is False]
    if unconverged:
        warnings.append(
            f"{len(unconverged)} start(s) did not reach {fmax_eVA:g} eV/A within {steps} steps; their energies "
            "and outcomes are those of unfinished relaxations (raise --steps)"
        )
    failed = [r for r in rows if r["outcome"] == "failed"]
    if failed:
        warnings.append(
            f"the calculator failed on {len(failed)} start(s), e.g. {failed[0]['state_id']}: {failed[0]['error']}"
        )
    if str(description.get("class", "")).endswith(".EMT"):
        warnings.append("EMT is a toy potential: use it to test the workflow, not to choose structures")
    result = {
        "campaign": str(campaign),
        "created": utc_now(),
        "calculator": description,
        "fingerprint": key,
        "settings": dict(
            settings, window_eV=float(window_eV), max_per_node=int(max_per_node), symmetry_operations=len(images)
        ),
        "counts": {name: sum(1 for r in rows if r["outcome"] == name) for name in OUTCOMES},
        "reused": sum(1 for r in rows if r.get("reused")),
        "starts": len(rows),
        "suggested_state_ids": suggested,
        "nodes": [dict(n, label=label(n["node_id"])) for n in nodes],
        "states": states_out,
        "warnings": warnings,
        "possible_new_intermediates": [
            {"state_id": s["state_id"], "node_id": s["node_id"], "label": s["new_intermediate"]}
            for s in states_out
            if s["new_intermediate"]
        ],
        "notes": [
            "Energies compare starts of one intermediate only (same atoms); they are not reaction energies.",
            "Only redundant starts are left out: among the starts that kept their intermediate, those that "
            "relaxed into the same structure as a lower one (or a symmetry image of it), those more than "
            f"{window_eV:g} eV above the lowest, and those beyond the {max_per_node} lowest. Every start the "
            "calculator changed or gave no verdict on is suggested: DFT decides, and a change may be a new "
            'intermediate (after DFT, `mechanet review --discover STATE "why"` adds one).',
        ],
        "files": {
            "summary": str(folder / "prescreen.json"),
            "table": str(folder / "prescreen.csv"),
            "suggested": str(folder / "suggested_states.txt"),
        },
    }
    dump_json(folder / "prescreen.json", result)
    return result


def prescreen_text(result):
    """Readable summary: outcomes, the suggestion, why starts were left out."""
    calc = result["calculator"]
    settings = result["settings"]
    labels = {n["node_id"]: n.get("label") or n["node_id"] for n in result["nodes"]}
    package = " ".join(str(x) for x in (calc.get("package"), calc.get("package_version")) if x)
    detail = ", ".join(str(x) for x in (calc.get("class"), package) if x)
    lines = [
        f"prescreen of {result['starts']} starts with {calc.get('calculator') or 'the calculator'}"
        + (f" ({detail})" if detail else "")
        + f": {calc.get('optimizer') or 'relaxed'} to {settings['fmax_eVA']:g} eV/A, at most {settings['steps']} steps"
    ]
    width = max(len(text) for text in OUTCOMES.values())
    lines += [f"  {OUTCOMES[name].ljust(width)}  {count}" for name, count in result["counts"].items() if count]
    if result["reused"]:
        lines.append(
            f"  ({result['reused']} reused from an earlier run with the same calculator; --redo relaxes again)"
        )
    states = result["states"]
    suggested = result["suggested_state_ids"]
    changed = [s for s in states if s["outcome"] in ("became_other", "released", "changed")]
    unknown = [s for s in states if s["outcome"] in ("failed", "unchecked")]
    parts = [f"{sum(1 for s in states if s['suggested'] and s['outcome'] == 'kept')} kept their intermediate"]
    parts += [f"{len(changed)} changed by the calculator (DFT decides)"] if changed else []
    parts += [f"{len(unknown)} without a verdict"] if unknown else []
    lines.append(f"suggested for DFT: {len(suggested)} of {result['starts']} starts: {', '.join(parts)}")
    if suggested:
        lines.append(f"  to compute only these: mechanet submit --states @{result['files']['suggested']}")

    def where(state):
        return f"{state['state_id']} ({labels.get(state['node_id'], state['node_id'])})"

    def listed(title, items, line):
        if items:
            lines.append(title)
            lines.extend(f"  {line(state)}" for state in items[:LIST_LIMIT])
            if len(items) > LIST_LIMIT:
                lines.append(f"  ... {len(items) - LIST_LIMIT} more ({result['files']['table']} lists every start)")

    new = [s for s in changed if s.get("new_intermediate")]
    listed(
        'possible new intermediates (after DFT, mechanet review --discover STATE "why" adds one):',
        new,
        lambda s: f"{where(s)}: the sites read as {s['new_intermediate']}",
    )

    def change(state):
        if state["outcome"] == "became_other" and state["assigned_node_id"]:
            what = f"relaxed into {state.get('assigned_label') or state['assigned_node_id']}"
        else:
            what = OUTCOMES[state["outcome"]]
        return f"{where(state)}: {what}" + (f": {'; '.join(state['changes'])}" if state["changes"] else "")

    listed("changed by the calculator, kept for DFT:", [s for s in changed if not s.get("new_intermediate")], change)
    listed("left out as redundant:", [s for s in states if not s["suggested"]], lambda s: f"{where(s)}: {s['why_not']}")
    held = [n for n in result["nodes"] if n.get("note")]
    if held:
        lines.append("intermediates no start kept with the calculator (all their starts are suggested):")
        for node in held[:LIST_LIMIT]:
            counts = ", ".join(f"{n} {OUTCOMES[name]}" for name, n in node["outcomes"].items())
            lines.append(f"  {node.get('label') or node['node_id']}: {counts}")
        if len(held) > LIST_LIMIT:
            lines.append(f"  ... {len(held) - LIST_LIMIT} more (--json shows all)")
    lines += [f"warning: {w}" for w in result["warnings"]]
    lines.append(f"details: {result['files']['summary']}")
    return "\n".join(lines)


def register_command(subparsers, registry):
    p = subparsers.add_parser(
        "prescreen",
        help="Relax the starts with an ML potential before DFT and suggest which to compute",
        description=__doc__.split("\n\n")[0],
    )
    p.add_argument("campaign", nargs="?")
    p.add_argument(
        "--calculator",
        required=True,
        help="emt, mace-mp, a name a plugin adds (`mechanet plugins` lists them) or module:callable",
    )
    p.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Option for the calculator, repeatable (e.g. --option model=medium --option device=cuda)",
    )
    p.add_argument(
        "--plugin",
        action="append",
        default=[],
        metavar="FILE",
        help="A plugin file that adds the calculator (registry.add_calculator), repeatable",
    )
    p.add_argument("--states", help="Comma-separated state IDs (or @file); default: every state")
    p.add_argument("--families", help="Every state used by these mechanism families")
    p.add_argument("--fmax", type=float, default=DEFAULT_FMAX_EVA, help="Force criterion in eV/A (default 0.05)")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Most optimizer steps per start (default 300)")
    p.add_argument("--optimizer", choices=OPTIMIZERS, default="BFGS", help="ASE optimizer (default BFGS)")
    p.add_argument(
        "--window",
        type=float,
        default=DEFAULT_WINDOW_EV,
        help="Suggest starts within this energy of their intermediate's lowest, eV (default 0.20)",
    )
    p.add_argument(
        "--max-per-node",
        type=int,
        default=DEFAULT_MAX_PER_NODE,
        help="Suggest at most this many starts per intermediate (default 3)",
    )
    p.add_argument("--redo", action="store_true", help="Relax again starts that have a result for this calculator")
    p.set_defaults(handler=_cli)


def _progress(k, total, row):
    energy = f" {row['energy_eV']:.4f} eV" if row.get("energy_eV") is not None else ""
    steps = f", {row['steps']} steps" if row.get("steps") is not None else ""
    note = " (reused)" if row.get("reused") else ""
    print(f"[{k}/{total}] {row['state_id']}: {OUTCOMES[row['outcome']]}{energy}{steps}{note}", file=sys.stderr)


def _cli(args, registry):
    from .campaign import find_campaign_dir, selection

    campaign = find_campaign_dir(args.campaign)
    states = selection(campaign, args.states, args.families)
    options = parse_options(args.option)
    if args.plugin:
        from .plugins import campaign_plugin_files, default_registry

        registry = default_registry([*campaign_plugin_files(campaign), *args.plugin])
    # Third-party potentials print banners on stdout; keep stdout for the result (--json).
    with contextlib.redirect_stdout(sys.stderr):
        calculator = load_calculator(args.calculator, options, registry, args.optimizer)
        result = prescreen(
            campaign,
            calculator,
            states,
            fmax_eVA=args.fmax,
            steps=args.steps,
            window_eV=args.window,
            max_per_node=args.max_per_node,
            redo=args.redo,
            registry=registry,
            progress=_progress,
        )
    return result, prescreen_text(result)
