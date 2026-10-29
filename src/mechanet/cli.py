"""mechanet command line.

A study runs as::

    mechanet init slab.vasp            # inspect the slab, answer a few questions -> project.yaml
    mechanet generate                  # intermediates, steps, cycles and starting structures
    mechanet plan                      # what it will cost
    mechanet submit --pilot            # preview; --execute submits to SLURM
    mechanet submit --execute          # everything (repeat to continue interrupted states)
    mechanet collect && mechanet select && mechanet submit --selected --through freq --execute
    mechanet thermo                    # free energies, limiting potentials, Gmax(U)
    mechanet pathways --rank           # every cycle of the network, with the project's references
    mechanet report

Generation, preparation and analysis never submit a job; ``submit`` only previews unless
``--execute`` is given. Every command prints a readable summary, or JSON with ``--json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from . import __version__

THROUGH_CHOICES = ["loose", "final", "refine", "static", "freq"]
THROUGH_HELP = "Last stage to run (default: final); 'refine' exists only with frequency.central = refine"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _scalar(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "-"
    return str(value)


def render(value, indent=0, max_items=15):
    """Readable text for nested dicts and lists (long lists are shortened)."""
    pad = "  " * indent
    lines = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, dict) and item:
                lines.append(f"{pad}{key}:")
                lines.append(render(item, indent + 1, max_items))
            elif isinstance(item, list) and item and any(isinstance(x, (dict, list)) for x in item):
                lines.append(f"{pad}{key}: ({len(item)})")
                lines.append(render(item, indent + 1, max_items))
            elif isinstance(item, list):
                shown = ", ".join(_scalar(x) for x in item[:max_items])
                more = f" ... (+{len(item) - max_items})" if len(item) > max_items else ""
                lines.append(f"{pad}{key}: [{shown}{more}]")
            else:
                lines.append(f"{pad}{key}: {_scalar(item)}")
    elif isinstance(value, list):
        for item in value[:max_items]:
            if isinstance(item, dict):
                text = ", ".join(
                    f"{k}={_scalar(v) if not isinstance(v, (dict, list)) else json.dumps(v)}" for k, v in item.items()
                )
                lines.append(f"{pad}- {text}")
            else:
                lines.append(f"{pad}- {_scalar(item)}")
        if len(value) > max_items:
            lines.append(f"{pad}... (+{len(value) - max_items} more; --json shows all)")
    else:
        lines.append(pad + _scalar(value))
    return "\n".join(line for line in lines if line != "")


def _plural(n, word, plural=None):
    """'1 state', '2 states' (``plural`` for irregular words)."""
    return word if n == 1 else (plural or word + "s")


def _table(rows, columns):
    if not rows:
        return "(none)"
    widths = [max(len(c), *(len(_scalar(r.get(c))) for r in rows)) for c in columns]
    out = ["  ".join(c.ljust(w) for c, w in zip(columns, widths)), "  ".join("-" * w for w in widths)]
    out += ["  ".join(_scalar(r.get(c)).ljust(w) for c, w in zip(columns, widths)) for r in rows]
    return "\n".join(line.rstrip() for line in out)


def _vector_text(vector):
    """'H2O -> H+ + e-' style text of a reservoir vector."""
    from .reactions import electrons

    left = [f"{-v if -v != 1 else ''}{k}" for k, v in sorted(vector.items()) if v < 0]
    right = [f"{v if v != 1 else ''}{k}" for k, v in sorted(vector.items()) if v > 0]
    n = electrons(vector)
    if n > 0:
        right.append(f"{n if n != 1 else ''}e-")
    elif n < 0:
        left.append(f"{-n if n != -1 else ''}e-")
    if not left and not right:
        return "no exchange with the electrolyte"
    if not left:
        return "releases " + " + ".join(right)
    if not right:
        return "takes up " + " + ".join(left)
    return f"{' + '.join(left)} -> {' + '.join(right)}"


# ---------------------------------------------------------------------------
# Helpers shared by commands
# ---------------------------------------------------------------------------


def _campaign(args):
    from .campaign import find_campaign_dir

    return find_campaign_dir(getattr(args, "campaign", None))


# Commands that build their registry from a project file (or need none).
_PROJECT_COMMANDS = ("init", "inspect", "generate", "examples", "plugins")


def _command_registry(args, registry):
    """The registry with the plugin files of the campaign a command works on.

    A campaign built with plugins needs their reactions, reservoir species and families in
    every later command (``show-cycle`` of an HER campaign needs the H2 species, say).
    Commands that do not name or find a campaign keep the given registry.
    """
    from .campaign import find_campaign_dir
    from .plugins import campaign_plugin_files, default_registry

    if args.command in _PROJECT_COMMANDS or not hasattr(args, "campaign"):
        return registry
    try:
        campaign = find_campaign_dir(args.campaign)
    except (FileNotFoundError, OSError):
        return registry
    plugins = campaign_plugin_files(campaign)
    if not plugins:
        return registry
    try:
        return default_registry(plugins)
    except Exception as exc:  # noqa: BLE001 - most commands (submit, list, cleanup, ...) need no plugin
        print(
            f"warning: the campaign's plugin files could not be loaded ({exc}); continuing without them, so "
            "commands that need their reactions or species will fail",
            file=sys.stderr,
        )
        return registry


def default_references(campaign):
    """thermodynamics.references of a campaign (set from project.yaml)."""
    from .campaign import find_campaign_dir
    from .fileio import read_json

    root = find_campaign_dir(campaign)
    path = read_json(root / "config.json").get("thermodynamics", {}).get("references")
    if not path:
        raise ValueError(
            "No references file: pass --references FILE or set thermodynamics.references in "
            "project.yaml (`mechanet references` writes a template)"
        )
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"References file {path} not found (`mechanet references --out {path}` writes a template)"
        )
    return path


def _selection_options(parser, pilot=True, selected=True):
    parser.add_argument("--states", help="Comma-separated state IDs (or @file)")
    parser.add_argument("--families", help="Every state used by these mechanism families")
    if pilot:
        parser.add_argument(
            "--pilot",
            action="store_true",
            help="The reference, one start per adsorbate type and the most crowded start",
        )
    if selected:
        parser.add_argument(
            "--selected", action="store_true", help="Only states listed in thermo_selection.json (written by `select`)"
        )


def _selection(args, campaign):
    from .campaign import selection

    return selection(
        campaign,
        getattr(args, "states", None),
        getattr(args, "families", None),
        getattr(args, "selected", False),
        getattr(args, "pilot", False),
    )


def _frequency_options(parser):
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--cutoff", type=float, default=None, help="Positive-frequency floor in cm^-1; 0 uses raw harmonic values"
    )
    parser.add_argument(
        "--imaginary-tolerance",
        type=float,
        default=None,
        help="Imaginary modes down to this magnitude (cm^-1) use the floor and are flagged; larger "
        "ones exclude the state (default: thermodynamics.imaginary_tolerance_cm1, 80)",
    )


def _floats(text):
    return [float(x) for x in str(text).split(",") if x.strip()] if text else None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_inspect(args, registry):
    from .inspector import inspect_slab, summary
    from .fileio import write_json

    report = inspect_slab(args.slab, mask_path=args.mask, vacuum_axis=args.vacuum_axis, surface_sign=args.surface_sign)
    if args.out:
        write_json(args.out, report)
    if args.site_map:
        from .sitemap import write_site_map
        from .structure import read_structure

        report["site_map"] = write_site_map(read_structure(args.slab), report, args.site_map)
    return report, summary(report) + (f"\n  site map: {report['site_map']['html']}" if args.site_map else "")


def cmd_init(args, registry):
    from .project import init_project

    result = init_project(
        args.slab,
        args.directory,
        name=args.name,
        reaction=args.reaction,
        scope=args.scope,
        cluster=args.cluster,
        job_script=args.job_script,
        incar_loose=args.incar_loose,
        incar_final=args.incar_final,
        kpoints=args.kpoints,
        preset=args.preset,
        frequency_mask=args.frequency_mask,
        plugins=args.plugin,
        interactive=False if args.yes else None,
        force=args.force,
        output=(lambda *a: print(*a, file=sys.stderr)) if getattr(args, "output_json", False) else print,
    )
    text = [
        f"wrote {result['project']} ({result['sites']} sites; families: {', '.join(result['families'])}; "
        f"protocol: {result['protocol']}; cluster: {result['cluster'] or 'not set'})",
        f"site map: {result['site_map']}",
        "next:",
    ] + [f"  {step}" for step in result["next"]]
    return result, "\n".join(text)


def cmd_generate(args, registry):
    from .project import generate

    report = generate(args.project, args.out, families=args.families.split(",") if args.families else None)
    keys = (
        "candidate_nodes",
        "computed_nodes",
        "symmetry_equivalent_nodes",
        "unique_initial_structures",
        "proposed_edges",
        "complete_stoichiometric_cycles",
        "symmetry_unique_cycles",
    )
    text = [f"campaign: {report['campaign']}"] + [f"  {k.replace('_', ' ')}: {report[k]}" for k in keys]
    if report["nodes_without_viable_start"]:
        text.append(
            f"  intermediates without a clash-free start: {len(report['nodes_without_viable_start'])} "
            "(see rejected_placements.json)"
        )
    text += [f"  WARNING: {w}" for w in report.get("warnings", [])]
    return report, "\n".join(text)


def cmd_plan(args, registry):
    from .campaign import plan

    result = plan(_campaign(args))
    n, runs = result["network"], result["dft_runs"]
    text = [
        f"{result['campaign']} ({result['reaction']}): {n['nodes']} intermediates ({n['computed_nodes']} computed, "
        f"{n['symmetry_images']} symmetry images), {n['edges']} steps, {n['cycles']} cycles "
        f"({n['symmetry_unique_cycles']} symmetry-unique)",
        "  cycles by family: " + ", ".join(f"{k} {v}" for k, v in n["cycles_by_family"].items()),
        f"  starting structures: {result['starts']}"
        + (
            f" ({len(result['nodes_without_start'])} "
            f"{_plural(len(result['nodes_without_start']), 'intermediate has', 'intermediates have')} none)"
            if result["nodes_without_start"]
            else ""
        ),
        f"  stages: {' -> '.join(result['protocol']['stages'])} (frequency.central = "
        f"{result['protocol']['frequency_central']})",
    ]
    text += [f"  {k}: {v}" for k, v in runs.items()]
    cost = result["cost"]
    text.append(
        f"  node-hours: {cost['node_hours_estimate']}"
        if cost["node_hours_estimate"] is not None
        else f"  node-hours: unknown. {cost['note']}"
    )
    text.append(f"  method card: {result['method_card']}")
    return result, "\n".join(text)


def cmd_list(args, registry):
    from .fileio import read_json

    campaign = _campaign(args)
    config = read_json(campaign / "config.json")
    nfree = int(config.get("frequency", {}).get("nfree", 2))
    rows = []
    for path in sorted((campaign / "states").glob("*/state.json")):
        meta = read_json(path)
        if args.family and args.family not in set(meta.get("family_aliases", [])) | {meta.get("family")}:
            continue
        ndof = sum(sum(bool(v) for v in x) for x in meta["frequency_mobile"])
        rows.append(
            {
                "id": meta["id"],
                "family": meta.get("family"),
                "site": meta.get("site"),
                "atoms": len(meta["atom_ids"]),
                "frequency_dof": ndof,
                "displacements": nfree * ndof,
            }
        )
    shown = rows[: args.limit] if args.limit else rows
    result = {"total_matching": len(rows), "states": shown}
    text = _table(shown, ["id", "family", "site", "atoms", "frequency_dof", "displacements"])
    if len(shown) < len(rows):
        text += f"\n... {len(rows) - len(shown)} more (--limit 0 lists all)"
    return result, text


def cmd_show_cycle(args, registry):
    from .fileio import read_json

    campaign = _campaign(args)
    network = read_json(campaign / "network.json")
    edges = {e["id"]: e for e in network["edges"]}
    cycle = next((c for c in network["cycles"] if c["id"] == args.cycle_id), None)
    if cycle is None and (campaign / "pathways" / "pathways.json").is_file():
        report = read_json(campaign / "pathways" / "pathways.json")
        cycle = next((c for c in report["cycles"] if c["id"] == args.cycle_id), None)
        edges.update({e["id"]: e for e in report.get("new_edges", [])})
    if cycle is None:
        raise ValueError(f"Unknown cycle {args.cycle_id} (not in network.json or pathways/pathways.json)")
    steps = [edges[x] for x in cycle["edge_ids"]]
    lines = [
        f"{cycle['id']}  ({cycle.get('family') or ', '.join(cycle.get('families', []))}; "
        f"{cycle.get('reaction', network.get('reaction', 'oer'))})"
    ]
    for k, step in enumerate(steps, start=1):
        from .reactions import clean

        vector = clean(step.get("reservoirs") or {"H2O": -step["water"], "H+": step["pairs"], "O2": step["oxygen"]})
        lines.append(
            f"  {k}. {step['from']} -> {step['to']}\n       {_vector_text(vector)}"
            + (f"   [{'; '.join(step.get('notes', [])[:1])}]" if step.get("notes") else "")
        )
    if cycle.get("evidence_scope"):
        lines.append(f"  evidence: {cycle['evidence_scope']}")
    return {"cycle": cycle, "steps": steps}, "\n".join(lines)


def cmd_cycles(args, registry):
    from .fileio import read_json

    campaign = _campaign(args)
    network = read_json(campaign / "network.json")
    rows = []
    for cycle in network["cycles"]:
        if args.family and cycle.get("family") != args.family:
            continue
        if args.unique and not cycle.get("symmetry_representative", True):
            continue
        rows.append(
            {
                "id": cycle["id"],
                "family": cycle.get("family"),
                "reaction": cycle.get("reaction", "oer"),
                "steps": len(cycle["edge_ids"]),
                "symmetry_representative": cycle.get("symmetry_representative", True),
                "all_starts_built": cycle.get("geometry_complete"),
            }
        )
    shown = rows[: args.limit] if args.limit else rows
    text = _table(shown, ["id", "family", "reaction", "steps", "symmetry_representative", "all_starts_built"])
    if len(shown) < len(rows):
        text += f"\n... {len(rows) - len(shown)} more (--limit 0 lists all)"
    return {"total_matching": len(rows), "cycles": shown}, text


def cmd_validate(args, registry):
    from .validation import validate_campaign

    result = validate_campaign(_campaign(args))
    text = (
        f"{'valid' if result['valid'] else 'INVALID'}: {result['n_states']} states, {result['n_nodes']} nodes, "
        f"{result['n_edges']} edges, {result['n_cycles']} cycles; {len(result['errors'])} error(s), "
        f"{len(result['warnings'])} warning(s)"
    )
    text += "".join(f"\n  ERROR {e}" for e in result["errors"][:20])
    text += "".join(f"\n  warning {w}" for w in result["warnings"][:10])
    if len(result["warnings"]) > 10:
        text += f"\n  ... {len(result['warnings']) - 10} more warnings in validation.json"
    return result, text, 0 if result["valid"] else 2


def cmd_prepare(args, registry):
    from .runner import prepare

    campaign = _campaign(args)
    result = prepare(campaign, selection=_selection(args, campaign), with_potcars=args.potcars, limit=args.limit)
    out = {
        "count": result["count"],
        "submitted": result["submitted"],
        "first_prepared": result["prepared"][:5],
        "note": result.get("note") or "Stage plans are in each state's plan.json; nothing was submitted.",
    }
    text = (
        f"prepared {result['count']} state(s); nothing was submitted.\n"
        + _table(result["prepared"][:10], ["state_id", "potcar_ready"])
        + (f"\n... {result['count'] - 10} more" if result["count"] > 10 else "")
        + (f"\nNOTE: {result['note']}" if result.get("note") else "")
    )
    return out, text


def cmd_submit(args, registry):
    from .slurm import submit

    campaign = _campaign(args)
    chosen = _selection(args, campaign)
    if args.execute:
        from .campaign import freeze_method_card

        freeze_method_card(campaign)
    result = submit(
        campaign,
        selection=chosen,
        limit=args.limit,
        through=args.through,
        execute=args.execute,
        retry=args.retry,
        max_active=args.max_active,
        refill=args.refill,
    )
    text = _submit_text(result, args)
    if result.get("error"):
        return result, text + f"\nERROR: {result['error']}", 2
    return result, text


SUBMIT_ORDER = ("submitted", "dry_run", "submission_failed", "submission_unknown", "deferred")


def _submit_text(result, args):
    """A compact summary of `submit`: counts, one row per state that matters, notes."""
    from .slurm import ATTENTION_STATUSES

    jobs = result["jobs"]
    counts = ", ".join(f"{n} {status}" for status, n in sorted(result["summary"].items(), key=lambda kv: -kv[1]))
    head = f"{'submit' if args.execute else 'preview'}: {result['count']} state(s): {counts or 'none'}"
    if result.get("max_active") is not None:
        active = result.get("active_jobs")
        head += f"; job cap {result['max_active']}" + (f" ({active} active)" if active is not None else "")
    if result.get("displacements_remaining"):
        head += f"; {result['displacements_remaining']} displacements left"

    def rank(entry):
        status = entry["status"]
        if status in SUBMIT_ORDER:
            return SUBMIT_ORDER.index(status)
        return len(SUBMIT_ORDER) + (0 if status in ATTENTION_STATUSES else 1)

    shown = [e for e in sorted(jobs, key=rank) if e["status"] != "skipped_complete"]
    rows = [
        {
            "state": e["state_id"],
            "status": e["status"],
            "next stage": e.get("stage", "-"),
            "through": e["through"],
            "job": e.get("job_id", ""),
            "note": str(e.get("reason") or e.get("warning") or e.get("error") or "")[:90],
        }
        for e in shown[:25]
    ]
    lines = [head]
    if rows:
        columns = ["state", "status", "next stage", "through", "job", "note"]
        lines.append(_table(rows, [c for c in columns if c in ("state", "status") or any(r[c] for r in rows)]))
    if len(shown) > 25:
        lines.append(f"... {len(shown) - 25} more (--json shows all)")
    complete = result["summary"].get("skipped_complete", 0)
    if complete:
        lines.append(f"{complete} state(s) already complete through the requested stage.")
    for key in ("refill_paused", "refill_changed", "stopped", "refill", "note"):
        if result.get(key):
            lines.append(f"{key.replace('_', ' ')}: {result[key]}")
    lines += [f"WARNING: {w}" for w in result.get("warnings") or []]
    if result.get("attention"):
        lines.append(f"NOTE: {len(result['attention'])} state(s) need a decision:")
        lines += [f"  {e['state_id']} ({e['status']}): {e.get('reason', '')}" for e in result["attention"][:10]]
        if len(result["attention"]) > 10:
            lines.append(f"  ... {len(result['attention']) - 10} more (--json shows all)")
    deferred = result["summary"].get("deferred", 0)
    if deferred and args.execute and not result.get("error"):
        lines.append(
            f"NOTE: {deferred} state(s) wait for a free slot under the job cap ({result['max_active']}); "
            + ("finishing jobs submit them (refill)." if args.refill else "repeat this command later, or add --refill.")
        )
    if not args.execute:
        lines.append("Preview only: add --execute to submit.")
    return "\n".join(lines)


def cmd_refill(args, registry):
    from .slurm import clear_refill, refill_status, refill_tick, set_refill

    campaign = _campaign(args)
    if args.clear:
        if args.max_active is not None or args.submit_from is not None:
            raise ValueError("--clear cannot be combined with other options")
        result = clear_refill(campaign)
    elif args.tick:
        if args.max_active is not None or args.submit_from is not None:
            set_refill(campaign, max_active=args.max_active, submit_from=args.submit_from)
        result = refill_tick(campaign)
        if result.get("error"):
            return result, render(result) + f"\nERROR: {result['error']}", 2
    elif args.off or args.on or args.max_active is not None or args.submit_from is not None:
        result = set_refill(
            campaign,
            enabled=True if args.on else False if args.off else None,
            max_active=args.max_active,
            submit_from=args.submit_from,
        )
    else:
        result = refill_status(campaign)
    return result, render(result)


def cmd_resolve(args, registry):
    from .slurm import resolve_submission

    result = resolve_submission(_campaign(args), args.state_id, job_id=args.job_id, not_submitted=args.not_submitted)
    return result, render(result)


def cmd_cleanup(args, registry):
    from .runner import cleanup

    campaign = _campaign(args)
    result = cleanup(campaign, selection=_selection(args, campaign), execute=args.execute)
    result["note"] = (
        "Deleted the listed files."
        if args.execute
        else "Dry run: nothing deleted; repeat with --execute to delete the listed files."
    )
    return result, render({k: v for k, v in result.items() if k != "states"})


def cmd_run_state(args, registry):
    from .runner import run_state

    result = run_state(args.state_dir, through=args.through, execute=args.execute, retry=args.retry)
    return result, render(result)


def cmd_select(args, registry):
    from .analysis import select_for_thermo
    from .campaign import _ids

    extra = {"cycles": _ids(args.cycles)} if args.cycles else {}
    result = select_for_thermo(
        _campaign(args),
        energy_window_eV=args.window,
        max_variants_per_node=args.max_per_node,
        include_partial=args.include_partial,
        **extra,
    )
    text = (
        f"selected {result['n_selected']} state(s) for the later stages (thermo_selection.json); "
        f"{len(result['waiting_nodes'])} node(s) wait for 02_final results, "
        f"{len(result['unresolved_nodes'])} have no accepted variant"
    )
    return result, text


def cmd_collect(args, registry):
    from .analysis import collect

    result = collect(
        _campaign(args),
        out_dir=args.out,
        temperature_K=args.temperature,
        low_frequency_cutoff_cm1=args.cutoff,
        imaginary_tolerance_cm1=args.imaginary_tolerance,
    )
    out = {
        "out_dir": result["out_dir"],
        "summary": result["summary"],
        "files": ["states.csv", "states.json", "magnetic_moments.csv", "frequencies.csv", "qc.json"],
    }
    s = result["summary"]
    lines = [
        f"collected {s['n_states']} {_plural(s['n_states'], 'state')} into {result['out_dir']}",
        f"  valid: {s['n_final_valid']} 02_final relaxations, {s['n_electronic_valid']} electronic energies, "
        f"{s['n_free_energy_valid']} free energies (05_freq)",
        f"  not run yet: {s['n_not_run']}; relaxed into another intermediate: {s['n_reassigned_to_other_node']}",
    ]
    moved = sorted((s.get("reassigned_states") or {}).items())
    lines += [f"    {state} -> {node}" for state, node in moved[:10]]
    if len(moved) > 10:
        lines.append(f"    ... {len(moved) - 10} more in qc.json")
    if s.get("energy_source"):
        lines.append(f"  energies: {s['energy_source']}")
    lines += [f"  note: {note}" for note in s.get("protocol_notes") or []]
    if s.get("limitations"):
        lines.append("  limits: " + " ".join(s["limitations"]))
    lines.append("  files: states.csv (each state, and why it is not valid), " + ", ".join(out["files"][1:]))
    return out, "\n".join(lines)


def cmd_thermo(args, registry):
    from .analysis import analyze_thermodynamics

    campaign = _campaign(args)
    references = args.references or default_references(campaign)
    activities = {}
    for item in args.activity or []:
        name, sep, value = item.partition("=")
        try:
            if not sep or not name.strip():
                raise ValueError
            activities[name.strip()] = float(value)
        except ValueError:
            raise ValueError(f"--activity {item!r}: write SPECIES=VALUE, e.g. --activity Cl-=0.5") from None
    result = analyze_thermodynamics(
        campaign,
        references,
        out_dir=args.out,
        potentials_V=_floats(args.potentials),
        include_vibrations=not args.electronic_only,
        temperature_K=args.temperature,
        low_frequency_cutoff_cm1=args.cutoff,
        imaginary_tolerance_cm1=args.imaginary_tolerance,
        energy_source=args.energy_source,
        scale=args.scale,
        pH=args.pH,
        activities=activities or None,
    )
    out = {
        "out_dir": str(Path(args.out) if args.out else campaign / "analysis"),
        "analysis_level": result["analysis_level"],
        "energy_source": result.get("energy_source"),
        "frequency_central": result.get("frequency_central"),
        "scale": result["scale"],
        "conditions": result["conditions"],
        "reservoirs_valid": result["reservoirs"]["valid"],
        "reservoir_errors": result["reservoirs"]["errors"],
        "warnings": result.get("warnings") or [],
        "notes": result.get("notes") or [],
        "nodes_with_energy": sum(n["status"] != "unresolved" for n in result["nodes"]),
        "nodes": len(result["nodes"]),
        "cycles_available": len({c["cycle_id"] for c in result["cycles"] if c["status"] == "available"}),
        "cycles": len({c["cycle_id"] for c in result["cycles"]}),
        "best_cycles": result["best_cycles"],
        "files": ["nodes.csv", "edges.csv", "cycles.csv", "resting_states.csv", "thermodynamics.json"],
    }
    lines = [
        f"{out['analysis_level']}; potentials vs {out['scale']}, pH {out['conditions']['pH']}",
        f"  intermediates with an energy: {out['nodes_with_energy']} of {out['nodes']}; cycles with every "
        f"energy: {out['cycles_available']} of {out['cycles']}",
    ]
    if not out["reservoirs_valid"]:
        lines.append("  references incomplete: " + "; ".join(e.rstrip(".") for e in out["reservoir_errors"]))
    lines += [f"  note: {n}" for n in out["notes"]]
    lines += [f"  WARNING: {w}" for w in out["warnings"]]
    for potential, by_reaction in out["best_cycles"].items():
        for reaction, best in by_reaction.items():
            eta = best.get("lowest_overpotential")
            lines.append(
                f"  {potential} V ({reaction}): lowest Gmax {best['lowest_Gmax']['cycle_id']} "
                f"{best['lowest_Gmax']['Gmax_U_eV']:.3f} eV"
                + (f"; lowest eta_TD {eta['cycle_id']} {eta['thermodynamic_overpotential_V']:.3f} V" if eta else "")
            )
    lines.append(f"  tables in {out['out_dir']}")
    return out, "\n".join(lines)


def cmd_plot_profile(args, registry):
    from .plotting import plot_profile

    campaign = _campaign(args)
    out = args.out or str(campaign / "analysis" / f"profile_{args.cycle}_{args.potential:g}V.png")
    result = plot_profile(campaign, args.cycle, args.potential, out)
    return result, f"wrote {result['figure']} and {result['vector']}"


def cmd_report(args, registry):
    from .campaign import report

    campaign = _campaign(args)
    text = report(campaign)
    return {"report": str(campaign / "report.md"), "text": text}, text


def cmd_cluster(args, registry):
    from . import clusters

    if args.cluster_command == "import":
        path, profile, notes = clusters.import_job_script(args.script, args.name, overwrite=args.force)
        result = {
            "profile": str(path),
            "notes": notes,
            "vasp_command": profile["vasp_command"],
            "sbatch": profile["sbatch"],
            "setup": profile["setup"],
        }
        text = (
            f"wrote {path}\n  vasp:   {' '.join(profile['vasp_command'])}\n  sbatch: "
            + ", ".join(f"{k}={v}" for k, v in profile["sbatch"].items())
            + f"\n  setup:  {len(profile['setup'])} line(s)"
            + "".join(f"\n  {n}" if n.startswith("warning:") else f"\n  note: {n}" for n in notes)
            + f"\nnext: `mechanet cluster show {args.name}` prints the job script this profile produces"
        )
        return result, text
    if args.cluster_command == "list":
        rows = clusters.list_profiles()
        if not rows:
            text = f"No cluster profiles in {clusters.clusters_dir()}; `mechanet cluster import JOB_SCRIPT --name NAME`"
        else:
            text = _table(rows, ["name", "status", "walltime", "vasp_command", "description"])
            text += "".join(
                f"\n{row['name']}: {row['status']}: {row['problem']}" for row in rows if row["status"] != "ok"
            )
        return {"folder": str(clusters.clusters_dir()), "profiles": rows}, text
    if args.cluster_command == "show":
        from .slurm import job_script_text

        profile = clusters.load_profile(args.name)
        placeholder = Path("/path/to/campaign/states/STATE")
        script = job_script_text(clusters.scheduler_section(profile), placeholder)
        warnings = profile.get("_warnings", [])
        result = {
            "path": profile["_path"],
            "profile": {k: v for k, v in profile.items() if not k.startswith("_")},
            "warnings": warnings,
            "job_script": script,
        }
        text = (
            f"{profile['_path']}: valid"
            + "".join(f"\nwarning: {w}" for w in warnings)
            + f"\nVASP is launched by the runner inside the job as: {' '.join(profile['vasp_command'])}"
            + "\n\nJob script for one state (STATE stands for the state folder):\n"
            + "\n".join("  " + line for line in script.splitlines())
        )
        return result, text
    if args.cluster_command == "use":
        from .fileio import read_json, write_json

        campaign = _campaign(args)
        config = read_json(campaign / "config.json")
        previous = (config.get("vasp") or {}).get("command")
        profile = clusters.load_profile(args.name)
        clusters.apply_profile(config, profile)
        write_json(campaign / "config.json", config)
        text = f"{campaign}/config.json now uses cluster profile {args.name} (new job scripts are written at submit)"
        if previous and previous != config["vasp"]["command"]:
            text += f"\nvasp.command changed from {' '.join(previous)!r} to {' '.join(config['vasp']['command'])!r}"
        text += "".join(f"\nwarning: {w}" for w in profile.get("_warnings", []))
        return {"campaign": str(campaign), "profile": args.name, "vasp_command": config["vasp"]["command"]}, text
    raise ValueError("Use `mechanet cluster import|list|show|use`")


def cmd_plugins(args, registry):
    from .fileio import read_yaml
    from .plugins import default_registry
    from .project import find_project

    errors = []  # the shared registry is not modified: its own errors are added below
    path = find_project(".")
    if path is not None:
        # Read the plugin list without validating the project: a broken plugin is what this
        # command should report, next to everything that does load.
        try:
            data = read_yaml(path)
        except Exception as exc:  # noqa: BLE001 - an unreadable project.yaml is reported, not fatal here
            data = {}
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
        listed = data.get("plugins") if isinstance(data, dict) else None
        if listed is not None and not isinstance(listed, list):
            errors.append(f"{path}: `plugins` must be a list of plugin files, not {listed!r}")
            listed = []
        for entry in listed or []:
            if not isinstance(entry, str):
                errors.append(f"{path}: plugin entry {entry!r} is not a file name")
        listed = [p for p in listed or [] if isinstance(p, str)]
        plugins = [str((path.parent / Path(p).expanduser()).resolve()) for p in listed]
        loaded = []
        for plugin in plugins:  # one at a time: list what loads and report every broken file
            try:
                default_registry(loaded + [plugin])
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc) if plugin in str(exc) else f"{plugin}: {type(exc).__name__}: {exc}")
            else:
                loaded.append(plugin)
        if loaded:
            registry = default_registry(loaded)
    errors = list(registry.errors) + errors
    reactions = [
        {
            "name": n,
            "title": registry.reaction(n).title,
            "net": _vector_text(registry.reaction(n).net),
            "scale": registry.reaction(n).potential_scale,
        }
        for n in sorted(registry.reactions)
    ]
    families = [
        {
            "name": f.name,
            "reaction": f.reaction or "-",
            "core": f.core,
            "origin": f.origin,
            "description": f.description,
        }
        for f in sorted(registry.families.values(), key=lambda f: (f.order, f.name))
    ]
    adsorbates = [
        {"name": a.name, "geometry": a.geometry, "description": a.description} for a in registry.adsorbates.values()
    ]
    result = {
        "reactions": reactions,
        "families": families,
        "adsorbates": adsorbates,
        "commands": sorted(registry.commands),
        "errors": errors,
    }
    text = [
        "Reactions:",
        _table(reactions, ["name", "scale", "net", "title"]),
        "",
        "Mechanism families:",
        _table(families, ["name", "reaction", "core", "origin", "description"]),
        "",
        "Adsorbates:",
        _table(adsorbates, ["name", "geometry", "description"]),
        "",
        "Plugin commands: " + ", ".join(result["commands"]),
    ]
    text += [f"PLUGIN ERROR: {e}" for e in errors]
    return result, "\n".join(text)


EXAMPLE_SLABS = {
    "RuO2_110": "RuO2(110) 2x1, 48 atoms: bridging O and coordinatively unsaturated Ru (OER benchmark)",
    "IrO2_110": "IrO2(110) 2x1, 48 atoms (the same rutile surface with Ir)",
    "Pt_111": "Pt(111) 3x3x4, 36 atoms, top sites (OER on a metal; HER through the plugin example of the sources)",
}


def cmd_examples(args, registry):
    from importlib import resources

    if not args.name:
        rows = [{"name": k, "slab": f"{k}.vasp", "description": v} for k, v in EXAMPLE_SLABS.items()]
        text = _table(rows, ["name", "description"]) + "\nwrite one with `mechanet examples NAME`"
        return {"examples": rows}, text
    matches = [k for k in EXAMPLE_SLABS if k.lower() == args.name.lower().removesuffix(".vasp")]
    if not matches:
        raise ValueError(f"Unknown example {args.name!r}; available: {', '.join(EXAMPLE_SLABS)}")
    name = matches[0]
    target = Path(args.out or ".") / f"{name}.vasp"
    if target.exists() and not args.force:
        raise FileExistsError(f"{target} exists; pass --force to overwrite it")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(resources.files("mechanet").joinpath(f"data/slabs/{name}.vasp").read_text())
    folder = name.split("_")[0].lower()
    return {"slab": str(target)}, (
        f"wrote {target} ({EXAMPLE_SLABS[name]})\nnext: mechanet init {shlex.quote(str(target))} {folder}"
    )


def cmd_references(args, registry):
    from . import references

    if args.collect:
        target = args.out or default_references(args.campaign)
        try:
            campaign = _campaign(args)
        except FileNotFoundError:
            if args.campaign:  # a campaign that was named must exist
                raise
            campaign = None
        result = references.collect_references(args.collect, target, campaign=campaign)
        lines = [f"updated {result['references']}"]
        for name, values in result["molecules"].items():
            correction = values.get("free_energy_correction_eV")
            lines.append(
                f"  {name}: E0 {values['energy_e0_eV']:.6f} eV; G_corr "
                + (f"{correction:.4f} eV" if correction is not None else "not yet (run freq/)")
            )
        lines.append(f"  protocol_id: {result['protocol_id']}")
        verified = [k for k, v in result["provenance"].items() if v is True]
        if verified:
            lines.append("  verified: " + ", ".join(verified))
        lines += [f"  WARNING: {w}" for w in result["warnings"]]
        if result.get("note"):
            lines.append(f"  note: {result['note']}")
        return result, "\n".join(lines)
    if args.molecules:
        from .fileio import read_json
        from .vasp import read_incar

        campaign = _campaign(args)
        config = read_json(campaign / "config.json")
        reaction = config.get("reaction", "oer")
        needed = getattr(registry.reaction(reaction), "required_references", None) or ()
        names = tuple(n for n in needed if n in references.MOLECULES) or references.REACTION_MOLECULES.get(
            reaction, ("H2", "H2O")
        )
        own = [n for n in needed if n not in references.MOLECULES]
        potcars = None
        if args.potcars:
            from .vasp import potcar_paths

            potcars = potcar_paths(config, sorted({s for n in names for s in references.MOLECULES[n]["symbols"]}))
        # the slab's protocol tags, with vasp.common_overrides and stage_overrides: what --collect compares
        slab = references.slab_protocol(campaign)
        template = read_incar(campaign / "inputs" / "INCAR_final")
        overridden = {
            k: v
            for k, v in (slab[0] if slab else {}).items()
            if v is not None and references._tag_value(v, k) != references._tag_value(template.get(k), k)
        }
        written = references.molecule_inputs(
            campaign / "inputs" / "INCAR_final", args.molecules, names, potcars=potcars, protocol=overridden or None
        )
        text = (
            "wrote "
            + ", ".join(written)
            + (
                " (INCAR_final with the slab's overrides: " + ", ".join(f"{k}={v}" for k, v in overridden.items()) + ")"
                if overridden
                else ""
            )
            + "\nrun relax/, copy CONTCAR to freq/POSCAR, run freq/, then "
            + f"`mechanet references --collect {args.molecules}`"
        )
        if own:
            text += (
                f"\n{reaction} also needs {', '.join(own)}, which has no built-in geometry here: compute "
                + ("it" if len(own) == 1 else "them")
                + " with the same settings and enter E0 and G_corr in the references file"
            )
        return {"written": written, "not_written": own}, text
    out = args.out or "references.yaml"
    path = references.write_template(out, args.reaction, force=args.force)
    return {"template": str(path)}, f"wrote {path}; fill in E0 and G_corr of each molecule"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _common_options(parser):
    """--json and --traceback after the command too (`mechanet thermo --json`); SUPPRESS keeps the
    value given before the command when they are absent."""
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="Print machine-readable JSON")
    parser.add_argument(
        "--traceback", action="store_true", default=argparse.SUPPRESS, help="Show the Python traceback of errors"
    )


def _with_common_options(subparsers):
    """Give every parser made by ``subparsers`` (plugin commands included) its own copy of the common
    options; a plugin that defines one of them itself replaces that copy (conflict_handler resolve)."""
    add_parser = subparsers.add_parser

    def add(name, **kwargs):
        kwargs.setdefault("conflict_handler", "resolve")
        parser = add_parser(name, **kwargs)
        _common_options(parser)
        return parser

    subparsers.add_parser = add
    return subparsers


def build_parser(registry):
    parser = argparse.ArgumentParser(
        prog="mechanet",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The commands are listed in the order of a study: slab and project, network, cluster and "
            "references, calculations, analysis.\nRun `mechanet COMMAND --help` for the options of a command."
        ),
    )
    parser.add_argument("--version", action="version", version=f"mechanet {__version__}")
    # Before the command; each command repeats them (own dest, so a plugin may define --json itself).
    parser.add_argument("--json", dest="json_global", action="store_true", help="Print machine-readable JSON")
    parser.add_argument(
        "--traceback", dest="traceback_global", action="store_true", help="Show the Python traceback of errors"
    )
    sub = _with_common_options(parser.add_subparsers(dest="command", required=True, metavar="COMMAND"))

    p = sub.add_parser("init", help="Inspect a slab and write a project (project.yaml, inputs, site map)")
    p.add_argument("slab")
    p.add_argument("directory", nargs="?", help="Project directory (default: the slab file name)")
    p.add_argument("--name")
    p.add_argument("--reaction", help="Reaction (default: asked; oer); a plugin reaction needs --plugin")
    p.add_argument("--scope", help="Scope preset (aem, aem+bifunctional, core, all) or comma-separated families")
    p.add_argument(
        "--plugin",
        action="append",
        metavar="FILE",
        help="Local plugin file (reactions, families, adsorbates); copied into the project. Repeatable",
    )
    p.add_argument("--cluster", help="Existing cluster profile name (with --job-script: the name to import it as)")
    p.add_argument("--job-script", help="Import this working VASP job script as the cluster profile")
    p.add_argument("--incar-loose", help="Your INCAR for the loose relaxation (give --incar-final too)")
    p.add_argument("--incar-final", help="Your INCAR for the final relaxation and frequencies (give --incar-loose too)")
    p.add_argument("--kpoints", help="Your KPOINTS (default: a Gamma mesh from the cell)")
    p.add_argument("--preset", default="pbe-d3", help="INCAR preset when no templates are given (default pbe-d3)")
    p.add_argument("--frequency-mask", help="POSCAR whose selective dynamics define the partial-Hessian region")
    p.add_argument("--yes", action="store_true", help="Accept every proposal without asking")
    p.add_argument(
        "--force",
        action="store_true",
        help="Rewrite an existing project.yaml and slab/ files (existing inputs/ files are kept)",
    )
    p.set_defaults(handler=cmd_init)

    p = sub.add_parser("inspect", help="Detect vacuum, active side, sites and symmetry of a slab")
    p.add_argument("slab")
    p.add_argument("--mask", help="Compare with the selective dynamics of a frequency-mask POSCAR")
    p.add_argument("--vacuum-axis", type=int, choices=[0, 1, 2])
    p.add_argument("--surface-sign", type=int, choices=[1, -1])
    p.add_argument("--out", help="Write the full report as JSON")
    p.add_argument("--site-map", help="Write an HTML site map here")
    p.set_defaults(handler=cmd_inspect)

    p = sub.add_parser("generate", help="Build the campaign of a project (network and starting structures)")
    p.add_argument("project", nargs="?", help="project.yaml or its folder (default: found from here upwards)")
    p.add_argument("--out", help="Campaign directory (default: <project>/campaign)")
    p.add_argument("--families", help="Comma-separated subset overriding scope.families")
    p.set_defaults(handler=cmd_generate)

    p = sub.add_parser("plan", help="Counts, DFT runs, node-hours (after the pilot) and the method card")
    p.add_argument("campaign", nargs="?")
    p.set_defaults(handler=cmd_plan)

    p = sub.add_parser("list", help="List starting structures and their frequency workload")
    p.add_argument("campaign", nargs="?")
    p.add_argument("--family")
    p.add_argument("--limit", type=int, default=25)
    p.set_defaults(handler=cmd_list)

    p = sub.add_parser("cycles", help="List the catalytic cycles of a campaign")
    p.add_argument("campaign", nargs="?")
    p.add_argument("--family")
    p.add_argument("--unique", action="store_true", help="Symmetry representatives only")
    p.add_argument("--limit", type=int, default=40)
    p.set_defaults(handler=cmd_cycles)

    p = sub.add_parser("show-cycle", help="Print the steps of one catalytic cycle")
    p.add_argument("cycle_id")
    p.add_argument("campaign", nargs="?")
    p.set_defaults(handler=cmd_show_cycle)

    p = sub.add_parser("validate", help="Check state identities, compositions and every reaction balance")
    p.add_argument("campaign", nargs="?")
    p.set_defaults(handler=cmd_validate)

    p = sub.add_parser("prepare", help="Write VASP stage plans (optionally assemble POTCARs on the cluster)")
    p.add_argument("campaign", nargs="?")
    _selection_options(p)
    p.add_argument("--limit", type=int)
    p.add_argument("--potcars", action="store_true", help="Assemble POTCARs from the licensed library")
    p.set_defaults(handler=cmd_prepare)

    p = sub.add_parser("submit", help="Preview or submit SLURM jobs (repeat to continue interrupted states)")
    p.add_argument("campaign", nargs="?")
    _selection_options(p)
    p.add_argument("--limit", type=int, help="Submit at most N jobs in this call")
    p.add_argument("--through", choices=THROUGH_CHOICES, default="final", help=THROUGH_HELP)
    p.add_argument("--execute", action="store_true", help="Really submit (default: preview)")
    p.add_argument("--retry", action="store_true", help="Retry stages whose last attempt failed")
    p.add_argument("--max-active", type=int, help="Keep at most N of this campaign's jobs pending or running")
    p.add_argument(
        "--refill", action="store_true", help="Every job that ends tops the campaign up to the cap from these states"
    )
    p.set_defaults(handler=cmd_submit)

    p = sub.add_parser("refill", help="Show, pause, resume, re-cap or remove the refill of a campaign")
    p.add_argument("campaign", nargs="?")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--off", action="store_true", help="Pause refilling (running jobs continue)")
    group.add_argument("--on", action="store_true", help="Resume refilling")
    group.add_argument("--clear", action="store_true", help="Remove the refill and its list of states")
    group.add_argument("--tick", action="store_true", help="Submit up to the cap now (e.g. in a login-node loop)")
    p.add_argument("--max-active", type=int, default=None, help="Change the refill's job cap")
    p.add_argument(
        "--submit-from",
        choices=["jobs", "login"],
        default=None,
        help="jobs: each ending job submits (default); login: jobs only record their end, --tick submits",
    )
    p.set_defaults(handler=cmd_refill)

    p = sub.add_parser("resolve-submission", help="Settle a submission whose outcome is unknown")
    p.add_argument("state_id")
    p.add_argument("campaign", nargs="?")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--job-id", help="The job exists in SLURM under this ID")
    group.add_argument("--not-submitted", action="store_true", help="No job was created")
    p.set_defaults(handler=cmd_resolve)

    p = sub.add_parser("cleanup", help="List (default) or delete WAVECAR/CHGCAR no pending stage needs")
    p.add_argument("campaign", nargs="?")
    _selection_options(p, pilot=False)
    p.add_argument("--execute", action="store_true", help="Delete the listed files (default: dry run)")
    p.set_defaults(handler=cmd_cleanup)

    # Run by the jobs themselves: kept out of the command list (no help=), documented in its own --help.
    p = sub.add_parser("run-state", description="Run one state's stages inside a compute allocation (used by jobs)")
    p.add_argument("state_dir")
    p.add_argument("--through", choices=THROUGH_CHOICES, default="final", help=THROUGH_HELP)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--retry", action="store_true")
    p.set_defaults(handler=cmd_run_state)

    p = sub.add_parser("select", help="Choose the variants that continue to 04_static and 05_freq")
    p.add_argument("campaign", nargs="?")
    p.add_argument("--window", type=float, default=None, help="02_final energy window above the node minimum (eV)")
    p.add_argument("--max-per-node", type=int, default=None, help="Maximum variants per node")
    p.add_argument(
        "--include-partial",
        action="store_true",
        help="Select from finished variants even if siblings are still running",
    )
    p.add_argument("--cycles", help="Restrict to the nodes of these cycles (comma-separated, or @file)")
    p.set_defaults(handler=cmd_select)

    p = sub.add_parser("collect", help="Parse results and write QC tables (no DFT is run)")
    p.add_argument("campaign", nargs="?")
    p.add_argument("--out")
    _frequency_options(p)
    p.set_defaults(handler=cmd_collect)

    p = sub.add_parser("thermo", help="Free energies, cycle descriptors and resting states (CHE)")
    p.add_argument("campaign", nargs="?")
    p.add_argument("--references", help="Molecular references (default: thermodynamics.references)")
    p.add_argument("--potentials", help="Comma-separated potentials (default: thermodynamics.potentials_V)")
    p.add_argument("--scale", choices=["RHE", "SHE"], type=str.upper)
    p.add_argument("--pH", type=float)
    p.add_argument("--activity", action="append", metavar="SPECIES=VALUE", help="e.g. --activity Cl-=0.5")
    p.add_argument("--electronic-only", action="store_true", help="Screening without 05_freq")
    p.add_argument("--energy-source", choices=["static", "final"], default=None)
    p.add_argument("--out")
    _frequency_options(p)
    p.set_defaults(handler=cmd_thermo)

    p = sub.add_parser("plot-profile", help="Free-energy profile of one cycle (needs matplotlib)")
    p.add_argument("cycle")
    p.add_argument("campaign", nargs="?")
    p.add_argument("--potential", type=float, required=True)
    p.add_argument("--out")
    p.set_defaults(handler=cmd_plot_profile)

    p = sub.add_parser("report", help="Write report.md: network, progress and every analysis so far")
    p.add_argument("campaign", nargs="?")
    p.set_defaults(handler=cmd_report)

    p = sub.add_parser("cluster", help="Cluster profiles: import a job script, list, show, use")
    cluster = _with_common_options(p.add_subparsers(dest="cluster_command", required=True, metavar="ACTION"))
    q = cluster.add_parser("import", help="Create a profile from a working VASP job script")
    q.add_argument("script")
    q.add_argument("--name", required=True)
    q.add_argument("--force", action="store_true")
    cluster.add_parser("list", help="List profiles")
    q = cluster.add_parser("show", help="Check a profile and preview its job script")
    q.add_argument("name")
    q = cluster.add_parser("use", help="Apply a profile to an existing campaign")
    q.add_argument("name")
    q.add_argument("campaign", nargs="?")
    p.set_defaults(handler=cmd_cluster)

    p = sub.add_parser("examples", help="List the example slabs, or write one to start a project from")
    p.add_argument("name", nargs="?", help="Example to write (e.g. RuO2_110)")
    p.add_argument("--out", help="Folder to write it to (default: here)")
    p.add_argument("--force", action="store_true", help="Overwrite an existing file")
    p.set_defaults(handler=cmd_examples)

    p = sub.add_parser("plugins", help="Reactions, mechanism families, adsorbates and plugin commands")
    p.set_defaults(handler=cmd_plugins)

    p = sub.add_parser("references", help="Molecular references: template, molecule inputs, collect results")
    p.add_argument("--reaction", default="oer", help="Reaction of the template (default oer)")
    p.add_argument("--out", help="References file to write (template) or update (--collect)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--molecules", metavar="DIR", help="Write H2/H2O(/Cl2) inputs from the campaign's INCAR_final")
    p.add_argument("--potcars", action="store_true", help="With --molecules: also write POTCARs (on the cluster)")
    p.add_argument("--collect", metavar="DIR", help="Read the molecule results into the references file")
    p.add_argument("--campaign")
    p.set_defaults(handler=cmd_references)

    for name, register in sorted(registry.commands.items()):
        try:
            register(sub, registry)
        except Exception as exc:  # noqa: BLE001 - a broken plugin command must not break the CLI
            registry.errors.append(f"command {name}: {type(exc).__name__}: {exc}")
    _order_command_list(sub)
    return parser


# The order of `mechanet --help`: a study from the slab to the report. Commands of
# third-party plugins follow, in the order they were registered.
COMMAND_ORDER = (
    "examples",
    "inspect",
    "init",
    "generate",
    "plan",
    "cycles",
    "show-cycle",
    "list",
    "validate",
    "cluster",
    "references",
    "prepare",
    "submit",
    "refill",
    "resolve-submission",
    "cleanup",
    "collect",
    "select",
    "thermo",
    "plot-profile",
    "pathways",
    "pourbaix",
    "report",
    "plugins",
)


def _order_command_list(subparsers):
    """Sort the command list of the main help into COMMAND_ORDER (argparse lists in creation order)."""
    actions = getattr(subparsers, "_choices_actions", None)
    if not isinstance(actions, list):  # pragma: no cover - argparse internals changed: keep its order
        return
    rank = {name: k for k, name in enumerate(COMMAND_ORDER)}
    actions.sort(key=lambda action: rank.get(getattr(action, "dest", None), len(rank)))


def main(argv=None):
    from .plugins import default_registry

    registry = default_registry()
    parser = build_parser(registry)
    args = parser.parse_args(argv)
    args.output_json = args.json_global or getattr(args, "json", None) is True
    show_traceback = args.traceback_global or getattr(args, "traceback", None) is True
    if getattr(args, "handler", None) is None:
        problems = [e for e in registry.errors if e.startswith(f"command {args.command}:")] or registry.errors
        print(
            f"ERROR: the command {args.command!r} did not load: {'; '.join(problems) or 'no handler'}", file=sys.stderr
        )
        return 2
    try:
        registry = _command_registry(args, registry)
        outcome = args.handler(args, registry)
        if isinstance(outcome, tuple):
            result, text, *code = outcome
            code = code[0] if code else 0
        else:
            result, text, code = outcome, None, 0
        if args.output_json:
            print(json.dumps(result, indent=2, allow_nan=False, default=str))
        else:
            print(text if text is not None else render(result))
        return code
    except BrokenPipeError:
        # The reader went away (`mechanet plugins | head`): stop quietly.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 1
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, RuntimeError, OSError, ImportError) as exc:
        if show_traceback:
            raise
        message = str(exc) if not isinstance(exc, KeyError) else f"missing key {exc}"
        if isinstance(exc, ImportError):
            message = f"a plugin could not be imported: {exc} (`mechanet plugins` lists what loads)"
        print(f"ERROR: {message}", file=sys.stderr)
        return 2
    except EOFError:
        if show_traceback:
            raise
        print(
            "ERROR: input ended before every question was answered (use --yes to accept the proposals)", file=sys.stderr
        )
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
