"""Campaign-level helpers: locating a campaign, state selections, the pilot set, the cost plan,
the method card and the progress report.
"""

from __future__ import annotations

import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

from . import __version__
from .fileio import atomic_text, file_hash, read_json, utc_now, write_json
from .vasp import STAGES, central_mode, configured_incar, energy_stage, protocol_stages, read_incar

CAMPAIGN_DIR = "campaign"


def find_campaign_dir(arg=None):
    """The campaign named on the command line, ./campaign, or the campaign of ./project.yaml."""
    if arg:
        path = Path(arg)
        if (path / "config.json").is_file() and (path / "network.json").is_file():
            return path
        if (path / "project.yaml").is_file() and (path / CAMPAIGN_DIR / "config.json").is_file():
            return path / CAMPAIGN_DIR
        raise FileNotFoundError(
            f"{path} is not a campaign (no config.json and network.json); create one with `mechanet generate`"
        )
    for candidate in (Path(CAMPAIGN_DIR), Path(".")):
        if (candidate / "config.json").is_file() and (candidate / "network.json").is_file():
            return candidate
    from .project import find_project

    project = find_project(".")
    if project is not None and (project.parent / CAMPAIGN_DIR / "config.json").is_file():
        return project.parent / CAMPAIGN_DIR
    raise FileNotFoundError("No campaign here: pass its directory, or run inside a project after `mechanet generate`")


def load_network(campaign):
    return read_json(Path(campaign) / "network.json")


def default_references(campaign):
    """thermodynamics.references of a campaign (set from project.yaml)."""
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


def _ids(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    if str(value).startswith("@"):
        value = Path(str(value)[1:]).read_text()
    return [x for x in str(value).replace(",", " ").split() if x]


def family_states(campaign, families):
    """Every state of the nodes the named families use (with the reference and symmetry representatives)."""
    wanted = set(_ids(families))
    network = load_network(campaign)
    present = {c["family"] for c in network["cycles"]} | {n.get("family") for n in network["nodes"]}
    if wanted - present:
        raise ValueError(f"Unknown or ungenerated families: {sorted(wanted - present)}")
    edges = {e["id"]: e for e in network["edges"]}
    nodes = {network.get("reference", "ref")}
    for cycle in network["cycles"]:
        if cycle["family"] in wanted:
            for key in cycle["edge_ids"]:
                nodes.update([edges[key]["from"], edges[key]["to"]])
    for node in network["nodes"]:
        if node.get("family") in wanted or wanted & set(node.get("families") or ()):
            nodes.add(node["id"])
    by_id = {n["id"]: n for n in network["nodes"]}
    for node_id in list(nodes):
        node = by_id.get(node_id, {})
        if node.get("computed_by") and not node.get("state_ids"):
            nodes.add(node["computed_by"])
    return sorted({s for n in network["nodes"] if n["id"] in nodes for s in n.get("state_ids", [])})


def pilot_states(campaign):
    """A small first batch: the reference, one start per adsorbate type, the most crowded start.

    Timings and convergence of these runs tell whether the protocol suits the system before
    the full campaign is submitted (and `mechanet plan` uses their timings).
    """
    network = load_network(campaign)
    states = {}
    for path in sorted((Path(campaign) / "states").glob("*/state.json")):
        meta = read_json(path)
        states[meta["id"]] = meta
    nodes = {n["id"]: n for n in network["nodes"]}
    reference = network.get("reference", "ref")
    chosen, why = [], {}

    def add(state_id, reason):
        if state_id in states and state_id not in chosen:
            chosen.append(state_id)
            why[state_id] = reason

    for sid in nodes.get(reference, {}).get("state_ids", []):
        add(sid, "reference surface")
        break
    native = {}
    if reference in nodes:
        native = dict(nodes[reference].get("occupancy") or {})
    seen = set()
    for node in sorted(network["nodes"], key=lambda n: (len(n.get("occupancy") or {}), n["id"])):
        if not node.get("state_ids"):
            continue
        changes = {k: v for k, v in (node.get("occupancy") or {}).items() if native.get(k) != v}
        if len(changes) != 1 or node.get("coupling"):
            continue
        status = next(iter(changes.values()))
        if status in seen:
            continue
        seen.add(status)
        add(node["state_ids"][0], f"first start with *{status}")
    crowded = max(
        states.values(), key=lambda m: (sum(abs(v) for v in m.get("delta_atoms", {}).values()), m["id"]), default=None
    )
    if crowded is not None:
        add(crowded["id"], "most added atoms")
    return chosen, why


def selection(campaign, states=None, families=None, selected=False, pilot=False):
    """None (every state) or a non-empty list of state IDs."""
    if sum(bool(x) for x in (states, families, pilot)) > 1:
        raise ValueError("Choose one of --states, --families and --pilot")
    chosen = (
        _ids(states)
        if states
        else family_states(campaign, families)
        if families
        else pilot_states(campaign)[0]
        if pilot
        else None
    )
    if selected:
        path = Path(campaign) / "thermo_selection.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run `mechanet select {campaign}` first")
        ids = read_json(path).get("state_ids", [])
        chosen = list(ids) if chosen is None else [s for s in chosen if s in set(ids)]
    if chosen is not None and not chosen:
        raise ValueError("The requested selection contains no states")
    return chosen


# ---------------------------------------------------------------------------
# Plan: what the campaign will cost
# ---------------------------------------------------------------------------


def _timings(campaign):
    """Measured elapsed seconds per stage (and per displacement unit) from finished runs."""
    from .results import parse_run

    per_stage = defaultdict(list)
    for state in sorted((Path(campaign) / "states").iterdir()) if (Path(campaign) / "states").is_dir() else []:
        for stage in STAGES:
            folder = state / stage
            if (folder / "OUTCAR").is_file():
                try:
                    seconds = parse_run(folder, include_xml=False).get("elapsed_seconds")
                except Exception:  # noqa: BLE001 - a broken OUTCAR must not stop planning
                    seconds = None
                if seconds:
                    per_stage[stage].append(float(seconds))
            units = folder / "units"
            if stage == "05_freq" and units.is_dir():
                for unit in units.iterdir():
                    if (unit / "OUTCAR").is_file():
                        try:
                            seconds = parse_run(unit, include_xml=False).get("elapsed_seconds")
                        except Exception:  # noqa: BLE001
                            seconds = None
                        if seconds:
                            per_stage["05_freq_unit"].append(float(seconds))
    return {k: sum(v) / len(v) for k, v in per_stage.items()}, {k: len(v) for k, v in per_stage.items()}


def plan(campaign):
    """Counts of nodes, cycles, starts, DFT runs and displacements; node-hours once timings exist."""
    campaign = Path(campaign)
    config = read_json(campaign / "config.json")
    network = load_network(campaign)
    report = read_json(campaign / "build_report.json")
    mode = central_mode(config)
    stages = protocol_stages(config)
    nfree = int(config.get("frequency", {}).get("nfree", 2))
    keep = int(config.get("thermo_selection", {}).get("max_variants_per_node", 2))
    states = {}
    for path in sorted((campaign / "states").glob("*/state.json")):
        meta = read_json(path)
        states[meta["id"]] = meta
    computed = [n for n in network["nodes"] if not n.get("computed_by")]
    displacements, later = 0, 0
    for node in computed:
        for sid in node.get("state_ids", [])[:keep]:
            meta = states.get(sid)
            if meta is None:
                continue
            later += 1
            displacements += nfree * sum(sum(bool(v) for v in row) for row in meta["frequency_mobile"])
    cycles = Counter(c["family"] for c in network["cycles"])
    out = {
        "campaign": str(campaign),
        "reaction": network.get("reaction", config.get("reaction")),
        "protocol": {
            "stages": stages,
            "frequency_central": mode,
            "nfree": nfree,
            "variants_per_node_after_select": keep,
        },
        "network": {
            "nodes": len(network["nodes"]),
            "computed_nodes": len(computed),
            "symmetry_images": len(network["nodes"]) - len(computed),
            "edges": len(network["edges"]),
            "cycles": len(network["cycles"]),
            "symmetry_unique_cycles": sum(bool(c.get("symmetry_representative", True)) for c in network["cycles"]),
            "cycles_by_family": dict(sorted(cycles.items())),
        },
        "starts": len(states),
        "nodes_without_start": report.get("nodes_without_viable_start", []),
        "dft_runs": {
            "relaxation_jobs (01_loose + 02_final)": len(states),
            "states_continuing_to_04_static_05_freq (estimate)": later,
            "frequency_displacements (estimate)": displacements,
        },
    }
    timing, samples = _timings(campaign)
    sched = config.get("scheduler", {}).get("sbatch", {})
    nodes_per_job = int(sched.get("nodes", 1) or 1)
    if timing:
        relax = timing.get("01_loose", 0.0) + timing.get("02_final", 0.0)
        single = timing.get("04_static", 0.0) + timing.get("03_refine", 0.0)
        unit = timing.get("05_freq_unit", 0.0)
        hours = (len(states) * relax + later * single + displacements * unit) * nodes_per_job / 3600.0
        out["cost"] = {
            "node_hours_estimate": round(hours, 1),
            "based_on_runs": samples,
            "mean_elapsed_s": {k: round(v, 1) for k, v in timing.items()},
            "note": "Mean measured times x counts; stages without a measurement count as zero.",
        }
    else:
        out["cost"] = {
            "node_hours_estimate": None,
            "note": "No finished runs yet: submit the pilot (`mechanet submit --pilot --execute`) and "
            "run `mechanet plan` again for a node-hour estimate.",
        }
    card = method_card(campaign, config)
    write_json(campaign / "method_card.json", card)
    out["method_card"] = str(campaign / "method_card.json")
    return out


# ---------------------------------------------------------------------------
# Method card
# ---------------------------------------------------------------------------


METHOD_CARD_SECTIONS = ("electronic_structure", "protocol", "thermodynamics", "families")


def _electronic_structure(incar, vasp, kpoints):
    """The electronic-structure block of the method card, from a map of INCAR tags."""

    def tag(*names):
        return {n: incar[n] for n in names if n in incar}

    return {
        "code": "VASP",
        "functional": tag("GGA", "METAGGA", "LHFCALC", "AEXX", "HFSCREEN"),
        "plane_wave_cutoff_eV": incar.get("ENCUT"),
        "spin": tag("ISPIN"),
        "hubbard_u": vasp.get("dft_u") or tag("LDAU", "LDAUTYPE", "LDAUL", "LDAUU", "LDAUJ"),
        "dispersion": tag("IVDW"),
        "solvation": tag("LSOL", "EB_K", "LAMBDA_D_K"),
        "dipole_correction": tag("LDIPOL", "IDIPOL", "DIPOL"),
        "smearing": tag("ISMEAR", "SIGMA"),
        "precision": tag("PREC", "LREAL", "EDIFF", "EDIFFG"),
        "kpoints": kpoints[:5],
        "paw_datasets": vasp.get("potcar_variants")
        or {k: Path(v).parent.name for k, v in (vasp.get("potcars") or {}).items()},
        "magnetic_moments_start": vasp.get("magmom_by_element"),
    }


def method_card(campaign, config=None):
    """Machine-readable description of the method (functional, U, dispersion, references, versions).

    The electronic-structure block describes the stage whose energies the analysis uses
    (04_static, or 02_final with frequency.central = final) as the jobs run it: INCAR_final
    with vasp.common_overrides and that stage's stage_overrides. Every override, of any stage,
    is recorded under protocol.vasp_overrides, so a later change shows in the change list.
    """
    from .plugins import campaign_registry

    campaign = Path(campaign)
    config = config or read_json(campaign / "config.json")
    vasp = config.get("vasp", {})
    thermo = config.get("thermodynamics", {})
    registry = campaign_registry(campaign, config=config)
    kpoints = (
        (campaign / "inputs" / "KPOINTS").read_text().splitlines()
        if (campaign / "inputs" / "KPOINTS").is_file()
        else []
    )
    references = thermo.get("references")
    stage = energy_stage(config)
    card = {
        "schema_version": 2,
        "written_utc": utc_now(),
        "mechanet_version": __version__,
        "python": platform.python_version(),
        "reaction": config.get("reaction"),
        "families": config.get("scope", {}).get("families"),
        "plugins": sorted({origin for _, _, origin in registry.loaded if origin != "builtin"}),
        "electronic_structure": {
            "stage": stage,
            **_electronic_structure(configured_incar(campaign, stage, config), vasp, kpoints),
        },
        "protocol": {
            "stages": protocol_stages(config),
            "frequency": config.get("frequency"),
            "max_attempts": vasp.get("max_attempts"),
            "max_continuations": vasp.get("max_continuations"),
            "vasp_overrides": {
                "common_overrides": vasp.get("common_overrides") or {},
                "stage_overrides": vasp.get("stage_overrides") or {},
            },
        },
        "thermodynamics": {
            "model": "computational hydrogen electrode",
            "temperature_K": thermo.get("temperature_K"),
            "potential_scale": thermo.get("potential_scale"),
            "pH": thermo.get("pH"),
            "references_file": references,
            "references_sha256": file_hash(references) if references and Path(references).is_file() else None,
        },
        "symmetry": {"tolerance_A": config.get("geometry", {}).get("symmetry_tolerance_A")},
        "scope": {"pair_max_distance_A": config.get("scope", {}).get("pair_max_distance_A")},
    }
    frozen = campaign / "method_card.frozen.json"
    card["frozen_at_first_submission"] = frozen.is_file()
    if frozen.is_file():
        old = read_json(frozen)
        current = card
        if int(old.get("schema_version") or 1) < 2:
            # Frozen by mechanet 0.1.4 or earlier from INCAR_final alone: compare like with like.
            template = read_incar(campaign / "inputs" / "INCAR_final")
            current = {
                **card,
                "electronic_structure": _electronic_structure(template, vasp, kpoints),
                "protocol": {k: v for k, v in card["protocol"].items() if k != "vasp_overrides"},
            }
            card["frozen_card_note"] = (
                "The frozen card was written by mechanet 0.1.4 or earlier from inputs/INCAR_final alone; "
                "changes of vasp.common_overrides or stage_overrides since the first submission cannot be detected."
            )
        card["changed_since_first_submission"] = sorted(
            k
            for k in METHOD_CARD_SECTIONS
            if json.dumps(old.get(k), sort_keys=True) != json.dumps(current.get(k), sort_keys=True)
        )
    return card


def freeze_method_card(campaign):
    """Keep the method card of the first submission (called by `submit --execute`)."""
    frozen = Path(campaign) / "method_card.frozen.json"
    if not frozen.is_file():
        write_json(frozen, method_card(campaign))
    return frozen


# ---------------------------------------------------------------------------
# Progress report
# ---------------------------------------------------------------------------


def progress(campaign):
    """Per-state stage status from the attempt records (no SLURM query)."""
    campaign = Path(campaign)
    config = read_json(campaign / "config.json")
    stages = protocol_stages(config)
    counts, rows = Counter(), []
    for path in sorted((campaign / "states").glob("*/state.json")):
        state = path.parent
        last, status = None, "not started"
        for stage in stages:
            attempt = state / stage / "attempt.json"
            if attempt.is_file():
                try:
                    status = read_json(attempt).get("status", "unknown")
                except (OSError, ValueError):
                    status = "unreadable"
                last = stage
        if last is None and (state / "plan.json").is_file():
            status = "prepared"
        key = f"{last} {status}" if last else status
        counts[key] += 1
        rows.append({"state_id": state.name, "stage": last, "status": status})
    return {"by_status": dict(sorted(counts.items())), "states": rows}


def _magnetism_section(result, label=None):
    """Report lines on local moments: spin checks, variant solutions, moments of the best cycles.

    ``label(cycle_id, node_id)`` names an intermediate (default: its node ID)."""
    from .magnetism import flag_text

    label = label or (lambda cycle_id, node_id: node_id)
    magnetism = result.get("magnetism") or {}
    cycles, variants = magnetism.get("cycles") or {}, magnetism.get("variants") or {}
    if not cycles and not variants:
        return []
    settings = magnetism["settings"]
    lines = [
        "## Magnetic moments",
        "",
        f"Projected local moments of the {magnetism['stage']} structures the energies come from. A step is "
        f"flagged when a moment of at least {settings['moment_change_muB']:g} muB changes sign, or when a moment "
        f"farther than {settings['site_radius_A']:g} A from every site metal changes by more than "
        f"{settings['moment_change_muB']:g} muB.",
        "",
    ]
    review = magnetism.get("cycles_needing_review") or []
    lines.append(
        f"- spin check: {len(review)} of {len(cycles)} cycles with moments need review"
        + (f": {', '.join(review[:10])}{', ...' if len(review) > 10 else ''}" if review else "")
    )
    for node_id, entry in sorted(variants.items())[:10]:
        solutions = entry["solutions"]
        second = solutions[1]
        differences = ", ".join(
            f"{d['element']} {d['atom_id']} {d['lowest_muB']:+.1f} -> {d['this_muB']:+.1f}"
            for d in second["differs_from_lowest"][:2]
        )
        lines.append(
            f"- variants of {label(None, node_id)} ({node_id}) converged to {len(solutions)} magnetic solutions; "
            f"the next lies {second['relative_eV']:+.3f} eV above the lowest ({differences})"
        )
    if len(variants) > 10:
        lines.append(f"- ... {len(variants) - 10} more intermediates with several solutions (thermodynamics.json)")
    shown = []
    for by_reaction in (result.get("best_cycles") or {}).values():
        for best in by_reaction.values():
            cycle_id = best["lowest_Gmax"]["cycle_id"]
            if cycle_id in cycles and cycle_id not in shown:
                shown.append(cycle_id)
    for cycle_id in shown[:3]:
        entry = cycles[cycle_id]
        heads = [label(cycle_id, n) for n in entry["node_ids"]]
        lines += [
            "",
            f"### {cycle_id} (spin check: {entry['status'].replace('_', ' ')})",
            "",
            "| atom (muB) | " + " | ".join(heads) + " |",
            "|---|" + "---|" * len(heads),
        ]
        for atom in entry["atoms"]:
            cells = ["–" if v is None else f"{v:+.2f}" for v in atom["moments_muB"]]
            lines.append(f"| {atom['element']} {atom['atom_id']} ({atom['role']}) | " + " | ".join(cells) + " |")
        if entry["flags"]:
            lines.append("")
            lines += [f"- flag: {flag_text(flag, lambda n, c=cycle_id: label(c, n))}" for flag in entry["flags"][:6]]
    lines.append("")
    return lines


def _tautomer_section(result, label=None):
    """Report lines on shared-proton tautomers: intermediates computed in several, cycles that mix them."""
    label = label or (lambda cycle_id, node_id: node_id)
    tautomers = result.get("tautomers") or {}
    cycles, nodes = tautomers.get("cycles") or {}, tautomers.get("nodes") or {}
    if not cycles and not nodes:
        return []
    review = tautomers.get("cycles_needing_review") or []
    lines = [
        "## Shared protons",
        "",
        "A proton shared between two reactive sites that an intermediate leaves native may sit on either "
        "oxygen: the same intermediate in two tautomers, whose lowest counts. A cycle needs review when its "
        "intermediates use different tautomers and some were not computed in the others'.",
        "",
        f"- {len(review)} of {len(cycles)} cycles with shared protons need review"
        + (f": {', '.join(review[:10])}{', ...' if len(review) > 10 else ''}" if review else ""),
    ]
    for node_id in sorted(nodes)[:10]:
        lines.append(
            f"- {label(None, node_id)} ({node_id}) was computed in {len(nodes[node_id]['tautomers'])} tautomers"
        )
    for cycle_id in review[:3]:
        for node_id, wanted in (cycles[cycle_id].get("not_computed") or {}).items():
            for key in wanted:
                where = " and ".join(f"H {h} on O {o}" for h, o in sorted(key.items()))
                lines.append(f"- {cycle_id}: {label(cycle_id, node_id)} was not computed with {where}")
    lines.append("")
    return lines


def _notation(campaign, network):
    from .notation import Notation
    from .plugins import campaign_registry

    try:
        registry = campaign_registry(campaign)
    except Exception:  # noqa: BLE001 - titles fall back to the family names
        registry = None
    return Notation.from_campaign(campaign, registry=registry, network=network)


def _cycle_table(network, notation):
    """Markdown table of the symmetry-unique cycles: mechanism, sites and intermediates."""
    rows = [c for c in network.get("cycles", []) if c.get("symmetry_representative", True)]
    if not rows:
        return []
    lines = [
        "### Cycles",
        "",
        "Intermediates in the order of the cycle, the step that releases the product last; `mechanet show-cycle ID` "
        "prints the steps. Notation: * terminal site, μ bridging site, □ vacancy, subscript = site number.",
        "",
        "| cycle | mechanism | intermediates |",
        "|---|---|---|",
    ]
    for cycle in rows:
        view = notation.cycle(cycle)
        title = view["title"] + (f" ({view['variant']})" if view.get("variant") else "")
        chain = " → ".join(view["labels"])
        lines.append(f"| `{cycle['id']}` | {title} | {chain} |")
    lines.append("")
    return lines


def report(campaign):
    """Markdown summary of the campaign and every analysis written so far."""
    campaign = Path(campaign)
    config = read_json(campaign / "config.json")
    network = load_network(campaign)
    build = read_json(campaign / "build_report.json")
    lines = [
        f"# {config.get('project', campaign.name)}: {network.get('reaction', 'oer').upper()} campaign",
        "",
        f"Generated by mechanet {build.get('mechanet_version')}; report written {utc_now()[:19]} UTC.",
        "",
        "## Network",
        "",
        f"- {build['candidate_nodes']} intermediates ({build['computed_nodes']} computed, "
        f"{build['symmetry_equivalent_nodes']} symmetry images), {build['proposed_edges']} steps, "
        f"{build['complete_stoichiometric_cycles']} cycles ({build['symmetry_unique_cycles']} symmetry-unique)",
        f"- families: {', '.join(build['families'])}",
        f"- {build['unique_initial_structures']} starting structures",
        "",
    ]
    notation = _notation(campaign, network)
    lines += _cycle_table(network, notation)
    status = progress(campaign)
    lines += ["## Calculations", "", "| stage and status | states |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in status["by_status"].items()]
    lines += ["", "`mechanet status` shows what is running, queued or waiting, per state and per cycle.", ""]
    thermo = campaign / "analysis" / "thermodynamics.json"
    if thermo.is_file():
        result = read_json(thermo)
        lines += [
            "## Thermodynamics",
            "",
            f"Level: {result['analysis_level']}; potentials on the "
            f"{result['scale']} scale; pH {result['conditions']['pH']}.",
            "",
        ]
        available = [c for c in result["cycles"] if c["status"] == "available"]
        lines.append(
            f"{len({c['cycle_id'] for c in available})} of {len(network['cycles'])} cycles have every energy they need."
        )
        titles = {}
        for cycle in network.get("cycles", []):
            view = notation.cycle(cycle)
            titles[cycle["id"]] = f"{view['title']}, {view['site_key']}" if view["site_key"] else view["title"]

        def named(cycle_id):
            return f"{cycle_id} ({titles[cycle_id]})" if cycle_id in titles else cycle_id

        for potential, by_reaction in result.get("best_cycles", {}).items():
            for reaction, best in by_reaction.items():
                eta = best.get("lowest_overpotential")
                lines.append(
                    f"- {potential} V, {reaction}: lowest Gmax {named(best['lowest_Gmax']['cycle_id'])}: "
                    f"{best['lowest_Gmax']['Gmax_U_eV']:.2f} eV"
                    + (
                        f"; lowest eta_TD {named(eta['cycle_id'])}: {eta['thermodynamic_overpotential_V']:.2f} V"
                        if eta
                        else ""
                    )
                )
        for potential, rest in result.get("network_resting_states", {}).items():
            transient = rest.get("transient_nodes") or {}
            lines.append(
                f"- {potential} V: most stable intermediate {notation.label(rest['node_id'])} ({rest['node_id']}, "
                f"{rest['formation_free_energy_eV']:.2f} eV from the native surface)"
                + (
                    (
                        f"; {len(transient)} intermediate that gives the product off downhill without electron "
                        "transfer is not counted as a sink"
                        if len(transient) == 1
                        else f"; {len(transient)} intermediates that give the product off downhill without "
                        "electron transfer are not counted as sinks"
                    )
                    if transient
                    else ""
                )
            )
        lines += [f"- warning: {w}" for w in result.get("warnings", [])]
        lines.append("")
        labels = {}
        for cycle in network.get("cycles", []):
            view = notation.cycle(cycle)
            labels.update({(cycle["id"], n): lab for n, lab in zip(view["node_ids"], view["labels"])})

        def label(cycle_id, node_id):
            return labels.get((cycle_id, node_id)) or notation.label(node_id)

        lines += _magnetism_section(result, label)
        lines += _tautomer_section(result, label)
    ranking = campaign / "pathways" / "ranking.json"
    if ranking.is_file():
        data = read_json(ranking)
        lines += ["## Pathways", ""]
        reactions = data.get("reactions", {})
        for reaction, best in data.get("global_minimum_limiting_potential", {}).items():
            info = reactions.get(reaction, {})
            scope = (
                f"the {info['steps_with_energies']} of {info['steps']} steps with energies"
                if "steps" in info
                else "the steps with energies"
            )
            lines.append(
                f"- {reaction}: lowest limiting potential over every cycle formed by {scope}: "
                f"{best['limiting_potential_V']:.2f} V vs {best['scale']} (eta_TD "
                f"{best['thermodynamic_overpotential_V']:.2f} V), cycle {best.get('cycle_id')}, "
                f"{len(best['edge_ids'])} steps"
            )
        lines.append("")
    pourbaix = campaign / "pourbaix" / "pourbaix.json"
    if pourbaix.is_file():
        data = read_json(pourbaix)
        lines += [
            "## Surface Pourbaix diagram",
            "",
            f"Window pH {data['pH_range'][0]:g} to {data['pH_range'][1]:g}, U {data['potential_range_V'][0]:g} "
            f"to {data['potential_range_V'][1]:g} V vs {data['scale']}.",
            "",
        ]
        lines += [
            f"- {p['label']} ({p['node_id']}): {100 * p['area_fraction']:.1f}% of the window"
            + (
                f"; gives the product off downhill {p['releases_product_downhill']} of its region "
                "(an intermediate under turnover)"
                if p.get("releases_product_downhill") == "in part"
                else "; gives the product off downhill throughout its region (an intermediate under turnover)"
                if p.get("releases_product_downhill")
                else ""
            )
            for p in data["phases"]
        ]
        left_out = data.get("excluded_transient") or []
        if left_out:
            lines.append(
                "- not shown, because they give the bound product off downhill at every potential and pH: "
                + ", ".join(f"{x['label']} ({x['node_id']})" for x in left_out)
            )
        lines.append("")
    card = campaign / "method_card.json"
    if card.is_file():
        lines += ["## Method", "", "See method_card.json (written by `mechanet plan`).", ""]
    text = "\n".join(lines) + "\n"
    atomic_text(campaign / "report.md", text)
    return text
