"""Campaign-level helpers: locating a campaign, state selections, the pilot set, the cost plan,
the method card and the progress report.
"""

from __future__ import annotations

import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

from . import __version__
from .fileio import file_hash, read_json, utc_now, write_json
from .vasp import STAGES, central_mode, protocol_stages, read_incar

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


def method_card(campaign, config=None):
    """Machine-readable description of the method (functional, U, dispersion, references, versions)."""
    from .plugins import campaign_registry

    campaign = Path(campaign)
    config = config or read_json(campaign / "config.json")
    incar = read_incar(campaign / "inputs" / "INCAR_final")
    vasp = config.get("vasp", {})
    thermo = config.get("thermodynamics", {})
    registry = campaign_registry(campaign, config=config)
    kpoints = (
        (campaign / "inputs" / "KPOINTS").read_text().splitlines()
        if (campaign / "inputs" / "KPOINTS").is_file()
        else []
    )
    references = thermo.get("references")

    def tag(*names):
        return {n: incar[n] for n in names if n in incar}

    card = {
        "schema_version": 1,
        "written_utc": utc_now(),
        "mechanet_version": __version__,
        "python": platform.python_version(),
        "reaction": config.get("reaction"),
        "families": config.get("scope", {}).get("families"),
        "plugins": sorted({origin for _, _, origin in registry.loaded if origin != "builtin"}),
        "electronic_structure": {
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
        },
        "protocol": {
            "stages": protocol_stages(config),
            "frequency": config.get("frequency"),
            "max_attempts": vasp.get("max_attempts"),
            "max_continuations": vasp.get("max_continuations"),
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
        changed = sorted(
            k
            for k in ("electronic_structure", "protocol", "thermodynamics", "families")
            if json.dumps(old.get(k), sort_keys=True) != json.dumps(card.get(k), sort_keys=True)
        )
        card["changed_since_first_submission"] = changed
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
    status = progress(campaign)
    lines += ["## Calculations", "", "| stage and status | states |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in status["by_status"].items()]
    lines.append("")
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
        for potential, by_reaction in result.get("best_cycles", {}).items():
            for reaction, best in by_reaction.items():
                eta = best.get("lowest_overpotential")
                lines.append(
                    f"- {potential} V, {reaction}: lowest Gmax {best['lowest_Gmax']['cycle_id']} "
                    f"({best['lowest_Gmax']['Gmax_U_eV']:.2f} eV)"
                    + (
                        f"; lowest eta_TD {eta['cycle_id']} ({eta['thermodynamic_overpotential_V']:.2f} V)"
                        if eta
                        else ""
                    )
                )
        for potential, rest in result.get("network_resting_states", {}).items():
            transient = rest.get("transient_nodes") or {}
            lines.append(
                f"- {potential} V: most stable intermediate {rest['node_id']} "
                f"({rest['formation_free_energy_eV']:.2f} eV from the native surface)"
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
    (campaign / "report.md").write_text(text)
    return text
