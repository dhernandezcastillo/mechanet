"""Surfaces of a campaign: their names, and the candidates for the mechanism.

A surface is a pattern of adsorbates on the reactive sites. Every computed node of a campaign
is one, and a coverage campaign (``mechanet generate --coverage``) holds nothing else.
``mechanet pourbaix`` says which surface is the most stable at each potential and pH. That is
not enough to choose where to study a mechanism: the surfaces that lie within the error of
the method of the stable one have to be carried as well. Usama, Razzaq, Hättig, Steinmann and
Exner (Nat. Commun. 16, 6137 (2025)) study four surfaces of IrO2(110) for this reason.

Everything here is at one pH and on one potential scale: the formation free energy of
surface i is DeltaG_i(U) = a_i + b_i U, with a_i = A_i + C_i pH and b_i = B_i = -(electrons
released) from :func:`mechanet.pourbaix.formation_lines`. The functions take these two
arrays, so they can be used and tested without a campaign.

* :func:`envelope_gap`: how far each surface lies above the stable one in a potential window.
* :func:`lead`: how far a stable surface stays below every other one.
* :func:`candidates`: the surfaces that are stable in the window or within a margin of it.
* :func:`compositions`: the surfaces grouped by composition (the isomers of one stoichiometry).
* :func:`hull`: the lower convex hull over the electrons released, the dual of the diagram.
* :func:`levels`: the stable surfaces with free energies, with tabulated corrections, with
  energies alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fileio import read_json
from .notation import surface_name

# A surface within this distance (eV, per cell) of the stable one is a candidate by default.
DEFAULT_MARGIN_EV = 0.2
# Two lines are the same free energy below this difference (eV).
TIE_EV = 1e-9
LEVEL_TITLES = {
    "free_energy": "free energy (frequencies)",
    "screening": "energy + tabulated ZPE − TS",
    "energy": "energy only",
}


def names(campaign, network):
    """Short name of every computed node of a campaign ({node ID: 'terminal 4 OH | bridge 2 O'}).

    Empty when the campaign has no sites.json (a network written by hand).
    """
    path = Path(campaign) / "sites.json"
    if not path.is_file():
        return {}
    sites = read_json(path).get("sites") or []
    if not sites:
        return {}
    return {
        node["id"]: surface_name(sites, node.get("occupancy") or {}, node.get("coupling"))
        for node in network.get("nodes", [])
        if not node.get("computed_by")
    }


def lines_at(lines, scale, nernst, pH):
    """(a, b) of DeltaG_i(U) = a_i + b_i U at one pH, on ``scale`` (SHE or RHE)."""
    from .pourbaix import _coefficients

    A, B, C = _coefficients(lines, scale, nernst)
    return A + C * float(pH), B


def _breakpoints(a, b, lo, hi):
    """The window ends and every potential in between where the stable line changes."""
    from .pourbaix import lower_envelope

    points = [float(lo)] + [float(u_to) for _, u_to, _ in lower_envelope(a, b, lo, hi)[:-1]] + [float(hi)]
    return sorted(set(points))


def envelope_gap(a, b, window):
    """How far each line lies above the lowest one inside a potential window, at its closest.

    Returns (gap, at): ``gap[i]`` is the minimum over U in the window of
    a_i + b_i U - min_j (a_j + b_j U), and ``at[i]`` the potential where it is reached. The
    distance is convex in U (a line minus the concave lower envelope), so its minimum lies
    at an end of the window or where the stable line changes: these points are evaluated,
    nothing is sampled.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    points = np.array(_breakpoints(a, b, *window))
    values = a[:, None] + b[:, None] * points[None, :]
    above = values - values.min(axis=0)[None, :]
    nearest = above.argmin(axis=1)
    return above[np.arange(len(a)), nearest], points[nearest]


def lead(a, b, index, segment):
    """How far line ``index`` stays below every other one on a potential segment, at most.

    Returns (lead in eV, potential where it is largest, index of the closest other line
    there), or (None, None, None) when it is the only line. A stable surface with a small
    lead is stable by less than the method can tell.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    others = [j for j in range(len(a)) if j != index]
    if not others:
        return None, None, None
    best = None
    for u in _breakpoints(a[others], b[others], *segment):
        values = a[others] + b[others] * u
        k = int(values.argmin())
        distance = float(values[k] - (a[index] + b[index] * u))
        if best is None or distance > best[0] + TIE_EV:
            best = (distance, float(u), others[k])
    return best


def candidates(lines, a, b, window, margin=DEFAULT_MARGIN_EV, potential_range=None, surface_names=None):
    """The surfaces to carry into the mechanism: stable somewhere in a potential window, or
    within ``margin`` eV of the stable surface there.

    One row per surface: the stable ones first, in the order of the potential at which they
    take over, then the others by their distance above the stable surface. ``stable_from_V``
    and ``stable_to_V`` are the range over which a surface is stable inside
    ``potential_range`` (default: the window); a range that ends at the edge of
    ``potential_range`` continues beyond it. ``lead_eV`` says by how much a stable surface
    beats its closest competitor at most, inside the window, and ``lead_over`` names it.
    """
    from .pourbaix import lower_envelope

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    lo, hi = (float(x) for x in window)
    if not hi > lo:
        raise ValueError("the potential window needs lo < hi")
    if margin < 0:
        raise ValueError("the margin must be >= 0")
    full_range = tuple(float(x) for x in (potential_range or window))
    surface_names = surface_names or {}
    gap, at = envelope_gap(a, b, (lo, hi))
    in_window = {index: (u_from, u_to) for u_from, u_to, index in lower_envelope(a, b, lo, hi)}
    in_range = {index: (u_from, u_to) for u_from, u_to, index in lower_envelope(a, b, *full_range)}
    rows = []
    for i, line in enumerate(lines):
        stable = i in in_window
        if not stable and gap[i] > margin + TIE_EV:
            continue
        row = {
            "node_id": line["node_id"],
            "surface": surface_names.get(line["node_id"]),
            "label": line["label"],
            "electrons": line["electrons"],
            "stable_in_window": stable,
            "stable_from_V": round(in_range[i][0], 6) if i in in_range else None,
            "stable_to_V": round(in_range[i][1], 6) if i in in_range else None,
            "above_stable_eV": 0.0 if stable else round(float(gap[i]), 6),
            "closest_at_V": None if stable else round(float(at[i]), 6),
            "lead_eV": None,
            "lead_at_V": None,
            "lead_over": None,
        }
        if stable:
            distance, where, other = lead(a, b, i, in_window[i])
            if distance is not None:
                row.update(lead_eV=round(distance, 6), lead_at_V=round(where, 6), lead_over=lines[other]["node_id"])
            row["_order"] = (0, in_window[i][0], line["node_id"])
        else:
            row["_order"] = (1, float(gap[i]), line["node_id"])
        rows.append(row)
    rows.sort(key=lambda row: row.pop("_order"))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def composition_text(delta_atoms):
    """'+1 O -2 H' for a change of composition relative to the slab; 'as the slab' for none."""
    parts = [f"{count:+d} {element}" for element, count in sorted((delta_atoms or {}).items()) if count]
    return " ".join(parts) or "as the slab"


def compositions(lines, a, b, potentials=(), surface_names=None):
    """The surfaces grouped by composition: the isomers of each stoichiometry.

    Surfaces of one composition release the same electrons, so their order does not depend on
    the potential. Each group gives its surfaces with their energy above the lowest one of the
    group, the spread, and the distance of that lowest one above the stable surface at each of
    ``potentials``. Groups are ordered by the electrons released.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    surface_names = surface_names or {}
    groups = {}
    for i, line in enumerate(lines):
        key = tuple(sorted((element, int(count)) for element, count in line["delta_atoms"].items() if count))
        groups.setdefault(key, []).append(i)
    out = []
    for key, members in groups.items():
        members = sorted(members, key=lambda i: (a[i], lines[i]["node_id"]))
        lowest = members[0]
        above = {}
        for u in potentials:
            values = a + b * float(u)
            above[f"{float(u):g}"] = round(float(values[lowest] - values.min()), 6)
        out.append(
            {
                "composition": composition_text(dict(key)),
                "delta_atoms": dict(key),
                "electrons": lines[lowest]["electrons"],
                "n_surfaces": len(members),
                "lowest": lines[lowest]["node_id"],
                "lowest_surface": surface_names.get(lines[lowest]["node_id"]),
                "spread_eV": round(float(a[members[-1]] - a[lowest]), 6),
                "second_eV": round(float(a[members[1]] - a[lowest]), 6) if len(members) > 1 else None,
                "above_stable_eV": above,
                "surfaces": [
                    {
                        "node_id": lines[i]["node_id"],
                        "surface": surface_names.get(lines[i]["node_id"]),
                        "above_lowest_eV": round(float(a[i] - a[lowest]), 6),
                    }
                    for i in members
                ],
            }
        )
    out.sort(key=lambda group: (group["electrons"], group["composition"]))
    return out


def hull(lines, a, b, reference):
    """The lower convex hull of the surfaces over the electrons they release.

    Each surface is the point (n_i, g_i): the electrons released while forming it and its
    formation free energy at the potential ``reference``. The surfaces on the lower convex
    hull are the ones that are stable at some potential, and two neighbours on it are equally
    stable at U = reference + (g_2 - g_1) / (n_2 - n_1): the transitions of the diagram.
    Returns {"points": one per surface, "vertices": the hull from few to many electrons, each
    with the potential ``to_next_V`` at which the next one takes over}.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    electrons = -b
    energy = a + b * float(reference)
    lowest = {}
    for i in range(len(a)):
        key = round(float(electrons[i]), 9)
        if key not in lowest or (energy[i], lines[i]["node_id"]) < (energy[lowest[key]], lines[lowest[key]]["node_id"]):
            lowest[key] = i
    chain = []
    for key in sorted(lowest):
        i = lowest[key]
        while len(chain) >= 2:
            p, q = chain[-2], chain[-1]
            cross = (electrons[q] - electrons[p]) * (energy[i] - energy[p]) - (energy[q] - energy[p]) * (
                electrons[i] - electrons[p]
            )
            if cross <= TIE_EV:  # q lies on or above the segment p-i
                chain.pop()
            else:
                break
        chain.append(i)
    vertices = []
    for k, i in enumerate(chain):
        nxt = chain[k + 1] if k + 1 < len(chain) else None
        vertices.append(
            {
                "node_id": lines[i]["node_id"],
                "electrons": float(electrons[i]),
                "deltaG_eV": round(float(energy[i]), 6),
                "to_next_V": (
                    round(float(reference + (energy[nxt] - energy[i]) / (electrons[nxt] - electrons[i])), 6)
                    if nxt is not None
                    else None
                ),
            }
        )
    on_hull = {lines[i]["node_id"] for i in chain}
    return {
        "reference_V": float(reference),
        "points": [
            {
                "node_id": lines[i]["node_id"],
                "electrons": float(electrons[i]),
                "deltaG_eV": round(float(energy[i]), 6),
                "lowest_for_its_electrons": lowest[round(float(electrons[i]), 9)] == i,
                "on_hull": lines[i]["node_id"] in on_hull,
            }
            for i in range(len(a))
        ],
        "vertices": vertices,
    }


def level_energies(data):
    """Node energies of a campaign at the three levels: {level: {node ID: energy} or an error text}.

    ``data`` comes from :func:`mechanet.analysis.energy_inputs`. ``free_energy`` needs a
    frequency calculation of the node; ``screening`` is the relaxed energy plus the tabulated
    ZPE - TS per added atom (``thermodynamics.electronic_mode_corrections``); ``energy`` is
    the relaxed energy alone.
    """
    from .energies import node_energies

    campaign, network, collected = data["campaign"], data["network"], data["collected"]
    central = data.get("central")
    source = data["energy_source"]
    tabulated = data["thermo_cfg"].get("electronic_mode_corrections", {"per_H_eV": 0.30, "per_O_eV": 0.05})
    wanted = {
        "free_energy": (True, "final" if central == "final" else "static", None),
        "screening": (False, source, tabulated),
        "energy": (False, source, {}),
    }
    out = {}
    for level, (vibrations, energy_source, corrections) in wanted.items():
        try:
            out[level] = node_energies(campaign, network, collected, vibrations, energy_source, corrections)[0]
        except (ValueError, KeyError) as error:  # a level that cannot be evaluated is said, not hidden
            out[level] = str(error)
    return out


def levels(data, scale, pH, potential_range, keep_transient=False, activities=None, surface_names=None):
    """The stable surfaces along the potential at the three levels of :func:`level_energies`.

    One entry per level: its title, the number of surfaces that have an energy at that level,
    and the stable segments [{from_V, to_V, node_id, label, surface}]; or ``unavailable`` with
    the reason (no frequency calculation yet, say). Comparing them shows what the vibrational
    terms change: energies alone usually give another sequence of surfaces.
    """
    from .pourbaix import formation_lines, lower_envelope

    surface_names = surface_names or {}
    out = []
    for level, selected in level_energies(data).items():
        entry = {"level": level, "title": LEVEL_TITLES[level]}
        if isinstance(selected, str):
            out.append({**entry, "unavailable": selected})
            continue
        try:
            all_lines, _, conditions = formation_lines({**data, "selected": selected}, activities)
        except ValueError as error:
            out.append({**entry, "unavailable": str(error)})
            continue
        shown = [line for line in all_lines if keep_transient or not line["transient"]] or list(all_lines)
        if not shown:
            out.append({**entry, "unavailable": "no surface has an energy at this level"})
            continue
        a, b = lines_at(shown, scale, conditions.nernst, pH)
        entry["n_surfaces"] = len(shown)
        entry["node_ids"] = [line["node_id"] for line in shown]
        entry["segments"] = [
            {
                "from_V": round(u_from, 6),
                "to_V": round(u_to, 6),
                "node_id": shown[index]["node_id"],
                "label": shown[index]["label"],
                "surface": surface_names.get(shown[index]["node_id"]),
            }
            for u_from, u_to, index in lower_envelope(a, b, *potential_range)
        ]
        out.append(entry)
    return out


CANDIDATE_COLUMNS = (
    "rank",
    "node_id",
    "surface",
    "electrons",
    "stable_in_window",
    "stable_from_V",
    "stable_to_V",
    "above_stable_eV",
    "closest_at_V",
    "lead_eV",
    "lead_at_V",
    "lead_over",
    "free_energy",
    "releases_product_downhill",
    "label",
)
LEVEL_OF_ANALYSIS = (
    ("CHE_with_partial_Hessian", "free_energy"),
    ("E0_plus_tabulated", "screening"),
    ("E0_without", "energy"),
)


def level_title(analysis_level):
    """'free energy (frequencies)' and so on for the ``analysis_level`` text of a result."""
    for marker, level in LEVEL_OF_ANALYSIS:
        if marker in str(analysis_level):
            return LEVEL_TITLES[level]
    return str(analysis_level)


def _write_csv(path, columns, rows):
    import csv

    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(["" if row.get(column) is None else row[column] for column in columns])
    return str(Path(path))


def analyse(
    data,
    lines,
    conditions,
    scale,
    potential_range,
    out,
    window=None,
    margin=None,
    pH=None,
    at_potentials=None,
    reference=None,
    figures=False,
    keep_transient=False,
    activities=None,
):
    """What ``mechanet pourbaix --window`` and ``--figures`` add to the diagram: (result keys, files).

    ``lines`` are the formation lines the diagram shows and ``data`` the energy inputs they
    came from. With a ``window`` (potentials on ``scale``) the candidates and the compositions
    are worked out at one pH (``pH``, default the campaign's ``thermodynamics.pH``) and written
    to candidates.csv and compositions.csv. With ``figures`` the five figures of
    :mod:`mechanet.surface_plots` are drawn over the window widened by 0.4 V (or over the whole
    potential range when there is no window).
    """
    from .pourbaix import release_downhill

    out = Path(out)
    margin = DEFAULT_MARGIN_EV if margin is None else float(margin)
    pH = float(data["thermo_cfg"].get("pH", 0.0) if pH is None else pH)
    u_lo, u_hi = (float(x) for x in potential_range)
    surface_names = names(data["campaign"], data["network"])
    a, b = lines_at(lines, scale, conditions.nernst, pH)
    energies = level_energies(data)
    with_free_energy = set(energies["free_energy"]) if isinstance(energies["free_energy"], dict) else set()
    extra = {"surface_names": surface_names, "surface_analysis": {"pH": pH, "scale": scale}}
    files = []
    found, groups = [], None
    if window is not None:
        lo, hi = (float(x) for x in window)
        full = (min(u_lo, lo), max(u_hi, hi))
        found = candidates(lines, a, b, (lo, hi), margin, full, surface_names)
        by_id = {line["node_id"]: line for line in lines}
        for row in found:
            row["free_energy"] = row["node_id"] in with_free_energy
            # Where the surface gives the product off downhill it is an intermediate under
            # turnover, not a resting surface: said for the part of the window where it is stable
            # (for the others, over the window).
            ends = (max(lo, row["stable_from_V"]), min(hi, row["stable_to_V"])) if row["stable_in_window"] else (lo, hi)
            row["releases_product_downhill"] = release_downhill(
                by_id[row["node_id"]].get("product_steps") or [], [(pH, u) for u in ends], scale, conditions.nernst
            )
        potentials = tuple(float(u) for u in at_potentials) if at_potentials else (lo, (lo + hi) / 2, hi)
        groups = compositions(lines, a, b, potentials, surface_names)
        files.append(_write_csv(out / "candidates.csv", CANDIDATE_COLUMNS, found))
        columns = ["composition", "electrons", "n_surfaces", "lowest", "lowest_surface", "spread_eV", "second_eV"]
        flat = []
        for group in groups:
            row = {key: group[key] for key in columns}
            for key, value in group["above_stable_eV"].items():
                row[f"above_stable_at_{key}V_eV"] = value
            flat.append(row)
        columns += [f"above_stable_at_{float(u):g}V_eV" for u in potentials]
        files.append(_write_csv(out / "compositions.csv", columns, flat))
        extra["surface_analysis"].update(
            window_V=[lo, hi], margin_eV=margin, potentials_V=list(potentials), potential_range_V=list(full)
        )
        extra.update(candidates=found, compositions=groups)
    reference = (
        float(reference) if reference is not None else ((window[0] + window[1]) / 2 if window else (u_lo + u_hi) / 2)
    )
    extra["hull"] = hull(lines, a, b, reference)
    if figures:
        from . import surface_plots as plots
        from .pourbaix import lower_envelope

        shown = (max(u_lo, window[0] - 0.4), min(u_hi, window[1] + 0.4)) if window else (u_lo, u_hi)
        title = level_title(data["level"])
        stable = [lines[index]["node_id"] for _, _, index in lower_envelope(a, b, *shown)]
        colours = plots.surface_colours(stable + [v["node_id"] for v in extra["hull"]["vertices"]])
        labels = {line["node_id"]: line["label"] for line in lines}
        files.append(
            plots.plot_lines(
                out / "pourbaix_lines.png",
                lines,
                a,
                b,
                shown,
                scale,
                pH,
                window,
                margin if window else None,
                [row["node_id"] for row in found],
                surface_names,
                colours,
                title,
            )
        )
        files.append(
            plots.plot_hull(out / "pourbaix_hull.png", extra["hull"], scale, pH, surface_names, labels, colours, title)
        )
        if groups is None:
            groups = compositions(lines, a, b, (), surface_names)
        else:
            files.append(
                plots.plot_margins(
                    out / "pourbaix_margins.png",
                    groups,
                    extra["surface_analysis"]["potentials_V"],
                    margin,
                    scale,
                    pH,
                    title,
                )
            )
        if any(group["n_surfaces"] > 1 for group in groups):
            files.append(plots.plot_isomers(out / "pourbaix_isomers.png", groups, with_free_energy, title))
        extra["levels"] = levels(data, scale, pH, shown, keep_transient, activities, surface_names)
        files.append(
            plots.plot_levels(
                out / "pourbaix_levels.png", extra["levels"], shown, scale, pH, window, surface_names, colours
            )
        )
    return extra, files


# ---------------------------------------------------------------------------
# Picking a surface: `mechanet generate --on`
# ---------------------------------------------------------------------------

NO_SOURCE = "none"


def relaxed_surface(campaign, node_id, registry=None):
    """The relaxed structure of a surface of a campaign, or None.

    Of the states of the intermediate whose 02_final relaxation finished and whose relaxed
    structure still is that intermediate, the one of lowest energy:
    {"state_id", "geometry" (path inside the state folder), "energy_eV"}.
    """
    from .plugins import campaign_registry
    from .results import parse_run
    from .topology import topology_check
    from .vasp import STAGE_FINAL

    campaign = Path(campaign)
    config = read_json(campaign / "config.json")
    registry = registry or campaign_registry(campaign, config=config)
    node = next((n for n in read_json(campaign / "network.json")["nodes"] if n["id"] == node_id), None)
    best = None
    for sid in (node or {}).get("state_ids", []):
        state_dir = campaign / "states" / sid
        final = parse_run(state_dir / STAGE_FINAL)
        if not final["completed"] or final.get("energy_e0_eV") is None:
            continue
        topology = topology_check(state_dir, read_json(state_dir / "state.json"), config, STAGE_FINAL, registry)
        if topology.get("assigned_node_id") != node_id or not topology.get("geometry"):
            continue
        if best is None or final["energy_e0_eV"] < best["energy_eV"]:
            best = {
                "state_id": sid,
                "geometry": Path(topology["geometry"]).as_posix(),
                "energy_eV": float(final["energy_e0_eV"]),
            }
    return best


def _same_inputs(source, config):
    """Refuse a source campaign that was built from another slab or run with other DFT inputs."""
    import hashlib

    provenance = read_json(Path(source) / "build_report.json").get("input_provenance") or {}
    names = {"seed": "slab file", "incar_loose": "INCAR_loose", "incar_final": "INCAR_final", "kpoints": "KPOINTS"}
    for key, what in names.items():
        recorded = (provenance.get(key) or {}).get("sha256")
        path = Path(config["inputs"][key])
        if recorded and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != recorded:
            raise ValueError(
                f"The {what} of the project ({path}) is not the one {source} was built with: energies of the two "
                "would not be comparable, so its calculations are not reused. Build on the surface without a source "
                f"(--from {NO_SOURCE}), or restore the file."
            )


def pick(project, config, registry, on, source=None):
    """The surface ``mechanet generate --on`` builds a campaign on: the ``scope.surface`` of its config.

    ``on`` is a name of the ``surfaces`` block of project.yaml or the ID of an intermediate of
    the source campaign (``source``; default ``<project>/coverage`` when it exists; "none" for
    no source). The result names the surface, the sites that differ from the slab, the source
    campaign whose finished calculations are reused, and the relaxed structure of the surface
    in it when there is one. ``notes`` says what could not be done.
    """
    project_dir = Path(project["_dir"])
    native = {str(s["id"]): str(s["native"]) for s in project["sites"]}
    declared = project.get("surfaces") or {}
    source_dir = None
    if source is None:
        if (project_dir / "coverage" / "network.json").is_file():
            source_dir = project_dir / "coverage"
    elif str(source).lower() != NO_SOURCE:
        source_dir = Path(source)
        if not (source_dir / "network.json").is_file():
            raise FileNotFoundError(f"{source_dir} is not a campaign (no network.json)")
    nodes = {n["id"]: n for n in read_json(source_dir / "network.json")["nodes"]} if source_dir else {}
    notes, node = [], None
    if on in declared:
        pattern = {str(k): v for k, v in declared[on].items()}
        coupling = pattern.pop("coupling", None)
        occupancy = {site: str(state) for site, state in pattern.items()}
        name = on
    elif on in nodes:
        node = nodes[on]
        if node.get("computed_by"):
            notes.append(
                f"{on} is the image of {node['computed_by']} under a translation of the slab: the campaign is built "
                f"on {node['computed_by']}, which holds the calculations."
            )
            node = nodes[node["computed_by"]]
        occupancy, coupling, name = dict(node.get("occupancy") or {}), node.get("coupling"), node["id"]
    else:
        where = (
            f"an intermediate of {source_dir}" if source_dir else "a campaign to take it from (no coverage campaign)"
        )
        known = f" (surfaces: {', '.join(sorted(declared))})" if declared else ""
        raise ValueError(
            f"No surface {on!r}: it is neither a name of `surfaces` in project.yaml{known} nor {where}. The IDs of "
            "the candidates are in pourbaix/candidates.csv of the coverage campaign."
        )
    unknown = sorted(set(occupancy) - set(native))
    if unknown:
        raise ValueError(f"The surface {name} names sites the project does not have: {unknown}")
    for state in occupancy.values():
        registry.adsorbate(state)
    occupancy = {site: state for site, state in sorted(occupancy.items()) if state != native[site]}
    surface = {"name": name, "occupancy": occupancy, "coupling": [str(x) for x in coupling] if coupling else None}
    if source_dir is not None:
        _same_inputs(source_dir, config)
        surface["source"] = {"campaign": str(source_dir.resolve())}
        if node is None:  # a declared surface: is it an intermediate of the source?
            wanted = (occupancy, sorted(surface["coupling"] or []))
            for candidate in nodes.values():
                theirs = {k: v for k, v in (candidate.get("occupancy") or {}).items() if v != native.get(k)}
                if not candidate.get("computed_by") and (theirs, sorted(candidate.get("coupling") or [])) == wanted:
                    node = candidate
                    break
        relaxed = relaxed_surface(source_dir, node["id"], registry) if node is not None else None
        if relaxed is not None:
            surface["relaxed"] = {"state_id": relaxed["state_id"], "geometry": relaxed["geometry"]}
        else:
            notes.append(
                f"{source_dir} has no finished relaxation of the surface {name} that kept it: new intermediates "
                "start from the slab as given, with every adsorbate placed anew."
            )
    surface["notes"] = notes
    return surface
