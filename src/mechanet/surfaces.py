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
    the relaxed energy alone. The relaxed energy is that of 02_final under either protocol:
    what a campaign has before any later stage is run.
    """
    from .energies import node_energies

    campaign, network, collected = data["campaign"], data["network"], data["collected"]
    central = data.get("central")
    tabulated = data["thermo_cfg"].get("electronic_mode_corrections", {"per_H_eV": 0.30, "per_O_eV": 0.05})
    wanted = {
        "free_energy": (True, "final" if central == "final" else "static", None),
        "screening": (False, "final", tabulated),
        "energy": (False, "final", {}),
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
BARE_SLAB = "slab"


def given_surface(carried):
    """``scope.surface`` for the adsorbates a slab file carried (``slab.carried`` of a campaign config):
    the pattern of the sites, and the atoms where the file had them."""
    return {
        "name": carried["name"],
        "occupancy": dict(carried["occupancy"]),
        "coupling": None,
        "given": {"positions": given_positions(carried)},
    }


def given_positions(carried):
    """Where the slab file had the atoms of the adsorbates it carried, for the sites ``occupancy`` still names.

    An atom is named after its site (``d_a0033`` on a0033, ``m0047`` itself on an open metal).
    """
    sites = set(carried["occupancy"])
    return {
        atom: xyz
        for atom, xyz in carried["positions"].items()
        if any(atom == site or atom.endswith(f"_{site}") for site in sites)
    }


def structures_of(campaign, node_id, registry=None):
    """The finished structures of a campaign that are the surface ``node_id``, lowest energy first.

    A structure is the surface when its 02_final relaxation finished and its relaxed atoms show
    the pattern of that surface (or review assigned it to it). Each entry:
    {"state_id", "geometry" (path inside the state folder), "energy_eV", "own", "built_for"}.
    ``own`` structures were built as this surface and kept it. The others were built as another
    intermediate (``built_for``) and relaxed into this one: a proton hopped, a water formed. For a
    surface added by ``mechanet review --discover`` there are no others.

    A structure that shows a lattice-translation image of the surface is left out: its energy
    counts for the surface, but its atoms sit on the sites of the image.
    """
    from .plugins import campaign_registry
    from .results import parse_run
    from .topology import topology_check
    from .vasp import STAGE_FINAL

    campaign = Path(campaign)
    config = read_json(campaign / "config.json")
    registry = registry or campaign_registry(campaign, config=config)
    found = []
    for state_dir in sorted((campaign / "states").iterdir()):
        if not (state_dir / "state.json").is_file():
            continue
        final = parse_run(state_dir / STAGE_FINAL)
        if not final["completed"] or final.get("energy_e0_eV") is None:
            continue
        meta = read_json(state_dir / "state.json")
        topology = topology_check(state_dir, meta, config, STAGE_FINAL, registry)
        if topology.get("assigned_node_id") != node_id or not topology.get("geometry"):
            continue
        if (topology.get("microstate") or {}).get("matched_node_id", node_id) != node_id:
            continue
        found.append(
            {
                "state_id": state_dir.name,
                "geometry": Path(topology["geometry"]).as_posix(),
                "energy_eV": float(final["energy_e0_eV"]),
                "own": node_id in {meta.get("node_id"), *meta.get("node_aliases", [])},
                "built_for": meta.get("node_id"),
            }
        )
    return sorted(found, key=lambda x: (x["energy_eV"], not x["own"], x["state_id"]))


def relaxed_surface(campaign, node_id, registry=None):
    """The relaxed structure that stands for a surface of a campaign, or None.

    The lowest of its finished structures (:func:`structures_of`), as in the Pourbaix diagram,
    where the lowest structure of a surface gives its energy. A structure built as the surface
    itself is preferred unless one that arrived from another start is lower by more than what
    ``mechanet select`` calls the same minimum (5 meV); the result then also carries the best of
    its own as ``own_structure``.
    """
    from .selection import DUPLICATE_ENERGY_EV

    found = structures_of(campaign, node_id, registry)
    own = next((x for x in found if x["own"]), None)
    arrived = next((x for x in found if not x["own"]), None)
    if own is None or arrived is None:
        return own or arrived
    if arrived["energy_eV"] < own["energy_eV"] - DUPLICATE_ENERGY_EV:
        return {**arrived, "own_structure": own}
    return own


def _roles_by_bonds(structure, ids, states, hysteresis, registry):
    """Who holds what in a structure, read from its distances and not from the names of its atoms.

    Returns ({site: {role label: atom index}}, {atom index: [H indices]}): the atoms of the
    adsorbate of every occupied site by the roles the builder uses (h, w, d, p, or the labels of
    a template), and the H of every other atom that holds some. ``states`` gives the state of
    every site. Raises ValueError when the atoms on a site are not bonded the way its state is
    written.
    """
    from .adsorbates import extra_elements
    from .carried import assign_roles
    from .structure import distance_matrix
    from .topology import _NATIVE_ID, _limits, _pair_type

    symbols = structure.symbols
    index = {aid: k for k, aid in enumerate(ids)}
    extra = extra_elements(registry.adsorbates.values())
    distances = distance_matrix(structure)
    hosts = [k for k, s in enumerate(symbols) if s == "O" or s in extra]
    bonded = {k: set() for k in range(len(symbols))}
    for h in (k for k, s in enumerate(symbols) if s == "H"):
        near = min(((float(distances[h, x]), x) for x in hosts), default=None)
        if near is None:
            continue
        kind = "O-H" if symbols[near[1]] == "O" else _pair_type(symbols[near[1]], "H", set(), extra)
        if near[0] <= _limits(kind, hysteresis)[2]:
            bonded[h].add(near[1])
            bonded[near[1]].add(h)
    heavy = [k for k, s in enumerate(symbols) if s in ("O", "Cl") or s in extra]
    for position, a in enumerate(heavy):
        for b in heavy[position + 1 :]:
            kind = _pair_type(symbols[a], symbols[b], set(), extra)
            if (
                kind is not None
                and not kind.startswith("M-")
                and float(distances[a, b]) <= _limits(kind, hysteresis)[1]
            ):
                bonded[a].add(b)
                bonded[b].add(a)
    site_atoms = {index[site] for site in states if site in index}
    roles, grouped = {}, set(site_atoms)
    for site, state in sorted(states.items()):
        if state == "empty":
            if site in index:
                raise ValueError(f"{site} is empty on the surface, but the structure has an atom there")
            continue
        if site not in index:
            raise ValueError(f"the structure has no atom on {site}, which holds {state} on the surface")
        proximal = index[site]
        members, frontier = {proximal}, [proximal]
        while frontier:
            current = frontier.pop()
            for other in bonded[current]:
                lattice = symbols[other] != "H" and _NATIVE_ID.match(ids[other])
                if other in members or other in site_atoms or lattice:
                    continue
                members.add(other)
                frontier.append(other)
        found = assign_roles(registry.adsorbate(state), proximal, members, bonded, symbols)
        if found is None:
            raise ValueError(f"the atoms on {site} are not bonded the way {state} is written")
        roles[site] = found
        grouped |= members
    held = {}
    for h in (k for k, s in enumerate(symbols) if s == "H" and k not in grouped):
        for host in bonded[h]:
            held.setdefault(host, []).append(h)
    return roles, held


def found_positions(campaign, state_id, geometry, changes, coupling=None, registry=None):
    """A relaxed structure of a campaign under the atom names the builder gives the surface ``changes``.

    The structure was built as another intermediate and relaxed into this surface, so its atoms
    carry the names of that other one: an H that hopped from one site to the next is still
    named after the site it left. Here every atom gets the name of the place it holds now. The
    atoms of the slab and the atom of each site keep theirs; the atoms of each adsorbate take
    the names of its roles; an H goes with the atom that holds it. The renamed structure is
    then read back, and must show exactly the pattern of the surface.

    Returns {atom name: [x, y, z]} for every atom of the surface. Raises ValueError with the
    reason when the structure cannot be written this way.
    """
    import tempfile

    from .builder import Builder
    from .plugins import campaign_registry
    from .structure import Structure, distance_matrix, read_poscar
    from .topology import DEFAULT_HYSTERESIS, classify_microstate

    campaign = Path(campaign)
    state_dir = campaign / "states" / state_id
    config = read_json(campaign / "config.json")
    registry = registry or campaign_registry(campaign, config=config)
    ids = list(read_json(state_dir / "state.json")["atom_ids"])
    relaxed = read_poscar(state_dir / geometry)
    pair = tuple(coupling) if coupling else None
    with tempfile.TemporaryDirectory() as scratch:  # the builder writes its reference when it is set up
        builder = Builder(config, Path(scratch) / "build", base_dir=campaign, registry=registry)
        native = dict(builder.native)
        wanted = {site: state for site, state in changes.items() if state != native[site]}
        built = builder.structure(dict(wanted), pair)[0]
    states = {**native, **wanted}
    if sorted(built.symbols) != sorted(relaxed.symbols):
        raise ValueError("it does not have the atoms of the surface")
    hysteresis = {
        k: tuple(v) for k, v in {**DEFAULT_HYSTERESIS, **(config.get("topology") or {}).get("hysteresis_A", {})}.items()
    }
    theirs, their_h = _roles_by_bonds(relaxed, ids, states, hysteresis, registry)
    ours, our_h = _roles_by_bonds(built, list(built.ids), states, hysteresis, registry)
    place = {aid: k for k, aid in enumerate(built.ids)}
    name = {}  # atom of the relaxed structure -> the name of its place on the surface
    for site, found in theirs.items():
        if set(found) != set(ours[site]):
            raise ValueError(f"the atoms on {site} do not match the roles of {states[site]}")
        for label, k in found.items():
            name[k] = built.ids[ours[site][label]]
    for host, hydrogens in their_h.items():
        slots = sorted(built.ids[k] for k in our_h.get(place.get(ids[host], -1), []))
        if len(slots) != len(hydrogens):
            raise ValueError(f"{ids[host]} holds {len(hydrogens)} H; on the surface as built it holds {len(slots)}")
        keep = [k for k in hydrogens if ids[k] in slots]  # an H that still sits where its name says
        free = [slot for slot in slots if slot not in {ids[k] for k in keep}]
        for k in keep:
            name[k] = ids[k]
        for k, slot in zip(sorted((k for k in hydrogens if k not in keep), key=lambda k: ids[k]), free):
            name[k] = slot
    for k, aid in enumerate(ids):
        name.setdefault(k, aid)
    back = {aid: k for k, aid in name.items()}
    if len(back) != len(ids) or set(back) != set(built.ids):
        missing = sorted(set(built.ids) - set(back))[:4]
        raise ValueError(f"its atoms cannot all be placed on the surface (no atom for {', '.join(missing) or 'some'})")
    order = [back[aid] for aid in built.ids]
    if [relaxed.symbols[k] for k in order] != list(built.symbols):
        raise ValueError("an atom would take the place of another element")
    renamed = Structure(
        list(built.symbols), relaxed.positions[order].copy(), relaxed.cell.copy(), built.mobile, list(built.ids)
    )
    read, read_pair, reason = classify_microstate(
        renamed, list(built.ids), distance_matrix(renamed), {"native": native}, hysteresis, registry=registry
    )
    if read is None or read != wanted or sorted(read_pair or []) != sorted(pair or []):
        raise ValueError(reason or "read back, it does not show the pattern of the surface")
    return {aid: [float(x) for x in renamed.positions[k]] for k, aid in enumerate(built.ids)}


def _found(source, name, best, occupancy, coupling, registry, notes):
    """``scope.surface.found`` for a structure that relaxed into the surface from another start, or
    None when its atoms cannot be renamed (``notes`` says which structure, and why)."""
    # PENDING(found-surface-recomputed): the structure is taken over, not the calculation, whose
    # folder belongs to the intermediate it was built as (docs/planned.md says what copying needs).
    which = f"The lowest structure of {name} in {source} is {best['state_id']}, a start built as {best['built_for']}"
    own = best.get("own_structure")
    try:
        positions = found_positions(source, best["state_id"], best["geometry"], occupancy, coupling, registry)
    except ValueError as error:
        notes.append(f"{which} that relaxed into it. It cannot be used as the start of the surface: {error}.")
        return None
    lower = (
        f", {own['energy_eV'] - best['energy_eV']:.3f} eV below its own calculation ({own['state_id']})" if own else ""
    )
    notes.append(
        f"{which} that relaxed into it{lower}. The surface starts from that structure here and is relaxed "
        "again: the calculation itself belongs to the other intermediate and is not copied."
    )
    return {
        "state_id": best["state_id"],
        "built_for": best["built_for"],
        "geometry": best["geometry"],
        "energy_eV": best["energy_eV"],
        "own_state_id": own["state_id"] if own else None,
        "own_energy_eV": own["energy_eV"] if own else None,
        "positions": positions,
    }


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


def pick(project, config, registry, on, source=None, structure=True):
    """The surface ``mechanet generate --on`` builds a campaign on: the ``scope.surface`` of its config.

    ``on`` is a name of the ``surfaces`` block of project.yaml or the ID of an intermediate of
    the source campaign (``source``; default ``<project>/coverage`` when it exists; "none" for
    no source). The result names the surface, the sites that differ from the slab, the source
    campaign whose finished calculations are reused, and the structure that stands for the
    surface in it: ``relaxed`` (a calculation of its own) or ``found`` (the lowest structure
    relaxed into it from a start built as another intermediate, with its atoms renamed, see
    :func:`found_positions`). ``notes`` says what could not be done. ``structure=False`` reads
    the pattern only (for a count).
    """
    project_dir = Path(project["_dir"])
    native = {str(s["id"]): str(s["native"]) for s in project["sites"]}
    declared = dict(project.get("surfaces") or {})
    carried = (config.get("slab") or {}).get("carried")
    if carried and carried["name"] not in declared:
        declared[carried["name"]] = dict(carried["occupancy"])  # the slab as it was given
    if BARE_SLAB not in declared:
        declared[BARE_SLAB] = {}  # the slab file of the project, without any pattern
    source_dir, notes = None, []
    default = project_dir / "coverage"
    if source is None:
        if (default / "network.json").is_file():
            source_dir = default
    elif str(source).lower() != NO_SOURCE:
        source_dir = Path(source)
        if not (source_dir / "network.json").is_file():
            raise FileNotFoundError(f"{source_dir} is not a campaign (no network.json)")
    # Without a source (--from none) an ID is still read from the coverage campaign of the
    # project: the pattern of its sites only, no calculation and no structure.
    listed_in = source_dir or (default if (default / "network.json").is_file() and on not in declared else None)
    nodes = {n["id"]: n for n in read_json(listed_in / "network.json")["nodes"]} if listed_in else {}
    if source_dir is None and on in nodes:
        notes.append(
            f"The pattern of {on} is read from {listed_in}; none of its calculations is reused and no relaxed "
            "structure is taken from it (--from none)."
        )
    node = None
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
        where = f"an intermediate of {listed_in}" if listed_in else "a campaign to take it from (no coverage campaign)"
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
        best = relaxed_surface(source_dir, node["id"], registry) if node is not None and structure else None
        if best is not None and not best["own"]:
            # The lowest structure of this surface started as another intermediate.
            surface["found"] = _found(source_dir, name, best, occupancy, surface["coupling"], registry, notes)
            if surface["found"] is None:  # not usable: its own calculation, when it has one
                del surface["found"]
                best = best.get("own_structure")
        if best is not None and "found" not in surface:
            surface["relaxed"] = {"state_id": best["state_id"], "geometry": best["geometry"]}
        if structure and "relaxed" not in surface and "found" not in surface:
            notes.append(
                f"{source_dir} has no finished relaxation of the surface {name} that kept it: new intermediates "
                + (
                    "start from the slab file as it was given."
                    if carried and name == carried["name"]
                    else "start from the slab as given, with every adsorbate placed anew."
                )
            )
    if carried and name == carried["name"] and "relaxed" not in surface and "found" not in surface:
        surface["given"] = {"positions": given_positions(carried)}  # the atoms where the slab file had them
    surface["notes"] = notes
    return surface
