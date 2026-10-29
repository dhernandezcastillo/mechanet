"""Surface Pourbaix diagrams from the computed intermediates of a campaign.

Every computed node is a surface termination that forms from the reference surface and the
formation species of the campaign reaction (H2O and H+ for OER; also Cl- for CER). Under the
computational hydrogen electrode its formation free energy is linear in the potential and
the pH::

    DeltaG_f(U_SHE, pH) = A + B * U_SHE + C * pH,     B = -n_e,  C = -nu(H+) * kT ln 10

where n_e is the number of electrons released and nu(H+) the protons released while
forming the node. The stable termination at (U, pH) is the node of lowest DeltaG_f. On the
RHE scale B stays and C becomes (n_e - nu(H+)) kT ln 10, which vanishes for terminations
reached by proton-coupled electron transfers only.

The diagram covers the computed nodes only: a coverage absent from the network cannot
appear. The ``coverage`` family (``scope.coverage`` in project.yaml) enumerates terminations
for this purpose. Configurational entropy and adsorbate-adsorbate interactions beyond the
computed cell are not included.

The reaction's product is not a reservoir of the diagram (above 1.23 V vs RHE evolving O2
would undercut every termination). A node that holds the product bound, such as \\*OO, can
therefore look stable although it gives the product off without electron transfer. Nodes
whose *chemical* product release (no electron, no proton: independent of U and pH) is
downhill are left out by default and listed; nodes whose release is electrochemical stay,
and every region where that release runs downhill is flagged as an intermediate under
turnover rather than a resting termination.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .fileio import dump_json
from .reactions import electrons as vector_electrons
from .thermo import DOWNHILL_TOLERANCE_EV, Conditions, species_free_energy


def node_label(node, native):
    """Readable occupancy of a node: changed sites only (``native`` for the reference)."""
    changes = {k: v for k, v in (node.get("occupancy") or {}).items() if native.get(k) != v}
    if not changes and not node.get("coupling"):
        return "native"
    text = " ".join(f"{site}:{status}" for site, status in sorted(changes.items()))
    if node.get("coupling"):
        text += " (" + "-".join(node["coupling"]) + " coupled)"
    return text


def _reactions_of(data):
    """The reactions of a campaign network (main reaction first), from :func:`analysis.energy_inputs` data."""
    network, main = data["network"], data["main"]
    names = {c.get("reaction") for c in network.get("cycles", []) if c.get("reaction")} - {main.name}
    registry = data.get("registry")
    return [main] + [registry.reaction(n) for n in sorted(names)] if registry is not None else [main]


def _plane(vector, reservoirs, conditions):
    """(A, B, C) of sum nu mu - n_e U_SHE as A + B U_SHE + C pH (mu at pH 0)."""
    mu = sum(float(nu) * species_free_energy(name, reservoirs, conditions) for name, nu in vector.items())
    return mu, -float(vector_electrons(vector)), -float(vector.get("H+", 0)) * conditions.nernst


def formation_lines(data, activities=None):
    """(A, B, C) of every computed node with an energy; ``data`` from :func:`analysis.energy_inputs`.

    Each line also lists the node's ``product_steps``: the steps from the node (or a symmetry
    image of it) that release a reaction's product (its turnover marker, when produced), with
    their DeltaG = A + B U_SHE + C pH. ``transient`` is set when one of them is chemical (no
    charged species, so independent of U and pH) and downhill: the node gives the product
    off at every potential and pH. A consumed reactant (O2 in ORR) is not part of the
    diagram's environment, so steps that take it up are not considered.
    """
    network, selected, main = data["network"], data["selected"], data["main"]
    reservoirs = data["reservoirs"]
    if not reservoirs["valid"]:
        raise ValueError("Reference chemical potentials are incomplete: " + "; ".join(reservoirs["errors"]))
    reference = network.get("reference", "ref")
    if reference not in selected:
        raise ValueError(
            f"The reference surface ({reference}) has no accepted energy yet; it anchors every formation free energy"
        )
    native = {}
    for node in network["nodes"]:
        if node["id"] == reference:
            native = dict(node.get("occupancy") or {})
    conditions = Conditions(
        data["settings"]["temperature_K"], 0.0, {**(data["thermo_cfg"].get("activities") or {}), **(activities or {})}
    )
    from .analysis import is_chemical
    from .pathways import product_release_steps

    representative = {n["id"]: n.get("computed_by") or n["id"] for n in network["nodes"]}
    reactions = _reactions_of(data)
    registry = data.get("registry")
    steps = product_release_steps(data.get("campaign") if registry is not None else None, network, registry, reactions)
    product_steps = {}
    for step in steps:
        if step["from"] not in selected or step["to"] not in selected:
            continue
        vector = step["vector"]
        a, b, c = _plane(vector, reservoirs, conditions)
        a += selected[step["to"]] - selected[step["from"]]
        product_steps.setdefault(representative.get(step["from"], step["from"]), []).append(
            {
                "edge_id": step["edge_id"],
                "reaction": step["reaction"],
                "species": step["species"],
                "proposed": step["proposed"],
                "kind": "chemical" if is_chemical(vector) else "electrochemical",
                "A_eV": a,
                "B_eV_per_V": b,
                "C_eV_per_pH": c,
            }
        )
    lines, missing = [], []
    for node in network["nodes"]:
        if node.get("computed_by"):
            continue
        if node["id"] not in selected:
            missing.append(node["id"])
            continue
        vector = main.formation_vector(node["delta_atoms"])
        a, b, c = _plane(vector, reservoirs, conditions)
        steps = product_steps.get(node["id"], [])
        downhill = [s for s in steps if s["kind"] == "chemical" and s["A_eV"] <= DOWNHILL_TOLERANCE_EV]
        lines.append(
            {
                "node_id": node["id"],
                "label": node_label(node, native),
                "A_eV": selected[node["id"]] - selected[reference] + a,
                "B_eV_per_V": b,
                "C_eV_per_pH": c,
                "electrons": vector_electrons(vector),
                "formation_vector": vector,
                "delta_atoms": node["delta_atoms"],
                "symmetry_images": sum(1 for n in network["nodes"] if n.get("computed_by") == node["id"]),
                "product_steps": steps,
                "transient": (
                    {
                        "edge_id": downhill[0]["edge_id"],
                        "deltaG_eV": downhill[0]["A_eV"],
                        "reason": f"bound {downhill[0]['species']} is given off downhill without electron transfer "
                        f"({downhill[0]['edge_id']}"
                        + (", a step no family wrote" if downhill[0]["proposed"] else "")
                        + f", {downhill[0]['A_eV']:+.3f} eV at every U and pH)",
                    }
                    if downhill
                    else None
                ),
            }
        )
    return lines, missing, conditions


def _coefficients(lines, scale, nernst):
    A = np.array([x["A_eV"] for x in lines])
    B = np.array([x["B_eV_per_V"] for x in lines])
    C = np.array([x["C_eV_per_pH"] for x in lines])
    if scale == "RHE":  # U_SHE = U_RHE - nernst * pH
        C = C - B * nernst
    return A, B, C


def lower_envelope(a, b, u_min, u_max, tolerance=1e-12):
    """Exact stable segments of the lines a_i + b_i U on [u_min, u_max]: [(u_from, u_to, index)]."""
    values = a + b * u_min
    current = int(min(range(len(a)), key=lambda i: (round(values[i], 12), b[i], i)))
    u, out = u_min, []
    while True:
        best_u, best = None, None
        for j in range(len(a)):
            if b[j] >= b[current] - tolerance:
                continue
            cross = (a[j] - a[current]) / (b[current] - b[j])
            if cross <= u + tolerance:
                continue
            if best_u is None or cross < best_u - tolerance or (abs(cross - best_u) <= tolerance and b[j] < b[best]):
                best_u, best = cross, j
        if best_u is None or best_u >= u_max:
            out.append((u, u_max, current))
            return out
        out.append((u, best_u, current))
        u, current = best_u, best


def _clip(polygon, a, b, c, tolerance=1e-12):
    """Part of a convex polygon [(x, y)] where a + b*y + c*x <= 0 (Sutherland-Hodgman, one edge)."""
    out = []
    n = len(polygon)
    for k in range(n):
        p, q = polygon[k], polygon[(k + 1) % n]
        fp, fq = a + b * p[1] + c * p[0], a + b * q[1] + c * q[0]
        if fp <= tolerance:
            out.append(p)
        if (fp < -tolerance and fq > tolerance) or (fp > tolerance and fq < -tolerance):
            t = fp / (fp - fq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def _area_centroid(polygon):
    area, cx, cy = 0.0, 0.0, 0.0
    n = len(polygon)
    for k in range(n):
        (x0, y0), (x1, y1) = polygon[k], polygon[(k + 1) % n]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area *= 0.5
    if abs(area) < 1e-15:
        return 0.0, (sum(p[0] for p in polygon) / n, sum(p[1] for p in polygon) / n)
    return abs(area), (cx / (6 * area), cy / (6 * area))


def release_downhill(steps, points, scale, nernst, tolerance=DOWNHILL_TOLERANCE_EV):
    """Where a node's product release runs downhill over a convex region given by its corners.

    ``steps``: the ``product_steps`` of a formation line; ``points``: [(pH, U)] on ``scale``.
    Returns "throughout", "in part" or None (DeltaG is linear, so the corners decide).
    """
    found = None
    for step in steps:
        a, b, c = step["A_eV"], step["B_eV_per_V"], step["C_eV_per_pH"]
        if scale == "RHE":
            c = c - b * nernst
        values = [a + b * u + c * ph for ph, u in points]
        if all(v <= tolerance for v in values):
            return "throughout"
        if any(v <= tolerance for v in values):
            found = "in part"
    return found


def stability_regions(A, B, C, pH_range, potential_range):
    """Exact region of every line, as a convex polygon in (pH, U): [(index, polygon, area, centroid)].

    Line k is stable where A_k + B_k U + C_k pH <= A_j + B_j U + C_j pH for every j; ties go to
    the lower index, so the regions tile the window without overlap.
    """
    (p_lo, p_hi), (u_lo, u_hi) = pH_range, potential_range
    window = [(p_lo, u_lo), (p_hi, u_lo), (p_hi, u_hi), (p_lo, u_hi)]
    out = []
    for k in range(len(A)):
        polygon = list(window)
        for j in range(len(A)):
            if j == k or not polygon:
                continue
            shift = 1e-10 if j < k else 0.0
            polygon = _clip(polygon, A[k] - A[j] + shift, B[k] - B[j], C[k] - C[j])
        if len(polygon) >= 3:
            area, centroid = _area_centroid(polygon)
            if area > 1e-9:
                out.append((k, polygon, area, centroid))
    return out


def pourbaix(
    campaign,
    references,
    pH_range=(0.0, 14.0),
    potential_range=(-0.5, 2.5),
    scale="SHE",
    pH_points=(0.0, 7.0, 14.0),
    include_vibrations=True,
    energy_source=None,
    activities=None,
    out=None,
    plot=False,
    registry=None,
    keep_transient=False,
):
    """Stable surface terminations over a pH x potential window; writes pourbaix/*.json|csv[|png].

    Nodes that give the product off chemically and downhill (bound O2, a coupled O-O pair
    that releases O2, ...) are left out unless ``keep_transient``; see the module docstring.
    """
    from .analysis import energy_inputs

    campaign = Path(campaign).resolve()
    scale = str(scale).upper()
    if scale not in ("SHE", "RHE"):
        raise ValueError("scale must be SHE or RHE")
    out = Path(out) if out else campaign / "pourbaix"
    data = energy_inputs(campaign, references, include_vibrations, energy_source=energy_source, registry=registry)
    all_lines, missing, conditions = formation_lines(data, activities)
    if not all_lines:
        raise ValueError("No computed node has an energy yet")
    notes = []
    lines = [line for line in all_lines if keep_transient or not line["transient"]]
    if not lines:
        lines = list(all_lines)
        notes.append("Every computed node gives the product off downhill; all are shown.")
    shown = {line["node_id"] for line in lines}
    excluded = [
        {"node_id": line["node_id"], "label": line["label"], **line["transient"]}
        for line in all_lines
        if line["node_id"] not in shown
    ]
    if excluded:
        notes.append(
            "Left out (bound product given off downhill without electron transfer, so not a stable termination): "
            + ", ".join(f"{x['label']} ({x['node_id']})" for x in excluded)
            + "; --keep-transient shows them."
        )
    A, B, C = _coefficients(lines, scale, conditions.nernst)
    p_lo, p_hi = map(float, pH_range)
    u_lo, u_hi = map(float, potential_range)
    if not (p_hi > p_lo and u_hi > u_lo):
        raise ValueError("Empty pH or potential range")
    total = (p_hi - p_lo) * (u_hi - u_lo)
    phases = []
    for index, polygon, area, centroid in stability_regions(A, B, C, (p_lo, p_hi), (u_lo, u_hi)):
        phases.append(
            {
                "node_id": lines[index]["node_id"],
                "label": lines[index]["label"],
                "area_fraction": round(area / total, 6),
                "pH_range": [round(min(p[0] for p in polygon), 6), round(max(p[0] for p in polygon), 6)],
                "potential_range_V": [round(min(p[1] for p in polygon), 6), round(max(p[1] for p in polygon), 6)],
                "centroid": [round(centroid[0], 6), round(centroid[1], 6)],
                "polygon": [[round(x, 6), round(y, 6)] for x, y in polygon],
                "releases_product_downhill": release_downhill(
                    lines[index]["product_steps"], polygon, scale, conditions.nernst
                ),
            }
        )
    phases.sort(key=lambda p: -p["area_fraction"])
    flagged = [p for p in phases if p["releases_product_downhill"]]
    if flagged:
        notes.append(
            "Releases the product downhill in its region (an intermediate under turnover, not a resting "
            "termination): " + ", ".join(f"{p['label']} ({p['releases_product_downhill']})" for p in flagged) + "."
        )
    cuts = []
    for value in [float(x) for x in pH_points]:
        a = A + C * value
        for u_from, u_to, index in lower_envelope(a, B, u_lo, u_hi):
            cuts.append(
                {
                    "pH": value,
                    "scale": scale,
                    "from_V": round(u_from, 6),
                    "to_V": round(u_to, 6),
                    "node_id": lines[index]["node_id"],
                    "label": lines[index]["label"],
                    "electrons": lines[index]["electrons"],
                    "releases_product_downhill": release_downhill(
                        lines[index]["product_steps"], [(value, u_from), (value, u_to)], scale, conditions.nernst
                    ),
                }
            )
    out.mkdir(parents=True, exist_ok=True)
    with (out / "pourbaix.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "node_id",
                "label",
                "A_eV",
                "B_eV_per_V",
                "C_eV_per_pH",
                "electrons",
                "stable",
                "area_fraction",
                "left_out",
            ]
        )
        stable = {p["node_id"]: p["area_fraction"] for p in phases}
        A_all, B_all, C_all = _coefficients(all_lines, scale, conditions.nernst)
        for line, a, b, c in zip(all_lines, A_all, B_all, C_all):
            writer.writerow(
                [
                    line["node_id"],
                    line["label"],
                    f"{a:.6f}",
                    f"{b:g}",
                    f"{c:.6f}",
                    line["electrons"],
                    line["node_id"] in stable,
                    stable.get(line["node_id"], 0.0),
                    "" if line["node_id"] in shown else line["transient"]["reason"],
                ]
            )
    with (out / "pourbaix_1d.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        columns = ("pH", "scale", "from_V", "to_V", "node_id", "label", "electrons", "releases_product_downhill")
        writer.writerow(columns)
        for row in cuts:
            writer.writerow([row[k] if row[k] is not None else "" for k in columns])
    for line, a, b, c in zip(lines, A, B, C):
        line.update({f"A_{scale}_eV": float(a), f"B_{scale}_eV_per_V": float(b), f"C_{scale}_eV_per_pH": float(c)})
    result = {
        "schema_version": 2,
        "scale": scale,
        "pH_range": [p_lo, p_hi],
        "potential_range_V": [u_lo, u_hi],
        "conditions": {**conditions.describe(), "nernst_V": conditions.nernst},
        "analysis_level": data["level"],
        "reaction": data["main"].name,
        "reference_node": data["network"].get("reference", "ref"),
        "lines": lines,
        "nodes_without_energy": missing,
        "excluded_transient": excluded,
        "phases": phases,
        "transitions": cuts,
        "notes": notes + list(data["reservoirs"].get("notes") or []),
        "warnings": list(data["reservoirs"].get("warnings") or []),
        "files": [str(out / "pourbaix.json"), str(out / "pourbaix.csv"), str(out / "pourbaix_1d.csv")],
        "interpretation": (
            "Most stable computed termination at each pH and potential (CHE formation free "
            "energy from the reference surface, with H2O and H+ (and Cl- for CER) as reservoirs). "
            "The product is not a reservoir: terminations that give it off chemically and downhill "
            "are left out, and regions where a termination releases it electrochemically and downhill "
            "are flagged. Terminations that were not computed cannot appear; configurational entropy "
            "is neglected."
        ),
        "formula": (
            "DeltaG_f = A + B U_SHE + C pH with B = -(electrons released) and C = -nu(H+) kT ln10; "
            "the *_RHE coefficients give the same on the RHE scale"
        ),
    }
    if plot:
        from .plotting import plot_pourbaix

        result["files"].append(plot_pourbaix(result, out / "pourbaix.png"))
    dump_json(out / "pourbaix.json", result)
    return result


def register_command(subparsers, registry):
    p = subparsers.add_parser(
        "pourbaix", help="Surface Pourbaix diagram of the computed terminations", description=__doc__.split("\n\n")[0]
    )
    p.add_argument("campaign", nargs="?")
    p.add_argument("--references", help="References file (default: thermodynamics.references of the campaign)")
    p.add_argument("--pH", default="0:14", help="pH window lo:hi (default 0:14)")
    p.add_argument("--potential", default="-0.5:2.5", help="Potential window lo:hi in V (default -0.5:2.5)")
    p.add_argument("--scale", default="SHE", choices=["SHE", "RHE"], type=str.upper)
    p.add_argument("--at-pH", default="0,7,14", help="pH values of the exact 1-D transitions (default 0,7,14)")
    p.add_argument("--electronic-only", action="store_true", help="Without 05_freq: E0 + tabulated ZPE-TS (screening)")
    p.add_argument(
        "--keep-transient",
        action="store_true",
        help="Also show terminations that give the product off downhill without electron transfer (e.g. bound O2)",
    )
    p.add_argument("--plot", action="store_true", help="Also write pourbaix.png (needs matplotlib)")
    p.set_defaults(handler=_cli)


def _window(text, name):
    try:
        lo, hi = (float(x) for x in str(text).split(":"))
    except ValueError:
        raise ValueError(f"--{name} must be lo:hi, e.g. 0:14") from None
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        raise ValueError(f"--{name}: need finite lo < hi")
    return lo, hi


def _cli(args, registry):
    from .campaign import find_campaign_dir
    from .cli import default_references

    args.campaign = find_campaign_dir(args.campaign)
    references = args.references or default_references(args.campaign)
    result = pourbaix(
        args.campaign,
        references,
        _window(args.pH, "pH"),
        _window(args.potential, "potential"),
        args.scale,
        [float(x) for x in args.at_pH.split(",") if x.strip()],
        include_vibrations=not args.electronic_only,
        plot=args.plot,
        keep_transient=args.keep_transient,
    )
    out = {
        "phases": [
            {k: p[k] for k in ("node_id", "label", "area_fraction", "releases_product_downhill")}
            for p in result["phases"]
        ],
        "transitions": result["transitions"],
        "left_out": [{k: x[k] for k in ("node_id", "label", "reason")} for x in result["excluded_transient"]],
        "nodes_without_energy": len(result["nodes_without_energy"]),
        "analysis_level": result["analysis_level"],
        "notes": result["notes"],
        "warnings": result["warnings"],
        "files": result["files"],
    }
    return out, _text(out, _window(args.pH, "pH"), _window(args.potential, "potential"), args.scale)


def _text(out, ph, potential, scale):
    """Readable summary of a Pourbaix result (phases, 1-D transitions, what was left out)."""
    from pathlib import Path

    def dagger(flag, where="region"):
        if not flag:
            return ""
        if flag == "throughout":
            return " (gives the product off downhill)"
        if flag == "in part":
            return f" (gives the product off downhill in part of its {where})"
        return f" ({flag})"

    lines = [
        f"surface Pourbaix diagram ({out['analysis_level']}): pH {ph[0]:g} to {ph[1]:g}, "
        f"U {potential[0]:g} to {potential[1]:g} V vs {scale}"
    ]
    lines.append("  most stable terminations, share of the window:")
    for p in out["phases"]:
        lines.append(f"    {p['label']:<24s} {100 * p['area_fraction']:5.1f}%{dagger(p['releases_product_downhill'])}")
    by_ph = {}
    for t in out["transitions"]:
        by_ph.setdefault(t["pH"], []).append(t)
    for value, segments in sorted(by_ph.items()):
        segments = sorted(segments, key=lambda t: t["from_V"])
        if len(segments) == 1:
            only = segments[0]
            text = f"{only['label']} throughout{dagger(only['releases_product_downhill'], 'range')}"
        else:
            parts = [
                f"{t['label']} up to {t['to_V']:.2f} V{dagger(t['releases_product_downhill'], 'range')}"
                for t in segments[:-1]
            ]
            last = segments[-1]
            parts.append(
                f"{last['label']} above {last['from_V']:.2f} V{dagger(last['releases_product_downhill'], 'range')}"
            )
            text = ", ".join(parts)
        lines.append(f"  at pH {value:g}: {text}")
    for x in out["left_out"]:
        lines.append(f"  left out: {x['label']}: {x['reason']}")
    if out["left_out"]:
        lines.append("    (--keep-transient shows them)")
    if out["nodes_without_energy"]:
        lines.append(f"  intermediates without an energy (not in the diagram): {out['nodes_without_energy']}")
    # the left-out terminations and the downhill flags are already on their own lines
    lines += [f"  note: {n}" for n in out["notes"] if not n.startswith(("Left out", "Releases the product downhill"))]
    lines += [f"  warning: {w}" for w in out["warnings"]]
    if out["files"]:
        folder = Path(out["files"][0]).parent
        lines.append(f"  files in {folder}: " + ", ".join(Path(f).name for f in out["files"]))
    return "\n".join(lines)
