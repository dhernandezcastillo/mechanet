"""Figures: free-energy profile of one cycle and the surface Pourbaix diagram (matplotlib).

Both figures carry their numbers in a table next to them (analysis/cycles.csv,
pourbaix/pourbaix.json), so nothing is read from colour alone.
"""

from __future__ import annotations

import json
from pathlib import Path

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
AXIS = "#c3c2b7"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SERIES_1 = "#2a78d6"


def _pyplot():
    try:
        import matplotlib
    except ImportError:
        raise RuntimeError("Figures need matplotlib: pip install 'mechanet[plots]'") from None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.titlecolor": INK,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
        }
    )
    return plt


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8)


def _short(node_id, release):
    if node_id == "ref":
        return "native"
    core = node_id[2:] if node_id.startswith("n_") else node_id
    core = core.rsplit("_", 1)[0] if "_" in core else core
    return core.replace("_", "\n")


def plot_profile(campaign, cycle_id, potential_V, output_path, scale=None):
    """Cumulative free energy along one cycle at one potential (from analysis/thermodynamics.json)."""
    plt = _pyplot()
    root = Path(campaign)
    path = root / "analysis" / "thermodynamics.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `mechanet thermo {campaign}` first")
    result = json.loads(path.read_text())
    rows = [x for x in result["cycles"] if x["cycle_id"] == cycle_id]
    if not rows:
        raise ValueError(
            f"Cycle {cycle_id} is not in {path}; if it is a discovered cycle, adopt it with "
            "`mechanet pathways --adopt ID` and run thermo again"
        )
    chosen = next((x for x in rows if abs(x["potential_V"] - float(potential_V)) < 1e-7), None)
    if chosen is None:
        raise ValueError(
            f"{cycle_id} was analysed at {sorted({x['potential_V'] for x in rows})} V; run thermo "
            f"with --potentials including {potential_V}"
        )
    if chosen["status"] != "available":
        raise ValueError(
            "This cycle has missing or rejected results; no profile is drawn from incomplete data "
            f"(missing edges: {chosen.get('missing_edges')})"
        )
    y = chosen["cumulative_deltaG_eV"]
    nodes = chosen.get("node_ids") or []
    reaction = chosen.get("reaction", "oer")
    release = {"oer": "O2", "cer": "Cl2"}.get(reaction, "product")
    labels = (
        [_short(n, release) for n in nodes] + [f"native\n+ {release}"]
        if len(nodes) == len(y) - 1
        else [str(i) for i in range(len(y))]
    )
    rest = chosen.get("resting_state_index")
    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * len(y)), 4.6))
    _clean(ax)
    ax.axhline(0.0, color=GRID, lw=0.8, zorder=0)
    for i, value in enumerate(y):
        ax.plot(
            [i - 0.28, i + 0.28],
            [value, value],
            lw=2.6 if i == rest else 2.0,
            color=SERIES_1,
            solid_capstyle="round",
            zorder=3,
        )
        if i + 1 < len(y):
            ax.plot([i + 0.28, i + 0.72], [value, y[i + 1]], lw=0.9, ls=(0, (2, 2)), color=MUTED, zorder=2)
    span = chosen.get("Gmax_U_span_indices")
    if span:
        i, j = span
        x = j + 0.36
        ax.annotate(
            "",
            xy=(x, y[j]),
            xytext=(x, y[i]),
            arrowprops={"arrowstyle": "<->", "color": INK_2, "lw": 0.9, "shrinkA": 0, "shrinkB": 0},
        )
        ax.text(
            x + 0.06,
            0.5 * (y[i] + y[j]),
            f"Gmax(U)\n{chosen['Gmax_U_eV']:.2f} eV",
            color=INK_2,
            fontsize=8.5,
            va="center",
        )
    if rest is not None:
        ax.text(
            rest,
            y[rest] + 0.025 * (max(y) - min(y) or 1.0),
            "lowest",
            color=INK_2,
            fontsize=8,
            ha="center",
            va="bottom",
        )
    ax.set_xticks(range(len(y)), labels, fontsize=8, color=INK_2)
    scale = scale or chosen.get("scale", "RHE")
    ax.set_ylabel("Cumulative free energy (eV)")
    descriptor = [f"U = {float(potential_V):g} V vs {scale}"]
    if chosen.get("largest_step_deltaG_eV") is not None:
        descriptor.append(f"largest step {chosen['largest_step_deltaG_eV']:.2f} eV")
    if chosen.get("thermodynamic_overpotential_V") is not None:
        descriptor.append(f"eta_TD {chosen['thermodynamic_overpotential_V']:.2f} V")
    ax.set_title(f"{cycle_id}\n" + " | ".join(descriptor), fontsize=9.5, loc="left")
    fig.text(
        0.01,
        0.01,
        "Computational hydrogen electrode profile; dotted connectors are not barriers.",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    return {
        "figure": str(out.resolve()),
        "vector": str(out.with_suffix(".svg").resolve()),
        "cycle": cycle_id,
        "potential_V": float(potential_V),
        "scale": scale,
    }


def plot_pourbaix(result, output_path):
    """Exact phase polygons of a Pourbaix result, labelled in place (numbered when too small)."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    _clean(ax)
    (p_lo, p_hi), (u_lo, u_hi) = result["pH_range"], result["potential_range_V"]
    footnotes = []

    def on_frame(p, q):
        return any(
            abs(p[i] - v) < 1e-9 and abs(q[i] - v) < 1e-9 for i, v in ((0, p_lo), (0, p_hi), (1, u_lo), (1, u_hi))
        )

    for k, phase in enumerate(result["phases"], start=1):
        polygon = phase["polygon"]
        xs = [p[0] for p in polygon] + [polygon[0][0]]
        ys = [p[1] for p in polygon] + [polygon[0][1]]
        for p, q in zip(polygon, polygon[1:] + polygon[:1]):
            if not on_frame(p, q):  # boundaries between phases only; the axes frame the window
                ax.plot([p[0], q[0]], [p[1], q[1]], color=INK_2, lw=1.0, zorder=3, solid_capstyle="round")
        cx, cy = phase["centroid"]
        width = (max(xs) - min(xs)) / (p_hi - p_lo)
        height = (max(ys) - min(ys)) / (u_hi - u_lo)
        label = phase["label"] + (" †" if phase.get("releases_product_downhill") else "")
        if width > 0.18 and height > 0.07 and len(label) <= 40:
            ax.text(
                cx,
                cy,
                label.replace(" ", "\n") if width < 0.4 else label,
                ha="center",
                va="center",
                fontsize=8,
                color=INK,
                zorder=4,
            )
        else:
            ax.text(
                cx,
                cy,
                str(k),
                ha="center",
                va="center",
                fontsize=8,
                color=INK,
                zorder=4,
                bbox={"boxstyle": "circle,pad=0.2", "fc": SURFACE, "ec": INK_2, "lw": 0.6},
            )
            footnotes.append(f"{k}: {label}")
    nernst = result["conditions"].get("nernst_V") or 0.0591597
    if result["scale"] == "SHE":
        lines = [(1.229, -nernst, "O2/H2O"), (0.0, -nernst, "H+/H2")]
    else:
        lines = [(1.229, 0.0, "O2/H2O"), (0.0, 0.0, "H+/H2")]
    for intercept, slope, name in lines:
        # the part of U = intercept + slope * pH inside the window
        xs = [p_lo, p_hi]
        if slope:
            xs += [(u - intercept) / slope for u in (u_lo, u_hi)]
        xs = sorted(
            x for x in xs if p_lo - 1e-9 <= x <= p_hi + 1e-9 and u_lo - 1e-9 <= intercept + slope * x <= u_hi + 1e-9
        )
        if len(xs) < 2:
            continue
        x0, x1 = xs[0], xs[-1]
        ax.plot(
            [x0, x1], [intercept + slope * x0, intercept + slope * x1], color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2
        )
        ax.annotate(
            name,
            (x1, intercept + slope * x1),
            xytext=(-4, 5),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color=MUTED,
            fontsize=8,
            bbox={"boxstyle": "square,pad=0.15", "fc": SURFACE, "ec": "none"},
        )
    ax.set_xlim(p_lo, p_hi)
    ax.set_ylim(u_lo, u_hi)
    ax.set_xlabel("pH")
    ax.set_ylabel(f"U (V vs {result['scale']})")
    ax.set_title(
        f"Most stable computed termination ({result['reaction'].upper()} network, "
        f"{result['analysis_level'].split('_')[0]})",
        fontsize=9.5,
        loc="left",
    )
    notes = ["   ".join(footnotes)] if footnotes else []
    if any(p.get("releases_product_downhill") for p in result["phases"]):
        notes.append("† gives the product off downhill in (part of) this region: an intermediate under turnover")
    left_out = [x["label"] for x in result.get("excluded_transient") or []]
    if left_out:
        notes.append("Not shown (bound product given off downhill at every U and pH): " + ", ".join(left_out))
    if notes:
        fig.text(0.01, 0.01, "\n".join(notes), fontsize=7.5, color=INK_2, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.035 * len(notes) + 0.02 if notes else 0.0, 0.93, 1))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".svg"))
    plt.close(fig)
    return str(out.resolve())
