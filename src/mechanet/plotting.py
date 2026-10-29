"""Figures: free-energy profile of one cycle and the surface Pourbaix diagram (matplotlib).

Both figures carry their numbers in a table next to them (analysis/cycles.csv,
pourbaix/pourbaix.json), so nothing is read from colour alone.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
AXIS = "#c3c2b7"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SERIES_1 = "#2a78d6"
# Moments panel: five categorical slots that stay apart pairwise (aqua, yellow, green, violet, red;
# all-pairs normal-vision dE >= 15, colour-blind dE >= 6.9 with the role markers as second encoding).
# Slot 1 (blue) is the energy profile. At most five atoms are drawn; the tables keep every one.
MOMENT_COLORS = ("#1baf7a", "#eda100", "#008300", "#6250d6", "#e34948")
ROLE_MARKERS = {"site metal": "o", "bound": "s", "outer": "^", "flagged": "D", "largest change": "D"}
WARNING = "#fab219"


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


def plot_profile(campaign, cycle_id, potential_V, output_path, scale=None, moments=True):
    """Cumulative free energy along one cycle at one potential (from analysis/thermodynamics.json).

    When the analysis has local moments for the cycle (thermodynamics.json ``magnetism``) and
    ``moments`` is true, a second panel shows the moments of the active-site atoms of every
    intermediate and shades the steps the spin check flags.
    """
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
        [_short(n, release) for n in nodes] + [f"{_short(nodes[0], release)}\n+ {release}"]
        if len(nodes) == len(y) - 1
        else [str(i) for i in range(len(y))]
    )
    magnetism = result.get("magnetism") or {}
    panel = (magnetism.get("cycles") or {}).get(cycle_id) if moments else None
    if panel is not None and (not panel.get("atoms") or panel.get("node_ids") != nodes or len(nodes) != len(y) - 1):
        panel = None
    rest = chosen.get("resting_state_index")
    width = max(7.0, 1.15 * len(y))
    if panel:
        fig, (ax, bx) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(width, 8.2),
            gridspec_kw={"height_ratios": [3, 2]},
        )
    else:
        fig, ax = plt.subplots(figsize=(width, 4.6))
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
    axis = bx if panel else ax
    axis.set_xticks(range(len(y)), labels, fontsize=8, color=INK_2)
    if panel:
        ax.tick_params(labelbottom=False)
    scale = scale or chosen.get("scale", "RHE")
    ax.set_ylabel("Cumulative free energy (eV)")
    descriptor = [f"U = {float(potential_V):g} V vs {scale}"]
    if chosen.get("largest_step_deltaG_eV") is not None:
        descriptor.append(f"largest step {chosen['largest_step_deltaG_eV']:.2f} eV")
    if chosen.get("thermodynamic_overpotential_V") is not None:
        descriptor.append(f"eta_TD {chosen['thermodynamic_overpotential_V']:.2f} V")
    ax.set_title(f"{cycle_id}\n" + " | ".join(descriptor), fontsize=9.5, loc="left")
    notes = ["Computational hydrogen electrode profile; dotted connectors are not barriers."]
    if panel:
        notes += _moments_panel(bx, panel, len(y), magnetism)
    fig_width, fig_height = fig.get_size_inches()
    size = 7.5 if panel else 8.0
    wrapped = [line for note in notes for line in textwrap.wrap(note, width=int(fig_width * 15))]
    fig.text(0.01, 0.01, "\n".join(wrapped), fontsize=size, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, (len(wrapped) * size * 1.3 / 72 + 0.1) / fig_height, 1, 1), h_pad=0.6)
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
        "moments_panel": bool(panel),
        "spin_check": (panel or {}).get("status")
        or ((magnetism.get("cycles") or {}).get(cycle_id) or {}).get("status", "not_available"),
    }


def _moments_panel(ax, panel, n_points, magnetism):
    """Local moments of the tracked atoms per intermediate; returns the footnote lines."""
    from .magnetism import flag_text, short_label

    _clean(ax)
    ax.axhline(0.0, color=GRID, lw=0.8, zorder=0)
    steps = sorted({flag["step"] for flag in panel["flags"]})
    for step in steps:  # step s goes from position s - 1 to position s
        ax.axvspan(step - 1 + 0.08, step - 0.08, color=WARNING, alpha=0.22, lw=0, zorder=0)
        ax.text(
            step - 0.5,
            0.97,
            "spin flag",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.5,
            color=INK_2,
        )
    xs = list(range(n_points))
    drawn = panel["atoms"][: len(MOMENT_COLORS)]
    for k, atom in enumerate(drawn):
        values = list(atom["moments_muB"]) + [atom["moments_muB"][0]]  # the cycle returns to its start
        ax.plot(
            xs,
            [float("nan") if v is None else v for v in values],
            color=MOMENT_COLORS[k],
            lw=1.6,
            marker=ROLE_MARKERS.get(atom["role"], "o"),
            ms=6.0,
            mec=SURFACE,
            mew=0.9,
            zorder=3,
            label=f"{atom['element']} {atom['atom_id']} ({atom['role']})",
        )
    ax.set_ylabel(r"Local moment ($\mu_\mathrm{B}$)")
    if steps:  # room above the data for the step labels
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + 0.16 * (high - low))
    handles, names = ax.get_legend_handles_labels()
    columns = min(len(handles), 3)
    rows = -(-len(handles) // columns)
    # matplotlib fills legend columns first: reorder so that the entries read row by row
    order = [r * columns + c for c in range(columns) for r in range(rows) if r * columns + c < len(handles)]
    ax.legend(
        [handles[i] for i in order],
        [names[i] for i in order],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=columns,
        frameon=False,
        fontsize=8,
        labelcolor=INK_2,
        handlelength=2.4,
        columnspacing=1.6,
    )
    settings = magnetism.get("settings") or {}
    notes = [
        f"Lower panel: projected local moments ({magnetism.get('stage', '')}) of the structure each energy comes "
        "from; the last column repeats the first intermediate."
        + (
            f" {len(panel['atoms']) - len(drawn)} more tracked atoms are in analysis/cycle_moments.csv."
            if len(panel["atoms"]) > len(drawn)
            else ""
        )
    ]
    status = panel["status"]
    if status == "needs_review":
        shown = "; ".join(flag_text(flag, short_label) for flag in panel["flags"][:2])
        more = len(panel["flags"]) - 2
        notes.append(
            "Spin check: needs review. "
            + shown
            + (f" (+{more} more; `mechanet show-cycle` lists them)" if more > 0 else "")
        )
    elif status == "incomplete":
        missing = ", ".join(short_label(m["node_id"]) for m in panel.get("missing", []))
        notes.append(f"Spin check incomplete: no moments for {missing or 'some intermediates'}.")
    else:
        notes.append(
            f"Spin check: consistent (no moment of {settings.get('moment_change_muB', 0.5):g} muB or more changes "
            f"sign; none farther than {settings.get('site_radius_A', 3.6):g} A from the site metals changes by more)."
        )
    return [note.replace("muB", "\u03bcB").replace(" -> ", " \u2192 ") for note in notes]


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
