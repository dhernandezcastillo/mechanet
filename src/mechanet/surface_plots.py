"""Figures of the surface analysis (matplotlib): which surfaces are stable, and how close the others are.

Five views of the formation free energies DeltaG_i(U) = a_i + b_i U of the surfaces of a
campaign at one pH (see :mod:`mechanet.surfaces`):

* :func:`plot_lines`: the lines, the stable surface along the potential, and the distance of
  every surface above it;
* :func:`plot_hull`: the same surfaces as points over the electrons they release, with the
  lower convex hull whose corners are the stable surfaces;
* :func:`plot_margins`: per composition, how far its lowest surface lies above the stable one;
* :func:`plot_levels`: the stable surfaces with free energies, with tabulated corrections and
  with energies alone;
* :func:`plot_isomers`: the surfaces of each composition above the lowest one.

Every number drawn is also in a table (candidates.csv, compositions.csv, pourbaix.json), and a
surface is named next to its colour: nothing is read from colour alone. A surface keeps its
colour in every figure of one run.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .plotting import AXIS, GRID, INK, INK_2, MUTED, SURFACE, _clean, _pyplot

# Categorical slots for the stable surfaces, in this fixed order (separated for normal and
# colour-deficient vision as neighbours on the light surface). A surface beyond the last slot
# is drawn in neutral ink and told apart by a number.
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
NEUTRAL = INK_2
CONTEXT = AXIS
WINDOW_WASH = 0.06
STRIP_HEIGHT_IN = 0.30
NOTE_BAND_IN = 0.42
# Most rows a ranked figure draws; the tables keep every one.
MAX_ROWS = 28


def surface_colours(node_ids):
    """{node ID: colour}: the first surfaces take the categorical slots in order, the rest neutral ink."""
    out = {}
    for node_id in node_ids:
        if node_id not in out:
            out[node_id] = SLOTS[len(out)] if len(out) < len(SLOTS) else NEUTRAL
    return out


def _name(node_id, surface_names, labels=None, limit=46):
    text = (surface_names or {}).get(node_id) or (labels or {}).get(node_id) or node_id
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _ink_on(colour):
    """White or near-black text for a label inside a fill of this colour."""
    r, g, b = (int(colour[k : k + 2], 16) / 255 for k in (1, 3, 5))

    def linear(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    luminance = 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b)
    return INK if luminance > 0.36 else "#ffffff"


def _save(fig, path):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".svg"))
    return str(out.resolve())


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _strip(ax, fig, segments, colours, texts, x_range, footnotes, window=None):
    """One row of stable segments: a block per surface with a surface-coloured gap between them.

    The name of a surface is written in its block where it fits; the legend of the figure
    names every colour. A surface in neutral ink (beyond the colour slots) gets a number, and
    its name goes to ``footnotes``. The potentials where the surface changes are written above
    the row (one that would run into the one before it is left to the table), and the ends of
    the window are marked below it.
    """
    lo, hi = x_range
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.axis("off")
    renderer = fig.canvas.get_renderer()
    for segment in segments:
        u_from, u_to = max(segment["from_V"], lo), min(segment["to_V"], hi)
        if u_to <= u_from:
            continue
        colour = colours.get(segment["node_id"], NEUTRAL)
        ax.barh(0.5, u_to - u_from, left=u_from, height=1.0, color=colour, edgecolor=SURFACE, linewidth=1.6)
        name = texts[segment["node_id"]]
        room = ax.transData.transform((u_to, 0))[0] - ax.transData.transform((u_from, 0))[0]
        label = ax.text(
            (u_from + u_to) / 2, 0.5, name, ha="center", va="center", fontsize=7.5, color=_ink_on(colour), zorder=4
        )
        if label.get_window_extent(renderer).width + 10 > room:
            if colour != NEUTRAL:
                label.set_visible(False)
                continue
            if name not in footnotes:
                footnotes.append(name)
            label.set_text(str(footnotes.index(name) + 1))
            if label.get_window_extent(renderer).width + 6 > room:
                label.set_visible(False)
    taken = -math.inf
    for segment in segments[:-1]:
        if not lo < segment["to_V"] < hi:
            continue
        label = ax.text(
            segment["to_V"],
            1.08,
            f"{segment['to_V']:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=INK_2,
            clip_on=False,
        )
        box = label.get_window_extent(renderer)
        if box.x0 < taken + 4:
            label.set_visible(False)
        else:
            taken = box.x1
    for edge in window or ():
        if lo <= edge <= hi:
            ax.plot([edge, edge], [-0.32, -0.06], color=INK, linewidth=0.9, clip_on=False, zorder=6)


def _legend_rows(node_ids, colours):
    named = [n for n in dict.fromkeys(node_ids) if colours.get(n, NEUTRAL) != NEUTRAL]
    return named, math.ceil(len(named) / 2)


def _surface_legend(fig, node_ids, colours, texts, bottom_in):
    """The legend of the surfaces that have a colour: a swatch and the name, two per row."""
    from matplotlib.patches import Patch

    named, _ = _legend_rows(node_ids, colours)
    if not named:
        return
    fig.legend(
        handles=[Patch(facecolor=colours[n], edgecolor="none", label=texts[n]) for n in named],
        loc="lower left",
        bbox_to_anchor=(0.004, bottom_in / fig.get_figheight()),
        ncol=2,
        fontsize=7.5,
        frameon=False,
        labelcolor=INK_2,
        handlelength=1.0,
        handleheight=0.9,
        columnspacing=1.6,
        borderaxespad=0.2,
    )


def _fit_rows(fig, ax, top_in, bottom_in):
    """Margins of a figure with one row per name on the left: room for the longest name."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widest = max((label.get_window_extent(renderer).width for label in ax.get_yticklabels()), default=0.0)
    width, height = fig.get_size_inches()
    left = min(0.62, (widest / fig.dpi + 0.22) / width)
    fig.subplots_adjust(left=left, right=0.97, top=1 - top_in / height, bottom=bottom_in / height)


def _notes(fig, footnotes, extra=()):
    notes = [f"{k}: {name}" for k, name in enumerate(footnotes, start=1)] + list(extra)
    if notes:
        fig.text(0.012, 0.012, "   ".join(notes), fontsize=7.5, color=INK_2, va="bottom", wrap=True)


def plot_lines(
    path,
    lines,
    a,
    b,
    potential_range,
    scale,
    pH,
    window=None,
    margin=None,
    candidate_ids=(),
    surface_names=None,
    colours=None,
    level=None,
):
    """Formation free energies along the potential, the stable surface, and the distance above it.

    Top: the stable surface at each potential, with the potentials where it changes. Middle:
    DeltaG_f(U) of every surface; the stable ones in colour, the other candidates in dark
    ink, the rest in light grey. Bottom: how far each surface lies above the stable one, with
    the margin of the candidates as a line; a surface is stable where its curve touches zero.
    """
    from .pourbaix import lower_envelope

    plt = _pyplot()
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    lo, hi = (float(x) for x in potential_range)
    ids = [line["node_id"] for line in lines]
    labels = {line["node_id"]: line["label"] for line in lines}
    segments = [
        {"from_V": u_from, "to_V": u_to, "node_id": ids[index]} for u_from, u_to, index in lower_envelope(a, b, lo, hi)
    ]
    stable = [s["node_id"] for s in segments]
    colours = colours or surface_colours(stable)
    texts = {node_id: _name(node_id, surface_names, labels) for node_id in ids}
    candidate_ids = set(candidate_ids) - set(stable)
    breaks = {lo, hi, *(s["to_V"] for s in segments)}
    grid = np.array(sorted(set(np.linspace(lo, hi, 400).tolist()) | breaks))
    values = a[:, None] + b[:, None] * grid[None, :]
    envelope = values.min(axis=0)
    above = values - envelope[None, :]
    span = max(float(envelope.max() - envelope.min()), 0.5)
    top_limit = max(1.0, 5.0 * margin) if margin else 1.0

    legend_in = 0.19 * _legend_rows(stable, colours)[1]
    height = 6.4 + legend_in
    bottom_in = NOTE_BAND_IN + legend_in + 0.52
    fig = plt.figure(figsize=(7.2, height))
    grid_spec = fig.add_gridspec(
        3,
        1,
        height_ratios=[STRIP_HEIGHT_IN, 3.0, 2.2],
        hspace=0.16,
        left=0.11,
        right=0.97,
        top=1 - 0.62 / height,
        bottom=bottom_in / height,
    )
    strip, main, zoom = (fig.add_subplot(grid_spec[k]) for k in range(3))
    fig.canvas.draw()
    footnotes = []
    _strip(strip, fig, segments, colours, texts, (lo, hi), footnotes)
    for ax in (main, zoom):
        _clean(ax)
        _grid(ax)
        if window:
            ax.axvspan(window[0], window[1], color=INK, alpha=WINDOW_WASH, linewidth=0)
        ax.set_xlim(lo, hi)

    def draw(ax, rows):
        for i, node_id in enumerate(ids):
            if node_id not in stable and node_id not in candidate_ids:
                ax.plot(grid, rows[i], color=CONTEXT, linewidth=0.5, zorder=1)
        for i, node_id in enumerate(ids):
            if node_id in candidate_ids:
                ax.plot(grid, rows[i], color=NEUTRAL, linewidth=0.7, zorder=2)
        for i, node_id in enumerate(ids):
            if node_id in stable:
                ax.plot(
                    grid, rows[i], color=colours.get(node_id, NEUTRAL), linewidth=1.6, zorder=3, solid_capstyle="round"
                )

    draw(main, values)
    main.set_ylim(float(envelope.min()) - 0.08 * span, float(envelope.max()) + 0.45 * span)
    main.set_ylabel("ΔG$_f$ (eV)")
    main.tick_params(labelbottom=False)
    draw(zoom, above)
    zoom.set_ylim(0, top_limit)
    zoom.set_ylabel("above the stable\nsurface (eV)")
    zoom.set_xlabel(f"U (V vs {scale})")
    if margin:
        zoom.axhline(margin, color=INK, linewidth=0.8, zorder=2)
        zoom.text(
            lo,
            margin,
            f" margin {margin:g} eV",
            ha="left",
            va="bottom",
            fontsize=7.5,
            color=INK_2,
            zorder=5,
            bbox={"boxstyle": "square,pad=0.1", "fc": SURFACE, "ec": "none"},
        )
    title = f"Stable surface along the potential (pH {pH:g}" + (f", {level}" if level else "") + ")"
    fig.text(0.012, 1 - 0.1 / height, title, fontsize=9.5, color=INK, va="top")
    key = [f"{len(ids)} surfaces", "colour: stable somewhere in the range drawn, grey: the others"]
    if candidate_ids:
        key.append(f"dark: {len(candidate_ids)} more within the margin in the window")
    if window:
        key.append(f"shaded: window {window[0]:g} to {window[1]:g} V")
    _surface_legend(fig, stable, colours, texts, NOTE_BAND_IN)
    _notes(fig, footnotes, ["; ".join(key)])
    out = _save(fig, path)
    plt.close(fig)
    return out


def plot_hull(path, hull, scale, pH, surface_names=None, labels=None, colours=None, level=None):
    """The surfaces as points over the electrons they release, at the reference potential.

    The lower convex hull joins the surfaces that are stable at some potential; the number
    under each of its edges is the potential at which the two surfaces at its ends are equally
    stable. A point just above the hull is a surface that is almost stable.
    """
    plt = _pyplot()
    points, vertices = hull["points"], hull["vertices"]
    colours = colours or surface_colours([v["node_id"] for v in vertices])
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    _clean(ax)
    _grid(ax)
    others = [p for p in points if not p["lowest_for_its_electrons"]]
    lowest = [p for p in points if p["lowest_for_its_electrons"] and not p["on_hull"]]
    ax.scatter(
        [p["electrons"] for p in others], [p["deltaG_eV"] for p in others], s=14, color=CONTEXT, linewidths=0, zorder=2
    )
    ax.scatter(
        [p["electrons"] for p in lowest],
        [p["deltaG_eV"] for p in lowest],
        s=30,
        color=NEUTRAL,
        edgecolors=SURFACE,
        linewidths=1.0,
        zorder=3,
    )
    xs, ys = [v["electrons"] for v in vertices], [v["deltaG_eV"] for v in vertices]
    ax.plot(xs, ys, color=INK_2, linewidth=1.2, zorder=3, solid_capstyle="round")
    y_span = (max(p["deltaG_eV"] for p in points) - min(p["deltaG_eV"] for p in points)) or 1.0
    footnotes = []
    for k, vertex in enumerate(vertices):
        footnotes.append(_name(vertex["node_id"], surface_names, labels))
        ax.scatter(
            [vertex["electrons"]],
            [vertex["deltaG_eV"]],
            s=62,
            color=colours.get(vertex["node_id"], NEUTRAL),
            edgecolors=SURFACE,
            linewidths=1.4,
            zorder=5,
        )
        ax.annotate(
            str(k + 1),
            (vertex["electrons"], vertex["deltaG_eV"]),
            xytext=(0, -9),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color=INK,
            zorder=6,
        )
        if vertex["to_next_V"] is not None:
            nxt = vertices[k + 1]
            ax.annotate(
                f"{vertex['to_next_V']:.2f} V",
                ((vertex["electrons"] + nxt["electrons"]) / 2, (vertex["deltaG_eV"] + nxt["deltaG_eV"]) / 2),
                xytext=(0, -7),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=7.5,
                color=INK_2,
                zorder=6,
            )
    ax.set_ylim(min(ys) - 0.16 * y_span, max(p["deltaG_eV"] for p in points) + 0.06 * y_span)
    ax.set_xlabel("electrons released on forming the surface")
    ax.set_ylabel(f"ΔG$_f$ at {hull['reference_V']:g} V vs {scale} (eV)")
    whole = sorted({round(p["electrons"]) for p in points if abs(p["electrons"] - round(p["electrons"])) < 1e-6})
    if whole and len(whole) <= 16:
        ax.set_xticks(whole)
    title = f"Surfaces over the electrons they release (pH {pH:g}" + (f", {level}" if level else "") + ")"
    ax.set_title(title, fontsize=9.5, loc="left")
    _notes(
        fig,
        footnotes,
        [
            "line: lower convex hull, its corners (numbered) are the stable surfaces; under each edge the "
            "potential of the transition; dark dots: lowest surface for its electrons; grey: the others"
        ],
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    out = _save(fig, path)
    plt.close(fig)
    return out


def plot_margins(path, groups, potentials, margin, scale, pH, level=None):
    """Per composition, how far its lowest surface lies above the stable surface.

    One row per composition, the closest first; one mark per potential, one under the other.
    A composition left of the margin line at any of the potentials is a candidate. At most
    MAX_ROWS rows are drawn: compositions.csv has all of them.
    """
    plt = _pyplot()
    keys = [f"{float(u):g}" for u in potentials]
    rows = sorted(groups, key=lambda g: (min(g["above_stable_eV"][k] for k in keys), g["electrons"], g["composition"]))
    hidden = max(0, len(rows) - MAX_ROWS)
    rows = rows[:MAX_ROWS]
    limit = max(1.0, 5.0 * margin)
    fig, ax = plt.subplots(figsize=(7.2, 1.6 + 0.27 * len(rows)))
    _clean(ax)
    _grid(ax, "x")
    markers = ("o", "s", "^", "D")
    step = 0.5 / max(len(keys), 1)
    for row, group in enumerate(rows):
        values = [group["above_stable_eV"][k] for k in keys]
        ax.plot([min(values), min(max(values), limit)], [row, row], color=CONTEXT, linewidth=1.0, zorder=2)
        for k, value in enumerate(values):
            if value > limit:
                continue
            ax.scatter(
                [value],
                [row + (k - (len(keys) - 1) / 2) * step],  # one mark under the other: equal values stay visible
                s=34,
                marker=markers[k % len(markers)],
                color=SLOTS[k % 3],
                edgecolors=SURFACE,
                linewidths=1.2,
                zorder=4,
            )
        if min(values) > limit:
            ax.text(limit, row, f"{min(values):.2f} ›", ha="right", va="center", fontsize=7.5, color=MUTED)
    ax.axvline(margin, color=INK, linewidth=0.8, zorder=3)
    ax.text(margin, -0.9, f" margin {margin:g} eV", ha="left", va="center", fontsize=7.5, color=INK_2)
    ax.set_xlim(-0.02 * limit, limit)
    ax.set_ylim(len(rows) - 0.4, -1.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [
            (g["lowest_surface"] or g["composition"]) + (f"  ({g['n_surfaces']})" if g["n_surfaces"] > 1 else "")
            for g in rows
        ],
        fontsize=7.5,
        color=INK,
    )
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("lowest surface of the composition above the stable surface (eV)")
    for k, key in enumerate(keys):
        ax.scatter([], [], s=34, marker=markers[k % len(markers)], color=SLOTS[k % 3], label=f"{key} V vs {scale}")
    fig.legend(
        loc="upper right",
        bbox_to_anchor=(0.985, 1 - 0.3 / fig.get_figheight()),
        ncol=len(keys),
        fontsize=7.5,
        frameon=False,
        labelcolor=INK_2,
        handletextpad=0.2,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    fig.text(
        0.012,
        1 - 0.1 / fig.get_figheight(),
        f"Compositions above the stable surface (pH {pH:g})",
        fontsize=9.5,
        color=INK,
        va="top",
    )
    notes = ([level] if level else []) + ["in brackets: surfaces computed with that composition"]
    if hidden:
        notes.append(f"{hidden} compositions further away are not drawn (compositions.csv)")
    fig.text(0.012, 0.012, "; ".join(notes), fontsize=7.5, color=INK_2, va="bottom", wrap=True)
    _fit_rows(fig, ax, 0.62, 0.78)
    out = _save(fig, path)
    plt.close(fig)
    return out


def plot_levels(path, levels, potential_range, scale, pH, window=None, surface_names=None, colours=None):
    """The stable surface along the potential at each level of the energies, one row per level.

    A level that cannot be evaluated (no frequency calculation yet) is said in its row. The
    rows share the potential axis, and a surface has the same colour in all of them.
    """
    plt = _pyplot()
    lo, hi = (float(x) for x in potential_range)
    order = []
    for level in levels:
        order += [s["node_id"] for s in level.get("segments", [])]
    colours = dict(colours or {})
    for node_id in dict.fromkeys(order):
        if node_id not in colours:
            free = [slot for slot in SLOTS if slot not in colours.values()]
            colours[node_id] = free[0] if free else NEUTRAL
    texts = {}
    for level in levels:
        for segment in level.get("segments", []):
            texts[segment["node_id"]] = _name(segment["node_id"], surface_names, {segment["node_id"]: segment["label"]})
    n = len(levels)
    legend_in = 0.19 * _legend_rows(order, colours)[1]
    row_in, gap_in = STRIP_HEIGHT_IN, 0.5
    axis_in = NOTE_BAND_IN + legend_in + 0.5
    height = axis_in + 0.2 + n * row_in + (n - 1) * gap_in + 0.95
    fig = plt.figure(figsize=(7.2, height))
    grid_spec = fig.add_gridspec(
        n,
        1,
        hspace=gap_in / row_in,
        left=0.03,
        right=0.97,
        top=1 - 0.95 / height,
        bottom=(axis_in + 0.2) / height,
    )
    fig.canvas.draw()
    footnotes = []
    for k, level in enumerate(levels):
        ax = fig.add_subplot(grid_spec[k])
        count = f" ({level['n_surfaces']} surfaces)" if "n_surfaces" in level else ""
        ax.set_title(level["title"] + count, fontsize=8.5, loc="left", color=INK, pad=13)
        if "segments" in level:
            _strip(ax, fig, level["segments"], colours, texts, (lo, hi), footnotes, window)
        else:
            ax.set_xlim(lo, hi)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.text(lo, 0.5, "not available: " + level["unavailable"][:110], fontsize=7.5, color=MUTED, va="center")
    axis = fig.add_axes([0.03, axis_in / height, 0.94, 0.02 / height])
    axis.set_xlim(lo, hi)
    axis.set_ylim(0, 1)
    for side in ("top", "left", "right"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.set_yticks([])
    axis.tick_params(length=3, width=0.8)
    axis.patch.set_alpha(0)
    axis.set_xlabel(f"U (V vs {scale})")
    fig.text(
        0.012,
        1 - 0.1 / height,
        f"Stable surface at each level of the energies (pH {pH:g})",
        fontsize=9.5,
        color=INK,
        va="top",
    )
    extra = [f"marks under each row: window {window[0]:g} to {window[1]:g} V"] if window else []
    _surface_legend(fig, order, colours, texts, NOTE_BAND_IN)
    _notes(fig, footnotes, extra)
    out = _save(fig, path)
    plt.close(fig)
    return out


def plot_isomers(path, groups, with_free_energy=(), level=None):
    """The surfaces of each composition above the lowest one of that composition.

    One row per composition, ordered by the electrons released; one mark per surface. A row
    whose marks lie close together has several surfaces that need the higher level to be told
    apart. Filled marks have a free energy (a frequency calculation), open ones do not.
    """
    plt = _pyplot()
    rows = [g for g in groups if g["n_surfaces"] > 1]
    hidden = max(0, len(rows) - 2 * MAX_ROWS)
    rows = rows[: 2 * MAX_ROWS]
    with_free_energy = set(with_free_energy)
    fig, ax = plt.subplots(figsize=(7.2, 1.5 + 0.22 * max(len(rows), 1)))
    _clean(ax)
    _grid(ax, "x")
    for row, group in enumerate(rows):
        ax.plot([0, group["spread_eV"]], [row, row], color=CONTEXT, linewidth=1.0, zorder=2)
        for surface in group["surfaces"]:
            filled = surface["node_id"] in with_free_energy
            ax.scatter(
                [surface["above_lowest_eV"]],
                [row],
                s=30,
                facecolors=SLOTS[0] if filled else SURFACE,
                edgecolors=SLOTS[0],
                linewidths=1.1,
                zorder=4 if filled else 3,
            )
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [f"{g['composition']}   {g['electrons']:g} e⁻   ({g['n_surfaces']})" for g in rows], fontsize=7.5, color=INK
    )
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(left=-0.02 * max((g["spread_eV"] for g in rows), default=1.0))
    ax.set_xlabel("above the lowest surface of the composition (eV)")
    title = "Surfaces of one composition" + (f" ({level})" if level else "")
    ax.set_title(title, fontsize=9.5, loc="left")
    notes = ["composition relative to the slab, electrons released, (surfaces)"]
    if with_free_energy:
        notes.append("filled: has a free energy; open: relaxed only")
    single = sum(1 for g in groups if g["n_surfaces"] == 1)
    if single:
        notes.append(f"{single} compositions with one surface are not drawn")
    if hidden:
        notes.append(f"{hidden} more rows in compositions.csv")
    fig.text(0.012, 0.012, "; ".join(notes), fontsize=7.5, color=INK_2, va="bottom", wrap=True)
    _fit_rows(fig, ax, 0.42, 0.85)
    out = _save(fig, path)
    plt.close(fig)
    return out
