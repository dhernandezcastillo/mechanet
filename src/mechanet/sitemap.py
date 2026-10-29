"""Static site map: a self-contained HTML page (plus an SVG) showing the reactive sites of a slab.

It needs no server and no JavaScript library, so it can be copied from a cluster and opened in
any browser. The top view looks down the surface normal onto the active side; the side view
shows the slab along the normal with the fixed region. Hovering an atom shows its ID.
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np

from .elements import covalent_radius
from .fileio import atomic_text
from .structure import slab_normal, unit

COLORS = {
    "H": "#f2f2f2",
    "O": "#e2453c",
    "N": "#3f6fd8",
    "C": "#555555",
    "S": "#e8c547",
    "Cl": "#3fb950",
    "F": "#9ad16a",
    "Br": "#a0522d",
    "I": "#7b3fa0",
    "K": "#8f55c9",
    "Na": "#ab5cf2",
    "Li": "#cc80ff",
    "Mg": "#8aff00",
    "Ca": "#3dff00",
    "Sr": "#00ff27",
    "Ba": "#00c900",
    "Al": "#bfa6a6",
    "Ti": "#bfc2c7",
    "V": "#a6a6ab",
    "Cr": "#8a99c7",
    "Mn": "#9c7ac7",
    "Fe": "#e06633",
    "Co": "#f090a0",
    "Ni": "#50d050",
    "Cu": "#c88033",
    "Zn": "#7d80b0",
    "Ru": "#248f8f",
    "Rh": "#0a7d8c",
    "Pd": "#006985",
    "Ag": "#c0c0c0",
    "Ir": "#175487",
    "Pt": "#d0d0e0",
    "Au": "#ffd123",
    "Mo": "#54b5b5",
    "W": "#2194d6",
    "La": "#70d4ff",
    "Ce": "#ffffc7",
    "Sn": "#668080",
    "Ga": "#c28f8f",
    "In": "#a67573",
    "Zr": "#94e0e0",
    "Nb": "#73c2c9",
    "Ta": "#4da6ff",
    "Hf": "#4dc2ff",
    "Y": "#94ffff",
    "Sc": "#e6e6e6",
}
PAGE_CSS = """
body{font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;color:#1f2933;margin:2rem auto;max-width:1300px;
  padding:0 1rem}
h1{font-size:1.4rem;margin:0 0 .3rem} h2{font-size:1.05rem;margin:1.6rem 0 .5rem}
.meta{color:#52606d} .views{display:flex;flex-wrap:wrap;gap:2rem;align-items:flex-start}
figure{margin:0} figcaption{color:#52606d;font-size:.9rem;margin-top:.3rem}
table{border-collapse:collapse;font-size:.92rem}
td,th{padding:.25rem .8rem;border-bottom:1px solid #e4e7eb;text-align:left}
.dot{display:inline-block;width:.75rem;height:.75rem;border-radius:50%;margin-right:.4rem;vertical-align:-1px}
.legend span{margin-right:1rem} code{background:#f3f4f6;padding:0 .25rem;border-radius:3px}
"""
SITE_COLORS = {
    "terminal": "#0b72b9",
    "bridge": "#d97706",
    "mu3": "#7c3aed",
    "framework": "#7c3aed",
    "open_metal": "#059669",
}


def _frame(structure, axis, sign):
    normal = slab_normal(structure.cell, axis, sign)
    first = [i for i in range(3) if i != axis][0]
    t1 = unit(structure.cell[first] - np.dot(structure.cell[first], normal) * normal)
    t2 = np.cross(normal, t1)
    return normal, t1, t2


def _svg_atoms(points, symbols, ids, scale, offset, fixed=None, faded=None, height=None):
    out = []
    order = np.argsort(points[:, 2])
    for k in order:
        x, y = (points[k, 0] - offset[0]) * scale, (points[k, 1] - offset[1]) * scale
        y = height - y if height is not None else y
        r = max(2.5, 0.40 * covalent_radius(symbols[k]) * scale)
        fill = COLORS.get(symbols[k], "#b0b0b0")
        opacity = 0.25 if faded is not None and faded[k] else 0.5 if fixed is not None and fixed[k] else 1.0
        stroke = "#9aa5b1" if fixed is not None and fixed[k] else "#33404d"
        dash = ' stroke-dasharray="2 2"' if fixed is not None and fixed[k] else ""
        out.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" fill-opacity="{opacity}" '
            f'stroke="{stroke}" stroke-opacity="{opacity}" stroke-width="0.8"{dash}>'
            f"<title>{html.escape(ids[k])} {symbols[k]}</title></circle>"
        )
    return out


def site_map_svg(structure, report, sites=None, fixed=None, width=620):
    """(top view SVG, side view SVG) strings."""
    axis = int(report["proposal"]["vacuum_axis"])
    sign = int(report["proposal"]["surface_sign"])
    sites = report["proposal"]["sites"] if sites is None else sites
    normal, t1, t2 = _frame(structure, axis, sign)
    pos = structure.positions
    depth = pos @ normal
    top = depth[[i for i, s in enumerate(structure.symbols) if s != "H"]].max()
    inplane = [i for i in range(3) if i != axis]
    # Top view: the cell and a margin of periodic images, atoms within 6 A of the top.
    images = []
    for n1 in (-1, 0, 1):
        for n2 in (-1, 0, 1):
            shift = n1 * structure.cell[inplane[0]] + n2 * structure.cell[inplane[1]]
            images.append((n1 == 0 and n2 == 0, shift))
    corners = np.array(
        [
            np.zeros(3),
            structure.cell[inplane[0]],
            structure.cell[inplane[0]] + structure.cell[inplane[1]],
            structure.cell[inplane[1]],
        ]
    )
    cx, cy = corners @ t1, corners @ t2
    margin = 2.5
    lo = np.array([cx.min() - margin, cy.min() - margin])
    hi = np.array([cx.max() + margin, cy.max() + margin])
    scale = width / float(max(hi - lo))
    height = float((hi - lo)[1] * scale)
    pts, syms, ids, faded = [], [], [], []
    for central, shift in images:
        p = pos + shift
        x, y = p @ t1, p @ t2
        keep = (x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1]) & (depth > top - 6.0)
        for i in np.where(keep)[0]:
            pts.append([x[i], y[i], depth[i]])
            syms.append(structure.symbols[i])
            ids.append(structure.ids[i] + ("" if central else "'"))
            faded.append(not central)
    pts = np.array(pts) if pts else np.zeros((0, 3))
    outline = " ".join(f"{(a - lo[0]) * scale:.1f},{height - (b - lo[1]) * scale:.1f}" for a, b in zip(cx, cy))
    body = [f'<polygon points="{outline}" fill="none" stroke="#64748b" stroke-dasharray="5 4"/>']
    body += _svg_atoms(pts, syms, ids, scale, lo, faded=np.array(faded, dtype=bool), height=height)
    index = {aid: i for i, aid in enumerate(structure.ids)}
    open_positions = {s["id"]: np.array(s["position_A"]) for s in report.get("open_metal_sites", [])}
    for site in sites:
        if site["kind"] == "open_metal":
            p = open_positions.get(site["id"])
            if p is None:
                continue
        else:
            p = pos[index[site["id"]]]
        x, y = (p @ t1 - lo[0]) * scale, height - (p @ t2 - lo[1]) * scale
        color = SITE_COLORS.get(site["kind"], "#111827")
        label = html.escape(f"{site['id']} {site['kind']} {site['native']}")
        body.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{0.75 * scale:.1f}" fill="none" stroke="{color}" '
            f'stroke-width="3"><title>{label}</title></circle>'
        )
        body.append(
            f'<text x="{x + 0.8 * scale:.1f}" y="{y - 0.6 * scale:.1f}" font-size="13" '
            f'font-family="ui-monospace,monospace" fill="{color}" font-weight="700" stroke="#ffffff" '
            f'stroke-width="3.5" paint-order="stroke">{html.escape(site["id"])}</text>'
        )
    top_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height:.0f}" width="{width}" '
        f'height="{height:.0f}" role="img" aria-label="Top view of the active surface">' + "".join(body) + "</svg>"
    )
    # Side view along t2: horizontal t1, vertical depth.
    fixed_mask = ~structure.mobile.any(axis=1)
    if fixed and fixed.get("fixed_ids") is not None:
        wanted = set(fixed["fixed_ids"])
        fixed_mask = np.array([aid in wanted for aid in structure.ids])
    sx = pos @ t1
    side = np.column_stack([sx, depth, pos @ t2])
    lo2 = np.array([sx.min() - 2.0, depth.min() - 2.0])
    hi2 = np.array([sx.max() + 2.0, depth.max() + 3.0])
    scale2 = min(scale, width / float(hi2[0] - lo2[0]), 380.0 / float(hi2[1] - lo2[1]))
    h2 = float((hi2[1] - lo2[1]) * scale2)
    order = np.argsort(-side[:, 2])
    side_sorted = side[order]
    side_sorted[:, 2] = -side_sorted[:, 2]
    body2 = _svg_atoms(
        side_sorted,
        [structure.symbols[i] for i in order],
        [structure.ids[i] for i in order],
        scale2,
        lo2,
        fixed=fixed_mask[order],
        height=h2,
    )
    ytop = h2 - (top - lo2[1]) * scale2
    body2.append(
        f'<line x1="0" x2="{(hi2[0] - lo2[0]) * scale2:.0f}" y1="{ytop:.1f}" y2="{ytop:.1f}" '
        'stroke="#64748b" stroke-dasharray="4 4"/>'
    )
    body2.append(
        f'<text x="6" y="{max(12.0, ytop - 6):.1f}" font-size="12" fill="#475569">vacuum (active side) &#8593;</text>'
    )
    side_svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {(hi2[0] - lo2[0]) * scale2:.0f} {h2:.0f}" '
        f'width="{(hi2[0] - lo2[0]) * scale2:.0f}" height="{h2:.0f}" role="img" '
        'aria-label="Side view with the fixed region">' + "".join(body2) + "</svg>"
    )
    return top_svg, side_svg


def write_site_map(structure, report, path, sites=None, fixed=None):
    """Write ``path`` (HTML) and the top view next to it as ``.svg``; returns both paths."""
    path = Path(path)
    sites = report["proposal"]["sites"] if sites is None else sites
    top_svg, side_svg = site_map_svg(structure, report, sites, fixed)
    rows = []
    metals = {s["id"]: s.get("metals", []) for s in report.get("ligand_sites", [])}
    for s in sites:
        where = s.get("metal") or ", ".join(metals.get(s["id"], []))
        rows.append(
            f"<tr><td><span class='dot' style='background:{SITE_COLORS.get(s['kind'], '#111')}'></span>"
            f"{html.escape(s['id'])}</td><td>{html.escape(s['kind'])}</td><td>{html.escape(s['native'])}</td>"
            f"<td>{html.escape(where)}</td></tr>"
        )
    v = report["vacuum"]
    legend = "".join(
        f"<span><span class='dot' style='background:{c}'></span>{k}</span>"
        for k, c in SITE_COLORS.items()
        if k in {s["kind"] for s in sites}
    )
    unbound = report.get("unbound_molecules") or []
    unbound_html = ""
    if unbound:
        listed = ", ".join(html.escape(f"{m['formula']} ({m['atoms'][0]})") for m in unbound)
        unbound_html = f"<h2>Unbound molecules</h2><p>{listed}</p>"
    name = Path(str(report.get("file") or "the slab")).name
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Site map: {html.escape(name)}</title>
<style>{PAGE_CSS}</style></head><body>
<h1>Reactive sites of {html.escape(name)}</h1>
<p class="meta">{report["n_atoms"]} atoms; vacuum along <b>{v["axis"]}</b>, normal {v["surface_normal"]};
active side {"+" if report["active_side"]["sign"] > 0 else "-"}{v["axis"]}
({html.escape(report["active_side"]["confidence"])}).
{len(sites)} site(s); {len(report["symmetry"]["translations"])} lattice translation(s) besides the identity map the slab
onto itself (equivalent sites are computed once).</p>
<p class="legend">{legend}</p>
<div class="views">
<figure>{top_svg}<figcaption>Top view of the active side (atoms within 6 &#8491; of the top; faded: periodic images;
dashed: the cell). Hover an atom for its ID.</figcaption></figure>
<div><h2>Sites</h2><table><tr><th>site</th><th>kind</th><th>native</th><th>metal(s)</th></tr>{"".join(rows)}</table>
{unbound_html}
<p class="meta">Edit <code>sites:</code> in project.yaml to add or remove a site.</p></div>
</div>
<h2>Side view</h2>
<figure>{side_svg}<figcaption>Pale atoms with a dashed outline are fixed during relaxations.</figcaption></figure>
</body></html>
"""
    atomic_text(path, page)
    svg_path = path.with_suffix(".svg")
    atomic_text(svg_path, top_svg + "\n")
    return {"html": str(path), "svg": str(svg_path)}
