"""Draw workflow.svg: the commands of a study in three bands (laptop, cluster, analysis).

    python docs/images/workflow.py docs/images/workflow.svg

workflow.png, which the README shows (PyPI does not show every SVG), is the same drawing
rendered by a browser at twice the size.
"""

import sys
from html import escape
from pathlib import Path

W, PAD = 760, 14
BX, BW = PAD, W - 2 * PAD  # band rectangle
IX, IW = BX + 12, BW - 24  # band content
GAP, BOX_H, HEAD, BOTTOM, ROW_GAP, BAND_GAP = 20, 56, 36, 14, 26, 22
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,'DejaVu Sans Mono','Liberation Mono',monospace"
SANS = "system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,'DejaVu Sans',sans-serif"

BANDS = [
    dict(
        title="On your laptop",
        note="no cluster needed",
        fill="#f1f0ec",
        stroke="#e1e0d9",
        accent="#52514e",
        rows=[
            [
                ("init SLAB", "finds the sites,", "writes project.yaml"),
                ("generate", "intermediates, steps,", "cycles, structures"),
                ("plan", "what the study", "will cost"),
                ("show-cycle ID", "one mechanism,", "as equations"),
            ]
        ],
    ),
    dict(
        title="On the cluster",
        note="SLURM, VASP and your PAW files",
        fill="#ebf2fb",
        stroke="#d3e2f4",
        accent="#1f62b4",
        rows=[
            [
                ("cluster import", "a profile from", "your job script"),
                ("references", "H₂ and H₂O with", "your settings"),
                ("submit --pilot", "test the protocol,", "get timings"),
                ("submit", "every structure;", "repeat any time"),
                ("status", "what is done, and", "what needs you"),
            ],
            [
                ("collect", "parse and check", "every run"),
                ("select", "the variants for", "frequencies"),
                ("submit --selected --through freq", "partial Hessians by finite displacements,", "restartable", 2),
            ],
        ],
    ),
    dict(
        title="Analysis",
        note="in the project folder, once results are in",
        fill="#ecf6ec",
        stroke="#d2e8d3",
        accent="#17761a",
        rows=[
            [
                ("thermo", "free energies,", "limiting potentials"),
                ("plot-profile ID", "the free-energy", "diagram"),
                ("pathways --rank", "the best route", "over every cycle"),
                ("pourbaix --plot", "stable surface", "terminations"),
                ("report", "everything in", "report.md"),
            ]
        ],
    ),
]


def text(x, y, s, size, fill, family=SANS, weight=None, anchor="middle"):
    w = f' font-weight="{weight}"' if weight else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" fill="{fill}"'
        f' text-anchor="{anchor}"{w}>{escape(s)}</text>'
    )


def main():
    out, y, number = [], PAD, 0
    for b_index, band in enumerate(BANDS):
        rows = band["rows"]
        height = HEAD + len(rows) * BOX_H + (len(rows) - 1) * ROW_GAP + BOTTOM
        out.append(
            f'<rect x="{BX}" y="{y}" width="{BW}" height="{height}" rx="12" fill="{band["fill"]}"'
            f' stroke="{band["stroke"]}"/>'
        )
        out.append(f'<rect x="{IX}" y="{y + 10}" width="4" height="15" rx="2" fill="{band["accent"]}"/>')
        out.append(text(IX + 11, y + 22.5, band["title"], 13.5, "#0b0b0b", weight=650, anchor="start"))
        out.append(text(IX + IW, y + 22.5, band["note"], 11.5, "#5f5e59", anchor="end"))
        row_boxes = []
        for r_index, row in enumerate(rows):
            top = y + HEAD + r_index * (BOX_H + ROW_GAP)
            ncols = 4 if len(rows) == 1 and len(row) == 4 else 5
            col_w = (IW - (ncols - 1) * GAP) / ncols
            col, placed = 0, []
            for item in row:
                cmd, line1, line2 = item[:3]
                span = item[3] if len(item) > 3 else 1
                number += 1
                x = IX + col * (col_w + GAP)
                w = span * col_w + (span - 1) * GAP
                cx = x + w / 2
                out.append(
                    f'<rect x="{x:.1f}" y="{top}" width="{w:.1f}" height="{BOX_H}" rx="8" fill="#ffffff"'
                    f' stroke="#d4d2ca"/>'
                )
                out.append(text(cx, top + 20.5, cmd, 12, "#0b0b0b", MONO, 600))
                out.append(text(cx, top + 36.5, line1, 11, "#5f5e59"))
                out.append(text(cx, top + 50, line2, 11, "#5f5e59"))
                # step number on the top-left corner
                out.append(
                    f'<circle cx="{x + 1:.1f}" cy="{top + 1}" r="8.5" fill="{band["accent"]}"'
                    f' stroke="{band["fill"]}" stroke-width="2"/>'
                )
                out.append(text(x + 1, top + 4.8, str(number), 10, "#ffffff", weight=700))
                placed.append((x, top, w))
                col += span
            for (x1, t1, w1), (x2, _, _) in zip(placed, placed[1:]):  # arrows along the row
                mid = t1 + BOX_H / 2
                out.append(
                    f'<path d="M{x1 + w1 + 4:.1f} {mid}H{x2 - 5:.1f}" stroke="#8d8b84" stroke-width="1.5"'
                    f' fill="none" marker-end="url(#arrow)"/>'
                )
            row_boxes.append(placed)
        for upper, lower in zip(row_boxes, row_boxes[1:]):  # wrap from the end of a row to the next row
            x1, t1, w1 = upper[-1]
            x2, t2, w2 = lower[0]
            sx, sy, ex = x1 + w1 / 2, t1 + BOX_H + 3, x2 + w2 / 2
            my = t1 + BOX_H + ROW_GAP / 2
            out.append(
                f'<path d="M{sx:.1f} {sy}V{my - 6}Q{sx:.1f} {my} {sx - 6:.1f} {my}H{ex + 6:.1f}'
                f'Q{ex:.1f} {my} {ex:.1f} {my + 6}V{t2 - 5}" stroke="#8d8b84" stroke-width="1.5"'
                f' fill="none" marker-end="url(#arrow)"/>'
            )
        y += height
        if b_index < len(BANDS) - 1:  # a short arrow down to the next band
            out.append(
                f'<path d="M{W / 2} {y + 3}V{y + BAND_GAP - 5}" stroke="#8d8b84" stroke-width="1.5"'
                f' marker-end="url(#arrow)"/>'
            )
            y += BAND_GAP
    H = y + PAD
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img"'
        ' aria-labelledby="t d">',
        '<title id="t">The mechanet workflow</title>',
        '<desc id="d">Seventeen mechanet commands in three bands. On your laptop: init, generate, plan, '
        "show-cycle. On the cluster: cluster import, references, submit --pilot, submit, status, collect, "
        "select, submit --selected --through freq. Analysis: thermo, plot-profile, pathways --rank, "
        "pourbaix --plot, report.</desc>",
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7"'
        ' orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#8d8b84"/></marker></defs>',
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="14" fill="#fbfbf9" stroke="#e1e0d9"/>',
        *out,
        "</svg>",
    ]
    return "\n".join(svg) + "\n"


if __name__ == "__main__":
    Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("workflow.svg")).write_text(
        main(), encoding="utf-8"
    )
