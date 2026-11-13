"""Draw the four schemes of docs/surface-first.md.

    python docs/images/surface_first.py            # writes the SVG files next to this script

* surface-carried.svg: how `mechanet init` reads adsorbates that a slab file carries;
* surface-changed.svg: what happens to a start whose relaxed structure is another surface;
* surface-orders.svg: how many placement orders there are, and how many are tried;
* surface-resting.svg: the cycle of a site that rests in *OOH on the surface the diagram gives.
"""

import sys
from html import escape
from pathlib import Path

W = 760
SANS = "system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,'DejaVu Sans',sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,'DejaVu Sans Mono','Liberation Mono',monospace"
INK, MUTED, LINE = "#0b0b0b", "#5f5e59", "#8d8b84"
WHITE, EDGE = "#ffffff", "#d4d2ca"
GRAY = ("#f1f0ec", "#e1e0d9", "#52514e")  # fill, edge, accent
BLUE = ("#ebf2fb", "#d3e2f4", "#1f62b4")
GREEN = ("#ecf6ec", "#d2e8d3", "#17761a")
AMBER = ("#fdf3e0", "#f1dcae", "#8a5200")
METAL, OXYGEN, HYDROGEN, CHLORINE = ("#c9c7be", LINE), ("#ef8a6f", "#b8452f"), (WHITE, LINE), ("#59c2a4", "#1f7a62")
SLAB = ("#e4e2da", "#c9c7be")


def text(x, y, s, size=12, fill=INK, family=SANS, weight=None, anchor="middle"):
    w = f' font-weight="{weight}"' if weight else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" fill="{fill}"'
        f' text-anchor="{anchor}"{w}>{escape(s)}</text>'
    )


def box(x, y, w, h, lines, colours=(WHITE, EDGE, INK), mono_first=False):
    """A rounded box with a first line in bold and the others muted, centred."""
    fill, edge, ink = colours
    out = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" fill="{fill}" stroke="{edge}"/>']
    top = y + h / 2 - 8.5 * (len(lines) - 1) + 4.5
    for k, line in enumerate(lines):
        first = k == 0
        family = MONO if first and mono_first else SANS
        out.append(
            text(
                x + w / 2,
                top + 17 * k,
                line,
                12.5 if first else 11.5,
                ink if first else MUTED,
                family,
                600 if first else None,
            )
        )
    return out


def arrow(d):
    return f'<path d="{d}" stroke="{LINE}" stroke-width="1.5" fill="none" marker-end="url(#arrow)"/>'


def atom(x, y, r, kind, ghost=False):
    fill, edge = kind
    if ghost:  # an atom taken out of the slab
        dashed = f'fill="none" stroke="{LINE}" stroke-width="1.2" stroke-dasharray="3 3"'
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" {dashed}/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{edge}"/>'


def card(height, title, description, body):
    return (
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}"'
                ' role="img" aria-labelledby="t d">',
                f'<title id="t">{escape(title)}</title>',
                f'<desc id="d">{escape(description)}</desc>',
                '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7"'
                f' orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{LINE}"/></marker></defs>',
                f'<rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1}" rx="14" fill="#fbfbf9"'
                ' stroke="#e1e0d9"/>',
                *body,
                "</svg>",
            ]
        )
        + "\n"
    )


def carried():
    """Three groups on a slab, and what `mechanet init` makes of each."""
    out, centres = [], (300, 470, 640)
    for cx, head in zip(centres, ("OOH on a metal", "Cl on a metal", "O₂ across two metals")):
        out.append(text(cx, 32, head, 12, MUTED))

    def slab(cx, top, metals=(0,)):
        parts = [f'<rect x="{cx - 62}" y="{top}" width="124" height="26" rx="5" fill="{SLAB[0]}" stroke="{SLAB[1]}"/>']
        return parts + [atom(cx + dx, top, 13, METAL) for dx in metals]

    for row, top in enumerate((106, 222)):
        kept = row == 1  # the second row is the slab the project keeps
        out += slab(centres[0], top)
        out.append(atom(centres[0], top - 22, 9, OXYGEN))
        out.append(atom(centres[0] + 13, top - 34.4, 9, OXYGEN, ghost=kept))
        out.append(atom(centres[0] + 24, top - 43.1, 5, HYDROGEN, ghost=kept))
        out += slab(centres[1], top)
        out.append(atom(centres[1], top - 24, 11, CHLORINE, ghost=kept))
        out += slab(centres[2], top, metals=(-20, 20))
        out += [atom(centres[2] - 9, top - 21, 9, OXYGEN), atom(centres[2] + 9, top - 21, 9, OXYGEN)]
    out.append(text(26, 104, "Your slab file", 13.5, INK, weight=650, anchor="start"))
    out.append(text(26, 212, "Slab kept in", 13.5, INK, weight=650, anchor="start"))
    out.append(text(26, 229, "the project", 13.5, INK, weight=650, anchor="start"))
    out.append(text(26, 317, "Read as", 13.5, INK, weight=650, anchor="start"))
    blue = (BLUE[0], BLUE[1], INK)
    out += box(centres[0] - 78, 284, 156, 56, ["State OOH", "on an oxygen site"], blue)
    out += box(centres[1] - 78, 284, 156, 56, ["State Cl", "on a metal site"], blue)
    out += box(
        centres[2] - 78, 284, 156, 56, ["Stays in the slab", "holds on through two atoms"], (GRAY[0], GRAY[1], INK)
    )
    out += [arrow(f"M{cx} 344V366") for cx in centres[:2]]
    out += box(
        222,
        372,
        326,
        56,
        ["The campaign is built on your surface", "these states, the atoms where your file has them"],
        blue,
    )
    x = 222
    for kind, label in ((METAL, "metal"), (OXYGEN, "oxygen"), (HYDROGEN, "hydrogen"), (CHLORINE, "chlorine")):
        out.append(atom(x + 6, 452, 6, kind))
        out.append(text(x + 18, 456, label, 11.5, MUTED, anchor="start"))
        x += 30 + 7 * len(label)
    out.append(atom(x + 6, 452, 6, HYDROGEN, ghost=True))
    out.append(text(x + 18, 456, "taken out of the slab", 11.5, MUTED, anchor="start"))
    return card(
        476,
        "How mechanet init reads adsorbates that a slab file carries",
        "Three columns. An OOH on a metal and a Cl on a metal are each read as the state of a site: the outer "
        "atoms of the OOH and the Cl are taken out of the slab the project keeps, and the campaign is built on "
        "the surface as given, with the atoms where the file has them. An O2 lying across two metals holds on "
        "through two atoms, cannot be the state of one site, and stays in the slab.",
        out,
    )


def changed():
    """A start is relaxed and read again: three outcomes."""
    out = box(250, 22, 260, 56, ["A start built as surface A", "is relaxed, then read again"])
    out += [arrow("M380 82V104H136V122"), arrow("M380 82V122"), arrow("M380 82V104H624V122")]
    out += box(26, 128, 220, 56, ["Still surface A", "nothing changed"])
    out += box(270, 128, 220, 56, ["Became surface B", "which the campaign has"])
    out += box(514, 128, 220, 56, ["Became surface C", "which the campaign did not list"])
    out += [arrow(f"M{x} 188V210") for x in (136, 380, 624)]
    out += box(26, 216, 220, 74, ["Counts for A"])
    out += box(
        270,
        216,
        220,
        74,
        ["Counts for B", "review names it;", "the lowest structure stands for B"],
        (BLUE[0], BLUE[1], INK),
    )
    out += box(
        514, 216, 220, 74, ["Reported as new", "you add it with", "review --discover"], (GREEN[0], GREEN[1], INK)
    )
    return card(
        312,
        "What happens to a start whose relaxed structure is another surface",
        "A start built as surface A is relaxed and read again. If it still is A, it counts for A. If it became "
        "surface B, which the campaign has, it counts for B: review names it and the lowest structure stands for "
        "B. If it became surface C, which the campaign did not list, it is reported as new and you add it with "
        "review --discover.",
        out,
    )


def orders():
    """How many orders there are to place n adsorbates, and how many are tried."""
    out = []
    for x, colours, title, cells in (
        (26, GREEN, "Every order is tried", (("2 to place", "2 orders"), ("3 to place", "6 orders"))),
        (392, AMBER, "Two orders are tried", (("4 to place", "24 orders"), ("8 to place", "40,320 orders"))),
    ):
        out.append(f'<rect x="{x}" y="18" width="342" height="118" rx="12" fill="{colours[0]}" stroke="{colours[1]}"/>')
        out.append(f'<rect x="{x + 14}" y="30" width="4" height="15" rx="2" fill="{colours[2]}"/>')
        out.append(text(x + 25, 42.5, title, 13.5, INK, weight=650, anchor="start"))
        for k, cell in enumerate(cells):
            out += box(x + 14 + k * 164, 62, 150, 56, list(cell))
    return card(
        154,
        "Placement orders for a starting structure with several adsorbates to orient",
        "With two or three adsorbates to orient there are 2 and 6 placement orders, and every one is tried. With "
        "four there are 24 and with eight 40,320: from four on, two orders are tried, the listed one and its reverse.",
        out,
    )


def resting():
    """The cycle of a site that rests in *OOH, read from that surface."""
    out, width, gap, left, top = [], 84, 42, 23, 74
    states = ("*OOH", "*OO", "*", "*OH", "*O", "*OOH")
    steps = (("− H⁺ − e⁻", None), ("− O₂", None), ("− H⁺ − e⁻", "+ H₂O"), ("− H⁺ − e⁻", None), ("− H⁺ − e⁻", "+ H₂O"))
    for k, state in enumerate(states):
        x = left + k * (width + gap)
        rests = k in (0, len(states) - 1)
        note = "empty site" if state == "*" else "the surface" if rests else None
        out += box(
            x,
            top,
            width,
            56,
            [state] + ([note] if note else []),
            (BLUE[0], BLUE[1], INK) if rests else (WHITE, EDGE, INK),
        )
        if k < len(steps):
            mid = x + width + gap / 2
            out.append(arrow(f"M{x + width + 4} {top + 28}H{x + width + gap - 5}"))
            leaves, enters = steps[k]
            electron = "e⁻" in leaves
            out.append(
                text(mid, top - 10, leaves, 11.5, BLUE[2] if electron else MUTED, weight=600 if electron else None)
            )
            if enters:
                out.append(text(mid, top - 26, enters, 11.5, MUTED))
    out.append(text(W / 2, 158, "Four steps release a proton and an electron; one releases O₂.", 12, INK))
    out.append(text(W / 2, 177, "Every other site keeps the state it has on that surface.", 11.5, MUTED))
    return card(
        196,
        "The cycle of a site that rests in OOH",
        "A site that holds OOH on the surface the Pourbaix diagram gives goes from OOH to OO, releases O2 to an "
        "empty site, and returns through OH and O to OOH. Four of the five steps release a proton and an "
        "electron. Every other site keeps the state it has on that surface.",
        out,
    )


FIGURES = {
    "surface-carried.svg": carried,
    "surface-changed.svg": changed,
    "surface-orders.svg": orders,
    "surface-resting.svg": resting,
}

if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    for name, draw in FIGURES.items():
        (folder / name).write_text(draw(), encoding="utf-8")
