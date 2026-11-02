"""Readable names for intermediates, steps and cycles.

An intermediate is written as the state of each site the cycle involves, one term per site,
joined by a centred dot::

    *OOH₆₅ · μ-O₆₁      OOH on terminal site a0065, a deprotonated bridging O at a0061
    *₆₅ · μ-OH₆₁        site a0065 empty (the bare metal), bridging OH at a0061
    μ-□₆₁               the bridging O of a0061 removed (a vacancy)
    *O₆₅–*O₆₆           the O atoms of a0065 and a0066 bonded to each other

``*`` marks a terminal site (one metal: a terminal O ligand or an open metal site), ``μ-`` a
bridging one (two metals; ``μ₃-`` three); the subscript is the site's number (a0065 -> 65,
m0012 -> 12). Formulas that contain digits are bracketed (``*(H₂O)₆₅``). A step reads as a
reaction equation, ``*O₆₅ + H₂O → *OOH₆₅ + H⁺ + e⁻``, and is electrochemical when electrons
cross the interface, chemical otherwise.

Text is built in Unicode; :func:`plain` gives the ASCII form (``*OOH65 & mu-O61``,
``->``, ``H+``) for terminals that cannot show it.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from .fileio import read_json
from .literature import LITERATURE
from .reactions import SPECIES, electrons

SITE_ID = re.compile(r"[am]\d{4}")
SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUPERSCRIPT = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
SEPARATOR = " · "
BOND = "–"
ARROW = "→"
VACANCY = "□"
TERMINAL_KINDS = ("terminal", "open_metal")
RELEASE = {"oer": "O2", "cer": "Cl2"}
DEFAULT_SCOPE = "proposed_on_this_surface"

# Unicode -> ASCII, for terminals whose encoding cannot show the notation.
ASCII = {
    **{chr(0x2080 + d): str(d) for d in range(10)},
    **{c: str(d) for d, c in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")},
    "⁺": "+",
    "⁻": "-",
    "μ": "mu",
    "µ": "mu",
    VACANCY: "[]",
    ARROW: "->",
    "←": "<-",
    "·": "&",
    BOND: "-",
    "—": "-",
    "…": "...",
    "≥": ">=",
    "≤": "<=",
    "≈": "~",
    "±": "+/-",
    "×": "x",
    "Δ": "d",
    "η": "eta",
    "°": " deg",
    "▶": ">",
    "◷": "q",
    "↻": "~",
    "⋅": ".",
    "✓": "#",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def subscript(value):
    return str(value).translate(SUBSCRIPT)


def formula(text):
    """Digits after an element or a bracket become subscripts: H2O -> H₂O."""
    return re.sub(r"(?<=[A-Za-z)\]])(\d+)", lambda m: subscript(m.group(1)), str(text))


def species(name):
    """A reservoir species in chemical notation: H2O -> H₂O, H+ -> H⁺, Cl- -> Cl⁻, e- -> e⁻."""
    match = re.fullmatch(r"(.+?)(\d*)([+-])", str(name))
    if match:
        core, charge, sign = match.groups()
        return formula(core) + (charge + sign).translate(SUPERSCRIPT)
    return formula(name)


def plain(text):
    """ASCII form of a text written with this notation (and of any other non-ASCII text)."""
    out = []
    for char in str(text):
        if ord(char) < 128:
            out.append(char)
        elif char in ASCII:
            out.append(ASCII[char])
        else:
            decomposed = unicodedata.normalize("NFKD", char).encode("ascii", "ignore").decode()
            out.append(decomposed or "?")
    return "".join(out)


def encodable(stream, text):
    """True when ``stream`` (e.g. sys.stdout) can write ``text`` in its encoding (a stream
    without one, such as io.StringIO, holds any text)."""
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return True
    try:
        str(text).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def site_number(site_id):
    """a0065 -> '65', m0012 -> '12'; other IDs are kept."""
    site_id = str(site_id)
    return str(int(site_id[1:])) if SITE_ID.fullmatch(site_id) else site_id


def short_authors(authors):
    """'Rossmeisl, Logadottir and Norskov' -> 'Rossmeisl et al.'; two names stay; 'X et al.' stays."""
    authors = str(authors).strip()
    if "et al" in authors:
        return authors
    names = [n.strip() for n in re.split(r",| and ", authors) if n.strip()]
    if len(names) > 2:
        return f"{names[0]} et al."
    return " and ".join(names)


def citation(source_id):
    record = LITERATURE.get(source_id)
    if not record:
        return str(source_id)
    return f"{short_authors(record['authors'])} {record['year']}"


def citations(source_ids):
    """'Rossmeisl et al. 2005, 2007; Man et al. 2011' (one entry per author group, in order)."""
    groups = {}
    for source_id in source_ids or ():
        record = LITERATURE.get(source_id)
        authors, year = (short_authors(record["authors"]), str(record["year"])) if record else (str(source_id), "")
        groups.setdefault(authors, [])
        if year and year not in groups[authors]:
            groups[authors].append(year)
    return "; ".join(f"{authors} {', '.join(years)}".strip() for authors, years in groups.items())


# ---------------------------------------------------------------------------
# Routes: what each family's cycles are called
# ---------------------------------------------------------------------------

# (families, pattern of the cycle ID once the family prefix and the site IDs are removed,
#  short name, title). Patterns must match the whole remainder; the first match wins.
ROUTES = (
    (("aem",), r"AEM", "AEM", "Adsorbate evolution (AEM)"),
    (("aem",), r"bound_OO", "AEM, bound O₂", "AEM with O₂ held on the site before it leaves"),
    (
        ("surface_bridge",),
        r"AEM",
        "bridge AEM",
        "Adsorbate evolution on a bridging O (its vacancy is refilled from water)",
    ),
    (
        ("surface_bridge",),
        r"bound_OO",
        "bridge AEM, bound O₂",
        "Adsorbate evolution on a bridging O, with O₂ held on the site before it leaves",
    ),
    (
        ("pair_coupling",),
        r"(?P<order>.+)_direct",
        "direct O–O coupling",
        "O–O coupling of two neighbouring O; O₂ leaves in the coupling step",
    ),
    (
        ("pair_coupling",),
        r"(?P<order>.+)_bound",
        "bound O–O coupling",
        "O–O coupling of two neighbouring O through a bound O–O intermediate",
    ),
    (
        ("terminal_bridge_coupling",),
        r"(?P<order>.+)_direct",
        "direct terminal–bridge coupling",
        "Coupling of a terminal O with a neighbouring bridging O; O₂ leaves in the coupling step",
    ),
    (
        ("terminal_bridge_coupling",),
        r"(?P<order>.+)_bound",
        "bound terminal–bridge coupling",
        "Coupling of a terminal O with a neighbouring bridging O through a bound O–O intermediate",
    ),
    (("bifunctional",), r"direct_water_attack", "direct water attack", "Bifunctional, direct water attack"),
    (
        ("bifunctional",),
        r"concerted",
        "concerted *OOH",
        "Bifunctional, *OOH proton transfer concerted with O₂ release",
    ),
    (("bifunctional",), r"resolved", "stepwise *OOH", "Bifunctional, *OOH proton transfer as a separate step"),
    (("bifunctional",), r"BF1", "BF-I", "Bifunctional mechanism I (BF-I)"),
    (("bifunctional",), r"BF2", "BF-II", "Bifunctional mechanism II (BF-II)"),
    (
        ("hydrated",),
        r"bound_water_AEM",
        "bound water",
        "AEM with molecular water bound to the site before it is deprotonated",
    ),
    (
        ("hydrated",),
        r"bound_water_OO",
        "bound water, bound O₂",
        "AEM with bound water, and O₂ held on the site before it leaves",
    ),
    (
        ("hydrated",),
        r"via_OOH_OH",
        "hydration via *OOH",
        "Local hydration with H transfer to the neighbour, resolved through *OOH next to OH",
    ),
    (
        ("hydrated",),
        r"via_OO_H2O",
        "hydration via *OO",
        "Local hydration with H transfer to the neighbour, resolved through *OO next to H₂O",
    ),
    (
        ("hydrated",),
        r"tautomer_via_OOH_OH",
        "tautomer via *OOH",
        "Local hydration starting from the O + H₂O tautomer, resolved through *OOH next to OH",
    ),
    (
        ("hydrated",),
        r"tautomer_via_OO_H2O",
        "tautomer via *OO",
        "Local hydration starting from the O + H₂O tautomer, resolved through *OO next to H₂O",
    ),
    (("oo_oo",), r"(?P<order>.+)", "OO/OO (O₄)", "Recombination of two *OO through O₄ (OO/OO)"),
    (
        ("cer_volmer_heyrovsky",),
        r"",
        "Volmer–Heyrovsky",
        "Volmer–Heyrovsky: Cl adsorbs, then a second Cl⁻ takes it off as Cl₂",
    ),
    (("cer_volmer_tafel",), r"", "Volmer–Tafel", "Volmer–Tafel: two adsorbed Cl recombine to Cl₂"),
    (("cer_ocl",), r"", "*OCl route", "Chlorine evolution through *OCl on an oxygen-covered site"),
    (("discovered",), r".*", "discovered", "Cycle found by enumerating the network"),
)
# Event letters in the order suffix of the two-site families.
ORDER_EVENTS = {
    "pair_coupling": {"R": "refill", "D": "deprotonate"},
    "terminal_bridge_coupling": {"R": "refill", "D": "deprotonate"},
    "oo_oo": {"A": "OOH on", "D": "deprotonate"},
}


def route_remainder(cycle_id, family):
    """The cycle ID without its family prefix and site IDs: 'bifunctional_a0065_a0061_BF1' -> 'BF1'."""
    text = str(cycle_id)
    if family and text.startswith(f"{family}_"):
        text = text[len(family) + 1 :]
    elif family and text == family:
        text = ""
    parts = [p for p in text.split("_") if p and not SITE_ID.fullmatch(p)]
    return "_".join(parts)


def _order_text(order, family):
    events = ORDER_EVENTS.get(family) or {}
    words = []
    for token in order.split("_"):
        match = re.fullmatch(r"(\d+)([A-Z])", token)
        if not match:
            return order.replace("_", " ")
        number, letter = match.groups()
        words.append(f"{events.get(letter, letter)} {int(number)}")
    return ", then ".join(words)


def route(cycle, families=None):
    """{'key', 'short', 'title', 'variant'} of a cycle record.

    ``key`` groups cycles of the same mechanism (``family:short``); ``variant`` tells apart
    the cycles of one key at the same sites (the event order of the two-site families).
    ``families``: family name -> description, for plugin families without an entry here.
    """
    family = cycle.get("family") or ("discovered" if cycle.get("discovered") else None)
    if family is None and cycle.get("families"):
        family = "+".join(cycle["families"])
    remainder = route_remainder(cycle.get("id", ""), family)
    for names, pattern, short, title in ROUTES:
        if family not in names:
            continue
        match = re.fullmatch(pattern, remainder)
        if match is None:
            continue
        order = match.groupdict().get("order")
        variant = _order_text(order, family) if order else None
        return {"key": f"{family}:{short}", "short": short, "title": title, "variant": variant}
    description = (families or {}).get(family) or (family or "cycle").replace("_", " ")
    short = remainder.replace("_", " ") or (family or "cycle").replace("_", " ")
    title = description[:1].upper() + description[1:]
    if remainder:
        title += f" ({remainder.replace('_', ' ')})"
    return {"key": f"{family}:{short}", "short": short, "title": title, "variant": None}


# ---------------------------------------------------------------------------
# Intermediates, steps and cycles of one network
# ---------------------------------------------------------------------------


class Notation:
    """Readable labels for the nodes, steps and cycles of a network."""

    def __init__(self, network, sites, registry=None, extra_edges=()):
        self.network = network
        self.nodes = {n["id"]: n for n in network.get("nodes", [])}
        self.edges = {e["id"]: e for e in network.get("edges", [])}
        self.edges.update({e["id"]: e for e in extra_edges or ()})
        self.cycles = {c["id"]: c for c in network.get("cycles", [])}
        self.sites = {s["id"]: s for s in sites or ()}
        self.native = {sid: s.get("native") for sid, s in self.sites.items()}
        self.reference = network.get("reference", "ref")
        self.registry = registry
        self.families = {}
        if registry is not None:
            self.families = {name: family.description for name, family in getattr(registry, "families", {}).items()}

    @classmethod
    def from_campaign(cls, campaign, registry=None, network=None):
        campaign = Path(campaign)
        network = network if network is not None else read_json(campaign / "network.json")
        sites_path = campaign / "sites.json"
        sites = read_json(sites_path).get("sites", []) if sites_path.is_file() else []
        extra = []
        discovered = campaign / "pathways" / "pathways.json"
        if discovered.is_file():
            try:
                extra = read_json(discovered).get("new_edges", [])
            except (OSError, ValueError):
                extra = []
        return cls(network, sites, registry=registry, extra_edges=extra)

    # ------------------------------------------------------------------ sites
    def _prefix(self, site_id):
        site = self.sites.get(site_id) or {}
        kind = site.get("kind")
        if kind in TERMINAL_KINDS:
            return "*"
        metals = len(site.get("metal_neighbors") or [])
        if kind == "bridge" or metals == 2:
            return "μ-"
        if metals >= 3:
            return f"μ{subscript(metals)}-"
        if kind is None and not site:
            return "*"
        return "μ-" if metals != 1 else "*"

    def term(self, site_id, status):
        """One site in one state: *OOH₆₅, μ-O₆₁, *₆₅ (empty terminal), μ-□₆₁ (bridge vacancy)."""
        prefix = self._prefix(site_id)
        number = subscript(site_number(site_id))
        if status in (None, "empty"):
            body = "" if prefix == "*" else VACANCY
        else:
            body = formula(status)
            if any(c.isdigit() for c in str(status)):
                body = f"({body})"
        return f"{prefix}{body}{number}"

    def symbol(self, site_id):
        """The site alone, for headings: *₆₅ (terminal), μ₆₁ (bridge), μ₃-₇₀ (three metals)."""
        prefix = self._prefix(site_id)
        return ("μ" if prefix == "μ-" else prefix) + subscript(site_number(site_id))

    def site_legend(self, site_id):
        """'65 = a0065: terminal O on Fe a0003 (native OH)'."""
        site = self.sites.get(site_id) or {}
        metals = [f"{m.get('element') or '?'} {m['id']}" for m in site.get("metal_neighbors") or []]
        joined = (
            metals[0]
            if len(metals) == 1
            else " and ".join(metals)
            if len(metals) == 2
            else ", ".join(metals[:-1]) + f" and {metals[-1]}"
            if metals
            else "?"
        )
        kind = site.get("kind")
        element = site.get("element") or "O"
        if kind == "open_metal":
            what = f"open site on {joined}"
        elif kind == "terminal":
            what = f"terminal {element} on {joined}"
        elif kind == "bridge":
            what = f"{element} bridging {joined}"
        elif kind in ("mu3", "framework"):
            what = f"{element} shared by {joined}"
        else:
            what = "site"
        native = site.get("native")
        native_text = "bare" if native == "empty" else f"native {formula(native)}" if native else None
        return f"{site_number(site_id)} = {site_id}: {what}" + (f" ({native_text})" if native_text else "")

    # ------------------------------------------------------------------ intermediates
    def occupancy(self, node_id):
        node = self.nodes.get(node_id) or {}
        return {**self.native, **(node.get("occupancy") or {})}

    def changed_sites(self, node_id):
        node = self.nodes.get(node_id) or {}
        out = [s for s, v in (node.get("occupancy") or {}).items() if v != self.native.get(s)]
        out += [s for s in node.get("coupling") or () if s not in out]
        return out

    def label(self, node_id, sites=None):
        """The intermediate as the state of ``sites`` (default: the sites it changes)."""
        sites = list(sites) if sites is not None else self.changed_sites(node_id)
        if not sites:
            return "native surface" if node_id == self.reference else str(node_id)
        node = self.nodes.get(node_id) or {}
        occupancy = self.occupancy(node_id)
        coupled = [s for s in node.get("coupling") or () if s in sites]
        terms, done = [], set()
        for site in sites:
            if site in done:
                continue
            if site in coupled and len(coupled) >= 2:
                terms.append(BOND.join(self.term(s, occupancy.get(s)) for s in coupled))
                done.update(coupled)
            else:
                terms.append(self.term(site, occupancy.get(site)))
                done.add(site)
        return SEPARATOR.join(terms)

    # ------------------------------------------------------------------ steps
    def release_species(self, cycle):
        reaction = cycle.get("reaction") or self.network.get("reaction")
        if self.registry is not None and reaction:
            try:
                return self.registry.reaction(reaction).release
            except (KeyError, ValueError, AttributeError):
                pass
        if reaction in RELEASE:
            return RELEASE[reaction]
        totals = cycle.get("reservoirs") or {}
        neutral = [s for s, v in totals.items() if abs(abs(float(v)) - 1) < 1e-9 and _charge(s) == 0]
        return neutral[0] if len(neutral) == 1 else None

    def vector(self, edge):
        if edge.get("reservoirs") is not None:
            return {k: v for k, v in edge["reservoirs"].items() if abs(float(v)) > 1e-12}
        return {
            k: v
            for k, v in {"H2O": -edge.get("water", 0), "H+": edge.get("pairs", 0), "O2": edge.get("oxygen", 0)}.items()
            if v
        }

    def step(self, edge, sites=None):
        """{'equation', 'kind', 'electrons', ...} of one edge; the intermediates as ``label`` writes them."""
        if isinstance(edge, str):
            edge = self.edges[edge]
        vector = self.vector(edge)
        n = electrons(vector) if all(name in SPECIES for name in vector) else _electrons(vector)
        left = [self.label(edge["from"], sites)] + [_amount(-v, k) for k, v in _ordered(vector) if v < 0]
        right = [self.label(edge["to"], sites)] + [_amount(v, k) for k, v in _ordered(vector) if v > 0]
        if n > 0:
            right.append(_amount(n, "e-"))
        elif n < 0:
            left.append(_amount(-n, "e-"))
        return {
            "edge_id": edge["id"],
            "from": edge["from"],
            "to": edge["to"],
            "equation": f"{' + '.join(left)} {ARROW} {' + '.join(right)}",
            "kind": "electrochemical" if n else "chemical",
            "electrons": n,
            "notes": list(edge.get("notes") or [])[:1],
        }

    # ------------------------------------------------------------------ cycles
    def ordered(self, cycle):
        """(edge IDs, node IDs) in the turnover convention: the release step last (OER: O₂)."""
        from .thermo import release_last_rotation

        edge_ids = list(cycle["edge_ids"])
        release = self.release_species(cycle)
        if release and all(e in self.edges for e in edge_ids):
            shift = release_last_rotation([{"reservoirs": self.vector(self.edges[e])} for e in edge_ids], release)
            edge_ids = edge_ids[shift:] + edge_ids[:shift]
        nodes = [self.edges[e]["from"] for e in edge_ids if e in self.edges]
        return edge_ids, nodes

    def cycle_sites(self, cycle, node_ids=None):
        """Sites the cycle involves, in the order its ID names them (donor before acceptor)."""
        if node_ids is None:
            node_ids = self.ordered(cycle)[1]
        involved = []
        for node_id in node_ids:
            for site in self.changed_sites(node_id):
                if site not in involved:
                    involved.append(site)
        named = [m.group(0) for m in SITE_ID.finditer(str(cycle.get("id", "")))]
        return [s for s in dict.fromkeys(named) if s in involved] + [s for s in involved if s not in named]

    def sources(self, cycle):
        """Evidence scope and literature of a cycle in one line."""
        scope = cycle.get("evidence_scope") or ""
        scope = scope.replace("_", " ") if scope == DEFAULT_SCOPE else scope
        refs = citations(cycle.get("source_ids"))
        if scope and refs:
            return f"{scope[:1].upper()}{scope[1:]}. Sources: {refs}"
        return scope[:1].upper() + scope[1:] if scope else (f"Sources: {refs}" if refs else "")

    def cycle(self, cycle):
        """Everything needed to show a cycle: title, sources, sites, intermediates and steps."""
        if isinstance(cycle, str):
            cycle = self.cycles[cycle]
        edge_ids, node_ids = self.ordered(cycle)
        sites = self.cycle_sites(cycle, node_ids)
        names = route(cycle, self.families)
        release = self.release_species(cycle)
        return {
            "cycle_id": cycle["id"],
            "family": cycle.get("family"),
            "reaction": cycle.get("reaction") or self.network.get("reaction"),
            "route": names["key"],
            "short": names["short"],
            "title": names["title"],
            "variant": names["variant"],
            "sources": self.sources(cycle),
            "sites": sites,
            "site_key": SEPARATOR.join(self.symbol(s) for s in sites),
            "site_legend": [self.site_legend(s) for s in sites],
            "node_ids": node_ids,
            "labels": [self.label(n, sites) for n in node_ids],
            "release": species(release) if release else None,
            "steps": [self.step(self.edges[e], sites) for e in edge_ids if e in self.edges],
            "symmetry_representative": cycle.get("symmetry_representative", True),
            "symmetry_equivalent_cycles": list(cycle.get("symmetry_equivalent_cycles") or []),
        }

    def cycle_lines(self, view, ascii=False, steps_only=False):
        """Lines for a terminal: title, sources, sites and the numbered step equations (ASCII if asked)."""
        text = plain if ascii else str
        lines = []
        if not steps_only:
            lines.append(text(view["title"] + (f" ({view['variant']})" if view.get("variant") else "")))
            if view.get("sources"):
                lines.append(text(f"  {view['sources']}"))
            if view.get("site_legend"):
                lines.append(text("  sites: " + "; ".join(view["site_legend"])))
        equations = [text(step["equation"]) for step in view["steps"]]
        width = max([0] + [len(e) for e in equations])
        for k, (equation, step) in enumerate(zip(equations, view["steps"]), start=1):
            lines.append(f"  {k}. {equation.ljust(width)}   {step['kind']}")
        return lines


def _charge(name):
    known = SPECIES.get(name)
    if known is not None:
        return known.charge
    match = re.fullmatch(r".+?(\d*)([+-])", str(name))
    if not match:
        return 0
    size = int(match.group(1) or 1)
    return size if match.group(2) == "+" else -size


def _electrons(vector):
    return int(round(sum(float(nu) * _charge(name) for name, nu in vector.items())))


def _ordered(vector):
    """Neutral molecules first, then ions, each alphabetically."""
    return sorted(vector.items(), key=lambda kv: (_charge(kv[0]) != 0, kv[0]))


def _amount(value, name):
    value = float(value)
    text = species(name)
    if abs(value - 1) < 1e-9:
        return text
    count = f"{int(round(value))}" if abs(value - round(value)) < 1e-9 else f"{value:g}"
    return f"{count} {text}"
