"""Adsorbate vocabulary: what a reactive site can hold.

An adsorbate is described by data, not code: its composition, the element bound to the
site (the *proximal* atom) and the internal geometry used for starting structures. The
builder places it with a scored orientation search; the analysis recognises it again in
relaxed structures from its signature.

Two kinds of geometry:

* six fixed shapes for small adsorbates (``atom``, ``XH``, ``water``, ``XY``, ``XYH``,
  ``none``), with the signature ``(proximal, distal, H on proximal, H on distal)``;
* ``template``: any small molecule given atom by atom (element, the atom it is bonded to,
  position), such as a carboxyl *COOH bound through C. Its signature is the canonical tree of
  its atoms rooted at the proximal one (``("graph", "C(O,O(H))")``).

Built-ins cover OER and CER: ``empty, O, OH, H2O, OO, OOH, Cl, OCl``. Plugins add more
through the ``mechanet.adsorbates`` entry-point group (see docs/plugins.md).

Atom-ID conventions for atoms an adsorbate adds at site ``s`` (stable across campaigns):
``h_s`` first H on the proximal atom, ``w_s`` second H on the proximal atom (water),
``d_s`` distal heavy atom, ``p_s`` H on the distal atom; ``<label>_s`` for a template atom.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

GEOMETRIES = ("none", "atom", "XH", "water", "XY", "XYH", "template")
# Elements of the built-in oxygen and chlorine chemistry. Other elements that registered
# adsorbates bring (C, N, S, ...) extend the bond checks and the bookkeeping to themselves.
CORE_ELEMENTS = frozenset({"O", "H", "Cl"})
_LABEL = re.compile(r"[a-z][a-z0-9]*$")


def tree_key(element, children=()):
    """Canonical text of a rooted tree of atoms: ``C(O,O(H))`` for a carboxyl bound through C."""
    return element + ("(" + ",".join(sorted(children)) + ")" if children else "")


def extra_elements(adsorbates):
    """Elements the adsorbates use beyond O, H and Cl (C and N for CO2 or N2 reduction, say)."""
    return frozenset(e for a in adsorbates for e, n in a.composition.items() if n and e not in CORE_ELEMENTS)


def binds_through_hydrogen(adsorbates):
    """True when an adsorbate is bound to its site through H (*H of hydrogen evolution): the
    checks of relaxed structures then watch metal-H contacts like the other metal-ligand ones."""
    return any(a.proximal == "H" for a in adsorbates)


@dataclass(frozen=True)
class Adsorbate:
    name: str
    composition: dict = field(default_factory=dict)
    # PENDING(two-site-adsorbates): one binding atom on one site. An adsorbate bound through two
    # atoms on two sites (formate through both O, CO on a bridge) cannot be written: docs/pending.md.
    proximal: str | None = None  # element bound to the site; None for "empty"
    geometry: str = "atom"  # one of GEOMETRIES
    distal: str | None = None  # distal heavy atom (XY, XYH)
    bond_A: float | None = None  # proximal-distal bond (XY, XYH)
    xh_A: float = 0.98  # X-H bond length
    tilts: str | None = None  # orientation_search key with the tilt list
    metal_distance_A: float = 1.95  # M-proximal distance for a new proximal atom
    side_on: bool = False  # can bind side-on (eta2) to a single metal
    description: str = ""
    # geometry "template": the atoms beyond the proximal one, in order, as (label, element,
    # label of the atom it is bonded to or "" for the proximal atom, (x, y, z) in A). The
    # proximal atom is at the origin, +z points away from the surface along the site's bond
    # and x is any direction across it. The search turns the molecule by these tilts from the
    # site's outward direction, every azimuth, and rolls about its own axis in this step.
    atoms: tuple = ()
    template_tilts_deg: tuple = (0.0, 25.0)
    template_roll_step_deg: float = 30.0
    # Two neighbouring sites can bond through their proximal atoms: a coupled pair, written by a
    # family with ``coupling=(site_a, site_b)`` (O-O in the OER coupling families, C-C between two
    # *CO). With ``couples`` on both adsorbates the checks read a relaxed pair as coupled, the step
    # rules propose the coupling and the enumeration writes the pair; a family may write any pair.
    # ``coupled_bond_A`` is the bond in the start (default: the covalent radii of the two atoms;
    # 1.45 A for O-O).
    couples: bool = False
    coupled_bond_A: float | None = None

    def __post_init__(self):
        if self.geometry not in GEOMETRIES:
            raise ValueError(f"Adsorbate {self.name}: geometry must be one of {GEOMETRIES}")
        if self.geometry in ("XY", "XYH") and (not self.distal or not self.bond_A):
            raise ValueError(f"Adsorbate {self.name}: XY/XYH geometry needs distal and bond_A")
        if self.geometry == "template":
            self._check_template()
        elif self.atoms:
            raise ValueError(f"Adsorbate {self.name}: atoms are given only with geometry 'template'")
        expected = self.expected_composition()
        if expected != {k: v for k, v in self.composition.items() if v}:
            raise ValueError(
                f"Adsorbate {self.name}: composition {self.composition} does not match its geometry ({expected})"
            )

    def _check_template(self):
        if not self.proximal:
            raise ValueError(f"Adsorbate {self.name}: a template needs the proximal element")
        if not self.atoms:
            raise ValueError(f"Adsorbate {self.name}: a template lists the atoms beyond the proximal one")
        seen = set()
        for entry in self.atoms:
            if len(entry) != 4:
                raise ValueError(f"Adsorbate {self.name}: template atom {entry!r} is not (label, element, parent, xyz)")
            label, element, parent, position = entry
            if not isinstance(label, str) or not _LABEL.match(label) or label in seen:
                raise ValueError(
                    f"Adsorbate {self.name}: template label {label!r} must be a new lowercase name such as o1 or h2"
                )
            if parent and parent not in seen:
                raise ValueError(
                    f"Adsorbate {self.name}: {label} is bonded to {parent!r}, which is not listed before it"
                )
            if not element or len(position) != 3 or not all(isinstance(x, (int, float)) for x in position):
                raise ValueError(f"Adsorbate {self.name}: template atom {label} needs an element and three coordinates")
            seen.add(label)
        if not self.template_tilts_deg or float(self.template_roll_step_deg) <= 0:
            raise ValueError(f"Adsorbate {self.name}: template tilts and a positive roll step are needed")

    @property
    def h_proximal(self):
        if self.geometry == "template":
            return sum(1 for _, element, parent, _ in self.atoms if element == "H" and not parent)
        return {"XH": 1, "water": 2}.get(self.geometry, 0)

    @property
    def h_distal(self):
        if self.geometry == "template":
            return sum(1 for _, element, parent, _ in self.atoms if element == "H" and parent)
        return 1 if self.geometry == "XYH" else 0

    @property
    def graph_key(self):
        """Canonical tree of a template's atoms rooted at the proximal one (None for other shapes)."""
        if self.geometry != "template":
            return None
        children = {"": []}
        for label, element, parent, _ in self.atoms:
            children.setdefault(parent, []).append((label, element))
            children.setdefault(label, [])

        def key(label, element):
            return tree_key(element, [key(child, sub) for child, sub in children[label]])

        return key("", self.proximal)

    @property
    def signature(self):
        if self.geometry == "template":
            return ("graph", self.graph_key)
        return (self.proximal, self.distal, self.h_proximal, self.h_distal)

    def expected_composition(self):
        if self.geometry == "none":
            return {}
        comp = {self.proximal: 1}
        if self.geometry == "template":
            for _, element, _, _ in self.atoms:
                comp[element] = comp.get(element, 0) + 1
            return comp
        if self.distal:
            comp[self.distal] = comp.get(self.distal, 0) + 1
        n_h = self.h_proximal + self.h_distal
        if n_h:
            comp["H"] = comp.get("H", 0) + n_h
        return comp

    def count(self, element):
        return int(self.composition.get(element, 0))

    @property
    def oriented(self):
        """Needs an orientation search (anything beyond a bare proximal atom)."""
        return self.geometry in ("XH", "water", "XY", "XYH", "template")


BUILTIN = (
    Adsorbate("empty", {}, None, "none", description="Vacant site (no adsorbate)"),
    Adsorbate("O", {"O": 1}, "O", "atom", metal_distance_A=1.75, couples=True, description="Oxo"),
    Adsorbate("OH", {"O": 1, "H": 1}, "O", "XH", tilts="OH_tilts_deg", metal_distance_A=1.95, description="Hydroxo"),
    Adsorbate(
        "H2O", {"O": 1, "H": 2}, "O", "water", tilts="H2O_tilts_deg", metal_distance_A=2.15, description="Bound water"
    ),
    Adsorbate(
        "OO",
        {"O": 2},
        "O",
        "XY",
        distal="O",
        bond_A=1.34,
        tilts="OO_tilts_deg",
        metal_distance_A=1.90,
        side_on=True,
        description="Superoxo/peroxo O-O (end-on; side-on variant on single-metal sites)",
    ),
    Adsorbate(
        "OOH",
        {"O": 2, "H": 1},
        "O",
        "XYH",
        distal="O",
        bond_A=1.43,
        tilts="OO_tilts_deg",
        metal_distance_A=1.95,
        description="Hydroperoxo",
    ),
    Adsorbate("Cl", {"Cl": 1}, "Cl", "atom", metal_distance_A=2.35, description="Chloro"),
    Adsorbate(
        "OCl",
        {"O": 1, "Cl": 1},
        "O",
        "XY",
        distal="Cl",
        bond_A=1.65,
        tilts="OO_tilts_deg",
        metal_distance_A=1.90,
        description="Hypochlorite-like O-Cl on an oxo site",
    ),
)


def register(registry):
    for adsorbate in BUILTIN:
        registry.add_adsorbate(adsorbate)
