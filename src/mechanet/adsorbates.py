"""Adsorbate vocabulary: what a reactive site can hold.

An adsorbate is described by data, not code: its composition, the element bound to the
site (the *proximal* atom), an optional *distal* heavy atom, how many H sit on each, and
the internal geometry used for starting structures. The builder places it with a scored
orientation search; the analysis recognises it again in relaxed structures from its
signature ``(proximal, distal, H on proximal, H on distal)``.

Built-ins cover OER and CER: ``empty, O, OH, H2O, OO, OOH, Cl, OCl``. Plugins add more
through the ``mechanet.adsorbates`` entry-point group (see docs/plugins.md).

Atom-ID conventions for atoms an adsorbate adds at site ``s`` (stable across campaigns):
``h_s`` first H on the proximal atom, ``w_s`` second H on the proximal atom (water),
``d_s`` distal heavy atom, ``p_s`` H on the distal atom.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GEOMETRIES = ("none", "atom", "XH", "water", "XY", "XYH")


@dataclass(frozen=True)
class Adsorbate:
    name: str
    composition: dict = field(default_factory=dict)
    proximal: str | None = None  # element bound to the site; None for "empty"
    geometry: str = "atom"  # one of GEOMETRIES
    distal: str | None = None  # distal heavy atom (XY, XYH)
    bond_A: float | None = None  # proximal-distal bond (XY, XYH)
    xh_A: float = 0.98  # X-H bond length
    tilts: str | None = None  # orientation_search key with the tilt list
    metal_distance_A: float = 1.95  # M-proximal distance for a new proximal atom
    side_on: bool = False  # can bind side-on (eta2) to a single metal
    description: str = ""

    def __post_init__(self):
        if self.geometry not in GEOMETRIES:
            raise ValueError(f"Adsorbate {self.name}: geometry must be one of {GEOMETRIES}")
        if self.geometry in ("XY", "XYH") and (not self.distal or not self.bond_A):
            raise ValueError(f"Adsorbate {self.name}: XY/XYH geometry needs distal and bond_A")
        expected = self.expected_composition()
        if expected != {k: v for k, v in self.composition.items() if v}:
            raise ValueError(
                f"Adsorbate {self.name}: composition {self.composition} does not match its geometry ({expected})"
            )

    @property
    def h_proximal(self):
        return {"XH": 1, "water": 2}.get(self.geometry, 0)

    @property
    def h_distal(self):
        return 1 if self.geometry == "XYH" else 0

    @property
    def signature(self):
        return (self.proximal, self.distal, self.h_proximal, self.h_distal)

    def expected_composition(self):
        if self.geometry == "none":
            return {}
        comp = {self.proximal: 1}
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
        return self.geometry in ("XH", "water", "XY", "XYH")


BUILTIN = (
    Adsorbate("empty", {}, None, "none", description="Vacant site (no adsorbate)"),
    Adsorbate("O", {"O": 1}, "O", "atom", metal_distance_A=1.75, description="Oxo"),
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
