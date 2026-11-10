"""Example plugin: CO2 reduction to CO, CO2 + 2 (H+ + e-) -> CO + H2O, on metal top sites.

It adds two reservoir species (CO2, CO), one reaction, two adsorbates given atom by atom
(*COOH and *CO, ``geometry="template"``) and one mechanism family (the carboxyl route). The
core is not changed: placement, the checks of relaxed structures (which now follow C as they
follow O), thermodynamics, pathways and Pourbaix diagrams come from mechanet.

Load it from a project (``plugins: [../plugins/co2rr.py]`` in project.yaml, or
``mechanet init SLAB --plugin co2rr.py --reaction co2rr_co``).

Thermodynamics stays the computational hydrogen electrode: give CO2 and CO in the references
file as E0 + G_corr of the gas molecules, with whatever gas-phase correction you apply to them.
"""

from mechanet.adsorbates import Adsorbate
from mechanet.plugins import Family
from mechanet.reactions import Reaction, Species

CO2 = Species("CO2", (("C", 1), ("O", 2)), 0, "gas", references=("CO2",))
CO = Species("CO", (("C", 1), ("O", 1)), 0, "gas", references=("CO",))


class CO2RRtoCO(Reaction):
    name = "co2rr_co"
    title = "CO2 reduction to CO"
    net = {"CO2": -1, "H+": -2, "CO": 1, "H2O": 1}  # cathodic: two electrons taken up
    release = "CO"  # the product whose release closes a cycle
    balance_species = {"H": "H+"}
    formation_species = ("CO2", "H2O", "H+")
    potential_scale = "RHE"
    default_potentials = (-0.2, -0.5, -0.8)
    required_references = ("H2", "H2O", "CO2", "CO")
    elementary = (
        {"CO2": -1, "H+": -1},  # * + CO2 + H+ + e- -> *COOH
        {"H+": -1, "H2O": 1},  # *COOH + H+ + e- -> *CO + H2O
        {"CO": 1},  # *CO -> * + CO
    )
    coverage_states = {"open_metal": ("empty", "CO", "COOH"), "terminal": ("empty", "CO", "COOH")}


# Carboxyl bound through C: C=O 1.22 A and C-OH 1.36 A at 117.5 degrees from the metal-C bond,
# O-H 0.98 A with C-O-H 106 degrees, the H pointing away from the surface.
CARBOXYL = Adsorbate(
    "COOH",
    {"C": 1, "O": 2, "H": 1},
    "C",
    "template",
    metal_distance_A=2.00,
    atoms=(
        ("o1", "O", "", (1.0822, 0.0, 0.5633)),
        ("o2", "O", "", (-1.2063, 0.0, 0.6279)),
        ("h", "H", "o2", (-1.0109, 0.0, 1.5882)),
    ),
    template_tilts_deg=(0.0, 25.0),
    template_roll_step_deg=30.0,
    description="Carboxyl bound through C",
)
CARBONYL = Adsorbate(
    "CO",
    {"C": 1, "O": 1},
    "C",
    "template",
    metal_distance_A=1.85,
    atoms=(("o", "O", "", (0.0, 0.0, 1.15)),),
    template_tilts_deg=(0.0, 15.0),
    template_roll_step_deg=360.0,  # a linear molecule: rolling it changes nothing
    couples=True,  # two neighbouring *CO can bond through C (the first step towards C2 products)
    description="CO on top, bound through C",
)

SOURCES = {
    "peterson2010": {
        "authors": "Peterson, Abild-Pedersen, Studt, Rossmeisl and Norskov",
        "year": 2010,
        "title": "How copper catalyzes the electroreduction of carbon dioxide into hydrocarbon fuels",
        "journal": "Energy & Environmental Science 3, 1311-1315",
        "doi": "10.1039/C0EE00071J",
        "url": "https://doi.org/10.1039/C0EE00071J",
        "material": "Cu(211)",
        "supported_use": "The carboxyl route of CO2 to CO under the computational hydrogen electrode.",
    }
}


def carboxyl_route(ctx):
    family = "co2rr_carboxyl"
    for site in ctx.sites:
        if ctx.native[site] != "empty":
            continue
        bare = ctx.node({site: "empty"}, family, site)
        carboxyl = ctx.node({site: "COOH"}, family, site)
        carbonyl = ctx.node({site: "CO"}, family, site)
        steps = [
            ctx.step(bare, carboxyl, family, "* + CO2 + H+ + e- -> *COOH", given={"CO2": -1}, reaction="co2rr_co"),
            ctx.step(carboxyl, carbonyl, family, "*COOH + H+ + e- -> *CO + H2O", given={"H2O": 1}, reaction="co2rr_co"),
            ctx.step(carbonyl, bare, family, "*CO -> * + CO", given={"CO": 1}, reaction="co2rr_co"),
        ]
        ctx.cycle(f"{family}_{site}", steps, family, "Carboxyl route to CO on a top site", reaction="co2rr_co")


def register(registry):
    registry.add_species(CO2)
    registry.add_species(CO)
    registry.add_reaction(CO2RRtoCO)
    registry.add_adsorbate(CARBOXYL)
    registry.add_adsorbate(CARBONYL)
    registry.add_sources(SOURCES)
    registry.add_family(
        Family(
            "co2rr_carboxyl",
            carboxyl_route,
            "co2rr_co",
            "CO2 to CO through *COOH on bare top sites",
            ["peterson2010"],
            core=True,
            order=300,
        )
    )
