"""Example plugin: nitrogen reduction to ammonia, N2 + 6 (H+ + e-) -> 2 NH3, on metal top sites.

It adds two reservoir species (N2, NH3), one reaction, seven nitrogen adsorbates and one
mechanism family: the distal associative route, in which the outer N is hydrogenated and
leaves as the first NH3 before the inner one. The adsorbates use the fixed shapes where they
fit (*N, *NH, *NH2) and atom-by-atom templates elsewhere (*N2, *NNH, *NNH2, *NH3). The core is
not changed: placement, the checks of relaxed structures (which follow N-N, N-H and M-N bonds
as they follow O), thermodynamics, pathways and Pourbaix diagrams come from mechanet.

Ammonia leaves twice per turnover, so the turnover marker (``Reaction.release``) is N2, which
is taken up once.

Load it from a project (``plugins: [../plugins/nrr.py]`` in project.yaml, or
``mechanet init SLAB --plugin nrr.py --reaction nrr``). Give N2 and NH3 in the references
file as E0 + G_corr of the gas molecules.
"""

from mechanet.adsorbates import Adsorbate
from mechanet.plugins import Family
from mechanet.reactions import Reaction, Species

N2 = Species("N2", (("N", 2),), 0, "gas", references=("N2",))
NH3 = Species("NH3", (("N", 1), ("H", 3)), 0, "gas", references=("NH3",))


class NRR(Reaction):
    name = "nrr"
    title = "Nitrogen reduction to ammonia"
    net = {"N2": -1, "H+": -6, "NH3": 2}  # cathodic: six electrons taken up
    release = "N2"  # taken up once per turnover (NH3 leaves twice)
    balance_species = {"H": "H+"}
    formation_species = ("N2", "H+")
    potential_scale = "RHE"
    default_potentials = (0.0, -0.4, -0.8)
    required_references = ("H2", "N2", "NH3")
    elementary = (
        {"N2": -1},  # * + N2 -> *N2
        {"H+": -1},  # a hydrogenation: *NxHy + H+ + e- -> *NxHy+1
        {"H+": -1, "NH3": 1},  # *NNH2 + H+ + e- -> *N + NH3
        {"NH3": 1},  # *NH3 -> * + NH3
    )
    coverage_states = {
        "open_metal": ("empty", "N2", "NNH", "NNH2", "N", "NH", "NH2", "NH3"),
        "terminal": ("empty", "N", "NH", "NH2", "NH3"),
    }


# Bond lengths: N-N 1.13 (N2), 1.23 (NNH), 1.33 A (NNH2); N-H 1.02 A; angles near 120 degrees
# at an N with three neighbours and 107 degrees in NH3. +z points away from the surface.
ADSORBATES = (
    Adsorbate(
        "N2",
        {"N": 2},
        "N",
        "template",
        metal_distance_A=1.95,
        atoms=(("n", "N", "", (0.0, 0.0, 1.13)),),
        template_tilts_deg=(0.0, 15.0),
        template_roll_step_deg=360.0,  # linear: rolling it changes nothing
        description="N2 end-on",
    ),
    Adsorbate(
        "NNH",
        {"N": 2, "H": 1},
        "N",
        "template",
        metal_distance_A=1.90,
        atoms=(("n", "N", "", (0.0, 0.0, 1.23)), ("h", "H", "n", (0.883, 0.0, 1.74))),
        description="Diazenyl, bound through N",
    ),
    Adsorbate(
        "NNH2",
        {"N": 2, "H": 2},
        "N",
        "template",
        metal_distance_A=1.85,
        atoms=(
            ("n", "N", "", (0.0, 0.0, 1.33)),
            ("h1", "H", "n", (0.883, 0.0, 1.84)),
            ("h2", "H", "n", (-0.883, 0.0, 1.84)),
        ),
        description="Hydrazido (NNH2), bound through the bare N",
    ),
    Adsorbate("N", {"N": 1}, "N", "atom", metal_distance_A=1.80, description="Nitrido"),
    Adsorbate(
        "NH", {"N": 1, "H": 1}, "N", "XH", xh_A=1.02, tilts="OH_tilts_deg", metal_distance_A=1.85, description="Imido"
    ),
    Adsorbate(
        "NH2",
        {"N": 1, "H": 2},
        "N",
        "water",
        xh_A=1.02,
        tilts="H2O_tilts_deg",
        metal_distance_A=1.95,
        description="Amido",
    ),
    Adsorbate(
        "NH3",
        {"N": 1, "H": 3},
        "N",
        "template",
        metal_distance_A=2.10,
        atoms=(
            ("h1", "H", "", (0.950, 0.0, 0.37)),
            ("h2", "H", "", (-0.475, 0.823, 0.37)),
            ("h3", "H", "", (-0.475, -0.823, 0.37)),
        ),
        template_tilts_deg=(0.0,),
        template_roll_step_deg=30.0,
        description="Ammonia, bound through N",
    ),
)

SOURCES = {
    "skulason2012": {
        "authors": "Skulason, Bligaard, Gudmundsdottir, Studt, Rossmeisl, Abild-Pedersen, Vegge, Jonsson and Norskov",
        "year": 2012,
        "title": "A theoretical evaluation of possible transition metal electro-catalysts for N2 reduction",
        "journal": "Physical Chemistry Chemical Physics 14, 1235-1245",
        "doi": "10.1039/C1CP22271F",
        "url": "https://doi.org/10.1039/C1CP22271F",
        "material": "close-packed and stepped transition-metal surfaces",
        "supported_use": "The associative route of N2 reduction under the computational hydrogen electrode.",
    }
}

ROUTE = (
    # (state, the same step written out, coefficients the family gives)
    ("N2", "* + N2 -> *N2", {"N2": -1}),
    ("NNH", "*N2 + H+ + e- -> *NNH", None),
    ("NNH2", "*NNH + H+ + e- -> *NNH2", None),
    ("N", "*NNH2 + H+ + e- -> *N + NH3", {"NH3": 1}),
    ("NH", "*N + H+ + e- -> *NH", None),
    ("NH2", "*NH + H+ + e- -> *NH2", None),
    ("NH3", "*NH2 + H+ + e- -> *NH3", None),
    ("empty", "*NH3 -> * + NH3", {"NH3": 1}),
)


def distal_route(ctx):
    family = "nrr_distal"
    for site in ctx.sites:
        if ctx.native[site] != "empty":
            continue
        previous = ctx.node({site: "empty"}, family, site)
        steps = []
        for state, text, given in ROUTE:
            node = ctx.node({site: state}, family, site)
            steps.append(ctx.step(previous, node, family, text, given=given, reaction="nrr"))
            previous = node
        ctx.cycle(f"{family}_{site}", steps, family, "Distal associative route on a top site", reaction="nrr")


def register(registry):
    registry.add_species(N2)
    registry.add_species(NH3)
    registry.add_reaction(NRR)
    for adsorbate in ADSORBATES:
        registry.add_adsorbate(adsorbate)
    registry.add_sources(SOURCES)
    registry.add_family(
        Family(
            "nrr_distal",
            distal_route,
            "nrr",
            "N2 to NH3 by the distal associative route on bare top sites",
            ["skulason2012"],
            core=True,
            order=300,
        )
    )
