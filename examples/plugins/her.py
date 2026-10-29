"""Example plugin: the hydrogen evolution reaction, 2 (H+ + e-) -> H2, on metal top sites.

Load it from a project (``plugins: [../plugins/her.py]`` in project.yaml) or package it with
an entry point in the ``mechanet.families`` group pointing at ``register``. It adds one
reservoir species (H2), one reaction (HER), one adsorbate (*H) and one mechanism family
(Volmer-Heyrovsky); everything else (placement, VASP stages, thermodynamics, pathways,
Pourbaix) comes from mechanet unchanged.
"""

from mechanet.adsorbates import Adsorbate
from mechanet.plugins import Family
from mechanet.reactions import Reaction, Species

# A neutral molecule: mu = E0 + G_corr of the H2 entry of the references file.
H2 = Species("H2", (("H", 2),), 0, "gas", references=("H2",))


class HER(Reaction):
    name = "her"
    title = "Hydrogen evolution reaction"
    net = {"H+": -2, "H2": 1}  # 2 H+ + 2 e- -> H2 (cathodic: two electrons taken up)
    release = "H2"
    balance_species = {"H": "H+"}
    formation_species = ("H+",)
    potential_scale = "RHE"
    default_potentials = (0.0, -0.1, -0.2)
    required_references = ("H2",)
    elementary = (
        {"H+": -1},  # Volmer: * + H+ + e- -> *H
        {"H+": -1, "H2": 1},  # Heyrovsky: *H + H+ + e- -> * + H2
        {"H2": 1},  # Tafel: 2 *H -> 2 * + H2
    )


HYDRIDE = Adsorbate("H", {"H": 1}, "H", "atom", metal_distance_A=1.6, description="Adsorbed hydrogen")


def volmer_heyrovsky(ctx):
    family = "her_volmer_heyrovsky"
    for site in ctx.sites:
        if ctx.native[site] != "empty":
            continue
        bare = ctx.node({site: "empty"}, family, site)
        hydride = ctx.node({site: "H"}, family, site)
        volmer = ctx.step(bare, hydride, family, "Volmer: * + H+ + e- -> *H", reaction="her")
        heyrovsky = ctx.step(
            hydride, bare, family, "Heyrovsky: *H + H+ + e- -> * + H2", given={"H2": 1}, reaction="her"
        )
        ctx.cycle(f"{family}_{site}", [volmer, heyrovsky], family, "Volmer-Heyrovsky on a top site", reaction="her")


def register(registry):
    registry.add_species(H2)
    registry.add_reaction(HER)
    registry.add_adsorbate(HYDRIDE)
    registry.add_family(
        Family("her_volmer_heyrovsky", volmer_heyrovsky, "her", "Volmer-Heyrovsky HER on bare top sites",
               ["norskov2004"], core=True, order=300)
    )
