"""CER mechanism families: chlorine evolution, 2 Cl- -> Cl2 + 2 e-.

* ``cer_volmer_heyrovsky``  * + Cl- -> *Cl + e-;  *Cl + Cl- -> * + Cl2 + e-   (bare metal sites)
* ``cer_volmer_tafel``      two Volmer steps on neighbouring bare metal sites, then 2 *Cl -> 2 * + Cl2
* ``cer_ocl``               *O + Cl- -> *OCl + e-;  *OCl + Cl- -> *O + Cl2 + e-  (oxygen-covered sites)

On rutile oxides the surface Pourbaix diagram decides which of these applies: under CER
conditions the cus sites of RuO2(110) and IrO2(110) are oxygen-covered, so the OCl route is
the usual one (Hansen et al. 2010). Run the ``coverage`` family and ``mechanet pourbaix``
first when the termination is not known.
"""

from __future__ import annotations

from ..plugins import Family

EVIDENCE_HANSEN = "CER intermediates on rutile (110) cus sites (Hansen et al. 2010); transferred to this surface"


def _bare_sites(ctx):
    return [aid for aid, site in ctx.sites.items() if site["native"] == "empty"]


def volmer_heyrovsky(ctx):
    family = "cer_volmer_heyrovsky"
    for aid in _bare_sites(ctx):
        bare = ctx.node({aid: "empty"}, family, aid)
        chloro = ctx.node({aid: "Cl"}, family, aid)
        volmer = ctx.step(bare, chloro, family, "Volmer: * + Cl- -> *Cl + e-", reaction="cer")
        heyrovsky = ctx.step(
            chloro, bare, family, "Heyrovsky: *Cl + Cl- -> * + Cl2 + e-", given={"Cl2": 1}, reaction="cer"
        )
        ctx.cycle(f"{family}_{aid}", [volmer, heyrovsky], family, EVIDENCE_HANSEN, reaction="cer")


def volmer_tafel(ctx):
    family = "cer_volmer_tafel"
    bare = set(_bare_sites(ctx))
    for a, b in ctx.nearby_pairs():
        if a not in bare or b not in bare:
            continue
        site = f"{a}+{b}"
        start = ctx.node({a: "empty", b: "empty"}, family, site)
        one = ctx.node({a: "Cl", b: "empty"}, family, site)
        two = ctx.node({a: "Cl", b: "Cl"}, family, site)
        e1 = ctx.step(start, one, family, f"Volmer on {a}.", reaction="cer")
        e2 = ctx.step(one, two, family, f"Volmer on {b}.", reaction="cer")
        e3 = ctx.step(
            two,
            start,
            family,
            "Tafel recombination: 2 *Cl -> 2 * + Cl2 (chemical step).",
            given={"Cl2": 1},
            reaction="cer",
        )
        ctx.cycle(
            f"{family}_{a}_{b}",
            [e1, e2, e3],
            family,
            "Volmer-Tafel recombination of neighbouring Cl; not established on oxides",
            reaction="cer",
        )


def ocl(ctx):
    family = "cer_ocl"
    for aid, site in ctx.sites.items():
        if ctx.site_class(aid) != "terminal" and site["native"] != "O":
            continue
        oxo = ctx.node({aid: "O"}, family, aid)
        hypo = ctx.node({aid: "OCl"}, family, aid)
        e1 = ctx.step(oxo, hypo, family, "*O + Cl- -> *OCl + e-", reaction="cer")
        e2 = ctx.step(hypo, oxo, family, "*OCl + Cl- -> *O + Cl2 + e-", given={"Cl2": 1}, reaction="cer")
        ctx.cycle(f"{family}_{aid}", [e1, e2], family, EVIDENCE_HANSEN, reaction="cer")


FAMILIES = (
    Family(
        "cer_volmer_heyrovsky",
        volmer_heyrovsky,
        "cer",
        "Cl adsorption and Heyrovsky release on bare metal sites",
        ["hansen2010"],
        core=True,
        order=110,
    ),
    Family(
        "cer_volmer_tafel",
        volmer_tafel,
        "cer",
        "Two Volmer steps and Tafel recombination on neighbouring bare sites",
        ["hansen2010"],
        core=False,
        order=120,
    ),
    Family("cer_ocl", ocl, "cer", "OCl intermediate on oxygen-covered sites", ["hansen2010"], core=True, order=130),
)


def register(registry):
    for family in FAMILIES:
        registry.add_family(family)
