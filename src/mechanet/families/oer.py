"""OER mechanism families.

Sites are 'terminal' (one metal: a terminal O/OH ligand or an open metal site) or
'bridge' (two or more metals). Open metal sites (e.g. CUS Ru on RuO2(110)) behave as
terminal sites whose native state is the bare metal, so the adsorbate-evolution cycle
reads * -> *OH -> *O -> *OOH -> * + O2 there, and vacancy -> OH -> O -> OOH -> vacancy on a
terminal O ligand.

Every cycle carries an ``evidence_scope``: these are hypotheses transferred to the surface
at hand, not demonstrated mechanisms.
"""

from __future__ import annotations

from ..plugins import Family
from ..builder import _two_site_event_orders


def singles(ctx, family, classes):
    for aid in ctx.sites:
        if ctx.site_class(aid) not in classes:
            continue
        n = {s: ctx.node({aid: s}, family, aid) for s in ("empty", "OH", "O", "OOH", "OO")}
        common = [
            (n["empty"], n["OH"], 1, 0, "Refill the vacant site from water (or adsorb OH on a bare metal)."),
            (n["OH"], n["O"], 0, 0, "Paired proton/electron removal."),
            (n["O"], n["OOH"], 1, 0, "Water attack / hydroperoxo formation hypothesis."),
        ]
        ctx.oer_cycle(
            f"{family}_{aid}_AEM",
            common + [(n["OOH"], n["empty"], 0, 1, "O2 release with one PCET; path unverified.")],
            family,
        )
        ctx.oer_cycle(
            f"{family}_{aid}_bound_OO",
            common
            + [
                (n["OOH"], n["OO"], 0, 0, "Explicit bound OO branch."),
                (n["OO"], n["empty"], 0, 1, "Chemical O2 release; not assumed barrierless."),
            ],
            family,
        )


def aem(ctx):
    singles(ctx, "aem", {"terminal"})


def surface_bridge(ctx):
    singles(ctx, "surface_bridge", {"bridge"})


def pairs(ctx, family, selected_pairs):
    for a, b in selected_pairs:
        site = f"{a}+{b}"
        n_oo = ctx.node({a: "O", b: "O"}, family, site)
        coupled = ctx.node({a: "O", b: "O"}, family, site, coupling=(a, b))
        empty = ctx.node({a: "empty", b: "empty"}, family, site)
        shared = ctx.shared_metals(a, b)
        if family == "terminal_bridge_coupling":
            evidence = (
                "Hao-type terminal/bridge coupling at a shared metal centre (shown for Fe in NiFe oxyhydroxide); "
                "transferred to this surface, spectator bridges retained"
                if shared
                else "Exploratory terminal/bridge analogue without a shared metal centre"
            )
        else:
            evidence = "Direct coupling hypothesis; two occupied O sites in this slab, not a demonstrated mechanism"
        # All six interleavings, including O/empty states where one site oxidizes
        # before the other vacancy is refilled.
        for event_order in _two_site_event_orders(a, b):
            occupation = {a: "empty", b: "empty"}
            current = empty
            refill = []
            for active, step in event_order:
                occupation[active] = "OH" if step == 0 else "O"
                next_node = ctx.node(occupation, family, site)
                refill.append(
                    (
                        current,
                        next_node,
                        1 if step == 0 else 0,
                        0,
                        f"Refill {active} from water." if step == 0 else f"Deprotonate OH at {active}.",
                    )
                )
                current = next_node
            if current != n_oo:
                raise AssertionError("Pair regeneration did not recover O/O")
            suffix = "_".join(f"{aid[1:]}{'R' if step == 0 else 'D'}" for aid, step in event_order)
            ctx.oer_cycle(
                f"{family}_{a}_{b}_{suffix}_direct",
                [(n_oo, empty, 0, 1, "Direct O/O coupling and O2 release hypothesis; no intermediate minimum assumed.")]
                + refill,
                family,
                evidence,
            )
            ctx.oer_cycle(
                f"{family}_{a}_{b}_{suffix}_bound",
                [
                    (n_oo, coupled, 0, 0, "O-O bond formation; bidentate product minimum must be checked."),
                    (coupled, empty, 0, 1, "O2 release leaves two local O vacancies."),
                ]
                + refill,
                family,
                evidence,
            )


def pair_coupling(ctx):
    pairs(ctx, "pair_coupling", [(a, b) for a, b in ctx.nearby_pairs() if ctx.site_class(a) == ctx.site_class(b)])


def terminal_bridge_coupling(ctx):
    pairs(
        ctx,
        "terminal_bridge_coupling",
        [
            (a, b)
            for a, b in ctx.nearby_pairs()
            if (ctx.site_class(a) == "terminal") != (ctx.site_class(b) == "terminal")
        ],
    )


def bifunctional(ctx):
    for a, b in ctx.nearby_pairs():
        for donor, acceptor in ((a, b), (b, a)):
            if ctx.site_class(donor) != "terminal":
                continue
            site = f"donor_{donor}_acceptor_{acceptor}"

            def n(d, h, donor=donor, acceptor=acceptor, site=site):
                return ctx.node({donor: d, acceptor: h}, "bifunctional", site)

            ooh_o, oo_oh, empty_oh = n("OOH", "O"), n("OO", "OH"), n("empty", "OH")
            oh_oh, o_oh, ooh_oh = n("OH", "OH"), n("O", "OH"), n("OOH", "OH")
            o_o = n("O", "O")
            empty_o, oh_o = n("empty", "O"), n("OH", "O")
            ctx.oer_cycle(
                f"bifunctional_{donor}_{acceptor}_direct_water_attack",
                [
                    (
                        o_o,
                        empty_oh,
                        1,
                        1,
                        "H-acceptor-assisted water attack and direct O2 formation in ONE step; OOH and "
                        "bound OO are bypassed, so this lumped step can hide an uphill *O -> *OO step "
                        "(compare BF1).",
                    ),
                    (empty_oh, oh_oh, 1, 0, "Donor-site refill from water."),
                    (oh_oh, o_oh, 0, 0, "Donor OH deprotonation."),
                    (o_oh, o_o, 0, 0, "Regenerate deprotonated proton acceptor."),
                ],
                "bifunctional",
                "Song/Bai direct bifunctional motif transferred from FeOOH/NiOOH chemistry to this "
                "surface; condensed step",
            )
            regen = [
                (empty_oh, oh_oh, 1, 0, "Donor-site refill from water."),
                (oh_oh, o_oh, 0, 0, "Donor OH deprotonation."),
                (o_oh, ooh_oh, 1, 0, "Donor hydroperoxo formation."),
                (ooh_oh, ooh_o, 0, 0, "Regenerate the deprotonated proton acceptor."),
            ]
            ctx.oer_cycle(
                f"bifunctional_{donor}_{acceptor}_concerted",
                [(ooh_o, empty_oh, 0, 1, "OOH H-transfer to acceptor coupled to O2 release; mechanistic hypothesis.")]
                + regen,
                "bifunctional",
                "Exploratory OOH-containing H-acceptor branch; not the direct OOH-bypassing Song mechanism",
            )
            ctx.oer_cycle(
                f"bifunctional_{donor}_{acceptor}_resolved",
                [
                    (ooh_o, oo_oh, 0, 0, "Internal H transfer; no proton/electron reservoir exchange."),
                    (oo_oh, empty_oh, 0, 1, "Chemical release after internal H transfer."),
                ]
                + regen,
                "bifunctional",
                "Exploratory stepwise OOH-containing H-transfer resolution; intermediate existence must be checked",
            )
            head = [
                (empty_o, oh_o, 1, 0, "Water refills the donor site (M + *O_A + H2O -> M-OH + *O_A + H+ + e-)."),
                (oh_o, o_o, 0, 0, "Donor OH deprotonation (M-OH + *O_A -> M-O + *O_A + H+ + e-)."),
            ]
            ctx.oer_cycle(
                f"bifunctional_{donor}_{acceptor}_BF1",
                head
                + [
                    (
                        o_o,
                        oo_oh,
                        1,
                        0,
                        "BF-I water attack: M-O + *O_A + H2O -> M-OO + *OH_A + H+ + e- (proton to the acceptor).",
                    ),
                    (oo_oh, empty_o, 0, 1, "BF-I release: M-OO + *OH_A -> M + *O_A + O2 + H+ + e-."),
                ],
                "bifunctional",
                "Exner (Mater. Horiz. 2023) bifunctional mechanism I after Halck et al. 2014, with the "
                "neighbouring O as proton acceptor",
            )
            ctx.oer_cycle(
                f"bifunctional_{donor}_{acceptor}_BF2",
                head
                + [
                    (
                        o_o,
                        ooh_oh,
                        1,
                        0,
                        "BF-II chemical water attack: M-O + *O_A + H2O -> M-OOH + *OH_A (no reservoir exchange).",
                    ),
                    (ooh_oh, ooh_o, 0, 0, "BF-II acceptor deprotonation: *OH_A -> *O_A + H+ + e-."),
                    (ooh_o, empty_o, 0, 1, "BF-II release: M-OOH -> M + O2 + H+ + e-."),
                ],
                "bifunctional",
                "Exner (Mater. Horiz. 2023) bifunctional mechanism II after Halck et al. 2014, with the "
                "neighbouring O as proton acceptor",
            )


def oo_oo(ctx):
    from ..structure import minimum_image, unit

    for a, b in ctx.nearby_pairs():
        # Literature analogue uses two adsorbed OO groups; start with terminal anchors.
        if ctx.site_class(a) != "terminal" or ctx.site_class(b) != "terminal":
            continue
        site = f"{a}+{b}"
        start_node = ctx.node({a: "O", b: "O"}, "oo_oo", site)
        for event_order in _two_site_event_orders(a, b):
            occupation = {a: "O", b: "O"}
            current, links = start_node, []
            for active, step in event_order:
                occupation[active] = "OOH" if step == 0 else "OO"
                next_node = ctx.node(occupation, "oo_oo", site)
                links.append(
                    (
                        current,
                        next_node,
                        1 if step == 0 else 0,
                        0,
                        f"Form OOH on {active}." if step == 0 else f"Deprotonate OOH on {active}.",
                    )
                )
                current = next_node
            links.append(
                (
                    current,
                    start_node,
                    0,
                    1,
                    "Hypothesized O4-mediated release of distal O atoms; IrO2 literature analogue, unverified here.",
                )
            )
            suffix = "_".join(f"{aid[1:]}{'A' if step == 0 else 'D'}" for aid, step in event_order)
            ctx.oer_cycle(
                f"oo_oo_{a}_{b}_{suffix}",
                links,
                "oo_oo",
                "IrO2 OO/OO motif transferred to this pair; all six precursor-event orders, microscopic O4 "
                "path unverified",
            )
        # Separate search seed, NEVER inserted as a normal minimum or submitted for frequencies.
        occ = {a: "OO", b: "OO"}
        st, _, _, _ = ctx.structure(occ, None, ctx.default_placement(occ, None, {}), {})
        ia, ib = st.ids.index(f"d_{a}"), st.ids.index(f"d_{b}")
        pa, pb = st.positions[ia].copy(), st.positions[ib].copy()
        delta = minimum_image(pb - pa, st.cell)
        center = pa + 0.5 * delta
        st.positions[ia] = center - 0.80 * unit(delta)
        st.positions[ib] = center + 0.80 * unit(delta)
        motif_id = f"O4_search_{a}_{b}"
        ctx.write_motif(
            motif_id,
            st,
            {
                "id": motif_id,
                "role": "transition_state_search_guess",
                "not_a_minimum": True,
                "not_in_normal_job_list": True,
                "atom_ids": st.ids,
                "source_ids": ["binninger2022"],
                "proximal_O": [a, b],
                "distal_O_released": [f"d_{a}", f"d_{b}"],
                "notes": [
                    "O4 is a transition-state search hypothesis from IrO2; no barrier or minimum is implied here.",
                    "Use an explicit path/TS-search method; ordinary minimum optimization cannot validate this motif.",
                ],
            },
        )


def hydrated(ctx):
    # A bound-water refill alternative on every selected site. Bridge-water guesses are
    # exploratory and may deprotonate or rebind.
    for aid in ctx.sites:
        n = {status: ctx.node({aid: status}, "hydrated", aid) for status in ("empty", "H2O", "OH", "O", "OOH", "OO")}
        prefix = [
            (n["empty"], n["H2O"], 1, 0, "Molecular water fills the vacant site without PCET."),
            (n["H2O"], n["OH"], 0, 0, "First bound-water deprotonation."),
            (n["OH"], n["O"], 0, 0, "OH deprotonation."),
            (n["O"], n["OOH"], 1, 0, "Water attack to form OOH."),
        ]
        scope = "Explicit bound-water refinement of a local CHE cycle; minimum and proton arrangement are unverified"
        ctx.oer_cycle(
            f"hydrated_{aid}_bound_water_AEM",
            prefix + [(n["OOH"], n["empty"], 0, 1, "OOH oxidation and O2 release.")],
            "hydrated",
            scope,
        )
        ctx.oer_cycle(
            f"hydrated_{aid}_bound_water_OO",
            prefix
            + [(n["OOH"], n["OO"], 0, 0, "Explicit OO branch."), (n["OO"], n["empty"], 0, 1, "Chemical O2 release.")],
            "hydrated",
            scope,
        )
    for a, b in ctx.nearby_pairs():
        for donor, acceptor in ((a, b), (b, a)):
            if ctx.site_class(donor) != "terminal":
                continue
            site = f"hydrated_donor_{donor}_neighbor_{acceptor}"

            def n(d, h, donor=donor, acceptor=acceptor, site=site):
                return ctx.node({donor: d, acceptor: h}, "hydrated", site)

            oh_oh, o_oh = n("OH", "OH"), n("O", "OH")
            ooh_water, ooh_oh = n("OOH", "H2O"), n("OOH", "OH")
            oo_water, oo_oh, empty_oh = n("OO", "H2O"), n("OO", "OH"), n("empty", "OH")
            o_water = n("O", "H2O")
            tail = [
                (oo_oh, empty_oh, 0, 1, "Release O2 from the donor while neighboring OH remains."),
                (empty_oh, oh_oh, 1, 0, "Refill the donor from water and one PCET."),
            ]
            head = [
                (oh_oh, o_oh, 0, 0, "Oxidize donor OH to O."),
                (
                    o_oh,
                    ooh_water,
                    1,
                    0,
                    "Local water addition and H transfer; neighboring OH becomes bound H2O, no PCET.",
                ),
            ]
            resolutions = {
                "via_OOH_OH": [
                    (ooh_water, ooh_oh, 0, 0, "First one-PCET resolution: deprotonate neighboring H2O."),
                    (ooh_oh, oo_oh, 0, 0, "Second one-PCET resolution: deprotonate OOH."),
                ],
                "via_OO_H2O": [
                    (ooh_water, oo_water, 0, 0, "First one-PCET resolution: deprotonate donor OOH."),
                    (oo_water, oo_oh, 0, 0, "Second one-PCET resolution: deprotonate neighboring H2O."),
                ],
            }
            scope = (
                "Shi-inspired local hydration/H-transfer bookkeeping transferred to this background; the original "
                "superoxohydroxide phase is not asserted; one-PCET subdivisions are hypotheses"
            )
            for name, resolution in resolutions.items():
                ctx.oer_cycle(f"hydrated_{donor}_{acceptor}_{name}", head + resolution + tail, "hydrated", scope)
                tautomer_head = [
                    (oh_oh, o_water, 0, 0, "Internal proton relocation OH/OH -> O/H2O; no reservoir exchange."),
                    (o_water, ooh_water, 1, 0, "Water attack with one PCET on the O/H2O tautomer."),
                ]
                ctx.oer_cycle(
                    f"hydrated_{donor}_{acceptor}_tautomer_{name}",
                    tautomer_head + resolution + tail,
                    "hydrated",
                    "Additional local proton-tautomer hypothesis; " + scope,
                )
            ctx.edge(o_water, oh_oh, 0, 0, "hydrated", "Reverse internal proton transfer; path not evaluated.")


FAMILIES = (
    Family(
        "aem",
        aem,
        "oer",
        "Adsorbate evolution (mononuclear) on terminal ligand and open metal sites",
        ["rossmeisl2005", "rossmeisl2007", "man2011", "dionigi2020", "shi2025"],
        core=True,
        order=10,
    ),
    Family(
        "surface_bridge",
        surface_bridge,
        "oer",
        "Vacancy refill / oxidation cycle on bridging O sites",
        ["dionigi2020"],
        core=True,
        order=20,
    ),
    Family(
        "pair_coupling",
        pair_coupling,
        "oer",
        "O/O coupling between two sites of the same class within the pair cutoff",
        ["hao2021", "exner2023"],
        core=True,
        order=30,
    ),
    Family(
        "terminal_bridge_coupling",
        terminal_bridge_coupling,
        "oer",
        "Coupling of a terminal-class O with a neighbouring bridge O",
        ["hao2021"],
        core=True,
        order=40,
    ),
    Family(
        "bifunctional",
        bifunctional,
        "oer",
        "Bifunctional cycles with a neighbouring proton acceptor, including Exner's BF-I and BF-II",
        ["song2019", "bai2021", "halck2014", "exner2023"],
        core=True,
        order=50,
    ),
    Family(
        "hydrated",
        hydrated,
        "oer",
        "Bound-water refill and local H-transfer variants",
        ["shi2025"],
        core=True,
        order=60,
    ),
    Family(
        "oo_oo",
        oo_oo,
        "oer",
        "OO/OO recombination (O4) hypothesis on terminal pairs; writes separate TS-search seeds",
        ["binninger2022", "exner2023"],
        core=False,
        order=70,
    ),
)


def register(registry):
    for family in FAMILIES:
        registry.add_family(family)
