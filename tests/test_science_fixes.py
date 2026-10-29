"""Turnover markers (consumed or released), windowed limiting potentials, transient intermediates,
exact formation vectors, activities, temperature guards and ideal-gas corrections."""

import itertools
import math
import random
import unittest
from pathlib import Path

import networkx as nx

from mechanet.analysis import releasing_nodes, transient_nodes
from mechanet.pathways import StepRules, build_graph, enumerate_cycles, minimum_limiting_cycle
from mechanet.plugins import Registry, default_registry
from mechanet.pourbaix import formation_lines, release_downhill
from mechanet.reactions import (
    CER,
    OER,
    SPECIES,
    Reaction,
    Species,
    add,
    clean,
    electrons,
    element_change,
    validate_reaction,
)
from mechanet.references import gas_correction
from mechanet.thermo import (
    KB_EV_K,
    Conditions,
    cycle_metrics,
    limiting_potential,
    reservoir_chemical_potentials,
    species_free_energy,
    vector_free_energy,
)

ROOT = Path(__file__).resolve().parents[1]
HER_PLUGIN = ROOT / "examples" / "plugins" / "her.py"
KT = KB_EV_K * 298.15


class ORR(Reaction):
    """Associative oxygen reduction, O2 + 4 (H+ + e-) -> 2 H2O: the marker O2 is consumed."""

    name = "orr"
    title = "Oxygen reduction reaction"
    net = {"O2": -1, "H+": -4, "H2O": 2}
    release = "O2"
    balance_species = {"H": "H+"}
    formation_species = ("H2O", "H+")
    potential_scale = "RHE"
    default_potentials = (0.9, 0.7)
    elementary = ({"O2": -1, "H+": -1}, {"H+": -1, "H2O": 1}, {"H+": -1})


def edge(key, start, end, vector):
    return {"id": key, "from": start, "to": end, "reservoirs": clean(vector)}


class MarkerTests(unittest.TestCase):
    def test_declarations_are_checked(self):
        for reaction in (OER(), CER(), ORR()):
            self.assertTrue(validate_reaction(reaction))
            self.assertTrue(reaction.closed_walks_are_turnovers)
        self.assertEqual(ORR().marker_sign, -1)
        self.assertEqual(OER().marker_sign, 1)

        class WaterMarker(ORR):
            name = "orr_water"
            release = "H2O"  # two per turnover: not a marker

        with self.assertRaisesRegex(ValueError, r"coefficient \+1.*'O2'"):
            validate_reaction(WaterMarker())
        with self.assertRaisesRegex(ValueError, "release = 'H2O'"):
            Registry().add_reaction(WaterMarker)

        class Unbalanced(ORR):
            name = "unbalanced"
            net = {"O2": -1, "H+": -4, "H2O": 1}

        with self.assertRaisesRegex(ValueError, "does not balance"):
            validate_reaction(Unbalanced())

    def test_species_with_several_balanced_combinations(self):
        SPECIES_BEFORE = dict(SPECIES)
        try:
            Registry().add_species(Species("H2O2", (("H", 2), ("O", 2)), 0, "aqueous", ("H2O2",)))

            class TwoElectronORR(ORR):
                name = "orr2"
                net = {"O2": -1, "H+": -2, "H2O2": 1}

            reaction = TwoElectronORR()
            self.assertTrue(validate_reaction(reaction))
            # O2, H+, H2O2 and H2O have two independent balanced combinations (2e- and 4e- ORR)
            self.assertFalse(reaction.closed_walks_are_turnovers)
            edges = {
                "e1": edge("e1", "a", "b", {"O2": -1, "H+": -1}),
                "e2": edge("e2", "b", "a", {"H+": -1, "H2O2": 1}),
            }
            self.assertIsNone(minimum_limiting_cycle(edges, {"e1": -0.5, "e2": -0.5}, reaction))
        finally:
            SPECIES.clear()
            SPECIES.update(SPECIES_BEFORE)


class OxygenReductionTests(unittest.TestCase):
    """A 4-step associative ORR: enumeration, the potential sweep and the cycle metrics agree."""

    def setUp(self):
        self.edges = {
            "e1": edge("e1", "ref", "ooh", {"O2": -1, "H+": -1}),
            "e2": edge("e2", "ooh", "o", {"H+": -1, "H2O": 1}),
            "e3": edge("e3", "o", "oh", {"H+": -1}),
            "e4": edge("e4", "oh", "ref", {"H+": -1, "H2O": 1}),
            "back": edge("back", "ooh", "ref", {"O2": 1, "H+": 1}),  # gives O2 off: never in an ORR cycle
        }
        # DeltaG at 0 V vs RHE, summing to -4.92 eV over the cycle
        self.dz = {"e1": -0.70, "e2": -1.60, "e3": -1.50, "e4": -1.12, "back": 0.70}

    def test_enumeration_sweep_and_metrics_agree(self):
        cycles, truncated = enumerate_cycles(build_graph(self.edges.values()), ORR())
        self.assertFalse(truncated)
        self.assertEqual(cycles, [("e1", "e2", "e3", "e4")])  # O2 uptake first
        best = minimum_limiting_cycle(self.edges, self.dz, ORR())
        self.assertEqual(best["edge_ids"], ["e1", "e2", "e3", "e4"])
        self.assertAlmostEqual(best["limiting_potential_V"], 0.70, places=9)
        keys = best["edge_ids"]
        metrics = cycle_metrics(
            [{"reservoirs": self.edges[k]["reservoirs"]} for k in keys],
            [self.dz[k] for k in keys],
            0.7,
            [electrons(self.edges[k]["reservoirs"]) for k in keys],
            1.23,
            release="O2",
        )
        self.assertEqual(metrics["direction"], "cathodic")
        self.assertEqual(metrics["rotation"], 0)
        self.assertAlmostEqual(metrics["limiting_potential_V"], 0.70, places=9)
        self.assertAlmostEqual(metrics["thermodynamic_overpotential_V"], 0.53, places=9)
        # declared in another order, the cycle is rotated so that O2 is taken up first
        rotated = ["e3", "e4", "e1", "e2"]
        metrics = cycle_metrics(
            [{"reservoirs": self.edges[k]["reservoirs"]} for k in rotated],
            [self.dz[k] for k in rotated],
            0.7,
            [electrons(self.edges[k]["reservoirs"]) for k in rotated],
            1.23,
            release="O2",
        )
        self.assertEqual(metrics["rotation"], 2)

    def test_only_a_chemical_product_release_makes_a_node_transient(self):
        nodes = {n: {} for n in ("ref", "ooh", "oo", "oo_image")}
        nodes["oo_image"] = {"computed_by": "oo"}

        def step(start, kind, dg, electrons):
            return {
                "edge_id": f"{start}_release",
                "from": start,
                "to": "ref",
                "vector": {"O2": 1, **({"H+": 1} if electrons else {})},
                "species": "O2",
                "proposed": start == "oo_image",
                "deltaG0_eV": dg,
                "electrons": electrons,
                "chemical": kind == "chemical",
            }

        steps = [step("oo_image", "chemical", -0.08, 0), step("ooh", "electrochemical", 1.12, 1)]
        transient = transient_nodes(["ref", "ooh", "oo"], nodes, steps)
        self.assertEqual(list(transient), ["oo"])  # found through its symmetry image, a step no family wrote
        self.assertIn("a step no family wrote", transient["oo"])
        self.assertEqual(releasing_nodes(["ref", "ooh", "oo"], nodes, steps, 1.0), {})
        self.assertEqual(list(releasing_nodes(["ref", "ooh", "oo"], nodes, steps, 1.2)), ["ooh"])
        # the clean surface taking up a consumed reactant (O2 in ORR) is never transient
        self.assertEqual(
            transient_nodes(["ref"], nodes, [{**steps[0], "from": "ref", "deltaG0_eV": -1.0, "chemical": False}]), {}
        )


class LimitingPotentialTests(unittest.TestCase):
    def test_window_with_a_step_against_the_direction(self):
        # anodic cycle with one cathodic step: downhill between 1.45 and 1.50 V
        dg = [1.0, 1.45, 1.2, -1.5, 2.77]
        n = [1, 1, 1, -1, 2]
        limit, window = limiting_potential(dg, n)
        self.assertAlmostEqual(limit, 1.45)
        self.assertEqual(window, (1.45, 1.5))
        # the reverse step stops being downhill before every forward step is: no window
        self.assertEqual(limiting_potential([1.0, 1.6, 0.8, -1.5, 3.02], n), (None, None))
        # an uphill chemical step blocks the cycle at every potential
        self.assertEqual(limiting_potential([1.0, 0.1, 3.82], [1, 0, 3]), (None, None))
        # cathodic mirror image: the anodic step is downhill above -0.5 V
        limit, window = limiting_potential([-0.7, -1.6, -0.5, -2.62], [-1, -1, 1, -2])
        self.assertAlmostEqual(limit, 0.7)
        self.assertEqual(window, (-0.5, 0.7))

    def test_reviewer_cycle_with_a_reverse_step_gets_a_limiting_potential(self):
        dg = [1.0, 1.3, 1.2, -1.5, 0.0, 2.92]
        n = [1, 1, 1, -1, 0, 2]
        metrics = cycle_metrics([{"reservoirs": {}}] * 6, dg, 1.47, n, 1.23)
        self.assertTrue(all(x <= 0 for x in metrics["step_deltaG_eV"]))
        self.assertAlmostEqual(metrics["limiting_potential_V"], 1.46)
        self.assertEqual(metrics["steps_against_reaction_direction"], 1)
        self.assertEqual(metrics["downhill_potential_range_V"], [1.46, 1.5])


def isomer_network():
    """A one-site OER network in which *OH, *O, *OOH and *OO each come as two isomers, joined by every
    elementary step and its reverse: cycles may climb back down (O -> OH') before going on."""
    composition = {"ref": (0, 0), "W0": (1, 2)}
    for name, comp in (("OH", (1, 1)), ("O", (1, 0)), ("OOH", (2, 1)), ("OO", (2, 0))):
        composition[f"{name}0"] = composition[f"{name}1"] = comp
    vectors = [clean(v) for v in OER.elementary]
    vectors += [clean({s: -x for s, x in v.items()}) for v in vectors]
    edges = {}
    for a, b in itertools.permutations(composition, 2):
        change = {"O": composition[b][0] - composition[a][0], "H": composition[b][1] - composition[a][1]}
        change = {k: v for k, v in change.items() if v}
        vector = next((v for v in vectors if v.get("O2", 0) >= 0 and element_change(v) == change), None)
        if vector is not None:
            edges[f"{a}>{b}"] = edge(f"{a}>{b}", a, b, vector)
    return list(composition), edges


class SweepWithReverseStepsTests(unittest.TestCase):
    def setUp(self):
        self.nodes, self.edges = isomer_network()
        graph = nx.DiGraph([(e["from"], e["to"]) for e in self.edges.values()])
        by_pair = {(e["from"], e["to"]): k for k, e in self.edges.items()}
        self.turnovers = []
        for cycle in nx.simple_cycles(graph):  # independent of mechanet's enumeration
            keys = [by_pair[pair] for pair in zip(cycle, cycle[1:] + cycle[:1])]
            if add(*(self.edges[k]["reservoirs"] for k in keys)) == clean(OER.net):
                self.turnovers.append(keys)

    def test_enumeration_finds_every_turnover_with_reverse_steps(self):
        cycles, truncated = enumerate_cycles(build_graph(self.edges.values()), OER(), max_length=len(self.nodes) + 1)
        self.assertFalse(truncated)
        self.assertEqual(len(cycles), len(self.turnovers))
        self.assertEqual({frozenset(c) for c in cycles}, {frozenset(c) for c in self.turnovers})
        against = {k for k, e in self.edges.items() if electrons(e["reservoirs"]) < 0}
        self.assertTrue(any(set(c) & against for c in cycles))

    def test_sweep_equals_brute_force(self):
        windows = 0
        for seed in range(25):
            rnd = random.Random(seed)
            g = {nid: rnd.uniform(-1.5, 1.5) for nid in self.nodes}
            dz = {k: g[e["to"]] - g[e["from"]] + 1.23 * electrons(e["reservoirs"]) for k, e in self.edges.items()}
            limits = {}
            for keys in self.turnovers:
                limit, window = limiting_potential(
                    [dz[k] for k in keys], [electrons(self.edges[k]["reservoirs"]) for k in keys]
                )
                if limit is not None:
                    limits[tuple(keys)] = limit
                    windows += window[1] is not None
            best = minimum_limiting_cycle(self.edges, dz, OER())
            if not limits:
                self.assertIsNone(best)
                continue
            expected = min(limits.values())
            self.assertAlmostEqual(best["limiting_potential_V"], expected, places=9)
            keys = best["edge_ids"]
            found, _ = limiting_potential([dz[k] for k in keys], [electrons(self.edges[k]["reservoirs"]) for k in keys])
            self.assertAlmostEqual(found, expected, places=9)
        self.assertGreater(windows, 0)  # some downhill cycles contain a step against the direction

    def test_the_only_route_climbs_back_down(self):
        # ref -> OH -> O -(+ H+ + e-)-> OH' -> O' -> OOH -> ref + O2: four electrons, one of them taken back
        steps = [
            ("ref", "OH0", {"H2O": -1, "H+": 1}),
            ("OH0", "O0", {"H+": 1}),
            ("O0", "OH1", {"H+": -1}),
            ("OH1", "O1", {"H+": 1}),
            ("O1", "OOH0", {"H2O": -1, "H+": 1}),
            ("OOH0", "ref", {"O2": 1, "H+": 1}),
        ]
        edges = {f"{a}>{b}": edge(f"{a}>{b}", a, b, v) for a, b, v in steps}
        for g_oh1, expected in ((-0.43, 1.45), (-0.33, None)):
            # node free energies at 1.23 V; O0 -> OH1 is downhill below 1.5 V (1.4 V for the second set)
            g = {"ref": 0.0, "OH0": -0.23, "O0": -0.16, "OH1": g_oh1, "O1": g_oh1 + 0.02, "OOH0": g_oh1 + 0.24}
            dz = {k: g[e["to"]] - g[e["from"]] + 1.23 * electrons(e["reservoirs"]) for k, e in edges.items()}
            best = minimum_limiting_cycle(edges, dz, OER())
            keys = list(edges)
            limit, _ = limiting_potential([dz[k] for k in keys], [electrons(edges[k]["reservoirs"]) for k in keys])
            if expected is None:
                self.assertIsNone(best)
                self.assertIsNone(limit)
            else:
                self.assertAlmostEqual(best["limiting_potential_V"], expected, places=9)
                self.assertAlmostEqual(limit, expected, places=9)
                self.assertEqual(best["edge_ids"][-1], "OOH0>ref")


class StepRuleTests(unittest.TestCase):
    def test_her_tafel_step_is_a_pair_release(self):
        registry = default_registry([HER_PLUGIN])
        native = {"s1": "empty", "s2": "empty"}
        rules = StepRules(native, registry.adsorbates, registry.reaction("her"), lambda a, b: True)
        both = {"id": "hh", "occupancy": {"s1": "H", "s2": "H"}}
        bare = {"id": "ref", "occupancy": {}}
        self.assertEqual(rules.step(both, bare), ({"H2": 1}, "pair release"))
        self.assertIsNone(rules.step(bare, both))  # dissociative H2 uptake runs against HER


class FormationVectorTests(unittest.TestCase):
    def tearDown(self):
        SPECIES.pop("OH-", None)

    def test_species_order_does_not_matter(self):
        class ProtonFirst(OER):
            formation_species = ("H+", "H2O")

        for reaction in (OER(), ProtonFirst()):
            self.assertEqual(reaction.formation_vector({"O": 1, "H": 1}), {"H+": 1, "H2O": -1})
            self.assertEqual(reaction.formation_vector({"O": 1}), {"H+": 2, "H2O": -1})
            self.assertEqual(reaction.formation_vector({"O": 2, "H": 1}), {"H+": 3, "H2O": -2})
        with self.assertRaisesRegex(ValueError, "cannot form"):
            OER().formation_vector({"Cl": 1})

    def test_alkaline_hydroxide_reservoir(self):
        Registry().add_species(
            Species(
                "OH-",
                (("O", 1), ("H", 1)),
                -1,
                "aqueous",
                ("H2", "H2O"),
                chemical_potential=lambda reservoirs, conditions: (
                    reservoirs["mu_H2O_eV"] - 0.5 * reservoirs["mu_H2_eV"]
                ),
            )
        )

        class Alkaline(OER):
            formation_species = ("H2O", "OH-")

        class HydroxideFirst(OER):
            formation_species = ("OH-", "H2O")

        for reaction in (Alkaline(), HydroxideFirst()):
            self.assertEqual(reaction.formation_vector({"O": 1, "H": 1}), {"OH-": -1})  # * + OH- -> *OH + e-
            self.assertEqual(reaction.formation_vector({"O": 1}), {"H2O": 1, "OH-": -2})
        self.assertEqual(electrons({"OH-": -1}), 1)


class ConditionsTests(unittest.TestCase):
    reservoirs = {"valid": True, "mu_H2_eV": -6.77, "mu_H2O_eV": -14.22, "mu_O2_eV": -9.86}

    def test_activities_of_o2_and_water(self):
        base = vector_free_energy(OER.net, self.reservoirs, Conditions(), 0.0)
        air = vector_free_energy(OER.net, self.reservoirs, Conditions(activities={"O2": 0.21}), 0.0)
        self.assertAlmostEqual(air - base, KT * math.log(0.21), places=12)
        brine = vector_free_energy(OER.net, self.reservoirs, Conditions(activities={"H2O": 0.8}), 0.0)
        self.assertAlmostEqual(brine - base, -2 * KT * math.log(0.8), places=12)
        # the activity of water does not move mu(O2), which the convention fixes from pure water
        self.assertEqual(
            species_free_energy("O2", self.reservoirs, Conditions(activities={"H2O": 0.5})),
            species_free_energy("O2", self.reservoirs, Conditions()),
        )

    def test_unknown_names_and_protons_are_refused(self):
        with self.assertRaisesRegex(ValueError, "Unknown species 'Cl'.*Cl-"):
            Conditions(activities={"Cl": 0.5})
        with self.assertRaisesRegex(ValueError, "pH"):
            Conditions(activities={"H+": 1e-3})
        with self.assertRaisesRegex(ValueError, "positive"):
            Conditions(activities={"O2": 0})


class TemperatureTests(unittest.TestCase):
    compact = {
        "references": {
            "H2": {"E": -6.77, "Gcorr": 0.0},
            "H2O": {"E": -14.22, "Gcorr": 0.0},
            "Cl2": {"E": -3.6, "Gcorr": 0.0},
        }
    }

    def test_298_values_warn_at_another_temperature(self):
        quiet = reservoir_chemical_potentials(self.compact, 298.15, needed=("H2", "H2O", "Cl2"))
        hot = reservoir_chemical_potentials(self.compact, 353.15, needed=("H2", "H2O", "Cl2"))
        self.assertTrue(quiet["valid"] and hot["valid"], hot["errors"])
        text = " ".join(hot["warnings"])
        for phrase in ("0.035 bar", "about 0.474 bar", "water-formation free energy", "about 4.729 eV", "U0(Cl2/Cl-)"):
            self.assertIn(phrase, text)
            self.assertNotIn(phrase, " ".join(quiet["warnings"]))

    def test_declared_values_for_the_temperature_are_used_quietly(self):
        refs = {
            "references": dict(self.compact["references"]),
            "oxygen_reference": {
                "method": "water_formation",
                "delta_G_2H2O_to_O2_2H2_eV": 4.73,
                "temperature_K": 353.15,
            },
            "chlorine_reference": {"U0_V": 1.289, "temperature_K": 353.15},
        }
        hot = reservoir_chemical_potentials(refs, 353.15, 0.474, needed=("H2", "H2O", "Cl2"))
        self.assertTrue(hot["valid"], hot["errors"])
        self.assertFalse([w for w in hot["warnings"] if "0.035" in w or "U0(Cl2" in w or "formation free energy" in w])
        conditions = Conditions(353.15)
        mu = species_free_energy("Cl-", hot, conditions)
        self.assertAlmostEqual(mu, 0.5 * hot["mu_Cl2_eV"] - 1.289, places=12)


class GasCorrectionTests(unittest.TestCase):
    def test_mode_order_and_rotations_do_not_matter(self):
        vibrations = [3850.0, 3740.0, 1590.0]
        noisy = [3850.0, 3740.0, 1590.0, 45.0, 20.0, 3.0, -8.0, -25.0, -40.0]
        reference = gas_correction("H2O", vibrations)
        self.assertAlmostEqual(gas_correction("H2O", noisy), reference, places=12)
        self.assertAlmostEqual(gas_correction("H2O", list(reversed(noisy))), reference, places=12)
        with self.assertRaisesRegex(ValueError, "3 vibrational modes expected"):
            gas_correction("H2O", [3850.0, 3740.0, -1590.0, 0, 0, 0, 0, 0, 0])


class PourbaixLineTests(unittest.TestCase):
    def data(self, g_oo, g_ooh):
        nodes = [
            {"id": "ref", "occupancy": {"s1": "empty"}, "delta_atoms": {}},
            {"id": "oo", "occupancy": {"s1": "OO"}, "delta_atoms": {"O": 2}},
            {"id": "ooh", "occupancy": {"s1": "OOH"}, "delta_atoms": {"O": 2, "H": 1}},
            {"id": "oo_image", "occupancy": {"s2": "OO"}, "delta_atoms": {"O": 2}, "computed_by": "oo"},
        ]
        edges = [
            edge("release", "oo_image", "ref", {"O2": 1}),
            edge("pcet_release", "ooh", "ref", {"O2": 1, "H+": 1}),
        ]
        reservoirs = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "mu_H2_eV": -6.8,
            "mu_H2O_eV": -14.2,
            "mu_O2_eV": -9.68,
        }
        return {
            "network": {"nodes": nodes, "edges": edges, "reference": "ref"},
            "selected": {"ref": -300.0, "oo": g_oo, "oo_image": g_oo, "ooh": g_ooh},
            "main": OER(),
            "reservoirs": reservoirs,
            "settings": {"temperature_K": 298.15},
            "thermo_cfg": {},
        }

    def test_chemical_release_marks_the_node_transient(self):
        g_ref, mu_o2, half_h2 = -300.0, -9.68, -3.4
        # bound O2 desorbs by -0.08 eV; *OOH releases O2 + (H+ + e-) at 1.12 V vs RHE
        lines, _, conditions = formation_lines(self.data(g_ref + mu_o2 + 0.08, g_ref + mu_o2 + half_h2 - 1.12))
        by_id = {line["node_id"]: line for line in lines}
        self.assertIn("-0.080 eV", by_id["oo"]["transient"]["reason"])
        self.assertIsNone(by_id["ooh"]["transient"])
        self.assertEqual([s["kind"] for s in by_id["ooh"]["product_steps"]], ["electrochemical"])
        steps = by_id["ooh"]["product_steps"]
        self.assertEqual(release_downhill(steps, [(0.0, 1.2), (0.0, 1.5)], "RHE", conditions.nernst), "throughout")
        self.assertEqual(release_downhill(steps, [(0.0, 1.0), (0.0, 1.5)], "RHE", conditions.nernst), "in part")
        self.assertIsNone(release_downhill(steps, [(0.0, 0.5), (14.0, 1.0)], "RHE", conditions.nernst))
        # bound O2 that holds on (+0.2 eV) is an ordinary termination
        lines, _, _ = formation_lines(self.data(g_ref + mu_o2 - 0.2, g_ref + mu_o2 + half_h2 - 1.12))
        self.assertIsNone({line["node_id"]: line for line in lines}["oo"]["transient"])


if __name__ == "__main__":
    unittest.main()
