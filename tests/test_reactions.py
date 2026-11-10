"""Reservoir vectors, electron counting and the built-in reactions."""

import unittest

from mechanet.families.coverage import default_states
from mechanet.plugins import default_registry
from mechanet.reactions import CER, OER, Reaction, add, clean, electrons, element_change, oer_trio
from mechanet.thermo import required_molecules


class ChemistryDeclarationTests(unittest.TestCase):
    """What used to be wired into the core now comes from the reaction and its species."""

    def test_coverage_defaults_come_from_the_reaction(self):
        self.assertEqual(default_states("open_metal", OER()), ["empty", "OH", "O", "H2O"])
        self.assertEqual(default_states("open_metal", "cer"), ["empty", "OH", "O", "H2O", "Cl", "OCl"])
        self.assertEqual(default_states("terminal", CER()), ["empty", "OH", "O", "H2O", "OCl"])
        self.assertEqual(default_states("bridge", CER()), ["empty", "O", "OH"])
        self.assertEqual(default_states("unknown_kind", OER()), ["O", "OH"])

        class MetalHydride(Reaction):
            coverage_states = {"open_metal": ("empty", "H")}

        self.assertEqual(default_states("open_metal", MetalHydride()), ["empty", "H"])
        self.assertEqual(default_states("terminal", Reaction()), ["empty", "OH", "O", "H2O"])  # oxide default

    def test_reference_molecules_come_from_the_species(self):
        self.assertEqual(required_molecules([OER.net]), {"H2", "H2O"})
        self.assertEqual(required_molecules([CER.net]), {"Cl2"})


class ReactionTests(unittest.TestCase):
    def test_turnovers_and_electrons(self):
        self.assertEqual(electrons(OER.net), 4)
        self.assertEqual(electrons(CER.net), 2)
        self.assertTrue(OER().is_turnover({"H2O": -2, "H+": 4, "O2": 1}))
        self.assertTrue(CER().is_turnover({"Cl-": -2, "Cl2": 1}))
        self.assertEqual(
            add({"H2O": -1, "H+": 1}, {"H+": 1}, {"H2O": -1, "H+": 1}, {"H+": 1, "O2": 1}),
            {"H+": 4, "H2O": -2, "O2": 1},
        )

    def test_element_change_matches_surface(self):
        # *O + H2O -> *OOH + H+ + e-: the surface gains O and H
        self.assertEqual(element_change({"H2O": -1, "H+": 1}), {"O": 1, "H": 1})
        self.assertEqual(element_change({"Cl-": -1}), {"Cl": 1})
        self.assertEqual(element_change({"O2": 1, "H+": 1}), {"O": -2, "H": -1})

    def test_balance_fills_balance_species(self):
        oer, cer = OER(), CER()
        self.assertEqual(oer.balance({"O": 1, "H": 1}, {"H2O": -1}), {"H+": 1, "H2O": -1})
        self.assertEqual(oer.balance({"H": -1}), {"H+": 1})
        self.assertEqual(cer.balance({"Cl": 1}), {"Cl-": -1})
        self.assertEqual(cer.balance({"Cl": -1}, {"Cl2": 1}), {"Cl-": -1, "Cl2": 1})
        with self.assertRaisesRegex(ValueError, "does not balance"):
            oer.balance({"O": 1}, {"H2O": 0})

    def test_formation_vectors(self):
        oer, cer = OER(), CER()
        self.assertEqual(oer.formation_vector({"O": 1, "H": 1}), {"H+": 1, "H2O": -1})  # *OH from water
        self.assertEqual(oer.formation_vector({"O": 1}), {"H+": 2, "H2O": -1})  # *O
        self.assertEqual(oer.formation_vector({"H": -1}), {"H+": 1})  # deprotonation
        self.assertEqual(cer.formation_vector({"Cl": 1}), {"Cl-": -1})
        self.assertEqual(electrons(cer.formation_vector({"Cl": 1, "O": 1})), 3)

    def test_oer_trio_and_clean(self):
        self.assertEqual(oer_trio({"H2O": -1, "H+": 1}), (1, 1, 0))
        self.assertIsNone(oer_trio({"Cl-": -1}))
        self.assertEqual(clean({"H+": 1.0, "O2": 0.0}), {"H+": 1})
        # A species no plug-in of the examples adds: the species table is shared by the whole
        # process, so NH3 is known once the nitrogen example has been loaded.
        with self.assertRaisesRegex(ValueError, "Unknown reservoir species"):
            clean({"SF6": 1})

    def test_registry_lists_builtins(self):
        registry = default_registry()
        self.assertEqual(sorted(registry.reactions), ["cer", "oer"])
        self.assertEqual(
            registry.core_families("oer"),
            ["aem", "surface_bridge", "pair_coupling", "terminal_bridge_coupling", "bifunctional", "hydrated"],
        )
        self.assertEqual(registry.core_families("cer"), ["cer_volmer_heyrovsky", "cer_ocl"])
        self.assertIn("pathways", registry.commands)
        self.assertIn("pourbaix", registry.commands)
        self.assertEqual(registry.errors, [])


if __name__ == "__main__":
    unittest.main()
