"""Reservoir vectors, electron counting and the built-in reactions."""

import unittest

from mechanet.plugins import default_registry
from mechanet.reactions import CER, OER, add, clean, electrons, element_change, oer_trio


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
        with self.assertRaisesRegex(ValueError, "Unknown reservoir species"):
            clean({"NH3": 1})

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
