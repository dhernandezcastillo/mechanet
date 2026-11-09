"""Provenance of the energies an analysis compares: the method card and the same-settings check."""

import json
import tempfile
import unittest
from pathlib import Path

from mechanet.analysis import protocol_consistency
from mechanet.campaign import _electronic_structure, freeze_method_card, method_card
from mechanet.vasp import read_incar

TEMPLATE = "GGA = PE\nENCUT = 400\nISPIN = 2\nIVDW = 12\n"


def make_campaign(root, common=None, stages=None, central="refine"):
    campaign = Path(root) / "campaign"
    (campaign / "inputs").mkdir(parents=True)
    (campaign / "inputs" / "INCAR_final").write_text(TEMPLATE)
    config = {
        "reaction": "oer",
        "scope": {"families": ["aem"]},
        "vasp": {"common_overrides": dict(common or {}), "stage_overrides": dict(stages or {})},
        "thermodynamics": {},
        "frequency": {"central": central},
    }
    write_config(campaign, config)
    return campaign, config


def write_config(campaign, config):
    (Path(campaign) / "config.json").write_text(json.dumps(config))


def write_run(campaign, state_id, incar, species=("Fe", "O", "H"), titles=None, stage="04_static"):
    """A finished run as the runner leaves it: the stage's INCAR and the state's PAW provenance."""
    state = Path(campaign) / "states" / state_id
    (state / stage).mkdir(parents=True)
    (state / stage / "INCAR").write_text(incar)
    titles = {e: f"PAW_PBE {e} 01Jan2000" for e in species} | dict(titles or {})
    datasets = [{"element": e, "titel": titles[e]} for e in species]
    (state / "potcar_provenance.json").write_text(json.dumps({"species_order": list(species), "datasets": datasets}))


class MethodCardTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name

    def test_the_card_shows_the_settings_the_jobs_run_with(self):
        campaign, _ = make_campaign(self.root, common={"ENCUT": 600}, stages={"04_static": {"IVDW": 11}})
        card = method_card(campaign)
        self.assertEqual(card["electronic_structure"]["stage"], "04_static")
        self.assertEqual(card["electronic_structure"]["plane_wave_cutoff_eV"], "600")
        self.assertEqual(card["electronic_structure"]["dispersion"], {"IVDW": "11"})
        self.assertEqual(card["protocol"]["vasp_overrides"]["common_overrides"], {"ENCUT": 600})

    def test_with_central_final_the_card_describes_02_final(self):
        campaign, _ = make_campaign(self.root, stages={"02_final": {"ENCUT": 520}}, central="final")
        card = method_card(campaign)
        self.assertEqual(card["electronic_structure"]["stage"], "02_final")
        self.assertEqual(card["electronic_structure"]["plane_wave_cutoff_eV"], "520")

    def test_a_changed_override_is_detected_after_freezing(self):
        campaign, config = make_campaign(self.root, common={"ENCUT": 600})
        freeze_method_card(campaign)
        self.assertEqual(method_card(campaign)["changed_since_first_submission"], [])
        config["vasp"]["common_overrides"]["ENCUT"] = 800
        write_config(campaign, config)
        self.assertIn("electronic_structure", method_card(campaign)["changed_since_first_submission"])
        # an override of a stage whose energy is not used still shows, under protocol
        config["vasp"]["common_overrides"]["ENCUT"] = 600
        config["vasp"]["stage_overrides"] = {"05_freq": {"EDIFF": 1e-8}}
        write_config(campaign, config)
        self.assertEqual(method_card(campaign)["changed_since_first_submission"], ["protocol"])

    def test_a_card_frozen_by_an_earlier_version_raises_no_false_alarm(self):
        campaign, config = make_campaign(self.root, common={"ENCUT": 600})
        card = method_card(campaign)
        legacy = {
            **card,
            "schema_version": 1,
            "electronic_structure": _electronic_structure(
                read_incar(campaign / "inputs" / "INCAR_final"), config["vasp"], []
            ),
            "protocol": {k: v for k, v in card["protocol"].items() if k != "vasp_overrides"},
        }
        (campaign / "method_card.frozen.json").write_text(json.dumps(legacy))
        again = method_card(campaign)
        self.assertEqual(again["changed_since_first_submission"], [])
        self.assertIn("cannot be detected", again["frozen_card_note"])


class ProtocolConsistencyTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.campaign = Path(tmp.name) / "campaign"

    def check(self, *state_ids):
        return protocol_consistency(self.campaign, state_ids, "04_static")

    def test_other_moments_and_parallelisation_keep_one_hamiltonian(self):
        write_run(self.campaign, "a", "GGA = PE\nENCUT = 500\nMAGMOM = 3*4.0\nNCORE = 8\n")
        write_run(self.campaign, "b", "GGA = PE\nENCUT = 500.0\nMAGMOM = 3*0.6\nNCORE = 16\n")
        check = self.check("a", "b")
        self.assertTrue(check["consistent"])
        self.assertIsNone(check["warning"])
        self.assertEqual(check["states_checked"], 2)

    def test_a_different_cutoff_is_reported_with_the_states(self):
        write_run(self.campaign, "a", "GGA = PE\nENCUT = 400\n")
        write_run(self.campaign, "b", "GGA = PE\nENCUT = 600\n")
        write_run(self.campaign, "c", "GGA = PE\nENCUT = 600\n")
        check = self.check("a", "b", "c")
        self.assertFalse(check["consistent"])
        self.assertEqual(check["differences"], {"ENCUT": {"400": ["a"], "600": ["b", "c"]}})
        self.assertIn("ENCUT 400 (1 state: a) vs 600 (2 states: b, c)", check["warning"])

    def test_hubbard_u_is_compared_element_by_element(self):
        incar = "GGA = PE\nENCUT = 500\nLDAU = .TRUE.\nLDAUL = {l}\nLDAUU = {u}\nLDAUJ = {j}\n"
        write_run(self.campaign, "fe_o_h", incar.format(l="2 -1 -1", u="5.3 0 0", j="0 0 0"), ("Fe", "O", "H"))
        write_run(self.campaign, "fe_o", incar.format(l="2 -1", u="5.3 0", j="0 0"), ("Fe", "O"))
        self.assertTrue(self.check("fe_o_h", "fe_o")["consistent"])  # other species, same U per element
        write_run(self.campaign, "other_u", incar.format(l="2 -1", u="4.0 0", j="0 0"), ("Fe", "O"))
        self.assertEqual(sorted(self.check("fe_o_h", "fe_o", "other_u")["differences"]), ["U Fe"])

    def test_paw_datasets_are_compared_and_missing_runs_listed(self):
        write_run(self.campaign, "a", "ENCUT = 500\n", ("Fe", "O"))
        write_run(self.campaign, "b", "ENCUT = 500\n", ("Fe", "O"), titles={"Fe": "PAW_PBE Fe_pv 02Aug2007"})
        check = self.check("a", "b", "never_ran")
        self.assertEqual(sorted(check["differences"]), ["PAW Fe"])
        self.assertEqual(check["unchecked"], ["never_ran"])


if __name__ == "__main__":
    unittest.main()
