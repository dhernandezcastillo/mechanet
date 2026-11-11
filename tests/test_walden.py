"""Walden steps of the oxygen evolution reaction: O2 leaves a site as water attaches to it.

Exner (Adv. Sci. 10, 2305505, 2023) adds two mechanisms to the usual ones, whose last step
releases O2 while water attaches, M-OO + H2O -> HO-M + O2 + H+ + e-, so that the bare site M is
never formed: mononuclear-Walden (eqs. 7-10) and bifunctional-Walden (eqs. 15-18). Usama et al.
(Nat. Commun. 16, 6137, 2025) find them preferred on IrO2(110) and also discuss the same
exchange without charge transfer, *OO + H2O -> *OH2 + O2.

The designed case (Pt(111), one site, free energies at 0 V vs RHE relative to the bare site): *OH
-0.30, *O 1.10, *OOH 2.60, *OO 3.75 eV. OH binds strongly, so leaving the site bare is costly:

* usual mechanism: -0.30, 1.40, 1.50 and 4.92 - 2.60 = 2.32 eV, so U_L = 2.32 V (eta 1.09 V);
* with *OO held first: its release to the bare site, 4.92 - 3.75 = 1.17 eV, has no electron and
  is uphill, so no potential drives the cycle;
* mononuclear-Walden: 1.40, 1.50, 1.15 and -0.30 + 4.92 - 3.75 = 0.87 eV, so U_L = 1.50 V
  (eta 0.27 V).
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from support import FAKE_VASP, write_potcars

from mechanet.cli import main
from mechanet.fileio import read_json
from mechanet.literature import LITERATURE
from mechanet.notation import route
from mechanet.pathways import StepRules, propose_steps
from mechanet.plugins import default_registry
from mechanet.project import generate
from mechanet.reactions import CER, OER, Reaction, add, electrons, validate_reaction
from mechanet.runner import run_state
from mechanet.structure import read_poscar

ROOT = Path(__file__).resolve().parents[1]
HER_PLUGIN = ROOT / "examples" / "plugins" / "her.py"
WITH_PROTON = {"H2O": -1, "O2": 1, "H+": 1}
WITHOUT_CHARGE = {"H2O": -1, "O2": 1}
E_H2, E_H2O, WATER_SPLITTING = -6.77, -14.22, 4.92
# free energy at 0 V vs RHE relative to the bare site, by (O, H) of the adsorbate
FORMATION = {(1, 1): -0.30, (1, 0): 1.10, (2, 1): 2.60, (2, 0): 3.75}
REFERENCES = {
    "temperature_K": 298.15,
    "protocol_id": "synthetic test references",
    "H2": {"energy_e0_eV": E_H2, "free_energy_correction_eV": 0.0, "phase": "gas"},
    "H2O": {"energy_e0_eV": E_H2O, "free_energy_correction_eV": 0.0, "phase": "liquid", "Gcorr_pressure_bar": 0.035},
    "oxygen_reference": {
        "method": "water_formation",
        "delta_G_2H2O_to_O2_2H2_eV": WATER_SPLITTING,
        "temperature_K": 298.15,
    },
}


def cli(*args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(["--json", *map(str, args)])
    if code != 0:
        raise AssertionError(f"mechanet {' '.join(map(str, args))} exited {code}: {out.getvalue()}")
    return json.loads(out.getvalue())


def node(**occupancy):
    return {"occupancy": occupancy, "coupling": None}


class DeclarationTests(unittest.TestCase):
    def test_the_oer_declares_both_forms(self):
        self.assertEqual(OER.walden, (WITH_PROTON, WITHOUT_CHARGE))
        self.assertTrue(validate_reaction(OER))
        self.assertEqual([electrons(v) for v in OER.walden], [1, 0])
        # they are not ordinary elementary steps: a step between two sites with the same bookkeeping
        # (the lumped bifunctional water attack) is not proposed because of them
        self.assertNotIn(WITH_PROTON, OER.elementary)
        self.assertEqual((CER.walden, Reaction.walden), ((), ()))

    def test_a_walden_step_releases_the_product_and_takes_something_up(self):
        for vector in ({"O2": 1}, {"H2O": -1, "H+": 1}, {"O2": -1, "H2O": -1}):
            with self.subTest(vector=vector):

                class Wrong(OER):
                    name = "wrong"
                    walden = (vector,)

                with self.assertRaisesRegex(ValueError, "Walden step .* must give off 'O2'"):
                    validate_reaction(Wrong)

    def test_the_sources_are_recorded(self):
        self.assertEqual(LITERATURE["exner2023walden"]["doi"], "10.1002/advs.202305505")
        self.assertEqual(LITERATURE["usama2025"]["doi"], "10.1038/s41467-025-61367-z")
        family = default_registry().families["walden"]
        self.assertEqual((family.reaction, family.core), ("oer", False))
        self.assertEqual(list(family.sources), ["exner2023walden", "usama2025"])
        self.assertNotIn("walden", default_registry().core_families("oer"))
        names = route({"id": "walden_a0065_mononuclear", "family": "walden"})
        self.assertEqual((names["short"], names["variant"]), ("mononuclear-Walden", None))
        names = route({"id": "walden_a0065_a0061_bifunctional", "family": "walden"})
        self.assertEqual((names["short"], names["variant"]), ("bifunctional-Walden", None))


class StepRuleTests(unittest.TestCase):
    """What ``mechanet pathways`` proposes between two intermediates."""

    def rules(self, reaction="oer", registry=None, reverse=False):
        registry = registry or default_registry()
        native = {"m1": "empty", "b1": "O"}
        return StepRules(native, registry.adsorbates, registry.reaction(reaction), lambda a, b: True, reverse)

    def test_from_a_site_that_holds_dioxygen_whole(self):
        rules = self.rules()
        self.assertEqual(rules.step(node(m1="OO"), node(m1="OH")), (WITH_PROTON, "Walden step"))
        self.assertEqual(rules.step(node(m1="OO"), node(m1="H2O")), (WITHOUT_CHARGE, "Walden step"))
        # a spectator next to it changes nothing
        self.assertEqual(rules.step(node(m1="OO", b1="OH"), node(m1="OH", b1="OH")), (WITH_PROTON, "Walden step"))
        # *OOH would have to lose its proton as well: not a step of the cited mechanisms
        self.assertIsNone(rules.step(node(m1="OOH"), node(m1="H2O")))
        # two sites with the same bookkeeping (O and O giving an empty site and OH) stay unproposed
        self.assertIsNone(rules.step(node(m1="O"), node(m1="empty", b1="OH")))
        # the reaction runs forward only, unless asked
        self.assertIsNone(rules.step(node(m1="OH"), node(m1="OO")))
        back = {name: -value for name, value in WITH_PROTON.items()}
        self.assertEqual(self.rules(reverse=True).step(node(m1="OH"), node(m1="OO")), (back, "Walden step"))

    def test_the_other_steps_are_proposed_as_before(self):
        rules = self.rules()
        self.assertEqual(rules.step(node(m1="OO"), node()), ({"O2": 1}, "single-site step"))
        self.assertEqual(rules.step(node(m1="OOH"), node()), ({"H+": 1, "O2": 1}, "single-site step"))
        self.assertEqual(rules.step(node(), node(m1="OH")), ({"H+": 1, "H2O": -1}, "single-site step"))
        self.assertEqual(rules.step(node(m1="OOH"), node(m1="OO")), ({"H+": 1}, "single-site step"))
        # chlorine and hydrogen evolution have no Walden step: their releases are untouched
        chlorine = self.rules("cer")
        self.assertEqual(chlorine.walden, [])
        self.assertEqual(chlorine.step(node(m1="Cl"), node()), ({"Cl-": -1, "Cl2": 1}, "single-site step"))
        self.assertIsNone(chlorine.step(node(m1="OO"), node(m1="OH")))
        hydrogen = self.rules("her", default_registry([HER_PLUGIN]))
        self.assertEqual(hydrogen.step(node(m1="H"), node()), ({"H+": -1, "H2": 1}, "single-site step"))


@unittest.skipIf(os.environ.get("MECHANET_SKIP_SLOW"), "slow build disabled by environment")
class FamilyTests(unittest.TestCase):
    """The two mechanisms on RuO2(110): open Ru sites m0038 and m0046, bridging O a0016 and a0019."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env = patch.dict(os.environ, {"MECHANET_CONFIG_DIR": str(cls.tmp / "config")})
        cls.env.start()
        project = cls.tmp / "p"
        shutil.copytree(ROOT / "examples" / "ruo2_110", project)
        data = yaml.safe_load((project / "project.yaml").read_text())
        cls.campaigns = {}
        for name, families in (("usual", ["aem", "bifunctional"]), ("walden", ["aem", "bifunctional", "walden"])):
            data["scope"]["families"] = families
            (project / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
            generate(project / "project.yaml", cls.tmp / name)
            cls.campaigns[name] = read_json(cls.tmp / name / "network.json")

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_each_cycle_is_one_turnover_without_the_bare_site(self):
        network = self.campaigns["walden"]
        nodes = {n["id"]: n for n in network["nodes"]}
        edges = {e["id"]: e for e in network["edges"]}
        cycles = {c["id"]: c for c in network["cycles"] if c["family"] == "walden"}
        self.assertEqual(
            sorted(cycles),
            [
                "walden_m0038_a0019_bifunctional",
                "walden_m0038_m0046_bifunctional",
                "walden_m0038_mononuclear",
                "walden_m0046_a0016_bifunctional",
                "walden_m0046_m0038_bifunctional",
                "walden_m0046_mononuclear",
            ],
        )
        for cycle_id, cycle in cycles.items():
            with self.subTest(cycle=cycle_id):
                steps = [edges[e] for e in cycle["edge_ids"]]
                self.assertEqual([electrons(e["reservoirs"]) for e in steps], [1, 1, 1, 1])
                self.assertEqual(add(*(e["reservoirs"] for e in steps)), {"H+": 4, "H2O": -2, "O2": 1})
                self.assertEqual(steps[-1]["reservoirs"], {"H+": 1, "H2O": -1, "O2": 1})
                self.assertEqual(steps[-1]["to"], steps[0]["from"])
                donor = cycle_id.split("_")[1]
                held = [(nodes[e["from"]]["occupancy"] or {}).get(donor) for e in steps]
                self.assertEqual(held[0], "OH")
                self.assertEqual(held[-1], "OO")
                self.assertNotIn("empty", held)
                self.assertNotIn(None, held)
                self.assertEqual(cycle["source_ids"], ["exner2023walden", "usama2025"])
        occupancies = [nodes[edges[e]["from"]]["occupancy"] for e in cycles["walden_m0038_mononuclear"]["edge_ids"]]
        self.assertEqual(occupancies, [{"m0038": s} for s in ("OH", "O", "OOH", "OO")])
        occupancies = [
            nodes[edges[e]["from"]]["occupancy"] for e in cycles["walden_m0038_a0019_bifunctional"]["edge_ids"]
        ]
        expected = [{"m0038": "OH"}, {"m0038": "O"}, {"m0038": "OO", "a0019": "OH"}, {"m0038": "OO"}]
        self.assertEqual(occupancies, expected)

    def test_it_reuses_the_intermediates_of_the_usual_mechanisms(self):
        usual = {n["id"]: n for n in self.campaigns["usual"]["nodes"]}
        added = [n for n in self.campaigns["walden"]["nodes"] if n["id"] not in usual]
        # only *OO next to an O on the other open site, an acceptor state that is not its native one
        self.assertEqual(
            sorted(json.dumps(n["occupancy"], sort_keys=True) for n in added),
            ['{"m0038": "O", "m0046": "OO"}', '{"m0038": "OO", "m0046": "O"}'],
        )
        usual_edges = {(e["from"], e["to"]) for e in self.campaigns["usual"]["edges"]}
        new_edges = [e for e in self.campaigns["walden"]["edges"] if (e["from"], e["to"]) not in usual_edges]
        # between intermediates that were there: the Walden step, and the deprotonation of the
        # acceptor next to *OO (eq. 17), which the usual bifunctional mechanisms do not need
        between_known = [e for e in new_edges if e["from"] in usual and e["to"] in usual]
        self.assertEqual(
            {json.dumps(e["reservoirs"], sort_keys=True) for e in between_known},
            {json.dumps(WITH_PROTON, sort_keys=True), json.dumps({"H+": 1})},
        )

    def test_reports_name_the_mechanisms(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["show-cycle", "walden_m0038_mononuclear", str(self.tmp / "walden")]), 0)
        text = out.getvalue()
        self.assertIn("Mononuclear-Walden: O₂ leaves the site as water attaches, without a bare site", text)
        self.assertIn("*OO₃₈ + H₂O → *OH₃₈ + O₂ + H⁺ + e⁻", text)
        self.assertIn("Sources: Exner 2023; Usama et al. 2025", text)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["validate", str(self.tmp / "walden")]), 0)

    def test_the_pathway_search_finds_the_step_without_the_family(self):
        network = self.campaigns["usual"]
        registry = default_registry()
        native = {s["id"]: s["native"] for s in read_json(self.tmp / "usual" / "sites.json")["sites"]}
        rules = StepRules(native, registry.adsorbates, registry.reaction("oer"), lambda a, b: True)
        nodes = {n["id"]: n for n in network["nodes"]}
        found = [
            (nodes[a]["occupancy"], nodes[b]["occupancy"])
            for a, b, _v, rule in propose_steps(network["nodes"], rules)
            if rule == "Walden step"
        ]
        self.assertIn(({"m0038": "OO"}, {"m0038": "OH"}), found)
        self.assertIn(({"m0046": "OO"}, {"m0046": "OH"}), found)
        self.assertTrue(all(len(a) == len(b) for a, b in found))  # one site changes; a neighbour only looks on


@unittest.skipIf(os.environ.get("MECHANET_SKIP_SLOW"), "slow workflow disabled by environment")
class ThermodynamicsTests(unittest.TestCase):
    """The designed case of the module docstring, through synthetic VASP, thermo and the pathway search."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env = patch.dict(os.environ, {"MECHANET_CONFIG_DIR": str(cls.tmp / "config")})
        cls.env.start()
        project = cls.tmp / "pt"
        shutil.copytree(ROOT / "examples" / "pt_111", project)
        data = yaml.safe_load((project / "project.yaml").read_text())
        fake = cls.tmp / "fake_vasp.py"
        fake.write_text(FAKE_VASP)
        data["sites"] = data["sites"][:1]
        data["scope"]["families"] = ["aem", "walden"]
        data["vasp"] = {
            "potcars": write_potcars(cls.tmp, ("Pt", "O", "H")),
            "magmom_by_element": None,
            "command": [sys.executable, "-I", str(fake)],
        }
        data["frequency"] = {"central": "final"}
        data["thermodynamics"] = {
            "references": "inputs/references.json",
            "potentials_V": [0.0, 1.5],
            "electronic_mode_corrections": {"per_H_eV": 0.0, "per_O_eV": 0.0},
        }
        (project / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        (project / "inputs" / "references.json").write_text(json.dumps(REFERENCES))
        cls.references = project / "inputs" / "references.json"
        cls.campaign = project / "campaign"
        cli("generate", project / "project.yaml")
        for state in sorted((cls.campaign / "states").iterdir()):
            delta = read_json(state / "state.json")["delta_atoms"]
            o, h = delta.get("O", 0), delta.get("H", 0)
            # surface + o H2O -> intermediate + (2 o - h) (H+ + e-)
            target = -100.0 + (FORMATION[(o, h)] + o * E_H2O - 0.5 * (2 * o - h) * E_H2 if o else 0.0)
            (state / "ctl").mkdir(exist_ok=True)
            (state / "ctl" / "eshift_02_final").write_text(f"{target + 100.003:.6f}")
            x0 = read_poscar(state / "POSCAR").positions.reshape(-1)
            (state / "harmonic.json").write_text(
                json.dumps({"x0": x0.tolist(), "K": (10.0 * np.eye(len(x0))).tolist()})
            )
        cli("prepare", cls.campaign)
        for state in sorted((cls.campaign / "states").iterdir()):
            assert run_state(state, through="final", execute=True)["status"] == "complete"

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_the_walden_cycle_is_driven_where_the_bare_site_blocks_the_others(self):
        cli("thermo", self.campaign, "--electronic-only")
        thermo = read_json(self.campaign / "analysis" / "thermodynamics.json")
        rows = {c["cycle_id"]: c for c in thermo["cycles"] if c["potential_V"] == 0.0}
        usual, held, walden = (
            rows[f"{name}_m0010_{tail}"]
            for name, tail in (("aem", "AEM"), ("aem", "bound_OO"), ("walden", "mononuclear"))
        )
        self.assertEqual(sorted(round(x, 6) for x in usual["step_deltaG_eV"]), [-0.30, 1.40, 1.50, 2.32])
        self.assertAlmostEqual(usual["limiting_potential_V"], 2.32, places=6)
        self.assertAlmostEqual(usual["thermodynamic_overpotential_V"], 1.09, places=6)
        # *OO -> bare site + O2 has no electron and is uphill: no potential drives that cycle
        self.assertEqual(sorted(round(x, 6) for x in held["step_deltaG_eV"]), [-0.30, 1.15, 1.17, 1.40, 1.50])
        self.assertIsNone(held["limiting_potential_V"])
        self.assertEqual(sorted(round(x, 6) for x in walden["step_deltaG_eV"]), [0.87, 1.15, 1.40, 1.50])
        self.assertAlmostEqual(sum(walden["step_deltaG_eV"]), WATER_SPLITTING, places=6)
        self.assertAlmostEqual(walden["limiting_potential_V"], 1.50, places=6)
        self.assertAlmostEqual(walden["thermodynamic_overpotential_V"], 0.27, places=6)

    def test_the_pathway_search_ranks_it_first(self):
        ranked = cli(
            "pathways",
            self.campaign,
            "--rank",
            "--references",
            self.references,
            "--electronic-only",
            "--potentials",
            "1.5",
        )
        best = ranked["ranking"]["global_minimum_limiting_potential"]["oer"]
        self.assertAlmostEqual(best["limiting_potential_V"], 1.50, places=6)


if __name__ == "__main__":
    unittest.main()
