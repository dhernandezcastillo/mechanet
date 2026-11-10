"""Bonds between the adsorbates of two neighbouring sites, beyond O-O.

The OER families couple two neighbouring O (``coupling=(a, b)``). Any adsorbate can now declare
that it bonds to a neighbour through its proximal atom (``Adsorbate.couples``): the builder
places the pair at its bond length, relaxed structures are read as a coupled pair, and the
pathway search proposes the coupling step. The test case is the dimerisation of two *CO on
Pt(111) through carbon, the first step towards C2 products, from the CO2 example plug-in plus a
small family written here.
"""

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from mechanet import elements
from mechanet.builder import Builder
from mechanet.cli import main
from mechanet.fileio import read_json
from mechanet.pathways import StepRules
from mechanet.plugins import default_registry
from mechanet.structure import distance_matrix, read_poscar
from mechanet.topology import DEFAULT_HYSTERESIS, _campaign_context, classify_microstate, topology_check

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "examples" / "plugins" / "co2rr.py"
DIMER = """

# --- added by tests/test_coupling.py: two neighbouring *CO couple through C -----------------
import dataclasses

CO2RRtoCO.elementary = CO2RRtoCO.elementary + ({},)  # a step that exchanges nothing: the coupling
# In OC-CO the oxygens point apart (C-C-O near 120 degrees): the pair needs tilts beyond the 15
# degrees of CO on its own, or the two O would start 2.1 A apart, below the 2.2 A floor.
CARBONYL = dataclasses.replace(CARBONYL, template_tilts_deg=(0.0, 15.0, 30.0, 45.0))


def co_dimer(ctx):
    family = "co_dimer"
    pairs = [(a, b) for a, b in ctx.nearby_pairs() if ctx.native[a] == "empty" and ctx.native[b] == "empty"]
    for a, b in pairs[:1]:
        label = f"{a}+{b}"
        apart = ctx.node({a: "CO", b: "CO"}, family, label)
        coupled = ctx.node({a: "CO", b: "CO"}, family, label, coupling=(a, b))
        ctx.step(apart, coupled, family, "2 *CO -> *OC-CO*", reaction="co2rr_co")


_register_carboxyl = register


def register(registry):
    _register_carboxyl(registry)
    registry.add_family(
        Family("co_dimer", co_dimer, "co2rr_co", "Two neighbouring *CO couple through C", ["peterson2010"], order=310)
    )
"""


def cli(*args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(["--json", *map(str, args)])
    if code != 0:
        raise AssertionError(f"mechanet {' '.join(map(str, args))} exited {code}: {out.getvalue()}")
    return json.loads(out.getvalue())


class StepRuleTests(unittest.TestCase):
    """The rule on its own: which pairs of nodes are a coupling step."""

    @staticmethod
    def rules(registry, reaction, native):
        return StepRules(native, registry.adsorbates, registry.reaction(reaction), lambda a, b: True)

    def test_oxygen_pairs_as_before(self):
        rules = self.rules(default_registry(), "oer", {"a": "O", "b": "O"})
        apart, coupled = {"occupancy": {}, "coupling": None}, {"occupancy": {}, "coupling": ["a", "b"]}
        self.assertEqual(rules.step(apart, coupled), ({}, "O-O coupling"))
        self.assertEqual(rules.step(coupled, apart), ({}, "O-O decoupling"))
        hydroxo = {"occupancy": {"a": "OH", "b": "OH"}, "coupling": ["a", "b"]}
        self.assertIsNone(rules.step({"occupancy": {"a": "OH", "b": "OH"}, "coupling": None}, hydroxo))

    def test_any_adsorbate_that_declares_it(self):
        registry = default_registry([PLUGIN])
        reaction = registry.reactions["co2rr_co"]
        with patch.object(reaction, "elementary", reaction.elementary + ({},)):
            rules = self.rules(registry, "co2rr_co", {"m1": "empty", "m2": "empty"})
            apart = {"occupancy": {"m1": "CO", "m2": "CO"}, "coupling": None}
            coupled = {"occupancy": {"m1": "CO", "m2": "CO"}, "coupling": ["m1", "m2"]}
            self.assertEqual(rules.step(apart, coupled), ({}, "C-C coupling"))
            self.assertEqual(rules.step(coupled, apart), ({}, "C-C decoupling"))
            # a carboxyl does not declare it
            carboxyl = {"occupancy": {"m1": "COOH", "m2": "COOH"}, "coupling": ["m1", "m2"]}
            self.assertIsNone(rules.step({"occupancy": {"m1": "COOH", "m2": "COOH"}, "coupling": None}, carboxyl))
        self.assertTrue(registry.adsorbate("CO").couples)
        self.assertTrue(default_registry().adsorbate("O").couples)
        self.assertFalse(default_registry().adsorbate("OH").couples)


@unittest.skipIf(os.environ.get("MECHANET_SKIP_SLOW"), "slow build disabled by environment")
class CarbonCouplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env = patch.dict(os.environ, {"MECHANET_CONFIG_DIR": str(cls.tmp / "config")})
        cls.env.start()
        cls.plugin = cls.tmp / "co2rr_dimer.py"
        cls.plugin.write_text(PLUGIN.read_text(encoding="utf-8") + DIMER, encoding="utf-8")
        project = cls.tmp / "pt"
        shutil.copytree(ROOT / "examples" / "pt_111", project)
        data = yaml.safe_load((project / "project.yaml").read_text())
        data.update(reaction="co2rr_co", plugins=[str(cls.plugin)])
        data["scope"]["families"] = ["co2rr_carboxyl", "co_dimer"]
        data.pop("thermodynamics", None)
        (project / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        cls.generated = cli("generate", project / "project.yaml")
        cls.campaign = project / "campaign"
        cls.registry = default_registry([cls.plugin])
        network = read_json(cls.campaign / "network.json")
        cls.coupled = next(n for n in network["nodes"] if n.get("coupling"))
        cls.sites = sorted(cls.coupled["coupling"])
        cls.apart = next(
            n
            for n in network["nodes"]
            if not n.get("coupling") and (n.get("occupancy") or {}) == {s: "CO" for s in cls.sites}
        )
        cls.network = network

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def start(self, node):
        state = self.campaign / "states" / node["state_ids"][0]
        return state, read_json(state / "state.json"), read_poscar(state / "POSCAR")

    def test_the_pair_is_built_at_its_bond_length(self):
        self.assertEqual(self.generated["nodes_without_viable_start"], [])
        self.assertTrue(cli("validate", self.campaign)["valid"])
        state, meta, structure = self.start(self.coupled)
        a, b = (meta["atom_ids"].index(s) for s in self.sites)
        self.assertEqual([structure.symbols[a], structure.symbols[b]], ["C", "C"])
        bond = 2 * elements.covalent_radius("C")
        self.assertAlmostEqual(float(distance_matrix(structure)[a, b]), bond, places=3)
        self.assertIn(self.sites, [sorted(pair) for pair in meta["expected_bonds"]])
        self.assertEqual(sorted(meta["coupled_metal_contacts"]), self.sites)
        self.assertNotIn("coupled_O_metal_contacts", meta)
        self.assertTrue(any("C-C bond between two sites" in note for note in meta["notes"]))
        # every O is still on its C, 1.15 A away
        for site in self.sites:
            c, o = meta["atom_ids"].index(site), meta["atom_ids"].index(f"o_{site}")
            self.assertAlmostEqual(float(distance_matrix(structure)[c, o]), 1.15, places=3)
        step = next(e for e in self.network["edges"] if {e["from"], e["to"]} == {self.apart["id"], self.coupled["id"]})
        self.assertEqual((step["reservoirs"], step["electrons"]), ({}, 0))

    def test_relaxed_pairs_are_read_and_reassigned(self):
        state, meta, structure = self.start(self.coupled)
        context = _campaign_context(self.campaign)
        occupancy = {s: "CO" for s in self.sites}
        read = classify_microstate(
            structure,
            meta["atom_ids"],
            distance_matrix(structure),
            context,
            dict(DEFAULT_HYSTERESIS),
            None,
            self.registry,
        )
        self.assertEqual(read, (occupancy, self.sites, None))
        apart_state, apart_meta, apart_structure = self.start(self.apart)
        self.assertEqual(apart_meta["atom_ids"], meta["atom_ids"])
        read = classify_microstate(
            apart_structure,
            apart_meta["atom_ids"],
            distance_matrix(apart_structure),
            context,
            dict(DEFAULT_HYSTERESIS),
            None,
            self.registry,
        )
        self.assertEqual(read, (occupancy, None, None))
        config = read_json(self.campaign / "config.json")
        # the coupled start relaxes apart: it counts for the uncoupled intermediate
        result = topology_check(
            state, meta, config, geometry=apart_state / "POSCAR", registry=self.registry, use_review=False
        )
        self.assertEqual(result["status"], "reassigned_to_existing_node")
        self.assertEqual(result["assigned_node_id"], self.apart["id"])
        self.assertIn(tuple(self.sites), {tuple(sorted(pair)) for pair in result["lost_bonds"]})
        # and two *CO that find each other count for the coupled one
        result = topology_check(
            apart_state, apart_meta, config, geometry=state / "POSCAR", registry=self.registry, use_review=False
        )
        self.assertEqual(result["status"], "reassigned_to_existing_node")
        self.assertEqual(result["assigned_node_id"], self.coupled["id"])

    def test_the_pathway_search_proposes_the_coupling(self):
        native = {site["id"]: site["native"] for site in read_json(self.campaign / "sites.json")["sites"]}
        rules = StepRules(native, self.registry.adsorbates, self.registry.reaction("co2rr_co"), lambda a, b: True)
        self.assertEqual(rules.step(self.apart, self.coupled), ({}, "C-C coupling"))
        self.assertEqual(rules.step(self.coupled, self.apart), ({}, "C-C decoupling"))

    def test_adsorbates_that_do_not_couple_are_refused(self):
        builder = Builder(
            read_json(self.campaign / "config.json"), self.tmp / "again", base_dir=self.campaign, registry=self.registry
        )
        a, b = self.sites
        with self.assertRaisesRegex(ValueError, "cannot be coupled"):
            builder.node({a: "COOH", b: "COOH"}, "co_dimer", f"{a}+{b}", coupling=(a, b))


if __name__ == "__main__":
    unittest.main()
