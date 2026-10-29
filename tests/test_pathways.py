"""Pathway discovery: elementary-step rules, cycle enumeration, the potential sweep, adoption."""

import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from mechanet.builder import build
from mechanet.fileio import read_json
from mechanet.pathways import (
    StepRules,
    adopt,
    build_graph,
    discover,
    enumerate_cycles,
    minimum_limiting_cycle,
    product_release_steps,
    propose_steps,
)
from mechanet.plugins import default_registry
from mechanet.project import load_project, resolve_config
from mechanet.reactions import OER, clean, electrons
from mechanet.thermo import validate_network

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def ruo2_campaign(out, families=("aem", "bifunctional")):
    config = resolve_config(load_project(EXAMPLES / "ruo2_110" / "project.yaml"), cluster=False)
    config["scope"]["families"] = list(families)
    build(config, out, base_dir=EXAMPLES / "ruo2_110")
    return Path(out)


class StepRuleTests(unittest.TestCase):
    def setUp(self):
        registry = default_registry()
        self.native = {"s1": "empty", "s2": "O", "s3": "O"}
        pairs = {("s1", "s2"), ("s2", "s3")}
        self.rules = StepRules(self.native, registry.adsorbates, OER(), lambda a, b: (a, b) in pairs or (b, a) in pairs)

    @staticmethod
    def node(occupancy, coupling=None, name="n"):
        return {"id": name, "occupancy": occupancy, "coupling": coupling}

    def test_adsorbate_evolution_steps(self):
        ref = self.node({})
        oh, o, ooh = self.node({"s1": "OH"}), self.node({"s1": "O"}), self.node({"s1": "OOH"})
        self.assertEqual(self.rules.step(ref, oh), ({"H+": 1, "H2O": -1}, "single-site step"))
        self.assertEqual(self.rules.step(oh, o), ({"H+": 1}, "single-site step"))
        self.assertEqual(self.rules.step(o, ooh), ({"H+": 1, "H2O": -1}, "single-site step"))
        self.assertEqual(self.rules.step(ooh, ref), ({"H+": 1, "O2": 1}, "single-site step"))
        self.assertIsNone(self.rules.step(oh, ref))  # cathodic steps are not proposed by default

    def test_two_site_rules(self):
        both_o = self.node({"s1": "O"})
        coupled = self.node({"s1": "O"}, coupling=["s1", "s2"])
        self.assertEqual(self.rules.step(both_o, coupled), ({}, "O-O coupling"))
        # a proton moves from *OOH on s1 to the bridging O on s2
        self.assertEqual(
            self.rules.step(self.node({"s1": "OOH"}), self.node({"s1": "OO", "s2": "OH"})),
            ({}, "proton transfer between neighbours"),
        )
        # two heavy-atom changes at once are not elementary
        self.assertIsNone(self.rules.step(self.node({"s1": "OH"}), self.node({"s1": "O", "s2": "empty"})))
        # s1 and s3 are not a pair
        self.assertIsNone(self.rules.step(self.node({}), self.node({"s1": "OH", "s3": "OH"})))

    def test_proposals_skip_existing_edges(self):
        nodes = [self.node({}, name="ref"), self.node({"s1": "OH"}, name="oh")]
        steps = propose_steps(nodes, self.rules)
        self.assertEqual([(a, b) for a, b, _, _ in steps], [("ref", "oh")])
        existing = {("ref", "oh", json.dumps({"H+": 1, "H2O": -1}, sort_keys=True))}
        self.assertEqual(propose_steps(nodes, self.rules, existing), [])


class DiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.campaign = ruo2_campaign(Path(cls.tmp) / "campaign")
        cls.report = discover(cls.campaign, max_length=6)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_every_curated_cycle_is_recovered(self):
        summary = self.report["summary"]
        network = read_json(self.campaign / "network.json")
        self.assertEqual(summary["curated_cycles_recovered"], len(network["cycles"]))
        self.assertEqual(summary["curated_not_recovered"], [])
        self.assertFalse(summary["truncated"])
        self.assertGreater(summary["discovered_cycles"], 0)
        self.assertTrue((self.campaign / "pathways" / "pathways.csv").is_file())

    def test_discovered_cycles_are_balanced_closed_turnovers(self):
        edges = {e["id"]: e for e in read_json(self.campaign / "network.json")["edges"]}
        edges.update({e["id"]: e for e in self.report["new_edges"]})
        for cycle in self.report["cycles"][:500]:
            steps = [edges[e] for e in cycle["edge_ids"]]
            for a, b in zip(steps, steps[1:] + steps[:1]):
                self.assertEqual(a["to"], b["from"])
            total = {}
            for step in steps:
                for k, v in clean(step["reservoirs"]).items():
                    total[k] = total.get(k, 0) + v
            self.assertEqual(clean(total), clean(OER.net))
            self.assertLessEqual(len(steps), 6)
            self.assertGreater(clean(steps[-1]["reservoirs"]).get("O2", 0), 0)  # release-last order

    def test_enumeration_respects_length_and_site_bounds(self):
        network = read_json(self.campaign / "network.json")
        graph = build_graph(network["edges"])
        cycles, truncated = enumerate_cycles(graph, OER(), max_length=4)
        self.assertFalse(truncated)
        self.assertTrue(all(len(c) <= 4 for c in cycles))
        aem = [c for c in network["cycles"] if c["family"] == "aem"]
        found = {tuple(c) for c in cycles}
        for cycle in aem:
            if len(cycle["edge_ids"]) <= 4:
                rotated = cycle["edge_ids"]
                self.assertTrue(any(tuple(rotated[k:] + rotated[:k]) in found for k in range(len(rotated))))
        capped, truncated = enumerate_cycles(graph, OER(), max_length=6, max_cycles=3)
        self.assertTrue(truncated)
        self.assertEqual(len(capped), 3)

    def test_potential_sweep_equals_brute_force(self):
        network = read_json(self.campaign / "network.json")
        edges = {e["id"]: e for e in network["edges"]}
        edges.update({e["id"]: e for e in self.report["new_edges"]})
        wide = discover(self.campaign, max_length=7, max_sites=None, out=Path(self.tmp) / "wide")

        def limiting(cycle, dz):
            values = []
            for key in cycle:
                n = electrons(clean(edges[key]["reservoirs"]))
                if n == 0 and dz[key] > 1e-12:
                    return float("inf")
                if n:
                    values.append(dz[key] / n)
            return max(values)

        for seed in range(4):
            rnd = random.Random(seed)
            g = {n["id"]: rnd.uniform(-1.5, 1.5) for n in network["nodes"]}
            dz = {k: g[e["to"]] - g[e["from"]] + 1.23 * electrons(clean(e["reservoirs"])) for k, e in edges.items()}
            best = minimum_limiting_cycle(edges, dz, OER())
            brute = min(limiting(c["edge_ids"], dz) for c in wide["cycles"])
            self.assertAlmostEqual(best["limiting_potential_V"], brute, places=9)
            self.assertAlmostEqual(limiting(best["edge_ids"], dz), brute, places=9)

    def test_adopt_adds_cycles_and_keeps_a_valid_network(self):
        out = Path(self.tmp) / "adopt"
        shutil.copytree(self.campaign, out)
        candidates = [
            c
            for c in self.report["cycles"]
            if not c["curated"] and c["symmetry_representative"] and c["uses_new_edges"]
        ][:2]
        adopted = adopt(out, [c["id"] for c in candidates])
        self.assertEqual(adopted, [c["id"] for c in candidates])
        network = read_json(out / "network.json")
        self.assertTrue((out / "network.before_pathways.json").is_file())
        self.assertTrue(validate_network(network, default_registry())["valid"])
        self.assertEqual(adopt(out, [c["id"] for c in candidates]), [])  # idempotent
        with self.assertRaisesRegex(ValueError, "Not in pathways"):
            adopt(out, ["no_such_cycle"])


if __name__ == "__main__":
    unittest.main()


class ReleaseStepTests(unittest.TestCase):
    def test_bound_o2_without_a_release_edge_gets_a_proposed_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = ruo2_campaign(Path(tmp) / "c", families=("aem", "hydrated"))
            network = read_json(campaign / "network.json")
            registry = default_registry()
            steps = product_release_steps(campaign, network, registry, [registry.reaction("oer")])
            written = {e["from"] for e in network["edges"] if (e.get("reservoirs") or {}).get("O2", 0) > 0}
            bound = [
                n["id"]
                for n in network["nodes"]
                if "OO" in (n.get("occupancy") or {}).values() and not n.get("computed_by") and n["id"] not in written
            ]
            self.assertTrue(bound)  # the hydrated family writes bound O2 next to bound water without its release
            for node in bound:
                proposed = [s for s in steps if s["from"] == node and s["proposed"]]
                self.assertTrue(proposed, node)
                self.assertTrue(all(s["vector"].get("O2") == 1 and s["species"] == "O2" for s in proposed))
            self.assertEqual(
                product_release_steps(None, network, registry, [registry.reaction("oer")]),
                [s for s in steps if not s["proposed"]],
            )
