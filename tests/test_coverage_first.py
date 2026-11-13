"""Coverage first: probe adsorbates on the sites, read the surface Pourbaix diagram, pick surfaces.

Part 1 (this file starts with it): the coverage campaign and its count, the placement of
surface patterns with many adsorbates to orient, and the short names of surfaces.
"""

import contextlib
import io
import os
import shutil
import tempfile
import unittest
from itertools import permutations, product
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from mechanet.builder import ALL_ORDERS_UP_TO, MANY_SITES_BEAM, Builder
from mechanet.cli import main
from mechanet.families.coverage import count, patterns
from mechanet.fileio import read_json
from mechanet.notation import surface_name
from mechanet.plugins import default_registry
from mechanet.project import coverage_count, generate, load_project, resolve_config
from mechanet.structure import minimum_image

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
RUO2 = EXAMPLES / "ruo2_110"
LDH = EXAMPLES / "nife_ldh"


def run(*args):
    """mechanet with text output: (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main([str(a) for a in args])
    return code, out.getvalue(), err.getvalue()


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {"MECHANET_CONFIG_DIR": str(self.root / "config")})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def project_copy(self, example=RUO2, name="p", **scope):
        shutil.copytree(example, self.root / name, ignore=shutil.ignore_patterns("reference", "campaign"))
        path = self.root / name / "project.yaml"
        data = yaml.safe_load(path.read_text())
        data["scope"].update(scope)
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return self.root / name

    def builder(self, project, name="b", **search):
        loaded = load_project(project / "project.yaml")
        config = resolve_config(loaded, cluster=False)
        config.setdefault("geometry", {}).setdefault("orientation_search", {}).update(search)
        return Builder(config, self.root / name, base_dir=loaded["_dir"], registry=default_registry(config["plugins"]))


class MinimumImageTests(unittest.TestCase):
    """The inverse cell and the image offsets are kept between calls: every digit must stay."""

    @staticmethod
    def reference(vector, cell, periodic=(True, True, True)):
        vector = np.asarray(vector, dtype=float)
        cell = np.asarray(cell, dtype=float)
        frac = vector @ np.linalg.inv(cell)
        frac = frac - np.where(periodic, np.rint(frac), 0)
        offsets = np.array(list(product(*[(-1, 0, 1) if p else (0,) for p in periodic])))
        trial = (frac[None, :] + offsets) @ cell
        return trial[np.argmin(np.einsum("ij,ij->i", trial, trial))]

    def test_same_digits_as_a_fresh_computation(self):
        rng = np.random.default_rng(7)
        cells = [np.diag([3.0, 4.0, 20.0]), np.array([[6.2, 0.0, 0.0], [-3.1, 5.4, 0.0], [0.9, -1.4, 27.1]])]
        cells += [rng.normal(size=(3, 3)) * 2 + np.diag([8.0, 9.0, 15.0]) for _ in range(4)]
        for cell in cells:
            for periodic in ((True, True, True), (True, True, False), (False, True, True)):
                for _ in range(40):
                    vector = rng.normal(size=3) * 12
                    new, old = minimum_image(vector, cell, periodic), self.reference(vector, cell, periodic)
                    self.assertTrue(np.array_equal(new, old), (cell, periodic, vector))

    def test_a_cell_given_as_lists_and_many_cells_in_turn(self):
        rng = np.random.default_rng(3)
        for _ in range(150):  # more cells than are kept at once
            cell = rng.normal(size=(3, 3)) + np.diag([7.0, 7.0, 7.0])
            vector = rng.normal(size=3) * 9
            self.assertTrue(np.array_equal(minimum_image(vector, cell.tolist()), self.reference(vector, cell)))


class PlacementOrderTests(TempCase):
    def test_every_order_up_to_three_sites_two_orders_beyond(self):
        builder = self.builder(self.project_copy())
        self.assertEqual(ALL_ORDERS_UP_TO, 3)
        for n in (1, 2, 3):
            sites = [f"s{k}" for k in range(n)]
            orders, beam = builder._placement_orders(sites)
            self.assertEqual(orders, list(permutations(sites)))
            self.assertEqual(beam, int(builder.search_cfg["beam_width"]))
        sites = ["s0", "s1", "s2", "s3", "s4"]
        orders, beam = builder._placement_orders(sites)
        self.assertEqual(orders, [tuple(sites), tuple(reversed(sites))])
        self.assertEqual(beam, MANY_SITES_BEAM)

    def test_four_adsorbates_to_orient_build_and_are_reported(self):
        builder = self.builder(self.project_copy())
        occupancy = {site: "OH" for site in builder.sites}  # two bridging O and two open metals: four OH
        self.assertEqual(len(occupancy), 4)
        with patch.object(Builder, "_beam", autospec=True, side_effect=Builder._beam) as beam:
            node_id = builder.node(occupancy, "coverage", "+".join(occupancy))
        tried = {call.args[1] for call in beam.call_args_list}
        self.assertEqual(tried, {tuple(sorted(occupancy)), tuple(sorted(occupancy, reverse=True))})
        self.assertTrue(builder.nodes[node_id]["state_ids"])
        self.assertEqual(builder.limited_searches, [node_id])
        two = builder.node({site: "OH" for site in list(builder.sites)[:2]}, "coverage", "two")
        self.assertEqual(builder.limited_searches, [node_id], two)
        bare = builder.node({"m0038": "O", "m0046": "O"}, "coverage", "two O")  # nothing to orient at all
        self.assertEqual(builder.limited_searches, [node_id], bare)
        report = builder.finish(["coverage"])
        self.assertEqual(report["limited_orientation_search"]["nodes"], 1)
        self.assertEqual(report["limited_orientation_search"]["all_orders_up_to"], 3)

    def test_a_build_without_such_intermediates_says_nothing(self):
        builder = self.builder(self.project_copy(), name="plain")
        report = builder.finish(["aem"])
        self.assertNotIn("limited_orientation_search", report)
        self.assertNotIn("all_orders_up_to", report["orientation_search"])

    def test_the_limit_is_a_setting(self):
        builder = self.builder(self.project_copy(), all_orders_up_to=4)
        self.assertEqual(builder.ignored_search_keys, [])
        self.assertEqual(len(builder._placement_orders(["a", "b", "c", "d"])[0]), 24)
        self.assertEqual(builder.search_cfg["all_orders_up_to"], 4)
        with self.assertRaisesRegex(ValueError, "all_orders_up_to must be at least 1"):
            self.builder(self.project_copy(name="q"), name="bad", all_orders_up_to=0)


class CoverageCountTests(TempCase):
    def test_count_matches_the_campaign_that_is_built(self):
        project = self.project_copy()
        found = coverage_count(project / "project.yaml")
        self.assertEqual((found["sites"], found["surfaces"], found["distinct"]), (4, 48, 27))
        self.assertIn("48 surfaces on 4 sites, 27 distinct after symmetry", found["text"])
        self.assertIn("up to 2 sites changed at once", found["text"])
        report = generate(project / "project.yaml", coverage=True)
        self.assertEqual(Path(report["campaign"]), project / "coverage")
        self.assertEqual(report["families"], ["coverage"])
        self.assertEqual((report["candidate_nodes"], report["computed_nodes"]), (48, 27))
        self.assertEqual(report["coverage"]["distinct"], 27)
        self.assertEqual(report["complete_stoichiometric_cycles"], 0)
        self.assertFalse((project / "campaign").exists())

    def test_every_combination_of_o_and_oh_on_the_ldh_edge(self):
        scope = {"coverage": {"max_changed_sites": None, "states": {"terminal": ["O", "OH"], "bridge": ["O", "OH"]}}}
        project = self.project_copy(LDH, **scope)
        found = coverage_count(project / "project.yaml")
        self.assertEqual((found["sites"], found["surfaces"], found["distinct"]), (8, 256, 136))
        self.assertIn("bridge: O, OH; terminal: O, OH; every combination", found["text"])

    def test_states_per_site_are_listed_site_by_site(self):
        scope = {"coverage": {"sites": ["m0038", "m0046"], "states": {"m0038": ["OH"], "m0046": ["OH", "O"]}}}
        project = self.project_copy(**scope)
        found = coverage_count(project / "project.yaml")
        self.assertEqual(found["states"], {"m0038": ["empty", "OH"], "m0046": ["empty", "OH", "O"]})
        self.assertEqual(found["surfaces"], 2 * 3)
        self.assertIn("m0038: OH, empty; m0046: O, OH, empty", found["text"])

    def test_patterns_are_what_the_family_builds(self):
        builder = self.builder(self.project_copy())
        listed = [occupancy for occupancy, _ in patterns(builder)]
        self.assertEqual(len(listed) + 1, count(builder)["surfaces"])
        builder.finish(["coverage"])
        built = [n["occupancy"] for n in builder.nodes.values() if n["id"] != "ref"]

        def as_set(occupancies):
            return {tuple(sorted(occupancy.items())) for occupancy in occupancies}

        self.assertEqual(len(as_set(listed)), len(listed))
        self.assertEqual(as_set(listed), as_set(built))

    def test_coverage_campaign_takes_no_family_list(self):
        project = self.project_copy()
        with self.assertRaisesRegex(ValueError, "builds the coverage family only"):
            generate(project / "project.yaml", families=["aem"], coverage=True)


class CoverageCommandTests(TempCase):
    def test_count_builds_nothing(self):
        project = self.project_copy()
        code, out, _ = run("generate", project, "--coverage", "--count")
        self.assertEqual(code, 0)
        self.assertIn("coverage: 48 surfaces on 4 sites, 27 distinct after symmetry", out)
        self.assertIn("nothing built", out)
        self.assertFalse((project / "coverage").exists())

    def test_count_needs_coverage(self):
        project = self.project_copy()
        code, _, err = run("generate", project, "--count")
        self.assertEqual(code, 2)
        self.assertIn("--count goes with --coverage", err)

    def test_the_count_is_said_before_the_build_and_in_the_summary(self):
        scope = {"coverage": {"sites": ["m0038", "m0046"], "states": {"open_metal": ["OH"]}}}
        project = self.project_copy(**scope)
        code, out, err = run("generate", project, "--coverage")
        self.assertEqual(code, 0)
        line = "coverage: 4 surfaces on 2 sites, 3 distinct after symmetry (open_metal: OH, empty; every combination)"
        self.assertIn(line, err)
        self.assertIn(line, out)
        self.assertTrue((project / "coverage" / "network.json").is_file())
        network = read_json(project / "coverage" / "network.json")
        self.assertEqual(network["families"], ["coverage"])
        code, out, err = run("--json", "generate", project, "--coverage", "--out", self.root / "again")
        self.assertEqual((code, err), (0, ""))  # JSON output: nothing but the report


class SurfaceNameTests(unittest.TestCase):
    SITES = [
        {"id": "a0061", "kind": "bridge", "native": "OH"},
        {"id": "a0062", "kind": "bridge", "native": "O"},
        {"id": "a0065", "kind": "terminal", "native": "OH"},
        {"id": "m0012", "kind": "open_metal", "native": "empty"},
    ]

    def test_counts_per_site_class(self):
        self.assertEqual(surface_name(self.SITES), "terminal 1 OH, 1 empty | bridge 1 O, 1 OH")
        self.assertEqual(
            surface_name(self.SITES, {"a0062": "OH", "m0012": "OOH"}), "terminal 1 OH, 1 OOH | bridge 2 OH"
        )
        by_id = {site["id"]: site for site in self.SITES}
        self.assertEqual(surface_name(by_id, {"a0061": "O"}), "terminal 1 OH, 1 empty | bridge 2 O")

    def test_coupled_sites_and_one_class_only(self):
        self.assertEqual(
            surface_name(self.SITES, {"a0061": "O"}, ["a0061", "a0062"]),
            "terminal 1 OH, 1 empty | bridge 2 O (61-62 coupled)",
        )
        self.assertEqual(surface_name(self.SITES[:2], {"a0061": "O"}), "bridge 2 O")


if __name__ == "__main__":
    unittest.main()
