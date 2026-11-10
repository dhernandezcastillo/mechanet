"""Starts of one intermediate that are rotations or mirror images of each other on the slab.

On a top site of Pt(111) (a three-fold axis and three mirrors) the orientation search used to
write the same *COOH twice, turned by 120 degrees, so DFT would have computed it twice. With
spglib, symmetry images of an orientation count once in the search and the second start is a
different orientation; a check after the search sets aside any start that is still an image
of a kept one. Without spglib, generate says that it could not merge them.
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

import numpy as np
import yaml

from mechanet.analysis import structure_difference
from mechanet.builder import Builder
from mechanet.cli import main
from mechanet.fileio import read_json
from mechanet.plugins import default_registry
from mechanet.structure import Structure, read_poscar
from mechanet.symmetry import describe_operation, spglib_available, surface_images

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "examples" / "plugins" / "co2rr.py"
HAS_SPGLIB = spglib_available()


def generate(tmp, **geometry):
    """The CO2-to-CO network on Pt(111) (generated only); returns (campaign, generate's JSON)."""
    project = tmp / "pt"
    shutil.copytree(ROOT / "examples" / "pt_111", project)
    data = yaml.safe_load((project / "project.yaml").read_text())
    data.update(reaction="co2rr_co", plugins=[str(PLUGIN)])
    data["scope"]["families"] = ["co2rr_carboxyl"]
    if geometry:
        data["geometry"] = {**(data.get("geometry") or {}), **geometry}
    (project / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(["--json", "generate", str(project / "project.yaml")])
    if code != 0:
        raise AssertionError(f"generate exited {code}: {out.getvalue()}")
    return project / "campaign", json.loads(out.getvalue())


def starts(campaign, adsorbate):
    return [
        read_poscar(state / "POSCAR")
        for state in sorted((campaign / "states").iterdir())
        if (read_json(state / "state.json").get("occupancy") or {}).get("m0010") == adsorbate
    ]


def closest_image(campaign, first, second):
    """The smallest largest-atom-shift between ``first`` and ``second`` under any operation of the slab."""
    reference = read_poscar(campaign / "states" / "ref" / "POSCAR")
    mobile = np.asarray(read_json(campaign / "states" / "ref" / "state.json")["relax_mobile"], dtype=bool)
    operations = surface_images(reference, [0, 0, 1], [(bool(row.any()), False, None) for row in mobile])
    return min(
        structure_difference(
            first, Structure(second.symbols, second.positions @ r.T + t, second.cell, second.mobile, second.ids)
        )["max_shift_A"]
        for r, t in operations
    )


class StartSymmetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        env = patch.dict(os.environ, {"MECHANET_CONFIG_DIR": str(self.tmp / "config")})
        env.start()
        self.addCleanup(env.stop)

    def test_operations_are_named(self):
        a = np.radians(120)
        turn = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        self.assertEqual(describe_operation(turn, [0, 0, 1]), "turned by 120°")
        self.assertEqual(describe_operation(np.diag([1.0, -1.0, 1.0]), [0, 0, 1]), "mirrored")

    @unittest.skipUnless(HAS_SPGLIB, "spglib is not installed")
    def test_no_two_starts_of_an_intermediate_are_images(self):
        campaign, result = generate(self.tmp)
        report = read_json(campaign / "build_report.json")
        self.assertEqual(report["start_symmetry"]["source"], "spglib")
        self.assertEqual(report["start_symmetry"]["operations"], 54)  # P3m1 on the 3x3 cell
        self.assertFalse([w for w in result["warnings"] if "spglib" in w])
        first, second = starts(campaign, "COOH")  # still two starts, now two different ones
        self.assertGreaterEqual(closest_image(campaign, first, second), 0.75)
        builder = Builder(
            read_json(campaign / "config.json"),
            self.tmp / "again",
            base_dir=campaign,
            registry=default_registry([PLUGIN]),
        )
        names = sorted(describe_operation(r, builder.normal) for r, _ in builder._node_images({"m0010": "COOH"}, None))
        self.assertEqual(names, ["mirrored", "mirrored", "mirrored", "turned by 120°", "turned by 240°"])

    @unittest.skipUnless(HAS_SPGLIB, "spglib is not installed")
    def test_a_start_left_an_image_by_the_search_is_set_aside(self):
        """The check after the search, with a search blind to symmetry (as before this change)."""
        original = Builder._diverse

        def blind(self, ranked, atoms_of, limit, images=()):
            return original(self, ranked, atoms_of, limit)

        with patch.object(Builder, "_diverse", blind):
            campaign, _ = generate(self.tmp)
        merged = read_json(campaign / "build_report.json")["start_symmetry"]["merged"]
        self.assertTrue(merged)
        self.assertIn(merged[0]["operation"], ("turned by 120°", "turned by 240°"))
        self.assertTrue(merged[0]["replaced_by_next_orientation"])
        self.assertLess(merged[0]["distance_after_operation_A"], 1e-3)
        written = starts(campaign, "COOH")
        for i in range(len(written)):
            for j in range(i + 1, len(written)):
                self.assertGreaterEqual(closest_image(campaign, written[i], written[j]), 0.75)

    def test_without_spglib_generate_says_so(self):
        with patch("mechanet.builder.spglib_available", return_value=False):
            campaign, result = generate(self.tmp)
        symmetry = read_json(campaign / "build_report.json")["start_symmetry"]
        self.assertIsNone(symmetry["source"])
        self.assertIn("spglib is not installed", symmetry["condition"])
        self.assertTrue([w for w in result["warnings"] if "spglib is not installed" in w])
        first, second = starts(campaign, "COOH")
        if HAS_SPGLIB:  # the old outcome: the second start is the first turned by 120 degrees
            self.assertLess(closest_image(campaign, first, second), 1e-6)

    def test_symmetry_reduction_off(self):
        campaign, result = generate(self.tmp, symmetry_reduction=False)
        symmetry = read_json(campaign / "build_report.json")["start_symmetry"]
        self.assertIsNone(symmetry["source"])
        self.assertIn("symmetry_reduction is off", symmetry["condition"])
        self.assertFalse([w for w in result["warnings"] if "spglib" in w])


if __name__ == "__main__":
    unittest.main()
