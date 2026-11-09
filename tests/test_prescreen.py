"""The calculator interface and `mechanet prescreen`: starts relaxed by a fast in-process
calculator (an ML potential in practice) and judged by the topology checks of collect, before
any DFT.

The campaign is the CO2-to-CO network on Pt(111) (examples/plugins/co2rr.py), generated only:
two starts of *COOH (the same COOH turned by 120 degrees about the site), one of *CO and the
bare surface. Scripted calculators give designed outcomes; EMT runs the command end to end.
"""

import contextlib
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from mechanet import symmetry
from mechanet.calculators import ASECalculator, Relaxation, load_calculator, parse_options, to_ase
from mechanet.cli import main
from mechanet.fileio import read_json
from mechanet.plugins import Registry, default_registry
from mechanet.prescreen import prescreen, prescreen_text
from mechanet.structure import Structure, from_ase, read_poscar

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "examples" / "plugins" / "co2rr.py"


def cli(*args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(["--json", *map(str, args)])
    if code != 0:
        raise AssertionError(f"mechanet {' '.join(map(str, args))} exited {code}: {out.getvalue()}")
    return json.loads(out.getvalue())


def start_key(positions):
    return np.round(np.asarray(positions, dtype=float), 6).tobytes()


def positions(state):
    return read_poscar(state / "POSCAR").positions


def same(xyz, symbols):
    return xyz


def lift(elements, dz):
    def move(xyz, symbols):
        for i, symbol in enumerate(symbols):
            if symbol in elements:
                xyz[i, 2] += dz
        return xyz

    return move


class Scripted:
    """Relaxes every start to a designed structure and energy, found by the start's positions."""

    def __init__(self, plan, tag="scripted"):
        self.plan, self.tag, self.calls = plan, tag, 0

    def describe(self):
        return {"calculator": self.tag, "class": "test_prescreen.Scripted"}

    def relax(self, structure, fmax=0.05, steps=300):
        self.calls += 1
        move, energy = self.plan[start_key(structure.positions)]
        if isinstance(energy, Exception):
            raise energy
        return Relaxation(energy, move(structure.positions.copy(), structure.symbols), 0.01, True, 7)


class CalculatorTests(unittest.TestCase):
    @staticmethod
    def slab():
        """Pt(111) 2x2x3 with O in an fcc hollow: bottom layer fixed, O free only along z."""
        from ase.build import add_adsorbate, fcc111

        atoms = fcc111("Pt", (2, 2, 3), vacuum=6.0)
        add_adsorbate(atoms, "O", 1.9, "fcc")
        mobile = np.ones((len(atoms), 3), dtype=bool)
        mobile[atoms.positions[:, 2] < atoms.positions[:, 2].min() + 0.1] = False
        mobile[-1] = (False, False, True)
        ids = [f"a{i + 1:04d}" for i in range(len(atoms))]
        return Structure(atoms.get_chemical_symbols(), atoms.positions, atoms.cell.array, mobile, ids)

    def test_selective_dynamics_carry_over(self):
        start = self.slab()
        np.testing.assert_array_equal(from_ase(to_ase(start)).mobile, start.mobile)
        relaxation = load_calculator("emt").relax(start, fmax=0.05, steps=60)
        moved = relaxation.positions - start.positions
        self.assertTrue(np.allclose(moved[~start.mobile], 0.0))  # fixed coordinates stay
        self.assertGreater(abs(moved[-1, 2]), 1e-3)  # the O moved, along z only
        self.assertGreater(relaxation.steps, 0)
        self.assertTrue(np.isfinite(relaxation.energy_eV))
        single = load_calculator("emt").relax(start, steps=0)
        self.assertEqual(single.steps, 0)
        np.testing.assert_allclose(single.positions, start.positions)

    def test_specs_options_and_registry(self):
        self.assertEqual(load_calculator("emt").describe()["class"], "ase.calculators.emt.EMT")
        spelled = load_calculator("ase.calculators.emt:EMT", optimizer="FIRE")
        self.assertIsInstance(spelled, ASECalculator)
        self.assertEqual(spelled.describe()["calculator"], "ase.calculators.emt:EMT")
        self.assertEqual(spelled.describe()["optimizer"], "FIRE")
        with self.assertRaisesRegex(ValueError, "Unknown calculator 'nope'"):
            load_calculator("nope")
        with self.assertRaisesRegex(ValueError, "cannot import"):
            load_calculator("no_such_module_for_mechanet:thing")
        with self.assertRaisesRegex(ValueError, "has no"):
            load_calculator("ase.calculators.emt:NoSuchCalculator")
        with self.assertRaisesRegex(ValueError, "optimizer"):
            ASECalculator(object(), "x", optimizer="CG")
        registry = Registry()
        registry.add_calculator("toy", lambda: object(), "returns nothing a relaxation can use")
        with self.assertRaisesRegex(ValueError, "neither an ASE calculator"):
            load_calculator("toy", registry=registry)
        registry.add_calculator("emt-again", lambda **options: load_calculator("emt").calc, "EMT by another name")
        self.assertEqual(load_calculator("emt-again", registry=registry).describe()["calculator"], "emt-again")
        self.assertEqual(
            parse_options(["model=medium", "dispersion=true", "n=3", "dtype="]),
            {"model": "medium", "dispersion": True, "n": 3, "dtype": ""},
        )
        with self.assertRaisesRegex(ValueError, "key=value"):
            parse_options(["model"])
        names = {c["name"] for c in cli("plugins")["calculators"]}
        self.assertLessEqual({"emt", "mace-mp"}, names)


class PrescreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env = patch.dict(os.environ, {"MECHANET_CONFIG_DIR": str(cls.tmp / "config")})
        cls.env.start()
        project = cls.tmp / "pt"
        shutil.copytree(ROOT / "examples" / "pt_111", project)
        data = yaml.safe_load((project / "project.yaml").read_text())
        data.update(reaction="co2rr_co", plugins=[str(PLUGIN)])
        data["scope"]["families"] = ["co2rr_carboxyl"]
        (project / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        cli("generate", project / "project.yaml")
        cls.campaign = project / "campaign"
        cls.registry = default_registry([PLUGIN])
        states = sorted((cls.campaign / "states").iterdir())

        def holds(state, name):
            return name in (read_json(state / "state.json").get("occupancy") or {}).values()

        cls.cooh = [s for s in states if holds(s, "COOH")]
        cls.co = [s for s in states if holds(s, "CO")]
        cls.ref = cls.campaign / "states" / "ref"

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        shutil.rmtree(self.campaign / "prescreen", ignore_errors=True)
        self.assertEqual((len(self.cooh), len(self.co)), (2, 1), "the network these tests were written for changed")

    def plan(self, cooh=((same, -10.0), (same, -10.0)), co=(same, -5.0), ref=(same, 0.0)):
        (v1, v2), (co_state,) = self.cooh, self.co
        return {
            start_key(positions(v1)): cooh[0],
            start_key(positions(v2)): cooh[1],
            start_key(positions(co_state)): co,
            start_key(positions(self.ref)): ref,
        }

    def run_prescreen(self, calculator, **options):
        return prescreen(self.campaign, calculator, registry=self.registry, **options)

    def test_outcomes_and_what_is_suggested(self):
        (v1, v2), (co,) = self.cooh, self.co
        plan = self.plan(
            cooh=((same, -10.0), (lift({"H"}, 3.0), -9.5)),  # the H of the second start leaves
            co=(lift({"C", "O"}, 5.0), -5.0),  # CO desorbs
            ref=(same, RuntimeError("no parameters for this element")),
        )
        # A review.json written for a DFT result must not decide for the ML structure.
        (v2 / "review.json").write_text(json.dumps({"accept_relaxed_topology": True, "comment": "DFT checked"}))
        try:
            result = self.run_prescreen(Scripted(plan))
        finally:
            (v2 / "review.json").unlink()
        rows = {s["state_id"]: s for s in result["states"]}
        self.assertEqual(rows[v1.name]["outcome"], "kept")
        self.assertTrue(rows[v1.name]["suggested"])
        self.assertEqual(rows[v2.name]["outcome"], "changed")
        self.assertFalse(rows[v2.name]["suggested"])
        self.assertEqual(rows[v2.name]["why_not"], "changed bonds (would need review)")
        self.assertIn("H h_m0010 is no longer bonded to O o2_m0010", rows[v2.name]["changes"])
        # every start of *CO gave CO off, so all are suggested: the calculator does not rule *CO out
        self.assertEqual(rows[co.name]["outcome"], "released")
        self.assertTrue(rows[co.name]["suggested"])
        self.assertEqual(rows["ref"]["outcome"], "failed")
        self.assertTrue(rows["ref"]["suggested"])
        self.assertIn("no parameters", rows["ref"]["error"])
        self.assertEqual(result["suggested_state_ids"], sorted([v1.name, co.name, "ref"]))
        counts = {name: n for name, n in result["counts"].items() if n}
        self.assertEqual(counts, {"kept": 1, "changed": 1, "released": 1, "failed": 1})
        notes = {n["node_id"]: n.get("note", "") for n in result["nodes"]}
        self.assertIn("no start kept", notes[read_json(co / "state.json")["node_id"]])
        self.assertIn("no verdict", notes["ref"])
        self.assertTrue(any("failed on 1 start" in w for w in result["warnings"]))
        folder = self.campaign / "prescreen"
        self.assertEqual((folder / "suggested_states.txt").read_text().split(), result["suggested_state_ids"])
        self.assertEqual(len((folder / "prescreen.csv").read_text().strip().splitlines()), 5)
        self.assertFalse((folder / "ref" / "CONTCAR").exists())
        np.testing.assert_allclose(read_poscar(folder / v1.name / "CONTCAR").positions, positions(v1), atol=1e-9)
        text = prescreen_text(result)
        self.assertIn("suggested for DFT: 3 of 4 starts", text)
        self.assertIn(f"{v2.name} (*COOH", text)

    def test_reuse_window_and_cap(self):
        v1, v2 = self.cooh
        calculator = Scripted(self.plan(cooh=((same, -10.0), (same, -9.7))))
        result = self.run_prescreen(calculator)
        self.assertEqual(calculator.calls, 4)
        rows = {s["state_id"]: s for s in result["states"]}
        self.assertEqual(rows[v2.name]["relative_eV"], 0.3)
        self.assertEqual(rows[v2.name]["why_not"], f"0.30 eV above {v1.name}")
        # The relaxations are reused; only the suggestion changes with the window and the cap.
        wider = self.run_prescreen(calculator, window_eV=0.5)
        self.assertEqual((calculator.calls, wider["reused"]), (4, 4))
        self.assertIn(v2.name, wider["suggested_state_ids"])
        capped = self.run_prescreen(calculator, window_eV=0.5, max_per_node=1)
        self.assertEqual({s["state_id"]: s for s in capped["states"]}[v2.name]["why_not"], "beyond the 1 lowest starts")
        self.run_prescreen(calculator, redo=True)
        self.assertEqual(calculator.calls, 8)
        # Another calculator (or other settings) relaxes every start again.
        other = Scripted(self.plan(), tag="another calculator")
        self.run_prescreen(other)
        self.assertEqual(other.calls, 4)

    def test_symmetry_images_are_merged(self):
        v1, v2 = self.cooh
        calculator = Scripted(self.plan())  # both COOH starts stay where they are, at one energy
        with patch.object(symmetry, "_dataset", return_value=None):  # no symmetry known
            result = self.run_prescreen(calculator)
        self.assertEqual(result["settings"]["symmetry_operations"], 1)
        self.assertLessEqual({v1.name, v2.name}, set(result["suggested_state_ids"]))
        # The slab's 3-fold rotations about the Pt atom under the site, as spglib gives them
        # (fractional coordinates): the second start is the first one turned by 120 degrees.
        reference = read_poscar(self.ref / "POSCAR")
        metal = read_json(self.ref / "state.json")["atom_ids"].index("a0010")
        lattice = reference.cell.T
        angle = 2 * np.pi / 3
        turn = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        w = np.rint(np.linalg.inv(lattice) @ turn @ lattice).astype(int)
        f = np.linalg.inv(lattice) @ reference.positions[metal]
        dataset = SimpleNamespace(
            rotations=[np.eye(3, dtype=int), w, w @ w], translations=[np.zeros(3), f - w @ f, f - w @ w @ f]
        )
        with patch.object(symmetry, "_dataset", return_value=dataset):
            result = self.run_prescreen(calculator)
        self.assertEqual(calculator.calls, 4)  # reused: only the comparison changed
        self.assertEqual(result["settings"]["symmetry_operations"], 3)
        rows = {s["state_id"]: s for s in result["states"]}
        self.assertEqual(rows[v2.name]["why_not"], f"a symmetry image of {v1.name}")
        self.assertEqual(rows[v2.name]["same_as"], v1.name)
        self.assertNotIn(v2.name, result["suggested_state_ids"])

    @unittest.skipUnless(importlib.util.find_spec("spglib"), "spglib is not installed")
    def test_symmetry_images_from_spglib(self):
        v1, v2 = self.cooh
        reference = read_poscar(self.ref / "POSCAR")
        fixed = ~np.asarray(read_json(self.ref / "state.json")["relax_mobile"], dtype=bool).all(axis=1)
        images = symmetry.surface_images(reference, [0, 0, 1], [(bool(x), False, None) for x in fixed])
        self.assertTrue(any(not np.allclose(rotation, np.eye(3)) for rotation, _ in images))
        result = self.run_prescreen(Scripted(self.plan()))
        rows = {s["state_id"]: s for s in result["states"]}
        self.assertEqual(rows[v2.name]["why_not"], f"a symmetry image of {v1.name}")

    def test_command_with_emt(self):
        out = cli("prescreen", self.campaign, "--calculator", "emt", "--steps", "3", "--states", "ref")
        self.assertEqual(out["starts"], 1)
        self.assertEqual(out["calculator"]["class"], "ase.calculators.emt.EMT")
        self.assertTrue(any("toy potential" in w for w in out["warnings"]))
        self.assertTrue((self.campaign / "prescreen" / "ref" / "CONTCAR").is_file())
        # A calculator from a plugin file the campaign was not generated with.
        plugin = self.tmp / "my_calculator.py"
        plugin.write_text(
            "def make(**options):\n"
            "    from ase.calculators.emt import EMT\n"
            "    return EMT()\n\n\n"
            "def register(registry):\n"
            "    registry.add_calculator('my-emt', make, 'EMT from a plugin file')\n"
        )
        out = cli(
            "prescreen", self.campaign, "--plugin", plugin, "--calculator", "my-emt", "--steps", "2", "--states", "ref"
        )
        self.assertEqual(out["calculator"]["calculator"], "my-emt")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = main(["prescreen", str(self.campaign), "--calculator", "emt", "--option", "nonsense"])
        self.assertEqual(code, 2)
        self.assertIn("key=value", err.getvalue())


if __name__ == "__main__":
    unittest.main()
