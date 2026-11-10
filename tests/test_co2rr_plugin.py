"""The generality gate: CO2 reduction to CO (examples/plugins/co2rr.py), with carbon adsorbates
given atom by atom, runs end to end with no change to the core: build, synthetic VASP, the
checks of relaxed structures, thermo, pathways and Pourbaix.

*COOH forms with DeltaG = +0.50 eV at 0 V vs RHE, *CO from it with -0.20 eV, and CO leaves
with -0.10 eV, so U_L = -0.50 V; the references put the equilibrium potential at -0.10 V, so
the overpotential is 0.40 V.
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

from mechanet.adsorbates import Adsorbate, extra_elements
from mechanet.analysis import DEFAULT_HYSTERESIS, _campaign_context, classify_microstate
from mechanet.cli import main
from mechanet.fileio import read_json
from mechanet.plugins import default_registry
from mechanet.runner import run_state
from mechanet.structure import Structure, distance_matrix, minimum_image, read_poscar, write_poscar
from mechanet.topology import topology_check

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "examples" / "plugins" / "co2rr.py"
MU = {"H2": -6.77, "H2O": -14.22, "CO2": -22.96}
DG = (0.50, -0.20, -0.10)  # *COOH formed, *CO formed, CO released; eV at 0 V vs RHE
MU["CO"] = MU["CO2"] + MU["H2"] - MU["H2O"] + sum(DG)  # CO2 + H2 -> CO + H2O: +0.20 eV
CORRECTIONS = {"H": 0.30, "O": 0.05}  # the default electronic_mode_corrections per atom


def cli(*args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(["--json", *map(str, args)])
    if code != 0:
        raise AssertionError(f"mechanet {' '.join(map(str, args))} exited {code}: {out.getvalue()}")
    return json.loads(out.getvalue())


def free_energy(delta):
    """G(node) - G(bare surface) designed from the node's composition change."""
    key = tuple(sorted((element, n) for element, n in delta.items() if n))
    if not key:
        return 0.0
    if key == (("C", 1), ("H", 1), ("O", 2)):  # *COOH
        return MU["CO2"] + 0.5 * MU["H2"] + DG[0]
    if key == (("C", 1), ("O", 1)):  # *CO
        return MU["CO2"] + MU["H2"] - MU["H2O"] + DG[0] + DG[1]
    raise ValueError(f"no designed energy for {delta}")


class TemplateAdsorbateTests(unittest.TestCase):
    def test_templates_are_checked_and_keyed_by_their_tree(self):
        registry = default_registry([PLUGIN])
        self.assertEqual(registry.adsorbate("COOH").signature, ("graph", "C(O,O(H))"))
        self.assertEqual(registry.adsorbate("CO").signature, ("graph", "C(O)"))
        self.assertEqual(extra_elements(registry.adsorbates.values()), {"C"})
        # the built-in oxygen and chlorine chemistry brings no other element: checks as before
        self.assertEqual(extra_elements(default_registry().adsorbates.values()), frozenset())
        with self.assertRaisesRegex(ValueError, "not listed before it"):
            Adsorbate(
                "X",
                {"C": 1, "O": 1, "H": 1},
                "C",
                "template",
                atoms=(("h", "H", "o", (0, 0, 2.1)), ("o", "O", "", (0, 0, 1.2))),
            )
        with self.assertRaisesRegex(ValueError, "does not match"):
            Adsorbate("Y", {"C": 1}, "C", "template", atoms=(("o", "O", "", (0, 0, 1.2)),))
        with self.assertRaisesRegex(ValueError, "only with geometry 'template'"):
            Adsorbate("Z", {"O": 1}, "O", "atom", atoms=(("h", "H", "", (0, 0, 1.0)),))


@unittest.skipIf(os.environ.get("MECHANET_SKIP_SLOW"), "slow plugin workflow disabled by environment")
class CO2RRPluginTests(unittest.TestCase):
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
        data.update(reaction="co2rr_co", plugins=[str(PLUGIN)])
        data["scope"]["families"] = ["co2rr_carboxyl"]
        data["vasp"] = {
            "potcars": write_potcars(cls.tmp, ("Pt", "C", "O", "H")),
            "magmom_by_element": None,
            "command": [sys.executable, "-I", str(fake)],
        }
        data["frequency"] = {"central": "final"}
        data["thermodynamics"] = {"references": "inputs/references.json", "potentials_V": [0.0, -0.5]}
        (project / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        references = {name: {"E": value, "Gcorr": 0.0} for name, value in MU.items()}
        (project / "inputs" / "references.json").write_text(json.dumps({"references": references}))
        cls.campaign = project / "campaign"
        cls.registry = default_registry([PLUGIN])
        cls.generated = cli("generate", project / "project.yaml")
        for state in sorted((cls.campaign / "states").iterdir()):
            delta = read_json(state / "state.json")["delta_atoms"]
            target = -100.0 + free_energy(delta) - sum(CORRECTIONS.get(e, 0.0) * n for e, n in delta.items())
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
        shutil.rmtree(cls.tmp)

    def state(self, status):
        return next(
            s
            for s in sorted((self.campaign / "states").iterdir())
            if status in (read_json(s / "state.json").get("occupancy") or {}).values()
        )

    def classify(self, state, move=None):
        """Read a structure of ``state`` with some atoms moved, as collect does."""
        meta = read_json(state / "state.json")
        structure = read_poscar(state / "POSCAR")
        initial = distance_matrix(structure)
        for atom, position in (move or {}).items():
            structure.positions[meta["atom_ids"].index(atom)] = position
        return classify_microstate(
            structure,
            meta["atom_ids"],
            distance_matrix(structure),
            _campaign_context(self.campaign),
            dict(DEFAULT_HYSTERESIS),
            initial,
            self.registry,
        )

    def test_the_plugin_builds_its_network(self):
        self.assertEqual(self.generated["complete_stoichiometric_cycles"], 9)
        self.assertEqual(self.generated["nodes_without_viable_start"], [])
        self.assertTrue(cli("validate", self.campaign)["valid"])

    def test_template_atoms_are_placed_as_drawn(self):
        state = self.state("COOH")
        meta = read_json(state / "state.json")
        site = next(iter(meta["occupancy"]))
        ids, d = meta["atom_ids"], distance_matrix(read_poscar(state / "POSCAR"))

        def dist(a, b):
            return float(d[ids.index(a), ids.index(b)])

        self.assertAlmostEqual(dist(site, f"o1_{site}"), 1.22, places=3)
        self.assertAlmostEqual(dist(site, f"o2_{site}"), 1.36, places=3)
        self.assertAlmostEqual(dist(f"o2_{site}", f"h_{site}"), 0.98, places=3)
        self.assertAlmostEqual(dist(site, "a" + site[1:]), 2.00, places=3)
        bonds = {tuple(sorted(b)) for b in meta["expected_bonds"]}
        self.assertEqual(bonds, {(site, f"o1_{site}"), (site, f"o2_{site}"), (f"h_{site}", f"o2_{site}")})

    def test_relaxed_carbon_structures_are_read_by_their_tree(self):
        state = self.state("COOH")
        meta = read_json(state / "state.json")
        site = next(iter(meta["occupancy"]))
        positions = dict(zip(meta["atom_ids"], read_poscar(state / "POSCAR").positions))
        self.assertEqual(self.classify(state), ({site: "COOH"}, None, None))
        # the proton on the other oxygen: still a carboxyl
        c, o1 = positions[site], positions[f"o1_{site}"]
        moved = {f"h_{site}": o1 + 0.98 * (o1 - c) / np.linalg.norm(o1 - c)}
        self.assertEqual(self.classify(state, moved), ({site: "COOH"}, None, None))
        # OH split off: the site holds CO and an OH floats away, outside the site model
        up = np.array([0.0, 0.0, 2.5])
        split = {f"o2_{site}": positions[f"o2_{site}"] + up, f"h_{site}": positions[f"h_{site}"] + up}
        changes, _, reason = self.classify(state, split)
        self.assertIsNone(changes)
        self.assertIn(f"adsorbate atom o2_{site} is not bonded to a reactive site", reason)
        self.assertEqual(self.classify(self.state("CO")), ({site: "CO"}, None, None))

    def test_two_point_binding_waits_for_review(self):
        """A carboxyl whose carbonyl O bends down onto the metals binds through two atoms. The
        site model has one binding atom per adsorbate, so the structure is neither accepted in
        silence nor called released: it waits for review, with the new contacts named."""
        state = self.state("COOH")
        meta = read_json(state / "state.json")
        site = next(iter(meta["occupancy"]))
        start = read_poscar(state / "POSCAR")
        ids = meta["atom_ids"]
        carbon, own, o1 = ids.index(site), ids.index("a" + site[1:]), ids.index(f"o1_{site}")
        # the surface Pt on the side where the carbonyl O points (away from the OH)
        across = minimum_image(start.positions[o1] - start.positions[carbon], start.cell)
        across[2] = 0.0
        across = across / np.linalg.norm(across)
        top = [
            i
            for i, symbol in enumerate(start.symbols)
            if symbol == "Pt" and i != own and abs(start.positions[i, 2] - start.positions[own, 2]) < 0.3
        ]

        def offset(i):
            return minimum_image(start.positions[i] - start.positions[own], start.cell)

        near = [i for i in top if np.linalg.norm(offset(i)) < 3.0]
        neighbour = max(near, key=lambda i: float(offset(i) @ across))
        along = offset(neighbour) / np.linalg.norm(offset(neighbour))
        moved = start.positions.copy()
        moved[o1] = start.positions[carbon] + 1.25 * along + np.array([0.0, 0.0, -0.45])
        path = self.tmp / "bidentate.vasp"
        write_poscar(path, Structure(start.symbols, moved, start.cell, start.mobile, start.ids))
        config = read_json(self.campaign / "config.json")
        result = topology_check(state, meta, config, geometry=path, registry=self.registry, use_review=False)
        self.assertFalse(result["valid_for_assigned_node"])
        self.assertEqual(result["status"], "changed_connectivity_needs_review")
        self.assertEqual(result["microstate"]["status"], "metal_attachment_changed")
        self.assertEqual(result["microstate"]["occupancy_changes"], {site: "COOH"})  # still a carboxyl
        self.assertEqual(result["released"], [])
        new = {tuple(pair) for pair in result["new_active_metal_contacts"]}
        self.assertIn(tuple(sorted((ids[neighbour], f"o1_{site}"))), new)
        self.assertEqual(result["lost_active_metal_contacts"], [])

    def test_carbon_intermediates_give_the_designed_thermodynamics(self):
        cli("thermo", self.campaign, "--electronic-only")
        thermo = read_json(self.campaign / "analysis" / "thermodynamics.json")
        states = read_json(self.campaign / "analysis" / "states.json")["states"]
        self.assertTrue(all(s["electronic_valid"] for s in states), [(s["state_id"], s["status"]) for s in states])
        row = next(c for c in thermo["cycles"] if c["cycle_id"] == "co2rr_carboxyl_m0010" and c["potential_V"] == 0.0)
        self.assertEqual(row["direction"], "cathodic")
        self.assertEqual([round(x, 6) for x in row["step_deltaG_eV"]], list(DG))
        self.assertAlmostEqual(row["limiting_potential_V"], -0.50, places=6)
        self.assertAlmostEqual(row["thermodynamic_overpotential_V"], 0.40, places=6)
        self.assertTrue(thermo["protocol_check"]["consistent"])

    def test_pathways_and_pourbaix_for_carbon_chemistry(self):
        ranked = cli(
            "pathways",
            self.campaign,
            "--rank",
            "--references",
            self.campaign.parent / "inputs" / "references.json",
            "--electronic-only",
            "--potentials",
            "0",
        )
        best = ranked["ranking"]["global_minimum_limiting_potential"]["co2rr_co"]
        self.assertAlmostEqual(best["limiting_potential_V"], -0.50, places=6)
        result = cli(
            "pourbaix", self.campaign, "--electronic-only", "--scale", "RHE", "--at-pH", "0", "--potential=-1:0.5"
        )
        # *CO gives CO off downhill without electron transfer: left out, as bound O2 is for OER
        self.assertEqual({x["label"] for x in result["left_out"]}, {"m0010:CO"})
        steps = [(t["label"], round(t["from_V"], 6)) for t in result["transitions"]]
        self.assertEqual(steps, [("m0010:COOH", -1.0), ("native", -0.5)])


if __name__ == "__main__":
    unittest.main()
