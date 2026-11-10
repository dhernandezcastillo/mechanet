"""A second generality gate: nitrogen reduction to ammonia (examples/plugins/nrr.py).

Seven nitrogen adsorbates (fixed shapes and templates), a turnover whose marker is taken up
(N2) while the product leaves twice (NH3), run end to end with no change to the core: build,
synthetic VASP, the checks of relaxed structures, thermo, pathways and Pourbaix.

Designed steps at 0 V vs RHE (eV): N2 adsorption -0.10, *NNH +0.90, *NNH2 +0.10, *N + NH3
-0.60, *NH -0.30, *NH2 -0.10, *NH3 -0.10, NH3 release -0.10. They sum to -0.30 eV, so the
equilibrium potential is +0.05 V, U_L = -0.90 V and the overpotential 0.95 V. (The two steps
without electron transfer run downhill: an uphill one could not be driven by any potential,
and mechanet then reports no limiting potential.)
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

from mechanet.adsorbates import extra_elements
from mechanet.cli import main
from mechanet.fileio import read_json
from mechanet.plugins import default_registry
from mechanet.runner import run_state
from mechanet.structure import Structure, distance_matrix, read_poscar, write_poscar
from mechanet.topology import DEFAULT_HYSTERESIS, _campaign_context, classify_microstate, topology_check

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "examples" / "plugins" / "nrr.py"
DG = (-0.10, 0.90, 0.10, -0.60, -0.30, -0.10, -0.10, -0.10)
MU = {"H2": -6.77, "N2": -16.63}
MU["NH3"] = 0.5 * (MU["N2"] + 3 * MU["H2"] + sum(DG))  # N2 + 3 H2 -> 2 NH3: -0.30 eV
CORRECTION_H = 0.30  # the default electronic_mode_corrections per H atom
# G(node) - G(bare surface) at 0 V vs RHE, by composition (N, H)
FORMATION = {
    (2, 0): MU["N2"] + sum(DG[:1]),
    (2, 1): MU["N2"] + 0.5 * MU["H2"] + sum(DG[:2]),
    (2, 2): MU["N2"] + MU["H2"] + sum(DG[:3]),
    (1, 0): MU["N2"] + 1.5 * MU["H2"] - MU["NH3"] + sum(DG[:4]),
    (1, 1): MU["N2"] + 2.0 * MU["H2"] - MU["NH3"] + sum(DG[:5]),
    (1, 2): MU["N2"] + 2.5 * MU["H2"] - MU["NH3"] + sum(DG[:6]),
    (1, 3): MU["N2"] + 3.0 * MU["H2"] - MU["NH3"] + sum(DG[:7]),
}
STATES = ("N2", "NNH", "NNH2", "N", "NH", "NH2", "NH3")


def cli(*args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(["--json", *map(str, args)])
    if code != 0:
        raise AssertionError(f"mechanet {' '.join(map(str, args))} exited {code}: {out.getvalue()}")
    return json.loads(out.getvalue())


@unittest.skipIf(os.environ.get("MECHANET_SKIP_SLOW"), "slow plugin workflow disabled by environment")
class NRRPluginTests(unittest.TestCase):
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
        data.update(reaction="nrr", plugins=[str(PLUGIN)])
        data["scope"]["families"] = ["nrr_distal"]
        data["vasp"] = {
            "potcars": write_potcars(cls.tmp, ("Pt", "N", "H")),
            "magmom_by_element": None,
            "command": [sys.executable, "-I", str(fake)],
        }
        data["frequency"] = {"central": "final"}
        data["thermodynamics"] = {"references": "inputs/references.json", "potentials_V": [0.0, -0.9]}
        (project / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        references = {name: {"E": value, "Gcorr": 0.0} for name, value in MU.items()}
        (project / "inputs" / "references.json").write_text(json.dumps({"references": references}))
        cls.campaign = project / "campaign"
        cls.registry = default_registry([PLUGIN])
        cls.generated = cli("generate", project / "project.yaml")
        for state in sorted((cls.campaign / "states").iterdir()):
            delta = read_json(state / "state.json")["delta_atoms"]
            n, h = delta.get("N", 0), delta.get("H", 0)
            target = -100.0 + (FORMATION[(n, h)] if n else 0.0) - CORRECTION_H * h
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

    def state(self, status):
        return next(
            s
            for s in sorted((self.campaign / "states").iterdir())
            if (read_json(s / "state.json").get("occupancy") or {}).get("m0010") == status
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

    def check(self, state, lift):
        """topology_check of ``state`` with the atoms in ``lift`` moved up by that many angstrom."""
        meta = read_json(state / "state.json")
        start = read_poscar(state / "POSCAR")
        positions = start.positions.copy()
        for atom, dz in lift.items():
            positions[meta["atom_ids"].index(atom), 2] += dz
        path = self.tmp / f"moved_{state.name}.vasp"
        write_poscar(path, Structure(start.symbols, positions, start.cell, start.mobile, start.ids))
        config = read_json(self.campaign / "config.json")
        return topology_check(state, meta, config, geometry=path, registry=self.registry, use_review=False)

    def test_the_plugin_builds_its_network(self):
        self.assertEqual(extra_elements(self.registry.adsorbates.values()), {"N"})
        self.assertEqual(self.generated["complete_stoichiometric_cycles"], 9)  # one per top site
        self.assertEqual(self.generated["nodes_without_viable_start"], [])
        self.assertTrue(cli("validate", self.campaign)["valid"])
        network = read_json(self.campaign / "network.json")
        cycle = next(c for c in network["cycles"] if c["id"] == "nrr_distal_m0010")
        self.assertEqual(cycle["reservoirs"], {"H+": -6, "N2": -1, "NH3": 2})
        self.assertEqual(len(cycle["edge_ids"]), 8)

    def test_every_nitrogen_intermediate_is_read_back(self):
        for status in STATES:
            self.assertEqual(self.classify(self.state(status)), ({"m0010": status}, None, None), status)

    def test_changed_nitrogen_structures_are_recognised(self):
        # an H of *NNH2 on the inner N: HN-NH is no registered adsorbate
        state = self.state("NNH2")
        meta = read_json(state / "state.json")
        positions = dict(zip(meta["atom_ids"], read_poscar(state / "POSCAR").positions))
        inner = positions["m0010"]
        changes, _, reason = self.classify(state, {"h1_m0010": inner + np.array([0.95, 0.0, -0.35])})
        self.assertIsNone(changes)
        self.assertIn("matches no registered adsorbate", reason)
        # the same H between the two N, inside the N-H hysteresis band: undecided, so outside the model
        changes, _, reason = self.classify(state, {"h1_m0010": inner + np.array([1.25, 0.0, 0.0])})
        self.assertIsNone(changes)
        self.assertIn("hysteresis band", reason)
        # N2 that left the metal: no bound minimum from this start
        result = self.check(self.state("N2"), {"m0010": 4.0, "n_m0010": 4.0})
        self.assertEqual(result["microstate"]["status"], "adsorbate_released")
        self.assertEqual([item["formula"] for item in result["released"]], ["N₂"])
        self.assertFalse(result["valid_for_assigned_node"])
        # NH3 that left: named as ammonia
        result = self.check(self.state("NH3"), {a: 4.0 for a in ("m0010", "h1_m0010", "h2_m0010", "h3_m0010")})
        self.assertEqual([item["formula"] for item in result["released"]], ["NH₃"])
        # *NNH that lost its H to the vacuum: the N-H bond is watched
        result = self.check(self.state("NNH"), {"h_m0010": 3.0})
        self.assertEqual(result["status"], "changed_connectivity_needs_review")
        self.assertIn(["h_m0010", "n_m0010"], [sorted(pair) for pair in result["lost_bonds"]])

    def test_nitrogen_intermediates_give_the_designed_thermodynamics(self):
        cli("thermo", self.campaign, "--electronic-only")
        thermo = read_json(self.campaign / "analysis" / "thermodynamics.json")
        states = read_json(self.campaign / "analysis" / "states.json")["states"]
        self.assertTrue(all(s["electronic_valid"] for s in states), [(s["state_id"], s["status"]) for s in states])
        row = next(c for c in thermo["cycles"] if c["cycle_id"] == "nrr_distal_m0010" and c["potential_V"] == 0.0)
        self.assertEqual(row["direction"], "cathodic")
        self.assertEqual(sorted(round(x, 6) for x in row["step_deltaG_eV"]), sorted(DG))
        self.assertAlmostEqual(row["limiting_potential_V"], -0.90, places=6)
        self.assertAlmostEqual(row["thermodynamic_overpotential_V"], 0.95, places=6)
        self.assertTrue(thermo["protocol_check"]["consistent"])

    def test_pathways_and_pourbaix_for_nitrogen_chemistry(self):
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
        best = ranked["ranking"]["global_minimum_limiting_potential"]["nrr"]
        self.assertAlmostEqual(best["limiting_potential_V"], -0.90, places=6)
        result = cli(
            "pourbaix", self.campaign, "--electronic-only", "--scale", "RHE", "--at-pH", "0", "--potential=-1.5:0.5"
        )
        # N2 binds downhill at every potential (-0.10 eV), so the bare surface is never the most
        # stable termination; *NH3 (-0.05 eV + 3 U) takes over below -0.05/3 V.
        steps = [(t["label"], round(t["from_V"], 6)) for t in result["transitions"]]
        self.assertEqual(steps, [("m0010:NH3", -1.5), ("m0010:N2", round(-0.05 / 3, 6))])
        self.assertEqual(result["left_out"], [])


if __name__ == "__main__":
    unittest.main()
