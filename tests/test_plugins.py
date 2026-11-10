"""A reaction plugin end to end: examples/plugins/her.py (HER on Pt(111)) through build, VASP
(synthetic), thermo, pathways and Pourbaix, with a new reservoir species and a cathodic reaction.

*H is designed to bind with DeltaG(* -> *H) = +0.09 eV at 0 V vs RHE, so the limiting
potential is -0.09 V, the overpotential 0.09 V, and *H covers the surface below -0.09 V.
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

from mechanet.adsorbates import binds_through_hydrogen
from mechanet.cli import main
from mechanet.fileio import read_json
from mechanet.plugins import default_registry
from mechanet.reactions import SPECIES
from mechanet.review import describe_changes
from mechanet.runner import run_state
from mechanet.structure import Structure, minimum_image, read_poscar, write_poscar
from mechanet.topology import DEFAULT_HYSTERESIS, _limits, _own_release, _pair_type, topology_check

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "examples" / "plugins" / "her.py"
E_H2 = -6.77
DG_H = 0.09


def cli(*args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(["--json", *map(str, args)])
    if code != 0:
        raise AssertionError(f"mechanet {' '.join(map(str, args))} exited {code}: {out.getvalue()}")
    return json.loads(out.getvalue())


@unittest.skipIf(os.environ.get("MECHANET_SKIP_SLOW"), "slow plugin workflow disabled by environment")
class HERPluginTests(unittest.TestCase):
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
        data.update(reaction="her", plugins=[str(PLUGIN)])
        data["scope"]["families"] = ["her_volmer_heyrovsky"]
        data["vasp"] = {
            "potcars": write_potcars(cls.tmp, ("Pt", "H")),
            "magmom_by_element": None,
            "command": [sys.executable, "-I", str(fake)],
        }
        data["frequency"] = {"central": "final"}
        data["thermodynamics"] = {"references": "inputs/references.json", "potentials_V": [0.0, -0.1]}
        (project / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        (project / "inputs" / "references.json").write_text(
            json.dumps({"references": {"H2": {"E": E_H2, "Gcorr": 0.0}}})
        )
        cls.campaign = project / "campaign"
        cls.generated = cli("generate", project / "project.yaml")
        for state in sorted((cls.campaign / "states").iterdir()):
            meta = read_json(state / "state.json")
            n_h = meta["delta_atoms"].get("H", 0)
            target = -100.0 + n_h * (0.5 * E_H2 + DG_H - 0.30)  # minus the tabulated ZPE - TS of one H
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

    def test_plugin_registers_species_reaction_adsorbate_family(self):
        registry = default_registry([PLUGIN])
        self.assertIn("her", registry.reactions)
        self.assertIn("H", registry.adsorbates)
        self.assertEqual(registry.families["her_volmer_heyrovsky"].origin, str(PLUGIN.resolve()))
        self.assertIn("H2", SPECIES)
        self.assertEqual(registry.reaction("her").electrons, -2)
        self.assertEqual(self.generated["complete_stoichiometric_cycles"], 9)
        self.assertEqual(self.generated["nodes_without_viable_start"], [])
        self.assertTrue(cli("validate", self.campaign)["valid"])

    def test_clean_metal_reference_has_no_frequency_region(self):
        ref = self.campaign / "states" / "ref"
        self.assertFalse(np.asarray(read_json(ref / "state.json")["frequency_mobile"]).any())
        (ref / "ctl" / "eshift_04_static").write_text((ref / "ctl" / "eshift_02_final").read_text())
        self.assertEqual(run_state(ref, through="freq", execute=True)["status"], "complete")
        self.assertEqual(read_json(ref / "05_freq" / "frequencies.json")["frequencies_cm1"], [])
        rows = {
            r["state_id"]: r
            for r in cli("collect", self.campaign) and read_json(self.campaign / "analysis" / "states.json")["states"]
        }
        self.assertTrue(rows["ref"]["free_energy_valid"])
        self.assertEqual(rows["ref"]["F_vib_eV"], 0.0)

    def test_cathodic_thermodynamics(self):
        cli("thermo", self.campaign, "--electronic-only")
        thermo = read_json(self.campaign / "analysis" / "thermodynamics.json")
        row = next(
            c for c in thermo["cycles"] if c["cycle_id"] == "her_volmer_heyrovsky_m0010" and c["potential_V"] == 0.0
        )
        self.assertEqual(row["direction"], "cathodic")
        self.assertEqual(
            [round(x, 6) for x in row["step_deltaG_eV"]], [DG_H, -DG_H]
        )  # Volmer, then the releasing Heyrovsky step
        self.assertAlmostEqual(row["limiting_potential_V"], -DG_H, places=6)
        self.assertAlmostEqual(row["thermodynamic_overpotential_V"], DG_H, places=6)

    def test_pathways_and_pourbaix_for_a_cathodic_reaction(self):
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
        best = ranked["ranking"]["global_minimum_limiting_potential"]["her"]
        self.assertAlmostEqual(best["limiting_potential_V"], -DG_H, places=6)
        self.assertAlmostEqual(best["thermodynamic_overpotential_V"], DG_H, places=6)
        result = cli(
            "pourbaix", self.campaign, "--electronic-only", "--scale", "RHE", "--at-pH", "0", "--potential=-0.5:0.5"
        )
        steps = [(t["label"], round(t["from_V"], 6)) for t in result["transitions"]]
        self.assertEqual(steps, [("m0010:H", -0.5), ("native", -DG_H)])

    def test_metal_hydrogen_bonds_are_watched(self):
        """An adsorbate that binds through H makes M-H a watched bond: an H that left its metal
        or went to another one is no longer accepted in silence."""
        registry = default_registry([PLUGIN])
        self.assertTrue(binds_through_hydrogen(registry.adsorbates.values()))
        self.assertFalse(binds_through_hydrogen(default_registry().adsorbates.values()))  # OER, CER: as before
        self.assertIsNone(_pair_type("Pt", "H", {"Pt"}))
        self.assertEqual(_pair_type("Pt", "H", {"Pt"}, metal_h=True), "M-H")
        state = next(s for s in sorted((self.campaign / "states").iterdir()) if s.name != "ref")
        meta = read_json(state / "state.json")
        config = read_json(self.campaign / "config.json")
        start = read_poscar(state / "POSCAR")
        ids = meta["atom_ids"]
        h, own = ids.index("m0010"), ids.index("a0010")
        top = [
            i
            for i, symbol in enumerate(start.symbols)
            if symbol == "Pt" and i != own and abs(start.positions[i, 2] - start.positions[own, 2]) < 0.3
        ]
        neighbour = min(top, key=lambda i: np.linalg.norm(start.positions[i] - start.positions[own]))

        def check(position):
            moved = start.positions.copy()
            moved[h] = position
            path = self.tmp / "moved_h.vasp"
            write_poscar(path, Structure(start.symbols, moved, start.cell, start.mobile, start.ids))
            return topology_check(state, meta, config, geometry=path, registry=registry, use_review=False)

        kept = check(start.positions[h])
        self.assertEqual((kept["status"], kept["valid_for_assigned_node"]), ("connectivity_retained", True))
        gone = check(start.positions[h] + np.array([0.0, 0.0, 3.0]))
        self.assertFalse(gone["valid_for_assigned_node"])
        self.assertEqual(gone["microstate"]["status"], "adsorbate_released")
        self.assertEqual([item["formula"] for item in gone["released"]], ["H"])
        self.assertEqual(describe_changes(gone, meta)[0], "H (H m0010) released from Pt a0010")
        hopped = check(start.positions[neighbour] + np.array([0.0, 0.0, 1.6]))
        self.assertFalse(hopped["valid_for_assigned_node"])
        self.assertEqual(hopped["microstate"]["status"], "metal_attachment_changed")
        self.assertEqual(hopped["released"], [])
        self.assertEqual(len(hopped["lost_active_metal_contacts"]), 1)
        self.assertEqual(len(hopped["new_active_metal_contacts"]), 1)
        # between two metals: it keeps its own metal and gains one, which the top-site model cannot
        # write, so it waits for review (as an O that moves to a bridge does)
        bridge = 0.5 * (start.positions[own] + start.positions[neighbour]) + np.array([0.0, 0.0, 1.15])
        bridged = check(bridge)
        self.assertEqual(bridged["microstate"]["status"], "metal_attachment_changed")
        self.assertEqual(bridged["lost_active_metal_contacts"], [])


@unittest.skipIf(os.environ.get("MECHANET_SKIP_SLOW"), "slow plugin workflow disabled by environment")
class HydrogenPairTests(unittest.TestCase):
    """Two *H on neighbouring Pt atoms, and what they may become: the H-H bond is watched."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env = patch.dict(os.environ, {"MECHANET_CONFIG_DIR": str(cls.tmp / "config")})
        cls.env.start()
        project = cls.tmp / "pt"
        shutil.copytree(ROOT / "examples" / "pt_111", project)
        data = yaml.safe_load((project / "project.yaml").read_text())
        data.update(reaction="her", plugins=[str(PLUGIN)])
        data["scope"]["families"] = ["enumerated"]
        data["scope"]["enumerate"] = {"sites": ["m0010", "m0011"], "states": {"open_metal": ["empty", "H"]}}
        data.pop("thermodynamics", None)
        (project / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        cli("generate", project / "project.yaml")
        cls.campaign = project / "campaign"
        cls.registry = default_registry([PLUGIN])
        cls.config = read_json(cls.campaign / "config.json")
        pair = {"m0010": "H", "m0011": "H"}
        node = next(n for n in read_json(cls.campaign / "network.json")["nodes"] if n.get("occupancy") == pair)
        cls.state = cls.campaign / "states" / node["state_ids"][0]
        cls.meta = read_json(cls.state / "state.json")
        cls.start = read_poscar(cls.state / "POSCAR")
        ids = cls.meta["atom_ids"]
        cls.h = [ids.index("m0010"), ids.index("m0011")]
        metals = [cls.start.positions[ids.index(m)] for m in ("a0010", "a0011")]
        vector = minimum_image(metals[1] - metals[0], cls.start.cell)
        cls.middle, cls.axis = metals[0] + 0.5 * vector, vector / np.linalg.norm(vector)

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        shutil.rmtree(cls.tmp)

    def check(self, separation, height):
        """The two H ``separation`` apart along the Pt-Pt line, ``height`` above the metals."""
        moved = self.start.positions.copy()
        for h, side in zip(self.h, (-0.5, 0.5)):
            moved[h] = self.middle + side * separation * self.axis + np.array([0.0, 0.0, height])
        path = self.tmp / "pair.vasp"
        start = self.start
        write_poscar(path, Structure(start.symbols, moved, start.cell, start.mobile, start.ids))
        return topology_check(
            self.state, self.meta, self.config, geometry=path, registry=self.registry, use_review=False
        )

    def test_the_bond_is_a_kind_only_with_hydrogen_sites(self):
        self.assertIsNone(_pair_type("H", "H", {"Pt"}))  # OER, CER: as before
        self.assertEqual(_pair_type("H", "H", {"Pt"}, metal_h=True), "H-H")
        self.assertEqual(_limits("H-H", DEFAULT_HYSTERESIS), (1.00, 1.15, 1.30))
        self.assertEqual(_limits("H-H", {**DEFAULT_HYSTERESIS, "H-H": (0.8, 1.2)}), (0.8, 1.0, 1.2))

    def test_molecular_hydrogen_on_the_surface_is_not_two_adsorbed_h(self):
        pair = ["m0010", "m0011"]
        kept = self.check(2.77, 1.6)
        self.assertEqual((kept["status"], kept["valid_for_assigned_node"]), ("connectivity_retained", True))
        # H2 lying over the bridge, each H still within reach of its own metal (2.14 A): it used
        # to pass as two *H with a remark about the distances
        molecule = self.check(0.80, 1.9)
        self.assertFalse(molecule["valid_for_assigned_node"])
        self.assertEqual(molecule["microstate"]["status"], "outside_site_model")
        self.assertIn("molecular hydrogen", molecule["microstate"]["reason"])
        self.assertEqual([sorted(p) for p in molecule["new_bonds"]], [pair])
        self.assertEqual((molecule["released"], molecule["lost_active_metal_contacts"]), ([], []))
        lines = describe_changes(molecule, self.meta)
        self.assertEqual([line for line in lines if "now bonded" in line], ["H m0010 is now bonded to H m0011"])
        # in between (1.0 to 1.3 A) no bond is claimed either way: reported, as for any other bond
        stretched = self.check(1.20, 1.6)
        self.assertTrue(stretched["valid_for_assigned_node"])
        self.assertIn(("H-H", pair), [(a["type"], sorted(a["pair"])) for a in stretched["threshold_ambiguities"]])
        # and H2 that left is the intermediate's own adsorbate, released
        gone = self.check(0.75, 4.5)
        self.assertEqual(gone["microstate"]["status"], "adsorbate_released")
        self.assertEqual([(item["formula"], item["atoms"]) for item in gone["released"]], [("H₂", pair)])


class OwnReleaseTests(unittest.TestCase):
    """Which released molecules are the intermediate's own adsorbate leaving (no bound minimum)."""

    SYMBOLS = {"m0010": "H", "m0012": "H", "m0011": "O", "h_m0011": "H", "a0010": "Pt", "a0099": "O", "h_w": "H"}
    SYMBOLS["o_x"] = "O"
    NATIVE = {"m0010": "empty", "m0011": "empty", "m0012": "empty"}

    def own(self, atoms, occupancy, ligand="m0010"):
        released = [{"formula": "", "atoms": atoms, "from": ["a0010"]}]
        return _own_release(released, [("a0010", ligand)], [], occupancy, None, self.NATIVE, self.SYMBOLS)

    def test_hydrogen_alone_must_be_the_sites_own(self):
        self.assertTrue(self.own(["m0010"], {"m0010": "H"}))
        self.assertTrue(self.own(["m0010", "m0012"], {"m0010": "H", "m0012": "H"}))
        self.assertFalse(self.own(["m0010"], {"m0011": "OH"}))  # the H of a site the intermediate leaves alone
        # H2 made of the site's H and the H of a neighbouring OH: that ligand changed too
        self.assertFalse(self.own(["h_m0011", "m0010"], {"m0010": "H", "m0011": "OH"}))

    def test_a_molecule_with_heavy_atoms_needs_one_of_its_own(self):
        # the site's H left with a spectator water of the slab: not its own
        self.assertFalse(self.own(["a0099", "h_w", "m0010"], {"m0010": "H"}))
        # as before: the O of a changed site, or an atom the intermediate added
        self.assertTrue(self.own(["m0011", "h_m0011"], {"m0011": "OH"}, ligand="m0011"))
        self.assertTrue(self.own(["o_x", "h_w"], {"m0010": "H"}, ligand="o_x"))
        self.assertFalse(self.own(["a0099", "h_w"], {"m0010": "H"}, ligand="a0099"))


if __name__ == "__main__":
    unittest.main()
