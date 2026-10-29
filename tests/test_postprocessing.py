"""Synthetic native-like fixtures; these are software tests, not DFT data."""

import contextlib
import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import support  # noqa: E402,F401 - puts src/ on sys.path

from mechanet.analysis import analyze_thermodynamics, collect, select_for_thermo, topology_check  # noqa: E402
from mechanet.cli import main  # noqa: E402
from mechanet.reactions import OER  # noqa: E402
from mechanet.results import parse_run  # noqa: E402
from mechanet.thermo import (
    cycle_metrics,
    cyclic_span,
    edge_free_energy,
    exner_gmax,  # noqa: E402
    formation_free_energy,
    reservoir_chemical_potentials,
    validate_network,
    vibrational_thermochemistry,
)

STAGES = ("01_loose", "02_final", "03_refine", "04_static", "05_freq")


def oer_cycle_metrics(edges, delta_g_at_0, potential_V, equilibrium_potential_V=1.23):
    """cycle_metrics for OER edges given by their (water, pairs, oxygen) counts: cycle_metrics
    takes reservoir vectors, the electrons of each edge and the equilibrium potential."""
    vectors = [
        {
            "reservoirs": {
                k: v
                for k, v in (("H2O", -e.get("water", 0)), ("H+", e.get("pairs", 0)), ("O2", e.get("oxygen", 0)))
                if v
            }
        }
        for e in edges
    ]
    electrons = [e.get("pairs", 0) for e in edges]
    return cycle_metrics(vectors, delta_g_at_0, potential_V, electrons, equilibrium_potential_V)


def write_sites(campaign, sites):
    """The reactive sites of a campaign, as sites.json."""
    (Path(campaign) / "sites.json").write_text(
        json.dumps(
            {"schema_version": 1, "sites": [{"id": aid, "kind": kind, "native": native} for aid, kind, native in sites]}
        )
    )


POSCAR = """synthetic O-H fixture
1.0
12 0 0
0 12 0
0 0 12
O H
1 1
Selective dynamics
Cartesian
2 2 2 F F F
2 2 3 T T T
"""


def outcar_text(n_evaluations=1, fail_scf=None, frequencies=None, footer=True, ionic=False, full_vectors=False):
    lines = ["NIONS = 2", "NELM = 4; EDIFF = 1.0E-07"]
    for i in range(n_evaluations):
        lines += [f"Iteration {i + 1}( 1)", f"Iteration {i + 1}( 2)"]
        if i != fail_scf:
            lines.append("aborting loop because EDIFF is reached")
        lines += [
            "POSITION TOTAL-FORCE (eV/Angst)",
            "------------",
            "2.0 2.0 2.0 0.001 0.002 0.003",
            "2.0 2.0 3.0 0.001 0.002 0.003",
            "------------",
            f"free  energy   TOTEN = {-10.1 + i * 0.001:.8f} eV",
            f"energy without entropy= -10.000000 energy(sigma->0) = {-10.0 + i * 0.001:.8f}",
        ]
    lines += [
        "magnetization (x)",
        "# of ion s p d tot",
        "---------",
        "1 0.0 0.1 0.0 0.1",
        "2 0.0 0.2 0.0 0.2",
        "---------",
        "tot 0 0.3 0 0.3",
    ]
    for i, freq in enumerate(frequencies or []):
        tag = "f/i" if freq < 0 else "f"
        lines.append(f"{i + 1} {tag} = 1.0 THz 2.0 2PiTHz {abs(freq):.8f} cm-1 3.0 meV")
        if full_vectors:
            vectors = [[0.0] * 3 for _ in range(2)]
            vectors[i // 3][i % 3] = 1.0
            lines += [
                "X Y Z dx dy dz",
                "2 2 2 " + " ".join(map(str, vectors[0])),
                "2 2 3 " + " ".join(map(str, vectors[1])),
            ]
    if ionic:
        lines.append("reached required accuracy - stopping structural energy minimisation")
    if footer:
        lines.append("General timing and accounting informations for this job:")
        lines.append("Total CPU time used (sec): 12.34")
        lines.append("Elapsed time (sec): 10.1")
    return "\n".join(lines) + "\n"


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        (self.path / "POSCAR").write_text(POSCAR)
        (self.path / "INCAR").write_text("IBRION=2; NELM=4; EDIFF=1D-7; EDIFFG=-0.03\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_static_outputs_and_moments(self):
        (self.path / "OUTCAR").write_text(outcar_text(ionic=True))
        result = parse_run(self.path)
        self.assertTrue(result["completed"])
        self.assertTrue(result["electronic_converged"])
        self.assertTrue(result["ionic_converged"])
        self.assertEqual(result["energy_e0_eV"], -10.0)
        self.assertEqual(result["magmoms"], [0.1, 0.2])
        self.assertAlmostEqual(result["max_mobile_force_eVA"], math.sqrt(14) * 0.001)

    def test_partial_footer_does_not_establish_convergence(self):
        (self.path / "OUTCAR").write_text(outcar_text(fail_scf=0))
        result = parse_run(self.path)
        self.assertTrue(result["completed"])
        self.assertFalse(result["electronic_converged"])
        (self.path / "OUTCAR").write_text(outcar_text(footer=False))
        self.assertFalse(parse_run(self.path)["completed"])

    def test_final_recovery_is_separate_from_all_scf_convergence(self):
        (self.path / "OUTCAR").write_text(outcar_text(n_evaluations=2, fail_scf=0, ionic=True))
        result = parse_run(self.path)
        self.assertFalse(result["electronic_converged"])
        self.assertTrue(result["last_electronic_converged"])

    def test_finite_difference_coverage_and_signed_imaginary_modes(self):
        (self.path / "INCAR").write_text("IBRION=5; NFREE=2; NELM=4; EDIFF=1e-7\n")
        (self.path / "OUTCAR").write_text(outcar_text(n_evaluations=7, frequencies=[3000, 600, -25]))
        result = parse_run(self.path)
        self.assertTrue(result["frequency_converged"])
        self.assertEqual(result["frequencies_cm1"], [3000, 600, -25])
        self.assertEqual(result["expected_frequency_modes"], 3)
        self.assertIn("DO_NOT_USE_FOR_CHE", result["energy_context"])
        (self.path / "OUTCAR").write_text(outcar_text(n_evaluations=2, frequencies=[3000, 600, 25]))
        self.assertFalse(parse_run(self.path)["frequency_converged"])

    def test_full_hessian_frozen_zeros_need_eigenvectors(self):
        (self.path / "INCAR").write_text("IBRION=5; NFREE=2; NELM=4; EDIFF=1e-7\n")
        values = [0, 0, 0, 100, 500, 3000]
        (self.path / "OUTCAR").write_text(outcar_text(7, frequencies=values, full_vectors=True))
        result = parse_run(self.path)
        self.assertTrue(result["frequency_converged"])
        self.assertEqual(result["frequencies_cm1"], [100, 500, 3000])
        self.assertEqual(result["excluded_frozen_mode_indices"], [1, 2, 3])
        (self.path / "OUTCAR").write_text(outcar_text(7, frequencies=values, full_vectors=False))
        self.assertFalse(parse_run(self.path)["frequency_converged"])

    def test_stale_input_hash_invalidates_completed_output(self):
        (self.path / "OUTCAR").write_text(outcar_text(ionic=True))
        manifest = {"files": {"INCAR": hashlib.sha256((self.path / "INCAR").read_bytes()).hexdigest()}}
        (self.path / "input_hashes.json").write_text(json.dumps(manifest))
        self.assertTrue(parse_run(self.path)["input_hashes_valid"])
        with (self.path / "INCAR").open("a") as stream:
            stream.write("ENCUT=500\n")
        result = parse_run(self.path)
        self.assertFalse(result["input_hashes_valid"])
        self.assertFalse(result["completed"])

    def test_xml_nelm_exhaustion_rejected(self):
        calculations = "<scstep><energy><i name='e_fr_energy'>-1</i><i name='e_0_energy'>-0.9</i></energy></scstep>" * 4
        (self.path / "vasprun.xml").write_text(
            "<modeling><parameters><i name='NELM'>4</i><i name='EDIFF'>1e-7</i></parameters>"
            "<calculation>"
            + calculations
            + "<energy><i name='e_0_energy'>-10</i><i name='e_fr_energy'>-10.1</i></energy>"
            "</calculation></modeling>"
        )
        result = parse_run(self.path)
        self.assertTrue(result["completed"])
        self.assertFalse(result["electronic_converged"])
        self.assertEqual(result["energy_e0_eV"], -10)

    def test_xml_energy_keeps_dispersion_and_reconstructs_entropy_correction(self):
        (self.path / "vasprun.xml").write_text(
            "<modeling><parameters><i name='NELM'>4</i><i name='EDIFF'>1e-7</i></parameters>"
            "<calculation><scstep><energy><i name='e_fr_energy'>-10.0</i>"
            "<i name='e_0_energy'>-9.9</i></energy></scstep>"
            "<energy><i name='e_0_energy'>999</i><i name='e_fr_energy'>-10.5</i></energy>"
            "</calculation></modeling>"
        )
        result = parse_run(self.path)
        self.assertAlmostEqual(result["energy_e0_eV"], -10.4)
        self.assertAlmostEqual(result["energy_free_eV"], -10.5)


class ThermochemistryTests(unittest.TestCase):
    def test_harmonic_and_frequency_floor(self):
        raw = vibrational_thermochemistry([10, 1000], low_frequency_cutoff_cm1=0)
        floor = vibrational_thermochemistry([10, 1000], low_frequency_cutoff_cm1=100)
        self.assertTrue(raw["valid"])
        self.assertTrue(floor["valid"])
        self.assertAlmostEqual(raw["F_vib_eV"], floor["raw"]["F_vib_eV"])
        self.assertGreater(floor["floor_sensitivity_eV"], 0)

    def test_imaginary_modes_are_never_silently_deleted(self):
        strict = vibrational_thermochemistry([-10, 1000])
        explicit = vibrational_thermochemistry([-10, 1000], imaginary_tolerance_cm1=20)
        self.assertFalse(strict["valid"])
        self.assertIsNone(strict["F_vib_eV"])
        self.assertTrue(explicit["valid"])
        self.assertEqual(explicit["n_modes"], 2)
        self.assertEqual(explicit["regularized_modes"][0]["original_cm1"], -10)

    def test_user_reference_format(self):
        refs = {
            "references": {"H2": {"E": -6.77398545, "Gcorr": -0.121865}, "H2O": {"E": -14.24029443, "Gcorr": -0.017671}}
        }
        result = reservoir_chemical_potentials(refs)
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["mu_H2_eV"], -6.89585045)
        self.assertAlmostEqual(result["mu_H2O_eV"], -14.25796543)
        self.assertAlmostEqual(result["mu_O2_eV"], -9.80422996)
        self.assertTrue(result["warnings"])
        self.assertFalse(reservoir_chemical_potentials({})["valid"])
        check = result["water_standard_state_check"]
        self.assertAlmostEqual(check["Gcorr_H2O_minus_Gcorr_H2_eV"], 0.104194)
        self.assertEqual(check["flag"], "molecular_corrections_inconsistent")
        self.assertEqual(result["H2O_pressure_correction_eV"], 0.0)
        # The supplied H2 value implies a ~3100 cm^-1 stretch (one H held fixed gives ~3060; PBE ~4330).
        self.assertAlmostEqual(check["implied_H2_stretch_cm1"]["ZPE_plus_Cp_minus_TS"], 3098, delta=10)

    def test_declared_water_pressure_does_not_hide_an_h2_problem(self):
        refs = {
            "references": {
                "H2": {"E": -6.77398545, "Gcorr": -0.121865},
                "H2O": {"E": -14.24029443, "Gcorr": -0.017671, "Gcorr_pressure_bar": 0.031},
            }
        }
        result = reservoir_chemical_potentials(refs)
        shift = 8.617333262145e-5 * 298.15 * math.log(0.035 / 0.031)
        self.assertAlmostEqual(result["H2O_pressure_correction_eV"], shift)
        check = result["water_standard_state_check"]
        self.assertEqual(check["flag"], "molecular_corrections_inconsistent")
        self.assertIn("every atom free", check["advice"])
        # Consistent corrections (H2 ~ -0.045 eV at 1 bar, H2O ~ 0 eV at 0.031 bar) pass.
        refs["references"]["H2"]["Gcorr"] = -0.045
        refs["references"]["H2O"]["Gcorr"] = -0.0025
        self.assertIsNone(reservoir_chemical_potentials(refs)["water_standard_state_check"]["flag"])

    def test_declared_one_bar_water_is_moved_to_liquid_equilibrium(self):
        refs = {
            "references": {
                "H2": {"E": -6.77398545, "Gcorr": -0.121865},
                "H2O": {"E": -14.24029443, "Gcorr": -0.017671, "Gcorr_pressure_bar": 1.0},
            }
        }
        result = reservoir_chemical_potentials(refs)
        shift = 8.617333262145e-5 * 298.15 * math.log(0.035)
        self.assertAlmostEqual(result["H2O_pressure_correction_eV"], shift)
        self.assertAlmostEqual(result["mu_H2O_eV"], -14.25796543 + shift)
        self.assertAlmostEqual(shift, -0.0861, places=4)
        self.assertIsNone(result["water_standard_state_check"]["flag"])
        # The 4.92 eV closure moves O2 with water: water steps -0.086 eV, O2 release +0.172 eV.
        base = reservoir_chemical_potentials(
            {"references": {"H2": refs["references"]["H2"], "H2O": {"E": -14.24029443, "Gcorr": -0.017671}}}
        )
        self.assertAlmostEqual(result["mu_O2_eV"] - base["mu_O2_eV"], 2 * shift)

    @staticmethod
    def network():
        nodes = [
            {"id": name, "delta_atoms": {"O": o, "H": h}}
            for name, o, h in [("empty", 0, 0), ("OH", 1, 1), ("O", 1, 0), ("OOH", 2, 1)]
        ]
        edges = [
            {"id": name, "from": left, "to": right, "water": w, "pairs": p, "oxygen": g}
            for name, left, right, w, p, g in [
                ("a", "empty", "OH", 1, 1, 0),
                ("b", "OH", "O", 0, 1, 0),
                ("c", "O", "OOH", 1, 1, 0),
                ("d", "OOH", "empty", 0, 1, 1),
            ]
        ]
        return {"nodes": nodes, "edges": edges, "cycles": [{"id": "AEM", "edge_ids": ["a", "b", "c", "d"]}]}

    def test_balanced_che_cycle_closes_at_every_potential(self):
        network = self.network()
        self.assertTrue(validate_network(network)["valid"])
        refs = reservoir_chemical_potentials(
            {
                "references": {
                    "H2": {"E": -6.77398545, "Gcorr": -0.121865},
                    "H2O": {"E": -14.24029443, "Gcorr": -0.017671},
                }
            }
        )
        energies = {"empty": -1000, "OH": -1010, "O": -1006, "OOH": -1014}
        dgs = [edge_free_energy(energies[e["from"]], energies[e["to"]], e, refs, 0) for e in network["edges"]]
        for potential in (0, 1.23, 1.53, 2.0):
            result = oer_cycle_metrics(network["edges"], dgs, potential)
            self.assertAlmostEqual(result["net_deltaG_eV"], 4.92 - 4 * potential)
        network["edges"][2]["water"] = 0
        self.assertFalse(validate_network(network)["valid"])

    def test_gmax_cyclic_span_and_resting_state(self):
        # Exner Gmax(U): largest G_j - G_i over i < j, excluding first -> last.
        profile = [0.0, 0.4, 0.2, 1.1, -0.3]
        value, pair = exner_gmax(profile)
        self.assertAlmostEqual(value, 1.1)
        self.assertEqual(pair, (0, 3))
        span, span_pair = cyclic_span(profile)
        self.assertAlmostEqual(span, 1.1)
        # A later intermediate lower than the start makes the cyclic span differ.
        profile = [0.0, -0.5, 0.6, 0.3, -0.2]
        self.assertAlmostEqual(exner_gmax(profile)[0], 1.1)
        self.assertAlmostEqual(cyclic_span(profile)[0], 1.1)
        metrics = oer_cycle_metrics([{"pairs": 1}] * 4, [-0.5 + 1.0, 1.1 + 1.0, -0.3 + 1.0, -0.5 + 1.0], 1.0)
        self.assertEqual(metrics["resting_state_index"], 1)
        self.assertAlmostEqual(metrics["largest_step_deltaG_eV"], 1.1)
        self.assertAlmostEqual(metrics["thermodynamic_overpotential_V"], 2.1 - 1.23)
        self.assertNotIn("deltaG_max_eV", metrics)

    def test_cycles_are_evaluated_with_o2_release_last(self):
        # A pair-coupling-shaped cycle written O2-first: O/O -> (release) -> empty -> ... -> O/O
        edges = [{"pairs": 1, "oxygen": 1, "water": 0}] + [{"pairs": 1, "oxygen": 0, "water": w} for w in (1, 0, 1, 0)]
        dg0 = [1.0, 1.0, 1.0, 1.0, 0.92]
        metrics = oer_cycle_metrics(edges, dg0, 1.53)
        self.assertEqual(metrics["rotation"], 1)
        rotated = oer_cycle_metrics(edges[1:] + edges[:1], dg0[1:] + dg0[:1], 1.53)
        self.assertEqual(rotated["rotation"], 0)
        for key in ("Gmax_U_eV", "cyclic_span_eV", "largest_step_deltaG_eV", "resting_state_index"):
            self.assertAlmostEqual(metrics[key], rotated[key])
        # The resting state of the rotated cycle is its lowest formation energy (the O/O state, index 4).
        self.assertEqual(metrics["resting_state_index"], 4)
        self.assertAlmostEqual(
            oer_cycle_metrics(edges, dg0, 1.53, equilibrium_potential_V=1.22)["thermodynamic_overpotential_V"],
            metrics["limiting_potential_V"] - 1.22,
        )

    def test_formation_free_energy(self):
        refs = {"mu_H2O_eV": -14.0, "mu_H2_eV": -7.0}
        # mechanet takes the reaction's formation vector instead of delta_atoms (here {H2O: -1, H+: 1}).
        vector = OER().formation_vector({"O": 1, "H": 1})
        self.assertEqual(vector, {"H2O": -1, "H+": 1})
        # ref + H2O -> OH* + (H+ + e-): G(OH*) - G(ref) - mu_H2O + (1/2 mu_H2 - U)
        self.assertAlmostEqual(formation_free_energy(-20.0, -10.0, vector, refs, 1.0), -20.0 + 10.0 + 14.0 - 3.5 - 1.0)
        self.assertAlmostEqual(
            formation_free_energy(-10.0, -10.0, OER().formation_vector({"O": 0, "H": 0}), refs, 1.0), 0.0
        )

    def test_potential_independent_step_cannot_be_removed_by_bias(self):
        metrics = oer_cycle_metrics([{"pairs": 0}, {"pairs": 1}], [0.2, 1.8], 2.0)
        self.assertIsNone(metrics["limiting_potential_V"])
        self.assertTrue(metrics["potential_independent_uphill_step"])


class CollectionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.campaign = Path(self.tmp.name)
        self.state = self.campaign / "states" / "reference"
        self.state.mkdir(parents=True)
        (self.campaign / "config.json").write_text(
            json.dumps({"analysis": {"require_input_hashes": False}, "thermodynamics": {"potentials_V": [1.23, 1.53]}})
        )
        (self.state / "POSCAR").write_text(POSCAR)
        (self.state / "state.json").write_text(
            json.dumps(
                {
                    "id": "reference",
                    "node_id": "reference",
                    "family": "reference",
                    "site": "whole_slab",
                    "variant": 0,
                    "atom_ids": ["a0001", "a0002"],
                    "symbols": ["O", "H"],
                    "relax_mobile": [[False] * 3, [True] * 3],
                    "frequency_mobile": [[False] * 3, [True] * 3],
                    "delta_atoms": {"O": 0, "H": 0},
                    "expected_bonds": [["a0001", "a0002"]],
                }
            )
        )
        for stage in STAGES:
            directory = self.state / stage
            directory.mkdir()
            (directory / "POSCAR").write_text(POSCAR)
            (directory / "CONTCAR").write_text(POSCAR)
            ibrion = 5 if stage == "05_freq" else -1 if stage == "04_static" else 2
            ediffg = -0.015 if stage == "03_refine" else -0.03
            (directory / "INCAR").write_text(
                f"IBRION={ibrion}; NELM=4; EDIFF=1e-7; EDIFFG={ediffg}; NFREE=2; ISMEAR=0\n"
            )
            (directory / "OUTCAR").write_text(
                outcar_text(
                    n_evaluations=7 if stage == "05_freq" else 1,
                    frequencies=[100, 500, 3000] if stage == "05_freq" else None,
                    ionic=stage in ("01_loose", "02_final", "03_refine"),
                )
            )

    def tearDown(self):
        self.tmp.cleanup()

    def test_central_energy_and_partial_hessian_are_combined(self):
        result = collect(self.campaign)
        state = result["states"][0]
        self.assertTrue(state["free_energy_valid"], state)
        self.assertEqual(state["energy_e0_eV"], -10.0)
        self.assertNotEqual(state["energy_e0_eV"], state["stages"]["05_freq"]["energy_e0_eV"])
        self.assertAlmostEqual(state["G_eV"], -10.0 + state["F_vib_eV"])
        self.assertTrue(state["final_valid"])
        self.assertTrue((self.campaign / "analysis" / "magnetic_moments.csv").exists())
        self.assertTrue(any("SIGMA is not set" in n for n in result["summary"]["protocol_notes"]))

    def test_small_imaginary_mode_is_floored_by_default_large_one_is_not(self):
        directory = self.state / "05_freq"
        (directory / "OUTCAR").write_text(outcar_text(n_evaluations=7, frequencies=[-60, 500, 3000]))
        state = collect(self.campaign)["states"][0]
        self.assertTrue(state["free_energy_valid"], state["frequency_exclusion_reasons"])
        self.assertEqual(state["vibrational"]["regularized_modes"][0]["original_cm1"], -60)
        (directory / "OUTCAR").write_text(outcar_text(n_evaluations=7, frequencies=[-150, 500, 3000]))
        state = collect(self.campaign)["states"][0]
        self.assertFalse(state["free_energy_valid"])

    def _hessian_json(self, soft_axis):
        """Stored displacement Hessian for the O-H fixture: H active, O frozen, O-H bond along z."""
        hessian = [[30.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]]
        axis = "xyz".index(soft_axis)
        hessian[axis][axis] = -((60.0 / 521.4709) ** 2) * 1.008  # a -60 cm^-1 mode
        units = [
            {"id": f"u{2 * k + s}", "dof": k, "atom_index_1based": 2, "axis": "xyz"[k]}
            for k in range(3)
            for s in (0, 1)
        ]
        (self.state / "05_freq" / "frequencies.json").write_text(
            json.dumps({"hessian_eV_A2": hessian, "masses_amu": {"O": 15.999, "H": 1.008}, "units": units})
        )

    def test_imaginary_mode_character_from_stored_hessian(self):
        from mechanet.analysis import imaginary_mode_character

        self._hessian_json("x")
        mode = imaginary_mode_character(self.state / "05_freq")[0]
        self.assertAlmostEqual(mode["cm1"], -60.0, places=3)
        self.assertEqual(mode["character"], "H_libration_or_rotation")
        self.assertEqual(mode["OH_stretch_fraction"], 0.0)
        self._hessian_json("z")
        self.assertEqual(imaginary_mode_character(self.state / "05_freq")[0]["character"], "proton_transfer_like")

    def test_small_proton_transfer_like_imaginary_mode_is_not_regularized(self):
        directory = self.state / "05_freq"
        (directory / "OUTCAR").write_text(outcar_text(n_evaluations=7, frequencies=[-60, 500, 3000]))
        self._hessian_json("x")
        state = collect(self.campaign)["states"][0]
        self.assertTrue(state["free_energy_valid"], state["frequency_exclusion_reasons"])
        self.assertEqual(state["imaginary_mode_character"][0]["character"], "H_libration_or_rotation")
        self.assertAlmostEqual(state["vibrational"]["floored_modes_F_vib_eV"], -0.0185, places=3)
        self._hessian_json("z")
        state = collect(self.campaign)["states"][0]
        self.assertFalse(state["free_energy_valid"])
        self.assertTrue(any("proton-transfer-like" in r for r in state["frequency_exclusion_reasons"]))

    def test_smearing_flag_is_per_cell(self):
        from mechanet.analysis import _smearing_warnings

        stages = {
            "02_final": {"eentro_eV": -0.05},
            "03_refine": {"eentro_eV": -0.001},
            "04_static": {"eentro_eV": None},
        }
        # The limit is per cell (analysis.max_eentro_eV), whatever the number of atoms.
        warnings = _smearing_warnings(stages)  # 0.05 eV per cell, 0.56 meV/atom
        self.assertEqual(len(warnings), 1)
        self.assertIn("0.050 eV per cell", warnings[0])

    def test_refine_must_converge_for_static_energy(self):
        (self.state / "03_refine" / "OUTCAR").write_text(outcar_text(ionic=False))
        state = collect(self.campaign)["states"][0]
        self.assertTrue(state["final_valid"])
        self.assertFalse(state["electronic_valid"])
        self.assertIn("03_refine: ionic convergence unresolved", state["electronic_exclusion_reasons"])

    def test_changed_product_is_excluded(self):
        moved = POSCAR.replace("2 2 3 T T T", "2 2 4 T T T")
        for path in [
            self.state / "02_final" / "CONTCAR",
            self.state / "03_refine" / "CONTCAR",
            self.state / "04_static" / "POSCAR",
            self.state / "05_freq" / "POSCAR",
        ]:
            path.write_text(moved)
        state = collect(self.campaign)["states"][0]
        self.assertFalse(state["electronic_valid"])
        self.assertEqual(state["topology"]["status"], "changed_connectivity_needs_review")

    def test_oxygen_migration_between_metals_requires_reclassification(self):
        initial = """metal migration fixture
1.0
20 0 0
0 20 0
0 0 20
Fe Ni O
1 1 1
Selective dynamics
Cartesian
2 2 2 F F F
8 2 2 F F F
2 2 3.8 T T T
"""
        final = initial.replace("2 2 3.8 T T T", "8 2 3.8 T T T")
        (self.state / "POSCAR").write_text(initial)
        (self.state / "04_static" / "POSCAR").write_text(final)
        meta = {
            "atom_ids": ["Fe1", "Ni1", "O1"],
            "frequency_mobile": [[False] * 3, [False] * 3, [True] * 3],
            "expected_bonds": [],
        }
        result = topology_check(self.state, meta)
        self.assertFalse(result["valid_for_assigned_node"])
        # *_active_metal_contacts track M-O and M-Cl contacts.
        self.assertEqual(result["lost_active_metal_contacts"], [("Fe1", "O1")])
        self.assertEqual(result["new_active_metal_contacts"], [("Ni1", "O1")])

    def test_shared_proton_transfer_is_a_different_tautomer(self):
        initial = """shared proton fixture
1.0
20 0 0
0 20 0
0 0 20
O H
2 1
Selective dynamics
Cartesian
2 2 2 T T T
2 2 4.47 T T T
2 2 3.09 T T T
"""
        meta = {
            "atom_ids": ["a0065", "a0068", "a0085"],
            "frequency_mobile": [[True] * 3] * 3,
            "expected_bonds": [["a0065", "a0085"]],
        }
        (self.state / "POSCAR").write_text(initial)
        # Moving inside the hysteresis band (1.09/1.38 -> 1.17/1.30 A) is a soft change only.
        (self.state / "04_static" / "POSCAR").write_text(initial.replace("2 2 3.09 T T T", "2 2 3.17 T T T"))
        result = topology_check(self.state, meta)
        self.assertTrue(result["valid_for_assigned_node"], result)
        self.assertEqual(result["shared_proton_transfers"], [])
        # A full transfer to the other oxygen is a different tautomer: flagged for review.
        (self.state / "04_static" / "POSCAR").write_text(initial.replace("2 2 3.09 T T T", "2 2 3.40 T T T"))
        result = topology_check(self.state, meta)
        self.assertFalse(result["valid_for_assigned_node"])
        self.assertEqual(result["shared_proton_transfers"][0]["to"], "a0068")
        # A native ID the builder reused for a new atom is not treated as a shared proton.
        reused = dict(meta, variant={"repositioned_native_ids": ["a0085"]})
        result = topology_check(self.state, reused)
        self.assertFalse(result["valid_for_assigned_node"])
        self.assertEqual(result["shared_proton_transfers"], [])

    def test_collect_excludes_attempts_the_runner_rejected(self):
        state = collect(self.campaign)["states"][0]
        self.assertTrue(state["electronic_valid"])
        (self.state / "04_static" / "attempt.json").write_text(json.dumps({"status": "failed", "returncode": 0}))
        state = collect(self.campaign)["states"][0]
        self.assertFalse(state["electronic_valid"])
        self.assertTrue(any("not accepted" in r for r in state["electronic_exclusion_reasons"]))
        (self.state / "04_static" / "attempt.json").unlink()
        text = (self.state / "04_static" / "OUTCAR").read_text()
        (self.state / "04_static" / "INCAR").write_text("IBRION=-1; NELM=4; EDIFF=1e-7; ISTART=0; ICHARG=1; ISMEAR=0\n")
        (self.state / "04_static" / "OUTCAR").write_text(
            " initial charge density was supplied:\n charge density of overlapping atoms calculated\n" + text
        )
        state = collect(self.campaign)["states"][0]
        self.assertFalse(state["electronic_valid"])
        self.assertTrue(any("fallback" in r for r in state["electronic_exclusion_reasons"]))

    def test_state_prepared_for_the_final_protocol_has_no_refine_energy(self):
        (self.state / "plan.json").write_text(json.dumps({"central": "final", "stages": {}}))
        state = collect(self.campaign)["states"][0]
        self.assertTrue(state["final_valid"])
        self.assertFalse(state["electronic_valid"])
        self.assertTrue(
            any("prepared with frequency.central = 'final'" in r for r in state["electronic_exclusion_reasons"])
        )

    def test_hysteresis_ignores_small_crossings(self):
        initial = POSCAR.replace("2 2 3 T T T", "2 2 3.22 T T T")
        crossing = POSCAR.replace("2 2 3 T T T", "2 2 3.28 T T T")
        (self.state / "POSCAR").write_text(initial)
        (self.state / "04_static" / "POSCAR").write_text(crossing)
        meta = {"atom_ids": ["a0001", "a0002"], "frequency_mobile": [[False] * 3, [True] * 3], "expected_bonds": []}
        result = topology_check(self.state, meta)
        self.assertTrue(result["valid_for_assigned_node"], result)
        self.assertEqual(len(result["soft_changes_within_hysteresis"]), 1)

    def test_no_outputs_produce_unresolved_not_zero_energies(self):
        for stage in STAGES:
            (self.state / stage / "OUTCAR").unlink()
        state = collect(self.campaign)["states"][0]
        self.assertEqual(state["status"], "not_run")
        self.assertIsNone(state["G_eV"])

    def test_missing_references_leave_graph_unresolved(self):
        network = {
            "nodes": [{"id": "reference", "state_ids": ["reference"], "delta_atoms": {"O": 0, "H": 0}}],
            "edges": [{"id": "overall", "from": "reference", "to": "reference", "water": 2, "pairs": 4, "oxygen": 1}],
            "cycles": [{"id": "overall_test_only", "edge_ids": ["overall"]}],
        }
        (self.campaign / "network.json").write_text(json.dumps(network))
        (self.campaign / "references.json").write_text("{}")
        result = analyze_thermodynamics(self.campaign, self.campaign / "references.json")
        self.assertFalse(result["reservoirs"]["valid"])
        self.assertTrue(all(e["deltaG_eV"] is None for e in result["edges"]))
        self.assertTrue(all(c["status"] == "unresolved" for c in result["cycles"]))

    def _network_with_twin(self):
        network = {
            "reference": "reference",
            "nodes": [
                {"id": "reference", "state_ids": ["reference"], "delta_atoms": {"O": 0, "H": 0}},
                {"id": "twin", "state_ids": [], "computed_by": "reference", "delta_atoms": {"O": 0, "H": 0}},
            ],
            "edges": [
                {"id": "e1", "from": "reference", "to": "twin", "water": 0, "pairs": 0, "oxygen": 0},
                {"id": "e2", "from": "twin", "to": "reference", "water": 2, "pairs": 4, "oxygen": 1},
            ],
            "cycles": [{"id": "toy", "edge_ids": ["e1", "e2"]}],
        }
        (self.campaign / "network.json").write_text(json.dumps(network))
        (self.campaign / "references.json").write_text(
            json.dumps(
                {
                    "references": {
                        "H2": {"E": -6.77398545, "Gcorr": -0.121865},
                        "H2O": {"E": -14.24029443, "Gcorr": -0.017671},
                    }
                }
            )
        )

    def test_twin_node_takes_representative_energy_and_resting_state_is_reported(self):
        self._network_with_twin()
        result = analyze_thermodynamics(self.campaign, self.campaign / "references.json")
        nodes = {n["node_id"]: n for n in result["nodes"]}
        self.assertEqual(nodes["twin"]["status"], "available_by_symmetry")
        self.assertEqual(nodes["twin"]["node_energy_eV"], nodes["reference"]["node_energy_eV"])
        # Rows carry potential_V and its scale.
        cycle = [c for c in result["cycles"] if c["potential_V"] == 1.23][0]
        self.assertEqual(cycle["scale"], "RHE")
        self.assertEqual(cycle["status"], "available")
        self.assertIn("Gmax_U_eV", cycle)
        self.assertEqual(result["network_resting_states"]["1.23"]["node_id"], "reference")

    def test_unlisted_state_folder_is_not_used_and_review_maps_images(self):
        self._network_with_twin()
        manual = self.campaign / "states" / "reference_manual"
        shutil.copytree(self.state, manual)
        meta = json.loads((manual / "state.json").read_text())
        meta["id"] = "reference_manual"
        (manual / "state.json").write_text(json.dumps(meta))
        (manual / "04_static" / "OUTCAR").write_text(
            outcar_text().replace("energy(sigma->0) = -10.00000000", "energy(sigma->0) = -10.50000000")
        )
        result = analyze_thermodynamics(self.campaign, self.campaign / "references.json", include_vibrations=False)
        node = {n["node_id"]: n for n in result["nodes"]}["reference"]
        self.assertEqual(node["selected_state_id"], "reference")  # not the folder no node lists
        self.assertEqual(node["reassigned_in"], [])
        shutil.rmtree(manual)
        # Without sites.json an explicit assignment is still validated and mapped to the representative.
        (self.state / "review.json").write_text(json.dumps({"assign_to_node": "twin", "comment": "checked"}))
        meta = json.loads((self.state / "state.json").read_text())
        result = topology_check(self.state, meta)
        self.assertEqual(result["assigned_node_id"], "reference")
        self.assertTrue(result["valid_for_assigned_node"])
        (self.state / "review.json").write_text(json.dumps({"assign_to_node": "nowhere", "comment": "typo"}))
        self.assertIsNone(topology_check(self.state, meta)["assigned_node_id"])

    def test_electronic_mode_adds_adsorbate_corrections_and_final_source(self):
        self._network_with_twin()
        meta = json.loads((self.state / "state.json").read_text())
        meta["delta_atoms"] = {"O": 1, "H": 1}
        (self.state / "state.json").write_text(json.dumps(meta))
        result = analyze_thermodynamics(self.campaign, self.campaign / "references.json", include_vibrations=False)
        self.assertIn("tabulated_adsorbate", result["analysis_level"])
        node = {n["node_id"]: n for n in result["nodes"]}["reference"]
        self.assertAlmostEqual(node["node_energy_eV"], -10.0 + 0.30 + 0.05)
        final = analyze_thermodynamics(
            self.campaign, self.campaign / "references.json", include_vibrations=False, energy_source="final"
        )
        self.assertEqual(final["energy_source"], "final")
        with self.assertRaises(ValueError):
            analyze_thermodynamics(self.campaign, self.campaign / "references.json", energy_source="final")

    def test_select_for_thermo_keeps_variants_within_window(self):
        second = self.campaign / "states" / "reference_v2"
        shutil.copytree(self.state, second)
        meta = json.loads((second / "state.json").read_text())
        meta["id"] = "reference_v2"
        (second / "state.json").write_text(json.dumps(meta))
        (second / "02_final" / "OUTCAR").write_text(
            outcar_text(ionic=True).replace("energy(sigma->0) = -10.00000000", "energy(sigma->0) = -9.95000000")
        )
        network = {
            "reference": "reference",
            "nodes": [{"id": "reference", "state_ids": ["reference", "reference_v2"], "delta_atoms": {"O": 0, "H": 0}}],
            "edges": [],
            "cycles": [],
        }
        (self.campaign / "network.json").write_text(json.dumps(network))
        result = select_for_thermo(self.campaign, energy_window_eV=0.10, max_variants_per_node=2)
        self.assertEqual(result["state_ids"], ["reference", "reference_v2"])
        result = select_for_thermo(self.campaign, energy_window_eV=0.01, max_variants_per_node=2)
        self.assertEqual(
            result["state_ids"], ["reference", "reference_v2"], "a state that already ran later stages stays selected"
        )
        (self.campaign / "thermo_selection.json").unlink()
        for stage in ("03_refine", "04_static", "05_freq"):
            (second / stage / "OUTCAR").unlink()
        result = select_for_thermo(self.campaign, energy_window_eV=0.01, max_variants_per_node=2)
        self.assertEqual(result["state_ids"], ["reference"])
        (second / "02_final" / "OUTCAR").unlink()
        result = select_for_thermo(self.campaign)
        self.assertEqual(result["waiting_nodes"], [{"node_id": "reference", "blocking_states": ["reference_v2"]}])
        self.assertEqual(select_for_thermo(self.campaign, include_partial=True)["state_ids"], ["reference"])
        # A recorded failure needing --retry does not block its sibling.
        (second / "02_final" / "OUTCAR").write_text(outcar_text(ionic=False, footer=False))
        (second / "02_final" / "attempt.json").write_text(json.dumps({"status": "failed"}))
        result = select_for_thermo(self.campaign)
        self.assertEqual(result["waiting_nodes"], [])
        self.assertEqual(result["state_ids"], ["reference"])

    def test_variants_in_other_magnetic_solutions_are_reported(self):
        second = self.campaign / "states" / "reference_v2"
        shutil.copytree(self.state, second)
        meta = json.loads((second / "state.json").read_text())
        meta["id"] = "reference_v2"
        (second / "state.json").write_text(json.dumps(meta))
        (second / "02_final" / "OUTCAR").write_text(
            outcar_text(ionic=True)
            .replace("energy(sigma->0) = -10.00000000", "energy(sigma->0) = -9.95000000")
            .replace("1 0.0 0.1 0.0 0.1", "1 0.0 -0.9 0.0 -0.9")
        )
        self._network_with_twin()
        network = json.loads((self.campaign / "network.json").read_text())
        network["nodes"][0]["state_ids"].append("reference_v2")
        (self.campaign / "network.json").write_text(json.dumps(network))
        result = select_for_thermo(self.campaign)
        self.assertEqual(result["nodes_with_several_magnetic_solutions"], ["reference"])
        solutions = next(n for n in result["nodes"] if n["node_id"] == "reference")["magnetic_solutions"]
        self.assertEqual([s["state_ids"] for s in solutions], [["reference"], ["reference_v2"]])
        self.assertAlmostEqual(solutions[1]["relative_eV"], 0.05)
        self.assertEqual(
            solutions[1]["differs_from_lowest"][0],
            {"atom_id": "a0001", "element": "O", "lowest_muB": 0.1, "this_muB": -0.9},
        )
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["select", str(self.campaign)]), 0)
        self.assertIn("different magnetic solutions", out.getvalue())
        # thermo compares the same variants from the collected tables
        result = analyze_thermodynamics(self.campaign, self.campaign / "references.json", include_vibrations=False)
        nodes = {n["node_id"]: n for n in result["nodes"]}
        self.assertEqual(nodes["reference"]["magnetic_solutions"], 2)
        self.assertEqual(result["magnetism"]["nodes_with_several_solutions"], ["reference"])
        # the twin has no recorded translation, so the toy cycle is checked only where it can be
        self.assertEqual(result["magnetism"]["cycles"]["toy"]["status"], "incomplete")
        self.assertIn("translation", result["magnetism"]["cycles"]["toy"]["missing"][0]["reason"])
        rows = (self.campaign / "analysis" / "cycles.csv").read_text().splitlines()
        self.assertIn("spin_check", rows[0])

    def test_select_for_thermo_can_be_restricted_to_cycles(self):
        nodes = {"reference": "reference", "a": "a_v1", "b": "b_v1", "c": "c_v1"}
        for node_id, sid in nodes.items():
            if sid != "reference":
                shutil.copytree(self.state, self.campaign / "states" / sid)
                meta = json.loads((self.campaign / "states" / sid / "state.json").read_text())
                meta.update(id=sid, node_id=node_id)
                (self.campaign / "states" / sid / "state.json").write_text(json.dumps(meta))
        delta = {"O": 0, "H": 0}
        network = {
            "reference": "reference",
            "nodes": [{"id": n, "state_ids": [sid], "delta_atoms": delta} for n, sid in nodes.items()]
            + [{"id": "c_image", "state_ids": [], "computed_by": "c", "delta_atoms": delta}],
            "edges": [
                {"id": "e1", "from": "a", "to": "c_image"},
                {"id": "e2", "from": "c_image", "to": "a"},
                {"id": "e3", "from": "b", "to": "b"},
            ],
            "cycles": [
                {"id": "C1", "edge_ids": ["e1", "e2"], "family": "f"},
                {"id": "C2", "edge_ids": ["e3"], "family": "f"},
            ],
        }
        (self.campaign / "network.json").write_text(json.dumps(network))
        result = select_for_thermo(self.campaign, cycles=["C1"])
        self.assertEqual(
            result["state_ids"],
            ["a_v1", "c_v1", "reference"],
            "the cycle's nodes (images mapped to their representative) plus the reference",
        )
        self.assertEqual(result["criteria"]["cycles"], ["C1"])
        self.assertEqual(select_for_thermo(self.campaign)["state_ids"], ["a_v1", "b_v1", "c_v1", "reference"])
        with self.assertRaisesRegex(ValueError, "Unknown cycle IDs"):
            select_for_thermo(self.campaign, cycles=["C9"])
        (self.campaign / "thermo_selection.json").unlink()
        ids = self.campaign / "top_cycles.txt"
        ids.write_text("C2\n")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["select", str(self.campaign), "--cycles", f"@{ids}"]), 0)
        chosen = json.loads((self.campaign / "thermo_selection.json").read_text())
        self.assertEqual(chosen["state_ids"], ["b_v1", "reference"])


SHARED_INITIAL = """shared proton campaign fixture
1.0
20 0 0
0 20 0
0 0 20
O H
3 1
Selective dynamics
Cartesian
2 2 2 T T T
2 2 4.47 T T T
8 8 8 F F F
2 2 3.09 T T T
"""
# H85 moves from O65 (1.09 A) to O68 (0.97 A): O65 becomes O, O68 becomes OH again.
SHARED_HOPPED = SHARED_INITIAL.replace("2 2 3.09 T T T", "2 2 3.50 T T T")


def outcar_n_atoms(n_atoms, energy=-10.0, ionic=True):
    lines = [
        f"NIONS = {n_atoms}",
        "NELM = 4; EDIFF = 1.0E-07",
        "Iteration 1( 1)",
        "Iteration 1( 2)",
        "aborting loop because EDIFF is reached",
        "POSITION TOTAL-FORCE (eV/Angst)",
        "------------",
    ]
    lines += ["0.0 0.0 0.0 0.001 0.002 0.003"] * n_atoms
    lines += [
        "------------",
        f"free  energy   TOTEN = {energy - 0.1:.8f} eV",
        f"energy without entropy= {energy:.8f} energy(sigma->0) = {energy:.8f}",
    ]
    if ionic:
        lines.append("reached required accuracy - stopping structural energy minimisation")
    lines += [
        "General timing and accounting informations for this job:",
        "Total CPU time used (sec): 1.0",
        "Elapsed time (sec): 1.0",
    ]
    return "\n".join(lines) + "\n"


class FinalProtocolCollectionTests(unittest.TestCase):
    """frequency.central = "final": node energy from 02_final, vibrations at that geometry."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.campaign = Path(self.tmp.name)
        self.state = self.campaign / "states" / "reference"
        self.state.mkdir(parents=True)
        (self.campaign / "config.json").write_text(
            json.dumps(
                {
                    "analysis": {"require_input_hashes": False},
                    "frequency": {"central": "final", "max_force_eVA": 0.035, "warn_force_eVA": 0.03},
                    "thermodynamics": {"potentials_V": [1.23, 1.53]},
                }
            )
        )
        (self.state / "POSCAR").write_text(POSCAR)
        (self.state / "state.json").write_text(
            json.dumps(
                {
                    "id": "reference",
                    "node_id": "reference",
                    "family": "reference",
                    "site": "whole_slab",
                    "variant": 0,
                    "atom_ids": ["a0001", "a0002"],
                    "symbols": ["O", "H"],
                    "relax_mobile": [[False] * 3, [True] * 3],
                    "frequency_mobile": [[False] * 3, [True] * 3],
                    "delta_atoms": {"O": 0, "H": 0},
                    "expected_bonds": [["a0001", "a0002"]],
                }
            )
        )
        for stage in ("01_loose", "02_final", "04_static", "05_freq"):
            directory = self.state / stage
            directory.mkdir()
            (directory / "POSCAR").write_text(POSCAR)
            (directory / "CONTCAR").write_text(POSCAR)
            ibrion = 5 if stage == "05_freq" else -1 if stage == "04_static" else 2
            (directory / "INCAR").write_text(f"IBRION={ibrion}; NELM=4; EDIFF=1e-6; EDIFFG=-0.03; NFREE=2; ISMEAR=0\n")
            # 02_final ends at E0 = -9.998 eV (three ionic steps), 04_static at -10.000 eV
            (directory / "OUTCAR").write_text(
                outcar_text(
                    n_evaluations=7 if stage == "05_freq" else 3 if stage == "02_final" else 1,
                    frequencies=[100, 500, 3000] if stage == "05_freq" else None,
                    ionic=stage in ("01_loose", "02_final"),
                )
            )
        (self.campaign / "network.json").write_text(
            json.dumps(
                {
                    "reference": "reference",
                    "nodes": [
                        {"id": "reference", "state_ids": ["reference"], "delta_atoms": {"O": 0, "H": 0}},
                        {"id": "twin", "state_ids": [], "computed_by": "reference", "delta_atoms": {"O": 0, "H": 0}},
                    ],
                    "edges": [
                        {"id": "e1", "from": "reference", "to": "twin", "water": 0, "pairs": 0, "oxygen": 0},
                        {"id": "e2", "from": "twin", "to": "reference", "water": 2, "pairs": 4, "oxygen": 1},
                    ],
                    "cycles": [{"id": "toy", "edge_ids": ["e1", "e2"]}],
                }
            )
        )
        (self.campaign / "references.json").write_text(
            json.dumps(
                {
                    "references": {
                        "H2": {"E": -6.77398545, "Gcorr": -0.121865},
                        "H2O": {"E": -14.24029443, "Gcorr": -0.017671},
                    }
                }
            )
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_node_energy_is_the_final_e0_and_vibrations_are_added_to_it(self):
        result = collect(self.campaign)
        state = result["states"][0]
        self.assertTrue(state["free_energy_valid"], state)
        self.assertNotIn("03_refine", state["stages"])
        self.assertAlmostEqual(state["energy_e0_eV"], -9.998)
        self.assertEqual(state["energy_e0_eV"], state["energy_final_e0_eV"])
        self.assertAlmostEqual(state["energy_static_e0_eV"], -10.0)
        self.assertAlmostEqual(state["static_minus_final_e0_eV"], -0.002)
        self.assertAlmostEqual(state["G_eV"], -9.998 + state["F_vib_eV"])
        self.assertIsNone(state["max_refine_mobile_force_eVA"])
        self.assertFalse(any("04_static: E0 differs" in w for w in state["warnings"]))
        self.assertEqual(result["summary"]["frequency_central"], "final")
        self.assertIn("02_final E0", result["summary"]["energy_source"])

    def test_static_problems_affect_only_the_free_energy(self):
        moved = POSCAR.replace("2 2 3 T T T", "2 2 3.1 T T T")
        (self.state / "04_static" / "POSCAR").write_text(moved)
        state = collect(self.campaign)["states"][0]
        self.assertTrue(state["electronic_valid"])
        self.assertFalse(state["free_energy_valid"])
        self.assertEqual(state["status"], "electronic_energy_ready")
        self.assertIn("04_static: geometry does not match the 02_final structure", state["frequency_exclusion_reasons"])
        (self.state / "04_static" / "POSCAR").write_text(POSCAR)
        (self.state / "04_static" / "OUTCAR").write_text(outcar_text(footer=False))
        state = collect(self.campaign)["states"][0]
        self.assertTrue(state["electronic_valid"])
        self.assertIn("04_static: incomplete", state["frequency_exclusion_reasons"])

    def test_another_static_solution_excludes_the_free_energy(self):
        (self.state / "04_static" / "OUTCAR").write_text(
            outcar_text().replace("energy(sigma->0) = -10.00000000", "energy(sigma->0) = -9.98000000")
        )
        state = collect(self.campaign)["states"][0]
        self.assertAlmostEqual(state["static_minus_final_e0_eV"], 0.018)
        self.assertTrue(state["electronic_valid"], "the 02_final energy itself is fine")
        self.assertFalse(state["free_energy_valid"])
        self.assertTrue(
            any(
                "04_static: E0 differs from the 02_final E0 by +18.0 meV" in r
                for r in state["frequency_exclusion_reasons"]
            )
        )
        config = json.loads((self.campaign / "config.json").read_text())
        config["frequency"]["max_static_e0_shift_eV"] = 0.02  # accepted after inspection
        (self.campaign / "config.json").write_text(json.dumps(config))
        self.assertTrue(collect(self.campaign)["states"][0]["free_energy_valid"])

    def test_a_moment_change_from_final_to_static_excludes_the_free_energy(self):
        (self.state / "04_static" / "OUTCAR").write_text(
            outcar_text().replace("2 0.0 0.2 0.0 0.2", "2 0.0 0.2 0.0 0.9")
        )
        state = collect(self.campaign)["states"][0]
        self.assertAlmostEqual(state["max_local_moment_change_final_to_static_muB"], 0.7)
        self.assertFalse(state["free_energy_valid"])
        self.assertTrue(
            any("a local moment changed by 0.70 muB from 02_final" in r for r in state["frequency_exclusion_reasons"])
        )

    def test_thermodynamics_default_to_the_final_energy(self):
        refs = self.campaign / "references.json"
        full = analyze_thermodynamics(self.campaign, refs)
        self.assertEqual((full["energy_source"], full["frequency_central"]), ("final", "final"))
        state = collect(self.campaign)["states"][0]
        node = {n["node_id"]: n for n in full["nodes"]}["reference"]
        self.assertAlmostEqual(node["node_energy_eV"], state["G_eV"])
        screening = analyze_thermodynamics(self.campaign, refs, include_vibrations=False)
        self.assertEqual(screening["energy_source"], "final")
        node = {n["node_id"]: n for n in screening["nodes"]}["reference"]
        self.assertAlmostEqual(node["node_energy_eV"], -9.998)
        with self.assertRaisesRegex(ValueError, "frequency.central = 'final'"):
            analyze_thermodynamics(self.campaign, refs, energy_source="static")

    def test_state_prepared_for_the_refine_protocol_has_no_free_energy(self):
        (self.state / "plan.json").write_text(json.dumps({"stages": {}}))
        state = collect(self.campaign)["states"][0]
        self.assertTrue(state["electronic_valid"])
        self.assertFalse(state["free_energy_valid"])
        self.assertTrue(
            any("prepared with frequency.central = 'refine'" in r for r in state["frequency_exclusion_reasons"])
        )


class MicrostateReassignmentTests(unittest.TestCase):
    """A converged structure that relaxed into another node's site occupancy is used for that node."""

    def setUp(self):
        import mechanet.analysis as analysis

        analysis._CONTEXT_CACHE.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.campaign = Path(self.tmp.name)
        (self.campaign / "config.json").write_text(
            json.dumps({"analysis": {"require_input_hashes": False}, "thermodynamics": {"potentials_V": [1.23]}})
        )
        write_sites(self.campaign, [("a0065", "terminal", "OH"), ("a0068", "terminal", "OH")])
        self.network = {
            "reference": "ref",
            "nodes": [
                {"id": "ref", "state_ids": [], "occupancy": {}, "coupling": None, "delta_atoms": {"O": 0, "H": 0}},
                {
                    "id": "n_0065O",
                    "state_ids": [],
                    "occupancy": {"a0065": "O"},
                    "coupling": None,
                    "delta_atoms": {"O": 0, "H": -1},
                },
                {
                    "id": "n_0068O",
                    "state_ids": ["n_0068O__v1"],
                    "occupancy": {"a0068": "O"},
                    "coupling": None,
                    "delta_atoms": {"O": 0, "H": -1},
                },
                {
                    "id": "n_0065O_0068H2O",
                    "state_ids": [],
                    "occupancy": {"a0065": "O", "a0068": "H2O"},
                    "coupling": None,
                    "delta_atoms": {"O": 0, "H": 0},
                },
            ],
            "edges": [
                {"id": "e1", "from": "ref", "to": "n_0065O", "water": 0, "pairs": 1, "oxygen": 0},
                {"id": "e2", "from": "ref", "to": "n_0068O", "water": 0, "pairs": 1, "oxygen": 0},
            ],
            "cycles": [],
        }
        self._write_network()
        self.state = self.campaign / "states" / "n_0068O__v1"
        self.state.mkdir(parents=True)
        self.meta = {
            "id": "n_0068O__v1",
            "node_id": "n_0068O",
            "node_aliases": ["n_0068O"],
            "family": "aem",
            "site": "a0068",
            "variant": {},
            "atom_ids": ["a0065", "a0068", "a0010", "a0085"],
            "symbols": ["O", "O", "O", "H"],
            "relax_mobile": [[True] * 3, [True] * 3, [False] * 3, [True] * 3],
            "frequency_mobile": [[True] * 3, [True] * 3, [False] * 3, [True] * 3],
            "delta_atoms": {"O": 0, "H": -1},
            "expected_bonds": [["a0065", "a0085"]],
        }
        (self.state / "state.json").write_text(json.dumps(self.meta))
        (self.state / "POSCAR").write_text(SHARED_INITIAL)
        for stage in STAGES:
            directory = self.state / stage
            directory.mkdir()
            geometry = SHARED_INITIAL if stage == "01_loose" else SHARED_HOPPED
            (directory / "POSCAR").write_text(geometry)
            (directory / "CONTCAR").write_text(geometry)
            ibrion = -1 if stage == "04_static" else 2
            (directory / "INCAR").write_text(f"IBRION={ibrion}; NELM=4; EDIFF=1e-7; EDIFFG=-0.03; ISMEAR=0\n")
            if stage != "05_freq":
                (directory / "OUTCAR").write_text(outcar_n_atoms(4, ionic=stage != "04_static"))

    def tearDown(self):
        self.tmp.cleanup()

    def _write_network(self):
        import mechanet.analysis as analysis

        analysis._CONTEXT_CACHE.clear()
        (self.campaign / "network.json").write_text(json.dumps(self.network))

    def test_shared_proton_hop_is_reassigned_to_the_matching_node(self):
        result = topology_check(self.state, self.meta)
        self.assertEqual(result["status"], "reassigned_to_existing_node", result)
        self.assertEqual(result["assigned_node_id"], "n_0065O")
        self.assertFalse(result["valid_for_assigned_node"])
        self.assertEqual(result["microstate"]["occupancy_changes"], {"a0065": "O"})
        self.assertEqual(result["shared_proton_transfers"][0]["to"], "a0068")

    def test_symmetry_image_maps_to_its_representative(self):
        self.network["nodes"][1].update(computed_by="n_0068O_rep")
        self.network["nodes"].append(
            {
                "id": "n_0068O_rep",
                "state_ids": [],
                "occupancy": {"a0068": "OH"},
                "coupling": ["x", "y"],
                "delta_atoms": {"O": 0, "H": -1},
            }
        )
        self._write_network()
        result = topology_check(self.state, self.meta)
        self.assertEqual(result["assigned_node_id"], "n_0068O_rep")
        self.assertEqual(result["microstate"]["via_symmetry_image"], "n_0065O")

    def test_proton_on_framework_oxygen_or_unlisted_occupancy_needs_review(self):
        to_framework = SHARED_INITIAL.replace("2 2 3.09 T T T", "8 8 7.03 T T T")
        (self.state / "04_static" / "POSCAR").write_text(to_framework)
        result = topology_check(self.state, self.meta)
        self.assertIsNone(result["assigned_node_id"])
        self.assertEqual(result["microstate"]["status"], "outside_site_model")
        self.assertIn("framework oxygen a0010", result["microstate"]["reason"])
        del self.network["nodes"][1]
        self._write_network()
        result = topology_check(self.state, self.meta)
        result = topology_check(self.state, self.meta, geometry="02_final")
        self.assertEqual(result["microstate"]["status"], "unlisted_microstate")
        self.assertIsNone(result["assigned_node_id"])

    def test_explicit_review_assignment(self):
        (self.state / "review.json").write_text(json.dumps({"assign_to_node": "n_0065O_0068H2O", "comment": "x"}))
        result = topology_check(self.state, self.meta)
        self.assertEqual(result["status"], "changed_connectivity_needs_review")
        self.assertIn("composition differs", result["microstate"]["reason"])
        (self.state / "review.json").write_text(json.dumps({"assign_to_node": "n_0068O", "comment": "kept"}))
        result = topology_check(self.state, self.meta)
        self.assertEqual(result["status"], "assigned_by_explicit_review")
        self.assertTrue(result["valid_for_assigned_node"])

    def test_variant_that_leaves_its_node_after_refinement_is_replaced(self):
        # v1: no hop at 02_final (stays n_0068O), hop during 03_refine (now n_0065O).
        (self.state / "02_final" / "CONTCAR").write_text(SHARED_INITIAL)
        second = self.campaign / "states" / "n_0068O__v2"
        shutil.copytree(self.state, second)
        meta = dict(self.meta, id="n_0068O__v2")
        (second / "state.json").write_text(json.dumps(meta))
        for stage in STAGES:
            for name in ("POSCAR", "CONTCAR"):
                (second / stage / name).write_text(SHARED_INITIAL)
        (second / "02_final" / "OUTCAR").write_text(outcar_n_atoms(4, energy=-9.9))
        for stage in ("03_refine", "04_static"):
            for name in ("OUTCAR", "POSCAR", "CONTCAR"):
                (second / stage / name).unlink()
        self.network["nodes"][2]["state_ids"] = ["n_0068O__v1", "n_0068O__v2"]
        self._write_network()
        selection = select_for_thermo(self.campaign, max_variants_per_node=1)
        node = {n["node_id"]: n for n in selection["nodes"]}["n_0068O"]
        self.assertEqual(node["selected"], ["n_0068O__v2"])
        self.assertEqual(node["left_node_after_refinement"], {"n_0068O__v1": "n_0065O"})

    def test_selection_and_thermodynamics_use_the_reassigned_node(self):
        selection = select_for_thermo(self.campaign)
        nodes = {n["node_id"]: n for n in selection["nodes"]}
        self.assertEqual(nodes["n_0065O"]["status"], "selected")
        self.assertEqual(nodes["n_0065O"]["selected"], ["n_0068O__v1"])
        self.assertEqual(nodes["n_0068O"]["status"], "no_accepted_variant")
        self.assertEqual(nodes["n_0068O"]["reassigned_out"], {"n_0068O__v1": "n_0065O"})
        (self.campaign / "references.json").write_text(
            json.dumps(
                {
                    "references": {
                        "H2": {"E": -6.77398545, "Gcorr": -0.121865},
                        "H2O": {"E": -14.24029443, "Gcorr": -0.017671},
                    }
                }
            )
        )
        result = analyze_thermodynamics(self.campaign, self.campaign / "references.json", include_vibrations=False)
        rows = {n["node_id"]: n for n in result["nodes"]}
        self.assertEqual(rows["n_0065O"]["selected_state_id"], "n_0068O__v1")
        self.assertEqual(rows["n_0065O"]["reassigned_in"], ["n_0068O__v1"])
        self.assertEqual(rows["n_0068O"]["status"], "unresolved")
        state = json.loads((self.campaign / "analysis" / "states.json").read_text())["states"][0]
        self.assertEqual(state["assigned_node_id"], "n_0065O")
        self.assertTrue(state["electronic_valid"], state["electronic_exclusion_reasons"])
        self.assertTrue(any("n_0065O" in w for w in state["warnings"]))


OO_END_ON = """end-on OO fixture
1.0
20 0 0
0 20 0
0 0 20
Fe O
1 2
Selective dynamics
Cartesian
5 5 5 F F F
5 5 6.75 T T T
5 6.0 7.59 T T T
"""
OO_SIDE_ON = OO_END_ON.replace("5 6.0 7.59 T T T", "5 6.25 6.40 T T T")  # distal O now 1.88 A from Fe


class BindingModeTests(unittest.TestCase):
    """End-on <-> side-on is a binding variant of the same OO node; a site O leaving its metal is not."""

    def setUp(self):
        import mechanet.analysis as analysis

        analysis._CONTEXT_CACHE.clear()
        self.tmp = tempfile.TemporaryDirectory()
        campaign = Path(self.tmp.name)
        write_sites(campaign, [("a0065", "terminal", "OH")])
        (campaign / "network.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {
                            "id": "ref",
                            "state_ids": [],
                            "occupancy": {},
                            "coupling": None,
                            "delta_atoms": {"O": 0, "H": 0},
                        },
                        {
                            "id": "n_0065OO",
                            "state_ids": ["s"],
                            "occupancy": {"a0065": "OO"},
                            "coupling": None,
                            "delta_atoms": {"O": 1, "H": -1},
                        },
                    ],
                    "edges": [],
                    "cycles": [],
                }
            )
        )
        self.state = campaign / "states" / "s"
        (self.state / "04_static").mkdir(parents=True)
        (self.state / "POSCAR").write_text(OO_END_ON)
        self.meta = {
            "id": "s",
            "node_id": "n_0065OO",
            "atom_ids": ["a0001", "a0065", "d_a0065"],
            "frequency_mobile": [[False] * 3, [True] * 3, [True] * 3],
            "delta_atoms": {"O": 1, "H": -1},
            "expected_bonds": [["a0065", "d_a0065"]],
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_end_on_to_side_on_keeps_the_node(self):
        (self.state / "04_static" / "POSCAR").write_text(OO_SIDE_ON)
        result = topology_check(self.state, self.meta)
        self.assertEqual(result["new_active_metal_contacts"], [("a0001", "d_a0065")])
        self.assertEqual(result["status"], "same_occupancy_binding_mode_changed", result)
        self.assertTrue(result["valid_for_assigned_node"])

    def test_site_oxygen_leaving_its_metal_needs_review(self):
        (self.state / "04_static" / "POSCAR").write_text(
            OO_END_ON.replace("5 5 6.75 T T T", "5 5 7.95 T T T").replace("5 6.0 7.59 T T T", "5 6.0 8.79 T T T")
        )
        result = topology_check(self.state, self.meta)
        self.assertEqual(result["lost_active_metal_contacts"], [("a0001", "a0065")])
        self.assertIsNone(result["assigned_node_id"])
        self.assertEqual(result["microstate"]["status"], "metal_attachment_changed")


if __name__ == "__main__":
    unittest.main()
