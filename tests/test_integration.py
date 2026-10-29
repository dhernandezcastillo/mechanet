"""End-to-end smoke test on the real seed: builder -> VASP stages -> selection -> analysis.

The VASP executable is the synthetic one from support.py (a harmonic force model around
each state's own starting geometry), so energies are meaningless; the test checks that every
layer hands the right files and numbers to the next. It runs both protocols:
frequency.central = "refine" (five stages) and "final" (no 03_refine).
Set MECHANET_SKIP_INTEGRATION=1 to skip it (it starts several hundred processes).
The campaign configuration comes from examples/nife_ldh/project.yaml.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import EXAMPLE, FAKE_VASP, FINAL_PROTOCOL, POMASS, REFINE_PROTOCOL, write_potcars  # noqa: E402

from mechanet.analysis import analyze_thermodynamics, collect, select_for_thermo  # noqa: E402
from mechanet.builder import Builder  # noqa: E402
from mechanet.fileio import read_json, write_json  # noqa: E402
from mechanet.project import load_project, resolve_config  # noqa: E402
from mechanet.runner import prepare, run_state  # noqa: E402
from mechanet.structure import read_poscar  # noqa: E402

# Molecular references in the full format (meaningless here, since the synthetic energies
# only test the bookkeeping and the closure of every cycle).
REFERENCES = {
    "temperature_K": 298.15,
    "protocol_id": "synthetic test references",
    "H2": {"energy_e0_eV": -6.77398545, "free_energy_correction_eV": -0.121865, "phase": "gas"},
    "H2O": {
        "energy_e0_eV": -14.24029443,
        "free_energy_correction_eV": -0.017671,
        "phase": "liquid",
        "Gcorr_pressure_bar": 0.031,
    },
    "oxygen_reference": {"method": "water_formation", "delta_G_2H2O_to_O2_2H2_eV": 4.92, "temperature_K": 298.15},
}


@unittest.skipIf(os.environ.get("MECHANET_SKIP_INTEGRATION"), "integration test disabled by environment")
class EndToEndTests(unittest.TestCase):
    def test_aem_cycle_from_builder_to_free_energies(self):
        for protocol in (FINAL_PROTOCOL, REFINE_PROTOCOL):
            with self.subTest(central=protocol["central"]):
                self._aem_cycle(protocol)

    def _aem_cycle(self, protocol):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = resolve_config(load_project(EXAMPLE / "project.yaml"), cluster=False)
            cfg["frequency"].update(protocol)
            cfg["scope"]["families"] = ["aem"]
            cfg["sites"] = [s for s in cfg["sites"] if s["id"] == "a0065"]
            potcars = write_potcars(root)
            fake = root / "fake_vasp.py"
            fake.write_text(FAKE_VASP)
            cfg["vasp"].update(potcars=potcars, command=[sys.executable, "-I", str(fake)])
            cfg["thermo_selection"] = {"energy_window_eV": 0.10, "max_variants_per_node": 1}
            cfg["analysis"]["require_input_hashes"] = True
            campaign = root / "campaign"
            report = Builder(cfg, campaign).finish()
            self.assertGreaterEqual(report["unique_initial_structures"], 5)
            # A stiff harmonic model around each start: relaxations stay put, Hessians are K.
            for state_dir in sorted((campaign / "states").glob("*")):
                st = read_poscar(state_dir / "POSCAR")
                (state_dir / "ctl").mkdir()
                masses = np.repeat([POMASS[s] for s in st.symbols], 3)
                k = np.diag(25.0 * np.sqrt(masses))
                write_json(state_dir / "harmonic.json", {"x0": st.positions.reshape(-1).tolist(), "K": k.tolist()})
            prepare(campaign, with_potcars=True)
            for state_dir in sorted((campaign / "states").glob("*")):
                result = run_state(state_dir, through="final", execute=True)
                self.assertEqual(result["status"], "complete", result)
            selection = select_for_thermo(campaign)
            net = read_json(campaign / "network.json")
            aem = next(c for c in net["cycles"] if c["id"] == "aem_a0065_AEM")
            edges = {e["id"]: e for e in net["edges"]}
            cycle_nodes = [edges[e]["from"] for e in aem["edge_ids"]]
            chosen = {}
            for entry in selection["nodes"]:
                if entry["node_id"] in cycle_nodes:
                    self.assertEqual(entry["status"], "selected")
                    chosen[entry["node_id"]] = entry["selected"][0]
            self.assertEqual(set(chosen), set(cycle_nodes))
            for sid in chosen.values():
                result = run_state(campaign / "states" / sid, through="freq", execute=True)
                self.assertEqual(result["status"], "complete", result)
            collected = collect(campaign)
            rows = {r["state_id"]: r for r in collected["states"]}
            final_mode = protocol["central"] == "final"
            for sid in chosen.values():
                row = rows[sid]
                self.assertTrue(
                    row["free_energy_valid"],
                    (sid, row["electronic_exclusion_reasons"], row["frequency_exclusion_reasons"]),
                )
                state_dir = campaign / "states" / sid
                self.assertEqual((state_dir / "03_refine").exists(), not final_mode)
                central = state_dir / ("02_final" if final_mode else "03_refine") / "CONTCAR"
                np.testing.assert_allclose(
                    read_poscar(state_dir / "04_static" / "POSCAR").positions, read_poscar(central).positions, atol=1e-9
                )
                node_energy = row["energy_final_e0_eV"] if final_mode else row["energy_static_e0_eV"]
                self.assertEqual(row["energy_e0_eV"], node_energy)
                self.assertAlmostEqual(row["G_eV"], node_energy + row["F_vib_eV"])
                st = read_poscar(campaign / "states" / sid / "POSCAR")
                active = np.asarray(
                    json.loads((campaign / "states" / sid / "state.json").read_text())["frequency_mobile"]
                ).any(axis=1)
                expected = sorted(
                    521.4709 * np.sqrt(25.0 * np.sqrt(POMASS[s]) / POMASS[s])
                    for s, a in zip(st.symbols, active)
                    if a
                    for _ in range(3)
                )
                np.testing.assert_allclose(sorted(row["stages"]["05_freq"]["frequencies_cm1"]), expected, rtol=2e-3)
            refs = root / "references.json"
            refs.write_text(json.dumps(REFERENCES))
            full = analyze_thermodynamics(campaign, refs, potentials_V=[1.23, 1.6])
            self.assertEqual(full["energy_source"], "final" if final_mode else "static")
            # Rows carry potential_V and its scale.
            row = next(c for c in full["cycles"] if c["cycle_id"] == "aem_a0065_AEM" and c["potential_V"] == 1.6)
            self.assertEqual(row["scale"], "RHE")
            self.assertEqual(row["status"], "available", row)
            self.assertAlmostEqual(row["net_deltaG_eV"], 4.92 - 4 * 1.6, places=6)
            for key in ("Gmax_U_eV", "cyclic_span_eV", "lowest_intermediate_node", "network_resting_state_node"):
                self.assertIn(key, row)
            screening = analyze_thermodynamics(
                campaign, refs, out_dir=root / "screening", include_vibrations=False, energy_source="final"
            )
            self.assertTrue(any(c["status"] == "available" for c in screening["cycles"]))
            try:
                import matplotlib  # noqa: F401
            except ImportError:
                return
            from mechanet.plotting import plot_profile

            plot = plot_profile(campaign, "aem_a0065_AEM", 1.6, root / "profile.png")
            self.assertTrue(Path(plot["figure"]).is_file())


if __name__ == "__main__":
    unittest.main()
