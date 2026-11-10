"""Collect the evidence of every calculation: convergence, forces, provenance, moments, frequencies.

:func:`collect` parses each stage of each state, checks it (input and output hashes, force and
energy criteria, the topology of the relaxed structure, imaginary modes) and writes the QC
tables. Collection is read-only with respect to calculation folders; unknown, failed, altered
and imaginary-mode results stay in the report but are excluded from free energies. Part of
:mod:`mechanet.analysis` up to 0.1.4, which still exports every name.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fileio import _csv, file_hash, _load, _write
from .plugins import campaign_registry
from .results import FREQUENCY_CM1_PER_SQRT_EV_A2_AMU, parse_run, read_incar, same_solution
from .structure import distance_matrix, minimum_image, read_poscar
from .thermo import vibrational_thermochemistry
from .topology import BOND_CUTOFF, topology_check
from .vasp import (
    STAGES,
    STAGE_FINAL,
    STAGE_REFINE,
    STAGE_STATIC,
    STAGE_FREQ,
    central_mode,
    frequency_gates,
    plan_mode,
    protocol_stages,
)

ENERGY_SOURCE_TEXT = {
    "refine": f"{STAGE_STATIC} E0 at the {STAGE_REFINE} geometry",
    "final": f"{STAGE_FINAL} E0 at the {STAGE_FINAL} geometry; 05_freq at the same geometry "
    f"with the {STAGE_FINAL} Hamiltonian (frequency.central = final)",
}


def _settings(campaign, temperature_K, low_frequency_cutoff_cm1, imaginary_tolerance_cm1):
    config = _load(Path(campaign) / "config.json")
    thermo = config.get("thermodynamics", {})
    return config, {
        "temperature_K": float(temperature_K if temperature_K is not None else thermo.get("temperature_K", 298.15)),
        "low_frequency_cutoff_cm1": float(
            low_frequency_cutoff_cm1
            if low_frequency_cutoff_cm1 is not None
            else thermo.get("low_frequency_cutoff_cm1", 100.0)
        ),
        "imaginary_tolerance_cm1": float(
            imaginary_tolerance_cm1
            if imaginary_tolerance_cm1 is not None
            else thermo.get("imaginary_tolerance_cm1", 80.0)
        ),
    }


def _force_max(forces, mask):
    if forces is None:
        return None
    values = np.asarray(forces, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape or not np.isfinite(values).all():
        return None
    return float(np.max(np.linalg.norm(np.where(mask, values, 0.0), axis=1), initial=0.0))


def _geometry_match(first_path, second_path, tolerance_A=1e-5):
    try:
        first, second = read_poscar(first_path), read_poscar(second_path)
        if first.symbols != second.symbols or not np.allclose(first.cell, second.cell, atol=1e-8):
            return False
        distances = [
            float(np.linalg.norm(minimum_image(b - a, first.cell))) for a, b in zip(first.positions, second.positions)
        ]
        return max(distances, default=0.0) <= tolerance_A
    except (OSError, ValueError, IndexError):
        return False


def _provenance_gate(state_dir, stage, result, require_hashes):
    reasons = []
    if result.get("input_hashes_valid") is False:
        reasons.append(stage + ": input hashes mismatch")
    if require_hashes and result.get("input_hashes_valid") is None:
        reasons.append(stage + ": no verified input hashes")
    record = _load(Path(state_dir) / stage / "input_hashes.json")
    expected = record.get("state_json_sha256")
    if expected and expected != file_hash(Path(state_dir) / "state.json"):
        reasons.append(stage + ": state metadata changed after calculation")
    expected_plan = record.get("plan_sha256")
    plan_path = Path(state_dir) / "plan.json"
    if expected_plan and plan_path.exists() and expected_plan != _load(plan_path).get("sha256"):
        reasons.append(stage + ": calculation plan changed")
    attempt = _load(Path(state_dir) / stage / "attempt.json")
    if attempt.get("returncode") not in (None, 0):
        reasons.append(stage + ": VASP launcher returned nonzero")
    if attempt and attempt.get("status") != "complete":
        reasons.append(f"{stage}: the runner has not accepted this attempt (status {attempt.get('status')!r})")
    if (result.get("restart_evidence") or {}).get("fallback"):
        reasons.append(stage + ": VASP reported a fallback of the required CHGCAR/WAVECAR restart")
    for name, expected_hash in attempt.get("output_hashes", {}).items():
        if name in ("WAVECAR", "CHGCAR"):
            continue
        output_path = Path(state_dir) / stage / name
        if isinstance(expected_hash, dict):
            expected_hash = expected_hash.get("sha256")
        if not expected_hash:
            continue
        if not output_path.is_file() or file_hash(output_path) != expected_hash:
            reasons.append(stage + ": accepted output changed or disappeared: " + name)
    return reasons


def _ediffg(result):
    try:
        return float(result.get("incar", {}).get("EDIFFG", "0"))
    except (TypeError, ValueError):
        return 0.0


def _relaxation_reasons(state_dir, stage, result, meta, require_hashes, require_ionic=True):
    reasons = []
    if not result["completed"]:
        reasons.append(stage + ": incomplete")
    if not result["last_electronic_converged"]:
        reasons.append(stage + ": electronic convergence unresolved")
    reasons.extend(_provenance_gate(state_dir, stage, result, require_hashes))
    force = _force_max(result.get("forces_eVA"), meta["relax_mobile"])
    if require_ionic:
        if not result["ionic_converged"]:
            reasons.append(stage + ": ionic convergence unresolved")
        ediffg = _ediffg(result)
        if ediffg >= 0:
            reasons.append(stage + ": no explicit negative EDIFFG force criterion")
        elif force is None or force > -ediffg + 1e-5:
            reasons.append(stage + ": final movable-atom force exceeds criterion or is missing")
    return reasons, force


def _usable(topology):
    """The relaxed structure represents some node: its own, or the one it relaxed into."""
    return bool(topology.get("valid_for_assigned_node") or topology.get("assigned_node_id"))


def _final_validity(state_dir, meta, final, require_hashes, config, registry=None):
    reasons, force = _relaxation_reasons(state_dir, STAGE_FINAL, final, meta, require_hashes)
    topology = topology_check(state_dir, meta, config, geometry=STAGE_FINAL, registry=registry)
    if not _usable(topology):
        reasons.append(STAGE_FINAL + ": relaxed connectivity is unavailable or needs explicit review")
    return reasons, force, topology


def imaginary_mode_character(freq_dir, h_fraction_min=0.5, stretch_min=0.5):
    """Displacement character of every imaginary mode of a displacements-method Hessian.

    H-dominated motion along O-H bonds is proton-transfer-like (the structure is not a
    minimum along a proton coordinate); H motion across the bond is a libration/rotation.
    Returns None when no stored Hessian is available.
    """
    freq_dir = Path(freq_dir)
    report = _load(freq_dir / "frequencies.json")
    hessian, masses, units = report.get("hessian_eV_A2"), report.get("masses_amu"), report.get("units")
    if not hessian or not masses or not units or not (freq_dir / "POSCAR").is_file():
        return None
    central = read_poscar(freq_dir / "POSCAR")
    active = [(int(u["atom_index_1based"]) - 1, "xyz".index(u["axis"])) for u in units[::2]]
    m = np.array([float(masses[central.symbols[a]]) for a, _ in active])
    dynamical = np.asarray(hessian, dtype=float) / np.sqrt(np.outer(m, m))
    eigenvalues, vectors = np.linalg.eigh(0.5 * (dynamical + dynamical.T))
    distances = distance_matrix(central)
    oxygen = [i for i, sym in enumerate(central.symbols) if sym == "O"]
    modes = []
    for k, value in enumerate(eigenvalues):
        if value >= 0:
            break
        pattern = vectors[:, k] / np.sqrt(m)
        disp = {}
        for (atom, axis), amplitude in zip(active, pattern):
            disp.setdefault(atom, np.zeros(3))[axis] = amplitude
        total = sum(float(u @ u) for u in disp.values()) or 1.0
        weights = {atom: float(u @ u) / total for atom, u in disp.items()}
        h_fraction = sum(w for atom, w in weights.items() if central.symbols[atom] == "H")
        along = across = 0.0
        for atom, u in disp.items():
            if central.symbols[atom] != "H" or not oxygen:
                continue
            o = min(oxygen, key=lambda j: distances[atom, j])
            if distances[atom, o] > BOND_CUTOFF["O-H"]:
                continue
            bond = minimum_image(central.positions[atom] - central.positions[o], central.cell)
            unit_bond = bond / np.linalg.norm(bond)
            relative = u - disp.get(o, np.zeros(3))
            along += float(relative @ unit_bond) ** 2
            across += float(relative @ relative)
        stretch = along / across if across > 0 else 0.0
        character = (
            "proton_transfer_like"
            if h_fraction >= h_fraction_min and stretch >= stretch_min
            else "H_libration_or_rotation"
            if h_fraction >= h_fraction_min
            else "heavy_atom_dominated"
        )
        top = sorted(weights.items(), key=lambda x: -x[1])[:3]
        modes.append(
            {
                "cm1": -FREQUENCY_CM1_PER_SQRT_EV_A2_AMU * float(np.sqrt(-value)),
                "character": character,
                "H_fraction": round(h_fraction, 3),
                "OH_stretch_fraction": round(stretch, 3),
                "dominant_atoms": [
                    {"poscar_index_1based": atom + 1, "element": central.symbols[atom], "weight": round(w, 3)}
                    for atom, w in top
                ],
            }
        )
    return modes


def _smearing_warnings(stages, limit_eV=0.02):
    """Flag stages whose smearing entropy term is large enough to matter for reaction energies."""
    warnings = []
    for stage in (STAGE_FINAL, STAGE_REFINE, STAGE_STATIC):
        value = (stages.get(stage) or {}).get("eentro_eV")
        if value is None:
            continue
        magnitude = abs(float(value))
        if magnitude > limit_eV:
            warnings.append(
                f"{stage}: smearing entropy |EENTRO| = {magnitude:.3f} eV per cell exceeds {limit_eV:g} eV; "
                "partial occupancies near the Fermi level make E(sigma->0) depend on SIGMA "
                "(VASP default 0.2 eV when SIGMA is not set); check a smaller SIGMA for this state."
            )
    return warnings


def collect(
    campaign,
    out_dir=None,
    temperature_K=None,
    low_frequency_cutoff_cm1=None,
    imaginary_tolerance_cm1=None,
    registry=None,
):
    """Write state, moment, frequency and QC tables; no DFT is run."""
    campaign = Path(campaign).resolve()
    out_dir = Path(out_dir) if out_dir else campaign / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    config, settings = _settings(campaign, temperature_K, low_frequency_cutoff_cm1, imaginary_tolerance_cm1)
    registry = registry or campaign_registry(campaign, config=config)
    require_hashes = bool(config.get("analysis", {}).get("require_input_hashes", True))
    eentro_limit = float(config.get("analysis", {}).get("max_eentro_eV", 0.02))
    gates = frequency_gates(config)
    max_frequency_force, warn_frequency_force = gates["max_force_eVA"], gates["warn_force_eVA"]
    central = central_mode(config)
    state_rows, moments, frequency_rows = [], [], []
    protocol_notes = set()
    for state_dir in sorted((campaign / "states").glob("*")):
        if not (state_dir / "state.json").is_file():
            continue
        meta = _load(state_dir / "state.json")
        stages = {stage: parse_run(state_dir / stage) for stage in protocol_stages(central)}
        final, static, freq = (stages[s] for s in (STAGE_FINAL, STAGE_STATIC, STAGE_FREQ))
        refine = stages.get(STAGE_REFINE)
        warnings = []
        written = read_incar(state_dir / STAGE_FINAL / "INCAR")
        if written and "SIGMA" not in written and str(written.get("ISMEAR", "1")).strip() in ("0", "-1"):
            protocol_notes.add(
                "SIGMA is not set in the stage INCARs: VASP uses its default SIGMA = 0.2 eV with ISMEAR = "
                + str(written.get("ISMEAR"))
                + ". Keep it for consistency with earlier data, but check EENTRO."
            )
        prepared = _load(state_dir / "plan.json")
        state_mode = plan_mode(prepared) if prepared else central
        mixed = (
            (
                f"plan.json: prepared with frequency.central = {state_mode!r}, but config.json says {central!r}; "
                "its later stages follow the other protocol"
            )
            if state_mode != central
            else None
        )
        final_reasons, final_force, final_topology = _final_validity(
            state_dir, meta, final, require_hashes, config, registry
        )
        final_valid = not final_reasons
        reasons = list(final_reasons)
        frequency_reasons = []
        refine_force = None
        if central == "refine":
            refine_reasons, refine_force = _relaxation_reasons(state_dir, STAGE_REFINE, refine, meta, require_hashes)
            reasons.extend(refine_reasons)
            if mixed:
                reasons.append(mixed)
            static_reasons, central_contcar, central_name = reasons, state_dir / STAGE_REFINE / "CONTCAR", "refined"
        else:
            if final["energy_e0_eV"] is None:
                reasons.append(STAGE_FINAL + ": electronic energy is missing")
            if mixed:
                frequency_reasons.append(mixed)
            static_reasons, central_contcar, central_name = (
                frequency_reasons,
                state_dir / STAGE_FINAL / "CONTCAR",
                STAGE_FINAL,
            )
        if not static["completed"]:
            static_reasons.append(STAGE_STATIC + ": incomplete")
        if not static["electronic_converged"]:
            static_reasons.append(STAGE_STATIC + ": electronic convergence unresolved")
        static_reasons.extend(_provenance_gate(state_dir, STAGE_STATIC, static, require_hashes))
        if not _geometry_match(central_contcar, state_dir / STAGE_STATIC / "POSCAR"):
            static_reasons.append(f"{STAGE_STATIC}: geometry does not match the {central_name} structure")
        topology = topology_check(state_dir, meta, config, registry=registry)
        if not _usable(topology):
            reasons.append("Relaxed connectivity is unavailable or needs explicit review")
        elif topology["assigned_node_id"] != meta["node_id"]:
            warnings.append(
                f"Relaxed into the occupancy of node {topology['assigned_node_id']}"
                + (
                    f" (symmetry image {topology['microstate'].get('via_symmetry_image')})"
                    if (topology.get("microstate") or {}).get("via_symmetry_image")
                    else ""
                )
                + f"; its energy is used for that node, not for {meta['node_id']}."
            )
        if static["energy_e0_eV"] is None:
            static_reasons.append(STAGE_STATIC + ": central electronic energy is missing")
        static_shift = None
        if central == "final":
            same = same_solution(final, static)
            static_shift, moment_change = (
                same["static_minus_final_e0_eV"],
                same["max_moment_change_final_to_static_muB"],
            )
            if static_shift is not None and abs(static_shift) > gates["max_static_e0_shift_eV"]:
                frequency_reasons.append(
                    f"{STAGE_STATIC}: E0 differs from the {STAGE_FINAL} E0 by {static_shift * 1000:+.1f} meV at "
                    "the same geometry and Hamiltonian "
                    f"(frequency.max_static_e0_shift_eV = {gates['max_static_e0_shift_eV']:g}); "
                    f"04_static reached another electronic solution, so the frequencies do not belong to the "
                    f"{STAGE_FINAL} energy"
                )
            if moment_change is not None and moment_change > gates["max_moment_change_muB"]:
                frequency_reasons.append(
                    f"{STAGE_STATIC}: a local moment changed by {moment_change:.2f} muB from {STAGE_FINAL} "
                    f"(frequency.max_moment_change_muB = {gates['max_moment_change_muB']:g}); 04_static reached "
                    f"another electronic solution, so the frequencies do not belong to the {STAGE_FINAL} energy"
                )
        electronic_valid = not reasons
        if not freq["frequency_converged"]:
            frequency_reasons.append(STAGE_FREQ + ": incomplete spectrum or unconverged displacement calculations")
        frequency_reasons.extend(_provenance_gate(state_dir, STAGE_FREQ, freq, require_hashes))
        if not _geometry_match(state_dir / STAGE_STATIC / "POSCAR", state_dir / STAGE_FREQ / "POSCAR"):
            frequency_reasons.append(
                STAGE_FREQ + ": displaced Hessian reference does not match the central static geometry"
            )
        try:
            frequency_atoms = read_poscar(state_dir / STAGE_FREQ / "POSCAR")
            if not np.array_equal(frequency_atoms.mobile, np.asarray(meta["frequency_mobile"], dtype=bool)):
                frequency_reasons.append(
                    STAGE_FREQ + ": active-coordinate mask differs from the recorded atom-ID-based region"
                )
        except (OSError, ValueError, IndexError):
            frequency_reasons.append(STAGE_FREQ + ": cannot verify the active-coordinate mask")
        active_force = _force_max(static.get("forces_eVA"), meta["frequency_mobile"])
        if active_force is None or active_force > max_frequency_force + 1e-5:
            frequency_reasons.append(
                STAGE_STATIC + ": force on a frequency-active atom exceeds the configured threshold or is missing"
            )
        elif active_force > warn_frequency_force:
            warnings.append(
                f"{STAGE_STATIC}: frequency-active force {active_force:.3f} eV/A is within the warning band "
                f"({warn_frequency_force:g}-{max_frequency_force:g} eV/A); soft modes may be less reliable."
            )
        expected_modes = sum(sum(row) for row in meta["frequency_mobile"])
        vibration = vibrational_thermochemistry(freq["frequencies_cm1"], expected_modes=expected_modes, **settings)
        frequency_reasons.extend(vibration["errors"])
        warnings.extend(vibration["warnings"])
        mode_character = None
        if vibration["imaginary_modes"]:
            try:
                mode_character = imaginary_mode_character(state_dir / STAGE_FREQ)
            except (OSError, ValueError, KeyError, IndexError) as exc:
                warnings.append(f"{STAGE_FREQ}: imaginary-mode character unavailable ({exc}).")
            if mode_character is None:
                warnings.append(
                    f"{STAGE_FREQ}: no stored Hessian, so imaginary modes were judged by magnitude only; "
                    "inspect their eigenvectors in OUTCAR."
                )
            else:
                for mode in mode_character:
                    if (
                        mode["character"] == "proton_transfer_like"
                        and abs(mode["cm1"]) <= settings["imaginary_tolerance_cm1"]
                    ):
                        frequency_reasons.append(
                            f"{STAGE_FREQ}: imaginary mode {mode['cm1']:.0f} cm^-1 moves H along its O-H bond "
                            "(proton-transfer-like); the structure is not a minimum along that coordinate, "
                            "so the mode is not regularized"
                        )
                    elif mode["character"] == "heavy_atom_dominated":
                        warnings.append(
                            f"{STAGE_FREQ}: imaginary mode {mode['cm1']:.0f} cm^-1 is dominated by heavy-atom "
                            "motion; check that it is numerical before accepting the floor."
                        )
        magnetic_continuity = {"status": "not_available", "max_change_muB": None}
        history = freq.get("magnetic_moment_history", [])
        reference_moments = static.get("magmoms")
        if history and reference_moments is not None:
            differences = np.asarray(history) - np.asarray(reference_moments)[None, :]
            maximum = float(np.max(np.abs(differences)))
            threshold_muB = gates["max_moment_change_muB"]
            magnetic_continuity = {
                "status": "within_diagnostic_threshold" if maximum <= threshold_muB else "needs_review",
                "max_change_muB": maximum,
                "threshold_muB": threshold_muB,
                "n_moment_snapshots": len(history),
                "note": "Projected-moment continuity diagnostic; it cannot prove identical electronic "
                "occupation patterns.",
            }
            if maximum > threshold_muB:
                frequency_reasons.append(
                    STAGE_FREQ + ": projected local moments changed beyond the configured continuity "
                    "threshold; inspect electronic branch consistency"
                )
        warnings.extend(_smearing_warnings(stages, eentro_limit))
        free_energy_valid = electronic_valid and not frequency_reasons and vibration["valid"]
        has_outputs = any(
            (state_dir / stage / "OUTCAR").exists() or (state_dir / stage / "vasprun.xml").exists() for stage in STAGES
        )
        status = (
            "free_energy_ready"
            if free_energy_valid
            else "electronic_energy_ready"
            if electronic_valid
            else "final_energy_ready"
            if final_valid
            else "needs_review"
            if has_outputs
            else "not_run"
        )
        energy = final["energy_e0_eV"] if central == "final" else static["energy_e0_eV"]
        row = {
            "state_id": meta["id"],
            "node_id": meta["node_id"],
            "assigned_node_id": topology.get("assigned_node_id"),
            "final_assigned_node_id": final_topology.get("assigned_node_id"),
            "microstate_status": topology["status"],
            "node_aliases": meta.get("node_aliases", [meta["node_id"]]),
            "family": meta.get("family"),
            "site": meta.get("site"),
            "variant": meta.get("variant"),
            "status": status,
            "n_atoms": len(meta["atom_ids"]),
            "n_frequency_dof": expected_modes,
            "delta_atoms": meta.get("delta_atoms"),
            "final_valid": final_valid,
            "electronic_valid": electronic_valid,
            "free_energy_valid": free_energy_valid,
            "energy_final_e0_eV": final["energy_e0_eV"],
            "energy_e0_eV": energy,
            "energy_static_e0_eV": static["energy_e0_eV"],
            "static_minus_final_e0_eV": static_shift,
            "F_vib_eV": vibration["F_vib_eV"] if not frequency_reasons else None,
            "G_eV": energy + vibration["F_vib_eV"] if free_energy_valid else None,
            "max_final_mobile_force_eVA": final_force,
            "max_refine_mobile_force_eVA": refine_force,
            "max_static_frequency_force_eVA": active_force,
            "final_exclusion_reasons": list(dict.fromkeys(final_reasons)),
            "electronic_exclusion_reasons": list(dict.fromkeys(reasons)),
            "frequency_exclusion_reasons": list(dict.fromkeys(frequency_reasons)),
            "warnings": list(dict.fromkeys(warnings)),
            "topology": topology,
            "final_topology": final_topology,
            "vibrational": vibration,
            "imaginary_mode_character": mode_character,
            "frequency_magnetic_continuity": magnetic_continuity,
            "stages": stages,
        }
        for stage in (STAGE_FINAL, STAGE_REFINE, STAGE_STATIC):
            values = (stages.get(stage) or {}).get("magmoms")
            if values is None or len(values) != len(meta["atom_ids"]):
                continue
            for index, (atom_id, symbol, value) in enumerate(zip(meta["atom_ids"], meta["symbols"], values)):
                moments.append(
                    {
                        "state_id": meta["id"],
                        "stage": stage,
                        "poscar_index_1based": index + 1,
                        "atom_id": atom_id,
                        "element": symbol,
                        "local_moment_muB": value,
                    }
                )
        if final.get("magmoms") is not None and static.get("magmoms") is not None:
            delta = np.asarray(static["magmoms"]) - np.asarray(final["magmoms"])
            row["max_local_moment_change_final_to_static_muB"] = float(np.max(np.abs(delta), initial=0.0))
            if central == "refine" and row["max_local_moment_change_final_to_static_muB"] > 0.5:
                row["warnings"].append(
                    "A local moment changed by >0.5 muB between final and static stages; inspect "
                    "electronic-state continuity."
                )
        for i, value in enumerate(freq["frequencies_cm1"]):
            frequency_rows.append(
                {
                    "state_id": meta["id"],
                    "mode": i + 1,
                    "frequency_cm1": value,
                    "imaginary": value < 0,
                    "floor_cm1": settings["low_frequency_cutoff_cm1"],
                }
            )
        state_rows.append(row)
    report = {
        "schema_version": 3,
        "campaign": str(campaign),
        "frequency_central": central,
        "settings": settings,
        "energy_source": ENERGY_SOURCE_TEXT[central],
        "vibrational_model": "same inherited active region plus new reactive atoms; partial Hessian",
        "protocol_notes": sorted(protocol_notes),
        "states": state_rows,
    }
    _write(out_dir / "states.json", report)
    _csv(
        out_dir / "states.csv",
        state_rows,
        [
            "state_id",
            "node_id",
            "assigned_node_id",
            "microstate_status",
            "family",
            "site",
            "variant",
            "status",
            "n_atoms",
            "n_frequency_dof",
            "energy_final_e0_eV",
            "energy_e0_eV",
            "energy_static_e0_eV",
            "static_minus_final_e0_eV",
            "F_vib_eV",
            "G_eV",
            "final_valid",
            "electronic_valid",
            "free_energy_valid",
            "max_final_mobile_force_eVA",
            "max_refine_mobile_force_eVA",
            "max_static_frequency_force_eVA",
            "final_exclusion_reasons",
            "electronic_exclusion_reasons",
            "frequency_exclusion_reasons",
            "warnings",
        ],
    )
    _csv(
        out_dir / "magnetic_moments.csv",
        moments,
        ["state_id", "stage", "poscar_index_1based", "atom_id", "element", "local_moment_muB"],
    )
    _csv(out_dir / "frequencies.csv", frequency_rows, ["state_id", "mode", "frequency_cm1", "imaginary", "floor_cm1"])
    qc = {
        "n_states": len(state_rows),
        "n_not_run": sum(r["status"] == "not_run" for r in state_rows),
        "n_reassigned_to_other_node": sum(r["assigned_node_id"] not in (None, r["node_id"]) for r in state_rows),
        "reassigned_states": {
            r["state_id"]: r["assigned_node_id"]
            for r in state_rows
            if r["assigned_node_id"] not in (None, r["node_id"])
        },
        "n_final_valid": sum(r["final_valid"] for r in state_rows),
        "n_electronic_valid": sum(r["electronic_valid"] for r in state_rows),
        "n_free_energy_valid": sum(r["free_energy_valid"] for r in state_rows),
        "settings": settings,
        "protocol_notes": sorted(protocol_notes),
        "frequency_central": central,
        "energy_source": ENERGY_SOURCE_TEXT[central],
        "limitations": [
            "Partial Hessians do not prove stability in frozen degrees of freedom.",
            "Electronic continuation does not prove a global magnetic ground state.",
            "Connectivity diagnostics require review for transformed products.",
        ],
    }
    _write(out_dir / "qc.json", qc)
    return {"out_dir": str(out_dir), "summary": qc, "states": state_rows}
