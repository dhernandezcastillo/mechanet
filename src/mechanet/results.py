"""Conservative extraction of native VASP results (OUTCAR, vasprun.xml) and displacement Hessians.

The OUTCAR timing footer, an energy, and a CONTCAR are different pieces of
evidence.  None by itself establishes convergence.  Frequency energies are
explicitly marked as displaced-geometry values and must not be used as the
central electronic energy in thermochemistry.

``parse_run(<state>/05_freq)`` dispatches on ``05_freq/stage.json``: the
``displacements`` method is assembled from its per-unit single points (see
``compute_displacement_frequencies``); the alternative ``vasp_ibrion5`` method and
every other folder are parsed as one VASP run.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"
RESTART_FILE_NAMES = ("WAVECAR", "CHGCAR")
# sqrt(eV / (Angstrom^2 amu)) expressed as a wavenumber: 1/(2 pi c) * sqrt(e / (1e-20 m^2 * amu)).
FREQUENCY_CM1_PER_SQRT_EV_A2_AMU = 521.4709
CHGCAR_SUCCESS = re.compile(r"charge[- ]density read from file")
CHGCAR_ERROR_MARKERS = ("charge density could not be read", "error reading chgcar")
ATOMIC_START_MARKER = "charge density of overlapping atoms calculated"
WAVECAR_SUCCESS_MARKER = "wavecar file was read successfully"
WAVECAR_FALLBACK_MARKERS = ("wavecar not read", "number of k-points has changed", "initial charge from atoms")


def _float(value):
    try:
        result = float(str(value).replace("D", "E").replace("d", "e"))
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def read_incar(path):
    """Return active, upper-case INCAR assignments, without trusting comments."""
    data = {}
    path = Path(path)
    if not path.exists():
        return data
    for line in path.read_text(errors="replace").splitlines():
        clean = re.split(r"[#!]", line, maxsplit=1)[0]
        for statement in clean.split(";"):
            if "=" in statement:
                key, value = statement.split("=", 1)
                data[key.strip().upper()] = value.strip()
    return data


def _int(data, name, default):
    try:
        return int(float(data[name]))
    except (KeyError, ValueError, TypeError):
        return default


def _active_dof(path):
    """Read only POSCAR's dimensions/flags; no assumptions about surface normal."""
    try:
        lines = Path(path).read_text().splitlines()
        count_line = 6
        try:
            counts = [int(x) for x in lines[count_line].split()]
        except ValueError:  # A VASP4 file is accepted only for counting flags.
            count_line = 5
            counts = [int(x) for x in lines[count_line].split()]
        n_atoms = sum(counts)
        cursor = count_line + 1
        selective = lines[cursor].lstrip().lower().startswith("s")
        if selective:
            cursor += 1
        cursor += 1
        if not selective:
            return n_atoms, [[True, True, True] for _ in range(n_atoms)]
        mask = [[s.upper().startswith("T") for s in line.split()[3:6]] for line in lines[cursor : cursor + n_atoms]]
        if len(mask) != n_atoms or any(len(row) != 3 for row in mask):
            return n_atoms, None
        return n_atoms, mask
    except (OSError, ValueError, IndexError):
        return None, None


def _parse_outcar(path):
    data = dict(
        completed=False,
        ionic_converged=False,
        energy_e0_eV=None,
        energy_free_eV=None,
        forces_eVA=None,
        positions_A=None,
        magmoms=None,
        magnetic_moment_history=[],
        frequencies_cm1=[],
        frequency_mode_indices=[],
        frequency_eigenvectors={},
        force_evaluations=0,
        ediff_markers=0,
        scf_evaluations=[],
        fatal_errors=[],
        total_cpu_seconds=None,
        elapsed_seconds=None,
        nions=None,
        nelm=None,
        ediff=None,
        eentro_eV=None,
        completed_ionic_steps=0,
    )
    if not Path(path).exists():
        return data
    current = None
    force_rows = None
    moment_rows = None
    mode_values = {}
    current_mode = None
    mode_rows = []
    iteration_re = re.compile(r"Iteration\s+(\d+)\(\s*(\d+)\s*\)")
    mode_re = re.compile(r"^\s*(\d+)\s+f(?P<imag>\s*/\s*i)?\s*=.*?" + r"(" + NUMBER + r")\s+cm-1")
    with Path(path).open(errors="replace") as stream:
        for line in stream:
            if "General timing and accounting informations for this job" in line:
                data["completed"] = True
            if "reached required accuracy" in line.lower():
                data["ionic_converged"] = True
            if "FREE ENERGIE OF THE ION-ELECTRON SYSTEM" in line:
                # Printed once per completed ionic step, after its forces.
                data["completed_ionic_steps"] += 1
            if any(
                phrase in line
                for phrase in (
                    "VERY BAD NEWS",
                    "Error EDDDAV",
                    "ZHEGV failed",
                    "EDDDAV:",
                    "BRMIX: very serious problems",
                    "internal error in subroutine",
                    "ZBRENT: fatal error",
                    "ERROR: the triple product",
                    "ERROR FEXCP",
                )
            ):
                data["fatal_errors"].append(line.strip())
            m = iteration_re.search(line)
            if m:
                ionic, electronic = map(int, m.groups())
                if electronic == 1 or current is None:
                    if current is not None:
                        data["scf_evaluations"].append(current)
                    current = {"ionic_index": ionic, "iterations": electronic, "ediff_reached": False}
                else:
                    current["iterations"] = max(current["iterations"], electronic)
            if "aborting loop because EDIFF is reached" in line:
                data["ediff_markers"] += 1
                if current is not None:
                    current["ediff_reached"] = True
            if "NIONS =" in line:
                m = re.search(r"NIONS\s*=\s*(\d+)", line)
                if m:
                    data["nions"] = int(m.group(1))
            if "NELM" in line and "=" in line:
                m = re.search(r"\bNELM\s*=\s*(\d+)", line)
                if m:
                    data["nelm"] = int(m.group(1))
            if "EDIFF" in line and "=" in line:
                m = re.search(r"\bEDIFF\s*=\s*(" + NUMBER + r")", line)
                if m:
                    data["ediff"] = _float(m.group(1))
            if "EENTRO" in line:
                m = re.search(r"EENTRO\s*=\s*(" + NUMBER + r")", line)
                if m:
                    data["eentro_eV"] = _float(m.group(1))
            if "free  energy   TOTEN" in line or "free energy TOTEN" in line:
                m = re.search(r"TOTEN\s*=\s*(" + NUMBER + r")", line)
                if m:
                    data["energy_free_eV"] = _float(m.group(1))
            if "energy(sigma->0)" in line:
                m = re.search(r"energy\(sigma->0\)\s*=\s*(" + NUMBER + r")", line)
                if m:
                    data["energy_e0_eV"] = _float(m.group(1))
            if "TOTAL-FORCE (eV/Angst)" in line:
                force_rows = []
                continue
            if force_rows is not None:
                fields = line.split()
                values = [_float(x) for x in fields]
                if len(values) == 6 and all(x is not None for x in values):
                    force_rows.append(values)
                    continue
                if force_rows:
                    data["positions_A"] = [row[:3] for row in force_rows]
                    data["forces_eVA"] = [row[3:] for row in force_rows]
                    data["force_evaluations"] += 1
                    force_rows = None
            if re.match(r"\s*magnetization\s*\(x\)", line):
                moment_rows = []
                continue
            if moment_rows is not None:
                fields = line.split()
                if fields and fields[0].isdigit() and len(fields) >= 2:
                    value = _float(fields[-1])
                    if value is not None:
                        moment_rows.append(value)
                        continue
                if moment_rows and (line.strip().startswith("tot") or line.strip().startswith("---")):
                    data["magmoms"] = moment_rows
                    data["magnetic_moment_history"].append(moment_rows)
                    moment_rows = None
            m = mode_re.search(line)
            if m:
                if current_mode is not None and mode_rows:
                    data["frequency_eigenvectors"][current_mode] = mode_rows
                index = int(m.group(1))
                value = _float(m.group(3))
                if value is not None:
                    mode_values[index] = -abs(value) if m.group("imag") else value
                    current_mode = index
                    mode_rows = []
                continue
            if current_mode is not None:
                fields = line.split()
                values = [_float(x) for x in fields]
                if len(values) == 6 and all(x is not None for x in values):
                    mode_rows.append(values[3:])
            for label, key in (
                ("Total CPU time used (sec)", "total_cpu_seconds"),
                ("Elapsed time (sec)", "elapsed_seconds"),
            ):
                if label in line:
                    data[key] = _float(line.split(":")[-1])
    if current is not None:
        data["scf_evaluations"].append(current)
    if force_rows:
        data["positions_A"] = [row[:3] for row in force_rows]
        data["forces_eVA"] = [row[3:] for row in force_rows]
        data["force_evaluations"] += 1
    if moment_rows:
        data["magmoms"] = moment_rows
        data["magnetic_moment_history"].append(moment_rows)
    if current_mode is not None and mode_rows:
        data["frequency_eigenvectors"][current_mode] = mode_rows
    data["frequency_mode_indices"] = sorted(mode_values)
    data["frequencies_cm1"] = [mode_values[i] for i in sorted(mode_values)]
    return data


def _parse_xml(path):
    data = {
        "xml_complete": False,
        "parameters": {},
        "calculations": [],
        "energy_e0_eV": None,
        "energy_free_eV": None,
        "forces_eVA": None,
        "warnings": [],
    }
    if not Path(path).exists():
        return data
    try:
        for _event, element in ET.iterparse(path, events=("end",)):
            if element.tag in ("incar", "parameters"):
                for item in element.iter("i"):
                    if item.get("name"):
                        data["parameters"][item.get("name").upper()] = (item.text or "").strip()
                element.clear()
            elif element.tag == "calculation":
                steps = element.findall("scstep")
                data["calculations"].append({"n_scf": len(steps)})
                energies = {i.get("name"): _float(i.text) for i in element.findall("energy/i")}
                if energies.get("e_fr_energy") is not None:
                    free_energy = energies["e_fr_energy"]
                    # VASP calculation/energy E0 is unreliable in some builds;
                    # SCF E0 alone omits classical dispersion.  Reconstruct its
                    # entropy correction on the calculation-level free energy,
                    # following the documented ASE VASP XML reader convention.
                    pressure_kbar = _float(data["parameters"].get("PSTRESS", 0.0)) or 0.0
                    volume = element.find("structure/crystal/i[@name='volume']")
                    if pressure_kbar:
                        volume_A3 = _float(volume.text) if volume is not None else None
                        if volume_A3 is None:
                            data["warnings"].append(
                                "Cannot remove XML PSTRESS enthalpy contribution without a volume; "
                                "XML energy unavailable."
                            )
                            free_energy = None
                        else:
                            free_energy -= pressure_kbar * volume_A3 * 0.0006241509074460763
                    data["energy_free_eV"] = free_energy
                    last_scf = steps[-1] if steps else None
                    scf_energy = (
                        {i.get("name"): _float(i.text) for i in last_scf.findall("energy/i")}
                        if last_scf is not None
                        else {}
                    )
                    if (
                        free_energy is not None
                        and scf_energy.get("e_0_energy") is not None
                        and scf_energy.get("e_fr_energy") is not None
                    ):
                        data["energy_e0_eV"] = free_energy + scf_energy["e_0_energy"] - scf_energy["e_fr_energy"]
                    else:
                        data["energy_e0_eV"] = None
                        data["warnings"].append(
                            "XML lacks the final SCF entropy correction; use OUTCAR E0 rather than an "
                            "unreliable calculation-level XML E0."
                        )
                varray = element.find("varray[@name='forces']")
                if varray is not None:
                    values = [[_float(x) for x in (v.text or "").split()] for v in varray.findall("v")]
                    if values and all(len(row) == 3 and None not in row for row in values):
                        data["forces_eVA"] = values
                element.clear()
            elif element.tag == "modeling":
                data["xml_complete"] = True
    except (ET.ParseError, OSError) as exc:
        data["warnings"].append("vasprun.xml is incomplete/unreadable: " + str(exc))
    return data


def _verify_input_hashes(directory):
    """Check recorded input files.  Restart-file copies are verified only before launch."""
    path = directory / "input_hashes.json"
    if not path.exists():
        return None, ["No input_hashes.json: input/output provenance is not verified."]
    try:
        manifest = json.loads(path.read_text())
        files = manifest.get("files", manifest)
        mismatches = []
        for name, expected in files.items():
            if not isinstance(expected, str) or name in RESTART_FILE_NAMES:
                continue
            target = directory / name
            if not target.exists():
                mismatches.append(name + " is missing")
                continue
            hasher = hashlib.sha256()
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            if hasher.hexdigest() != expected:
                mismatches.append(name + " changed after the recorded run inputs")
        return not mismatches, mismatches
    except (ValueError, OSError) as exc:
        return False, ["Cannot verify input provenance: " + str(exc)]


def required_restart(incar):
    """Which restart file a run reads according to its INCAR (None for a fresh start)."""
    if _int(incar, "ISTART", 0) >= 1:
        return "WAVECAR"
    if _int(incar, "ICHARG", 2) in (1, 11):
        return "CHGCAR"
    return None


def restart_evidence(folder, required):
    """Inspect the VASP startup report for the restart actually used.

    Output wording differs among VASP builds, so absent positive wording is
    recorded as unverified rather than invented as a successful read.  An
    atomic-density start is flagged only when no successful read is reported.
    """
    if required not in ("CHGCAR", "WAVECAR"):
        return {"required": None, "verified": None, "fallback": False, "markers": []}
    text = ""
    for name in ("stdout.log", "OUTCAR"):
        path = Path(folder) / name
        if path.exists():
            with path.open(errors="replace") as f:
                text += f.read(300000) + "\n"
    lower = text.lower()
    atomic = ATOMIC_START_MARKER in lower
    if required == "CHGCAR":
        success = bool(CHGCAR_SUCCESS.search(lower))
        markers = [m for m in CHGCAR_ERROR_MARKERS if m in lower]
    else:
        success = WAVECAR_SUCCESS_MARKER in lower
        markers = [m for m in WAVECAR_FALLBACK_MARKERS if m in lower]
    if atomic and not success:
        markers.append(ATOMIC_START_MARKER)
    return {"required": required, "verified": True if success else None, "fallback": bool(markers), "markers": markers}


def _empty_xml():
    return {
        "xml_complete": False,
        "parameters": {},
        "calculations": [],
        "energy_e0_eV": None,
        "energy_free_eV": None,
        "forces_eVA": None,
        "warnings": [],
    }


def is_displacement_driver(directory):
    path = Path(directory) / "stage.json"
    try:
        return path.is_file() and json.loads(path.read_text()).get("method") == "displacements"
    except (OSError, ValueError, AttributeError):
        return False


def parse_run(directory, include_xml=True):
    """Parse one VASP stage and return JSON-safe results with explicit QC.

    ``electronic_converged`` is deliberately aggregate over all recorded energy
    evaluations.  ``last_electronic_converged`` allows a caller to distinguish an
    ionic relaxation that recovered from an earlier SCF failure.  Frequency runs
    require all displacement SCFs and the full selected Hessian spectrum.
    """
    directory = Path(directory)
    if is_displacement_driver(directory):
        return _parse_displacement_driver(directory)
    incar = read_incar(directory / "INCAR")
    out = _parse_outcar(directory / "OUTCAR")
    xml = _parse_xml(directory / "vasprun.xml") if include_xml else _empty_xml()
    params = dict(xml["parameters"])
    params.update(incar)
    nions, mask = _active_dof(directory / "POSCAR")
    nions = nions or out["nions"]
    nelm = _int(params, "NELM", out["nelm"] or 60)
    ediff = _float(params.get("EDIFF", out["ediff"] or 1e-4))
    ibrion = _int(params, "IBRION", -1)
    warnings = list(xml["warnings"])
    evaluations = out["scf_evaluations"]
    flags = []
    convergence_evidence = "missing"
    if evaluations:
        flags = [bool(e["ediff_reached"]) for e in evaluations]
        convergence_evidence = "OUTCAR explicit EDIFF termination for each evaluation"
        if not any(flags) and xml["calculations"]:
            # VASP XML carries no explicit EDIFF boolean.  An SCF loop that
            # terminated before NELM in a completed calculation is the VASP
            # control-flow evidence used here; NELM termination is never accepted.
            flags = [0 < e["n_scf"] < nelm for e in xml["calculations"]]
            convergence_evidence = "vasprun.xml SCF count below NELM"
    elif xml["calculations"]:
        flags = [0 < e["n_scf"] < nelm for e in xml["calculations"]]
        if out["ediff_markers"] >= len(flags):
            flags = [True] * len(flags)
            convergence_evidence = "OUTCAR EDIFF markers cover XML evaluations"
        else:
            convergence_evidence = "vasprun.xml SCF count below NELM"
    elif out["force_evaluations"]:
        count = out["force_evaluations"]
        flags = [True] * count if out["ediff_markers"] >= count else [False] * count
        convergence_evidence = "OUTCAR EDIFF markers cover force evaluations"
    elif out["ediff_markers"]:
        flags = [True] * out["ediff_markers"]
        convergence_evidence = "OUTCAR explicit EDIFF termination"
    if not ediff or ediff <= 0:
        flags = [False for _ in flags]
        warnings.append("EDIFF is nonpositive/missing; SCF convergence is not established.")
    # A normal OUTCAR footer is required if OUTCAR exists; complete XML alone
    # is accepted only when OUTCAR is absent (e.g. a deliberate XML-only import).
    completed = bool(out["completed"] if (directory / "OUTCAR").exists() else xml["xml_complete"])
    if out["fatal_errors"]:
        completed = False
        warnings.extend(out["fatal_errors"])
    if not completed:
        warnings.append("No clean VASP completion evidence.")
    if not flags:
        warnings.append("Electronic convergence could not be established.")
    elif not all(flags):
        warnings.append("One or more electronic evaluations did not establish EDIFF convergence.")
    energy_e0 = out["energy_e0_eV"]
    energy_free = out["energy_free_eV"]
    if energy_e0 is None:
        energy_e0 = xml["energy_e0_eV"]
    if energy_free is None:
        energy_free = xml["energy_free_eV"]
    forces = out["forces_eVA"] or xml["forces_eVA"]
    if forces is not None and nions is not None and len(forces) != nions:
        warnings.append("Force array length does not match POSCAR atom count.")
        forces = None
    magmoms = out["magmoms"]
    if magmoms is not None and nions is not None and len(magmoms) != nions:
        warnings.append("Local magnetic-moment array length does not match POSCAR.")
        magmoms = None
    expected_modes = sum(sum(row) for row in mask) if mask is not None else None
    frequency_run = ibrion in (5, 6, 7, 8)
    modes = out["frequencies_cm1"]
    output_mode_indices = out["frequency_mode_indices"]
    retained_mode_indices = output_mode_indices[:]
    excluded_frozen_modes = []
    # Some VASP builds print the full 3N spectrum, including exactly frozen
    # coordinates.  Exclude a mode only when its eigenvector proves it is
    # confined to frozen degrees of freedom AND its frequency is numerically
    # zero.  A frequency-only zero cutoff would erase genuine soft active modes.
    if (
        frequency_run
        and nions is not None
        and mask is not None
        and len(modes) == 3 * nions
        and expected_modes < 3 * nions
    ):
        retained = []
        for mode_index, value in zip(output_mode_indices, modes):
            vector = out["frequency_eigenvectors"].get(mode_index)
            purely_frozen = False
            if vector and len(vector) == nions:
                total = sum(x * x for row in vector for x in row)
                active = sum(
                    x * x for row, flags_row in zip(vector, mask) for x, movable in zip(row, flags_row) if movable
                )
                purely_frozen = total > 1e-14 and active / total < 1e-10
            if purely_frozen and abs(value) < 1e-3:
                excluded_frozen_modes.append(mode_index)
            else:
                retained.append((mode_index, value))
        retained_mode_indices = [x[0] for x in retained]
        modes = [x[1] for x in retained]
        if excluded_frozen_modes:
            warnings.append(
                f"Excluded {len(excluded_frozen_modes)} zero-frequency modes whose eigenvectors are confined "
                "to frozen coordinates."
            )
    raw_index_complete = output_mode_indices == list(range(1, len(output_mode_indices) + 1))
    spectrum_complete = bool(
        frequency_run and expected_modes is not None and len(modes) == expected_modes and raw_index_complete
    )
    minimum_displacements = None
    displacement_coverage = None
    if frequency_run and ibrion == 5 and expected_modes is not None:
        nfree = _int(params, "NFREE", 2)
        if nfree in (1, 2, 4):
            minimum_displacements = nfree * expected_modes
            # VASP may additionally evaluate the undisplaced geometry.  Require
            # at least all displacements, never assume a fixed extra count.
            displacement_coverage = len(flags) >= minimum_displacements
            if not displacement_coverage:
                warnings.append("Not all finite-difference displacement SCFs were found.")
    if frequency_run and not spectrum_complete:
        warnings.append(
            f"Frequency spectrum has {len(modes)} modes; expected {expected_modes} selected degrees of freedom."
        )
    hashes_valid, hash_warnings = _verify_input_hashes(directory)
    warnings.extend(hash_warnings)
    if hashes_valid is False:
        completed = False
    all_scf = bool(flags and all(flags))
    freq_ok = bool(completed and all_scf and spectrum_complete and displacement_coverage is not False)
    if out["completed_ionic_steps"]:
        completed_steps = out["completed_ionic_steps"]
    elif out["force_evaluations"]:
        completed_steps = out["force_evaluations"]
    else:
        completed_steps = len(xml["calculations"])
    evidence = restart_evidence(directory, required_restart(incar))
    result = {
        "directory": str(directory),
        "completed": completed,
        "electronic_converged": all_scf,
        "all_electronic_converged": all_scf,
        "last_electronic_converged": bool(flags and flags[-1]),
        "ionic_converged": bool(out["ionic_converged"]),
        "energy_e0_eV": energy_e0,
        "energy_free_eV": energy_free,
        "eentro_eV": out["eentro_eV"],
        "energy_context": "last_displaced_geometry_DO_NOT_USE_FOR_CHE"
        if frequency_run
        else "central_or_final_ionic_geometry",
        "forces_eVA": forces,
        "positions_A": out["positions_A"],
        "magmoms": magmoms,
        "magnetic_moment_history": [
            row for row in out["magnetic_moment_history"] if nions is not None and len(row) == nions
        ],
        "frequencies_cm1": modes,
        "retained_frequency_mode_indices": retained_mode_indices,
        "excluded_frozen_mode_indices": excluded_frozen_modes,
        "frequency_converged": freq_ok,
        "frequency_spectrum_complete": spectrum_complete,
        "expected_frequency_modes": expected_modes if frequency_run else None,
        "frequency_method": "vasp_ibrion5" if frequency_run else None,
        "n_displacements_expected": minimum_displacements,
        "n_displacements_done": max(out["force_evaluations"] - 1, 0) if frequency_run else None,
        "minimum_displacement_evaluations": minimum_displacements,
        "displacement_coverage": displacement_coverage,
        "n_ionic_steps": len(flags),
        "n_completed_ionic_steps": completed_steps,
        "n_scf_iterations": sum(e["iterations"] for e in evaluations)
        if evaluations
        else sum(e["n_scf"] for e in xml["calculations"]),
        "n_scf_evaluations": len(flags),
        "scf_convergence_flags": flags,
        "convergence_evidence": convergence_evidence,
        "fatal_errors": list(out["fatal_errors"]),
        "restart_evidence": evidence,
        "stopcar_present": (directory / "STOPCAR").exists(),
        "n_atoms": nions,
        "input_hashes_valid": hashes_valid,
        "total_cpu_seconds": out["total_cpu_seconds"],
        "elapsed_seconds": out["elapsed_seconds"],
        "incar": params,
        "warnings": list(dict.fromkeys(warnings)),
    }
    if forces is not None and mask is not None:
        result["max_mobile_force_eVA"] = max(
            (
                math.sqrt(sum(value * value for value, movable in zip(row, flags_row) if movable))
                for row, flags_row in zip(forces, mask)
            ),
            default=0.0,
        )
    else:
        result["max_mobile_force_eVA"] = None
    return result


# ---------------------------------------------------------------------------
# Restartable finite-difference frequencies (05_freq, method "displacements")
# ---------------------------------------------------------------------------


def same_solution(final_result, static_result):
    """How far a single point is from the relaxation it repeats (parse_run results of both).

    Returns the E0 shift (static - final, eV) and the largest change of a local moment (muB),
    None where either value is unavailable. With frequency.central = "final", 04_static repeats
    the converged 02_final point with the same geometry and Hamiltonian, so both should be small.
    """
    shift = change = None
    if final_result.get("energy_e0_eV") is not None and static_result.get("energy_e0_eV") is not None:
        shift = float(static_result["energy_e0_eV"]) - float(final_result["energy_e0_eV"])
    before, after = final_result.get("magmoms"), static_result.get("magmoms")
    if before is not None and after is not None and len(before) == len(after):
        change = float(np.max(np.abs(np.asarray(after, dtype=float) - np.asarray(before, dtype=float)), initial=0.0))
    return {"static_minus_final_e0_eV": shift, "max_moment_change_final_to_static_muB": change}


def displacement_units(structure, displacement_A):
    """Deterministic +h/-h unit list over the selective-dynamics (frequency) mask.

    Units are ordered by POSCAR atom, Cartesian axis, then sign; ``dof`` indexes
    the active Cartesian coordinates in that order.
    """
    h = float(displacement_A)
    width = max(3, len(str(len(structure.symbols))))
    units, dof = [], 0
    for atom, flags in enumerate(np.asarray(structure.mobile, dtype=bool)):
        for axis, movable in enumerate(flags):
            if not movable:
                continue
            for sign, label in ((1, "plus"), (-1, "minus")):
                units.append(
                    {
                        "id": f"u{dof + 1:03d}_atom{atom + 1:0{width}d}_{'xyz'[axis]}_{label}",
                        "dof": dof,
                        "atom": atom,
                        "axis": axis,
                        "sign": sign,
                        "displacement_A": sign * h,
                    }
                )
            dof += 1
    return units


def displaced_structure(central, unit):
    displaced = central.copy()
    displaced.positions[unit["atom"], unit["axis"]] += unit["displacement_A"]
    return displaced


def displacement_error(central, displaced, unit, tolerance_A=1e-8):
    """None if ``displaced`` differs from ``central`` by exactly the unit's +/-h; else a reason."""
    if displaced.symbols != central.symbols or displaced.positions.shape != central.positions.shape:
        return "atom list differs from the central geometry"
    if not np.allclose(displaced.cell, central.cell, rtol=0, atol=1e-10):
        return "cell differs from the central geometry"
    expected = np.zeros_like(central.positions)
    expected[unit["atom"], unit["axis"]] = unit["displacement_A"]
    error = float(np.max(np.abs(displaced.positions - central.positions - expected)))
    return None if error <= tolerance_A else f"displacement deviates from the intended +/-h by {error:.3e} A"


def mass_table(potcar_path, species):
    """Masses (amu) per element from POTCAR POMASS, with a recorded standard-mass fallback."""
    from .vasp import potcar_records

    try:
        records = potcar_records(potcar_path)
        elements = [element for element, _ in records]
        masses = [mass for _, mass in records]
        if elements == list(species) and all(m is not None and m > 0 for m in masses):
            return dict(zip(elements, masses)), "POTCAR POMASS", []
        reason = f"POTCAR datasets {elements} do not match the POSCAR species {list(species)} or lack POMASS"
    except OSError as exc:
        reason = f"POTCAR unavailable ({exc})"
    from .elements import standard_mass

    return (
        {s: standard_mass(s) for s in species},
        "standard atomic masses",
        [f"Used standard atomic masses: {reason}."],
    )


def frequencies_from_hessian(hessian, masses_amu):
    """Signed harmonic wavenumbers (cm^-1, imaginary negative) from a Cartesian Hessian in eV/A^2."""
    hessian = np.asarray(hessian, dtype=float)
    m = np.asarray(masses_amu, dtype=float)
    dynamical = hessian / np.sqrt(np.outer(m, m))
    eigenvalues = np.linalg.eigvalsh(0.5 * (dynamical + dynamical.T))
    values = [math.copysign(FREQUENCY_CM1_PER_SQRT_EV_A2_AMU * math.sqrt(abs(x)), x) if x else 0.0 for x in eigenvalues]
    return sorted(values, reverse=True), [float(x) for x in eigenvalues]


def unit_status(unit_dir, central, unit):
    """Parse one displacement unit and decide whether it is a usable force evaluation."""
    from .structure import read_poscar

    unit_dir = Path(unit_dir)
    if not (unit_dir / "OUTCAR").exists() and not (unit_dir / "vasprun.xml").exists():
        return {"id": unit["id"], "done": False, "reason": "not run", "result": None}
    result = parse_run(unit_dir, include_xml=False)
    reasons = []
    if not result["completed"]:
        reasons.append("no clean completion")
    if not result["electronic_converged"]:
        reasons.append("SCF not converged")
    if result["input_hashes_valid"] is False:
        reasons.append("input hash mismatch")
    if result["forces_eVA"] is None or len(result["forces_eVA"]) != len(central.symbols):
        reasons.append("forces missing")
    if result["restart_evidence"]["fallback"]:
        reasons.append("WAVECAR restart fallback reported")
    try:
        error = displacement_error(central, read_poscar(unit_dir / "POSCAR"), unit)
    except (OSError, ValueError, IndexError) as exc:
        error = f"unreadable displaced POSCAR ({exc})"
    if error:
        reasons.append(error)
    return {"id": unit["id"], "done": not reasons, "reason": "; ".join(reasons) or "accepted", "result": result}


def compute_displacement_frequencies(freq_dir):
    """Assemble the central-difference Hessian of a displacements-method 05_freq folder.

    H_ij = -(F_i(+h_j) - F_i(-h_j)) / (2h) over the active Cartesian DOF, then
    symmetrized and mass-weighted with POTCAR POMASS (standard masses as a
    recorded fallback); nu = 521.4709 sqrt(lambda) cm^-1, negative for lambda < 0.
    """
    from .structure import read_poscar

    freq_dir = Path(freq_dir)
    spec = json.loads((freq_dir / "stage.json").read_text())
    h = float(spec["displacement_A"])
    central = read_poscar(freq_dir / "POSCAR")
    units = displacement_units(central, h)
    statuses = [unit_status(freq_dir / "units" / u["id"], central, u) for u in units]
    species = list(dict.fromkeys(central.symbols))
    report = {
        "schema_version": 1,
        "method": "displacements",
        "displacement_A": h,
        "nfree": 2,
        "n_active_dof": len(units) // 2,
        "n_displacements_expected": len(units),
        "n_displacements_done": sum(s["done"] for s in statuses),
        "units": [
            {
                "id": u["id"],
                "dof": u["dof"],
                "atom_index_1based": u["atom"] + 1,
                "axis": "xyz"[u["axis"]],
                "displacement_A": u["displacement_A"],
                "done": s["done"],
                "status": s["reason"],
            }
            for u, s in zip(units, statuses)
        ],
        "frequencies_cm1": [],
        "eigenvalues_eV_A2_amu": [],
        "hessian_eV_A2": None,
        "masses_amu": None,
        "mass_source": None,
        "max_hessian_asymmetry_eV_A2": None,
        "warnings": [],
        "complete": False,
    }
    if not units:
        # No frequency-active atom (a clean metal reference): an empty, complete spectrum.
        report["warnings"].append("The frequency mask selects no active coordinates: F_vib = 0 by construction.")
        report.update(frequencies_cm1=[], eigenvalues_eV_A2_amu=[], hessian_eV_A2=[], complete=True)
        return report, statuses
    if report["n_displacements_done"] != len(units):
        return report, statuses
    forces = {s["id"]: np.asarray(s["result"]["forces_eVA"], dtype=float) for s in statuses}
    active = [(u["atom"], u["axis"]) for u in units[::2]]
    hessian = np.zeros((len(active), len(active)))
    for j, (plus, minus) in enumerate(zip(units[::2], units[1::2])):
        delta = forces[plus["id"]] - forces[minus["id"]]
        hessian[:, j] = -np.array([delta[a, k] for a, k in active]) / (2.0 * h)
    asymmetry = float(np.max(np.abs(hessian - hessian.T))) if hessian.size else 0.0
    symmetric = 0.5 * (hessian + hessian.T)
    masses, source, mass_warnings = mass_table(freq_dir / "POTCAR", species)
    per_dof = [masses[central.symbols[a]] for a, _ in active]
    frequencies, eigenvalues = frequencies_from_hessian(symmetric, per_dof)
    report.update(
        frequencies_cm1=frequencies,
        eigenvalues_eV_A2_amu=eigenvalues,
        hessian_eV_A2=symmetric.tolist(),
        masses_amu=masses,
        mass_source=source,
        max_hessian_asymmetry_eV_A2=asymmetry,
        complete=True,
    )
    report["warnings"].extend(mass_warnings)
    return report, statuses


def _parse_displacement_driver(directory):
    warnings = []
    if not (directory / "POSCAR").is_file():
        report, statuses = None, []
        warnings.append("Displacement set not started (no central 05_freq/POSCAR yet).")
    else:
        try:
            report, statuses = compute_displacement_frequencies(directory)
        except (OSError, ValueError, KeyError, IndexError) as exc:
            report, statuses = None, []
            warnings.append(f"Cannot assemble displacement frequencies: {exc}")
    incar = read_incar(directory / "INCAR")
    hashes_valid, hash_warnings = _verify_input_hashes(directory)
    warnings.extend(hash_warnings)
    unit_hashes = [s["result"]["input_hashes_valid"] for s in statuses if s["result"] is not None]
    if hashes_valid is not False and any(v is False for v in unit_hashes):
        hashes_valid = False
        warnings.append("A displacement unit's inputs changed after its calculation.")
    done = [s for s in statuses if s["done"]]
    expected = report["n_displacements_expected"] if report else None
    expected_modes = report["n_active_dof"] if report else None
    complete = bool(report and report["complete"] and hashes_valid is not False)
    ran = [s for s in statuses if s["result"] is not None]
    all_scf = bool(
        report
        and len(ran) == len(statuses)
        and (statuses or report["n_displacements_expected"] == 0)
        and all(s["result"]["electronic_converged"] for s in ran)
    )
    frequencies = report["frequencies_cm1"] if report and report["complete"] else []
    spectrum_complete = bool(report and report["complete"] and len(frequencies) == expected_modes)
    if report:
        warnings.extend(report["warnings"])
        if not complete:
            pending = [s["id"] + ": " + s["reason"] for s in statuses if not s["done"]]
            warnings.append(
                f"{len(done)}/{expected} displacement single points are usable; first pending: "
                + "; ".join(pending[:3])
            )
        if report["max_hessian_asymmetry_eV_A2"] is not None and report["max_hessian_asymmetry_eV_A2"] > 0.5:
            warnings.append(
                "Finite-difference Hessian is strongly asymmetric "
                f"({report['max_hessian_asymmetry_eV_A2']:.3f} eV/A^2); inspect unit convergence."
            )
    moments = [s["result"]["magmoms"] for s in done if s["result"]["magmoms"] is not None]
    nions = len(done[0]["result"]["forces_eVA"]) if done else None
    last = done[-1]["result"] if done else None
    return {
        "directory": str(directory),
        "completed": complete,
        "electronic_converged": all_scf,
        "all_electronic_converged": all_scf,
        "last_electronic_converged": all_scf,
        "ionic_converged": False,
        "energy_e0_eV": None,
        "energy_free_eV": None,
        "eentro_eV": last["eentro_eV"] if last else None,
        "energy_context": "displaced_geometries_DO_NOT_USE_FOR_CHE",
        "forces_eVA": None,
        "positions_A": None,
        "magmoms": None,
        "magnetic_moment_history": moments,
        "frequencies_cm1": frequencies,
        "retained_frequency_mode_indices": list(range(1, len(frequencies) + 1)),
        "excluded_frozen_mode_indices": [],
        "frequency_converged": bool(complete and all_scf and spectrum_complete),
        "frequency_spectrum_complete": spectrum_complete,
        "expected_frequency_modes": expected_modes,
        "frequency_method": "displacements",
        "n_displacements_expected": expected,
        "n_displacements_done": len(done),
        "minimum_displacement_evaluations": expected,
        "displacement_coverage": bool(expected and len(done) == expected),
        "n_ionic_steps": len(ran),
        "n_completed_ionic_steps": len(done),
        "n_scf_iterations": sum(s["result"]["n_scf_iterations"] for s in ran),
        "n_scf_evaluations": len(ran),
        "scf_convergence_flags": [bool(s["result"]["electronic_converged"]) for s in ran],
        "convergence_evidence": "per-unit OUTCAR EDIFF termination",
        "fatal_errors": [e for s in ran for e in s["result"]["fatal_errors"]],
        "restart_evidence": {
            "required": "WAVECAR",
            "verified": None,
            "fallback": any(s["result"]["restart_evidence"]["fallback"] for s in ran),
            "markers": [],
        },
        "stopcar_present": False,
        "n_atoms": nions,
        "input_hashes_valid": hashes_valid,
        "total_cpu_seconds": sum((s["result"]["total_cpu_seconds"] or 0.0) for s in ran) or None,
        "elapsed_seconds": sum((s["result"]["elapsed_seconds"] or 0.0) for s in ran) or None,
        "incar": incar,
        "max_mobile_force_eVA": None,
        "mass_source": report["mass_source"] if report else None,
        "units": report["units"] if report else [],
        "warnings": list(dict.fromkeys(warnings)),
    }
