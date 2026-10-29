"""Molecular references: the template file, ready-to-run molecule inputs, and reading results back.

The computational hydrogen electrode needs mu(H2) and mu(H2O) (and mu(Cl2) for CER), each
E0 + G_corr from a molecule computed with the slab's functional and settings. mechanet never
substitutes literature values; these helpers make the user's own values easy to produce:

1. ``mechanet references --molecules DIR`` writes, for each molecule, ``relax/`` (POSCAR in a
   cubic box, INCAR derived from the campaign's INCAR_final, Gamma-point KPOINTS) and
   ``freq/`` (the same Hamiltonian with IBRION = 5).
2. Run ``relax``, copy its CONTCAR to ``freq/POSCAR``, run ``freq``.
3. ``mechanet references --collect DIR`` reads E0 and the frequencies and writes E0 and
   G_corr = ZPE + int Cp dT - TS (ideal gas, 1 bar, ASE thermochemistry) into the references file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fileio import read_yaml, write_yaml
from .structure import Structure, write_poscar
from .thermo import protocol_documented
from .vasp import read_incar, render_incar

CM1_TO_EV = 1.239841984e-4
MOLECULES = {
    "H2": {
        "symbols": ["H", "H"],
        "positions": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]],
        "geometry": "linear",
        "symmetry_number": 2,
        "spin": 0,
    },
    "H2O": {
        "symbols": ["O", "H", "H"],
        "positions": [[0.0, 0.0, 0.0], [0.7586, 0.5865, 0.0], [-0.7586, 0.5865, 0.0]],
        "geometry": "nonlinear",
        "symmetry_number": 2,
        "spin": 0,
    },
    "Cl2": {
        "symbols": ["Cl", "Cl"],
        "positions": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.99]],
        "geometry": "linear",
        "symmetry_number": 2,
        "spin": 0,
    },
}
REACTION_MOLECULES = {"oer": ("H2", "H2O"), "cer": ("H2", "H2O", "Cl2")}
DROP_TAGS = (
    "LDAU",
    "LDAUTYPE",
    "LDAUL",
    "LDAUU",
    "LDAUJ",
    "LDAUPRINT",
    "MAGMOM",
    "LDIPOL",
    "IDIPOL",
    "DIPOL",
    "KPAR",
    "NSW",
    "IBRION",
    "ISIF",
    "POTIM",
    "NFREE",
    "EDIFFG",
    "ICHARG",
    "ISTART",
    "LMAXMIX",
)


def write_template(path, reaction="oer", force=False):
    from .project import references_template

    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; pass --force to overwrite it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(references_template(reaction))
    return path


def _molecule_incar(template, stage, protocol=None):
    values = {k: v for k, v in read_incar(template).items() if k not in DROP_TAGS}
    # the slab's effective protocol tags (vasp.common_overrides and stage_overrides included)
    values.update({k: str(v) for k, v in (protocol or {}).items() if v is not None})
    values.update(
        ISPIN="1",
        ISMEAR="0",
        SIGMA="0.01",
        ISYM="0",
        LREAL=".FALSE.",
        PREC="Accurate",
        LWAVE=".FALSE.",
        LCHARG=".FALSE.",
        ISTART="0",
        ICHARG="2",
    )
    if stage == "relax":
        values.update(IBRION="2", ISIF="2", NSW="100", EDIFF="1E-6", EDIFFG="-0.01")
    else:
        values.update(IBRION="5", NFREE="2", POTIM="0.015", NSW="1", EDIFF="1E-7")
    return values


def molecule_inputs(incar_template, out_dir, molecules=("H2", "H2O"), box_A=15.0, potcars=None, protocol=None):
    """Write relax/ and freq/ inputs for each molecule; returns the folders written.

    ``incar_template``: the slab's INCAR_final (functional, ENCUT, dispersion and solvation
    tags are kept; +U, moments, dipole and ionic tags are replaced). ``protocol``: protocol
    tags that replace the template's (the slab's overrides; see :func:`slab_protocol`).
    ``potcars``: optional element -> POTCAR path map, to also write POTCAR files.
    """
    out_dir = Path(out_dir)
    written = []
    for name in molecules:
        if name not in MOLECULES:
            raise ValueError(f"No built-in geometry for {name}; known: {sorted(MOLECULES)}")
        spec = MOLECULES[name]
        positions = np.array(spec["positions"], dtype=float)
        positions += 0.5 * box_A - positions.mean(axis=0)
        species = list(dict.fromkeys(spec["symbols"]))
        order = sorted(range(len(spec["symbols"])), key=lambda i: species.index(spec["symbols"][i]))
        structure = Structure(
            [spec["symbols"][i] for i in order],
            positions[order],
            np.eye(3) * box_A,
            np.ones((len(order), 3), dtype=bool),
            [f"a{i + 1:04d}" for i in range(len(order))],
        )
        for stage in ("relax", "freq"):
            folder = out_dir / name / stage
            folder.mkdir(parents=True, exist_ok=True)
            if stage == "relax":
                write_poscar(folder / "POSCAR", structure, comment=f"{name} in a {box_A:g} A box (mechanet references)")
            values = _molecule_incar(incar_template, stage, protocol)
            values["SYSTEM"] = f"{name}_{stage}"
            (folder / "INCAR").write_text(
                "# mechanet references: molecule "
                + stage
                + ", derived from "
                + Path(incar_template).name
                + (" with the slab's protocol overrides" if protocol else "")
                + "\n"
                + render_incar(values)
            )
            (folder / "KPOINTS").write_text("Gamma point\n0\nGamma\n1 1 1\n0 0 0\n")
            if potcars:
                missing = [s for s in species if s not in potcars]
                if missing:
                    raise ValueError(f"No POTCAR for {missing}")
                (folder / "POTCAR").write_bytes(b"".join(Path(potcars[s]).read_bytes() for s in species))
        (out_dir / name / "README.txt").write_text(
            f"{name}: 1) run VASP in relax/  2) cp relax/CONTCAR freq/POSCAR  3) run VASP in freq/\n"
            "Then: mechanet references --collect "
            + str(out_dir)
            + "\n"
            + (
                ""
                if potcars
                else "POTCAR: concatenate the PAW datasets of the slab in the POSCAR species order "
                f"({' '.join(species)}).\n"
            )
        )
        written.append(str(out_dir / name))
    return written


def gas_correction(name, frequencies_cm1, temperature_K=298.15, pressure_bar=1.0):
    """ZPE + int Cp dT - TS (eV) of an ideal-gas molecule from its harmonic frequencies.

    The 3N - 6 (3N - 5 for a linear molecule) highest real modes are the vibrations; the
    translations and rotations (near zero, real or imaginary) are dropped here, in any input
    order, so the result does not depend on how the ASE version sorts or filters modes.
    """
    from ase import Atoms
    from ase.thermochemistry import IdealGasThermo

    spec = MOLECULES[name]
    atoms = Atoms(spec["symbols"], positions=spec["positions"])
    count = 3 * len(atoms) - (5 if spec["geometry"] == "linear" else 6)
    real = sorted(float(f) for f in frequencies_cm1 if float(f) > 0)
    if len(real) < count:
        raise ValueError(
            f"{name}: {count} vibrational modes expected, only {len(real)} real modes found "
            f"({len(list(frequencies_cm1)) - len(real)} imaginary or zero)"
        )
    energies = [f * CM1_TO_EV for f in real[-count:]]
    thermo = IdealGasThermo(
        energies,
        spec["geometry"],
        potentialenergy=0.0,
        atoms=atoms,
        symmetrynumber=spec["symmetry_number"],
        spin=spec["spin"],
    )
    return float(thermo.get_gibbs_energy(temperature_K, pressure_bar * 1e5, verbose=False))


# INCAR tags that define the electronic-structure protocol of the molecules (compared with
# the slab's by `references --collect`; +U, moments and ionic tags differ on purpose).
PROTOCOL_TAGS = ("GGA", "METAGGA", "LHFCALC", "HFSCREEN", "AEXX", "IVDW", "ENCUT", "LASPH", "LSOL", "EB_K")
LOGICAL_TAGS = ("LHFCALC", "LASPH", "LSOL")
# VASP defaults of unset tags (AEXX and HFSCREEN as they apply with LHFCALC = .TRUE.)
TAG_DEFAULTS = {"LHFCALC": "FALSE", "LASPH": "FALSE", "LSOL": "FALSE", "IVDW": "0", "AEXX": "0.25", "HFSCREEN": "0"}
# Tags that matter only when another one is on: the hybrid mixing and screening, the solvent's permittivity.
CONDITIONAL_TAGS = {"AEXX": "LHFCALC", "HFSCREEN": "LHFCALC", "EB_K": "LSOL"}
COLLECT_MARK = "(mechanet references --collect)"
ENTRY_KEYS = ("energy_e0_eV", "E", "free_energy_correction_eV", "Gcorr", "G_eV")
RUNS = ("relax", "freq")


def _tag_value(value, tag=None):
    """An INCAR value in a comparable form: T, .TRUE. and .T. agree; 500 and 500.0 agree; unset is the default."""
    text = str(value if value is not None else "").strip().strip("'\"").upper()
    if not text:
        return TAG_DEFAULTS.get(tag, "")
    if tag in LOGICAL_TAGS:
        bare = text.strip(".")
        return {"T": "TRUE", "F": "FALSE"}.get(bare[:1], bare)
    try:
        return f"{float(text.rstrip('.').replace('D', 'E')):g}"
    except ValueError:
        return text.replace(".", "")


def _differences(mine, theirs):
    """Protocol tags in which two tag maps differ (by value; AEXX, HFSCREEN and EB_K only when switched on)."""
    out = []
    for tag in PROTOCOL_TAGS:
        switch = CONDITIONAL_TAGS.get(tag)
        if switch and _tag_value(mine.get(switch), switch) == "FALSE" == _tag_value(theirs.get(switch), switch):
            continue
        if _tag_value(mine.get(tag), tag) != _tag_value(theirs.get(tag), tag):
            out.append(tag)
    return out


def _titles(text):
    import re

    return list(dict.fromkeys(" ".join(t.split()) for t in re.findall(r"TITEL\s*=\s*(.+)", text)))


def _paw_titles(folder):
    """TITEL lines of the PAW datasets a run used (POTCAR, or OUTCAR when POTCAR is absent)."""
    for name in ("POTCAR", "OUTCAR"):
        path = Path(folder) / name
        if path.is_file():
            titles = _titles(path.read_text(errors="replace"))
            if titles:
                return titles
    return []


def _paw_element(title):
    import re

    match = re.match(r"\s*\S+\s+([A-Z][a-z]?)", title)
    return match.group(1) if match else None


def _run_protocol(folder):
    """{"incar": protocol tags, "paw": PAW titles} of one molecule run."""
    from .vasp import read_incar

    incar = Path(folder) / "INCAR"
    values = read_incar(incar) if incar.is_file() else {}
    return {"incar": {k: values[k] for k in PROTOCOL_TAGS if k in values}, "paw": _paw_titles(folder)}


def _runs(entry):
    """The recorded protocol of each run of a molecule entry: {run: {"incar", "paw"}} ({} when none)."""
    protocol = entry.get("protocol")
    if not isinstance(protocol, dict):
        return {}
    if "incar" in protocol:  # a single record: the relaxation
        return {"relax": protocol}
    return {run: record for run, record in protocol.items() if run in RUNS and isinstance(record, dict)}


def _describe(record):
    tags = " ".join(f"{k}={v}" for k, v in (record.get("incar") or {}).items())
    paw = record.get("paw") or []
    return (tags or "no functional, cutoff or solvation tags in the INCAR") + ("; PAW " + ", ".join(paw) if paw else "")


def slab_protocol(campaign, elements=()):
    """The slab's protocol: the INCAR tags of the stage whose energies the analysis uses, and PAW titles.

    The INCAR is inputs/INCAR_final with vasp.common_overrides and the stage_overrides of
    04_static (frequency.central = refine) or 02_final (central = final). PAW titles come from
    the POTCARs the campaign's jobs staged (potcars/*/provenance.json) and, for elements not
    staged yet (the slab's and ``elements``), from the configured POTCAR files when readable.
    Returns (tags, description, paw: element -> set of titles), or None without INCAR_final.
    """
    from .fileio import read_json
    from .vasp import STAGE_FINAL, STAGE_STATIC, central_mode, incar_value, potcar_paths, read_incar

    campaign = Path(campaign)
    template = campaign / "inputs" / "INCAR_final"
    if not template.is_file():
        return None
    config = read_json(campaign / "config.json") if (campaign / "config.json").is_file() else {}
    vasp = config.get("vasp") or {}
    try:
        stage = STAGE_STATIC if central_mode(config) == "refine" else STAGE_FINAL
    except ValueError:
        stage = STAGE_STATIC
    values = read_incar(template)
    for overrides in (vasp.get("common_overrides") or {}, (vasp.get("stage_overrides") or {}).get(stage) or {}):
        values.update({str(k).upper(): incar_value(v) for k, v in overrides.items()})
    description = f"the slab's {stage} INCAR (inputs/INCAR_final with vasp.common_overrides and stage_overrides)"
    paw = {}
    for path in sorted((campaign / "potcars").glob("*/provenance.json")):
        for record in read_json(path).get("datasets", []):
            paw.setdefault(record["element"], set()).add(" ".join(str(record["titel"]).split()))
    wanted = [e for e in dict.fromkeys([*((config.get("slab") or {}).get("species_order") or []), *elements])]
    for element in [e for e in wanted if e not in paw]:  # not staged by a job yet: the configured file
        try:
            path = Path(potcar_paths(config, [element])[element])
        except (ValueError, KeyError):
            continue
        if path.is_file():
            titles = _titles(path.read_text(errors="replace"))
            if titles:
                paw.setdefault(element, set()).update(titles)
    return {k: values.get(k) for k in PROTOCOL_TAGS}, description, paw


def _entries(data):
    """The molecule entries of a references file (full or compact format) that carry values."""
    block = data["references"] if isinstance(data.get("references"), dict) else data
    return {
        name: entry
        for name, entry in block.items()
        if isinstance(entry, dict) and any(entry.get(k) is not None for k in ENTRY_KEYS)
    }


def _has_correction(entry):
    return any(entry.get(k) is not None for k in ("free_energy_correction_eV", "Gcorr", "G_eV"))


def collect_references(folder, references_path, temperature_K=None, campaign=None):
    """Read E0 (relax/) and G_corr (freq/) of every molecule folder into the references file.

    Each molecule entry records how it was computed (``protocol``: the INCAR protocol tags and
    PAW titles of its relax and freq runs; ``Gcorr_temperature_K``). From the file as it then
    stands, the provenance flags say what mechanet itself can vouch for, and are set again on
    every collect:

    * ``temperature_verified``: every free-energy correction was computed here at temperature_K;
    * ``water_standard_state_verified``: the H2O correction was computed here at 1 bar and its
      pressure declared (the analysis moves it to the liquid-water pressure);
    * ``DFT_protocol_compatibility_verified``: with the campaign, every run of every molecule
      has the protocol tags and PAW datasets of the slab (see :func:`slab_protocol`).

    Values entered by hand are kept but cannot be vouched for. ``protocol_id`` is written when
    the file has none or a previous collect wrote it; a description of the user's is kept.
    The compact format (``references: {H2: {E, Gcorr}}``) is updated in place.
    """
    from .results import parse_run

    folder = Path(folder)
    references_path = Path(references_path)
    data = read_yaml(references_path) if references_path.is_file() else {}
    data = {} if data is None else data
    if not isinstance(data, dict):
        raise ValueError(f"{references_path} is not a references file (a mapping of molecules)")
    compact = isinstance(data.get("references"), dict)
    block = data["references"] if compact else data
    energy_key, correction_key = ("E", "Gcorr") if compact else ("energy_e0_eV", "free_energy_correction_eV")
    previous_temperature = data.get("temperature_K")
    temperature = float(temperature_K or previous_temperature or 298.15)
    found, warnings = {}, []
    for name in MOLECULES:
        relax, freq = folder / name / "relax", folder / name / "freq"
        if not (relax / "OUTCAR").is_file():
            continue
        result = parse_run(relax, include_xml=False)
        if not result.get("completed") or result.get("energy_e0_eV") is None:
            raise ValueError(f"{relax}: the relaxation did not finish normally")
        previous = block.get(name)
        if previous is not None and not isinstance(previous, dict):
            raise ValueError(
                f"{name} in {references_path} must be a mapping (energy_e0_eV, free_energy_correction_eV, ...), "
                f"not {previous!r}"
            )
        entry = dict(previous or {})
        with_freq = (freq / "OUTCAR").is_file()
        if entry.get("G_eV") is not None:
            if with_freq:
                warnings.append(f"{name}: G_eV {entry['G_eV']} eV entered by hand is replaced by E0 + G_corr")
                entry.pop("G_eV")
            else:
                warnings.append(
                    f"{name}: G_eV {entry['G_eV']} eV entered by hand is kept and still used; E0 + G_corr "
                    "replaces it once freq/ has run"
                )
        for key in ("energy_e0_eV", "E"):
            entry.pop(key, None)  # one spelling, the one the file's format uses
        entry[energy_key] = float(result["energy_e0_eV"])
        runs = {"relax": _run_protocol(relax)}
        if with_freq:
            modes = parse_run(freq, include_xml=False).get("frequencies_cm1") or []
            if not modes:
                raise ValueError(f"{freq}: no frequencies found (IBRION = 5 run incomplete?)")
            for key in ("free_energy_correction_eV", "Gcorr"):
                entry.pop(key, None)
            entry[correction_key] = round(gas_correction(name, modes, temperature), 6)
            entry["Gcorr_temperature_K"] = temperature
            entry["frequencies_cm1"] = [round(float(x), 2) for x in modes]
            if name == "H2O":
                entry["Gcorr_pressure_bar"] = 1.0
            runs["freq"] = _run_protocol(freq)
            differs = _differences(runs["relax"]["incar"], runs["freq"]["incar"])
            if differs:
                warnings.append(f"{name}: the relax and freq INCARs differ in {', '.join(differs)}")
        elif any(entry.get(k) is not None for k in ("free_energy_correction_eV", "Gcorr")):
            kept = entry.get(correction_key, entry.get("free_energy_correction_eV", entry.get("Gcorr")))
            warnings.append(
                f"{name}: no freq/ result; the free-energy correction already in the file was kept ({kept} eV)"
            )
        entry["protocol"] = runs
        entry.setdefault("phase", "liquid" if name == "H2O" else "gas")
        entry["source"] = f"mechanet references --collect {folder}"
        block[name] = entry
        found[name] = {
            "energy_e0_eV": entry[energy_key],
            "free_energy_correction_eV": entry.get(correction_key) if with_freq else None,
        }
    if not found:
        raise FileNotFoundError(f"No molecule results under {folder} (expected NAME/relax/OUTCAR)")
    data["temperature_K"] = temperature
    entries = _entries(data)
    records = {n: _runs(e) for n, e in entries.items() if _runs(e)}

    # protocol_id: the protocol of the computed molecules, unless the user wrote one.
    relaxations = {n: r["relax"] for n, r in records.items() if "relax" in r}
    by_element = {}
    for runs in records.values():
        for record in runs.values():
            for title in record.get("paw") or []:
                by_element.setdefault(_paw_element(title) or title, set()).add(title)
    first = next(iter(relaxations.values()), {})
    same_tags = all(not _differences(first.get("incar") or {}, r.get("incar") or {}) for r in relaxations.values())
    if same_tags and all(len(t) == 1 for t in by_element.values()):
        titles = [next(iter(t)) for t in by_element.values()]
        protocol_text = "molecules: " + _describe({"incar": first.get("incar") or {}, "paw": titles})
    else:
        protocol_text = "molecules: " + "; ".join(f"{n}: {_describe(r)}" for n, r in relaxations.items())
        for tag in PROTOCOL_TAGS:
            values = {n: _tag_value((r.get("incar") or {}).get(tag), tag) for n, r in relaxations.items()}
            if len(set(values.values())) > 1:
                warnings.append(
                    f"the molecules were computed with different {tag}: "
                    + ", ".join(f"{n} {(relaxations[n].get('incar') or {}).get(tag, 'unset')}" for n in values)
                )
    current = data.get("protocol_id")
    if not protocol_documented(current) or str(current).rstrip().endswith(COLLECT_MARK):
        data["protocol_id"] = f"{protocol_text} {COLLECT_MARK}"

    # Provenance flags, from the file as it now stands.
    provenance = data.get("provenance")
    if provenance is not None and not isinstance(provenance, dict):
        warnings.append(f"provenance was {provenance!r}, not a mapping of flags; it is written again")
    provenance = dict(provenance) if isinstance(provenance, dict) else {}
    corrected = {n: e for n, e in entries.items() if _has_correction(e)}
    by_hand = sorted(
        n for n, e in corrected.items() if e.get("Gcorr_temperature_K") is None or e.get("G_eV") is not None
    )
    off_temperature = sorted(
        n
        for n, e in corrected.items()
        if e.get("Gcorr_temperature_K") is not None and abs(float(e["Gcorr_temperature_K"]) - temperature) > 1e-6
    )
    provenance["temperature_verified"] = bool(corrected) and not by_hand and not off_temperature
    if by_hand:
        warnings.append(
            f"{', '.join(by_hand)}: free-energy correction entered by hand; its temperature cannot be verified"
        )
    if off_temperature:
        warnings.append(
            f"{', '.join(off_temperature)}: free-energy correction computed at another temperature than "
            f"{temperature:g} K; collect it again"
        )
    water = entries.get("H2O") or {}
    provenance["water_standard_state_verified"] = bool(
        water.get("Gcorr_temperature_K") is not None
        and water.get("Gcorr_pressure_bar") == 1.0
        and water.get("G_eV") is None
    )
    if previous_temperature is not None and abs(float(previous_temperature) - temperature) > 1e-6:
        warnings.append(f"temperature_K changed from {float(previous_temperature):g} K to {temperature:g} K")

    provenance["DFT_protocol_compatibility_verified"] = False
    provenance.pop("checked", None)
    elements = sorted({s for n in records if n in MOLECULES for s in MOLECULES[n]["symbols"]})
    slab = slab_protocol(campaign, elements) if campaign is not None else None
    if campaign is None:
        warnings.append(
            "protocol not compared with the slab: no campaign here (run in the project folder or pass --campaign)"
        )
    elif slab is None:
        warnings.append(f"protocol not compared with the slab: {Path(campaign) / 'inputs' / 'INCAR_final'} not found")
    else:
        slab_tags, description, slab_paw = slab
        unchecked = sorted(set(entries) - set(records))
        problems, paw_unchecked = [], {}
        for n, runs in records.items():
            for run, record in runs.items():
                label = n if run == "relax" else f"{n} ({run})"
                incar = record.get("incar") or {}
                problems += [
                    f"{label} {tag}: molecule {incar.get(tag, 'unset')}, slab {slab_tags.get(tag) or 'unset'}"
                    for tag in _differences(incar, slab_tags)
                ]
                titles = record.get("paw") or []
                if not titles:
                    reason = "no POTCAR or OUTCAR with TITEL lines in the molecule runs"
                elif not slab_paw:
                    reason = (
                        "no POTCAR record in the campaign yet (potcars/ is written when jobs start) and the "
                        "configured POTCAR files are not readable here"
                    )
                else:
                    missing = sorted({_paw_element(t) or t for t in titles} - set(slab_paw))
                    reason = f"no slab dataset of {', '.join(missing)} recorded or readable here" if missing else None
                    for title in titles:
                        element = _paw_element(title)
                        if element in slab_paw and title not in slab_paw[element]:
                            slab_titles = ", ".join(sorted(slab_paw[element]))
                            problems.append(f"{label} PAW {element}: molecule {title}, slab {slab_titles}")
                if reason:
                    paw_unchecked.setdefault(reason, [])
                    if n not in paw_unchecked[reason]:
                        paw_unchecked[reason].append(n)
        warnings += [f"protocol differs from the slab's: {p}" for p in problems]
        if unchecked:
            warnings.append(f"{', '.join(unchecked)}: entered by hand; its protocol cannot be compared with the slab")
        for reason, names in paw_unchecked.items():
            warnings.append(f"PAW datasets of {', '.join(names)} not compared: {reason}")
        provenance["DFT_protocol_compatibility_verified"] = not problems and not unchecked and not paw_unchecked
        provenance["checked"] = f"INCAR tags {', '.join(PROTOCOL_TAGS)} against {description}" + (
            "; PAW datasets of the shared elements" if slab_paw else ""
        )
    data["provenance"] = provenance
    write_yaml(references_path, data, header="# Molecular references (updated by mechanet references --collect)\n")
    return {
        "references": str(references_path),
        "molecules": found,
        "protocol_id": data["protocol_id"],
        "provenance": provenance,
        "warnings": warnings,
        "note": (
            "G_corr is the ideal-gas value at 1 bar; for H2O the analysis moves it to the liquid-water "
            "pressure (thermodynamics.water_equilibrium_pressure_bar)."
            if "H2O" in found
            else None
        ),
    }
