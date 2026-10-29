"""VASP stage protocols and input writing.

Every state runs through staged calculations built from the user's own INCAR templates
(``inputs/INCAR_loose`` and ``inputs/INCAR_final`` of the campaign):

======== ===================================================================
01_loose preliminary relaxation (the loose template)
02_final final relaxation (the final template); writes the magnetic CHGCAR
03_refine short re-relaxation with reciprocal-space projectors (protocol "refine" only)
04_static single point that seeds the frequency displacements
05_freq  restartable finite-difference partial Hessian (or VASP IBRION=5)
======== ===================================================================

``frequency.central`` selects the protocol: ``"final"`` (frequencies at the 02_final
geometry with the 02_final Hamiltonian; node energy = 02_final E0) or ``"refine"`` (the
default; node energy = 04_static E0 after 03_refine). Every generated stage records its
values, overrides and reasons in ``stage.json``. No licensed PAW data are bundled.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .fileio import atomic_bytes, file_hash, read_json, write_json

STAGES = ("01_loose", "02_final", "03_refine", "04_static", "05_freq")
STAGE_LOOSE, STAGE_FINAL, STAGE_REFINE, STAGE_STATIC, STAGE_FREQ = STAGES
RELAXATION_STAGES = (STAGE_LOOSE, STAGE_FINAL, STAGE_REFINE)
THROUGH_NAMES = {
    "loose": STAGE_LOOSE,
    "final": STAGE_FINAL,
    "refine": STAGE_REFINE,
    "static": STAGE_STATIC,
    "freq": STAGE_FREQ,
}
FREQUENCY_METHODS = ("displacements", "vasp_ibrion5")
CENTRAL_MODES = ("refine", "final")
GEOMETRY_SOURCE = {
    STAGE_LOOSE: "POSCAR",
    STAGE_FINAL: f"{STAGE_LOOSE}/CONTCAR",
    STAGE_REFINE: f"{STAGE_FINAL}/CONTCAR",
    STAGE_STATIC: f"{STAGE_REFINE}/CONTCAR",
    STAGE_FREQ: f"{STAGE_STATIC}/POSCAR",
}
RESTART_SOURCE = {
    STAGE_LOOSE: None,
    STAGE_FINAL: None,
    STAGE_REFINE: f"{STAGE_FINAL}/CHGCAR",
    STAGE_STATIC: f"{STAGE_REFINE}/CHGCAR",
    STAGE_FREQ: f"{STAGE_STATIC}/WAVECAR",
}
FINAL_GEOMETRY_SOURCE = {
    **{k: v for k, v in GEOMETRY_SOURCE.items() if k != STAGE_REFINE},
    STAGE_STATIC: f"{STAGE_FINAL}/CONTCAR",
}
FINAL_RESTART_SOURCE = {
    **{k: v for k, v in RESTART_SOURCE.items() if k != STAGE_REFINE},
    STAGE_STATIC: f"{STAGE_FINAL}/CHGCAR",
}
NON_SCF_ALGOS = frozenset(
    {
        "EIGENVAL",
        "EXACT",
        "NONE",
        "NOTHING",
        "CHI",
        "G0W0",
        "GW0",
        "GW",
        "SCGW0",
        "SCGW",
        "EVGW0",
        "EVGW",
        "QPGW0",
        "QPGW",
        "ACFDT",
        "ACFDTR",
        "RPA",
        "RPAR",
        "BSE",
        "TDHF",
        "TIMEEV",
        "CRPA",
    }
)
# With frequency.central = "final", 04_static and 05_freq may differ from 02_final only in
# these tags (restart, SCF tolerance and algorithm, symmetry, ionic control, output and
# parallelization); every other tag defines the energy surface and must be identical.
FINAL_HAMILTONIAN_FREE_TAGS = frozenset(
    {
        "SYSTEM",
        "ISTART",
        "ICHARG",
        "EDIFF",
        "EDIFFG",
        "NELM",
        "NELMIN",
        "NELMDL",
        "ISYM",
        "IBRION",
        "NSW",
        "POTIM",
        "NFREE",
        "ISIF",
        "LWAVE",
        "LCHARG",
        "LMAXMIX",
        "LAECHG",
        "LVTOT",
        "LVHAR",
        "LORBIT",
        "LELF",
        "NWRITE",
        "LDAUPRINT",
        "ALGO",
        "AMIX",
        "BMIX",
        "AMIX_MAG",
        "BMIX_MAG",
        "AMIN",
        "IMIX",
        "MAXMIX",
        "INIMIX",
        "MIXPRE",
        "WC",
        "TIME",
        "IWAVPR",
        "NCORE",
        "NPAR",
        "KPAR",
        "NSIM",
        "LPLANE",
        "LSCALU",
        "LSCALAPACK",
        "NBLOCK",
        "KBLOCK",
    }
)
F_ELEMENTS = frozenset("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Ac Th Pa U Np Pu Am Cm".split())
FORCE_GATE_DEFAULTS = {"refine": (0.03, 0.02), "final": (0.035, 0.03)}
DEFAULT_MAX_STATIC_E0_SHIFT_EV = 0.01
DEFAULT_MAX_MOMENT_CHANGE_MUB = 0.5


def find_campaign(state_dir):
    state_dir = Path(state_dir).resolve()
    campaign = state_dir.parent.parent
    if state_dir.parent.name != "states" or not (campaign / "config.json").is_file():
        raise ValueError(f"Expected campaign/states/STATE with campaign/config.json: {state_dir}")
    return campaign


def read_incar(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        line = re.split(r"[#!]", line, maxsplit=1)[0]
        for statement in line.split(";"):
            if "=" not in statement:
                continue
            key, value = statement.split("=", 1)
            key, value = key.strip().upper(), value.strip()
            if key in values:
                raise ValueError(f"Repeated INCAR tag {key} in {path}; make the intended value explicit")
            if key and value:
                values[key] = value
    return values


def incar_value(value):
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    if isinstance(value, (list, tuple)):
        return " ".join(incar_value(v) for v in value)
    return str(value)


def render_incar(values):
    return "# Generated by mechanet; the original templates are in <campaign>/inputs.\n" + "".join(
        f"{key} = {incar_value(value)}\n" for key, value in values.items()
    )


def expand_values(value):
    out = []
    for token in str(value).split():
        if "*" in token:
            number, scalar = token.split("*", 1)
            out.extend([float(scalar)] * int(number))
        else:
            out.append(float(token))
    return out


def species_order(config):
    return list((config or {}).get("slab", {}).get("species_order") or [])


def state_species(meta, config=None):
    """Species of a state in POSCAR order; atoms must be grouped (builder order)."""
    symbols = meta["symbols"]
    species = list(dict.fromkeys(symbols))
    grouped = [s for s in species for _ in range(symbols.count(s))]
    if grouped != symbols:
        raise ValueError("State atoms must be grouped by element; reorder metadata and masks together")
    order = species_order(config)
    if order:
        expected = [s for s in order if s in species] + sorted(s for s in species if s not in order)
        if expected != species:
            raise ValueError(f"State species order {species} differs from the campaign order {expected}")
    return species


def template_path(campaign, stage):
    return Path(campaign) / "inputs" / ("INCAR_loose" if stage == STAGE_LOOSE else "INCAR_final")


def frequency_method(config):
    method = str(config.get("frequency", {}).get("method", "displacements"))
    if method not in FREQUENCY_METHODS:
        raise ValueError(f"frequency.method must be one of {FREQUENCY_METHODS}, not {method!r}")
    return method


def central_mode(config):
    """frequency.central of a configuration: "refine" (default) or "final"."""
    mode = str((config or {}).get("frequency", {}).get("central", "refine"))
    if mode not in CENTRAL_MODES:
        raise ValueError(f"frequency.central must be one of {CENTRAL_MODES}, not {mode!r}")
    return mode


def frequency_gates(config):
    """Checks on 04_static before 05_freq starts (and again in collect); defaults follow the protocol."""
    freq = (config or {}).get("frequency", {})
    mode = central_mode(config)
    limit, warn = FORCE_GATE_DEFAULTS[mode]
    return {
        "max_force_eVA": float(freq.get("max_force_eVA", limit)),
        "warn_force_eVA": float(freq.get("warn_force_eVA", warn)),
        "max_static_e0_shift_eV": float(freq.get("max_static_e0_shift_eV", DEFAULT_MAX_STATIC_E0_SHIFT_EV)),
        "max_moment_change_muB": float(freq.get("max_moment_change_muB", DEFAULT_MAX_MOMENT_CHANGE_MUB)),
        "same_solution_as_final": mode == "final",
    }


def force_gate(config):
    gates = frequency_gates(config)
    return gates["max_force_eVA"], gates["warn_force_eVA"]


def plan_mode(plan):
    """The frequency.central a prepared plan.json was written for (plans without the key: "refine")."""
    mode = str((plan or {}).get("central", "refine"))
    if mode not in CENTRAL_MODES:
        raise ValueError(f"plan.json names an unknown frequency.central {mode!r}")
    return mode


def protocol_stages(config_or_mode):
    """The stages of the protocol, in order (no 03_refine when frequency.central = "final")."""
    mode = config_or_mode if isinstance(config_or_mode, str) else central_mode(config_or_mode)
    if mode not in CENTRAL_MODES:
        raise ValueError(f"frequency.central must be one of {CENTRAL_MODES}, not {mode!r}")
    return tuple(s for s in STAGES if s != STAGE_REFINE) if mode == "final" else STAGES


def stage_sources(mode):
    return (FINAL_GEOMETRY_SOURCE, FINAL_RESTART_SOURCE) if mode == "final" else (GEOMETRY_SOURCE, RESTART_SOURCE)


def resolve_through(through, config_or_mode):
    """The stage folder a --through value names, checked against the protocol."""
    mode = config_or_mode if isinstance(config_or_mode, str) else central_mode(config_or_mode)
    stage = THROUGH_NAMES.get(through, through)
    if stage not in STAGES:
        raise ValueError(f"Unknown final stage {through}; choose one of {', '.join(THROUGH_NAMES)}")
    if stage not in protocol_stages(mode):
        names = ", ".join(k for k, v in THROUGH_NAMES.items() if v in protocol_stages(mode))
        raise ValueError(
            f"{stage} is not part of this campaign's protocol (frequency.central = {mode!r}: "
            f"04_static follows 02_final directly); choose one of {names}"
        )
    return stage


def _flag(value):
    return str(value).strip().upper().strip(".") in ("T", "TRUE")


def _num(values, key):
    return int(float(values[key]))


def mpi_ranks(config):
    sched = config.get("scheduler", {})
    if sched.get("mpi_ranks"):
        return int(sched["mpi_ranks"])
    sbatch = sched.get("sbatch", {})
    if sbatch.get("ntasks"):
        return int(sbatch["ntasks"])
    nodes = int(sbatch.get("nodes", 1) or 1)
    if sbatch.get("ntasks-per-node"):
        return nodes * int(sbatch["ntasks-per-node"])
    return None


def _dft_u_arrays(values, species, config):
    """LDAUL/LDAUU/LDAUJ for the present species, from vasp.dft_u or the template arrays."""
    table = config.get("vasp", {}).get("dft_u")
    if table:
        rows = [table.get(s, {}) for s in species]
        return {
            "LDAUL": " ".join(str(int(r.get("L", -1))) for r in rows),
            "LDAUU": " ".join(f"{float(r.get('U', 0.0)):g}" for r in rows),
            "LDAUJ": " ".join(f"{float(r.get('J', 0.0)):g}" for r in rows),
        }
    order = config.get("vasp", {}).get("template_species_order") or species_order(config)
    out = {}
    for tag in ("LDAUL", "LDAUU", "LDAUJ"):
        if tag not in values:
            continue
        parts = values[tag].split()
        if len(parts) != len(order):
            raise ValueError(
                f"{tag} in the INCAR template has {len(parts)} entries, but the template species order "
                f"{order} has {len(order)}; set vasp.template_species_order or vasp.dft_u"
            )
        lookup = dict(zip(order, parts))
        default = {"LDAUL": "-1", "LDAUU": "0", "LDAUJ": "0"}[tag]
        out[tag] = " ".join(lookup.get(s, default) for s in species)
    return out


def stage_settings(campaign, meta, stage, config=None):
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage}")
    config = config or read_json(Path(campaign) / "config.json")
    mode = central_mode(config)
    if stage not in protocol_stages(mode):
        raise ValueError(f"{stage} is not part of the frequency.central = {mode!r} protocol")
    vasp = config.get("vasp", {})
    freq = config.get("frequency", {})
    refine = config.get("refine", {})
    original = read_incar(template_path(campaign, stage))
    values = dict(original)
    notes = []
    species = state_species(meta, config)
    values.update(_dft_u_arrays(values, species, config))
    if vasp.get("dft_u") and "LDAU" not in values:
        values.update(LDAU=".TRUE.", LDAUTYPE="2")
        notes.append("DFT+U arrays from vasp.dft_u (LDAUTYPE 2).")
    common = {str(k).upper(): incar_value(v) for k, v in vasp.get("common_overrides", {}).items()}
    values.update(common)
    if "NCORE" in common:
        notes.append("NCORE runtime override is numerical parallelization only; the template is retained.")
    method = frequency_method(config) if stage == STAGE_FREQ else None
    lmaxmix = "6" if set(species) & F_ELEMENTS else "4"
    if stage in (STAGE_LOOSE, STAGE_FINAL):
        values.update(ISTART="0", ICHARG="2")
        notes.append("Fresh atomic-density SCF start, as in the two-stage loose/final procedure.")
    if stage == STAGE_FINAL:
        values.update(LWAVE=".FALSE.", LCHARG=".TRUE.", LMAXMIX=lmaxmix)
        notes.append(
            "Write the magnetic CHGCAR only (LMAXMIX keeps the d/f on-site density for DFT+U density "
            "restarts); no WAVECAR is written because no stage reads it."
        )
    if stage == STAGE_REFINE:
        values.update(
            ISTART="0",
            ICHARG="1",
            PREC="Accurate",
            LREAL=".FALSE.",
            ISYM="0",
            EDIFF=str(refine.get("ediff", 1e-6)),
            EDIFFG=str(refine.get("ediffg", -0.015)),
            IBRION=str(refine.get("ibrion", 2)),
            NSW=str(refine.get("nsw", 60)),
            LWAVE=".FALSE.",
            LCHARG=".TRUE.",
            LMAXMIX=lmaxmix,
        )
        notes.append(
            "Short re-relaxation with the static/frequency projector settings (LREAL=.FALSE., ISYM=0), "
            "starting from the 02_final magnetic CHGCAR, so the central geometry is stationary under the "
            "settings used for the energy and Hessian."
        )
    if stage in (STAGE_STATIC, STAGE_FREQ) and mode == "refine":
        values.update(
            ISTART="0",
            ICHARG="1",
            EDIFF=str(freq.get("ediff", 1e-7)),
            NELM=str(freq.get("nelm", 120)),
            PREC="Accurate",
            LREAL=".FALSE.",
            ISYM="0",
            IBRION="-1",
            NSW="0",
            LWAVE=".TRUE.",
            LCHARG=".FALSE.",
            LMAXMIX=lmaxmix,
        )
        notes.append("Tighter electronic tolerance and reciprocal-space projectors for central energy and Hessian.")
        notes.append("ISYM=0 avoids imposing crystal symmetry during electronic continuation/displacements.")
    if stage in (STAGE_STATIC, STAGE_FREQ) and mode == "final":
        values.update(
            ISTART="0",
            ICHARG="1",
            EDIFF=str(freq.get("ediff", 1e-7)),
            NELM=str(freq.get("nelm", 120)),
            ISYM="0",
            IBRION="-1",
            NSW="0",
            LWAVE=".TRUE.",
            LCHARG=".FALSE.",
            LMAXMIX=lmaxmix,
        )
        notes.append(
            "frequency.central = final: the 02_final Hamiltonian at the 02_final geometry; only EDIFF "
            "(frequency.ediff), NELM and ISYM differ, so the Hessian belongs to the energy surface on which "
            "02_final was relaxed."
        )
        notes.append("ISYM=0 avoids imposing crystal symmetry during electronic continuation/displacements.")
    if stage == STAGE_STATIC and mode == "refine":
        notes.append("Continue the 03_refine magnetic CHGCAR using ISTART=0, ICHARG=1; WAVECAR is written for 05_freq.")
    if stage == STAGE_STATIC and mode == "final":
        notes.append(
            "Single point at 02_final/CONTCAR continuing the 02_final magnetic CHGCAR (ISTART=0, ICHARG=1); "
            "it seeds the displacements (WAVECAR) and supplies the force check. The node energy is the "
            "02_final E0."
        )
    if stage == STAGE_FREQ:
        values.update(ISTART="1", ICHARG="0", LWAVE=".FALSE.", LCHARG=".FALSE.")
        notes.append("Restart the 04_static WAVECAR (same cell, basis, k-points and ISYM).")
        if method == "vasp_ibrion5":
            values.update(
                IBRION="5", NSW="1", NFREE=str(freq.get("nfree", 2)), POTIM=str(freq.get("potim", 0.015)), ISIF="2"
            )
            notes.append("Native VASP partial Hessian (IBRION=5); only frequency-active atoms are displaced.")
        else:
            notes.append(
                "Restartable finite differences: one single point per active Cartesian coordinate "
                "and sign (+h, -h); WAVECAR is copied into every unit."
            )
    explicit = {str(k).upper(): incar_value(v) for k, v in vasp.get("stage_overrides", {}).get(stage, {}).items()}
    values.update(explicit)
    moments = vasp.get("magmom_by_element")
    per_atom = vasp.get("magmom_by_id") or {}
    if moments is not None or per_atom:
        moments = moments or {}
        default = vasp.get("magmom_default")
        vals = []
        for atom_id, symbol in zip(meta["atom_ids"], meta["symbols"]):
            if atom_id in per_atom:
                val = per_atom[atom_id]
            elif symbol in moments:
                val = moments[symbol]
            elif default is not None:
                val = default
            else:
                raise ValueError(
                    f"Explicit MAGMOM map has no value for {atom_id} ({symbol}); add it to "
                    "vasp.magmom_by_element or set vasp.magmom_default"
                )
            vals.append(float(val))
        values["MAGMOM"] = " ".join(f"{v:.8g}" for v in vals)
        notes.append("Explicit user-configured moment initialization; not a local-spin or oxidation constraint.")
    elif "MAGMOM" in values and len(expand_values(values["MAGMOM"])) != len(meta["symbols"]):
        raise ValueError("Active template MAGMOM length no longer matches the generated state; use an element/ID map")
    else:
        notes.append("MAGMOM is absent in the active tags; VASP's default applies unless configured.")
    _check_stage_consistency(stage, values, method, freq, mode)
    if mode == "final" and stage in (STAGE_STATIC, STAGE_FREQ):
        _check_same_hamiltonian(stage, values, stage_settings(campaign, meta, STAGE_FINAL, config)["values"])
    overrides = {k: {"original": original.get(k), "generated": v} for k, v in values.items() if original.get(k) != v}
    ranks = mpi_ranks(config)
    ncore, kpar = int(values.get("NCORE", 1)), int(values.get("KPAR", 1))
    if ranks and (ranks % kpar or (ranks // kpar) % ncore):
        notes.append(f"WARNING: {ranks} MPI ranks / KPAR={kpar} is not divisible by NCORE={ncore}.")
    geometry_sources, restart_sources = stage_sources(mode)
    spec = {
        "stage": stage,
        "values": values,
        "overrides": overrides,
        "notes": notes,
        "template_sha256": file_hash(template_path(campaign, stage)),
        "species": species,
        "geometry_source": geometry_sources[stage],
        "restart_source": restart_sources[stage],
    }
    if stage == STAGE_FREQ:
        spec["method"] = method
        if method == "displacements":
            spec["displacement_A"] = float(freq.get("potim", 0.015))
            spec["nfree"] = int(freq.get("nfree", 2))
    return spec


def _check_stage_consistency(stage, values, method, freq, mode="refine"):
    """Reject overrides that would silently change the documented stage protocol."""
    if int(float(values.get("ISPIN", 1))) not in (1, 2) or _flag(values.get("LNONCOLLINEAR", "F")):
        raise ValueError("This protocol supports collinear ISPIN = 1 or 2 only")
    # Avoid a misleading declaration that a new-charge or constrained-spin run is the same protocol.
    if "NELECT" in values or "NUPDOWN" in values or "I_CONSTRAINED_M" in values:
        raise ValueError("NELECT/NUPDOWN/local-spin constraints need a separately defined protocol")
    istart, icharg = _num(values, "ISTART"), _num(values, "ICHARG")
    if stage in (STAGE_LOOSE, STAGE_FINAL) and (istart != 0 or icharg != 2):
        raise ValueError("Loose/final stages must keep a fresh ISTART=0, ICHARG=2 start in this protocol")
    if stage in (STAGE_REFINE, STAGE_STATIC) and (istart != 0 or icharg != 1):
        raise ValueError(f"{stage} must continue the preceding CHGCAR with ISTART=0, ICHARG=1")
    if stage == STAGE_FREQ and (istart != 1 or icharg != 0):
        raise ValueError("Frequency stage must continue the static WAVECAR with ISTART=1, ICHARG=0")
    if stage in (STAGE_FINAL, STAGE_REFINE) and not _flag(values.get("LCHARG", "F")):
        raise ValueError(f"{stage} must write CHGCAR (LCHARG=.TRUE.) for the next density restart")
    if (
        _flag(values.get("LDAU", "F"))
        and stage in (STAGE_FINAL, STAGE_REFINE, STAGE_STATIC, STAGE_FREQ)
        and int(float(values.get("LMAXMIX", 2))) < 4
    ):
        raise ValueError(f"{stage} needs LMAXMIX>=4 so DFT+U on-site densities survive the restart chain")
    if stage in RELAXATION_STAGES:
        if _num(values, "IBRION") not in (1, 2, 3) or _num(values, "NSW") < 1:
            raise ValueError(f"{stage} must be an ionic relaxation (IBRION 1/2/3, NSW>=1)")
    if stage in (STAGE_FINAL, STAGE_REFINE) and float(values.get("EDIFFG", "0")) >= 0:
        raise ValueError(f"{stage} must use an explicit negative (force-based) EDIFFG")
    reciprocal = (STAGE_REFINE, STAGE_STATIC, STAGE_FREQ) if mode == "refine" else (STAGE_REFINE,)
    if stage in reciprocal and (
        _flag(values.get("LREAL", "F")) or str(values.get("LREAL", "")).strip().upper() in ("AUTO", "A", "O", "ON")
    ):
        raise ValueError(f"{stage} must use reciprocal-space projectors (LREAL=.FALSE.)")
    if stage in (STAGE_REFINE, STAGE_STATIC, STAGE_FREQ) and _num(values, "ISYM") != 0:
        raise ValueError(f"{stage} must use ISYM=0 so WAVECAR/CHGCAR k-point sets stay compatible")
    algo = str(values.get("ALGO", "Normal")).strip().upper()
    if stage in (STAGE_STATIC, STAGE_FREQ) and (
        algo in NON_SCF_ALGOS or algo.startswith(("EIG", "EXA", "NONE", "NOTH"))
    ):
        raise ValueError(f"{stage} needs a self-consistent electronic algorithm, not ALGO = {values['ALGO']}")
    if stage == STAGE_STATIC:
        if _num(values, "IBRION") != -1 or _num(values, "NSW") != 0:
            raise ValueError("04_static must be a single point (IBRION=-1, NSW=0)")
        if not _flag(values.get("LWAVE", "F")):
            raise ValueError("04_static must write WAVECAR (LWAVE=.TRUE.) for 05_freq")
    if stage == STAGE_FREQ:
        if method == "vasp_ibrion5":
            if _num(values, "IBRION") != 5:
                raise ValueError("vasp_ibrion5 frequencies require IBRION=5")
            if int(values["NFREE"]) not in (2, 4):
                raise ValueError("Only central-difference NFREE=2 or 4 is supported")
        else:
            if _num(values, "IBRION") != -1 or _num(values, "NSW") != 0:
                raise ValueError("Displacement units must be single points (IBRION=-1, NSW=0)")
            if int(freq.get("nfree", 2)) != 2:
                raise ValueError("The displacements method implements central differences only (frequency.nfree=2)")
            if not 0 < float(freq.get("potim", 0.015)) < 0.1:
                raise ValueError("frequency.potim must be a small positive displacement in Angstrom")


def _canonical_tag_value(value):
    text = str(value).strip()
    if text.upper().strip(".") in ("T", "TRUE", "F", "FALSE"):
        return _flag(text)
    try:
        return tuple(expand_values(text))
    except ValueError:
        return text.upper()


def _check_same_hamiltonian(stage, values, final_values):
    """frequency.central = "final": 04_static/05_freq must use the energy surface of 02_final."""
    differ = sorted(
        tag
        for tag in set(values) | set(final_values)
        if tag not in FINAL_HAMILTONIAN_FREE_TAGS
        and (
            tag not in values
            or tag not in final_values
            or _canonical_tag_value(values[tag]) != _canonical_tag_value(final_values[tag])
        )
    )
    if differ:
        shown = "; ".join(
            f"{tag}: 02_final {final_values.get(tag, '(unset)')}, {stage} {values.get(tag, '(unset)')}"
            for tag in differ
        )
        raise ValueError(
            f"frequency.central = 'final' computes the Hessian with the 02_final Hamiltonian, but "
            f"{stage} differs from 02_final in {shown}. Remove the override, or use "
            "frequency.central = 'refine' (03_refine re-relaxes with the frequency settings)."
        )


# ---------------------------------------------------------------------------
# POTCAR
# ---------------------------------------------------------------------------


def potcar_records(path):
    """(element, POMASS) for every dataset in a (concatenated) POTCAR, in file order."""
    text = Path(path).read_bytes().decode("latin-1")
    records = []
    for block in text.split("End of Dataset"):
        element = re.search(r"VRHFIN\s*=\s*([A-Za-z]+)\s*:", block)
        mass = re.search(r"POMASS?\s*=\s*([0-9]+(?:\.[0-9]*)?)", block)
        if element:
            records.append((element.group(1), float(mass.group(1)) if mass else None))
    return records


def potcar_paths(config, species):
    """Element -> POTCAR path from vasp.potcars, or vasp.potcar_dir plus vasp.potcar_variants."""
    vasp = config.get("vasp", {})
    explicit = vasp.get("potcars") or {}
    root = vasp.get("potcar_dir")
    variants = vasp.get("potcar_variants") or {}
    out = {}
    for element in species:
        if element in explicit:
            out[element] = Path(explicit[element]).expanduser()
        elif root:
            out[element] = Path(root).expanduser() / variants.get(element, element) / "POTCAR"
        else:
            raise ValueError(f"No POTCAR for {element}: set vasp.potcars.{element} or vasp.potcar_dir")
    return out


def assemble_potcar(campaign, meta, config=None):
    """Validate one dataset per species, cache the concatenation, return (path, provenance)."""
    campaign = Path(campaign).resolve()
    config = config or read_json(campaign / "config.json")
    species = state_species(meta, config)
    paths = potcar_paths(config, species)
    variants = config.get("vasp", {}).get("potcar_variants") or {}
    parts, records = [], []
    for element in species:
        path = paths[element]
        if not path.is_file():
            raise FileNotFoundError(f"Licensed PAW dataset unavailable here: {path}. Assemble on your cluster.")
        data = path.read_bytes()
        text = data.decode("latin-1")
        labels = re.findall(r"TITEL\s*=\s*(.+)", text)
        rhfin = re.findall(r"RHFIN\s*=\s*([A-Za-z]+)\s*:", text)
        if len(labels) != 1 or len(rhfin) != 1 or rhfin[0] != element:
            raise ValueError(f"Expected one {element} dataset in {path}; TITEL={labels}, RHFIN={rhfin}")
        if "PAW" not in labels[0] or "End of Dataset" not in text:
            raise ValueError(f"Expected a complete PAW dataset: {path}")
        variant = variants.get(element)
        if variant and not re.search(rf"\b{re.escape(variant)}\b", labels[0]):
            raise ValueError(f"vasp.potcar_variants asks for {variant}, but TITEL is {labels[0]}")
        digest = hashlib.sha256(data).hexdigest()
        records.append({"element": element, "path": str(path), "titel": labels[0].strip(), "sha256": digest})
        parts.append(data + (b"\n" if not data.endswith(b"\n") else b""))
    combined = b"".join(parts)
    digest = hashlib.sha256(combined).hexdigest()
    cache = campaign / "potcars" / digest
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "POTCAR"
    if target.exists() and file_hash(target) != digest:
        raise ValueError(f"POTCAR cache hash mismatch: {target}")
    if not target.exists():
        atomic_bytes(target, combined)
    provenance = {"species_order": species, "sha256": digest, "datasets": records}
    write_json(cache / "provenance.json", provenance)
    return target, provenance
