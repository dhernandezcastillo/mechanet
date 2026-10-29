"""Harmonic thermochemistry, reservoir chemical potentials and cycle descriptors.

Energies are eV per simulation cell or per reservoir molecule. For an edge ``i -> j`` with
reservoir vector ``nu`` (amount produced of each species, see ``mechanet.reactions``) and
``n_e = sum(nu * charge)`` electrons released,

    DeltaG = G_j - G_i + sum_s nu_s * mu_s + n_e * mu_e,     mu_e = -e U_SHE.

The chemical potentials follow the computational hydrogen electrode (CHE) and its analogues:

* ``mu(H+) + mu(e-) = 1/2 mu(H2) - e U_RHE``, with ``U_RHE = U_SHE + (kT ln10 / e) pH``;
* ``mu(H2O)`` from the user's E + Gcorr (liquid-water equilibrium pressure), plus kT ln a(H2O);
* ``mu(O2)`` from the water-formation convention (4.92 eV at 298.15 K) or an explicit value,
  plus kT ln a(O2);
* ``mu(Cl-) - mu(e-) = 1/2 mu(Cl2) + e (U_SHE - U0) + kT ln a(Cl-)`` with U0 = 1.358 V
  (Cl2/Cl-), and ``mu(Cl2)`` from the user's E + Gcorr at ``p(Cl2)``.

The water-formation free energy, U0(Cl2/Cl-) and the liquid-water equilibrium pressure are
298.15 K values; at another temperature the analysis warns unless the references (or the
thermodynamics settings) declare values for that temperature.

No rate, barrier or constant-potential electronic energy is inferred.
"""

from __future__ import annotations

import math

from .reactions import SPECIES, electrons as vector_electrons

KB_EV_K = 8.617333262145e-5
HC_EV_CM = 1.2398419843320026e-4
LN10 = math.log(10.0)
U0_CL2_CL = 1.358  # V vs SHE, Cl2(g) + 2 e- -> 2 Cl-(aq), standard state, 298.15 K
DU0_CL2_CL_DT = -1.25e-3  # V/K, from the standard entropies (Cl2 + H2 -> 2 HCl(aq): -241 J/mol/K)
S_H2_298_J_MOL_K = 130.68
EV_PER_J_MOL = 1.0 / 96485.33212
STANDARD_TEMPERATURE_K = 298.15
TEMPERATURE_TOLERANCE_K = 0.5
WATER_EQUILIBRIUM_PRESSURE_BAR = 0.035  # CHE convention at 298.15 K [Norskov 2004]
# Free energy of 2 H2O(l) -> O2 + 2 H2 from tabulated standard enthalpies and entropies
# (571.66 kJ/mol, 326.61 J/mol/K): 4.916 eV at 298.15 K, used only to estimate other temperatures.
WATER_SPLITTING_DH_EV = 571.66e3 * EV_PER_J_MOL
WATER_SPLITTING_DS_EV_K = 326.61 * EV_PER_J_MOL
# One tolerance (eV, and V for potential thresholds) decides "downhill" everywhere.
DOWNHILL_TOLERANCE_EV = 1e-8


def water_vapour_pressure_bar(temperature_K):
    """Saturation vapour pressure of water (Antoine equation, NIST parameters for 256-373 K)."""
    return 10.0 ** (4.6543 - 1435.264 / (float(temperature_K) - 64.848))


def _finite(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Vibrations
# ---------------------------------------------------------------------------


def _harmonic(positive_cm1, temperature_K):
    energies = [value * HC_EV_CM for value in positive_cm1]
    zpe = 0.5 * sum(energies)
    if temperature_K == 0:
        return {"ZPE_eV": zpe, "F_vib_eV": zpe, "thermal_free_energy_eV": 0.0}
    kT = KB_EV_K * temperature_K
    # -expm1(-x) is accurate for soft modes, unlike 1 - exp(-x).
    thermal = kT * sum(math.log(-math.expm1(-e / kT)) for e in energies)
    return {"ZPE_eV": zpe, "F_vib_eV": zpe + thermal, "thermal_free_energy_eV": thermal}


def vibrational_thermochemistry(
    frequencies_cm1,
    temperature_K=298.15,
    low_frequency_cutoff_cm1=100.0,
    imaginary_tolerance_cm1=0.0,
    expected_modes=None,
):
    """Raw and regularized harmonic corrections with explicit validity.

    The default rejects every imaginary active mode. A positive imaginary tolerance is an
    explicit user choice: modes in [-tolerance, 0] use the positive frequency floor, are
    recorded individually and are never silently deleted.
    """
    temperature_K = float(temperature_K)
    cutoff = float(low_frequency_cutoff_cm1)
    tolerance = float(imaginary_tolerance_cm1)
    if temperature_K < 0 or cutoff < 0 or tolerance < 0:
        raise ValueError("Temperature, frequency floor and imaginary tolerance must be nonnegative.")
    values = [_finite(x) for x in frequencies_cm1]
    errors, warnings = [], []
    if not values and expected_modes == 0:
        # An empty active region (the clean reference of a metal surface, whose metal atoms
        # stay frozen in every node): no modes, F_vib = 0 by construction.
        zero = {"ZPE_eV": 0.0, "F_vib_eV": 0.0, "thermal_free_energy_eV": 0.0}
        return {
            "valid": True,
            "model": "partial_Hessian_harmonic_active_region",
            "temperature_K": temperature_K,
            "low_frequency_cutoff_cm1": cutoff,
            "imaginary_tolerance_cm1": tolerance,
            "n_modes": 0,
            "imaginary_modes": [],
            "regularized_modes": [],
            "raw": zero,
            "regularized": zero,
            "F_vib_eV": 0.0,
            "ZPE_eV": 0.0,
            "floor_sensitivity_eV": 0.0,
            "floored_modes_F_vib_eV": 0.0,
            "errors": [],
            "warnings": ["No frequency-active atoms: F_vib = 0 by construction."],
        }
    if not values:
        errors.append("No active vibrational modes are available.")
    if any(x is None for x in values):
        errors.append("A frequency is missing or nonfinite.")
    if expected_modes is not None and len(values) != int(expected_modes):
        errors.append("Mode count differs from the selected Hessian dimension.")
    numerical = [x for x in values if x is not None]
    imaginary = [{"mode": i + 1, "cm1": x} for i, x in enumerate(numerical) if x < 0]
    significant = [row for row in imaginary if abs(row["cm1"]) > tolerance]
    if significant:
        errors.append("Imaginary active modes exceed the configured tolerance.")
    overridden = [row for row in imaginary if abs(row["cm1"]) <= tolerance]
    if overridden:
        warnings.append(
            "Small imaginary modes were explicitly regularized to the configured floor; verify "
            "displacement/force convergence before accepting a minimum."
        )
    if any(x <= 0 for x in numerical) and cutoff <= 0:
        errors.append("Nonpositive modes require a positive frequency floor, or a better-converged Hessian.")
    raw = _harmonic(numerical, temperature_K) if numerical and all(x > 0 for x in numerical) else None
    floored = [max(x, cutoff) for x in numerical]
    floor_result = _harmonic(floored, temperature_K) if floored and all(x > 0 for x in floored) else None
    regularized = [
        {"mode": i + 1, "original_cm1": x, "used_cm1": max(x, cutoff)} for i, x in enumerate(numerical) if x < cutoff
    ]
    if regularized:
        warnings.append(f"{len(regularized)} modes use the explicit {cutoff:g} cm^-1 frequency floor.")
    floored_part = (
        _harmonic([row["used_cm1"] for row in regularized], temperature_K)["F_vib_eV"]
        if regularized and cutoff > 0
        else 0.0
    )
    valid = not errors and floor_result is not None
    return {
        "valid": valid,
        "model": "partial_Hessian_harmonic_active_region",
        "temperature_K": temperature_K,
        "low_frequency_cutoff_cm1": cutoff,
        "imaginary_tolerance_cm1": tolerance,
        "n_modes": len(values),
        "imaginary_modes": imaginary,
        "regularized_modes": regularized,
        "raw": raw,
        "regularized": floor_result if valid else None,
        "F_vib_eV": floor_result["F_vib_eV"] if valid else None,
        "ZPE_eV": floor_result["ZPE_eV"] if valid else None,
        "floor_sensitivity_eV": (floor_result["F_vib_eV"] - raw["F_vib_eV"]) if valid and raw is not None else None,
        "floored_modes_F_vib_eV": floored_part if valid else None,
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Molecular references
# ---------------------------------------------------------------------------


def _implied_h2_frequency_cm1(g_h2, temperature_K, with_cp):
    """H2 stretch implied by a Gcorr(H2) at 1 bar, for ZPE + int Cp dT - TS (with_cp) or ZPE - TS."""
    kT = KB_EV_K * temperature_K
    entropy = (S_H2_298_J_MOL_K + 3.5 * 8.314462618 * math.log(temperature_K / 298.15)) * EV_PER_J_MOL
    zpe = g_h2 + temperature_K * entropy - (3.5 * kT if with_cp else 0.0)
    return 2.0 * zpe / HC_EV_CM if zpe > 0 else None


def _water_standard_state_check(references, temperature_K, water_equilibrium_pressure_bar):
    """Consistency diagnostic for the H2 and H2O corrections (no automatic change).

    With one convention for both molecules (H2 at 1 bar, H2O at the liquid-water equilibrium
    pressure) and PBE frequencies, Gcorr(H2O) - Gcorr(H2) is about +0.03 to +0.04 eV. A larger
    difference has two common causes: H2O evaluated as vapour at 1 bar, or a too-small H2
    zero-point energy from a frequency run with one H held fixed.
    """
    h2, h2o = references.get("H2", {}), references.get("H2O", {})
    if not isinstance(h2, dict) or not isinstance(h2o, dict):
        return None
    g_h2 = _finite(h2.get("Gcorr", h2.get("free_energy_correction_eV")))
    g_h2o = _finite(h2o.get("Gcorr", h2o.get("free_energy_correction_eV")))
    if g_h2 is None or g_h2o is None:
        return None
    kT = KB_EV_K * temperature_K
    declared = _finite(h2o.get("Gcorr_pressure_bar"))
    moved = kT * math.log(water_equilibrium_pressure_bar / declared) if declared and declared > 0 else 0.0
    difference = g_h2o - g_h2
    at_equilibrium = difference + moved
    implied = {
        "ZPE_plus_Cp_minus_TS": _implied_h2_frequency_cm1(g_h2, temperature_K, True),
        "ZPE_minus_TS": _implied_h2_frequency_cm1(g_h2, temperature_K, False),
    }
    inconsistent = not (0.0 <= at_equilibrium <= 0.07)
    advice = None
    if inconsistent:
        nu_cp, nu_plain = implied["ZPE_plus_Cp_minus_TS"], implied["ZPE_minus_TS"]
        frequency_text = (
            f"The supplied Gcorr(H2) implies an H2 stretch of about {nu_cp:.0f} cm^-1 for ZPE + int Cp dT - TS "
            f"at 1 bar ({nu_plain:.0f} cm^-1 for ZPE - TS); PBE gives about 4300-4350 cm^-1, and a "
            "frequency run with one H held fixed gives about 3060 cm^-1."
            if nu_cp and nu_plain
            else ""
        )
        causes = (
            "Recompute the molecular frequencies with every atom free and one thermochemistry recipe "
            "(H2 at 1 bar, H2O at the declared pressure)."
            if declared
            else "Either the H2O correction is for vapour at 1 bar (then declare Gcorr_pressure_bar: 1.0), "
            "or one of the corrections is inconsistent, e.g. a frequency computed with atoms held fixed."
        )
        advice = (
            f"Gcorr(H2O) - Gcorr(H2) = {at_equilibrium:+.3f} eV with H2O at {water_equilibrium_pressure_bar:g} bar; "
            f"one convention for both molecules gives 0 to +0.07 eV. {causes} {frequency_text}"
        ).strip()
    return {
        "Gcorr_H2O_minus_Gcorr_H2_eV": difference,
        "difference_with_H2O_at_equilibrium_pressure_eV": at_equilibrium,
        "H2O_declared_pressure_bar": declared,
        "consistent_convention_range_eV": [0.0, 0.07],
        "one_bar_to_equilibrium_shift_eV": kT * math.log(water_equilibrium_pressure_bar),
        "water_equilibrium_pressure_bar": water_equilibrium_pressure_bar,
        "implied_H2_stretch_cm1": implied,
        "flag": "molecular_corrections_inconsistent" if inconsistent else None,
        "advice": advice,
    }


def normalize_references(references, temperature_K):
    """Accept the full format or the compact {references: {H2: {E, Gcorr}, ...}} format."""
    warnings = []
    if isinstance(references.get("references"), dict):
        original = references
        references = dict(original["references"])
        for key in (
            "temperature_K",
            "potential_reference",
            "protocol_id",
            "oxygen_reference",
            "chlorine_reference",
            "provenance",
        ):
            if key in original:
                references[key] = original[key]
        assumed = []
        if "temperature_K" not in references:
            references["temperature_K"] = temperature_K
            assumed.append("the analysis temperature")
        if isinstance(references.get("H2O"), dict) and "phase" not in references["H2O"]:
            references["H2O"] = {**references["H2O"], "phase": "liquid"}
            assumed.append("H2O as the CHE liquid-water reservoir")
        if "oxygen_reference" not in references:
            references["oxygen_reference"] = {
                "method": "water_formation",
                "delta_G_2H2O_to_O2_2H2_eV": 4.92,
                "temperature_K": 298.15,
            }
            assumed.append("the 4.92 eV water-formation convention at 298.15 K")
        if assumed:
            warnings.append(
                "Compact E/Gcorr references without complete standard-state metadata: assumed "
                + ", ".join(assumed)
                + ". Verify these assumptions."
            )
    return references, warnings


def reservoir_chemical_potentials(
    references, temperature_K=298.15, water_equilibrium_pressure_bar=0.035, needed=("H2", "H2O")
):
    """Validate user-supplied molecular references; never invent them.

    Returns mu(H2), mu(H2O), mu(O2) and, when Cl2 is supplied, mu(Cl2), plus errors and
    warnings. ``needed`` lists molecules whose absence is an error.
    """
    errors, warnings, notes = [], [], []
    if not (0 < float(water_equilibrium_pressure_bar) <= 1.01325):
        raise ValueError("water_equilibrium_pressure_bar must be in (0, 1.01325]")
    references, assumed = normalize_references(references, temperature_K)
    warnings += assumed
    temperature = _finite(references.get("temperature_K"))
    if temperature is None or abs(temperature - temperature_K) > 1e-6:
        errors.append("Reference temperature must equal the requested analysis temperature.")
    if not protocol_documented(references.get("protocol_id")):
        warnings.append(
            "Reference DFT protocol identity has not been documented; verify functional, PAWs, U, "
            "dispersion and solvation conventions."
        )
    provenance = references.get("provenance", {}) or {}
    for flag, explanation in (
        ("temperature_verified", "The temperature of the supplied molecular corrections is an unverified assumption."),
        (
            "water_standard_state_verified",
            "The liquid-water standard-state interpretation of the supplied H2O correction is unverified.",
        ),
        (
            "DFT_protocol_compatibility_verified",
            "Compatibility of the molecular reference calculations with the surface DFT protocol is unverified.",
        ),
    ):
        if flag == "water_standard_state_verified" and "H2O" not in needed:
            continue  # no water in the analysis (HER, say)
        if provenance.get(flag) is False:
            warnings.append(explanation)
    for name, entry in references.items():
        # `references --collect` records the temperature of each correction it computes.
        at = _finite(entry.get("Gcorr_temperature_K")) if isinstance(entry, dict) else None
        if at is not None and abs(at - temperature_K) > TEMPERATURE_TOLERANCE_K and name in needed:
            errors.append(
                f"{name}: the free-energy correction was computed at {at:g} K, the analysis runs at "
                f"{temperature_K:g} K; collect it again at this temperature."
            )

    def species_mu(name, required=True):
        entry = references.get(name)
        if not isinstance(entry, dict):
            if required:
                errors.append(f"Missing {name} reference.")
            return None
        mu = _finite(entry.get("G_eV"))
        if mu is not None:
            return mu
        energy = _finite(entry.get("energy_e0_eV", entry.get("E")))
        correction = _finite(entry.get("free_energy_correction_eV", entry.get("Gcorr")))
        if energy is None or correction is None:
            if required:  # an unused entry left empty (H2O for HER, say) is no error
                errors.append(f"{name} needs G_eV, or both energy_e0_eV and free_energy_correction_eV.")
            return None
        return energy + correction

    hydrogen = species_mu("H2", "H2" in needed)
    water = species_mu("H2O", "H2O" in needed)
    water_check = _water_standard_state_check(references, temperature_K, float(water_equilibrium_pressure_bar))
    pressure_correction = 0.0
    declared = references.get("H2O", {}).get("Gcorr_pressure_bar") if isinstance(references.get("H2O"), dict) else None
    if declared is not None:
        declared = _finite(declared)
        if declared is None or declared <= 0:
            errors.append("H2O.Gcorr_pressure_bar must be a positive pressure in bar.")
        elif water is not None:
            pressure_correction = KB_EV_K * temperature_K * math.log(float(water_equilibrium_pressure_bar) / declared)
            water += pressure_correction
            if not math.isclose(declared, float(water_equilibrium_pressure_bar), rel_tol=1e-9):
                notes.append(
                    f"mu(H2O) moved from {declared:g} bar (the declared Gcorr_pressure_bar) to the liquid-water "
                    f"pressure {float(water_equilibrium_pressure_bar):g} bar: {pressure_correction:+.4f} eV."
                )
    if water_check and water_check["flag"]:
        warnings.append(water_check["advice"])
    if water is not None and str(references.get("H2O", {}).get("phase", "")).lower() != "liquid":
        errors.append("H2O chemical potential must be explicitly for liquid water (phase: liquid).")
    off_standard = abs(temperature_K - STANDARD_TEMPERATURE_K) > TEMPERATURE_TOLERANCE_K
    if (
        "H2O" in needed
        and off_standard
        and abs(float(water_equilibrium_pressure_bar) - WATER_EQUILIBRIUM_PRESSURE_BAR) < 1e-12
    ):
        p_t = water_vapour_pressure_bar(temperature_K)
        shift = KB_EV_K * temperature_K * math.log(p_t / WATER_EQUILIBRIUM_PRESSURE_BAR)
        warnings.append(
            f"Liquid water is represented by its vapour at {WATER_EQUILIBRIUM_PRESSURE_BAR:g} bar, the 298.15 K "
            f"equilibrium pressure; at {temperature_K:g} K the vapour pressure of water is about {p_t:.3g} bar "
            f"(mu(H2O) {shift:+.3f} eV). Set thermodynamics.water_equilibrium_pressure_bar for this temperature and "
            "declare H2O.Gcorr_pressure_bar so that the analysis moves mu(H2O)."
        )
    oxygen_spec = references.get("oxygen_reference", {}) or {}
    method = oxygen_spec.get("method", "water_formation")
    oxygen = None
    if method == "water_formation":
        reaction = _finite(oxygen_spec.get("delta_G_2H2O_to_O2_2H2_eV"))
        oxygen_temperature = _finite(oxygen_spec.get("temperature_K", temperature))
        if reaction is None:
            errors.append("Water-formation convention needs delta_G_2H2O_to_O2_2H2_eV (e.g. 4.92 eV at 298.15 K).")
        if oxygen_temperature is None:
            errors.append("oxygen_reference.temperature_K must be a temperature in K.")
        elif reaction is not None and abs(oxygen_temperature - temperature_K) > TEMPERATURE_TOLERANCE_K:
            estimate = WATER_SPLITTING_DH_EV - temperature_K * WATER_SPLITTING_DS_EV_K
            warnings.append(
                f"The water-formation free energy ({reaction:g} eV) is given at {oxygen_temperature:g} K but the "
                f"analysis runs at {temperature_K:g} K: mu(O2) and U_eq(OER) carry the mismatch (tabulated standard "
                f"enthalpy and entropy give about {estimate:.3f} eV at {temperature_K:g} K). Declare "
                "oxygen_reference.delta_G_2H2O_to_O2_2H2_eV and its temperature_K for the analysis temperature."
            )
        if reaction is not None and water is not None and hydrogen is not None:
            oxygen = reaction + 2 * water - 2 * hydrogen
    elif method == "explicit":
        oxygen = species_mu("O2")
    else:
        errors.append("oxygen_reference.method must be water_formation or explicit.")
    chlorine = species_mu("Cl2", "Cl2" in needed)
    if chlorine is not None and str(references.get("Cl2", {}).get("phase", "gas")).lower() != "gas":
        errors.append("Cl2 reference must be the gas-phase molecule (phase: gas).")
    chlorine_spec = references.get("chlorine_reference") or {}
    u0_chlorine = _finite(chlorine_spec.get("U0_V", U0_CL2_CL)) if isinstance(chlorine_spec, dict) else None
    u0_temperature = (
        _finite(chlorine_spec.get("temperature_K", STANDARD_TEMPERATURE_K)) if isinstance(chlorine_spec, dict) else None
    )
    if u0_chlorine is None or u0_temperature is None:
        errors.append("chlorine_reference needs a numeric U0_V (V vs SHE) and temperature_K.")
    elif "Cl2" in needed and abs(u0_temperature - temperature_K) > TEMPERATURE_TOLERANCE_K:
        estimate = U0_CL2_CL + DU0_CL2_CL_DT * (temperature_K - STANDARD_TEMPERATURE_K)
        warnings.append(
            f"U0(Cl2/Cl-) = {u0_chlorine:g} V is given at {u0_temperature:g} K but the analysis runs at "
            f"{temperature_K:g} K (about {estimate:.3f} V there, -1.25 mV/K): mu(Cl-) and U_eq(CER) carry the "
            "mismatch. Declare chlorine_reference: {U0_V: ..., temperature_K: ...} in the references file."
        )
    # Molecules of plugin species (N2, NH3, CO2, ...): E0 + G_corr as supplied.
    others = {f"mu_{name}_eV": species_mu(name) for name in sorted(set(needed) - {"H2", "H2O", "O2", "Cl2"})}
    overall = oxygen + 2 * hydrogen - 2 * water if None not in (oxygen, hydrogen, water) else None
    return {
        "valid": not errors,
        "mu_H2_eV": hydrogen,
        "mu_H2O_eV": water,
        "mu_O2_eV": oxygen,
        "mu_Cl2_eV": chlorine,
        "U0_Cl2_Cl_V": u0_chlorine,
        **others,
        "overall_OER_at_0V_eV": overall,
        "temperature_K": temperature_K,
        "oxygen_method": method,
        "H2O_pressure_correction_eV": pressure_correction,
        "water_standard_state_check": water_check,
        "errors": errors,
        "warnings": warnings,
        "notes": notes,
    }


def protocol_documented(value):
    """False for an empty ``protocol_id`` or a placeholder ("user", "REPLACE with ...", "REPLACE_ME")."""
    import re

    text = str(value or "").strip().lower()
    return bool(text) and text not in ("user", "none", "null", "todo", "tbd") and not re.match(r"replace(\b|_)", text)


# ---------------------------------------------------------------------------
# Conditions and reservoir free energies
# ---------------------------------------------------------------------------


def validate_activities(activities):
    """Activities by species name; unknown names and H+ are errors (H+ is set through the pH)."""
    checked = {}
    for name, value in (activities or {}).items():
        name = str(name).strip()
        if name == "H+":
            raise ValueError("The H+ activity is set through the pH (--pH or thermodynamics.pH), not as an activity")
        if name not in SPECIES:
            valid = ", ".join(sorted(s for s in SPECIES if s != "H+"))
            raise ValueError(f"Unknown species {name!r} in the activities; valid names: {valid}")
        number = _finite(value)
        if number is None or number <= 0:
            raise ValueError(f"The activity of {name} must be a positive number, got {value!r}")
        checked[name] = number
    return checked


class Conditions:
    """pH, temperature and activities shared by every reaction of an analysis.

    Activities (relative to each species' standard state: 1 bar for gases, 1 M for ions, pure
    liquid water) enter as kT ln a for H2O, O2, Cl-, Cl2 and neutral plugin species. They do
    not move the electrode scale: H+ is set through the pH, and the CHE electrode stays at
    p(H2) = 1 bar.
    """

    def __init__(self, temperature_K=298.15, pH=0.0, activities=None):
        self.temperature_K = float(temperature_K)
        self.pH = float(pH)
        self.activities = {"Cl-": 1.0, "Cl2": 1.0, **validate_activities(activities)}

    @property
    def kT(self):
        return KB_EV_K * self.temperature_K

    @property
    def nernst(self):
        """kT ln(10)/e in V (0.05916 V at 298.15 K)."""
        return self.kT * LN10

    def to_she(self, potential, scale):
        scale = str(scale).upper()
        if scale == "SHE":
            return float(potential)
        if scale == "RHE":
            return float(potential) - self.nernst * self.pH
        raise ValueError(f"Unknown potential scale {scale!r}; use RHE or SHE")

    def from_she(self, potential_she, scale):
        scale = str(scale).upper()
        if scale == "SHE":
            return float(potential_she)
        if scale == "RHE":
            return float(potential_she) + self.nernst * self.pH
        raise ValueError(f"Unknown potential scale {scale!r}; use RHE or SHE")

    def describe(self):
        return {"temperature_K": self.temperature_K, "pH": self.pH, "activities": dict(self.activities)}


def species_free_energy(name, reservoirs, conditions):
    """mu of a reservoir species at U_SHE = 0 (electrons are added separately)."""
    from .reactions import BUILTIN_SPECIES, SPECIES

    species = SPECIES.get(name)
    if species is not None and name not in BUILTIN_SPECIES:
        if species.chemical_potential is not None:
            mu = species.chemical_potential(reservoirs, conditions)
        else:
            base = reservoirs.get(f"mu_{name}_eV")
            mu = None if base is None else base + conditions.kT * math.log(conditions.activities.get(name, 1.0))
        if mu is None:
            raise ValueError(f"The references do not define mu({name}); add the molecule to the references file")
        return float(mu)

    def log_activity(species):
        return conditions.kT * math.log(conditions.activities.get(species, 1.0))

    if name == "H2O":
        mu = None if reservoirs.get("mu_H2O_eV") is None else reservoirs["mu_H2O_eV"] + log_activity("H2O")
    elif name == "H+":
        mu = (
            None
            if reservoirs.get("mu_H2_eV") is None
            else 0.5 * reservoirs["mu_H2_eV"] - conditions.nernst * conditions.pH
        )
    elif name == "O2":
        # the water-formation convention fixes mu(O2) at 1 bar from pure water: a(H2O) does not enter
        mu = None if reservoirs.get("mu_O2_eV") is None else reservoirs["mu_O2_eV"] + log_activity("O2")
    elif name == "Cl2":
        mu = None if reservoirs.get("mu_Cl2_eV") is None else reservoirs["mu_Cl2_eV"] + log_activity("Cl2")
    elif name == "Cl-":
        u0 = reservoirs.get("U0_Cl2_Cl_V")
        mu = (
            None
            if reservoirs.get("mu_Cl2_eV") is None
            else 0.5 * reservoirs["mu_Cl2_eV"] - (U0_CL2_CL if u0 is None else u0) + log_activity("Cl-")
        )
    else:
        raise ValueError(f"No chemical potential rule for reservoir species {name!r}")
    if mu is None:
        raise ValueError(f"The references do not define mu({name}); add the molecule to the references file")
    return mu


def vector_free_energy(vector, reservoirs, conditions, potential_she):
    """sum_s nu_s mu_s + n_e mu_e for a reservoir vector at a potential on the SHE scale."""
    total = sum(float(nu) * species_free_energy(name, reservoirs, conditions) for name, nu in vector.items())
    return total - vector_electrons(vector) * float(potential_she)


def edge_free_energy(G_i, G_j, edge, reservoirs, potential_V, conditions=None, scale="RHE"):
    """CHE free energy of an edge at ``potential_V`` on ``scale`` (RHE by default)."""
    if not reservoirs.get("valid"):
        raise ValueError("Reservoir chemical potentials are unresolved.")
    conditions = conditions or Conditions(reservoirs.get("temperature_K", 298.15))
    vector = (
        edge.get("reservoirs")
        if "reservoirs" in edge
        else {"H2O": -edge["water"], "H+": edge["pairs"], "O2": edge["oxygen"]}
    )
    return (
        float(G_j)
        - float(G_i)
        + vector_free_energy(vector, reservoirs, conditions, conditions.to_she(potential_V, scale))
    )


def formation_free_energy(G_node, G_reference, vector, reservoirs, potential_V, conditions=None, scale="RHE"):
    """Free energy of reference + reservoirs -> node, with the formation vector of the reaction."""
    conditions = conditions or Conditions(reservoirs.get("temperature_K", 298.15))
    return (
        float(G_node)
        - float(G_reference)
        + vector_free_energy(vector, reservoirs, conditions, conditions.to_she(potential_V, scale))
    )


# ---------------------------------------------------------------------------
# Cycle descriptors
# ---------------------------------------------------------------------------


def exner_gmax(profile):
    """Gmax after Exner (ACS Catal. 2020; Mater. Horiz. 2023): the largest G_j - G_i over every
    ordered pair i < j of the cycle profile (starting right after product release), excluding
    the overall reaction (first -> last). Returns (value, (i, j))."""
    n = len(profile) - 1
    best, pair = None, None
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if i == 0 and j == n:
                continue
            value = profile[j] - profile[i]
            if best is None or value > best:
                best, pair = value, (i, j)
    return best, pair


def cyclic_span(profile):
    """Largest free-energy span over the repeating cycle (resting-state independent)."""
    n = len(profile) - 1
    net = profile[-1]
    best, pair = None, None
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            value = profile[j] - profile[i] + (net if j < i else 0.0)
            if best is None or value > best:
                best, pair = value, (i, j)
    return best, pair


def release_last_rotation(ordered_edges, release):
    """Offset that rotates a cycle into the turnover convention.

    The cycle's net amount of the turnover marker ``release`` decides the convention: a
    product released once per turnover (OER: O2) puts its releasing step last, so the cycle
    starts right after the release; a reactant consumed once per turnover (ORR: O2) puts its
    consuming step first. Returns 0 unless exactly one step moves the marker that way.
    """
    amounts = [float((e.get("reservoirs") or {}).get(release, 0)) for e in ordered_edges]
    total = sum(amounts)
    if abs(total) < 1e-12:
        return 0
    sign = 1 if total > 0 else -1
    marked = [k for k, x in enumerate(amounts) if x * sign > 0]
    if len(marked) != 1:
        return 0
    return (marked[0] + 1) % len(ordered_edges) if sign > 0 else marked[0]


def limiting_potential(delta_g_at_0, electrons_per_edge, tolerance=DOWNHILL_TOLERANCE_EV):
    """(U_L, (lowest, highest) potential at which every step runs downhill), or (None, None).

    A step with n electrons released has DeltaG(U) = DeltaG(0) - n U. Steps in the cycle's
    direction (n of the sign of the cycle's electrons) are downhill above (anodic) or below
    (cathodic) DeltaG(0)/n; steps against it the other way round; chemical steps must be
    downhill on their own. U_L is the potential closest to equilibrium at which every step is
    downhill, provided the steps against the direction still are there (otherwise None).
    """
    total = sum(electrons_per_edge)
    if total == 0:
        return None, None
    sign = 1 if total > 0 else -1
    if any(n == 0 and dg > tolerance for n, dg in zip(electrons_per_edge, delta_g_at_0)):
        return None, None
    # in s = sign * U, a forward step needs s >= t and a reverse step s <= t, with t = sign * DeltaG(0) / n
    forward = [sign * dg / n for n, dg in zip(electrons_per_edge, delta_g_at_0) if sign * n > 0]
    reverse = [sign * dg / n for n, dg in zip(electrons_per_edge, delta_g_at_0) if sign * n < 0]
    low = max(forward)
    high = min(reverse) if reverse else None
    if high is not None and low > high + tolerance:
        return None, None
    window = (sign * low, None if high is None else sign * high)
    return sign * low, (window if sign > 0 else (window[1], window[0]))


def cycle_metrics(
    ordered_edges, delta_g_at_0, potential_V, electrons_per_edge, equilibrium_potential_V, release="O2", scale="RHE"
):
    """Descriptors of one closed cycle at ``potential_V`` (on ``scale``).

    ``delta_g_at_0`` are the step free energies at 0 V on the same scale; a step with ``n``
    electrons changes by ``-n U``. The cycle is first rotated into the turnover convention
    (:func:`release_last_rotation`; ``rotation`` = number of leading edges moved to the end);
    indices in the result refer to that rotated order. U_L follows :func:`limiting_potential`;
    eta = U_L - U_eq for anodic cycles (electrons released) and U_eq - U_L for cathodic ones.
    """
    shift = release_last_rotation(ordered_edges, release)
    ordered_edges = list(ordered_edges[shift:]) + list(ordered_edges[:shift])
    delta_g_at_0 = list(delta_g_at_0[shift:]) + list(delta_g_at_0[:shift])
    electrons_per_edge = list(electrons_per_edge[shift:]) + list(electrons_per_edge[:shift])
    values = [value - n * float(potential_V) for n, value in zip(electrons_per_edge, delta_g_at_0)]
    profile = [0.0]
    for value in values:
        profile.append(profile[-1] + value)
    positive = max([0.0] + values)
    gmax, gmax_pair = exner_gmax(profile)
    span, span_pair = cyclic_span(profile)
    resting = min(range(len(profile) - 1), key=lambda k: (profile[k], k)) if len(profile) > 1 else 0
    total_electrons = sum(electrons_per_edge)
    anodic = total_electrons > 0
    chemical_uphill = any(n == 0 and dg > DOWNHILL_TOLERANCE_EV for n, dg in zip(electrons_per_edge, delta_g_at_0))
    against = sum(1 for n in electrons_per_edge if (n < 0 if anodic else n > 0))
    limit, window = limiting_potential(delta_g_at_0, electrons_per_edge)
    overpotential = None
    if limit is not None:
        overpotential = (limit - equilibrium_potential_V) if anodic else (equilibrium_potential_V - limit)
    return {
        "potential_V": float(potential_V),
        "potential_scale": scale,
        "step_deltaG_eV": values,
        "cumulative_deltaG_eV": profile,
        "largest_step_deltaG_eV": max(values) if values else None,
        "largest_uphill_step_eV": positive,
        "Gmax_U_eV": gmax,
        "Gmax_U_span_indices": list(gmax_pair) if gmax_pair else None,
        "cyclic_span_eV": span,
        "cyclic_span_indices": list(span_pair) if span_pair else None,
        "resting_state_index": resting,
        "net_deltaG_eV": sum(values),
        "limiting_potential_V": limit,
        "downhill_potential_range_V": list(window) if window else None,
        "thermodynamic_overpotential_V": overpotential,
        "equilibrium_potential_V": equilibrium_potential_V,
        "rotation": shift,
        "direction": "anodic" if anodic else "cathodic",
        "steps_against_reaction_direction": against,
        "potential_independent_uphill_step": chemical_uphill,
        "contains_condensed_multi_electron_step": any(abs(n) > 1 for n in electrons_per_edge),
        "interpretation": "thermodynamic_descriptor_not_activation_barrier_or_rate",
    }


def validate_network(network, registry=None):
    """Endpoints, element/charge balance of every edge and turnover totals of every cycle."""
    from .reactions import add, clean, element_change

    errors = []
    nodes = {n["id"]: n for n in network.get("nodes", [])}
    edges = {e["id"]: e for e in network.get("edges", [])}
    if len(nodes) != len(network.get("nodes", [])):
        errors.append("Duplicate node identifiers.")
    if len(edges) != len(network.get("edges", [])):
        errors.append("Duplicate edge identifiers.")
    for key, edge in edges.items():
        if edge.get("from") not in nodes or edge.get("to") not in nodes:
            errors.append(f"Edge {key}: missing endpoint.")
            continue
        left, right = nodes[edge["from"]]["delta_atoms"], nodes[edge["to"]]["delta_atoms"]
        vector = edge.get("reservoirs")
        if vector is None:
            vector = {"H2O": -edge["water"], "H+": edge["pairs"], "O2": edge["oxygen"]}
        try:
            implied = element_change(clean(vector))
        except ValueError as exc:
            errors.append(f"Edge {key}: {exc}")
            continue
        actual = {el: right.get(el, 0) - left.get(el, 0) for el in set(left) | set(right)}
        if any(abs(implied.get(el, 0) - actual.get(el, 0)) > 1e-8 for el in set(implied) | set(actual)):
            errors.append(f"Edge {key}: element balance fails (surface {actual}, reservoirs {implied}).")
        if "electrons" in edge and int(edge["electrons"]) != vector_electrons(clean(vector)):
            errors.append(f"Edge {key}: electron count disagrees with charge balance.")
    reactions = network.get("reactions") or {}
    cycle_ids = set()
    for cycle in network.get("cycles", []):
        cid = cycle.get("id")
        if cid in cycle_ids:
            errors.append(f"Duplicate cycle {cid}.")
        cycle_ids.add(cid)
        edge_ids = cycle.get("edge_ids", [])
        if not edge_ids or any(e not in edges for e in edge_ids):
            errors.append(f"Cycle {cid}: empty or missing edge.")
            continue
        ordered = [edges[e] for e in edge_ids]
        if any(ordered[i]["to"] != ordered[(i + 1) % len(ordered)]["from"] for i in range(len(ordered))):
            errors.append(f"Cycle {cid}: ordered edges do not form a closed cycle.")
        name = cycle.get("reaction") or network.get("reaction", "oer")
        net = (reactions.get(name) or {}).get("net")
        if net is None and registry is not None:
            net = registry.reaction(name).net
        if net is None and name == "oer":
            net = {"H2O": -2, "H+": 4, "O2": 1}
        total = add(
            *[
                (
                    e.get("reservoirs")
                    if e.get("reservoirs") is not None
                    else {"H2O": -e["water"], "H+": e["pairs"], "O2": e["oxygen"]}
                )
                for e in ordered
            ]
        )
        if net is not None and clean(total) != clean(net):
            errors.append(f"Cycle {cid}: totals {total} are not one {name} turnover {net}.")
    return {
        "valid": not errors,
        "errors": errors,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_cycles": len(network.get("cycles", [])),
    }


def required_molecules(reactions):
    names = set()
    for vector in reactions:
        for species in vector:
            if species in ("H2O",):
                names.add("H2O")
            if species in ("H+",):
                names.add("H2")
            if species in ("Cl-", "Cl2"):
                names.add("Cl2")
            if species == "O2":
                names.update(("H2", "H2O"))
    return names


__all__ = [
    "Conditions",
    "SPECIES",
    "cycle_metrics",
    "cyclic_span",
    "edge_free_energy",
    "exner_gmax",
    "formation_free_energy",
    "reservoir_chemical_potentials",
    "validate_network",
    "vector_free_energy",
    "vibrational_thermochemistry",
]
