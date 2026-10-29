"""Project files: one slab, its sites, the mechanism scope, the DFT protocol and a cluster.

``project.yaml`` records every decision of a study. ``mechanet init`` writes it from a slab
file (inspecting the slab and asking only what the structure cannot answer), and
``mechanet generate`` turns it into a campaign directory. Sections that configure the
calculations (``vasp``, ``frequency``, ``thermodynamics``, ...) are copied into the
campaign's ``config.json`` as written, completed with the defaults below; ``cluster`` is
resolved into the ``scheduler`` section and ``vasp.command``.

Layout written by ``mechanet init``::

    <project>/project.yaml
    <project>/slab/<slab file>, inspection.json, site_map.html, site_map.svg
    <project>/inputs/INCAR_loose, INCAR_final, KPOINTS, references.yaml
    <project>/campaign/            (written by `mechanet generate`)
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import sys
from importlib import resources
from pathlib import Path

import numpy as np

from . import __version__, elements
from .fileio import read_yaml, write_json

PROJECT_FILE = "project.yaml"
SCHEMA = 1
TOP_LEVEL = (
    "mechanet",
    "name",
    "reaction",
    "slab",
    "sites",
    "scope",
    "inputs",
    "vasp",
    "frequency",
    "refine",
    "thermo_selection",
    "thermodynamics",
    "analysis",
    "topology",
    "geometry",
    "cluster",
    "plugins",
    "origins",
    "notes",
)
SLAB_KEYS = (
    "file",
    "vacuum_axis",
    "surface_sign",
    "fixed",
    "frequency_atoms",
    "frequency_mask",
    "roles",
    "species_order",
    "spectator_molecules",
)
INPUT_KEYS = ("incar_loose", "incar_final", "kpoints")
# Keys each section accepts (a typo is an error, not a silently ignored setting).
SECTION_KEYS = {
    "scope": {"families", "pair_max_distance_A"},
    "inputs": set(INPUT_KEYS),
    "vasp": {
        "command",
        "common_overrides",
        "dft_u",
        "magmom_by_element",
        "magmom_by_id",
        "magmom_default",
        "max_attempts",
        "max_continuations",
        "potcar_dir",
        "potcar_variants",
        "potcars",
        "stage_overrides",
        "template_species_order",
    },
    "frequency": {
        "central",
        "method",
        "ediff",
        "potim",
        "nfree",
        "nelm",
        "max_force_eVA",
        "warn_force_eVA",
        "max_static_e0_shift_eV",
        "max_moment_change_muB",
        "delete_wavecar_after_use",
        "max_unit_failures_per_job",
        "prune_unit_outputs",
        "unit_prune_files",
    },
    "refine": {"ediff", "ediffg", "ibrion", "nsw"},
    "thermo_selection": {"energy_window_eV", "max_variants_per_node"},
    "thermodynamics": {
        "references",
        "potentials_V",
        "potential_scale",
        "pH",
        "temperature_K",
        "activities",
        "low_frequency_cutoff_cm1",
        "imaginary_tolerance_cm1",
        "water_equilibrium_pressure_bar",
        "electronic_mode_corrections",
    },
    "analysis": {"require_input_hashes", "max_eentro_eV", "moment_change_muB", "moment_site_radius_A"},
    "topology": {"shared_proton_max_OH_A", "hysteresis_A"},
    "geometry": {
        "symmetry_reduction",
        "symmetry_tolerance_A",
        "pair_max_distance_A",
        "anchor_min_M_O_X_angle_deg",
        "clash_floors_A",
        "coupled_pair_lift_A",
        "orientation_search",
        "side_on_OO",
    },
}

# Calculation settings written into every campaign config.json unless the project sets them.
DEFAULT_SECTIONS = {
    "vasp": {"max_attempts": 2, "max_continuations": 8, "common_overrides": {}, "stage_overrides": {}},
    "frequency": {
        "central": "final",
        "method": "displacements",
        "ediff": 1e-6,
        "potim": 0.015,
        "nfree": 2,
        "nelm": 120,
        "max_static_e0_shift_eV": 0.01,
        "max_moment_change_muB": 0.5,
        "delete_wavecar_after_use": True,
        "max_unit_failures_per_job": 3,
        "prune_unit_outputs": True,
    },
    "refine": {"ediff": 1e-6, "ediffg": -0.015, "ibrion": 2, "nsw": 60},
    "thermo_selection": {"energy_window_eV": 0.10, "max_variants_per_node": 2},
    "thermodynamics": {
        "temperature_K": 298.15,
        "pH": 0.0,
        "low_frequency_cutoff_cm1": 100.0,
        "imaginary_tolerance_cm1": 80.0,
        "water_equilibrium_pressure_bar": 0.035,
        "electronic_mode_corrections": {"per_H_eV": 0.30, "per_O_eV": 0.05},
    },
    "analysis": {"require_input_hashes": True, "max_eentro_eV": 0.02},
    "topology": {"shared_proton_max_OH_A": 1.6},
    "geometry": {"symmetry_reduction": True, "symmetry_tolerance_A": 0.005},
}

SCOPE_HELP = {
    "aem": "adsorbate evolution (*OH, *O, *OOH) on each terminal site and open metal",
    "aem+bifunctional": "AEM plus proton transfer to a neighbouring acceptor site",
    "core": "every core family of the reaction (the curated literature set)",
    "all": "every registered family of the reaction, including the expensive ones",
}


def _scope_presets(reaction, registry):
    core = registry.core_families(reaction)
    every = [
        f.name for f in sorted(registry.families.values(), key=lambda f: (f.order, f.name)) if f.reaction == reaction
    ]
    presets = {"core": core, "all": every}
    if reaction == "oer":
        presets = {"aem": ["aem"], "aem+bifunctional": ["aem", "bifunctional"], **presets}
    return presets


def available_presets():
    """Names of the packaged INCAR presets (data/presets/<name>/INCAR_loose, INCAR_final)."""
    base = resources.files("mechanet").joinpath("data/presets")
    return sorted(entry.name for entry in base.iterdir() if entry.joinpath("INCAR_loose").is_file())


def materials_project_defaults():
    """PAW dataset and Hubbard-U choices of the Materials Project input set (packaged table)."""
    text = resources.files("mechanet").joinpath("data/materials_project.json").read_text()
    return json.loads(text)


def kpoints_text(cell, vacuum_axis, spacing=0.25):
    """A Gamma-centred mesh with about ``spacing`` (1/A, 2*pi included) between points; 1 along the vacuum."""
    reciprocal = 2 * math.pi * np.linalg.inv(np.asarray(cell, dtype=float)).T
    mesh = [max(1, math.ceil(float(np.linalg.norm(reciprocal[i])) / spacing)) for i in range(3)]
    mesh[int(vacuum_axis)] = 1
    return (
        f"Gamma-centred mesh, {spacing} 1/A spacing (mechanet init; review for your system)\n0\nGamma\n"
        f"{mesh[0]} {mesh[1]} {mesh[2]}\n0 0 0\n"
    ), mesh


def references_template(reaction):
    text = resources.files("mechanet").joinpath("data/references_template.yaml").read_text()
    block = ""
    if reaction == "cer":
        block = (
            "Cl2:                         # CER: mu(Cl-) = 1/2 mu(Cl2) - e U0(Cl2/Cl-) + kT ln a(Cl-)\n"
            "  energy_e0_eV: null\n  free_energy_correction_eV: null\n  phase: gas\n\n"
            "chlorine_reference:          # U0(Cl2/Cl-) vs SHE; 1.358 V is the 298.15 K value (-1.25 mV/K)\n"
            "  U0_V: 1.358\n  temperature_K: 298.15\n\n"
        )
    return text.replace("{cl2_block}", block)


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def find_project(start=".", name=PROJECT_FILE):
    """project.yaml in ``start`` or its parents (None when there is none)."""
    here = Path(start).resolve()
    if here.is_file():
        return here
    for folder in (here, *here.parents):
        if (folder / name).is_file():
            return folder / name
    return None


def load_project(path=PROJECT_FILE):
    path = Path(path)
    if path.is_dir():
        path = path / PROJECT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"No project file {path}; create one with `mechanet init SLAB_FILE`")
    data = read_yaml(path)
    registry, problem = _project_registry(data, path.resolve().parent)
    if problem:
        # The plugin's own scope blocks cannot be checked without it: report the load error.
        errors = [problem] + [e for e in validate_project(data) if not e.startswith("unknown scope keys")]
    else:
        errors = validate_project(data, registry)
    if errors:
        raise ValueError(f"{path} is not a valid mechanet project:\n  - " + "\n  - ".join(errors))
    data = copy.deepcopy(data)
    data["_path"] = str(path.resolve())
    data["_dir"] = str(path.resolve().parent)
    return data


def _project_registry(data, folder):
    """(registry, problem): built-ins plus the project's plugin files, or the load error.

    The registry is None when the project lists no plugins (the built-ins apply) or when a
    plugin file is missing or fails to load (``problem`` then says why).
    """
    from .plugins import default_registry

    plugins = data.get("plugins") if isinstance(data, dict) else None
    if not isinstance(plugins, list) or not plugins or not all(isinstance(p, str) for p in plugins):
        return None, None
    paths = []
    for plugin in plugins:
        path = Path(plugin).expanduser()
        paths.append(str((path if path.is_absolute() else Path(folder) / path).resolve()))
    try:
        return default_registry(paths), None
    except (ValueError, FileNotFoundError, ImportError) as exc:
        return None, f"plugins: {exc}"


def validate_project(data, registry=None):
    """Problems of a project mapping (empty list when valid); file existence is checked too.

    ``registry`` supplies the scope blocks that mechanism families declare (``scope.coverage``,
    ``scope.enumerate``, blocks of plugin families); without it the built-in families are used.
    """
    if not isinstance(data, dict):
        return ["the file must contain a mapping"]
    errors = []
    if data.get("mechanet") != SCHEMA:
        errors.append(f"`mechanet: {SCHEMA}` (the schema version) must be the first line")
    unknown = sorted(set(data) - set(TOP_LEVEL), key=str)
    if unknown:
        errors.append(_unknown("top-level keys", unknown, TOP_LEVEL))
    if not isinstance(data.get("name"), str) or not data.get("name"):
        errors.append("`name` must be a non-empty string")
    if not isinstance(data.get("reaction"), str):
        errors.append("`reaction` must name a reaction (e.g. oer, cer)")
    slab = data.get("slab")
    if not isinstance(slab, dict) or not slab.get("file"):
        errors.append("`slab.file` (the slab structure) is required")
    else:
        bad = sorted(set(slab) - set(SLAB_KEYS), key=str)
        if bad:
            errors.append(_unknown("slab keys", bad, SLAB_KEYS))
        if slab.get("vacuum_axis") not in (0, 1, 2):
            errors.append("slab.vacuum_axis must be 0, 1 or 2")
        if slab.get("surface_sign") not in (1, -1):
            errors.append("slab.surface_sign must be 1 or -1")
        fixed = slab.get("fixed")
        if fixed is not None and not (
            isinstance(fixed, dict) and (fixed.get("source") == "file" or isinstance(fixed.get("fixed_ids"), list))
        ):
            errors.append("slab.fixed must be {source: file} or {fixed_ids: [...]}")
    sites = data.get("sites")
    if not isinstance(sites, list):
        errors.append("`sites` must be a list of {id, kind, native} entries")
    else:
        for k, site in enumerate(sites):
            if not isinstance(site, dict) or not {"id", "kind", "native"} <= set(site):
                errors.append(f"sites[{k}] needs id, kind and native")
    scope = data.get("scope")
    if not isinstance(scope, dict) or not isinstance(scope.get("families"), list) or not scope.get("families"):
        errors.append("`scope.families` must list at least one mechanism family")
    inputs = data.get("inputs")
    if not isinstance(inputs, dict) or any(not inputs.get(k) for k in INPUT_KEYS):
        errors.append(f"`inputs` must give {', '.join(INPUT_KEYS)}")
    for key in (
        "vasp",
        "frequency",
        "refine",
        "thermo_selection",
        "thermodynamics",
        "analysis",
        "topology",
        "geometry",
        "origins",
    ):
        if data.get(key) is not None and not isinstance(data[key], dict):
            errors.append(f"`{key}` must be a mapping")
    for section, known in SECTION_KEYS.items():
        if section == "scope":
            continue
        if isinstance(data.get(section), dict):
            bad = sorted(set(data[section]) - known, key=str)
            if bad:
                errors.append(_unknown(f"{section} keys", bad, known))
    if isinstance(scope, dict):
        errors += _scope_problems(scope, registry)
    errors += _value_problems(data)
    if data.get("plugins") is not None and not isinstance(data["plugins"], list):
        errors.append("`plugins` must be a list of plugin files")
    if data.get("cluster") is not None and not isinstance(data["cluster"], str):
        errors.append("`cluster` must be a profile name (`mechanet cluster list`) or a profile file")
    return errors


def _scope_problems(scope, registry=None):
    """Unknown keys of the scope section and of the family blocks it holds."""
    from .plugins import default_registry

    blocks = (registry or default_registry()).scope_blocks()
    known = set(SECTION_KEYS["scope"]) | set(blocks)
    errors = []
    bad = sorted(set(scope) - known, key=str)
    if bad:
        errors.append(_unknown("scope keys", bad, known))
    for key, family in sorted(blocks.items()):
        if key not in scope or scope[key] is None:
            continue
        block = scope[key]
        if not isinstance(block, dict):
            errors.append(f"scope.{key} configures the {family.name} family and must be a mapping")
            continue
        fields = set(family.scope_fields or ())
        if fields:
            wrong = sorted(set(block) - fields, key=str)
            if wrong:
                errors.append(_unknown(f"scope.{key} keys", wrong, fields))
    return errors


def _value_problems(data):
    """Values that would only fail later (at plan or submit) or be ignored silently."""
    from .clusters import _command_list, command_problems
    from .vasp import CENTRAL_MODES, FREQUENCY_METHODS, STAGES

    errors = []
    vasp = data.get("vasp") if isinstance(data.get("vasp"), dict) else {}
    overrides = vasp.get("stage_overrides")
    if overrides is not None:
        if not isinstance(overrides, dict):
            errors.append("vasp.stage_overrides must map stage names to INCAR tags, e.g. {01_loose: {NSW: 80}}")
        else:
            bad = sorted(set(map(str, overrides)) - set(STAGES))
            if bad:
                errors.append(_unknown("vasp.stage_overrides stages", bad, STAGES))
    for key in ("common_overrides", "potcar_variants", "magmom_by_element", "dft_u"):
        if vasp.get(key) is not None and not isinstance(vasp[key], dict):
            errors.append(f"vasp.{key} must be a mapping (or null)")
    command = vasp.get("command")
    if command is not None:
        if not isinstance(command, (str, list)) or (
            isinstance(command, list) and not all(isinstance(w, (str, int, float)) for w in command)
        ):
            problems = ["give a list of words, e.g. [srun, vasp_std], or a command line"]
        else:
            try:
                words = _command_list(command)
                problems = command_problems(words) if words else ["it is empty"]
            except ValueError as exc:
                problems = [str(exc)]
        errors += [f"vasp.command: {problem}" for problem in problems]
    frequency = data.get("frequency") if isinstance(data.get("frequency"), dict) else {}
    for key, allowed in (("central", CENTRAL_MODES), ("method", FREQUENCY_METHODS)):
        if key in frequency and frequency[key] not in allowed:
            errors.append(_unknown(f"frequency.{key} values", [frequency[key]], allowed))
    analysis = data.get("analysis") if isinstance(data.get("analysis"), dict) else {}
    for key in ("moment_change_muB", "moment_site_radius_A"):
        value = analysis.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0):
            errors.append(f"analysis.{key} must be a positive number, not {value!r}")
    thermo = data.get("thermodynamics") if isinstance(data.get("thermodynamics"), dict) else {}
    scale = thermo.get("potential_scale")
    if scale is not None and str(scale).upper() not in ("RHE", "SHE"):
        errors.append(f"thermodynamics.potential_scale must be RHE or SHE, not {scale!r}")
    return errors


def _unknown(what, bad, known):
    """'unknown <what> [...]' with the closest known key for each typo."""
    import difflib

    hints = []
    for key in bad:
        close = difflib.get_close_matches(str(key), [str(k) for k in known], n=1, cutoff=0.6)
        if not close:
            close = [k for k in known if str(k).lower() == str(key).lower()]
        hints.append(f"{key!r} (did you mean {close[0]!r}?)" if close else repr(key))
    return f"unknown {what}: {', '.join(hints)}; known: {', '.join(sorted(map(str, known)))}"


def _path(project, value):
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else Path(project["_dir"]) / path


def _merge(defaults, given):
    out = copy.deepcopy(defaults)
    for key, value in (given or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_config(project, registry=None, cluster=True):
    """The campaign configuration of a loaded project (absolute paths, defaults filled in)."""
    from .plugins import default_registry

    plugins = [str(_path(project, p).resolve()) for p in project.get("plugins") or []]
    registry = registry or default_registry(plugins)
    reaction = registry.reaction(project["reaction"])
    slab = project["slab"]
    files = {"seed": _path(project, slab["file"])}
    if slab.get("frequency_mask"):
        files["frequency_mask"] = _path(project, slab["frequency_mask"])
    files.update({k: _path(project, project["inputs"][k]) for k in INPUT_KEYS})
    missing = [f"{k}: {p}" for k, p in files.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Project inputs not found: " + "; ".join(missing))
    unknown = sorted(set(project["scope"]["families"]) - set(registry.families))
    if unknown:
        raise ValueError(f"Unknown mechanism families {unknown}; `mechanet plugins` lists them")
    config = {
        "schema_version": 1,
        "project": project["name"],
        "reaction": reaction.name,
        "mechanet_version": __version__,
        "project_file": project.get("_path"),
        "inputs": {k: str(p.resolve()) for k, p in files.items()},
        "slab": {k: copy.deepcopy(v) for k, v in slab.items() if k not in ("file", "frequency_mask")},
        "sites": copy.deepcopy(project["sites"]),
        "scope": copy.deepcopy(project["scope"]),
        "plugins": plugins,
    }
    for section, defaults in DEFAULT_SECTIONS.items():
        config[section] = _merge(defaults, project.get(section))
    if config["vasp"].get("command") is not None:
        from .clusters import _command_list

        config["vasp"]["command"] = _command_list(config["vasp"]["command"])  # words as text: [srun, -n, "4"]
    thermo = config["thermodynamics"]
    thermo.setdefault("potential_scale", reaction.potential_scale)
    thermo.setdefault("potentials_V", list(reaction.default_potentials))
    if thermo.get("references"):
        thermo["references"] = str(_path(project, thermo["references"]).resolve())
    if cluster and project.get("cluster"):
        from .clusters import apply_profile, load_profile

        apply_profile(config, load_profile(cluster_reference(project)), keep_command=_project_command(project))
    return config


def cluster_reference(project):
    """The project's `cluster:` value: a profile name, or a profile file relative to the project."""
    value = str(project["cluster"])
    if "/" in value or value.endswith((".yaml", ".yml")):
        return str(_path(project, value))
    return value


def _project_command(project):
    return bool((project.get("vasp") or {}).get("command"))


def generate(project_path=None, out=None, families=None):
    """Build the campaign of a project (default ``<project>/campaign``); returns the build report.

    Without ``project_path`` the project.yaml of the current directory or a parent is used.
    """
    from .builder import build
    from .plugins import default_registry

    if project_path is None:
        project_path = find_project(".")
        if project_path is None:
            raise FileNotFoundError(
                f"No {PROJECT_FILE} in {Path.cwd()} or its parents; create one with `mechanet init SLAB_FILE`"
            )
    project = load_project(project_path)
    config = resolve_config(project, cluster=bool(project.get("cluster")))
    registry = default_registry(config["plugins"])
    out = Path(out) if out else Path(project["_dir"]) / "campaign"
    report = build(config, out, base_dir=project["_dir"], families=families, registry=registry)
    shutil.copy2(project["_path"], out / "project.snapshot.yaml")
    report["warnings"] = []
    if not project.get("cluster"):
        report["warnings"].append(
            "No cluster profile in project.yaml: set `cluster:` and run "
            f"`mechanet cluster use NAME {_quote(out)}` before submitting."
        )
    elif _project_command(project):
        report["warnings"].append("vasp.command of project.yaml is used instead of the cluster profile's vasp_command.")
    stray = _stray_potentials(project, registry.reaction(project["reaction"]))
    if stray:
        report["warnings"].append(stray)
    listed = set(config["scope"].get("families") if families is None else families)
    for key, family in sorted(registry.scope_blocks().items()):
        if key in project["scope"] and family.name not in listed:
            report["warnings"].append(
                f"scope.{key} configures the {family.name} family, which this campaign does not build "
                f"(scope.families: {sorted(listed)}): the block has no effect"
            )
    report["campaign"] = str(out)
    return report


def _stray_potentials(project, reaction):
    """A warning when every analysis potential is far on the wrong side of the reaction (None otherwise).

    Potentials left over from another reaction (OER values after switching a project to HER,
    say) analyse a window where nothing of interest happens.
    """
    from .thermo import Conditions

    thermo = {**DEFAULT_SECTIONS.get("thermodynamics", {}), **(project.get("thermodynamics") or {})}
    potentials = thermo.get("potentials_V")
    defaults = [float(u) for u in reaction.default_potentials or ()]
    if not potentials or not defaults:
        return None
    try:
        values = [float(u) for u in potentials]
        scale = str(thermo.get("potential_scale") or reaction.potential_scale).upper()
        conditions = Conditions(float(thermo.get("temperature_K") or 298.15), float(thermo.get("pH") or 0.0))
        # the project's potentials on the scale of the reaction's defaults (RHE and SHE differ by 0.059 pH)
        on_reaction_scale = [conditions.from_she(conditions.to_she(u, scale), reaction.potential_scale) for u in values]
    except (TypeError, ValueError):
        return None
    # Beyond 1 V on the side where the reaction does not run: below the defaults of an anodic
    # reaction, above those of a cathodic one.
    if reaction.electrons > 0:
        stray = all(u < min(defaults) - 1.0 for u in on_reaction_scale)
    else:
        stray = all(u > max(defaults) + 1.0 for u in on_reaction_scale)
    if stray:
        rounded = [round(u, 3) for u in on_reaction_scale]
        converted = (
            ""
            if scale == reaction.potential_scale.upper()
            else f" ({rounded} V vs {reaction.potential_scale} at pH {conditions.pH:g})"
        )
        return (
            f"thermodynamics.potentials_V {values} V vs {scale}{converted} are all more than 1 V on the side where "
            f"{reaction.name} does not run (its defaults: {defaults} V vs {reaction.potential_scale}); left over "
            f"from another reaction? Set them for {reaction.name}, unless this window is deliberate."
        )
    return None


# ---------------------------------------------------------------------------
# mechanet init
# ---------------------------------------------------------------------------


class Asker:
    """Questions on a terminal; defaults when not interactive (records what was decided how)."""

    def __init__(self, interactive, input_func=input, output=print):
        self.interactive = interactive
        self.input = input_func
        self.output = output
        self.origins = {}

    def yes_no(self, key, question, default=True):
        if not self.interactive:
            self.origins[key] = "auto"
            return default
        suffix = "[Y/n]" if default else "[y/N]"
        while True:
            answer = self.input(f"  [ask] {question} {suffix} ").strip().lower()
            if not answer:
                self.origins[key] = "confirmed"
                return default
            if answer in ("y", "yes", "n", "no"):
                value = answer.startswith("y")
                self.origins[key] = "confirmed" if value == default else "overridden"
                return value
            self.output("        please answer y or n")

    def choose(self, key, question, options, default):
        """One of ``options`` (list of (value, label)); returns the value."""
        if not self.interactive:
            self.origins[key] = "auto"
            return default
        self.output(f"  [ask] {question}")
        for k, (value, label) in enumerate(options, start=1):
            mark = "*" if value == default else " "
            self.output(f"     {mark}{k}) {value}: {label}" if label else f"     {mark}{k}) {value}")
        while True:
            answer = self.input(f"        choice [{[v for v, _ in options].index(default) + 1}]: ").strip()
            if not answer:
                self.origins[key] = "confirmed"
                return default
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                value = options[int(answer) - 1][0]
                self.origins[key] = "confirmed" if value == default else "overridden"
                return value
            if answer in [v for v, _ in options]:
                self.origins[key] = "confirmed" if answer == default else "overridden"
                return answer
            self.output("        please enter a number from the list")

    def text(self, key, question, default=""):
        if not self.interactive:
            self.origins[key] = "auto"
            return default
        answer = self.input(f"  [ask] {question} [{default}] ").strip()
        self.origins[key] = "confirmed" if not answer or answer == default else "overridden"
        return answer or default


def init_project(
    slab,
    directory=None,
    *,
    name=None,
    reaction=None,
    scope=None,
    cluster=None,
    job_script=None,
    incar_loose=None,
    incar_final=None,
    kpoints=None,
    preset="pbe-d3",
    frequency_mask=None,
    plugins=None,
    interactive=None,
    force=False,
    input_func=input,
    output=print,
):
    """Inspect a slab and write a project directory; returns a summary of what was written and decided.

    ``plugins`` are local plugin files: they are loaded before the reaction and scope are
    chosen (so ``reaction`` may name a plugin reaction) and copied into ``<project>/plugins``.
    """
    from .clusters import clusters_dir, import_job_script, list_profiles, load_profile
    from .inspector import inspect_slab, summary
    from .plugins import default_registry
    from .sitemap import write_site_map
    from .structure import grouped_copy, read_structure, species_grouped, write_poscar

    slab = Path(slab)
    if not slab.is_file():
        raise FileNotFoundError(f"Slab file not found: {slab}")
    # Everything that can be checked before the questions is checked here, so that a typo
    # does not cost the whole interview (and nothing is written before it passes).
    if bool(incar_loose) != bool(incar_final):
        raise ValueError(
            "Give both --incar-loose and --incar-final (the same file twice is fine), or neither to use a preset"
        )
    for option, path in (
        ("--incar-loose", incar_loose),
        ("--incar-final", incar_final),
        ("--kpoints", kpoints),
        ("--frequency-mask", frequency_mask),
        ("--job-script", job_script),
        *(("--plugin", plugin) for plugin in plugins or ()),
    ):
        if path and not Path(path).is_file():
            raise FileNotFoundError(f"{option}: {path} not found")
    plugins = [Path(p) for p in plugins or ()]
    names = [p.name for p in plugins]
    if len({n.lower() for n in names}) != len(names):  # case-insensitive, as on macOS and Windows
        raise ValueError(f"--plugin: two plugin files share a name ({', '.join(names)}); rename one")
    if not incar_loose and preset not in available_presets():
        raise ValueError(f"Unknown DFT preset {preset!r}; available: {', '.join(available_presets())}")
    registry = default_registry([str(p.resolve()) for p in plugins]) if plugins else default_registry()
    if reaction is not None and reaction not in registry.reactions:
        hint = "" if plugins else " (a reaction from a plugin file needs --plugin FILE)"
        raise ValueError(f"Unknown reaction {reaction!r}; available: {sorted(registry.reactions)}{hint}")
    if cluster and not job_script:
        load_profile(cluster)  # an unknown or invalid profile fails now, with the known names
    name = name or slab.stem.replace(" ", "_")
    if directory:
        root = Path(directory)
    else:
        root = Path(name)
        if root.exists() and not root.is_dir():  # `mechanet init POSCAR`: the slab file has the default name
            root = Path(f"{name}_project")
    if root.exists() and not root.is_dir():
        raise FileExistsError(f"{root} is a file; give the project directory: `mechanet init {slab} DIRECTORY`")
    if (root / PROJECT_FILE).exists() and not force:
        raise FileExistsError(f"{root / PROJECT_FILE} exists; edit it, or pass --force to start over")
    if interactive is None:
        interactive = sys.stdin.isatty()
        if not interactive:
            output(
                "  [note] standard input is not a terminal, so every proposal is accepted (as with --yes); "
                "run `mechanet init` in a terminal to answer the questions"
            )
    ask = Asker(interactive, input_func, output)
    profile_name = None
    if job_script:
        from .clusters import profile_from_job_script

        profile_from_job_script(job_script)  # a script that cannot become a profile fails before the questions
        profile_name = _profile_name_for(job_script, cluster, ask, output, check_only=True)
    structure = read_structure(slab)
    # A POSCAR holds each element in one block; a file that lists an element twice (as some
    # builders write) is regrouped here, so that atom IDs, sites and every structure written
    # later follow one order: the project's copy of the slab.
    regrouped = not species_grouped(structure.symbols)
    slab_source, mask_source = slab, frequency_mask
    if regrouped:
        original_symbols = list(structure.symbols)
        structure, _ = grouped_copy(structure)
        slab_source = structure
        if frequency_mask:
            mask = read_structure(frequency_mask)
            if mask.symbols != original_symbols:
                raise ValueError("The --frequency-mask file must have the same atoms in the same order as the slab")
            mask_source = grouped_copy(mask)[0]
        output(
            f"  [note] {slab.name} lists an element in more than one block; the project's copy groups the "
            "atoms by element (a POSCAR needs one block per element), and atom IDs follow the new order"
        )
    report = inspect_slab(slab_source, mask_path=mask_source)
    if regrouped:
        report["file"] = str(slab)
    output(summary(report))
    proposal = report["proposal"]
    origins = {"slab.vacuum_axis": "auto", "slab.surface_sign": "auto"}
    if report["active_side"]["confidence"].startswith("low"):
        axis = report["vacuum"]["axis"]
        keep = ask.yes_no("slab.surface_sign", f"No constraints in the file. Is the active side +{axis}?", True)
        proposal["surface_sign"] = 1 if keep else -1
        if not keep:  # re-inspect from the other side so sites and the fixed region follow
            report = inspect_slab(slab_source, mask_path=mask_source, surface_sign=-1)
            if regrouped:
                report["file"] = str(slab)
            proposal = report["proposal"]
    if report["fixed_region"]["source"] == "proposal":
        fr = report["fixed_region"]
        if ask.yes_no(
            "slab.fixed",
            f"Fix the bottom {fr['fix_bottom_layers']} of {fr['n_layers']} layers "
            f"({len(fr['fixed_ids'])} atoms) during relaxations?",
            True,
        ):
            fixed = {"fixed_ids": fr["fixed_ids"]}
        else:
            fixed = {"fixed_ids": []}
            output("        nothing fixed; edit slab.fixed.fixed_ids in project.yaml to choose atoms")
    else:
        fixed = {"source": "file"}
        origins["slab.fixed"] = "file"
    spectators = []
    if report["unbound_molecules"]:
        molecules = report["unbound_molecules"]
        if ask.yes_no(
            "slab.spectator_molecules",
            f"Treat the {len(molecules)} unbound molecule(s) "
            f"({', '.join(m['formula'] for m in molecules)}) as spectators?",
            True,
        ):
            spectators = [m["atoms"] for m in molecules]
        else:
            output(
                "        mechanet has no family for reactive solvent molecules yet; they stay in the slab "
                "unchanged. Remove them from the slab file if they should not be there."
            )
            spectators = [m["atoms"] for m in molecules]
    sites = proposal["sites"]
    if not sites:
        output("  [warn] no reactive site was found; add sites to project.yaml by hand")
    elif not ask.yes_no("sites", f"Use the {len(sites)} proposed reactive site(s)?", True):
        known_ids = [s["id"] for s in sites]
        while True:
            keep = ask.text("sites.keep", "Site IDs to keep (comma-separated)", ",".join(known_ids))
            wanted = {x.strip() for x in keep.split(",") if x.strip()}
            unknown = sorted(wanted - set(known_ids))
            if not unknown:
                break
            output(f"        unknown site IDs {unknown}; the proposed sites are {', '.join(known_ids)}")
        sites = [s for s in sites if s["id"] in wanted]
        if not sites:
            output("  [warn] no site kept; add sites to project.yaml by hand")
    reactions = sorted(registry.reactions)
    reaction = reaction or ask.choose(
        "reaction", "Reaction", [(r, registry.reaction(r).title) for r in reactions], "oer"
    )
    presets = _scope_presets(reaction, registry)
    if scope is None:
        options = [(k, SCOPE_HELP.get(k, "") + f" [{', '.join(v)}]") for k, v in presets.items()]
        scope = ask.choose("scope.families", "Mechanism scope", options, "core")
    if scope in presets:
        families = presets[scope]
    else:
        families = [f.strip() for f in str(scope).split(",") if f.strip()]
        unknown = sorted(set(families) - set(registry.families))
        if unknown:
            raise ValueError(f"Unknown mechanism families {unknown}; `mechanet plugins` lists them")
    frequency_atoms = proposal["frequency_atoms"]
    if frequency_mask:
        origins["slab.frequency_mask"] = "given"
    elif frequency_atoms and not ask.yes_no(
        "slab.frequency_atoms",
        f"Partial-Hessian region: the {len(frequency_atoms)} reactive ligand atoms and "
        "their H (every added atom is active too)?",
        True,
    ):
        output("        edit slab.frequency_atoms in project.yaml (atom IDs a0001...)")

    # Spin, +U and PAW proposals.
    present = list(dict.fromkeys(structure.symbols))
    roles = report["roles"]
    mp = materials_project_defaults()
    anion = next((a for a in ("F", "O") if a in present), None)
    u_table = {el: mp["dft_u"][anion][el] for el in present if anion and el in mp["dft_u"][anion]}
    dft_u = None
    own_templates = bool(incar_loose and incar_final)
    if own_templates:
        # The user's protocol is kept as written: its LDAU and MAGMOM tags are used unchanged.
        u_table = {}
        origins["vasp.dft_u"] = origins["vasp.magmom_by_element"] = "INCAR templates"
    if u_table and ask.yes_no(
        "vasp.dft_u",
        "Use the Materials Project Hubbard U values for "
        + ", ".join(f"{el} (U = {v['U']} eV)" for el, v in u_table.items())
        + "?",
        True,
    ):
        dft_u = u_table
    magmom = {el: float(elements.HIGH_SPIN_MAGMOM.get(el, 0.6 if roles.get(el) == "metal" else 0.0)) for el in present}
    magmom.setdefault("O", 0.0)
    magmom.setdefault("H", 0.0)
    if own_templates:
        magmom = None
    potcar_variants = {el: mp["potcar"].get(el, el) for el in dict.fromkeys(present + ["O", "H"])}
    if reaction == "cer":
        potcar_variants.setdefault("Cl", mp["potcar"].get("Cl", "Cl"))
        if magmom is not None:
            magmom.setdefault("Cl", 0.0)

    # Cluster profile.
    cluster_name, cluster_note = None, None
    if job_script:
        cluster_name = _profile_name_for(job_script, cluster or profile_name, ask, output)
        if (clusters_dir() / f"{cluster_name}.yaml").exists():
            cluster_note = f"cluster profile {cluster_name} already matches {job_script}; using it"
            output("  " + cluster_note)
        else:
            written, _, notes = import_job_script(job_script, cluster_name)
            cluster_note = f"imported {job_script} as cluster profile {cluster_name} ({written})"
            output("  " + cluster_note + "".join(f"\n    {n}" for n in notes))
        ask.origins["cluster"] = "imported"
    elif cluster:
        cluster_name = cluster
        ask.origins["cluster"] = "given"
    else:
        valid = [p["name"] for p in list_profiles() if p.get("status") == "ok"]
        if valid and ask.interactive:
            cluster_name = ask.choose(
                "cluster",
                "Cluster profile",
                [(k, "") for k in valid] + [("none", "set later")],
                valid[0] if len(valid) == 1 else "none",
            )
            cluster_name = None if cluster_name == "none" else cluster_name
        elif len(valid) == 1:
            cluster_name = valid[0]
            ask.origins["cluster"] = "auto"
            output(f"  [note] cluster profile {cluster_name} (the only one) is used; check it in project.yaml")
        elif valid:
            output(
                f"  [note] {len(valid)} cluster profiles exist ({', '.join(valid)}); set `cluster:` in "
                "project.yaml, or pass --cluster NAME"
            )
        else:
            output(
                "  [note] no cluster profile yet: run `mechanet cluster import JOB_SCRIPT --name NAME` with a "
                "job script that runs VASP on your cluster, then set `cluster:` in project.yaml"
            )

    # Write the project directory.
    (root / "slab").mkdir(parents=True, exist_ok=True)
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    slab_copy = root / "slab" / slab.name
    mask_copy = None
    if regrouped:
        poscar_like = slab.name.upper().startswith(("POSCAR", "CONTCAR")) or slab.suffix.lower() in (
            ".vasp",
            ".poscar",
        )
        slab_copy = root / "slab" / (slab.name if poscar_like else f"{slab.stem}.vasp")
        if slab_copy.resolve() == slab.resolve():  # never overwrite the file given on the command line
            slab_copy = root / "slab" / f"{slab.stem}_grouped.vasp"
        write_poscar(slab_copy, structure, comment=f"{slab.name}, atoms grouped by element (mechanet init)")
        origins["slab.file"] = f"{slab.name}, atoms grouped by element"
        if frequency_mask:
            mask_copy = root / "slab" / f"{Path(frequency_mask).stem}_grouped.vasp"
            write_poscar(mask_copy, mask_source, comment=f"{Path(frequency_mask).name}, atoms grouped by element")
    else:
        if slab.resolve() != slab_copy.resolve():
            shutil.copy2(slab, slab_copy)
        if frequency_mask:
            mask_copy = root / "slab" / Path(frequency_mask).name
            if Path(frequency_mask).resolve() != mask_copy.resolve():
                shutil.copy2(frequency_mask, mask_copy)
    report["file"] = f"slab/{slab_copy.name}"
    write_json(root / "slab" / "inspection.json", report)
    write_site_map(structure, report, root / "slab" / "site_map.html", sites=sites, fixed=fixed)
    axis = int(proposal["vacuum_axis"])
    # Files given on the command line are copied; generated ones never replace an existing
    # file (it may hold your edits or computed reference energies; delete it to regenerate).
    inputs, kept = root / "inputs", []
    if incar_loose and incar_final:
        shutil.copy2(incar_loose, inputs / "INCAR_loose")
        shutil.copy2(incar_final, inputs / "INCAR_final")
        protocol = "your INCAR templates"
        origins["inputs"] = "given"
    elif (inputs / "INCAR_loose").exists() and (inputs / "INCAR_final").exists():
        kept += ["INCAR_loose", "INCAR_final"]
        protocol = "the existing inputs/INCAR_* (kept)"
        origins["inputs"] = "kept from an earlier init"
    else:
        base = resources.files("mechanet").joinpath(f"data/presets/{preset}")
        for stage in ("INCAR_loose", "INCAR_final"):
            text = base.joinpath(stage).read_text().replace("{name}", name).replace("{idipol}", str(axis + 1))
            (inputs / stage).write_text(text)
        protocol = f"preset {preset} (review inputs/INCAR_*)"
        origins["inputs"] = f"preset {preset}"
    if kpoints:
        shutil.copy2(kpoints, inputs / "KPOINTS")
        origins["inputs.kpoints"] = "given"
    elif (inputs / "KPOINTS").exists():
        kept.append("KPOINTS")
        origins["inputs.kpoints"] = "kept from an earlier init"
    else:
        text, mesh = kpoints_text(structure.cell, axis)
        (inputs / "KPOINTS").write_text(text)
        origins["inputs.kpoints"] = f"auto {mesh[0]}x{mesh[1]}x{mesh[2]}"
    references = inputs / "references.yaml"
    if references.exists():
        kept.append("references.yaml")
    else:
        references.write_text(references_template(reaction))
    if kept:
        output(f"  [note] kept the existing inputs/{', inputs/'.join(kept)} (delete a file to regenerate it)")
    plugin_copies = []
    if plugins:
        (root / "plugins").mkdir(exist_ok=True)
        for plugin in plugins:
            dest = root / "plugins" / plugin.name
            if plugin.resolve() != dest.resolve():
                shutil.copy2(plugin, dest)
            plugin_copies.append(f"plugins/{plugin.name}")
        origins["plugins"] = "given (copied into plugins/)"
        output(f"  [note] plugin file(s) copied into {root / 'plugins'}; the project loads these copies")
    origins.update(ask.origins)
    decisions = {
        "name": name,
        "reaction": reaction,
        "slab_file": f"slab/{slab_copy.name}",
        "frequency_mask": f"slab/{mask_copy.name}" if mask_copy else None,
        "vacuum_axis": axis,
        "surface_sign": int(proposal["surface_sign"]),
        "fixed": fixed,
        "frequency_atoms": None if mask_copy else frequency_atoms,
        "roles": {},
        "spectators": spectators,
        "sites": sites,
        "families": families,
        "pair_max_distance_A": report["thresholds"]["pair_max_A"],
        "dft_u": dft_u,
        "magmom": magmom,
        "potcar_variants": potcar_variants,
        "potentials": list(registry.reaction(reaction).default_potentials),
        "scale": registry.reaction(reaction).potential_scale,
        "cluster": cluster_name,
        "plugins": plugin_copies,
        "origins": origins,
    }
    (root / PROJECT_FILE).write_text(render_project(decisions))
    load_project(root / PROJECT_FILE)  # what we wrote must load
    return {
        "project": str(root / PROJECT_FILE),
        "site_map": str(root / "slab" / "site_map.html"),
        "sites": len(sites),
        "families": families,
        "protocol": protocol,
        "cluster": cluster_name,
        "cluster_note": cluster_note,
        "origins": origins,
        "next": [
            f"review {root / PROJECT_FILE} and inputs/INCAR_* (open slab/site_map.html to check the sites)",
            f"mechanet generate {_quote(root)}",
            f"mechanet plan {_quote(root / 'campaign')}",
            f"mechanet submit {_quote(root / 'campaign')} --pilot            # preview; add --execute to submit",
        ],
    }


def _quote(path):
    import shlex

    return shlex.quote(str(path))


def _profile_name_for(job_script, name, ask, output, check_only=False):
    """The profile name for an imported job script: free, or already holding the same script.

    ``check_only`` (before the interview) only rejects a name given on the command line, or the
    default name when nobody can be asked; the question itself comes at the cluster step.
    """
    from .clusters import load_profile, parse_job_script, profile_path

    default = name or Path(job_script).stem.replace(" ", "_")
    if check_only and ask.interactive and not name:
        return None
    while True:
        chosen = default if (check_only or name) else ask.text("cluster", "Name for the cluster profile", default)
        try:
            path = profile_path(chosen)
            problem = None
            if path.exists():
                existing = load_profile(path)
                parsed = parse_job_script(Path(job_script).read_text())[0]
                same = all(
                    (existing.get(key) or None) == (parsed.get(key) or None)
                    for key in ("sbatch", "shell", "setup", "vasp_command")
                )
                if not same:
                    problem = (
                        f"cluster profile {chosen!r} exists ({path}) and differs from {job_script}; choose another "
                        "name, use it with --cluster NAME, or replace it with `mechanet cluster import "
                        f"{job_script} --name {chosen} --force`"
                    )
        except ValueError as exc:
            problem = str(exc)
        if problem is None:
            return chosen
        if check_only or name or not ask.interactive:
            raise ValueError(problem)
        output(f"        {problem}")
        default = f"{default}_2"


def _flow(value):
    """One-line YAML for short lists and mappings of scalars."""
    import yaml

    text = yaml.safe_dump(value, default_flow_style=True, width=10**6, sort_keys=False).strip()
    if text.endswith("\n..."):  # end-of-document marker after a plain scalar
        text = text[:-4].strip()
    return text


def _wrap_flow(line, width=100):
    """Wrap a long `key: [a, b, ...]` flow sequence over several lines (valid YAML)."""
    key, sep, value = line.partition(": ")
    if len(line) <= width or not sep or not value.startswith("[") or not value.endswith("]"):
        return line
    items, depth, current = [], 0, ""
    for char in value[1:-1]:
        if char == "," and depth == 0:
            items.append(current.strip())
            current = ""
            continue
        depth += char in "[{"
        depth -= char in "]}"
        current += char
    items.append(current.strip())
    pad = " " * (len(key) + 3)
    out, row = [], f"{key}: ["
    for k, item in enumerate(items):
        piece = item + ("]" if k == len(items) - 1 else ",")
        if len(row) + len(piece) + 1 > width and row.strip() not in (f"{key.strip()}: [",):
            out.append(row.rstrip())
            row = pad
        row += piece + " "
    out.append(row.rstrip())
    return "\n".join(out)


def render_project(d):
    """project.yaml text with comments (the file a person reads and edits)."""
    o = d["origins"]

    def tag(key, default="auto"):
        return f"[{o.get(key, default)}]"

    def c(content, comment=""):
        """A line with its comment aligned at column 34; a long value gets its comment on the line above."""
        indent = " " * (len(content) - len(content.lstrip()))
        content = _wrap_flow(content)
        if not comment:
            return content
        if "\n" in content or len(content) + len(comment) > 110:
            return f"{indent}# {comment}\n{content}"
        return f"{content.ljust(32)}  # {comment}" if len(content) < 32 else f"{content}  # {comment}"

    fixed = d["fixed"]
    lines = [
        "# mechanet project. Every value is used as written; [auto] marks values detected by",
        "# `mechanet init` that should be checked once ([confirmed]/[overridden]: answered at init).",
        f"mechanet: {SCHEMA}",
        f"name: {_flow(d['name'])}",
        c(f"reaction: {d['reaction']}", "`mechanet plugins` lists the reactions"),
        "",
        "slab:",
        f"  file: {_flow(d['slab_file'])}",
        c(f"  vacuum_axis: {d['vacuum_axis']}", f"{tag('slab.vacuum_axis')} lattice vector across the vacuum"),
        c(f"  surface_sign: {d['surface_sign']}", f"{tag('slab.surface_sign')} +1: the active side faces +normal"),
    ]
    if fixed.get("source") == "file":
        lines.append(c("  fixed: {source: file}", "[file] selective dynamics of the slab file"))
    else:
        lines.append(c("  fixed:", f"{tag('slab.fixed')} atoms frozen in every relaxation"))
        lines.append(f"    fixed_ids: {_flow(fixed['fixed_ids'])}")
    if d["frequency_mask"]:
        lines.append(c(f"  frequency_mask: {_flow(d['frequency_mask'])}", "partial-Hessian region from a mask file"))
    else:
        lines.append(
            c(
                f"  frequency_atoms: {_flow(d['frequency_atoms'])}",
                f"{tag('slab.frequency_atoms')} partial-Hessian region (added atoms join it)",
            )
        )
    lines.append(c("  roles: {}", "element role overrides, e.g. {Mg: spectator, La: metal}"))
    if d["spectators"]:
        lines.append(
            c(
                f"  spectator_molecules: {_flow(d['spectators'])}",
                f"{tag('slab.spectator_molecules')} unbound molecules, kept unchanged",
            )
        )
    lines += ["", c("sites:", f"{tag('sites')} reactive sites; delete an entry to exclude it")]
    lines += [f"  - {_flow(s)}" for s in d["sites"]] or ["  []"]
    dft_u = (
        c(f"  dft_u: {_flow(d['dft_u'])}", f"{tag('vasp.dft_u')} Materials Project U (LDAUTYPE 2)")
        if d["dft_u"]
        else c("  dft_u: null", "LDAU tags of the INCAR templates, or e.g. {Ni: {L: 2, U: 6.2, J: 0}}")
    )
    cluster = (
        c(f"cluster: {d['cluster']}", f"{tag('cluster', 'given')} profile name (`mechanet cluster list`)")
        if d["cluster"]
        else c("cluster: null", "set to a profile from `mechanet cluster list`")
    )
    lines += [
        "",
        "scope:",
        c(f"  families: {_flow(d['families'])}", tag("scope.families")),
        c(f"  pair_max_distance_A: {d['pair_max_distance_A']}", "two-site families pair sites within this distance"),
        "",
        c("inputs:", "DFT templates; INCAR tags are used as written"),
        "  incar_loose: inputs/INCAR_loose",
        "  incar_final: inputs/INCAR_final",
        c("  kpoints: inputs/KPOINTS", f"[{o.get('inputs.kpoints', 'given')}]"),
        "",
        "vasp:",
        c(f"  potcar_variants: {_flow(d['potcar_variants'])}", "[auto] Materials Project PAW choices"),
        (
            c(f"  magmom_by_element: {_flow(d['magmom'])}", "[auto] starting moments (muB)")
            if d["magmom"] is not None
            else c("  magmom_by_element: null", "MAGMOM as in your INCAR templates (or VASP's default)")
        ),
        dft_u,
        c("  common_overrides: {}", "tags for every stage, e.g. {NCORE: 16, KPAR: 2}"),
        "  max_attempts: 2",
        "  max_continuations: 8",
        "",
        c("frequency:", "partial Hessian by finite differences"),
        c("  central: final", "04_static and 05_freq at the 02_final geometry (or: refine)"),
        "  ediff: 1.0e-06",
        "  potim: 0.015",
        "  nfree: 2",
        "",
        "thermodynamics:",
        "  references: inputs/references.yaml",
        c(f"  potentials_V: {_flow(d['potentials'])}", f"vs {d['scale']}"),
        "  pH: 0",
        "  temperature_K: 298.15",
        "",
        cluster,
        c(f"plugins: {_flow(d.get('plugins') or [])}", "local plugin files, e.g. [./my_families.py]"),
        "",
        c("origins:", "how each value was decided at init"),
    ]
    lines += [f"  {key}: {_flow(value)}" for key, value in sorted(o.items())]
    return "\n".join(lines) + "\n"
