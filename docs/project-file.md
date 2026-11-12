# project.yaml reference

A project file records every decision of a study. `mechanet init` writes it; you edit it;
`mechanet generate` turns it into a campaign. Relative paths are relative to the project
directory. Sections that configure calculations are copied into the campaign's
`config.json` as written and completed with the defaults below, so a campaign is complete
on its own. Unknown keys are refused at every level, with the closest known key as a
suggestion (`thermodynamics.ph` → did you mean `pH`?), so typos do not pass silently.

```yaml
mechanet: 1                 # schema version (required, first line)
name: RuO2_110              # free text, used in reports
reaction: oer               # oer | cer | a plugin reaction (`mechanet plugins`)
slab: {...}
sites: [...]
scope: {...}
inputs: {...}
vasp: {...}
frequency: {...}
thermodynamics: {...}
cluster: hpc                # a profile name (`mechanet cluster list`) or a path to a profile file
plugins: []                 # local plugin files
# optional: geometry, refine, thermo_selection, analysis, topology, origins, notes
```

## slab

| Key | Meaning |
|---|---|
| `file` | The slab (POSCAR/CONTCAR, or any format ASE reads). Required. |
| `vacuum_axis` | Lattice vector (0, 1, 2) across the vacuum. |
| `surface_sign` | +1 if the active side faces the +normal, −1 otherwise. |
| `fixed` | `{source: file}` keeps the file's selective dynamics; `{fixed_ids: [a0001, ...]}` fixes these atoms (and frees the rest). |
| `frequency_atoms` | Atom IDs of the reference's partial-Hessian region (default: reactive ligand atoms and their H). Added atoms always join it. May be empty (a clean metal surface). |
| `frequency_mask` | Alternative: a POSCAR whose selective dynamics define the region. |
| `roles` | Element role overrides, e.g. `{Mg: spectator, La: metal}` (roles: metal, ligand, H, spectator, nonmetal, other). |
| `species_order` | POSCAR species order of generated structures (default: the slab's order, new elements appended). |
| `spectator_molecules` | Unbound molecules kept unchanged (informational; recorded by `init`). |

Atom IDs are `a` plus the 1-based index in the slab file (`a0065`).

## sites

```yaml
sites:
  - {id: a0016, kind: bridge, native: O}                  # a ligand atom (O)
  - {id: a0065, kind: terminal, native: OH}
  - {id: m0038, kind: open_metal, native: empty, metal: a0038}
```

`kind`: `terminal` (one metal), `bridge` (two), `mu3`, `framework` (a lattice O made
reactive; it joins the frequency region of every state), or `open_metal`. `native` must
match the slab (O, OH or H2O for ligand sites; `empty` for open metal sites). Open metal
sites take an optional `direction` (unit vector) for the adsorbate; by default it is the
direction of the missing ligand.

## scope

| Key | Meaning |
|---|---|
| `families` | Mechanism families to build (`mechanet plugins` lists them). |
| `pair_max_distance_A` | Two-site families pair sites within this distance (3.3 Å). |
| `coverage` | Settings of the `coverage` family: `sites`, `max_changed_sites` (2; null for all combinations), `states` per site kind or per site ID. `mechanet generate --coverage` builds these surfaces as a campaign of their own ([surface first](surface-first.md)). |
| `enumerate` | Settings of the `enumerated` family: `sites` (one to three), `states` per site or kind, `max_length` (8), `max_cycles`. |

## surfaces

Named surfaces a campaign can be built on with `mechanet generate --on NAME`
([surface first](surface-first.md#3-pick-the-surfaces-and-build-the-mechanism-on-each)):

```yaml
surfaces:
  oxidised: {a0061: O, a0063: O}                     # the sites that differ from the slab, and their state
  peroxo: {a0061: O, a0062: O, coupling: [a0061, a0062]}   # two sites bonded to each other
```

A name may hold letters, digits, `.`, `_` and `-`: it names the campaign folder
(`<project>/on_NAME`). A surface picked from the coverage campaign by its ID needs no entry.

## geometry

| Key | Default | Meaning |
|---|---|---|
| `symmetry_reduction` | true | Compute one representative of lattice-translation images, and never keep two starts of an intermediate that a rotation or mirror of the slab maps onto each other (needs spglib; [science §3](science.md#3-starting-structures)). |
| `symmetry_tolerance_A` | 0.005 | Maximum atom displacement for a translation, rotation or mirror to count as a symmetry. |
| `anchor_min_M_O_X_angle_deg` | 100 | Minimum M–O–X angle for atoms bonded to a metal-coordinated O (lowered to the best angle less 5° at an O bridging two metals in their plane). |
| `clash_floors_A` | H-H 1.5, O-H 1.5, O-O 2.2, M-H 1.8, M-O 1.6, S-any 2.0, X-any 2.0, default 1.35 | Non-bonded floors for new or moved atoms (S: spectator cations, X: halogens). |
| `orientation_search` | azimuth_step_deg 15; OH_tilts_deg [20, 35, 50]; OO_tilts_deg [35, 55, 70]; OOH_H_torsion_step_deg 30; H2O_tilts_deg [0, 25, 50]; H2O_roll_step_deg 15; variants_per_site 2; hbond_bonus 0.15; clearance_cap_A 0.6; beam_width 6; diversity_min_displacement_A 0.75 | The search that places new atoms ([science §3](science.md#3-starting-structures)); `mechanet init` writes these values into project.yaml. Unknown keys are reported and ignored. One more key is read when given: `all_orders_up_to` (3), the number of sites to orient up to which every placement order is tried. |

## inputs

`incar_loose`, `incar_final`, `kpoints`: the DFT templates. INCAR tags are used as written,
except those mechanet sets per stage (docs/execution.md §1), MAGMOM and the LDAU arrays.

## vasp

| Key | Default | Meaning |
|---|---|---|
| `potcar_variants` | `init` proposes the Materials Project choices | PAW dataset per element (`{Ru: Ru_pv, O: O}`): `<potcar_dir>/<dataset>/POTCAR`. `mechanet init --potcar-variants Fe=Fe_pv,...` sets some and keeps the proposal for the rest. |
| `potcars` | | Alternative: explicit POTCAR path per element. |
| `potcar_dir` | the profile's `potcar_dir` | The licensed PAW library (`mechanet init --potcar-dir DIR`); a relative path is relative to project.yaml, `~` is expanded where the job runs. |
| `magmom_by_element` | | Starting moments per element (μB); null leaves MAGMOM to the templates. |
| `magmom_by_id` | | Starting moments per atom ID (overrides the element value; breaking a translation disables that symmetry reduction). |
| `magmom_default` | | Moment for elements missing from `magmom_by_element`. |
| `dft_u` | `init` proposes the Materials Project U for oxides and fluorides | `{Ni: {L: 2, U: 6.2, J: 0}}`: LDAU arrays for the species present (LDAUTYPE 2); species not listed get no U. `mechanet init --dft-u Fe=5.3,Ni=6.2` (U on d, f for lanthanides and actinides; `Fe=5.3:2:0.5` sets L and J) or `--dft-u none`. |
| `template_species_order` | slab order | Species order of LDAU arrays already present in the templates. |
| `common_overrides` | {} | Tags for every stage (NCORE, KPAR, ...); recorded in the method card. |
| `stage_overrides` | {} | Tags for one stage, e.g. `{01_loose: {NSW: 80}}`; changes that would break the protocol are refused. Recorded in the method card. |
| `max_attempts` | 2 | Attempts per stage (a failed attempt needs `--retry`). |
| `max_continuations` | 8 | Continuations of interrupted attempts per stage or unit. |
| `command` | from the cluster | The VASP launch command. Normally the profile's `vasp_command`; set here, it wins over the profile (for example `[srun, vasp_gam]` for one project), and `generate` says so. |

## frequency

| Key | Default | Meaning |
|---|---|---|
| `central` | final | `final`: 04_static and 05_freq at the 02_final geometry; `refine`: an extra 03_refine re-relaxation first. |
| `method` | displacements | Restartable finite differences, or `vasp_ibrion5`. |
| `ediff` | 1e-6 | EDIFF of 04_static and the displacements. |
| `potim` | 0.015 | Displacement (Å). |
| `nfree` | 2 | Central differences. |
| `nelm` | 120 | NELM of 04_static and the displacements. |
| `max_static_e0_shift_eV` | 0.01 | Final protocol: 04_static must reproduce the 02_final E0. |
| `max_moment_change_muB` | 0.5 | Final protocol: and its local moments. |
| `max_force_eVA`, `warn_force_eVA` | 0.035, 0.03 (final); 0.03, 0.02 (refine) | Force gate on frequency-active atoms before 05_freq. |
| `delete_wavecar_after_use` | true | Delete each unit's WAVECAR copy once accepted. |
| `max_unit_failures_per_job` | 3 | New displacement failures tolerated per job. |
| `prune_unit_outputs`, `unit_prune_files` | true, PROCAR … WAVECAR | Remove bulky outputs of accepted units. |

## refine (refine protocol only)

`ediff` (1e-6), `ediffg` (−0.015), `ibrion` (2), `nsw` (60) of 03_refine.

## thermo_selection

`energy_window_eV` (0.10) and `max_variants_per_node` (2): which 02_final variants of each
intermediate continue to 04_static and 05_freq (`mechanet select`). Before that, variants that
relaxed into the same minimum as a lower one are merged: energies within
`duplicate_energy_eV` (0.005), every atom within `duplicate_max_shift_A` (0.10 Å) of a partner
of its element (labels ignored, minimum image) and local moments within
`analysis.moment_change_muB` (`mechanet select --keep-duplicates` keeps them).

## thermodynamics

| Key | Default | Meaning |
|---|---|---|
| `references` | inputs/references.yaml | Molecular references (`mechanet references`). |
| `potentials_V` | the reaction's | Analysis potentials. |
| `potential_scale` | the reaction's (RHE for OER, SHE for CER) | RHE or SHE. |
| `pH` | 0 | |
| `temperature_K` | 298.15 | Must equal the references' temperature. |
| `activities` | {Cl-: 1, Cl2: 1} | Activities of solution species. |
| `low_frequency_cutoff_cm1` | 100 | Floor for soft positive modes. |
| `imaginary_tolerance_cm1` | 80 | Small imaginary modes floored (with the eigenvector rules of docs/science.md §8). |
| `water_equilibrium_pressure_bar` | 0.035 | Liquid-water standard state of the CHE. |
| `electronic_mode_corrections` | per_H_eV 0.30, per_O_eV 0.05 | Tabulated ZPE − TS per added atom for `--electronic-only` screening. |

## analysis and topology

`analysis.require_input_hashes` (true): refuse results whose inputs changed after the run.
`analysis.max_eentro_eV` (0.02): flag stages with a large electronic entropy.
`analysis.moment_change_muB` (0.5): spin check of a cycle and comparison of variants: a local
moment of at least this size that changes sign, or a change larger than this (away from the
active site, or between variants), is flagged.
`analysis.moment_site_radius_A` (3.6): atoms within this distance of a site metal belong to
the active site, where moment changes are the chemistry and are not flagged.
`topology.shared_proton_max_OH_A` (1.6): O···H distance of a proton shared between two O.
`topology.hysteresis_A`: bond-formation/breaking thresholds of the relaxed-structure check
(`O-H [1.20, 1.35]`, `O-O [1.75, 1.95]`, `M-O [2.20, 2.50]`, `O-Cl`, `M-Cl`).

## cluster and plugins

`cluster` names a profile in `~/.config/mechanet/clusters/` (or gives a path to one, relative
to the project folder); it is resolved into the campaign's `scheduler` section and
`vasp.command` at `generate`, and can be changed later with `mechanet cluster use NAME`.
`plugins` lists local plugin files, relative to the project folder (`mechanet init --plugin
FILE` copies a file into `plugins/` and lists it); their reactions, families and scope blocks
are known as soon as the project loads, and `generate` copies them into the campaign
(`inputs/plugins/`), so a campaign keeps them wherever it is copied (docs/plugins.md).

## origins

Written by `init`: how each value was decided (`auto`, `confirmed`, `overridden`, `given`,
`file`, a preset). Informational only.
