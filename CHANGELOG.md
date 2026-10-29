# Changelog

## 0.1.1 (2026-10-29)

Local magnetic moments become part of the analysis.

- **Moments along cycles**: `thermo` follows the projected local moments (LORBIT) of the
  structures the energies come from, atom by atom through every cycle, symmetry images
  relabelled through their lattice translation: the site metals, the bound and outer atoms
  of every site the cycle changes, and the atoms whose moments change most. They are written
  to `analysis/cycle_moments.csv` and `thermodynamics.json` (`magnetism`), printed by
  `show-cycle` and the report, and drawn by `plot-profile` in a panel under the free-energy
  profile (`--no-moments` leaves it out).
- **Spin check of cycles**: a step is flagged when a moment of at least
  `analysis.moment_change_muB` (0.5 μB) changes sign, or when a moment farther than
  `analysis.moment_site_radius_A` (3.6 Å) from the site metals changes by more than that.
  `cycles.csv` (`spin_check`, `spin_flags`), the pathway ranking and the report carry it.
- **Magnetic solutions of variants**: `select` and `thermo` report the variants of an
  intermediate whose 02_final moments differ in the same way, with their energy gap
  (`magnetic_solutions` in `thermo_selection.json` and `nodes.csv`).
- `plot-profile` labels the last point of a cycle with its first intermediate plus the
  released product; it read "native + O2" also for cycles that start elsewhere.
- Documentation: magnetic states are not sampled (the README said they were); every
  structure starts from the same MAGMOM.

## 0.1.0 (2026-10-29)

First release.

- **Slab inspection** (`mechanet inspect`): vacuum axis and surface normal (skewed cells
  included), active side, fixed region, element roles, ligand sites (terminal and bridging
  O, OH, H2O) and open metal sites (dopants judged against the host lattice, single-atom
  catalysts in 2D hosts), unbound molecules, shared protons, lattice-translation symmetry
  and site pairs.
- **Projects** (`mechanet init`): `project.yaml` with every detected value marked, INCAR
  presets, a k-point mesh, Materials Project PAW and U proposals, a references template and
  a static HTML site map; slab files that list an element in more than one block are
  regrouped into a copy. Project files are validated, with suggestions for typos.
- **Reaction networks** (`mechanet generate`): reactions as data with reservoir vectors, OER
  (RHE) and CER (SHE) built in. OER families: adsorbate evolution, bridge-O oxidation, O–O
  coupling between neighbouring sites, bifunctional BF-I and BF-II, hydrated routes and O4
  recombination; CER families: Volmer–Heyrovsky, Volmer–Tafel and the OCl route; a coverage
  family for Pourbaix diagrams and an enumerated family (every occupancy of chosen sites).
  Starting structures come from a scored, clash-free placement search, with label-free
  deduplication and lattice-translation symmetry reduction; every cycle is checked to sum to
  one turnover.
- **Plugins**: reactions, reservoir species, adsorbates, mechanism families (with their own
  `scope` blocks in project.yaml) and CLI commands, through entry points or local files
  (`mechanet init --plugin FILE`); campaigns carry copies of their plugin files. Example:
  the hydrogen evolution reaction on Pt(111).
- **Any SLURM cluster**: cluster profiles imported from a working job script (environment
  lines kept in order, the VASP launch line found and cleaned, personal options dropped);
  a staged, restartable VASP protocol (loose and final relaxations, an optional refinement,
  a single point and a partial Hessian by restartable finite displacements); continuation
  versus retry, clean stops before the time limit, walltime checks, a submission ledger, a
  job cap and refill from compute or login nodes; `plan` with node-hour estimates from pilot
  timings.
- **Results**: `collect` and `select` check convergence, forces, magnetic moments and
  topology, and assign relaxed structures to the intermediates they became.
- **Thermodynamics** (`mechanet thermo`): CHE free energies with partial-Hessian vibrations
  or a screening estimate; limiting potential, thermodynamic overpotential, Gmax(U), cyclic
  span and resting states, for any reaction and potential scale.
- **Pathway discovery** (`mechanet pathways`, networkx): elementary-step proposal, bounded
  cycle enumeration, ranking per reaction, and the exact lowest limiting potential over all
  cycles whose steps have energies.
- **Surface Pourbaix diagrams** (`mechanet pourbaix`) with exact phase polygons and
  transitions; terminations that give the product off downhill are flagged or left out.
- **Molecular references**: `references --molecules` writes H2, H2O and Cl2 inputs with the
  slab's protocol; `references --collect` reads E0 and the ideal-gas correction and records
  which checks it could make (temperature, water standard state, DFT protocol and PAW
  datasets).
- **Reports and output**: `mechanet report` (report.md), `cycles`, `show-cycle`,
  `plot-profile`, readable summaries and `--json` for scripts; `mechanet examples` writes
  the packaged example slabs (RuO2(110), IrO2(110), Pt(111)).
