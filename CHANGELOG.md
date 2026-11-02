# Changelog

## 0.1.3 (2026-11-02)

Seeing why a finished calculation gives no energy, and documentation with pictures.

- **`mechanet review`**: collects the campaign, then lists every finished structure that gives
  no energy to its own intermediate, says in plain words what changed during the relaxation
  ("H p_a0065 moved from O d_a0065 to O a0066", "O a0065 is no longer bound to Fe a0003", a
  proton relay through a water molecule in order), names the complete cycles each one holds up
  and the other starts of the intermediate, and says what to do. `--accept STATE "why"` accepts
  a relaxed structure as its own intermediate, `--assign STATE NODE "why"` counts it for
  another intermediate of the same composition (checked), `--clear STATE` removes the
  decision; each writes `states/STATE/review.json` with your reason and reports the new
  verdict at once. Decisions take effect at the next collect.
- **`status` shows what the last collect did not accept**: such finished structures are listed
  and marked `?` (on the page: a question mark with a red ring), and a complete cycle with an
  intermediate that has no accepted structure counts as held up (its fraction is no longer in
  bold). Structures that ran or were decided after that collect are not judged until the next.
- `collect` and `thermo` say how many finished structures gave no energy to their own
  intermediate; `show-cycle` points to `review` when an intermediate has no accepted structure,
  and notes a review.json written after the last `thermo`.
- **New documentation site**: a landing page with pictures of what mechanet produces (the
  site map, the status grid, free-energy profiles, Pourbaix diagrams) and a diagram of the
  workflow; Getting started; five tutorials (a first network on a laptop, your own slab,
  running on a cluster, from energies to answers, other reactions and your own mechanisms);
  Following a campaign, with screenshots of the status page and an interactive example page;
  Checking results (`mechanet review`); and Troubleshooting. The README shows the workflow and
  the status grid and links to the pages of the site. `docs/requirements.txt` and the `docs`
  extra add sphinx-design.
- Installation from the repository, until mechanet is on PyPI:
  `pip install "mechanet[plots] @ git+https://github.com/dhernandezcastillo/mechanet.git@v0.1.3"`.

## 0.1.2 (2026-11-02)

Seeing what a campaign computes, and fewer manual steps between runs.

- **`mechanet status`**: every state in one group (finished, running, queued, to continue,
  needs you, not started), measured against the stage its last submission or the refill asked
  for (`--through` sets one). It reads the run records, the ledger and one `squeue` listing
  (`--no-slurm` skips it) and changes nothing. The summary counts the groups, the
  intermediates and cycles that are complete per family, what needs you and the running jobs;
  `--cycles` shows every cycle with one mark per intermediate; `--list GROUP` prints state IDs
  for `--states @FILE`. It writes `status.html`, a self-contained page with a grid of sites ×
  mechanisms (one square per structure) and every cycle's steps and structures.
- **Readable notation** for intermediates, steps and cycles (`mechanet.notation`):
  `*OOH₆₅ · μ-O₆₁` (terminal and bridging sites, site numbers as subscripts, `μ-□` for a
  bridge vacancy), steps as equations (`*O₆₅ + H₂O → *OOH₆₅ + H⁺ + e⁻`) marked electrochemical
  or chemical, cycle titles (Adsorbate evolution (AEM), Bifunctional mechanism II (BF-II), ...)
  with their evidence and literature. `show-cycle` prints the steps as equations in the
  turnover order (the step that releases the product last); `report` has a table of the
  cycles with their intermediates and names the best cycles and resting states; the x labels
  and title of `plot-profile` use it. Terminals that cannot show it get an ASCII form.
- **`select` merges duplicate variants**: a variant that relaxed into the same minimum as a
  lower one (energies within `thermo_selection.duplicate_energy_eV`, 5 meV; every atom within
  `thermo_selection.duplicate_max_shift_A`, 0.1 Å, of a partner of its element, labels
  ignored; local moments within `analysis.moment_change_muB`) is not selected for
  frequencies; `thermo_selection.json` lists them (`duplicates_merged`, `nodes[].duplicates`).
  `--keep-duplicates` keeps them.
- **`select --families aem,...`** restricts the selection to the nodes of those families'
  cycles (as `--cycles` does for listed cycles).
- **`init --dft-u`, `--potcar-dir`, `--potcar-variants`**: your U values
  (`Fe=5.3,Ni=6.2`, `U:L:J`, or `none`), PAW library and PAW datasets go into project.yaml at
  `init`, marked as given; the Materials Project proposals fill the rest. A relative
  `vasp.potcar_dir` is read relative to project.yaml.
- **Refill after the hard time limit**: a job that SLURM kills at its time limit (a stage that
  overran the stop signal) is continued by the refill like one that stopped cleanly. The
  runner records the seconds left in the allocation when SIGTERM came; at most 120 s (or past
  the end), with the job not requeued, is the time limit, otherwise SLURM's `TIMEOUT` state or
  `TimeLimit` reason decides. Cancelled and preempted jobs still get no replacement.
- **Documentation site** for Read the Docs (`.readthedocs.yaml`, Sphinx with MyST in
  `docs/`): the README as the landing page, the guides, a reference of every command
  generated from the CLI at each build, and the changelog. CI builds it with warnings as
  errors; `pip install "mechanet[docs]"` installs the tools.
- `report.md` is written as UTF-8 on every platform.
- Documentation: on metal surfaces adsorbates are placed on top sites only; bridge and hollow
  sites are not generated yet.

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
