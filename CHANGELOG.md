# Changelog

## Unreleased

Chemistry beyond oxygen and chlorine without changes to the core (carbon, nitrogen, hydrogen on
metals, bonds between two sites), pre-screening with ML potentials, no repeated starts on
symmetric sites, Windows and macOS support, a smaller `analysis.py`, and three corrections from
an independent review of the thermodynamics. Everything 0.1.4 offered is still there, and a
test now checks it.

- **A coverage campaign of its own.** `mechanet generate --coverage` builds the surfaces of
  `scope.coverage` (the adsorbates you list per kind of site or per site, in every combination
  or up to `max_changed_sites` at once) and nothing else, in `<project>/coverage`, for the
  surface Pourbaix diagram that comes before the mechanism. It says how many surfaces that is
  before it builds, and how many are left once copies shifted by a lattice translation are
  merged (256 and 136 for O and OH on the eight sites of the NiFe LDH example); `--count` says
  it and builds nothing. `families.coverage.patterns` and `count` give the same to a script,
  and `notation.surface_name` names a surface by its counts (`terminal 4 OH | bridge 2 O, 2
  OH`). See `docs/surface-first.md`.
- **A slab may carry adsorbates.** A slab file with an OOH, an O2 or an OCl on an oxygen, or
  with Cl, H or another registered adsorbate on top of a metal, used to be built with that group
  as rigid scenery: no site there, no chemistry, and no word about it. `mechanet inspect` now
  names such a group, and `mechanet init` reads it as the state of a site: the atoms beyond the
  site leave the slab the campaign is referenced to (`slab/NAME_bare.vasp`), what was read is
  kept in `slab/as_given.json` (`slab.carried` in project.yaml), and `mechanet generate` builds
  the campaign on the slab as given, whose start is the file atom for atom. The adsorbate can
  then react, and the mechanism next to it starts from the given structure. A group that is no
  registered adsorbate, binds through several atoms, or sits between metals through an element
  other than oxygen (CO on a bridge, H in a hollow) stays part of the framework, and the
  inspection says so with the atoms and the reason; answering `n` at `init` keeps a recognised group there too. `init` warns when the
  reaction has no reservoir for an element the slab carries. Slabs without such groups are read
  as before. New module `carried`; `inspect_slab` takes the registry, so adsorbates of a
  plugin are recognised.
- **The mechanism on a surface you pick.** `mechanet generate --on SURFACE` builds a campaign on
  a surface: a candidate of the coverage campaign, by its ID, or a pattern written in the new
  `surfaces` block of project.yaml. Every family writes its usual intermediates, and each
  site the family does not name keeps the state it has on that surface; the families and the
  names of the cycles are untouched. The campaign goes to `<project>/on_NAME` (repeat `--on`
  for one campaign per surface) and shares slab, sites and zero of energy with the coverage
  campaign, so the calculations that campaign has finished for these intermediates are copied
  instead of repeated (the folders as they are, with `imported.json`; the coverage campaign is
  only read), and new intermediates start from the relaxed surface with only the sites that
  differ rebuilt (`seeded_from` in state.json). `--from` names another source campaign or
  `none`; a campaign built from another slab or run with other INCAR or KPOINTS files is
  refused. Only the lattice translations that map the surface onto itself merge
  intermediates. `--coverage --on` builds the coverage states around a surface (a second pass
  with more adsorbates). `mechanet pourbaix` labels the intermediates of such a campaign by
  what differs from the surface. A campaign without `--on` is built exactly as before.
- **Candidates for the mechanism from the surface Pourbaix diagram.** `mechanet pourbaix
  --window LO:HI` lists every surface that is stable somewhere in a potential window or within
  a margin (`--margin`, 0.2 eV) of the stable one there, with its ID, its short name, the
  range over which it is stable, the lead of a stable surface over its closest competitor, and
  a note when it gives the product off downhill. The distances are exact (evaluated where
  the stable surface changes, not sampled). `candidates.csv` holds the table and
  `compositions.csv` the surfaces grouped by composition. `--figures` draws five views: the
  lines with the stable surface along the potential and the distance of every surface above
  it, the lower convex hull over the electrons released, the distance per composition, the
  stable surfaces at three levels of the energies (free energy, energy with tabulated ZPE − TS,
  energy alone), and the surfaces of each composition. `--screening` is a second name of
  `--electronic-only`. `mechanet select --candidates` sends only the candidates on to the
  frequency stage (`--nodes` takes any list of intermediates), so the diagram is first read
  from relaxed energies and then redone with free energies where it matters. Without the new
  options `pourbaix` and `select` write exactly what they wrote. New modules: `surfaces`
  (the analysis, usable on plain arrays) and `surface_plots`.
- **Surface patterns with many adsorbates build in seconds.** The orientation search tried
  every order in which the sites of an intermediate can be placed: 6 orders for three sites,
  24 for four, 40,320 for eight, and a surface with OOH on four sites took three minutes
  where one with a single OOH takes one second. Up to three sites to orient, which covers
  every mechanism family and every example, nothing changes: the same orders, the same starts.
  Beyond that, two orders are tried (the sites in their order and in reverse) with two partial
  structures kept, and four to six OOH or OH now take 5 to 11 s. The build report lists the
  intermediates placed this way (`limited_orientation_search`), and
  `geometry.orientation_search.all_orders_up_to` moves the limit. Two computations the search
  repeated for every candidate (the inverse of the cell, a metal–oxygen distance) are now done
  once, with the same digits: three sites take a third less time.
- **Walden steps for the OER.** O2 can leave a site as water attaches to it, so that the bare
  site is never formed: M–OO + H2O → HO–M + O2 + H⁺ + e⁻ (Exner, *Adv. Sci.* **10**, 2305505
  (2023)), and the same exchange without charge transfer, \*OO + H2O → \*OH2 + O2 (Usama et
  al., *Nat. Commun.* **16**, 6137 (2025), after Ping 2017 and González 2021). Both are
  declared for the OER (`Reaction.walden`); `mechanet pathways` proposes them on a site that
  holds \*OO, and the new family `walden` writes the mononuclear-Walden and
  bifunctional-Walden cycles of Exner. It is a core family: `mechanet init` proposes it for new
  OER projects (RuO2(110) then has 65 intermediates, 154 steps, 106 cycles and 66 starts instead
  of 63, 146, 100 and 62), and the RuO2(110) and IrO2(110) examples list it. Projects that
  exist keep the families their project.yaml lists; the NiFe LDH example keeps the six families
  of its reference build. A campaign in which leaving
  the site bare is costly now shows the cycle that avoids it. Consequences for existing
  campaigns: `pathways` proposes more steps and cycles, and a bound O2 whose Walden release is
  downhill is treated as a bound product that leaves (resting states, Pourbaix). The families
  that were there write the same intermediates, steps and cycles as before. Applying the step
  to the \*OO of a bridging (lattice) site is an extension made here, not from the cited works.
- **An adsorbate that leaves with an atom of the surface points to a possible intermediate.**
  O₂ made of an adsorbed O and a lattice O used to be filed as "released, no bound minimum",
  like any desorption. It is the step of a lattice-oxygen route: the checks now say which
  surface atom left, read the surface left behind from the structure without the molecule, and
  name it, as an intermediate of the campaign, as a possible one the campaign does not list, or
  as outside the declared sites (with the atom to declare). `review` and `prescreen` report
  it, and `review --discover` adds a surface the campaign lacks, with a start of its own: the
  relaxed structure without the molecule, with the atom order, masks and bookkeeping the
  builder gives that intermediate, so it is prepared, run and judged like any other start
  (`--undiscover` removes it, and sets a start with calculations aside instead of deleting
  it). The structure it was found in still gives no energy, since it holds the molecule. An
  adsorbate that leaves by itself is reported as before.
- **Open design questions are marked.** `docs/pending.md` lists what was left open on purpose
  (the order of equally scored orientations on different processors, adsorbates bound through
  two sites); the code carries
  `PENDING(id)` comments where a fix would go, and a test keeps the two in step. `generate`
  warns on processors other than x86-64, where a few starts can differ.
- **No two starts of an intermediate are symmetry images of each other.** `generate` used to
  merge only lattice translations, so on a symmetric site it could write the same start twice
  (on a top site of Pt(111), \*COOH and the same \*COOH turned by 120°), and DFT computed it
  twice. The orientation search now measures how different two starts are also after every
  rotation and mirror of the slab (spglib) that keeps its normal, fixed and frequency regions and
  any explicit MAGMOM and maps the intermediate's sites onto sites in the same state; the second
  start is then a genuinely different orientation, and a start still found to be an image is set
  aside and replaced. `build_report.json` records the operations and the starts set aside
  (`start_symmetry`). Without spglib nothing changes and `generate` warns. On slabs with
  rotations or mirrors this changes which starts are written: two starts must now differ by
  `diversity_min_displacement_A` (0.75 Å) after any such operation, not only as written. The
  examples lose starts that were turned or mirrored near-copies (within 0.20–0.67 Å) of a kept
  one: RuO2(110) 69 → 62, IrO2(110) 69 → 59, Pt(111) OER 10 → 8, the SrCoO3(001) tutorial
  10 → 8; for \*COOH on Pt(111) the copy was exact, and another orientation replaces it. The
  NiFe LDH reference build is unchanged.
- **`mechanet prescreen`: ML potentials before DFT.** Every start is relaxed inside Python with
  a calculator (`mace-mp`, any `module:callable` returning an ASE calculator, or one a plugin
  adds) with the selective dynamics of its DFT relaxation, and judged by the topology check of
  `collect`. Only redundancy is removed: among the starts that kept their intermediate, those
  that relaxed into the same structure as a lower one or a symmetry image of it, those more than
  0.20 eV above the lowest, and those beyond the 3 lowest. Every start the potential changed
  (into another intermediate, without its adsorbate, into a new arrangement of the sites) or
  failed on stays in `prescreen/suggested_states.txt` (for `submit --states @FILE`), since DFT
  may confirm a change as a new intermediate (`review --discover`); an arrangement that reads as
  an intermediate the network does not list is named as a possible new intermediate. The ML
  energies only compare starts of one intermediate; they are not used for thermodynamics. The
  approach follows AdsorbML (Lan et al., npj Comput. Mater. 2023, doi:10.1038/s41524-023-01121-5),
  which discards such changed structures; a reaction network keeps them. Results are reused
  while the POSCAR, calculator and settings are unchanged. Nothing in the campaign is modified.
- **Calculator interface** (`mechanet.calculators`): `Calculator` (`describe`, `relax`), an
  ASE adapter that carries selective dynamics over exactly (FixAtoms, FixCartesian), the
  `mechanet.calculators` entry-point group and `registry.add_calculator`; `mechanet plugins`
  lists the calculators. DFT still runs through the staged VASP runner.
- `topology_check` takes a structure file as `geometry` and can ignore review.json
  (`use_review=False`); `symmetry.surface_images` gives the slab's rotations and mirrors that
  keep its normal, fixed region, frequency region and MAGMOM (spglib).
- **Adsorbates given atom by atom.** `geometry="template"` describes any small adsorbate by
  its atoms (element, the atom it is bonded to, position relative to the proximal atom), such
  as \*COOH bound through C. The builder places it with the usual scored orientation search
  (tilts, azimuths and rolls about its own axis), and relaxed structures are recognised by the
  canonical tree of the site's atoms (`Adsorbate.graph_key`, e.g. `C(O,O(H))`). The six fixed
  shapes are unchanged.
- **The checks of relaxed structures follow every element an adsorbate brings.** C, N or any
  other element of a registered adsorbate beyond O, H and Cl can hold H, and its bonds to other
  heavy atoms and to the metals are watched, with lengths from covalent radii; released
  molecules containing it are recognised, and the starting-structure bond checks and atom
  bookkeeping include it. For oxygen and chlorine chemistry every check, starting structure and
  ID is unchanged (the frozen NiFe LDH build still matches to 10⁻⁶ Å).
- **CO2 reduction to CO** (`examples/plugins/co2rr.py`): a plugin with CO2 and CO, \*COOH and
  \*CO templates and the carboxyl route, run end to end in the tests.
- **Nitrogen reduction to ammonia** (`examples/plugins/nrr.py`): N2 + 6 (H⁺ + e⁻) → 2 NH3 by
  the distal route, with seven nitrogen adsorbates (fixed shapes and templates) and a turnover
  whose marker (N2) is taken up while the product leaves twice. Run end to end in the tests,
  with no change to the core beyond naming a released NH₃.
- **Metal–H and H–H bonds are watched** when an adsorbate is bound through H (the \*H of the
  HER example). An H that left the surface, alone or as H2, is reported as released; one that
  moved to another metal waits for review; H2 lying on the surface (two H closer than 1.00 Å)
  is not taken for two \*H; and an H2 that took the H of another ligand along is not counted as
  the intermediate's own adsorbate leaving. Before, a hydrogen 6 Å above the surface still
  counted as "connectivity retained". Without such an adsorbate (OER, CER) nothing changes.
- **Bonds between the adsorbates of two sites, beyond O–O.** An adsorbate declares
  `couples=True` (and optionally `coupled_bond_A`); the builder places the pair at its bond
  length, relaxed structures are read as a coupled pair or as two adsorbates, the pathway
  search proposes "C-C coupling" (the text "O-O coupling" is unchanged), and the `enumerated`
  family couples every pair that allows it, each once. A family can still couple any two
  occupied sites, as in 0.1.4 (two OH on RuO2(110) give the starts 0.1.4 wrote); the flag
  decides what the checks, the step rules and the enumeration take up by themselves. Pairs
  other than two O now start at their own bond length instead of 1.45 Å. Tested with two \*CO
  on Pt(111) that bond through C; the CO example itself leaves the flag off, since its
  reaction has no coupling step.
- **Two-point binding is flagged, not guessed.** A relaxed adsorbate in which a second atom
  binds a metal waits for review with the new contacts named (a test pins this down); the
  end-on/side-on rule now applies to O and Cl atoms only, as it was written for.
- **`analysis.py` is split** into `topology.py` (bond kinds, classification, `topology_check`),
  `collection.py` (`collect`), `selection.py` (`select_for_thermo`), `energies.py` (node
  energies, `energy_inputs`) and the thermodynamic analysis that stays. Every name is still
  importable from `mechanet.analysis`. No science module imports the command line any more
  (`default_references` moved to `campaign.py`; `cli.default_references` still works). Code
  that replaces a function on `mechanet.analysis` at run time (a monkey-patch) no longer
  reaches the calls made inside the new modules: patch the module that defines the function.
- **Stages can be driven one at a time** from another workflow tool: `run_state(state,
  through=STAGE, execute=True)` runs what is not finished up to that stage and nothing twice
  (docs/execution.md §11; now covered by a test).
- **Nothing of 0.1.4 is lost**: every command, option and public name of 0.1.4 is still there
  (27 commands, 233 options, 1,150 names checked); the tables `analysis.RESERVOIR_MOLECULES`,
  `families.coverage.DEFAULT_STATES` and `CER_EXTRA`, replaced by declarations, are importable
  again.
- **Starts on other processors.** Orientations with exactly equal scores are told apart by
  rounding in the last digits; x86-64 machines agree with each other, Apple Silicon keeps
  another of the tied orientations for a few starts (4 of 128 in the NiFe LDH example). The
  frozen-build test is exact on x86-64 and checks states, atoms and masks elsewhere
  (docs/science.md §3).
- **Reactions declare their coverage states** (`Reaction.coverage_states`), and the reference
  molecules an analysis needs come from `Species.references`; nothing in the core names OER or
  CER for either any more.
- **Windows.** `runner.py` no longer imports `fcntl` unconditionally (the directory lock takes
  over), only the stop signals the platform has are registered, and a POTCAR that cannot be
  symlinked is copied. Checking whether a lock's owner is still running no longer uses
  `os.kill(pid, 0)`, which on Windows sends Ctrl-C to the whole console (mechanet included):
  the process table is asked instead. Tests that need POSIX signals or shell-script SLURM fakes
  skip on Windows, as do those that run job scripts when `bash` cannot run them; CI runs the
  tests on Windows and macOS as well.

Three corrections from an independent review of the thermodynamics:

- **Pourbaix planes of pH-dependent plugin species.** The pH coefficient of a formation free
  energy is now the pH slope of the species' own chemical potentials, not only of explicit H⁺.
  A termination formed in an OH⁻ basis used to be off by n kT ln10 · pH (0.83 eV per OH⁻ at
  pH 14, enough to change the stable phase); equivalent H⁺ and OH⁻ bases now give the same
  plane, and the plane reproduces the scalar free energy at every pH. A plugin rule that is not
  linear in the pH is refused by the Pourbaix construction. Built-in OER and CER results are
  unchanged.
- **The method card shows the settings the jobs run with.** Its electronic-structure block now
  describes the stage whose energies the analysis uses (04_static, or 02_final with
  `frequency.central = final`) with `vasp.common_overrides` and `stage_overrides` applied; it
  used to read `inputs/INCAR_final` alone, so an ENCUT override was reported as the template
  value and a later change of an override went undetected. Every override is recorded under
  `protocol.vasp_overrides` (card schema 2). A card frozen by an earlier version is compared
  as before, with a note that override changes cannot be detected for it.
- **Same settings across intermediates.** `thermo`, `pourbaix` and `pathways --rank` compare
  the runs whose energies they use (functional, hybrid, dispersion, cutoff, LASPH and
  solvation tags of the energy's stage; PAW dataset and DFT+U values per element) and warn,
  naming the states, when they differ (`protocol_check` in thermodynamics.json). Starting
  moments, parallelisation and convergence tags may differ.
- `vasp.energy_stage` and `vasp.configured_incar` give the energy's stage and its configured
  tags; the references check uses them too.

## 0.1.4 (2026-11-09)

What a relaxation can turn an intermediate into: another tautomer, a new intermediate, or nothing bound.

- **Shared protons are tautomers.** A proton of the seed shared between two oxygens in a short
  hydrogen bond (the inspector's `shared_protons`) that a relaxation moved to its other oxygen
  (or left between both), both oxygens being reactive sites the intermediate leaves native, no
  longer makes the structure another intermediate: it is accepted for its own
  (`same_node_other_tautomer`), and `analysis/states.json` records where each shared proton
  sits (`tautomer`, read from the H counts on the two oxygens, not from atom labels). An exact
  match with another node still comes first, the reading holds only if it leaves both oxygens
  native, and a shared proton next to a site the intermediate changes is still chemistry.
  `select` lists the intermediates computed in several tautomers (`nodes_with_several_tautomers`,
  with energies); the lowest counts. `thermo` checks every cycle (`tautomer_check` in
  cycles.csv, a `tautomers` section in thermodynamics.json) and flags those whose intermediates
  use different tautomers without each having been computed in the others'; `show-cycle` and
  the report show where each shared H sits and what was not computed. `review` lists the
  structures accepted in another tautomer.
- **Discovered intermediates**: `mechanet review --discover STATE "why"` adds the relaxed site
  occupancy of a structure that matches no intermediate to network.json as a new one (same
  definition and notation as the generated ones, with its symmetry images and where it came
  from) and collects again; the structure counts for it, `pathways` proposes the cycles
  through it, `validate` and `status` know it. `--undiscover NODE` removes it with its images
  and the adopted steps and cycles that use it (also from pathways/pathways.json). `review`
  names the new arrangement ("the sites are now *OO₆₅ · *(H₂O)₆₈, which is not an
  intermediate of this campaign"). `pathways --adopt` refuses a cycle through an intermediate
  that is no longer in network.json.
- **No bound minimum**: a molecule that left the surface during a relaxation (no O or Cl atom
  within the upper M–O (M–Cl) edge of any metal after a lost contact) is reported in chemical
  terms ("O₂ (O a0065, O d_a0065) released from Fe a0003"; `released` in the topology record).
  When it is the intermediate's own adsorbate and nothing else changed at the metals
  (`adsorbate_released`), the structure is listed apart from those that need a decision; an
  intermediate whose finished starts all end that way shows "no bound minimum found from N
  starts" instead of holding its cycles up, and `status` marks it ∅ and counts its cycles
  separately. A spectator ligand that left still needs review.
- `review` names a reassignment through a symmetry image by the image ("*O₆₇, image of *O₆₅").
- `mechanet init` writes `geometry.orientation_search` with its defaults into project.yaml, and
  the science page describes the starting-structure recipe in full (outward direction, grids,
  bond lengths, clash floors, score, beam search, the second start as a hedge).
- Installation from the repository, until mechanet is on PyPI:
  `pip install "mechanet[plots] @ git+https://github.com/dhernandezcastillo/mechanet.git@v0.1.4"`.

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
