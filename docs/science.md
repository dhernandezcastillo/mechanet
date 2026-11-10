# What mechanet computes, and how to read it

mechanet builds a finite, literature-guided set of hypotheses on one surface, computes the
intermediates with DFT, and evaluates thermodynamic descriptors under the computational
hydrogen electrode (CHE). It does not compute barriers or rates. This page states the
conventions behind every number.

## 1. The reference surface and its sites

The slab you supply is the common reference ("native" surface) of every intermediate.
`mechanet inspect` / `init` analyse it:

* **Vacuum axis and normal.** The largest periodic gap along each lattice vector, after
  small unbound molecules (waters, intercalants) are set aside; the normal is the cross
  product of the two in-plane vectors, so skewed cells are handled exactly (the NiFe LDH
  example has its vacuum along *a* and a normal 7° off the Cartesian axis).
* **Active side.** Opposite the fixed atoms; without constraints the +axis side is assumed
  and `init` asks.
* **Element roles.** Metals carry ligands and form sites; O, N, S and halides are ligands;
  alkali and alkaline-earth cations are spectators. Roles can be overridden per element
  (`slab.roles`), for example framework Mg in an MgAl LDH.
* **Ligand sites.** Exposed O groups bound to one metal (*terminal*) or two (*bridge*),
  with their protonation (O, OH, H2O). μ3 and lattice O are reported but not proposed;
  add them explicitly to make them reactive.
* **Open metal sites.** Exposed metals coordinated below the slab interior (the bulk-like
  coordination of each element is taken from interior atoms, or from the host lattice for a
  dopant). The site sits along the direction of the missing ligand. On metal surfaces every
  exposed atom below bulk coordination is a top site. A thin slab (fewer than three layers
  and at most `interior_min_thickness_A`, 2.5 Å, thick) has no interior; stepped surfaces,
  whose atomic planes merge into one layer, keep theirs. When no interior atom of an element
  can serve as the reference (a 2D sheet, or a single-atom catalyst such as Fe–N4 in
  graphene), an exposed metal whose axial position is free and on the surface (no atom
  within `open_site_clearance_A`, 1.2 Å, of the probe position, and no heavy atom within
  `open_site_probe_radius_A`, 2.2 Å, laterally that is higher than the probe less 0.2 Å) is
  an open metal site; a buried dopant is not. Where the ligands nearly cancel (four in a
  plane), the site points along the surface normal.
* **Symmetry.** Lattice translations that map the slab, its constraints, the
  partial-Hessian mask and any explicit initial moments onto themselves (tolerance
  `geometry.symmetry_tolerance_A`, 0.005 Å). A node and its translation image are the same
  calculation: one representative is computed and the image inherits its energy
  (`computed_by`). This assumes both converge to the same electronic solution; an explicit
  `vasp.magmom_by_id` that breaks the translation disables the reduction. Rotations and
  mirrors of the slab are used only to avoid repeating a start of one intermediate (§3).
* **Partial-Hessian region.** The reactive ligand atoms and their H, plus every added atom,
  with the same background in every node so vibrational corrections are comparable.

Site IDs are atom IDs (`a0065` is the 65th atom of the slab file) or, for open metal sites,
`m` plus the metal's number (`m0038`). They follow atoms through every generated structure
(`state.json`).

## 2. Mechanism families

Every family is a set of proposed paths. Geometry optimization establishes which starting
guesses become stationary structures; nothing here identifies the catalytic flux.

| Family | Intermediates and steps | Motivation |
|---|---|---|
| `aem` | \*OH, \*O, \*OOH, \*OO on each terminal site and open metal; conventional release and a bound-OO branch | Adsorbate evolution mechanism, the CHE standard [Rossmeisl 2005, 2007; Man 2011] |
| `surface_bridge` | Oxidation of bridging OH/O, bridge-bound OOH/OO, O vacancy and water refill | Surface-bridge cycles on LDH edges [Dionigi 2020] |
| `pair_coupling` | Adjacent O/O of the same site class, coupled O2-like structure, both sites emptied, every ordering of refill and deprotonation | O–O coupling (lattice-oxygen-like) hypotheses [Hao 2021; Exner 2023] |
| `terminal_bridge_coupling` | A terminal O coupling to a neighbouring bridge O | [Hao 2021] |
| `bifunctional` | Proton transfer from the donor's OOH or water attack to a neighbouring acceptor; Exner's BF-I and BF-II and their resolved variants | [Halck 2014; Song 2019; Bai 2021; Exner 2023] |
| `hydrated` | Bound-water refill, H2O/OH/O tautomers, local proton relays | [Shi 2025] |
| `oo_oo` | Two neighbouring OO groups and distal-O release; O4 kept as a transition-state seed | [Binninger 2022; Exner 2023] |
| `cer_volmer_heyrovsky` | \* + Cl⁻ → \*Cl + e⁻; \*Cl + Cl⁻ → \* + Cl2 + e⁻ on bare metal sites | [Hansen 2010] |
| `cer_volmer_tafel` | Two Volmer steps and 2 \*Cl → Cl2 on neighbouring sites | [Hansen 2010] |
| `cer_ocl` | \*O + Cl⁻ → \*OCl + e⁻; \*OCl + Cl⁻ → \*O + Cl2 + e⁻ on O-covered sites | [Hansen 2010] |
| `coverage` | Terminations only, for surface Pourbaix diagrams | [Hansen 2008] |
| `enumerated` | Every occupancy of chosen sites, every elementary step, every balanced cycle | [Govind Rajan 2020] |

Two-site families pair sites within `scope.pair_max_distance_A` (3.3 Å by default). This is
a declared search boundary, not proof that more distant chemistry is impossible.

## 3. Starting structures

Every intermediate is built from the reference surface, the seed. Sites the intermediate
leaves native keep their atoms where they are. A site it changes keeps its own atom (its O
stays in place, its native H are removed); only an open metal site, which has no ligand of
its own, gets a new proximal atom, at the distance in the table below along the site's
outward direction. The new atoms of an adsorbate (the H of OH or water, the outer O of OO and
OOH, the H of OOH) are then placed by a scored search.

**The outward direction** of a site is the sum of its metal→ligand unit vectors. When that
sum points more than about 75° from the surface normal (cosine below 0.25) it is blended with
the normal; when it nearly cancels (an O bridging two metals in their plane) or the site has
no metal, the normal is used. Tilts are measured from this direction and azimuths around it.

**The grid** of each adsorbate (`geometry.orientation_search`, written into project.yaml by
`mechanet init`):

| Adsorbate | Built with | Searched | Candidates |
|---|---|---|---|
| OH | O–H 0.98 Å | tilt 20, 35, 50° × azimuth in 15° steps (24) | 72 |
| H2O | O–H 0.98 Å, H–O–H 104.5° | bisector tilt 0, 25, 50° × 24 azimuths (one at 0°) × roll about the bisector in 15° steps (12) | 588 |
| OO, end-on | O–O 1.34 Å | tilt 35, 55, 70° × 24 azimuths | 72 |
| OO, side-on | O–O 1.34 Å, both M–O 1.95 Å (single-metal sites) | 12 azimuths | 12 |
| OOH | O–O 1.43, O–H 0.98 Å, O–O–H 105° | tilt 35, 55, 70° × 24 azimuths × H torsion about O–O in 30° steps (12) | 864 |
| OCl | O–Cl 1.65 Å | tilt 35, 55, 70° × 24 azimuths | 72 |

At a terminal O, where the outward direction lies along the M–O bond, the OO/OOH tilts make
M–O–O angles of 145, 125 and 110°. A new proximal atom on an open metal site starts at M–O
1.75 (O), 1.95 (OH, OOH), 2.15 (H2O), 1.90 (OO, OCl) or M–Cl 2.35 Å.

**Feasibility.** Every non-bonded contact of a new or moved atom must respect
`geometry.clash_floors_A` (H···H 1.5, O···H 1.5, O···O 2.2, M···H 1.8, M···O 1.6, a
spectator cation or a halogen with any atom 2.0, any other pair 1.35 Å), and an atom bonded
to a metal-coordinated O keeps
the M–O–X angle at or above `geometry.anchor_min_M_O_X_angle_deg` (100°). A site whose
geometry cannot reach that angle, an O bridging two metals in their plane (a CoO2-terminated
perovskite: the outward direction is the surface normal, 90° from both M–O bonds), keeps the
best angle it allows less 5°, so X stands perpendicular to the M–O–M axis.

**Score.** A feasible candidate scores min(its clearance above the floors, 0.6 Å) plus 0.15
for each H-bond one of its new H donates (H···O 1.55–2.25 Å, angle at the H at least 130°;
the O that holds the H, and for OOH the inner O, do not count). The clearance term prefers open placements
up to the cap; the H-bond term prefers an H that points at a neighbouring O.

**Selection.** The sites of an intermediate are placed one after another, keeping the
`beam_width` (6) best partial structures, each extended by up to three of the best diverse
candidates of the next site. Every placement order is tried and the results are pooled, so
an early greedy choice cannot block a later site. The `variants_per_site` (2) best
structures that differ by at least `diversity_min_displacement_A` (0.75 Å) are kept, with
atoms matched by element and labels ignored (a water rotated by 180° is the same start).
Intermediates without any feasible placement are reported in `nodes_without_viable_start`
(with reasons in `rejected_placements.json`), never started from an unphysical geometry.

**Symmetry images of a start.** The difference is also measured after every rotation and
mirror of the slab that keeps its normal, its fixed and frequency regions and any explicit
MAGMOM, and maps the intermediate's sites onto sites in the same state (found with spglib,
within `symmetry_tolerance_A`). On a top site of Pt(111), with a three-fold axis and three
mirrors, a \*COOH turned by 120° is the same start as the original and gives the same DFT
result, so it is not kept twice: the second start is the next distinct orientation. A start
still found to be an image after the search is set aside and replaced
(`build_report.json`: `start_symmetry`). Without spglib this is skipped and `generate` says
so. Lattice translations, which relate whole intermediates rather than starts, are handled
separately (§1).

**Equal scores.** Orientations that a symmetry of the site alone maps onto each other, without
being symmetry images on the whole slab, score the same up to rounding in the last digits (a
difference of 10⁻¹⁵). Which of them is kept then follows that rounding. The choice is the same
on every x86-64 machine (Linux and Windows, Python 3.10 to 3.14), so a build is reproducible
there; another processor family (Apple Silicon) can keep another of the tied orientations for
a few starts (4 of 128 in the NiFe LDH example). The network, the intermediates and the scores
do not depend on it.

**Shared protons.** An H of the seed within `topology.shared_proton_max_OH_A` (1.6 Å) of two
O, one of them within 1.25 Å, is a proton shared in a short hydrogen bond (the inspector
lists them). When the intermediate changes the site of its acceptor O, the proton is moved
onto its donor at 0.98 Å; the relaxation decides where it ends (§8).

These numbers are construction choices, not fitted values or predictions; the relaxation
moves every atom it is allowed to. One limit is worth knowing: the H-bond term cannot tell a
neighbouring O that only accepts an H-bond from a base that will take the proton during the
relaxation. An OOH whose H points at such a base can lose it (the structure then relaxes into
another intermediate, or into one the network does not list). The second start, at least
0.75 Å away and usually with another H orientation, is the hedge.

## 4. Thermodynamic bookkeeping

Every edge exchanges atoms and electrons with reservoirs. It carries a **reservoir vector**
ν: the amount of each species it produces (negative when consumed), for example
`{H2O: -1, H+: 1}` for \*O + H2O → \*OOH + H⁺ + e⁻. Electrons follow from charge balance,
n = Σ ν_s z_s, positive when released to the electrode. The surface composition change of
an edge i → j must satisfy, element by element, Δn(j) − Δn(i) = −Σ ν_s n_s(element), and
every cycle must sum to exactly one turnover of its reaction (OER: `{H2O: -2, H+: 4, O2: 1}`;
CER: `{Cl-: -2, Cl2: 1}`). `mechanet validate` checks both for every edge and cycle. Every
coadsorbed configuration is one whole-slab calculation; adsorption energies are never added.

The free energy of an edge at potential U is

    ΔG_ij(U) = G_j − G_i + Σ_s ν_s μ_s − n e U_SHE

with, under the CHE,

| Species | μ |
|---|---|
| H⁺ (+ e⁻ at U) | ½ μ(H2) − kT ln10 · pH (and −eU_SHE per electron) |
| H2O | μ(H2O), liquid water through the vapour at its equilibrium pressure, + kT ln a(H2O) |
| O2 | 2 μ°(H2O) − 2 μ(H2) + ΔG(2 H2O → O2 + 2 H2), 4.92 eV at 298.15 K by convention, + kT ln a(O2) |
| Cl⁻ | ½ μ(Cl2) − e U⁰(Cl2/Cl⁻) + kT ln a(Cl⁻), U⁰ = 1.358 V vs SHE at 298.15 K |
| Cl2 | μ(Cl2) + kT ln a(Cl2) |
| plugin species without a rule | E0 + G_corr of the molecule of the same name + kT ln a |

Potentials are reported on the reaction's scale (RHE for OER, SHE for CER; `--scale`
changes it), with U_SHE = U_RHE − kT ln10 · pH. The potentials configured in project.yaml
(`thermodynamics.potentials_V`, on `thermodynamics.potential_scale`) keep their physical
meaning when `--scale` asks for the other scale: they are converted at the analysis pH and the
conversion is noted. pH and activities are analysis settings: changing them needs no new
calculation. On the RHE scale an OER step's free energy does not depend on pH; a CER step
mixing Cl⁻ and protons does.

**Activities** (`thermodynamics.activities` for every analysis; `thermo --activity NAME=VALUE`
for one run) are relative to each
species' standard state: 1 bar for gases, 1 M for ions, pure liquid water. They apply to H2O,
O2, Cl⁻, Cl2 and neutral plugin species; μ°(O2) stays fixed by the water-formation convention
from pure water. Unknown names are refused, and so is H⁺: its activity is the pH. Activities
never move the electrode scale (the CHE keeps p(H2) = 1 bar); for HER, an H2 activity applies
to the released H2 only.

**Temperature.** The molecular corrections G_corr must be computed at the analysis
temperature (the references file declares it). Three constants of the scheme are 298.15 K
values: the water-formation free energy (4.92 eV), U⁰(Cl2/Cl⁻) (1.358 V, −1.25 mV/K) and the
liquid-water equilibrium pressure (0.035 bar). At another temperature the analysis warns,
with estimates (about 4.73 eV, 1.289 V and 0.47 bar at 353 K), and records the warning in its
output, unless the references declare `oxygen_reference: {delta_G_2H2O_to_O2_2H2_eV, temperature_K}`
and `chlorine_reference: {U0_V, temperature_K}` for that temperature and
`thermodynamics.water_equilibrium_pressure_bar` is set. The `--electronic-only` corrections are
298.15 K values too, and warn the same way.

**References.** μ(H2), μ(H2O) and, for CER, μ(Cl2) come from your references file, as
E0 + G_corr computed with the slab's functional, PAW datasets, cutoff, dispersion and
solvation conventions. mechanet never substitutes literature values. Liquid water is
represented by the vapour at 0.035 bar (`thermodynamics.water_equilibrium_pressure_bar`)
[Nørskov 2004]; declare with `Gcorr_pressure_bar` the pressure at which your H2O correction
was computed and the analysis moves it by kT ln(p_eq/p) (a note, not a warning: it is the
convention). `mechanet references --collect` records with each molecule how it was computed
(`protocol`: INCAR protocol tags and PAW datasets; `Gcorr_temperature_K`) and sets the
`provenance` flags again on every collect, from the file as it then stands:
`temperature_verified` when every correction was computed there at the file's temperature,
`water_standard_state_verified` when the H2O correction was computed at 1 bar and declared
so, and `DFT_protocol_compatibility_verified` when, with the campaign at hand, the relax
and freq runs of every molecule have the slab's protocol tags (GGA, METAGGA, LHFCALC,
HFSCREEN, AEXX, IVDW, ENCUT, LASPH, LSOL, EB_K; the hybrid and solvent tags only where
switched on) and PAW datasets. The slab's are those of the stage whose energies the analysis
uses, with `vasp.common_overrides` and `stage_overrides`, which `references --molecules`
also writes into the molecule inputs. Values entered by hand are kept but not
vouched for, and an analysis refuses a correction recorded at another temperature. The
references check reports G_corr(H2O) − G_corr(H2) and the H2 stretch your correction
implies: a correction from a finite-difference run with one H held fixed (a common mistake)
shows up as a stretch near 4330/√2 cm⁻¹ instead of about 4330 cm⁻¹ with PBE.

**Same settings across intermediates.** Every analysis (`thermo`, `pourbaix`,
`pathways --rank`) also compares the intermediates with each other: for each state whose
energy it uses, the protocol tags above, the PAW dataset and the DFT+U values of every
element, read from the INCAR of the energy's stage (04_static, or 02_final with
`frequency.central = final`) and from the state's `potcar_provenance.json`. A difference, such as a cutoff changed
in project.yaml while part of the campaign had already been prepared, is reported as a
warning that names the states (`protocol_check` in thermodynamics.json). It is not refused,
since a converged change can be legitimate. Starting moments, parallelisation and
convergence tags are not compared: they may differ without changing the Hamiltonian.

## 5. Descriptors

For every complete cycle at each potential U, rotated into the turnover convention (a
released product's step last, so the cycle starts right after the release; a consumed
reactant's step first, as for O2 in ORR):

* `largest_step_deltaG_eV`: the largest single-step ΔG(U), the classic limiting step;
* `limiting_potential_V` U_L: the potential closest to equilibrium at which every step runs
  downhill. A step in the reaction's direction (n electrons released in an anodic reaction,
  taken up in a cathodic one) is downhill beyond ΔG(0)/n; a step against it (a family's
  back-step, or one proposed with `pathways --allow-reverse`) only on the near side of its
  own threshold; a chemical step must be downhill at every potential. U_L is None when a
  chemical step is uphill or when no potential satisfies both kinds of step;
  `downhill_potential_range_V` gives the window. η_TD = U_L − U_eq (anodic) or U_eq − U_L
  (cathodic), with U_eq from the references (1.23 V with the 4.92 eV convention);
* `Gmax_U_eV`: Exner's Gmax(U), the largest span G_j − G_i over ordered pairs i < j of the
  cycle, excluding the overall reaction [Exner 2020, 2023; Razzaq 2023];
* `cyclic_span_eV`: the span over the repeating cycle (start-independent);
* `lowest_intermediate_node`: the lowest intermediate of the cycle's own profile at U. It
  belongs to the cycle, so it can be an intermediate that is transient for the network (below).

Independently, every computed node's formation free energy from the reference surface
(with the reaction's formation species) is ranked at each potential (`resting_states.csv`).
The formation species exclude the product, so a node that holds the product bound can rank
lowest although it gives it off: in the RuO2 campaign of the test suite, bound O2 at 1.63 V
vs RHE lies 1.52 eV below the native surface, yet it desorbs downhill to the native surface
plus O2, 1.60 eV below. A node that gives a reaction's product off downhill **without
electron transfer** (independent of U and pH) is therefore **transient**: it is listed with
the reason (`transient` in `resting_states.csv`, `transient_nodes` in `thermodynamics.json`)
and is not a candidate resting state. The release steps considered are the network's edges
and the elementary steps that `mechanet pathways` would propose between computed nodes, so
bound O2 that a family wrote without its release (next to bound water, `hydrated`) is
covered. A node whose *electrochemical* release runs downhill at U (\*OOH above its release
potential) stays a candidate and is flagged (`releases_product_downhill`): its formation free
energy counts the electrons, so it is a genuine termination, but under turnover it gives
the product off. The Pourbaix diagram applies the same two rules. The most stable candidate
is the network's resting state; if an intermediate outside a cycle is more stable than every
candidate intermediate of that cycle, the difference is reported as
`resting_state_penalty_eV` (never negative). This identifies off-cycle sinks within the
sampled space; a kinetically populated resting state would need rates.

**Screening without frequencies.** `--electronic-only` uses E0 plus tabulated
per-atom ZPE − TS corrections (`thermodynamics.electronic_mode_corrections`, 0.30 eV per
added H and 0.05 eV per added O by default) and is labelled as a screening level everywhere.
The defaults suit O-bound OER intermediates on oxides; for H bound to a metal (HER) they are
about 0.1 eV per H too large, so set your own values for other chemistry.

## 6. Pathway discovery

`mechanet pathways` turns the network into a directed multigraph (networkx; one edge per
step, keyed by its ID) and searches it:

1. **Elementary steps** between existing intermediates are proposed when they are local
   (one site, or two sites that form a pair), when at most one site changes its heavy atoms
   (the partner of a two-site step may only gain or lose one proton, or both sites couple,
   decouple or release the product together), and when the step's reservoir vector is one
   of the reaction's elementary vectors (one PCET, water attack or adsorption, an internal
   transfer, a product release, a Cl⁻ adsorption; `Reaction.elementary`). A step that gives
   off the product is a release whatever the sites do, so the HER Tafel step 2 \*H → 2 \* + H2
   is proposed. Only steps in the reaction's direction are proposed unless `--allow-reverse`.
2. **Cycles.** A turnover moves the reaction's *marker* (`Reaction.release`: O2 given off in
   OER, Cl2 in CER, H2 in HER; O2 taken up in ORR) exactly once, so a depth-first search runs
   from the end of each marker step back to its start, over steps of the reaction's species
   that never move the marker backwards. Every closed walk found is kept only if it sums to
   exactly one turnover; for OER, CER, HER and ORR every such walk does, because their
   species admit a single balanced combination. The search prunes branches that move more
   electrons than a turnover (when every step runs in the reaction's direction) or touch
   more than `--max-sites` sites, and stops at `--max-length` steps or `--max-cycles` cycles
   (reported as truncated). Cycles mapped onto each other by a lattice translation form one
   class.
3. **Ranking.** With energies, every cycle is evaluated at each potential (the descriptors
   of §5), and the cycles of each reaction are ranked against that reaction's own U_eq; the
   lists show one cycle per translation class. For every reaction of the network the lowest
   limiting potential is found exactly, without enumeration (so also when `--max-length` or
   `--max-sites` enumerate nothing), over every cycle of any length formed by the steps whose
   two ends have energies (curated and proposed steps; a step to an intermediate without an
   energy cannot take part, and the output says how many steps had energies). A cycle runs
   downhill at U when all its steps do (§5); sweeping U over the thresholds of the steps in
   the reaction's direction, the first U at which a marker step u → v is closed by a
   downhill path v → u is the minimum (a bottleneck-path formulation of the
   minimum-overpotential search of Govind Rajan and Carter 2020). Without steps against
   the reaction's direction the thresholds are bisected; with them (`--allow-reverse`)
   they are scanned in increasing order, because such a step stops being downhill as U
   grows. The best route is reported under its own ID when it is a curated cycle (by the
   representative of its translation class) or an enumerated one, and recorded as a new
   `discovered_*` cycle only otherwise; `--adopt` never copies a cycle that `network.json`
   already holds. For a reaction whose species admit several balanced combinations (for
   example 2e⁻ and 4e⁻ ORR with H2O2 as a species), a closed walk need not be one turnover:
   the sweep is skipped, the output says so, and the best *enumerated* cycle is reported.
   The ranking computes its own free energies in `pathways/analysis/` and leaves the results
   of `mechanet thermo` in `analysis/` alone.
   The test suite checks the sweep against brute-force enumeration, with and without steps
   against the reaction's direction.

Discovered cycles reuse computed intermediates, so they cost no DFT. They combine steps
from different families or proposed automatically; they are hypotheses with the same
status as the curated ones (evidence scope recorded in `network.json` after `--adopt`).

## 7. Surface Pourbaix diagrams

Every computed node forms from the reference surface and the reaction's formation species
(H2O and H⁺ for OER; also Cl⁻ for CER). Its CHE formation free energy is linear in the
potential and the pH,

    ΔG_f(U_SHE, pH) = A + B U_SHE + C pH,   B = −n_e,

where C is the pH slope of the formation species' chemical potentials: −ν(H⁺) kT ln10 when
H⁺ is the only one that depends on the pH, plus the slope of any plugin species whose rule
does (OH⁻, a buffer ion). On the RHE scale C becomes C + n_e kT ln10, which vanishes for
terminations reached by proton-coupled electron transfers only. A plugin rule that is not
linear in the pH is refused here, since the regions would no longer be polygons. The stable region of each termination
is the convex polygon where its plane is lowest (computed exactly by half-plane clipping);
the transitions at a given pH are the exact breakpoints of the lower envelope. Symmetry
images are not repeated; configurational entropy and interactions beyond the computed cell
are neglected, and terminations that were not computed cannot appear. The `coverage` family
enumerates terminations of chosen sites for this purpose [Hansen 2008].

The product is not a reservoir of the diagram (above 1.23 V vs RHE, evolving O2 would
undercut every termination), so a node that holds it bound can look stable although it
gives it off. A node whose chemical release of the product (no electron, no proton: the
same at every U and pH), such as \*OO → \* + O2, is downhill is left out and listed
(`--keep-transient` shows it); the release steps are the network's edges and the steps
`mechanet pathways` would propose, as for the resting states (§5). A node whose release is electrochemical, such as
\*OOH → \* + O2 + H⁺ + e⁻, stays, and every region and transition where that release runs
downhill is flagged (`releases_product_downhill`, a † on the plot): there it is an
intermediate under turnover rather than a resting termination.

## 8. Vibrations, electronic states and assignment

The surface energy and the Hessian belong to the same geometry and energy surface
(docs/execution.md §1). The active-region harmonic contribution is

    F_vib(T) = Σ [hν/2 + kT ln(1 − exp(−hν/kT))],

summed over every active mode of the state, so the reference keeps its own correction. At
298.15 K positive frequencies below 100 cm⁻¹ are raised to that floor (raw and floored values
are both reported; `--cutoff 0` gives the raw expression). Imaginary modes are classified
from the Hessian eigenvectors: H-dominated motion across an O–H bond (a libration) down to
80i cm⁻¹ is floored and listed; H motion along the bond (proton-transfer-like) invalidates
the state at any magnitude; soft modes of O or metal atoms are floored but flagged; larger
imaginary modes always invalidate the state. Partial Hessians cannot rule out instabilities
in the frozen region.

Relaxed connectivity is compared with the starting structure with hysteresis (an O–H bond
is lost beyond 1.35 Å and forms below 1.20 Å; O–O 1.75/1.95 Å; M–O 2.20/2.50 Å; O–Cl and
M–Cl analogous; for elements that plugin adsorbates bring, such as C or N, 1.15 and 1.35
times the sum of the covalent radii, with 1.40 Å for a metal; metal–H the same way, 1.97 and
2.31 Å, when an adsorbate is bound through H). Two site atoms bonded to each other are a
coupled pair when both adsorbates allow it: two O, or two \*CO through C. When it changed, the relaxed
structure is classified site by site with the
adsorbate signatures of the registry and assigned to the node with that occupancy, where it
competes with that node's own variants. Ambiguous structures (a proton in transit, a
detached O–O fragment, an occupancy outside the network) are not assigned automatically and
are listed for review: `mechanet review` describes the change, and a decision in
`states/<state>/review.json` (accept the structure as its own node, or assign it to another
node of the same composition, each with a comment) is applied at the next collect. Large
electronic entropies (|EENTRO| above `analysis.max_eentro_eV`) are flagged, since E(σ→0)
still depends on SIGMA for partially occupied states.

**Shared protons are tautomers, not intermediates.** In a short hydrogen bond (O···O near
2.5 Å) a proton is not owned by either oxygen in any chemically meaningful sense: in
water–hydroxyl layers on metals, path-integral simulations find the distinction between the
covalent bond and the hydrogen bond partly or almost entirely lost, depending on the O···O
distances the substrate imposes [Li 2010], and protons hop quickly between surface oxygens
at wet oxide surfaces [Merte 2012; Tocci 2014]. Which O a
geometry optimization leaves such a proton on is therefore a configuration of the
intermediate, as long as the intermediate does not act on either O. The rule: a shared proton
of the seed (§3) is *background* for an intermediate when both of its oxygens are reactive
sites the intermediate leaves at their native occupancy. Where it sits is read from the H
atoms around its two oxygens, not from atom labels (protons are indistinguishable): as in the
seed, transferred (one H went from the seed's oxygen to the other), or centred (an H within
the upper O–H edge of both); an H bonded to neither, or also close to a third O, leaves it
unread. When the relaxed occupancy, read literally, matches no node, it is read again with
transferred or centred background protons put back as in the seed, and that reading is kept
only if it leaves both of their oxygens native. A structure whose second reading is its own
intermediate is accepted for it (`same_node_other_tautomer`); `analysis/states.json` records
where each shared proton sits (`tautomer`). An exact match comes first: if the relaxed
occupancy, read literally, is another node of the network, the structure counts for that
node. A shared proton next to a site the intermediate changes is chemistry: the builder puts
it on its donor (§3), and wherever it ends up decides the intermediate.

The free energy of an intermediate is that of its lowest tautomer. Another tautomer weighs
exp(−ΔE/kT) relative to it: 14 % at 0.05 eV, 2 % at 0.1 eV and 0.04 % at 0.2 eV (298 K), so
the lowest one is a fair thermodynamic value whatever the gap; there is no threshold on it.
The gap matters for consistency instead: when two intermediates of a cycle were computed in
different tautomers, neither in the other's, the step between them contains the energy between
two tautomers, which nobody computed. `select` lists the intermediates computed in several
tautomers with their energies; `thermo` checks every cycle (`tautomer_check`:
`consistent`, `mixed_both_computed` or `needs_review`, written to cycles.csv and to the
`tautomers` section of thermodynamics.json), and `show-cycle` shows where each shared H sits
and which intermediate was not computed in which tautomer. A proton is compared only between
the intermediates that leave both its oxygens native, and the tautomer of an intermediate is
compared as a whole (all its shared protons at once); a centred proton matches either side.
"Computed" means an accepted structure of the intermediate: its 02_final geometry, the level at
which `select` compares the structures of one intermediate, or the geometry its analysis-level
energy belongs to.

**Discovered intermediates.** A relaxation can end in a sensible intermediate the network did
not list: the H of \*OOH handed to a neighbouring bridge O gives \*OO₆₅ · μ-OH₆₄. Its site
occupancy is read as in the classification above (background protons back on their seed
oxygens, so a tautomer never becomes a node) and named with the same notation as every other
intermediate. `mechanet review --discover STATE "why"` adds it to network.json as a node of
the campaign, with its composition, its lattice-translation images and where it came from;
the structure then counts for it, and `mechanet pathways` proposes the balanced steps and
cycles through it. As in automated reaction-network exploration, where a new minimum found
along the way becomes a member of the network [Unsleber 2022], the node is defined by what
the relaxed structure is; here the user decides, because a single accidental structure is
thin evidence: no orientation search and no spin variants of its own.

**No bound minimum.** A relaxation in which a molecule leaves the surface (an O₂ that lost
its metal, a water that drifted off) is reported in chemical terms ("O₂ (O a0065, O d_a0065)
released from Fe a0003") and gives its intermediate no energy, as relaxations that end in
desorption or dissociation are discarded as anomalies in high-throughput adsorption work
[Lan 2023]. A molecule is released when, after it lost a metal contact, none of its O or Cl
atoms is within the upper M–O (M–Cl) hysteresis edge of any metal; the verdict needs it to be
the intermediate's own adsorbate (an atom of a site the intermediate changes, or one it added)
and no other contact to the metals to have changed. A spectator ligand that left still needs
review. An intermediate whose finished
starts all end that way has no bound minimum from those starts; its cycles are listed apart
from the held-up ones (`mechanet review`, the ∅ mark of `mechanet status`). Every start began
from the same magnetic moments (below), so a start from other moments could still find a
bound one.

Local magnetic moments (the projected moments VASP prints with LORBIT) are reported per atom
and stage, and used in three diagnostics; none of them changes an energy:

* within one state, the moments of 04_static and of every displacement are compared with the
  central structure's (`frequency.max_moment_change_muB`): another electronic solution there
  excludes the free energy;
* along a cycle, the moments of the structures whose energies the analysis uses (02_final
  or 04_static, the energy's own stage) are matched atom by atom through their stable IDs
  (a symmetry image's atoms are relabelled through its lattice translation). The active-site
  atoms are followed per intermediate. Between consecutive intermediates, a moment of at
  least `analysis.moment_change_muB` (0.5 μB) that changes sign anywhere, or a moment that
  changes by more than that farther than `analysis.moment_site_radius_A` (3.6 Å) from every
  metal of the cycle's sites, flags the step: its two intermediates have their spectator
  atoms in different magnetic solutions, or the oxidation went away from the site, and the
  step's free energy contains that difference;
* among the accepted 02_final variants of one intermediate, those whose moments differ in
  the same way are reported as separate magnetic solutions with their energy gap, since the
  gap is then partly a spin-state gap.

Every structure starts from the same MAGMOM (`vasp.magmom_by_element`, `magmom_by_id`);
magnetic states are not enumerated. Moments are not converted to oxidation states, a moment
change is a diagnostic rather than proof of an electronic transition, and a consistent cycle
does not prove the magnetic ground state.

## 9. What is not included

* Activation barriers, constant-potential or explicit-field energetics, microkinetics.
* A complete ensemble of solvent configurations, proton wires or reconstructions;
  placements and explicit-water arrangements are sampled, not exhausted.
* Spin-state or magnetic-order enumeration: one starting MAGMOM per structure (§8).
* Bulk phase changes, dissolution, intercalant exchange, alkaline hypochlorite chemistry.
* Lattice or buried oxygen unless its atom is added as a site.

## 10. Literature

The keys are those of `mechanet.literature.LITERATURE`, recorded in `network.json`.

* rossmeisl2005: Rossmeisl, Logadottir, Nørskov, *Chem. Phys.* **319**, 178 (2005).
* rossmeisl2007: Rossmeisl, Qu, Zhu, Kroes, Nørskov, *J. Electroanal. Chem.* **607**, 83 (2007). https://doi.org/10.1016/j.jelechem.2006.11.008
* norskov2004: Nørskov et al., *J. Phys. Chem. B* **108**, 17886 (2004). https://doi.org/10.1021/jp047349j
* man2011: Man et al., *ChemCatChem* **3**, 1159 (2011). https://doi.org/10.1002/cctc.201000397
* hansen2008: Hansen, Rossmeisl, Nørskov, *Phys. Chem. Chem. Phys.* **10**, 3722 (2008). https://doi.org/10.1039/B803956A
* hansen2010: Hansen, Man, Studt, Abild-Pedersen, Bligaard, Rossmeisl, *Phys. Chem. Chem. Phys.* **12**, 283 (2010). https://doi.org/10.1039/B917459A
* halck2014: Halck, Petrykin, Krtil, Rossmeisl, *Phys. Chem. Chem. Phys.* **16**, 13682 (2014). https://doi.org/10.1039/C4CP00571F
* song2019: Song et al., *ACS Cent. Sci.* **5**, 558 (2019). https://doi.org/10.1021/acscentsci.9b00053
* dionigi2020: Dionigi et al., *Nat. Commun.* **11**, 2522 (2020). https://doi.org/10.1038/s41467-020-16237-1
* exner2020: Exner, *ACS Catal.* **10**, 12607 (2020). https://doi.org/10.1021/acscatal.0c03865
* govindrajan2020: Govind Rajan, Carter, *J. Phys. Chem. C* (2020). https://doi.org/10.1021/acs.jpcc.0c08120
* hao2021: Hao et al., *J. Am. Chem. Soc.* **143**, 1493 (2021). https://doi.org/10.1021/jacs.0c11307
* bai2021: Bai, Lee, Hu, *Angew. Chem. Int. Ed.* **60**, 3095 (2021). https://doi.org/10.1002/anie.202011388
* binninger2022: Binninger, Doublet, *Energy Environ. Sci.* **15**, 2519 (2022). https://doi.org/10.1039/D2EE00158F
* exner2023: Exner, *Mater. Horiz.* **10**, 2086 (2023). https://doi.org/10.1039/D3MH00047H
* razzaq2023: Razzaq, Exner, *ACS Catal.* **13**, 1740 (2023). https://doi.org/10.1021/acscatal.2c03997
* shi2025: Shi et al., *Nat. Chem.* **17**, 1607 (2025). https://doi.org/10.1038/s41557-025-01898-6

Sources motivate the chemical possibilities. They do not establish their energies, kinetic
accessibility or occurrence on a new surface.

Method references (§8), not recorded in `network.json`:

* Li 2010: Li, Probert, Alavi, Michaelides, *Phys. Rev. Lett.* **104**, 066102 (2010). https://doi.org/10.1103/PhysRevLett.104.066102
* Merte 2012: Merte et al., *Science* **336**, 889 (2012). https://doi.org/10.1126/science.1219468
* Tocci 2014: Tocci, Michaelides, *J. Phys. Chem. Lett.* **5**, 474 (2014). https://doi.org/10.1021/jz402646c
* Unsleber 2022: Unsleber, Grimmel, Reiher, *J. Chem. Theory Comput.* **18**, 5393 (2022). https://doi.org/10.1021/acs.jctc.2c00193
* Lan 2023: Lan et al., *npj Comput. Mater.* **9**, 172 (2023). https://doi.org/10.1038/s41524-023-01121-5
