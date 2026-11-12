# Design: a coverage shortlist from an ML potential

**Status: designed, not implemented.** This page is the specification for a feature that does
not exist yet. It holds what a later implementation needs: the idea, why it is exact, the
evidence from a benchmark, the algorithms, the traps, where it goes in mechanet and how to test
it. The open question is listed in [Open design questions](pending.md#ml-coverage-shortlist),
and the code is marked `PENDING(ml-coverage-shortlist)` where it would go.

## The idea in one paragraph

A surface Pourbaix diagram needs the free energy of every coverage (every termination the
`coverage` family enumerates), but only the few that are stable somewhere in the window of
potential and pH appear in it. Each coverage is a plane in (U, pH), and the diagram is their
lower envelope, so removing coverages that are never lowest changes nothing: **the envelope of
any subset that contains every truly stable coverage is the true envelope.** An ML potential
does not have to draw the right diagram to be useful. If the coverages it predicts stable (plus
a margin) contain the ones DFT finds stable, then DFT on that shortlist gives the exact
diagram, however wrong the potential's own energies are. A verify loop decides how deep to go
without knowing the answer: compute the shortlist with DFT, measure the potential's error on
those, and keep computing while the next candidate could still be stable within that error.

`mechanet prescreen` today uses the potential only to compare starts of one intermediate.
This feature would use it, for the first time, across coverages: not for their energies, which
stay DFT, but to decide which ones need DFT.

## Evidence: a NiFe LDH benchmark

The design comes from a benchmark of MACE foundation potentials on the hydrogen coverage of a
γ-NiFe LDH edge (the `mlpourbaix` package; 2026). Eight edge sites, four terminal (O or OH) and
four bridging (H or empty), give 256 coverages, 136 distinct up to symmetry, and 25
compositions. DFT: VASP, spin-polarised PBE+U (U(Fe) = 2.56, U(Ni) = 5.20 eV), D3(BJ), VASPsol;
229 converged relaxations (171,122 core-hours) and 24 frequency runs (19,945 core-hours), one
per composition on its lowest isomer. Eighteen model runs were scored: eight zero-shot (seven
MACE foundation models; MACE-MH-1 twice, with FIRE and with BFGS) and ten fine-tunes.

DFT diagram, 0.8 to 2.0 V vs RHE (the composition of the 8 sites, terminal / bridging):

```text
G:  4OH/4H |1.42 V| 4OH/2H |1.71 V| 4OH          (|2.25 V| 4O, outside the window)
E:  4OH/4H |1.74 V| 1O/3OH/3H |2.01 V| 4OH       (no vibrations: another middle phase)
```

What the benchmark found:

| Finding | Number |
|---|---|
| Runs with the right sequence and every transition within 0.05 V | 0 of 18 |
| Zero-shot models: rank (in G) by which the 3 stable coverages are all in | 4 to 10 |
| MACE-MH-1 (head omat_pbe): coverages on its own window hull | 4, holding all 3 |
| Verify loop (below), G: exact diagram | 18 of 18 runs |
| ... DFT runs it spent, zero-shot models | 7 to 15 (MACE-MH-1: 8) |
| ... DFT runs, fine-tunes, beyond their training states | 9 to 23 (18 to 34 in all) |
| ... with the model's own vibrations instead of DFT's (zero-shot) | 6 to 8 runs, 7 of 7 exact |
| Verify loop on energies alone (E), zero-shot models | 0 of 8 exact |
| Verify loop against a second reference (one doubtful vibration on its trend) | 3 of 18 exact |
| Saving in relaxations | 15 to 33 times |
| Saving with the 24 frequency runs counted (they give the vibrational corrections) | 6.5 to 8 times |
| Within one composition, MACE-MH-1's lowest isomer is DFT's lowest | 67 % of 21 compositions (in its 3 lowest: 69 % of 13) |

Every "exact" above is a *replay*: a DFT run of a shortlisted coverage was taken from the
stored relaxation of that coverage, which started from the generated start, not from the
potential's geometry. That a new DFT run from the potential's geometry ends in the same minimum
was not tested; four relaxations from MACE-MH-1's shortlist would test it (about 2,950
core-hours, under 2 % of the reference).

## Why it is exact

Write the formation free energy of coverage *i* as mechanet does
([science §7](science.md#7-surface-pourbaix-diagrams)), `ΔG_i(U, pH) = A_i + B_i U + C_i pH`, a plane over the window
W = [U_min, U_max] × [pH_min, pH_max]. The diagram is the lower envelope
`E_S(x) = min over i in S of ΔG_i(x)` over the set S of coverages, and T, the coverages stable
somewhere in W, are those with `ΔG_i(x) = E_S(x)` for some x in W.

For any subset S' with T ⊆ S' ⊆ S: `E_S'(x) ≥ E_S(x)` (fewer planes), and at every x the minimum
of S is attained by a member of T ⊆ S', so `E_S'(x) = E_S(x)` on all of W. Same envelope, same
regions, same transitions. A shortlist that holds T is enough; its size is what costs.

## The ranking: window gap

The **window gap** of a coverage is how far its plane lies above the envelope at its closest
point in the window:

```text
gap_i = min over x in W of [ΔG_i(x) − E(x)]       (0 for the coverages on the envelope)
```

Ranking by gap is ranking by "how much must the potential be wrong for this coverage to be
stable". It is exact without a grid: ΔG_i is affine and E is concave and piecewise linear, so
ΔG_i − E is convex and piecewise linear, and its minimum over W lies at a vertex of the pieces,
which are the stability regions of the envelope clipped to W. So evaluate ΔG_i − E at every
vertex of the clipped regions (`pourbaix.stability_regions` gives the polygons) and at the
corners of W, and take the smallest. On the RHE scale a coverage reached by proton-coupled
electron transfers only has C = 0 and the problem is one-dimensional: the vertices are the
window edges and the transitions inside the window. The benchmark code did exactly this:

```python
# 1-D (U only): lines F_i - n_i U
us = [u_min, u_max] + [u for u in transitions(envelope) if u_min < u < u_max]
lines = F[:, None] - n[:, None] * us[None, :]
gap = (lines - lines.min(axis=0)).min(axis=1)        # gap < 1e-9 eV counts as on the hull
```

Ties: coverages on the hull all have gap 0, so any count of "how many to compute" must count
ties against the shortlist (a margin that reaches one reaches all at that gap), never by the
order of names. Symmetry twins (images of each other under a rotation or mirror of the slab)
are one structure and one DFT run: rank orbits, not labels, each at the gap of its closer member.

The **shortlist** is the reference coverage plus every orbit with gap 0 in the potential's own
diagram, and the verify loop extends it. Scored against a known DFT answer (a benchmark, not
production), the useful numbers are `n_to_cover` (the rank of the last member of T: DFT runs
that contain T) and `delta_to_cover` (the gap that takes T in).

## The vibrational corrections (Gcorr)

The benchmark's G result leans on the vibrations, and that is where most of the cost hides. A
coverage's free energy is G = E + Gcorr; four sources, best first:

1. **DFT Gcorr of the composition**, from a frequency run on its lowest isomer (mechanet's
   `select` and frequency stage). The benchmark had these for 24 of 25 compositions: 24 DFT
   runs on top of the 7 to 15 relaxations of the loop, which is why, counted in runs, the saving
   drops from 15–33 to 6.5–8 times (a frequency run cost about as much as a relaxation there).
2. **A per-adsorbate trend** fitted on a few DFT frequency runs:
   `Gcorr ≈ g0 + g_OH · n(terminal OH) + g_H · n(bridging H)` (one term per adsorbate kind; the
   bare-site count follows from the others and needs no term). On the LDH: g0 = 0.409,
   g_OH = 0.299, g_H = 0.321 eV; leave-one-out error 34 meV on average, 59 meV at most, except
   4OH/2H at −150 meV (two short O–H···O bonds, O–O 2.47 Å, stretches near 1600 cm⁻¹, where
   the harmonic picture is doubtful). Fit on 3 or more compositions and report the
   leave-one-out residuals: an outlier is a warning, not a correction.
3. **The potential's own vibrations**: a finite-difference Hessian of the frequency region,
   after a tight relaxation of it (the benchmark used ±0.02 Å displacements after relaxing to
   0.005 eV/Å, with the same atoms free as the DFT frequency runs). Free of DFT, and on the LDH
   it worked (6 to 8 runs, 7 of 7 exact), but it inherits the potential's errors.
4. **None** (E only). Do not: the benchmark's loop failed in E for every zero-shot model,
   because the middle phase of E is a coverage in which DFT forms a water molecule during the
   relaxation, which no potential predicted.

Use one thermochemistry recipe for DFT and potential frequencies. mechanet floors frequencies
at 100 cm⁻¹ by default (`vibrational_thermochemistry`); the benchmark used plain harmonic
modes with imaginary ones dropped. The recipe changes Gcorr by meV to tens of meV.

Within one composition, isomers can differ in Gcorr by tens of meV, and the composition's G is
the lowest over its isomers. Computing Gcorr for one isomer only can make a composition look
less stable than it is, never more. On the LDH, 8 of 25 compositions have a second isomer
within 60 meV of the lowest in E (4OH/2H at 43 meV). Where isomers are that close, rank them
in G, not E.

## The verify loop

Production does not know T. The loop decides what to compute from what it has computed:

```text
known    = {reference coverage} ∪ (model hull in W)        # run with DFT first
repeat:
    fit the model error on the known coverages (suspect ones excluded):
        r_i = F_DFT,i − F_model,i,   r ≈ a + b · n_i       # n: H removed (electrons); slope once 3 points exist
        sigma = max(sigma_min, 1.4826 · MAD(residuals))    # robust: one bad state does not open the search
    hybrid diagram: DFT for known coverages, F_model + a + b·n for the others
    take the unknown orbit of smallest gap in the hybrid diagram (ties: first name)
    stop when its gap > z · sigma and at least min_known non-suspect coverages are known
    else compute it with DFT (a failed run costs one and leaves the diagram)
result: the diagram of the known coverages
```

Benchmark settings: `z = 2.0`, `sigma_min = 0.05 eV`, `min_known = 3`. The error model
`a + b·n` matters: each removed H is a different chemical change, and a potential trained on
another level of theory (other U values, no D3, no solvent) errs per H, not by a constant. The
sigma of the known coverages runs low, because they are the ones the potential ranked best (a
random error of 0.12 eV read about 0.08 eV on the LDH): raise z to search deeper.

On a cluster, compute in rounds rather than one coverage at a time: each round, submit every
unknown orbit with gap ≤ z·sigma (or the k smallest), collect, refit. That is fewer rounds for
possibly a few more runs; report the runs spent either way.

## Traps the benchmark found

- **Chemistry changes during the DFT relaxation.** A coverage may relax into another
  (an H hops; water forms, as 1O/3OH/3H did on the LDH; O–O couples and O₂ leaves). Its DFT
  energy then belongs to the coverage it became. mechanet's `collect` and `review` already
  assign a relaxed structure to the intermediate it became; the loop must use that assignment,
  not the label.
- **Doubtful electronic solutions.** ML potentials know no spin. A DFT result on another
  magnetic solution (mechanet's magnetism checks) stays in the diagram but must not enter the
  error fit (`suspect`), nor count towards `min_known`.
- **Gas references.** The formation planes mix slab energies with the reservoirs (H₂O, H₂ or
  H⁺/e⁻, Cl₂ ...). With potential energies for the slabs, take the reservoirs from the same
  potential (a molecule relaxed in a 15 Å box), or DFT references plus the fitted `a + b·n`,
  which absorbs a constant error per H. MACE-MH-1's own H₂ was within 8 meV of DFT; OMAT-0's
  was 0.36 eV off, which tilts every transition by 0.18 V. OC22-trained models (UMA `oc22`,
  SevenNet-Omni `oc22`) never saw gas molecules: use DFT references there.
- **States that break.** Exclude coverages in which the potential couples two O into O₂
  (`exclude_oo_coupled` in the benchmark), and list them: they can still be real.
- **The window edge.** A coverage stable just outside the window can matter. Screening a margin
  past the window (the benchmark tried 0.25 V) costs little.
- **Configurational entropy** and lateral interactions beyond the cell are not in the planes,
  as for DFT Pourbaix diagrams in mechanet today.
- **The reference coverage's level of theory.** Absolute energies of different potentials (and
  of UMA tasks or SevenNet modals) are not comparable; only differences within one potential
  are. Never mix two potentials in one diagram.

## Which potential

Built-in calculators ([pre-screening](prescreening.md#choosing-a-potential)): `mace-mh1`
(MACE-MH-1, head omat_pbe; the best zero-shot model of the LDH benchmark), `uma` (task oc22)
and `sevennet` (SevenNet-Omni, modal oc22), all with FIRE, 0.03 eV/Å, 1500 steps. Only
MACE-MH-1 has been scored on this task. Score a new potential the same way before trusting it:
the rank that contains T and the DFT runs the loop spends, against a system with a full DFT
reference.

## Where it goes in mechanet

Read these first: `prescreen.py` (relaxations, outcomes, `prescreen.json`), `pourbaix.py`
(`formation_lines`, `lower_envelope`, `stability_regions`), `energies.py` (`energy_inputs`,
`node_energies`), `families/coverage.py` (`max_changed_sites`: set `null` to enumerate every
coverage, as the benchmark did), `symmetry.py` (`surface_images`), `thermo.py`
(`vibrational_thermochemistry`).

A possible shape, to be settled when it is built:

1. **Relax every coverage start** with `mechanet prescreen` (exists). Per coverage node, the
   potential's energy is the lowest among its starts that kept it; a start that relaxed into
   another coverage node gives an energy for that node; released or changed starts give none
   and are listed.
2. **The potential's references**: relax the reservoir molecules the reaction needs with the
   same calculator (new; or `--gas dft`).
3. **Gcorr**: `--gcorr dft|trend|model|none` as above; `trend` reads the collected frequency
   results, `model` computes a finite-difference Hessian of each node's frequency region.
4. **Shortlist**: the formation planes of `pourbaix.formation_lines`, built from these energies
   instead of collected DFT ones; the window gap of every node; write
   `prescreen/shortlist.json` (gap, rank, orbit, Gcorr source of every node) and
   `prescreen/shortlist_states.txt` (the lowest kept start of each shortlisted node) for
   `mechanet submit --states @FILE`.
5. **Verify**: after `collect` (and `review` where needed), refit, rebuild the hybrid diagram,
   write the next round's states and a `verify.json` log (one record per decision: node, gap,
   sigma, runs spent), and say when the loop stops. Then `mechanet pourbaix` on the computed
   nodes is the result.

The two `PENDING(ml-coverage-shortlist)` markers sit where the planes are built
(`pourbaix.formation_lines`) and where `prescreen` compares energies (`_suggest`).

## How to test it

- **Window gap.** Random planes: the vertex formula against a dense grid of (U, pH), in 1-D and
  2-D windows, with ties.
- **Subset property.** The envelope of (hull ∪ any extra coverages) equals the full envelope.
- **Verify loop**, deterministic on designed data: a potential equal to DFT plus noise of known
  size gives the exact diagram; one that puts a stable coverage more than z·sigma too high
  misses it, and the test documents that limit; a failed run costs one; a suspect state stays
  out of the fit; ties go to the first name.
- **The benchmark.** The LDH data (DFT and MACE-MH-1 energies of all 256 coverages, the 24
  frequency runs; `hpc_results.tar.gz` of the mlpourbaix project) should give: a 4-coverage
  shortlist in G holding the 3 stable ones, a loop that stops after 8 DFT runs with the exact
  diagram, and a loop in E that does not end exact.
- **End to end** on the NiFe LDH example with a scripted calculator, as `tests/test_prescreen.py`
  does for the prescreen.

## Sources

- Computational hydrogen electrode: Nørskov et al., *J. Phys. Chem. B* **108**, 17886 (2004).
- Surface Pourbaix diagrams: Hansen, Rossmeisl, Nørskov, *Phys. Chem. Chem. Phys.* **10**, 3722 (2008).
- ML relaxation, then DFT: AdsorbML, Lan et al., *npj Comput. Mater.* **9**, 172 (2023); DFT single
  points after ML relaxations of adsorbates: Kazhiyev et al., *J. Phys. Chem. Lett.* **17**, 9686 (2026).
- Potentials: MACE-MH-1, Batatia et al., arXiv:2510.25380; UMA, Wood et al., arXiv:2506.23971;
  SevenNet-Omni, Kim et al., arXiv:2510.11241; OC22, Tran et al., arXiv:2206.08917.
- Benchmarks of foundation potentials for catalysis: Kempen, Cheula, Andersen, arXiv:2512.16702.
- Vibrational corrections from ML Hessians: Wander et al., arXiv:2410.01650.
