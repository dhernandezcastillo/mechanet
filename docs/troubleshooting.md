# Troubleshooting and quick reference

## Messages and what to do

| Message | Meaning | What to do |
|---|---|---|
| `No campaign here: pass its directory, or run inside a project after mechanet generate` | The command works on a campaign and found none | Run it in the project folder after `mechanet generate`, or pass the campaign folder |
| `This campaign has no cluster profile` (or the warning `No cluster profile in project.yaml`) | Nothing tells mechanet how to submit | `mechanet cluster import JOB_SCRIPT --name NAME`, then `mechanet cluster use NAME`, and `cluster: NAME` in project.yaml ([tutorial 3](tutorials/cluster.md#2-describe-the-cluster-once)) |
| `references incomplete: H2 needs G_eV, or both energy_e0_eV and free_energy_correction_eV` | inputs/references.yaml has no value for a molecule | `mechanet references --molecules refs`, run them, then `--collect refs` ([tutorial 3, step 4](tutorials/cluster.md#4-compute-the-molecular-references)) |
| `Reference DFT protocol identity has not been documented` | `protocol_id` is empty | `mechanet references --collect` writes it; in a file you fill in yourself, describe the protocol ([tutorial 4, step 1](tutorials/analysis.md#1-screen-before-the-frequencies)) |
| `node-hours: unknown. No finished runs yet` | `plan` has no timings from this cluster | Run the pilot, then `mechanet plan` again |
| A `skipped_...` status from `submit` | A state that needs you | See the table in [tutorial 3, step 6](tutorials/cluster.md#6-run-everything) |
| `Unknown reaction 'her' ... (a reaction from a plugin file needs --plugin FILE)` from `init` | The reaction comes from a plugin file that mechanet has not loaded | Give the file: `mechanet init SLAB --plugin FILE --reaction her` ([tutorial 5, part B](tutorials/other-reactions.md#b-a-new-reaction-from-a-plugin-her-on-pt111)) |
| `init` proposes no sites, or the wrong ones | Nothing on the surface looks undercoordinated to the inspector | Add or edit sites in project.yaml, e.g. `- {id: m0040, kind: open_metal, native: empty, metal: a0040}` with your atom's ID, and check slab/site_map.html |
| `The slab file ... lists an element in more than one block` from `generate` | A project written by hand points at a slab file whose elements are not grouped, which a POSCAR cannot hold | Run `mechanet init` on the slab file in a new folder: it writes a grouped copy and proposes the sites on it. Atom IDs follow the new order, so site IDs written for the old order point to other atoms. `init` groups the slab of a new project by itself |
| `PAW datasets of H2, H2O not compared` from `references --collect` | The slab's PAW datasets are not readable where you ran it (no job of the campaign has started and the PAW library of the cluster profile is not on this machine), or the molecule runs have no POTCAR or OUTCAR | Run `mechanet references --collect refs` again in the project folder on the cluster; until then `DFT_protocol_compatibility_verified` stays `false` |
| `protocol differs from the slab's: H2O ENCUT: molecule 400, slab 500` | That molecule was computed with other settings than the INCAR whose energies the analysis uses (overrides included) | Compute it again from the inputs `mechanet references --molecules refs` writes, then collect again |
| `spin check (02_final moments): 2 of 9 cycles with moments need review` from `thermo` | In those cycles a local moment changed sign between two intermediates, or a moment away from the active site changed by more than 0.5 μB | `mechanet show-cycle ID` names the atoms and steps. If it is not the chemistry, that step's ΔG mixes two magnetic solutions: recompute one of the two intermediates from the other's moments and compare ([tutorial 4, step 3](tutorials/analysis.md#3-look-at-one-cycle)) |

| `N finished structures gave no energy to their own intermediate` from `collect` or `thermo` | Those structures finished, but their bonds changed during the relaxation (or a check failed), so their energy is not used for their intermediate | `mechanet review` says what changed and what to do ([Checking results](checking-results.md)) |
| `no moments for X: no structure with an accepted energy` from `show-cycle` | No structure of that intermediate was accepted | `mechanet review` |
| `cycles with every energy: 0 of N` while `status` shows complete cycles | The references are incomplete, or an intermediate of each complete cycle has no accepted structure | Look for `references incomplete` in the `thermo` output; then `mechanet review` |
| `a review.json changed after the last thermo` from `show-cycle` | A decision was written after the last analysis | `mechanet thermo` again |

For anything else, run the command again with `--traceback` and keep the output for the bug report.

## Commands at a glance

| Stage | Commands |
|---|---|
| Start | `examples NAME`, `inspect SLAB`, `init SLAB [DIR]` |
| Build and look | `generate`, `plan`, `cycles`, `show-cycle ID`, `list`, `validate` |
| Set up the cluster | `cluster import / list / show / use`, `prepare --potcars`, `references --molecules DIR`, `references --collect DIR` |
| Run | `submit --pilot`, `submit --execute`, `submit --retry`, `submit --max-active N --refill`, `submit --selected --through freq`, `status`, `refill`, `resolve-submission`, `cleanup` |
| Analyse | `collect`, `review`, `select`, `thermo`, `plot-profile ID --potential U`, `pathways --rank`, `pourbaix --plot`, `report` |
| Extend | `plugins` |

`run-state` is what each job runs; you never call it yourself. Every command takes `--json` for scripts and `--help` for its options. Only `submit --execute`, `refill --tick` (for a refill that `submit --execute --refill` set up) and `cleanup --execute` submit jobs or delete files.

## Files at a glance

| Path | Contents |
|---|---|
| `project.yaml` | Every decision about the project; `[auto]` marks what `init` detected and you should check |
| `slab/site_map.html` | The sites on a picture of the slab |
| `inputs/` | `INCAR_loose`, `INCAR_final`, `KPOINTS`, `references.yaml` |
| `~/.config/mechanet/clusters/NAME.yaml` | Cluster profiles |
| `campaign/network.json` | Intermediates, steps and cycles |
| `campaign/method_card.json` | The protocol and counts, written by `plan` |
| `campaign/states/STATE/` | One calculation: `run.json`, a folder per stage, the job's `slurm-ID.out`, and `review.json` when you decided about its relaxed structure |
| `campaign/thermo_selection.json` | The variants `select` kept |
| `campaign/analysis/` | Tables, profiles and the moments along each cycle (cycle_moments.csv) from `collect` and `thermo` |
| `campaign/pathways/` | `pathways.json`, `pathways.csv`, `ranking.json` |
| `campaign/pourbaix/` | `pourbaix.json`, `pourbaix.csv`, `pourbaix_1d.csv`, the figure |
| `campaign/report.md` | The summary from `report` |
| `campaign/status.html` | Every state and cycle with its progress, from `status` |
