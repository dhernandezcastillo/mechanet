# Execution: stages, continuation, time limits, frequencies and SLURM

This page describes how `mechanet run-state` and `mechanet submit` drive VASP for one state,
what they record, and how work resumes. The physical settings of each stage are fixed in
the state's `plan.json` when the campaign is prepared.

## 1. Stage sequence

`frequency.central` (project.yaml, copied into `campaign/config.json`) chooses one of two
protocols before `prepare`. A prepared state keeps its protocol; changing the setting
afterwards is refused for that state.

* `final` (the default written by `mechanet init`): four stages,
  `01_loose` → `02_final` → `04_static` → `05_freq`. `04_static` is a single point at the
  `02_final` geometry with the `02_final` settings and a tighter EDIFF; it seeds the
  `05_freq` displacements and is checked against `02_final` (below). The node energy is the
  `02_final` E0, and the frequencies belong to the same geometry and Hamiltonian.
* `refine`: five stages, `01_loose` → `02_final` → `03_refine` → `04_static` → `05_freq`.
  The node energy is the `04_static` E0 at the `03_refine` geometry.

`--through loose|final|refine|static|freq` selects the last stage (`refine` exists only in
the refine protocol); earlier accepted stages are verified and skipped.

| Stage | Base INCAR | Changes from the base | Geometry from | Electronic start |
|---|---|---|---|---|
| `01_loose` | INCAR_loose | ISTART 0, ICHARG 2 | `POSCAR` | atomic densities |
| `02_final` | INCAR_final | ISTART 0, ICHARG 2, LWAVE .FALSE., LCHARG .TRUE., LMAXMIX 4 (6 with f elements) | `01_loose/CONTCAR` | atomic densities |
| `03_refine` (refine only) | INCAR_final | ISTART 0, ICHARG 1, PREC Accurate, LREAL .FALSE., ISYM 0, EDIFF `refine.ediff` (1e-6), EDIFFG `refine.ediffg` (-0.015), IBRION `refine.ibrion` (2), NSW `refine.nsw` (60), LWAVE .FALSE., LCHARG .TRUE. | `02_final/CONTCAR` | copy of `02_final/CHGCAR` |
| `04_static`, refine | INCAR_final | ISTART 0, ICHARG 1, EDIFF `frequency.ediff`, NELM `frequency.nelm`, PREC Accurate, LREAL .FALSE., ISYM 0, IBRION -1, NSW 0, LWAVE .TRUE., LCHARG .FALSE. | `03_refine/CONTCAR` | copy of `03_refine/CHGCAR` |
| `04_static`, final | INCAR_final | ISTART 0, ICHARG 1, EDIFF `frequency.ediff`, NELM `frequency.nelm`, ISYM 0, IBRION -1, NSW 0, LWAVE .TRUE., LCHARG .FALSE.; PREC, LREAL and every other tag as in `02_final` | `02_final/CONTCAR` | copy of `02_final/CHGCAR` |
| `05_freq` | INCAR_final | `frequency.method`: `displacements` (default) or `vasp_ibrion5`; the `04_static` settings with ISTART 1, ICHARG 0 | `04_static/POSCAR` | copies of `04_static/WAVECAR` |

Per stage, mechanet also writes MAGMOM from `vasp.magmom_by_element` / `vasp.magmom_by_id`
and the LDAU arrays from `vasp.dft_u` (or reorders the template's arrays for the species
present), so the INCAR templates stay species-independent. `vasp.common_overrides` apply to
every stage (parallelization tags such as NCORE and KPAR belong there).

`stage_settings` rejects overrides that would break the chain: ISYM other than 0 in
static/freq, a non-self-consistent ALGO in static/freq, no CHGCAR from final/refine, no
WAVECAR from static, NFREE other than 2 for the displacement method, or, in the refine
protocol, LREAL Auto in refine/static/freq. In the final protocol `04_static` and `05_freq`
may differ from `02_final` only in restart, SCF tolerance and algorithm, symmetry,
ionic-step, output and parallelization tags; a `vasp.stage_overrides` entry that changes any
other tag for one of these stages, or for `02_final` alone, is refused, because the Hessian
would then belong to another energy surface. An override for a stage the protocol does not
run is refused as well.

Acceptance of a finished VASP run:

* every stage: normal OUTCAR footer, no fatal VASP error, inputs unchanged
  (`input_hashes.json`), and no reported restart fallback;
* relaxations (`01`–`03`): the last ionic step's SCF converged and CONTCAR is valid; a run
  stopped through STOPCAR is never accepted unless it reached its criterion;
* `02_final` and `03_refine`: "reached required accuracy" and the largest force on a
  relaxation-mobile atom at most |EDIFFG|;
* `04_static` and frequency single points: every SCF reached EDIFF.

Before `05_freq` starts, `04_static` is checked; a state that fails gets
`05_freq/BLOCKED.json` with the measured values and is skipped by `submit`
(`skipped_blocked`). The thresholds are read again at every run, so raising one releases a
block.

* Every protocol: the largest `04_static` force on a frequency-active atom must not exceed
  `frequency.max_force_eVA`; above `frequency.warn_force_eVA` a warning is recorded.
  Defaults: 0.035 and 0.03 eV/Å in the final protocol, 0.03 and 0.02 eV/Å in the refine
  protocol.
* Final protocol only: `04_static` must reproduce the `02_final` electronic solution,
  because the node energy comes from `02_final` and the displacements continue the
  `04_static` wavefunction: |E0(04_static) − E0(02_final)| at most
  `frequency.max_static_e0_shift_eV` (0.01 eV) and no local moment changing by more than
  `frequency.max_moment_change_muB` (0.5 μB). A failure means another SCF solution or a
  loosely converged `02_final`. `collect` applies the same rules.

Restart evidence comes from the VASP startup text. A CHGCAR read is recognised by
"charge-density read from file"; "charge density of overlapping atoms calculated" without
that line, or an explicit read error, is a fallback and rejects the run. For WAVECAR, "the
WAVECAR file was read successfully" is the success line; "WAVECAR not read" or a changed
k-point count is a fallback. When neither appears, the attempt carries an "unverified
restart" warning.

## 2. Records in a stage folder

| File | Content |
|---|---|
| `INCAR`, `KPOINTS`, `POTCAR`, `stage.json` | Prepared inputs (refused if edited later) |
| `input_hashes.json` | Hashes of INCAR, POSCAR, KPOINTS, POTCAR; source, size and hash of copied CHGCAR/WAVECAR; geometry source and continuation |
| `attempt.json` | `status` (`running`, `complete`, `interrupted`, `failed`, `launcher_error`), command, host, SLURM job, return code, wall time, parsed result, warnings; for accepted runs the hashes of OUTCAR, vasprun.xml and CONTCAR |
| `previous_attempts/NNN/` | Archived attempts with `archive.json` (`interrupted`, `failed` or `launcher_error`) |
| `continuation.json` | Present only between archiving an attempt and recording the next one |
| `PENDING.json`, `BLOCKED.json` | Not started yet / blocked by the static force gate |

`<state>/run.json` records each run: `status` is `complete`, `incomplete_time_limit`
(stopped cleanly, exit code 0) or `failed` (exit code 2), with the stage reached, signals
received, `vasp_launches` and, for a failure, `failure_kind` (`stage`, `launcher` or
`other`). Restart files are never part of the output hashes, so deleting a WAVECAR or CHGCAR
after the stage that needs it has been accepted affects nothing.

## 3. Continuation versus retry

Every unsuccessful attempt is classified:

* **interrupted**: cut short (no OUTCAR footer and no fatal error, a STOPCAR stop of a
  relaxation, or a SIGUSR1/SIGTERM recorded by the runner). The next run continues
  automatically, **without `--retry`**, up to `vasp.max_continuations` (8) per stage or
  unit. A relaxation continues from the archived CONTCAR when at least one ionic step was
  completed; each continuation receives the stage's full NSW. Single points run again.
* **failed**: VASP finished but the result is unacceptable (unconverged, force criterion
  missed, restart fallback, fatal error, nonzero launcher code). These need **`--retry`**
  and count toward `vasp.max_attempts` (2 attempts in total). Settings are never changed
  automatically.
* **launcher_error**: VASP never started (the launcher could not be executed, or returned
  nonzero without creating an OUTCAR). Fix the environment; the next run re-launches without
  `--retry` and without using either budget.

An attempt left `running` by a process that died is classified from its outputs when the
state lock is free; the verdict is written into `attempt.json` so that `submit` does not
queue the state again.

## 4. Time limits and resubmission

Generated job scripts contain `#SBATCH --signal=B:USR1@<stop_signal_lead_s>` (set in the
cluster profile; by default 1800 s, or a quarter of the walltime for jobs shorter than 2 h)
and `exec` the runner, so the runner receives the signal:

* during a relaxation it writes `STOPCAR` (`LSTOP = .TRUE.`): VASP finishes the current
  ionic step, writes CONTCAR and stops; the attempt is recorded as interrupted;
* a single point or displacement unit in progress is allowed to finish;
* SIGTERM (hard time limit or `scancel`) is recorded immediately as an interruption;
* the run then ends with `run.json` status `incomplete_time_limit` and exit code 0.

Before every stage and every displacement unit the runner compares the remaining
allocation time with `min_remaining_s` (`relaxation` 1800 s, `single_point` 5400 s,
`displacement` 2700 s; settable in the cluster profile) and with measured durations of
earlier attempts (×1.25). A requirement larger than 90% of the allocation is capped at 90%,
so such a stage still starts at the beginning of a fresh job, one per job; a single point
then has to finish within one job. The allocation end is read from `SLURM_JOB_END_TIME`, else
from `squeue`, and its start from `SLURM_JOB_START_TIME`, else from the moment the runner
started; outside SLURM there is no limit.

`mechanet cluster show`, `cluster import` and `submit` check the walltime against these
numbers: a stop signal that would come at the start of every job is an error, and
walltimes shorter than a stage's `min_remaining_s` give a warning. `submit` also flags a
state whose last job ended without starting VASP, and the refill leaves such a state to
you instead of resubmitting it.

To continue, repeat the same command:

```bash
mechanet submit --families aem --through freq --execute
```

Completed stages are skipped, interrupted ones continue, and states whose requested stage
is complete are not submitted again.

## 5. Frequencies by restartable displacements

`05_freq/POSCAR` is the `04_static` central geometry with the frequency selective-dynamics
mask. For every active Cartesian coordinate two single points are run with the atom
displaced by +h and −h, h = `frequency.potim` (0.015 Å):

```
05_freq/units/u001_atom061_x_plus/   05_freq/units/u001_atom061_x_minus/   ...
```

Each unit has its own INCAR (the `04_static` settings with ISTART 1, ICHARG 0,
LWAVE .FALSE., LCHARG .FALSE.), KPOINTS, POTCAR link, displaced POSCAR, a **copied**
WAVECAR from `04_static`, `input_hashes.json`, `attempt.json` and the VASP outputs. Every
displaced POSCAR is read back and must differ from the central geometry by exactly ±h on its
intended coordinate. With `frequency.delete_wavecar_after_use` (default) the unit's WAVECAR
copy is deleted once the unit is accepted, and with `frequency.prune_unit_outputs` (default)
the files in `frequency.unit_prune_files` (PROCAR, DOSCAR, vasprun.xml, EIGENVAL, CHG, PCDAT,
XDATCAR, REPORT, WAVECAR) are removed from accepted units; units are judged from OUTCAR and
CONTCAR only.

Units run one after the other inside the state's single job (they are never separate SLURM
jobs). An interrupted unit reruns automatically; a failed unit needs `--retry`; other units
still run, up to `frequency.max_unit_failures_per_job` (3) new failures per job. When all
2·N units are accepted:

```
H_ij = -(F_i(+h_j) - F_i(-h_j)) / (2h)      (eV/Å², active coordinates only)
H    = (H + Hᵀ)/2
D_ij = H_ij / sqrt(m_i m_j)                 (m from POTCAR POMASS, amu)
ν    = 521.4709 · sqrt(λ) cm⁻¹ for each eigenvalue λ of D (negative ν for λ < 0)
```

Results go to `05_freq/frequencies.json`. `frequency.method: vasp_ibrion5` uses a single
IBRION=5 run instead, which cannot resume individual displacements.

## 6. Cleaning up restart files

```bash
mechanet cleanup                # dry run: lists files and bytes
mechanet cleanup --execute      # deletes them
```

A WAVECAR/CHGCAR is listed only when no pending stage needs it (for example `02_final/CHGCAR`
once `04_static` is accepted in the final protocol, `04_static/WAVECAR` once `05_freq` is
complete). States with a running job are skipped. `--states`, `--families` and `--selected`
restrict the scope.

## 7. Submission ledger and SLURM

`submit` records every sbatch in `<campaign>/submissions.json`. Without `--execute` it only
previews. For each selected state, in this order:

1. `skipped_complete`: the requested stage is accepted;
2. `skipped_submission_unknown`: the outcome of an earlier sbatch is unknown (a lost
   connection, a timeout). Look for a job named `mn_<state>` with `squeue`/`sacct`, then
   record what you found:
   ```bash
   mechanet resolve-submission <state_id> [campaign] --job-id 1234567   # the job exists
   mechanet resolve-submission <state_id> [campaign] --not-submitted    # no job was created
   ```
3. `skipped_active`: the state's last job is still pending or running (checked with
   `--execute` only). Any SLURM state other than a final one counts as active. If the job
   listing fails, nothing is submitted;
4. `skipped_needs_retry`, `skipped_exhausted`, `skipped_blocked`: a job could not make
   progress (a failed attempt waits for `--retry`, a budget is used up, or the static force
   gate blocks `05_freq`);
5. otherwise the state is a candidate. Candidates furthest along the stage sequence go
   first. With a job cap or `--limit N`, only as many as allowed are submitted and the rest
   are `deferred`. A submission writes `run_job.sh` from the campaign's cluster profile,
   records `submitting`, calls `sbatch --parsable` and records the job ID.

States in items 2 and 4 are listed under `attention`: they wait for a decision. Everything
else resolves itself when the same command is repeated.

`--selected` restricts `prepare`, `submit` and `cleanup` to the states in
`thermo_selection.json` (written by `mechanet select`); `--pilot` selects the reference, one
start per adsorbate type and the most crowded start.

The job script is rebuilt from `config.json` at every submission. To change walltime,
modules or the VASP command for an existing campaign, edit the profile and run
`mechanet cluster use NAME`. `mechanet cluster show NAME` prints the exact job script a
profile produces.

What a job script contains, in order: the profile's `shell` (default `/bin/bash`), its
`#SBATCH` options and mechanet's own (`--job-name mn_<state>`, `--signal`, `--chdir`,
`--output`, `--error`); `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` unless the profile
sets them; the `setup` lines verbatim (so `export`, `module`, `source`, `conda` lines work as
in your own script); the `environment` entries (double-quoted, so `$VARIABLES` expand); and
`exec python -m mechanet run-state ...`. The runner launches `vasp_command` itself, without a
shell: `$VAR`, `${VAR}` and `${VAR:-default}` of the job's exported environment are expanded,
and command substitution, pipes and redirections are refused when the profile is checked.

## 8. Locks

A run holds `flock` on `<state>/.run.lock` and the ledger on `<campaign>/.submission.lock`.
On file systems without `flock` an atomic directory lock with host, PID, SLURM job and start
time is used instead; it counts as stale only when its SLURM job is no longer active, or
when it belongs to a dead process on the same host.

## 9. Cluster load: job cap, refill and the amount of work

**One job per state.** A job runs one state's stages in order; in `05_freq` all 2·N
displacement units of that state run inside the same job. The number of jobs a campaign
queues is the number of states submitted, not the number of displacements.

**Job cap.** `--max-active N` (default: `max_active_jobs` of the cluster profile, unset
means no cap) keeps at most N of the campaign's jobs pending or running. It counts the
active jobs recorded in this campaign's `submissions.json`, any other job running in one of
its state folders, and one per submission whose outcome is unknown.

```bash
mechanet submit --selected --through freq --max-active 10 --execute
```

**Refill.** With `--refill` the selected states are added to `<campaign>/refill.json`.
Whenever `run-state` ends inside a job this campaign submitted, it submits the next states
up to the cap, from the compute node, with the rules of §7.

```bash
mechanet submit --selected --through freq --max-active 10 --refill --execute
mechanet refill                    # status: cap, states, recent decisions
mechanet refill --max-active 15    # change the cap
mechanet refill --off              # pause (running jobs continue)
mechanet refill --on               # resume
mechanet refill --clear            # remove the refill
mechanet refill --tick             # submit up to the cap now, with the refill's rules
```

Each job's decision is logged in `refill.json` and as one line `mechanet refill: ...` in the
job's `slurm-<jobid>.err`. The refill pauses itself when a problem would repeat in every job:
a launcher error, a walltime too short for any stage, `sbatch` unavailable inside jobs, three
sbatch errors in a row, or `refill_max_consecutive_failures` (3) jobs in a row whose state
failed without progress. States whose last job made no progress are listed as
`skipped_stalled` instead of being resubmitted. `--retry` is never passed on.

**Clusters that refuse sbatch from compute nodes.** Let a login node submit instead: in
`--submit-from login` mode the jobs only record their end, and each `--tick` submits up to
the cap with all the rules above. Run the ticks in `tmux`/`screen`, from `cron`, or with
`scrontab`:

```bash
mechanet refill --submit-from login --on
while true; do mechanet refill --tick; sleep 1800; done
```

**Amount of work.** The cost of frequencies is the number of displacement single points:
twice the number of frequency-active Cartesian coordinates per state. `mechanet plan`
reports it, and node-hours once the pilot has run. To reduce it before queuing:

* `mechanet select --max-per-node 1` keeps only the lowest 02_final variant of each node;
* `mechanet select --cycles @cycles.txt` restricts the selection to the nodes of the listed
  cycles plus the reference surface.
