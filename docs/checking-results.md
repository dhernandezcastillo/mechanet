# Checking results: `mechanet review`

A structure counts as *finished* when its 02_final relaxation ran to the end and the runner
accepted the run. Its energy only enters the analysis once `collect` also accepts it:

- **its files** are still the ones the runner accepted, and its inputs those it ran with;
- **the relaxation converged**: electronically, and with every movable atom below the EDIFFG
  force criterion;
- **the relaxed structure is still the intermediate it was built for.** `collect` compares its
  bonds with the starting structure: which H sits on which O, whether O–O bonds survived, and
  which O atoms are bound to which metal.

On real surfaces the third check is the one that matters. A proton hops to a neighbouring O,
an OOH gives its H to a water molecule, an O₂ leaves the metal, a water molecule binds to an
open site. Such a structure is still a valid calculation, but no longer of the intermediate
it was meant to be, so its energy cannot be used for it.

## The command

```bash
mechanet review
```

It collects the campaign, then lists every finished structure that gives no energy to its own
intermediate, says in plain words what changed, which complete cycles that holds up, and what
to do:

```text
7 finished structures: 6 accepted for their intermediate, 1 not

Needs your review: the relaxed structure changed
  n_0065OOH_cfb68d__v1  *OOH₆₅
      H p_a0065 moved from O d_a0065 to O a0066
      outside the site model: the proton count on framework oxygen a0066 changed
      another start of *OOH₆₅: n_0065OOH_cfb68d__v2 (not run yet)

Cycles (2 symmetry-unique): 0 with an accepted structure for every intermediate
Held up (every intermediate finished, some without an accepted structure):
  aem_a0065_AEM  Adsorbate evolution (AEM)
      *OOH₆₅: needs review; 1 more start not finished
  aem_a0065_bound_OO  AEM with O₂ held on the site before it leaves
      *OOH₆₅: needs review; 1 more start not finished

What to do: look at each structure (states/STATE/02_final/CONTCAR), then
  mechanet review --accept STATE "why"         it is still its intermediate
  mechanet review --assign STATE NODE "why"    it is another intermediate of the same composition
  mechanet submit --states STATE --execute     run another start of the intermediate
and run mechanet thermo again: it reads the decisions.
```

Atoms are named by element and ID: `a0066` is the 66th atom of the slab file, and the atoms an
adsorbate adds carry the site they belong to (`d_a0065` is the outer O and `p_a0065` the H of
the OOH on site a0065). `collect` and `thermo` say how many finished structures were not
accepted, and `show-cycle` points here when an intermediate has no energy.

## The four outcomes

| Outcome | What it means | What to do |
|---|---|---|
| Accepted | The structure is its intermediate; its energy is used | Nothing |
| Relaxed into another intermediate | Its bonds now describe another intermediate of the campaign, typically because a proton moved to a neighbouring site. Its energy counts for that one, where it competes with that intermediate's own structures | Nothing for the structure. The intended intermediate needs another start that stays intact, or this is the chemistry: the intermediate is not a minimum there |
| Needs your review | The bonds changed into something no intermediate of the campaign describes, or a bond to the metal changed | Look at the relaxed structure, then decide (below), or run another start |
| Rejected | Files changed after the run, or convergence or forces fail | Read the reasons; a run whose OUTCAR or CONTCAR changed afterwards has to be restored or run again |

## Deciding

Look at the relaxed structure (`states/STATE/02_final/CONTCAR`) next to its starting
structure (`states/STATE/POSCAR`). Then:

- **It is still the intended intermediate.** For example, a proton shared between two
  neighbouring OH groups moved to the other O, far from the cycle's sites, and nothing at the
  site itself changed. Accept it, with your reason:

  ```bash
  mechanet review --accept n_0065empty_a500dc__v1 "Site 65 stays a bare metal; only the shared proton next door moved"
  ```

- **It is another intermediate of the campaign with the same atoms.** Count it for that one:

  ```bash
  mechanet review --assign STATE NODE "why"
  ```

  `mechanet show-cycle ID` lists the IDs of a cycle's intermediates under its steps. A
  structure can only count for an intermediate with the same composition.

- **It is neither.** Leave it. If the intermediate has another start that has not run, run it
  (`mechanet submit --states STATE --execute`). If every start ends the same way, the
  intermediate is not a minimum on this surface, which is itself a result about the mechanism.

Each decision is a small file, `states/STATE/review.json`, with your comment, so it stays with
the structure and can be read later:

```json
{"accept_relaxed_topology": true, "comment": "Site 65 stays a bare metal; only the shared proton next door moved",
 "written_by": "mechanet review", "written_utc": "2026-11-02T15:51:39+00:00"}
```

`{"assign_to_node": "NODE", "comment": "..."}` counts the structure for another intermediate.
`mechanet review --clear STATE` removes the decision. After any change, run `mechanet thermo`
again: decisions take effect when the results are collected.

Accept only what you would defend in a paper. A structure accepted for an intermediate brings
everything that moved with it into that intermediate's energy: the ΔG of the steps on both
sides includes it. When the change is chemistry, such as an OOH that gives its proton away as
soon as it forms, the honest answer is a different mechanism, not a relabelled energy.

## On the status page

`mechanet status` shows what the last collect did not accept. These structures keep their
group, finished, but their square carries a question mark and a red ring, and a table lists
them with what happened:

![The status page with a structure the last collect did not accept: the table "Finished, but not accepted" and the marked square in the AEM grid](images/status-not-accepted.png)

A complete cycle with such a structure is held up: its fraction is no longer in bold, and the
summary counts it. `mechanet status --cycles` marks the intermediate with `?`. The marks come
from the last `collect` (or `review` or `thermo`, which collect too); a structure that ran or
was decided since then shows without a mark until the next one.

The thresholds of the bond comparison and the classification of changed structures are in
[What mechanet computes](science.md#8-vibrations-electronic-states-and-assignment).
