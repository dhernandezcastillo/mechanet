# Planned improvements

Changes that are decided and not built yet. Each one is written for someone who was not part
of the work that led to it: what happens today, what to build, where in the code, which
tests to add and how to check that nothing else changed. Questions that still need a decision
are on another page, [open design questions](pending.md); an entry there with the same ID
points here, and a `PENDING(<id>)` comment marks the place in the code.

## Before you start

**Words.** A *campaign* is a folder with `network.json`, `config.json` and one folder per
*state* under `states/`. An *intermediate* (a `node` in the code) is a pattern of the
reactive sites: which site is in which state. A *surface* is an intermediate seen as the
resting pattern of the whole slab. A *state* (`n_<sites>_<hash>__v1`) is one starting
structure of an intermediate and the calculations run from it. `state.json` names every atom
(`atom_ids`): the atoms of the slab keep the ID they have in the slab file (`a0016`), the
atom that binds to an open metal site carries the site's ID (`m0038`), and the other atoms of
an adsorbate carry a role and the site (`h_m0038`, `w_m0038`, `d_m0038`, `p_m0038`: the H on
the first atom, the second H of water, the outer atom, the H on the outer atom).

**How a campaign on a surface is built** ([surface first](surface-first.md), section 3).
`mechanet generate --on SURFACE` calls `project.generate`, which asks `surfaces.pick` for
`scope.surface`: the pattern, the *source campaign* whose finished calculations are reused
(`<project>/coverage` by default) and the structure the surface starts from. `Builder`
(`builder.py`) reads it in `_read_surface` and `_read_source`. Every intermediate a family
writes goes through `Builder.node` and `_node`: `_import_states` copies the finished
calculations of that intermediate from the source campaign, and when there are none
`_build_starts` makes new starts from the structure of the surface (`self.environment`),
rebuilding only the sites that differ from it.

**Own and arrived.** A finished structure is an intermediate's *own* when it was built as
that intermediate and still shows its pattern after relaxation. It *arrived* when it was
built as another intermediate and relaxed into this one (a proton hopped, a water formed).
`surfaces.structures_of(campaign, node_id)` lists both, lowest energy first.
`mechanet pourbaix` and `mechanet select` give an intermediate the energy of its lowest
structure, own or arrived.

**Checks before a change is done.**

```bash
python -m ruff check src tests
```

```bash
python -m ruff format --check src tests
```

```bash
python -m pytest -q
```

The suite takes about 20 minutes. On Windows 45 tests are skipped (they need POSIX signals)
and run on Linux in the checks of a pull request, which also build these pages with
`sphinx -W`: a broken link or a page missing from a `toctree` fails there. If the checkout
lies in a synchronised folder (OneDrive, Dropbox), run the suite from a copy outside it.

Three tests guard what must not move. `tests/test_public_surface.py` fails when a command,
an option or a public name that a version offered disappears (one list per version in
`tests/data`, from 0.1.4 on), and when a new version has no list of its own.
`tests/test_regression_ldh.py` compares a fresh build of the NiFe LDH example with a frozen
one. `tests/test_pending.py` fails when an
entry of `pending.md` has no `PENDING(<id>)` comment in the code, or a comment has no entry:
when an improvement below is built, remove its entry there, its comments and its ID from the
list in that test, and move its section from this page to the changelog.

**Rules that tests cannot check.**

* A campaign built without `--on` must come out byte for byte as before. Build the example
  projects (`examples/*/project.yaml`) with the code before and after the change and compare
  `network.json`, every `state.json` and every `POSCAR`.
* A source campaign is only read. `tests/test_surface_campaign.py`
  (`test_finished_calculations_are_copied_byte_for_byte`) hashes every file of it.
* A copied state keeps its `state.json` unchanged: `mechanet prepare` hashes that file into
  `plan.json`, and a changed file is refused as a changed protocol. What is known about the
  copy goes to `imported.json` next to it.
* New tests must not depend on which of two equally scored orientations the builder keeps
  (it differs between processors, see `tie-break` in [open design questions](pending.md)),
  nor on how many starts an intermediate gets (with and without symmetry merging), and must
  compare folders as resolved paths (`Path.resolve()`).
* Say in `CHANGELOG.md`, under "Unreleased", what changed for the user (add that heading above
  the newest version if it is not there), and describe a new behaviour in the guide it belongs
  to.

## `copied-intermediates`

**Every intermediate of a campaign on a surface should be represented by its lowest
finished structure. Today only the picked surface is.**

### What happens today

For the picked surface, `surfaces.relaxed_surface` takes the lowest finished structure of the
source campaign that shows the surface, own or arrived (an arrived one must be lower by more
than `selection.DUPLICATE_ENERGY_EV`, 5 meV). An arrived structure is renamed by
`surfaces.found_positions` and the surface starts from it (`scope.surface.found`,
`surface.found_in` in the build report).

For every other intermediate, `Builder._import_states` looks only at the states of the source
intermediate with the same pattern, that is at its own structures:

* If one of them finished and still shows the pattern, it is copied, even when an arrived
  structure of that intermediate lies lower in the source campaign.
* If none did, the intermediate gets a new start from the structure of the picked surface,
  even when an arrived structure of it exists.
* Nothing says that a lower structure exists.

The energy of such an intermediate in the new campaign can therefore lie above the lowest one
the source campaign has, and a step energy that involves it is off by the difference. The
Pourbaix diagram of the source campaign uses the lower structure, so the two disagree.

### What to build

One rule for all: an intermediate of the new campaign for which the source campaign has
finished structures is represented by the lowest of them.

1. The lowest is its own (or no arrived one is lower by more than 5 meV): copy its own
   finished states, as today.
2. The lowest arrived from another start: the intermediate gets one new start at the
   positions of that structure, with its atoms renamed, and is relaxed again in the new
   campaign. Its own finished states are not copied; they are listed in
   `surface.not_copied` with the reason. This is what the picked surface does today.
3. The arrived structure cannot be renamed (`found_positions` raises `ValueError` with the
   reason): fall back to today's behaviour and record the reason.
4. `generate` says how many intermediates start from a structure found in another start,
   and the build report lists them with both energies.

A start made this way does not start from the structure of the picked surface. That is
intended: the rule is the lowest known structure of each pattern, the one the diagram uses.

### Where

`src/mechanet/surfaces.py`

* `structures_of(campaign, node_id)` scans every state of the campaign for one
  intermediate. Add `structures_by_node(campaign, registry=None)`, which scans once and
  returns `{node_id: [records]}` with the same records and the same order, and let
  `structures_of` return one entry of it. A scan of 112 finished states takes about 4 s.
* Move the choice inside `relaxed_surface` (its own unless an arrived one is lower by more
  than 5 meV, the best own kept as `own_structure`) into a function that takes the records,
  so that the builder can use it without scanning again.

`src/mechanet/builder.py`

* `_read_source` indexes the intermediates of the source by pattern (`self.source_nodes`)
  and skips those without states of their own. An intermediate added with
  `mechanet review --discover` has none and is skipped. Keep a second index of the ID of
  every intermediate that is not a symmetry image, by the same key, and call
  `structures_by_node` once (`self.source_structures`).
* `_node` decides, after the symmetry images: `_import_states`, else `_build_starts`. Decide
  there with the records of the intermediate. Case 1 goes to `_import_states` unchanged.
* Case 2 needs a start at given positions. The builder can already make one: for the picked
  surface, every site is in the state it has on the surface, so `_base` takes every atom
  from `self.environment` and the orientation search has nothing to place. Use the same
  path by building the intermediate as if it were the surface: for the one call of
  `_build_starts`, set `self.base` and `self.base_coupling` to its pattern,
  `self.environment` to `{"state_id", "campaign", "positions", "binding_modes": {},
  "found": {...}}` with the positions from `found_positions`, and `self._env_active` to
  True; restore all four afterwards (`try`/`finally`). `_write_state` then records
  `seeded_from` with `structure: "the structure the surface was found in"`; give it a
  wording that fits an intermediate too.
* `_write_state` adds the intermediate to `self.seeded`, which feeds
  `nodes_started_from_the_relaxed_surface`. Count the intermediates of case 2 apart
  (a list of `{node_id, state_id, built_for, energy_eV, own_state_id, own_energy_eV}`) and
  report them as `surface.started_from_found`.
* `_import_states` already lists the own states of the picked surface in `not_copied` when
  the surface starts from an arrived structure (reason: `a lower structure of this surface
  was found in ...`). Do the same for an intermediate, with a reason that starts the same way:
  `cli.cmd_generate` recognises these entries by the words `a lower structure`.

`src/mechanet/cli.py`, `cmd_generate`: one line with the number of intermediates of case 2;
the line about calculations that were not copied says "of the surface itself" and has to
fit both.

### Tests

`tests/test_surface_campaign.py`, class `FoundSurfaceTests`, has the campaign that is
needed: a coverage of the bridging O `a0016` (O, OH) and the metal `m0038` (empty, OH, O) on
RuO2(110), relaxed with the synthetic VASP of `tests/support.py`. One state built as
`a0016: OH` + `m0038: O` is given the relaxed structure of `m0038: OH`, 0.1 eV below that
intermediate's own calculation (`hop_state`, `hydroxo_id`, `oxo_id`), and one state built as
`a0016: OH` + `m0038: OH` is given the structure of `m0038: H2O`, which `review --discover`
adds (`water_state`, `water_id`).

* `mechanet generate PROJECT --on slab --families aem`: the adsorbate evolution on `m0038`
  passes through `m0038: OH`. Expect that intermediate to have one start, at the positions of
  `hop_state` (the helper `farthest` compares two structures atom by atom), no `02_final`
  folder, `seeded_from.state_id == hop_state`; its own states in `surface.not_copied`; the
  entry in `surface.started_from_found` with an energy difference of 0.1 eV; the line of
  `generate`.
* The same call on a campaign whose own calculation is the lower one (give the own state the
  lower energy with `ctl/eshift_02_final`, as `relax_into` does): the own states are copied
  and nothing is listed.
* An intermediate that only has an arrived structure: on a builder made with
  `reuse_source=True` and the source campaign in `scope.surface.source`, ask for
  `node({"m0038": "H2O"}, "aem", "m0038")` on another surface and expect its start at the
  positions of `water_state`.
* A structure that cannot be renamed: patch `mechanet.surfaces.found_positions` to raise
  `ValueError("reason")` and expect today's behaviour and the reason in the build report.
* `SurfaceWorkflowTests` and `PartialSourceTests` in the same file must pass unchanged: their
  source campaigns hold no arrived structure of an intermediate that is used.

### A smaller first step

If the change above has to wait, say it at least. With `structures_by_node` and the index of
`_read_source`, list in the build report every copied intermediate for which the source
campaign has an arrived structure lower by more than 5 meV (`surface.lower_elsewhere`: the
intermediate, its own state and energy, the other state and energy), and print the number in
`generate`. Nothing else changes, so no start and no copied file moves.

## `found-surface-recomputed`

**Not planned. Recorded so that the reason is not found again the hard way.**

When the lowest structure of a surface arrived from another start, the campaign on that
surface relaxes it again, and its frequencies have to be run there too. The finished
calculation is not copied, because its folder belongs to the intermediate it was built as:
`state.json` carries that intermediate's `node_id`, its occupancy, its expected bonds and the
atom names of that pattern, and the VASP files keep that atom order. The new campaign needs a
state whose names match where the atoms are: the builder edits the active site by name, the
frequency region is chosen by name, and the check of a relaxed structure compares bonds by
name.

Copying the folder under its own identity would need the new campaign to hold a state of an
intermediate it does not have, and every command that walks the states (`validate`,
`collect`, `review`, `select`, `status`) to accept it. The run that is saved starts from a
relaxed structure and should take a few ionic steps; the frequencies are the larger cost.
If that cost matters, the cheaper way is to run the frequencies of the surface once, in the
campaign on the surface, and not in the coverage campaign.
