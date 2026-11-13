# Open design questions

Gaps that were looked at and left open on purpose, each with what is known and what a fix
would change. Every entry has an ID. The code carries the same ID as a comment, `PENDING(<id>)`,
at the places a fix would touch:

```bash
grep -rn "PENDING(" src tests
```

`tests/test_pending.py` fails when an entry here has no comment in the code, or a comment has
no entry here. Settle a question by removing its entry together with its comments.

## `tie-break`

**Which of two equally scored orientations is kept depends on the processor.**

The orientation search scores every orientation of a new adsorbate and keeps the best ones
(docs/science.md §3). Two orientations can score exactly the same, mirror images on a symmetric
site for instance. The order of the two is then decided by the last digit of the scores, which
Intel and AMD processors (x86-64) round one way and Apple Silicon (arm64) another.

- x86-64 machines agree with each other: Windows, Linux and the usual clusters give the same
  starts. On Apple Silicon, 4 of the 128 starts of the NiFe LDH example are the other of two
  equally good orientations. Both are valid starts; the files differ.
- `generate` warns on a processor other than x86-64. `tests/test_regression_ldh.py` compares
  the frozen build exactly on x86-64 and within a tolerance elsewhere.
- A rule every processor follows (scores rounded to 10⁻⁹, then the position in the search grid)
  was tried. Taking the lower grid position among ties changes 77 starts of the examples, taking
  the higher one 152, in both cases including starts of the frozen NiFe LDH build (17 of 48).
- Left open because either rule changes starts that existing campaigns and the frozen reference
  were built with. Decide before a release that may break that, and regenerate the reference
  with it.

## `two-site-adsorbates`

**An adsorbate holds the surface through one atom on one site.**

`Adsorbate.proximal` is one element on one site. An adsorbate bound through two atoms on two
sites cannot be written: formate through both O, a carboxyl lying down with C on one metal and
O on the next, CO bridging two metals, a peroxo group between two metals other than the coupled
O–O pair.

- A relaxation that ends that way is not lost: it waits for review with the new metal contacts
  named (`tests/test_co2rr_plugin.py`, `test_two_point_binding_waits_for_review`). It cannot be
  accepted as an intermediate of its own, because no intermediate describes it.
- A coupled pair (`coupling=(a, b)`) covers the one case where the two site atoms bond to each
  other.
- A fix needs an adsorbate that occupies two sites: its template placed over a pair of sites,
  a signature the checks recognise, and step rules for binding and releasing the second atom.
  Rings cannot be written as templates either.
- Not needed for oxygen and chlorine evolution on oxides; needed for CO2 reduction to formate
  and for most C2 chemistry.
- The same limit decides what a slab file may carry (`carried.py`): O2 bound side-on, a
  peroxo pair between two sites, CO on a bridge and H in a hollow are kept as part of the
  framework, with the reason said, instead of being read as states of sites.

## `many-site-placement`

**A surface with more than three adsorbates to orient is placed with two orders, not all.**

The orientation search places the sites of an intermediate one after another and tries every
order, so that an early choice cannot block a later site. The orders grow as a factorial: 24
for four sites, 40,320 for eight, at seconds each. Up to three sites nothing changed. Beyond
that two orders are tried (the sites in their order and in reverse) with two partial
structures kept, which builds a coverage state with OOH on four sites in 10 s instead of 190 s.

- Every start still passes the clash and bond checks, and DFT relaxes it; what the shorter
  search may miss is a better hydrogen-bond arrangement among many adsorbates.
- Not measured: how often the two-order start differs from the best of all orders for four
  and five sites, and whether the relaxed energies differ. Four sites can be compared directly
  (`geometry.orientation_search.all_orders_up_to: 4`).
- A coverage campaign is the case that needs it. Mechanism campaigns on a picked surface start
  from the relaxed surface and orient only the sites that differ, so they do not depend on it.

## `surface-from-discovered`

**A surface that exists only as a structure another start relaxed into cannot be reused yet.**

`mechanet generate --on ID` copies finished calculations and takes the relaxed surface from
the states of that intermediate in the source campaign. A surface added by `review
--discover` (a pattern the coverage scope did not list, found because a start relaxed into
it) has no state of its own with that atom order: its structure belongs to the start it was
found in.

- Such a surface can be picked: the campaign is built on its pattern. Nothing is copied for
  it and its intermediates start from the slab with every adsorbate placed anew; `generate`
  says so.
- A fix writes the found structure as a start of the discovered surface with the builder's
  atom order, as `review --discover` already does for a surface left behind by a release.
