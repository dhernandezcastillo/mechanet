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
