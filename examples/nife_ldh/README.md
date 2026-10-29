# NiFe layered double hydroxide edge (OER)

A 90-atom NiFe LDH edge (Fe4Ni12K4O48H22) with its INCAR templates and k-point mesh. Its
vacuum lies along the first lattice vector of a skewed cell, eight interlayer waters are
spectators, and two protons are shared between neighbouring O.

- `project.yaml`: the eight reactive O sites (O61–O68), the six core OER families, the
  partial-Hessian mask of `slab/CONTCAR_frequency_mask` (O61–O68 and H85–H90) and the
  placement settings.
- `reference/campaign.tar.xz`: a frozen build of this project (network and 128 starting
  structures). `tests/test_regression_ldh.py` rebuilds the campaign from `project.yaml` and
  checks that every intermediate, step, cycle, starting structure (coordinates within
  1e-6 Å) and mask is unchanged.
- `inputs/references.yaml`: a template; add the H2 and H2O references computed with this
  protocol (`mechanet references --molecules`).

The INCAR templates use DFT+U for Fe and Ni with their LDAU arrays in the order Fe, Ni, K,
O, H (`vasp.template_species_order`), D3(BJ) dispersion and VASPsol implicit solvation.
