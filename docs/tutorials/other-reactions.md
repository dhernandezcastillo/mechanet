# Tutorial 5: Other reactions and your own mechanisms

The same workflow runs chlorine evolution, reactions added by a plugin, and mechanisms you write yourself. Parts A and B use files from the `examples` folder of the repository (and of the source archive), which the installed package does not include:

```bash
git clone --depth 1 --branch v0.1.3 https://github.com/dhernandezcastillo/mechanet.git mechanet-src
```

## A. Chlorine evolution on RuO2(110)

```bash
cp -r mechanet-src/examples/ruo2_110_cer cer && cd cer
mechanet generate
mechanet cycles
```

```text
id                          family                reaction  steps  symmetry_representative  all_starts_built
--------------------------  --------------------  --------  -----  -----------------------  ----------------
cer_volmer_heyrovsky_m0038  cer_volmer_heyrovsky  cer       2      True                     True
cer_volmer_heyrovsky_m0046  cer_volmer_heyrovsky  cer       2      False                    True
cer_ocl_a0016               cer_ocl               cer       2      True                     True
cer_ocl_a0019               cer_ocl               cer       2      False                    True
cer_ocl_m0038               cer_ocl               cer       2      True                     True
cer_ocl_m0046               cer_ocl               cer       2      False                    True
```

The project differs from the OER one in three lines: `reaction: cer`, the families, and potentials on the SHE scale. `cer_volmer_heyrovsky` adsorbs Cl on the bare Ru and releases Cl2 with a second chloride; `cer_ocl` forms OCl on an oxygen-covered site, the bridging O or a Ru that carries O. That is 9 intermediates, 12 steps and 6 cycles from 7 starting structures. `cer_volmer_tafel`, two Cl recombining, is available but not in the default scope.

Cl2 joins the references, so `mechanet references --molecules refs` now writes refs/H2, refs/H2O and refs/Cl2; continue as in [tutorial 3](cluster.md#4-compute-the-molecular-references). The analysis runs on the SHE scale, from the Cl2/Cl⁻ equilibrium at 1.358 V:

```bash
mechanet thermo --activity Cl-=0.5
```

**Selectivity against OER.** Run the OER network of the same slab (`examples/ruo2_110`) with identical INCAR files, KPOINTS and references, then analyse both reactions at the same potentials and pH:

```bash
# in the OER project
mechanet thermo --scale SHE --pH 0 --potentials 1.36,1.46,1.56 --out campaign/analysis_she
# in the CER project
mechanet thermo --pH 0 --potentials 1.36,1.46,1.56
```

At each potential, compare the lowest Gmax(U) of each reaction in the two `cycles.csv` files: the lower one points to the reaction that wins thermodynamically, without saying anything about barriers. The oxygen-covered Ru is an intermediate of both networks, so the two campaigns compute it twice; keep the protocols identical, or the comparison is meaningless.

## B. A new reaction from a plugin: HER on Pt(111)

`examples/plugins/her.py` adds the hydrogen evolution reaction in about 60 lines: a reservoir species (H2), the reaction (2 H⁺ + 2 e⁻ → H2 on the RHE scale), an adsorbate (H on top of a metal, 1.6 Å) and a family (Volmer, then Heyrovsky, on every bare site). Placement, the VASP stages, thermodynamics, pathways and Pourbaix diagrams come from mechanet unchanged.

```bash
mechanet examples Pt_111
mechanet init Pt_111.vasp pt --plugin mechanet-src/examples/plugins/her.py --reaction her --yes
cd pt
mechanet plugins     # in a project folder it also lists the project's plugins
mechanet generate
mechanet plan
```

`--plugin` loads the file before the questions, so `--reaction` can name the reaction it adds. `init` copies the file into `pt/plugins/`, lists it under `plugins:` in `project.yaml`, chooses the HER family and writes the HER potentials: 0, −0.1 and −0.2 V vs RHE. `generate` copies it into the campaign as well, so the campaign keeps its reaction wherever you copy it.

```text
campaign (her): 10 intermediates (2 computed, 8 symmetry images), 18 steps, 9 cycles (1 symmetry-unique)
  cycles by family: her_volmer_heyrovsky 9
  starting structures: 2
```

Nine top sites are equivalent, so two calculations, the bare slab and H on one site, cover all nine cycles. To share a plugin, package it with an entry point in the `mechanet.families` group pointing at its `register` function; `pip install` then makes it available to every project.

## C. Write your own mechanism family

A family is a Python function that writes intermediates, steps and cycles for the sites of a slab. This one runs the adsorbate-evolution cycle on a site whose neighbour is held as O, a question of coverage the built-in families do not ask. Save it as `my_families.py` in an OER project such as `examples/ruo2_110`:

```python
"""My mechanisms: the adsorbate-evolution cycle next to an oxidised neighbour."""
from mechanet.plugins import Family


def aem_o_neighbour(ctx):
    family = "aem_o_neighbour"
    for a, b in ctx.nearby_pairs():
        for site, neighbour in ((a, b), (b, a)):
            # the cycle runs on a terminal-class site; the neighbour is held as O,
            # unless O is already its native state (then this is plain aem)
            if ctx.site_class(site) != "terminal" or ctx.native[neighbour] == "O":
                continue
            n = {s: ctx.node({site: s, neighbour: "O"}, family, f"{site}+{neighbour}")
                 for s in ("empty", "OH", "O", "OOH")}
            ctx.oer_cycle(
                f"{family}_{site}_{neighbour}",
                [  # (from, to, H2O consumed, O2 released, note); each step moves one H+ + e-
                    (n["empty"], n["OH"], 1, 0, "OH forms from water"),
                    (n["OH"], n["O"], 0, 0, "OH deprotonates"),
                    (n["O"], n["OOH"], 1, 0, "Water attacks the oxo"),
                    (n["OOH"], n["empty"], 0, 1, "OOH deprotonates and O2 leaves"),
                ],
                family,
            )


def register(registry):
    registry.add_family(
        Family("aem_o_neighbour", aem_o_neighbour, "oer",
               "Adsorbate evolution next to an O-covered neighbour")
    )
```

Then in `project.yaml`:

```yaml
scope:
  families: [aem, aem_o_neighbour]
plugins: [./my_families.py]
```

```bash
mechanet generate
mechanet cycles --family aem_o_neighbour
mechanet show-cycle aem_o_neighbour_m0038_m0046
```

```text
aem_o_neighbour_m0038_m0046: Adsorbate evolution next to an O-covered neighbour  [aem_o_neighbour; oer]
  Proposed on this surface
  sites: 38 = m0038: open site on Ru a0038 (bare); 46 = m0046: open site on Ru a0046 (bare)
  1. *₃₈ · *O₄₆ + H₂O → *OH₃₈ · *O₄₆ + H⁺ + e⁻     electrochemical
       n_m0046O_940db2 -> n_m0038OH_m0046O_b0631e   [OH forms from water]
  2. *OH₃₈ · *O₄₆ → *O₃₈ · *O₄₆ + H⁺ + e⁻          electrochemical
       n_m0038OH_m0046O_b0631e -> n_m0038O_m0046O_45a0e8   [OH deprotonates]
  3. *O₃₈ · *O₄₆ + H₂O → *OOH₃₈ · *O₄₆ + H⁺ + e⁻   electrochemical
       n_m0038O_m0046O_45a0e8 -> n_m0038OOH_m0046O_8adfe2   [Water attacks the oxo]
  4. *OOH₃₈ · *O₄₆ → *₃₈ · *O₄₆ + O₂ + H⁺ + e⁻     electrochemical
       n_m0038OOH_m0046O_8adfe2 -> n_m0046O_940db2   [OOH deprotonates and O2 leaves]
  symmetry-equivalent: aem_o_neighbour_m0046_m0038
```

On RuO2(110) this adds two cycles, one per Ru, related by symmetry. The pieces you used:

| In your family | What it gives you |
|---|---|
| `ctx.sites`, `ctx.native[site]` | The reactive sites of project.yaml and their native states |
| `ctx.site_class(site)` | `terminal` (one metal: a terminal ligand or an open metal) or `bridge` |
| `ctx.nearby_pairs()` | Site pairs closer than `scope.pair_max_distance_A` |
| `ctx.node({site: status, ...}, family, label)` | An intermediate. Statuses are adsorbate names (`mechanet plugins` lists them); the same occupation written by two families is one intermediate |
| `ctx.oer_cycle(id, links, family)` | An OER cycle from (from, to, H2O consumed, O2 released, note) links; the H⁺ + e⁻ pairs follow from the atoms |
| `ctx.step(a, b, family, note, given={...})`, `ctx.cycle(id, steps, family)` | The general form for any reaction, as in her.py |

A cycle whose species do not add up to exactly one turnover is refused at `generate`, so a mistake costs seconds, not node-hours. Your intermediates also join the graph that `pathways` searches: with this family, `mechanet pathways` on RuO2(110) finds 36 cycles, 30 of them discovered across the two families.

**Next:** [Troubleshooting](../troubleshooting.md) lists common messages and what to do about them.
