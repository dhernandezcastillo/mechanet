# Examples

| Folder | System | Reaction, scope | Built network (`mechanet generate`) |
|---|---|---|---|
| `ruo2_110` | RuO2(110) 2×1, 48 atoms: 2 bridging O, 2 CUS Ru | OER, core families | 63 intermediates, 146 steps, 100 cycles, 69 starts |
| `iro2_110` | IrO2(110) 2×1, 48 atoms | OER, core families | 63 intermediates, 146 steps, 100 cycles, 69 starts |
| `pt_111` | Pt(111) 3×3×4, 36 atoms, 9 top sites (one class) | OER, adsorbate evolution | 37 intermediates (5 computed), 54 steps, 18 cycles, 10 starts |
| `ruo2_110_cer` | RuO2(110) 2×1 | CER, Volmer-Heyrovsky and OCl routes | 9 intermediates, 12 steps, 6 cycles, 7 starts |
| `nife_ldh` | NiFe LDH edge, 90 atoms, skewed cell, interlayer water | OER, core families | 189 intermediates, 488 steps, 368 cycles, 128 starts |
| `plugins/her.py` | A plugin adding the hydrogen evolution reaction | HER on metal top sites | used by `tests/test_plugins.py` on Pt(111) |

Each project folder has `project.yaml`, the slab under `slab/` (with the inspection report
and the site map `slab/site_map.html`), and `inputs/` with INCAR templates, KPOINTS and a
references template. To run one:

```bash
cd examples/ruo2_110
# set `cluster:` in project.yaml (`mechanet cluster import JOB_SCRIPT --name NAME` first)
mechanet generate
mechanet plan
```

The oxide and metal slabs come from `build_slabs.py` (pymatgen and ASE) with experimental
lattice constants (Bolzan et al., Acta Cryst. B 53, 373 (1997) for the rutiles). Relax the
bulk with your own functional and rebuild before production calculations; the INCAR
templates are the generic `pbe-d3` preset and need the same review.
