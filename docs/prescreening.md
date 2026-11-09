# Pre-screening the starts with an ML potential

`mechanet generate` writes several starts for most intermediates: the same adsorbate at
different tilts and turns, or bound in different ways. DFT relaxes every one of them.
`mechanet prescreen` relaxes them first with a fast calculator, usually a machine-learned
interatomic potential such as MACE or UMA, checks each relaxed structure the way `collect`
checks a DFT result, and suggests which starts are worth DFT.

Nothing in the campaign changes: the starts, the network and the DFT inputs stay as
`generate` wrote them. The prescreen writes its own folder and a list of states that
`submit` reads.

```bash
mechanet prescreen --calculator mace-mp --option model=medium
mechanet submit --states @campaign/prescreen/suggested_states.txt
```

`submit` shows a preview until you add `--execute`. Without the list, `submit` runs every
start as before: the prescreen is optional, and its suggestion is advice.

## What it checks

Each start is relaxed with the atoms its DFT relaxation would move (the same selective
dynamics), at fixed cell, to `--fmax` (0.05 eV/Å) in at most `--steps` (300) steps. The
relaxed structure then goes through the topology check of `collect`, with the same bond
lengths and hysteresis, and gets one of these outcomes:

| Outcome | Meaning |
|---|---|
| kept its intermediate | the bonds of the intermediate are still there (protons that only changed places count as kept, as in `collect`) |
| relaxed into another intermediate | the sites now hold another intermediate of the network |
| gave its adsorbate off | the adsorbate left the surface (no bound minimum from this start) |
| changed bonds (would need review) | something changed that matches no intermediate of the network |
| could not be checked, calculator failed | no verdict |

A `review.json` written for a state's DFT result is not applied to the prescreen's structure.
The changes are described in the same words as in `mechanet review`, for example "H h_m0010
is no longer bonded to O o2_m0010".

## What it suggests

For each intermediate, the starts that kept it are ranked by the calculator's energy. Starts
that relaxed into the same structure are merged: the same positions within 0.10 Å and the same
energy within 5 meV, directly or after a symmetry operation of the slab (a rotation or mirror
that keeps the fixed atoms, the frequency region and any per-atom MAGMOM in place; this needs
spglib, a dependency of mechanet). Of the rest, those within `--window` (0.20 eV) of the lowest
are suggested, at most `--max-per-node` (3).

A start is left out only for a reason the prescreen found, and the summary gives it:

* it relaxed into the same structure as a lower start, or a symmetry image of one;
* it is more than the window above the lowest start, or beyond the cap;
* it relaxed into another intermediate, gave its adsorbate off or changed bonds, while another
  start kept the intermediate.

Two rules keep the potential from deciding what only DFT can:

* An intermediate none of whose starts kept it keeps **all** of them. A potential that pulls
  an adsorbate apart may be wrong, and dropping the intermediate would remove every mechanism
  through it.
* A start the calculator failed on, or whose structure could not be checked, is suggested.

The window is wider than that of `select` (0.10 eV) because ML energies are less certain than
DFT ones.

## What it does not do

The energies only compare starts of one intermediate, which have the same atoms. They are not
used for reaction energies, limiting potentials or Pourbaix diagrams: adsorption energies from
universal ML potentials are not reliably within 0.1 eV of DFT, and differences of that size
decide which mechanism wins.

An ML potential knows no magnetic moments, charges or Hubbard U, and was trained on another
protocol than yours. Two starts that differ only in their spin arrangement look the same to it.
Treat its relaxations as a cheap first look at the geometry, not as a result.

## Calculators

| `--calculator` | What it is |
|---|---|
| `mace-mp` | a MACE foundation potential (needs `pip install mace-torch`). Options: `model` (a name mace-torch knows, such as `small`, `medium` or `large`, or the path of a model file; default `medium`), `device` (`cpu` or `cuda`), `default_dtype` (default `float64`, as MACE recommends for relaxations), `dispersion` (D3) |
| `emt` | ASE's effective-medium toy potential, for testing the workflow only |
| `module:callable` | any factory Python can import that returns an ASE calculator, called with the options, for example `mace.calculators:MACECalculator --option model_paths=my.model --option device=cpu` for a fine-tuned MACE model |
| a name a plugin adds | see below; `mechanet plugins` lists the calculators that are loaded |

Options are given as `--option key=value`, repeated, and read as YAML: `--option
dispersion=true` passes `True`, `--option n=3` passes `3`. `--optimizer` chooses ASE's `BFGS`
(default), `LBFGS` or `FIRE`.

A calculator that needs more than keyword options, such as UMA through fairchem, is added by a
plugin file:

```python
# uma_calculator.py
def make_uma(model="uma-s-1p1", task="oc20", device="cpu"):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    return FAIRChemCalculator(pretrained_mlip.get_predict_unit(model, device=device), task_name=task)


def register(registry):
    registry.add_calculator("uma", make_uma, "UMA through fairchem (task oc20: slabs with adsorbates)")
```

```bash
mechanet prescreen --plugin uma_calculator.py --calculator uma --option device=cuda
```

The UMA weights are gated on Hugging Face (request access and log in first), and fairchem's
interface changes between releases: check its documentation. A package can offer a calculator
through the `mechanet.calculators` entry-point group instead
([Extending mechanet](plugins.md#calculators)).

## Files and reruns

In `<campaign>/prescreen/`:

| File | Content |
|---|---|
| `STATE/CONTCAR` | the relaxed structure, atoms in the order of the state's POSCAR |
| `STATE/result.json` | energy, largest remaining force, steps, outcome, the changes found, the calculator and its versions |
| `prescreen.json` | the run: calculator, settings, every start, every intermediate, the suggestion, warnings |
| `prescreen.csv` | one row per start |
| `suggested_states.txt` | one state ID per line, for `--states @FILE` |

A start whose POSCAR, calculator (class, versions, options, optimizer), `--fmax` and `--steps`
are unchanged is not relaxed again: changing `--window` or `--max-per-node` only recomputes the
suggestion. `--redo` relaxes everything again; starts the calculator failed on are always tried
again. `--states` and `--families` restrict the run (the suggestion then covers only those
starts).

Most ML potentials print a banner when they load; it goes to stderr, with the progress lines,
so `--json` output stays clean.
