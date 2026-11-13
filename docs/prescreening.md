# Pre-screening the starts with an ML potential

`mechanet generate` writes several starts for most intermediates: the same adsorbate at
different tilts and turns, or bound in different ways. DFT relaxes every one of them.
`mechanet prescreen` relaxes them first with a fast calculator, usually a machine-learned
interatomic potential such as MACE or UMA, checks each relaxed structure the way `collect`
checks a DFT result, and suggests which starts are worth DFT. It removes only redundancy:
copies of a start and higher-energy starts of an intermediate that other starts kept. Every
start the potential changes stays in the list, because a change may be a new intermediate.

Nothing in the campaign changes: the starts, the network and the DFT inputs stay as
`generate` wrote them. The prescreen writes its own folder and a list of states that
`submit` reads.

```bash
mechanet prescreen --calculator mace-mh1 --option device=cuda
mechanet submit --states @campaign/prescreen/suggested_states.txt
```

`submit` shows a preview until you add `--execute`. Without the list, `submit` runs every
start as before: the prescreen is optional, and its suggestion is advice.

## What it checks

Each start is relaxed with the atoms its DFT relaxation would move (the same selective
dynamics), at fixed cell, with the settings the calculator recommends (for the
foundation-potential presets FIRE, to 0.03 eV/Å, in at most 1500 steps; see
[below](#relaxation-settings)) or else mechanet's defaults (BFGS, 0.05 eV/Å, 300 steps).
`--optimizer`, `--fmax` and `--steps` override them. The relaxed structure then goes through
the topology check of `collect`, with the same bond lengths and hysteresis, and gets one of
these outcomes:

| Outcome | Meaning |
|---|---|
| kept its intermediate | the bonds of the intermediate are still there (protons that only changed places count as kept, as in `collect`) |
| relaxed into another intermediate | the sites now hold another intermediate of the network |
| gave its adsorbate off | the adsorbate left the surface (no bound minimum from this start) |
| changed bonds (would need review) | something changed that matches no intermediate of the network; this includes an adsorbate that left with an atom of the surface (a lattice O in the O₂), whose surface left behind is named as a possible new intermediate |
| could not be checked, calculator failed | no verdict |

A `review.json` written for a state's DFT result is not applied to the prescreen's structure.
The changes are described in the same words as in `mechanet review`, for example "H h_m0010
is no longer bonded to O o2_m0010".

## What it suggests

Only redundant starts are left out, among the starts that kept their intermediate. They are
ranked by the calculator's energy, and a start is left out when:

* it relaxed into the same structure as a lower start: the same positions within 0.10 Å and
  the same energy within 5 meV, directly or after a symmetry operation of the slab (a rotation
  or mirror that keeps the fixed atoms, the frequency region and any per-atom MAGMOM in place;
  this needs spglib, a dependency of mechanet). `generate` already writes no two starts that
  are images of each other ([science §3](science.md#3-starting-structures)), so this catches
  different starts that relax into the same minimum;
* it lies more than `--window` (0.20 eV) above the lowest start;
* it is beyond the `--max-per-node` (3) lowest starts.

The summary gives the reason for each. The window is wider than that of `select` (0.10 eV)
because ML energies are less certain than DFT ones. How far to trust the ranking: on the
coverages of a NiFe LDH edge, MACE-MH-1 put the DFT-lowest arrangement of a composition first
in 67 % of 21 compositions, and among its three lowest in 69 % of 13. Widening `--max-per-node`
to 5 costs a few more DFT runs and loses fewer minima.

Every other start is suggested, whatever the potential made of it:

* **A start the potential changed** (it relaxed into another intermediate, gave its adsorbate
  off, or rearranged its sites) stays in the list, even when another start kept the
  intermediate. The potential can be wrong either way, and a change confirmed by DFT is how
  mechanet finds intermediates the network did not list ([`review --discover`](checking-results.md#a-new-intermediate)).
  When the new arrangement of the sites reads as such an intermediate, the summary names it as a
  possible new intermediate, for example "the sites read as \*OO₆₅ · \*(H₂O)₆₈", so you know in
  advance where a discovery may come from.
* **An intermediate none of whose starts kept it** keeps all of them: dropping it would remove
  every mechanism through it.
* **A start the calculator failed on**, or whose structure could not be checked, is suggested.

The savings therefore come from copies and from higher-energy orientations of intermediates
that held, not from anything the potential changed.

Relaxing many placements with an ML potential and computing the best with DFT follows AdsorbML
(Lan et al., *npj Comput. Mater.* 2023, [doi:10.1038/s41524-023-01121-5](https://doi.org/10.1038/s41524-023-01121-5)).
AdsorbML discards relaxations in which the adsorbate desorbs, dissociates or changes the
surface, since it computes one adsorption energy. A reaction network keeps them, because
they may be new intermediates.

## What it does not do

The energies only compare starts of one intermediate, which have the same atoms. They are not
used for reaction energies, limiting potentials or Pourbaix diagrams: adsorption energies from
universal ML potentials are not reliably within 0.1 eV of DFT, and differences of that size
decide which mechanism wins. Using a potential to choose which *coverages* of a surface
Pourbaix diagram need DFT, without using its energies in the diagram, is designed but not built
yet: [a coverage shortlist from an ML potential](coverage-shortlist.md).

An ML potential knows no magnetic moments, charges or Hubbard U, and was trained on another
protocol than yours. Two starts that differ only in their spin arrangement look the same to it.
Treat its relaxations as a cheap first look at the geometry, not as a result.

## Calculators

| `--calculator` | What it is |
|---|---|
| `mace-mh1` | MACE-MH-1 on its OMAT head (`omat_pbe`), in float64 (needs `pip install "mace-torch>=0.3.14"`). Options: `head`, `device` (`auto`, `cpu`, `cuda`), `dispersion` (D3, below), `model` (`mh-1`, or the path of a model file, such as an MH-1 fine-tuned on your data), `default_dtype`, and any option of mace-torch's `MACECalculator` (`enable_cueq`, ...) |
| `uma` | UMA through fairchem (needs `pip install "fairchem-core>=2.24"` and access to the gated weights), model `uma-s-1p2p1`, task `oc22`, `high-precision` inference. Options: `model` (a name fairchem knows, or the path of a checkpoint file), `task`, `device`, `inference_settings`, `dispersion`, and any option of `pretrained_mlip.get_predict_unit` (`overrides`, ...) |
| `sevennet` | SevenNet-Omni (needs `pip install "sevenn>=0.13"`), modal `oc22`. Options: `model`, `modal` (lower case), `device`, `dispersion`, and any option of `SevenNetCalculator` (`enable_cueq`, ...) |
| `mace-mp` | any MACE foundation potential by name (needs `pip install mace-torch`). Options: `model` (a name mace-torch knows, such as `small`, `medium` or `large`, or the path of a model file; default `medium`), `device` (`cpu` or `cuda`), `default_dtype` (default `float64`, as MACE recommends for relaxations), `dispersion` (D3). No recommended settings: mechanet's defaults apply |
| `emt` | ASE's effective-medium toy potential, for testing the workflow only |
| `module:callable` | any factory Python can import that returns an ASE calculator, called with the options, for example `mace.calculators:MACECalculator --option model_paths=my.model --option device=cpu` for a fine-tuned MACE model |
| a name a plugin adds | see below; `mechanet plugins` lists the calculators that are loaded, with the settings each recommends |

Options are given as `--option key=value`, repeated, and read as YAML: `--option
dispersion=true` passes `True`, `--option n=3` passes `3`. `--optimizer` chooses ASE's `BFGS`,
`LBFGS` or `FIRE`; without it, the calculator's recommendation, else BFGS.

### Choosing a potential

The three presets are foundation potentials trained on surfaces with adsorbates, with forces
that are the gradient of the energy (so every optimizer, and a Hessian, behaves as it should).
Which training data a potential reproduces matters more than its size: a potential reproduces
the level of theory of its data, and none was trained on yours.

| Preset | Model | Level of theory it reproduces | Evidence |
|---|---|---|---|
| `mace-mh1` | MACE-MH-1 (Batatia et al., arXiv:2510.25380), head `omat_pbe` | PBE/PBE+U (OMat24), with surface and molecular data through its other heads | On the hydrogen coverage of a NiFe LDH edge (256 states, DFT reference), the closest zero-shot diagram of seven MACE foundation models; its four lowest coverages held every DFT-stable one. The model card recommends this head for surfaces |
| `uma` | UMA (Wood et al., arXiv:2506.23971), `uma-s-1p2p1`, task `oc22` | spin-polarised PBE+U slabs with adsorbates (OC22, oxides) | The `oc20` task came within about 0.1 eV of (non-spin-polarised) DFT adsorption energies on Pt, the `omat` task within about 0.3 eV on IrO₂ (a 2026 benchmark of adsorption on Pt and IrO₂); with organic molecules on Zn it ranked adsorption energies well after ML relaxations but placed the molecules too far from the surface (Kazhiyev et al., *J. Phys. Chem. Lett.* 2026). Not scored on the LDH |
| `sevennet` | SevenNet-Omni (Kim et al., arXiv:2510.11241), modal `oc22` | the same OC22 data, one channel of a model trained on fifteen datasets | State-of-the-art cross-domain accuracy reported by its authors, including adsorption on catalytic surfaces. Not scored on the LDH |

Other UMA tasks: `oc20` (metal slabs with adsorbates, RPBE, no oxides), `omat` (bulk, PBE/PBE+U
at Materials Project settings), `oc25` (slabs with explicit water, RPBE+D3). `oc22` exists in
the UMA 1.2 models only (`uma-s-1p2`, `uma-s-1p2p1`); fairchem lists the tasks a checkpoint has
when you ask for one it lacks. Do not compare absolute energies across potentials, UMA tasks or
SevenNet modals: each has its own energy zero. Within one prescreen only starts of one
intermediate are compared, so this never mixes.

Left out, with the reason:

* **MatterSim** ranked first in one benchmark of ML relaxations followed by DFT single points
  (organic molecules on Zn), but it is built for bulk materials, and its authors advise caution
  outside that scope. Use it through `module:callable` if you want to try it.
* **EquiformerV2 and GemNet-OC trained on OC22**: their forces are a separate output, not the
  gradient of the energy, and they run only on the retired fairchem 1.x stack. UMA's `oc22`
  task covers the same data with conservative forces.
* **eSEN trained on OC25** (explicit water, RPBE+D3): another level of theory from the slabs in
  vacuum that mechanet builds.

### Relaxation settings

The three presets recommend FIRE, 0.03 eV/Å and 1500 steps. 0.03 eV/Å is the force criterion
of mechanet's DFT relaxations (`EDIFFG = -0.03` in the `pbe-d3` preset). FIRE, because on the
LDH coverages MACE-MH-1 gave the same diagram with FIRE and with BFGS, but BFGS formed an O–O
bond in 18 relaxations where FIRE formed none; LBFGSLineSearch stopped with an error. 1500
steps, because FIRE needed up to 457 (87 on average) there. `prescreen.json` records which of
the three settings were given, recommended or defaults (`settings.chosen_by`).

### Dispersion

None of these potentials learned dispersion: OMat24, OC20 and OC22 are PBE (RPBE) without it.
`--option dispersion=true` adds D3 with Becke–Johnson damping and PBE parameters, as VASP's
`IVDW = 12` in the `pbe-d3` preset, through torch-dftd (`pip install torch-dftd`) and with its
pair cutoff of 95 bohr (VASP uses 50.2 Å); `dispersion_cutoff_A` changes it. The MACE-MH-1
model card advises it for systems where dispersion matters. On the LDH, D3 was worth −0.05 to
−0.06 eV per adsorbed H in the DFT energies, but whether it brings a potential closer to DFT
has to be tested: adding it to MACE-MPA-0 made that model's diagram worse. Use it the same way
for every start of a campaign.

### Installing

mace-torch pins an e3nn version that fairchem and sevenn do not accept, so each potential needs
an environment of its own, with mechanet in each (it is light):

```bash
pip install -e ".[mace]"           # from a clone; or [uma], [sevennet]; add [d3] for dispersion
```

Each calculator loads its model file from the path given as `--option model=PATH`, else from
mechanet's models folder (`$MECHANET_MODELS`, else `~/.local/share/mechanet/models`), else from
its package, which downloads it on the first run. UMA's weights are gated. As UMA's
documentation says, request access at <https://huggingface.co/facebook/UMA>, create a Hugging
Face token with "Read access to contents of all public gated repos you can access" (a token of
type Read has it), and give it to huggingface_hub with `hf auth login` (`huggingface-cli login`
in older versions) or the `HF_TOKEN` variable: the first prescreen then downloads the model.
Or download the file in your browser and put it in the models folder.

On a cluster whose compute nodes have no internet, or where you would rather not log in to
Hugging Face there, fill the models folder once: [Model files on a cluster](model-files.md)
gives three ways, among them `mechanet models uma --download` with your token, and
`mechanet models` checks what a job would find before you submit.

### Which model file was used

Every preset stores, in each `result.json` and in `prescreen.json`, the model file it loaded,
its SHA-256, the head, task or modal, the device, the dtype and the package versions
(`calculator.model`). A start is relaxed again when any of these changes, so a retrained file
under the same name is not mistaken for the old one. The file is recorded by its official name,
so its copy in the models folder and the package's own download of the same file are one model. Ask for a head the file lacks and
`mace-mh1` stops with the heads it has: mace-torch alone would quietly use another.

## Your own calculator

A calculator that needs more than keyword options, or one you want under a name of your own,
is added by a plugin file. It may recommend relaxation settings (`relax_settings`):

```python
# my_calculators.py
from mechanet.calculators import make_uma


def make_uma_metals(**options):
    return make_uma(task="oc20", **options)  # metal slabs with adsorbates


def register(registry):
    registry.add_calculator(
        "uma-metals",
        make_uma_metals,
        "UMA, task oc20 (metal slabs with adsorbates)",
        relax_settings={"optimizer": "FIRE", "fmax_eVA": 0.03, "steps": 1500},
    )
```

```bash
mechanet prescreen --plugin my_calculators.py --calculator uma-metals --option device=cuda
```

A package can offer a calculator through the `mechanet.calculators` entry-point group instead
([Extending mechanet](plugins.md#calculators)).

## Files and reruns

In `<campaign>/prescreen/`:

| File | Content |
|---|---|
| `STATE/CONTCAR` | the relaxed structure, atoms in the order of the state's POSCAR |
| `STATE/result.json` | energy, largest remaining force, steps, outcome, the changes found, the calculator, its model file and versions |
| `prescreen.json` | the run: calculator, settings and where each came from, every start, every intermediate, the suggestion, the possible new intermediates, warnings |
| `prescreen.csv` | one row per start |
| `suggested_states.txt` | one state ID per line, for `--states @FILE` |

A start whose POSCAR, calculator (class, versions, options, optimizer, model file and its
SHA-256), force criterion and step limit are unchanged is not relaxed again: changing `--window`
or `--max-per-node` only recomputes the suggestion. `--redo` relaxes everything again; starts
the calculator failed on are always tried again. `--states` and `--families` restrict the run
(the suggestion then covers only those starts).

Most ML potentials print a banner when they load; it goes to stderr, with the progress lines,
so `--json` output stays clean.
