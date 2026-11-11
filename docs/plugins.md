# Extending mechanet: reactions, species, adsorbates, mechanisms, commands and calculators

Everything chemistry-specific in mechanet is a plugin, including the built-in OER and CER
chemistry. A plugin can add:

| Piece | Class | Registry call | Entry-point group |
|---|---|---|---|
| Reservoir species (H2, N2, NH3, ...) | `mechanet.reactions.Species` | `registry.add_species(species)` | (inside a `register` function) |
| Reaction | subclass of `mechanet.reactions.Reaction` | `registry.add_reaction(cls)` | `mechanet.reactions` (the class) |
| Adsorbate | `mechanet.adsorbates.Adsorbate` | `registry.add_adsorbate(ads)` | `mechanet.adsorbates` (`register`) |
| Mechanism family | `mechanet.plugins.Family` | `registry.add_family(family)` | `mechanet.families` (`register`) |
| CLI command | `register_command(subparsers, registry)` | `registry.add_command(name, fn)` | `mechanet.commands` |
| Calculator for `prescreen` | a factory returning an ASE calculator | `registry.add_calculator(name, factory, description)` | `mechanet.calculators` (the factory) |

Two ways to load one:

* **A local file** listed in project.yaml, `plugins: [./my_chemistry.py]`. The file defines
  `register(registry)`. `mechanet init SLAB --plugin FILE` loads it before the questions (so
  `--reaction` may name a reaction it adds) and copies it into the project's `plugins/`
  folder. `mechanet generate` copies the plugin files into the campaign
  (`inputs/plugins/`, listed relative to the campaign in `config.json`), so every later
  command (show-cycle, collect, thermo, pathways, pourbaix, ...) loads them again, also
  from a campaign copied to a cluster. A file is executed once per process for each
  content: loading it again, or a copy of it, reuses the module, so the objects it defines
  are the same in every registry.

  Because the file is copied, a plugin should not read data files next to itself through
  `__file__` (the copies have no such neighbours): put the data in the file, or package the
  plugin and load it through an entry point.
* **An installed package** declaring entry points, for example in its `pyproject.toml`:

  ```toml
  [project.entry-points."mechanet.families"]
  her = "mechanet_her:register"
  ```

`mechanet plugins` lists everything that is loaded (inside a project, including the
project's plugin files) and reports plugins that failed to load. A broken third-party plugin
never breaks the core.

## A complete example: hydrogen evolution

`examples/plugins/her.py` adds HER, 2 (H⁺ + e⁻) → H2, on metal top sites. It is about sixty
lines; `tests/test_plugins.py` runs it on Pt(111) through generation, synthetic VASP runs,
thermodynamics, pathway ranking and the Pourbaix analysis.

```python
from mechanet.adsorbates import Adsorbate
from mechanet.plugins import Family
from mechanet.reactions import Reaction, Species

# A neutral molecule: mu = E0 + G_corr of the H2 entry of the references file.
H2 = Species("H2", (("H", 2),), 0, "gas", references=("H2",))


class HER(Reaction):
    name = "her"
    title = "Hydrogen evolution reaction"
    net = {"H+": -2, "H2": 1}            # 2 H+ + 2 e- -> H2 (cathodic)
    release = "H2"                       # the product whose release closes a cycle
    balance_species = {"H": "H+"}        # H left over in a step goes to/comes from H+
    formation_species = ("H+",)          # forms terminations from the reference (Pourbaix)
    potential_scale = "RHE"
    default_potentials = (0.0, -0.1, -0.2)
    required_references = ("H2",)
    elementary = ({"H+": -1}, {"H+": -1, "H2": 1}, {"H2": 1})   # Volmer, Heyrovsky, Tafel


HYDRIDE = Adsorbate("H", {"H": 1}, "H", "atom", metal_distance_A=1.6, description="Adsorbed hydrogen")


def volmer_heyrovsky(ctx):
    family = "her_volmer_heyrovsky"
    for site in ctx.sites:
        if ctx.native[site] != "empty":
            continue
        bare = ctx.node({site: "empty"}, family, site)
        hydride = ctx.node({site: "H"}, family, site)
        volmer = ctx.step(bare, hydride, family, "Volmer: * + H+ + e- -> *H", reaction="her")
        heyrovsky = ctx.step(hydride, bare, family, "Heyrovsky: *H + H+ + e- -> * + H2",
                             given={"H2": 1}, reaction="her")
        ctx.cycle(f"{family}_{site}", [volmer, heyrovsky], family, "Volmer-Heyrovsky on a top site",
                  reaction="her")


def register(registry):
    registry.add_species(H2)
    registry.add_reaction(HER)
    registry.add_adsorbate(HYDRIDE)
    registry.add_family(Family("her_volmer_heyrovsky", volmer_heyrovsky, "her",
                               "Volmer-Heyrovsky HER on bare top sites", ["norskov2004"], core=True, order=300))
```

`mechanet init Pt_111.vasp pt --plugin examples/plugins/her.py --reaction her` writes a
Pt(111) project with `reaction: her`, the core HER families, the HER potentials (0, −0.1,
−0.2 V vs RHE) and `plugins: [plugins/her.py]`; `mechanet generate` builds the cycles, and
the analysis treats the reaction as cathodic: limiting potentials are the highest potentials
at which every step runs downhill, and η_TD = U_eq − U_L.

## A second example: CO2 reduction to CO

`examples/plugins/co2rr.py` adds CO2 + 2 (H⁺ + e⁻) → CO + H2O through \*COOH on metal top
sites: two gas species (CO2, CO), the reaction, \*COOH and \*CO given atom by atom as
templates (see [Adsorbates](#adsorbates)), the carboxyl family and its literature record.
Nothing in mechanet itself is specific to it: the relaxed-structure checks follow C because an
adsorbate brings it. `tests/test_co2rr_plugin.py` runs it on Pt(111) through generation,
synthetic VASP runs, the checks of relaxed structures, thermodynamics, pathway ranking and the
Pourbaix analysis. The references file needs CO2 and CO (E0 + G_corr, with whatever gas-phase
correction you apply to them).

## A third example: nitrogen reduction to ammonia

`examples/plugins/nrr.py` adds N2 + 6 (H⁺ + e⁻) → 2 NH3 by the distal associative route on
metal top sites: \*N2, \*NNH, \*NNH2, \*N, \*NH, \*NH2 and \*NH3. Three of the adsorbates use
the fixed shapes with N as the proximal atom (`atom`, `XH`, `water`), four are templates. Two
things differ from the other examples, and neither needs anything from the core:

* **The product leaves twice.** A turnover gives two NH3, so the turnover marker
  (`Reaction.release`) is N2, which is taken up once.
* **Steps without electron transfer.** N2 adsorption and NH3 release exchange no electron, so
  no potential drives them. If one of them is uphill, the cycle has no limiting potential and
  `thermo` and `pathways --rank` say so instead of reporting a number.

`tests/test_nrr_plugin.py` runs it on Pt(111) end to end, and checks that the seven
intermediates are read back from their structures and that an N2 or NH3 that left, an N–H bond
that broke and an H on the wrong N are recognised.

## Reactions

A `Reaction` declares data only:

* `net`: the reservoir vector of one turnover; electrons follow from charge balance
  (positive when released, so anodic reactions have positive `electrons`);
* `release`: the *turnover marker*, a species of `net` with coefficient +1 (a product given
  off once per turnover: O2 for OER, Cl2 for CER, H2 for HER) or −1 (a reactant taken up
  once per turnover: O2 for ORR, N2 for NRR). Cycles are rotated to end with a released
  marker or to start with a consumed one; pathway discovery searches from the steps that
  move the marker in the reaction's direction and never uses a step that moves it back;
  the resting-state and Pourbaix analyses treat a node with a downhill marker step as
  transient. A species that appears twice per turnover (H2O in ORR, NH3 in NRR) cannot be
  the marker;
* `balance_species`: which species absorbs each element when a family gives only some
  coefficients of a step (`ctx.step(..., given={"H2": 1})`);
* `formation_species`: the species that form any termination from the reference surface
  (used for resting states and Pourbaix diagrams);
* `elementary`: the reservoir vectors that count as one elementary step for
  `mechanet pathways` (it proposes steps between intermediates only with these vectors);
* `walden` (optional): reservoir vectors of Walden steps, in which the product leaves a site as
  a reactant attaches to it, so the bare site is never formed (OER: `{"H2O": -1, "O2": 1,
  "H+": 1}`). Each must give off the turnover marker and take up another species; `pathways`
  proposes it only on one site that holds the product whole (docs/science.md §2);
* `potential_scale` (`RHE` or `SHE`), `default_potentials`, `required_references`;
* `coverage_states`: the states the `coverage` family gives each site kind when project.yaml
  lists none, for example `{"open_metal": ("empty", "H")}` for HER; without it the oxide
  states (O, OH, H2O and the vacancy) are used.

`registry.add_reaction` checks the declaration (registered species, element balance of
`net`, at least one electron, a valid marker) and says what to change. `Reaction.balance` and
`Reaction.formation_vector` solve the element balance exactly, for any species in any order;
when the species admit several solutions (H2O, H⁺ and OH⁻ are related by water
autoionization) the species listed first are used. Every step a family writes is checked for
element and charge balance.

For example, associative ORR on the RHE scale:

```python
class ORR(Reaction):
    name = "orr"
    title = "Oxygen reduction reaction"
    net = {"O2": -1, "H+": -4, "H2O": 2}      # O2 + 4 (H+ + e-) -> 2 H2O (cathodic)
    release = "O2"                            # consumed once per turnover: cycles start with it
    balance_species = {"H": "H+"}
    formation_species = ("H2O", "H+")
    potential_scale = "RHE"
    default_potentials = (0.9, 0.8, 0.7)
    required_references = ("H2", "H2O")
    elementary = ({"O2": -1, "H+": -1}, {"H+": -1, "H2O": 1}, {"H+": -1}, {"O2": -1})
```

The pathway sweep is exact when a closed walk over the reaction's species that moves the
marker once is necessarily one turnover (`Reaction.closed_walks_are_turnovers`), which holds
for OER, CER, HER, ORR and NRR. If you add species that allow two independent balanced
reactions (H2O2 next to H2O in ORR), enumerated cycles are still checked one by one, but the
sweep is skipped.

## Species

Built-in species: H2O, H⁺, O2, Cl⁻, Cl2, with the CHE chemical potentials described in
docs/science.md. A plugin species gives its composition, charge, phase and the reference
molecules its chemical potential needs. Without a `chemical_potential` rule it is treated as
a neutral molecule, μ = E0 + G_corr of the molecule of the same name in the references file
plus kT ln(activity) (`--activity NAME=VALUE`; registered species are the valid names). A
charged species needs a rule:

```python
from mechanet.reactions import Species

def mu_hydroxide(reservoirs, conditions):          # eV at U_SHE = 0
    ...

OH_MINUS = Species("OH-", (("O", 1), ("H", 1)), -1, "aqueous", references=("H2", "H2O"),
                   chemical_potential=mu_hydroxide)
```

## Adsorbates

An `Adsorbate` is data: its composition, the proximal element bound to the site and an
internal geometry. Small adsorbates use one of six fixed shapes (`none`, `atom`, `XH`,
`water`, `XY`, `XYH`) with bond lengths, a tilt list and the metal distance; they are
recognised in relaxed structures by the signature `(proximal, distal, H on proximal, H on
distal)`. Anything larger is a **template**: the atoms beyond the proximal one, each with its
element, the atom it is bonded to and a position. The builder places every kind with the same
scored, clash-checked orientation search as the built-ins. Two adsorbates may not share a
signature.

A carboxyl bound through C, from `examples/plugins/co2rr.py`:

```python
CARBOXYL = Adsorbate(
    "COOH",
    {"C": 1, "O": 2, "H": 1},
    "C",                                   # proximal element, bonded to the site
    "template",
    metal_distance_A=2.00,                 # M-C on an open metal site
    atoms=(
        # label, element, bonded to ("" = the proximal atom), (x, y, z) in A
        ("o1", "O", "", (1.0822, 0.0, 0.5633)),
        ("o2", "O", "", (-1.2063, 0.0, 0.6279)),
        ("h", "H", "o2", (-1.0109, 0.0, 1.5882)),
    ),
    template_tilts_deg=(0.0, 25.0),        # tilts from the site's outward direction
    template_roll_step_deg=30.0,           # rotations about the molecule's own axis
)
```

The proximal atom sits at the origin and +z points away from the surface along its bond to
the site. The atoms get the IDs `o1_<site>`, `o2_<site>`, `h_<site>`, and the bonds between
them are the intended bonds of the starting structure. In a relaxed structure the
adsorbate is recognised by the canonical tree of its atoms rooted at the site atom
(`Adsorbate.graph_key`: `C(O,O(H))` here), so an H that moved to the other O still reads as
\*COOH, while an OH that split off makes the structure need review.

The relaxed-structure checks follow O, Cl and the H bound to O. Elements that registered
adsorbates bring beyond O, H and Cl (C and N, say) join them automatically: they may hold H,
their bonds to each other, to O and to the metals are watched, with bond lengths from covalent
radii (lower hysteresis edge, initial bond and upper edge at 1.15, 1.25 and 1.35 times the sum
of the radii), and released molecules that contain them are recognised. For oxygen and
chlorine chemistry nothing changes.

**Hydrogen on a metal.** When an adsorbate is bound through H (`proximal="H"`, the \*H of the
HER example), metal–H contacts are watched like metal–O ones (bonded below 2.14 Å for a typical
metal; override with `topology.hysteresis_A: {"M-H": [lower, upper]}`). An H that left the
surface, alone or as H2, is reported as released (no bound minimum from that start); one that
moved to another metal atom, or sits between two, waits for review, as an O that changes its
metal does. The H–H bond is watched too: two H closer than 1.00 Å are molecular hydrogen, so H2
lying on the surface is not taken for two \*H and waits for review (between 1.00 and 1.30 Å the
pair is listed as ambiguous; `"H-H": [lower, upper]` overrides the limits), and an H2 that took
the H of another ligand along is not the intermediate's own adsorbate leaving. Without such an
adsorbate (OER, CER) neither metal–H nor H–H distances are looked at, as before.

**Bonds between two sites.** Two neighbouring sites can bond through their proximal atoms: the
O–O pairs of the OER coupling families, two \*CO through C. A family writes such an
intermediate with `ctx.node({a: "CO", b: "CO"}, family, label, coupling=(a, b))`, and the
builder places the two proximal atoms at their bond length (the adsorbate's `coupled_bond_A`;
by default 1.45 Å for two O and the sum of the covalent radii otherwise). It does so for any
two occupied sites, as it always did. An adsorbate that sets `couples=True` gets the rest:
relaxed structures are read as a coupled pair or as two separate adsorbates (and assigned to
the intermediate they became), the `enumerated` family writes the pair by itself, and the
pathway search proposes the step as "C-C coupling" when the reaction's `elementary` vectors
include `{}`, a step that exchanges nothing with the reservoirs (without it no step leads to
the pair). Without the flag, a pair that stays bonded waits for review. The CO example leaves
the flag off, since its reaction has no such step; `tests/test_coupling.py` adds the flag, the
step and tilts wide enough for the pair (two \*CO need about 30° to keep their oxygens apart).

**One binding atom per adsorbate.** An adsorbate is bound through its proximal atom. A relaxed
structure in which a second atom of it also binds a metal (a carboxyl or a formate lying down
on two metals) is neither accepted in silence nor called released: it waits for review, with
the new contacts named. Rings cannot be written as templates.

## Mechanism families

`build(ctx)` receives the build context:

| Method | Does |
|---|---|
| `ctx.sites`, `ctx.native` | site records (kind, native state, metal neighbours, position) and native occupancies |
| `ctx.node(occupancy, family, site, coupling=None)` | an intermediate (deduplicated; translation images become `computed_by` twins) |
| `ctx.step(start, end, family, note, given=None, reaction=None)` | a balanced step; the reaction completes the vector |
| `ctx.cycle(label, edge_ids, family, evidence, reaction=None)` | a cycle; refused unless it is one turnover |
| `ctx.site_class(site)`, `ctx.nearby_pairs()`, `ctx.shared_metals(a, b)` | site classes and pairs within `scope.pair_max_distance_A` |
| `ctx.scope`, `ctx.reaction`, `ctx.registry` | the project scope (family settings live there), the campaign reaction, the registry |
| `ctx.write_motif(id, structure, meta)` | a separate search seed (e.g. a transition-state guess), never a normal minimum |

`Family(name, build, reaction, description, sources, core=False, order=100, scope_key=None,
scope_fields=())`: `reaction` names the reaction whose cycles the family writes (None for
node-only families such as `coverage`), `sources` are literature keys (add records with
`registry.add_sources`), `core` puts the family in the default scope of its reaction, and
`order` fixes the build order so node and state IDs are reproducible. A family configured
from project.yaml names its block with `scope_key` (it reads `ctx.scope[scope_key]`; the
built-in `coverage` and `enumerated` families use `scope.coverage` and `scope.enumerate`)
and lists the keys the block accepts in `scope_fields`; project validation then accepts the
block and refuses typos in it, as for every other section.

## Commands

A command plugin adds a subcommand to `mechanet`:

```python
def register_command(subparsers, registry):
    p = subparsers.add_parser("bader", help="Bader charges of every relaxed state")
    p.add_argument("campaign", nargs="?")
    p.set_defaults(handler=run)

def run(args, registry):
    ...
    return {"states": 42, "file": "analysis/bader.csv"}   # printed readably, or as JSON with --json
```

`mechanet pathways`, `mechanet pourbaix` and `mechanet prescreen` are registered this way.
Every subcommand gets `--json` and `--traceback` automatically (a plugin that defines them
itself keeps its own). The handler returns a result (printed readably, or as JSON), or
`(result, text)`, or `(result, text, exit_code)`; a `ValueError` is printed as `ERROR: ...`
with exit code 2.

## Calculators

A calculator relaxes structures inside Python for
[`mechanet prescreen`](prescreening.md), usually through a machine-learned potential. A plugin
registers a factory, which is called with the `--option key=value` pairs and returns an ASE
calculator:

```python
def make_uma(model="uma-s-1p1", task="oc20", device="cpu"):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    return FAIRChemCalculator(pretrained_mlip.get_predict_unit(model, device=device), task_name=task)


def register(registry):
    registry.add_calculator("uma", make_uma, "UMA through fairchem (task oc20: slabs with adsorbates)")
```

A package offers it through an entry point (the entry-point name is the calculator's name):

```toml
[project.entry-points."mechanet.calculators"]
uma = "my_package.calculators:make_uma"
```

`emt` and `mace-mp` are registered this way. Import the potential's package inside the
factory, so that mechanet loads without it. The factory may also return an object of its own
with `describe()` (a dict: what was used, stored with every result) and
`relax(structure, fmax, steps)` returning a `mechanet.calculators.Relaxation`, for an engine
that is not an ASE calculator; it must keep fixed the coordinates that the structure's
`mobile` flags fix.
