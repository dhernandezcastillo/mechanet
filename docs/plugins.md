# Extending mechanet: reactions, species, adsorbates, mechanisms and commands

Everything chemistry-specific in mechanet is a plugin, including the built-in OER and CER
chemistry. A plugin can add:

| Piece | Class | Registry call | Entry-point group |
|---|---|---|---|
| Reservoir species (H2, N2, NH3, ...) | `mechanet.reactions.Species` | `registry.add_species(species)` | (inside a `register` function) |
| Reaction | subclass of `mechanet.reactions.Reaction` | `registry.add_reaction(cls)` | `mechanet.reactions` (the class) |
| Adsorbate | `mechanet.adsorbates.Adsorbate` | `registry.add_adsorbate(ads)` | `mechanet.adsorbates` (`register`) |
| Mechanism family | `mechanet.plugins.Family` | `registry.add_family(family)` | `mechanet.families` (`register`) |
| CLI command | `register_command(subparsers, registry)` | `registry.add_command(name, fn)` | `mechanet.commands` |

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
* `potential_scale` (`RHE` or `SHE`), `default_potentials`, `required_references`.

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

An `Adsorbate` is data: its composition, the proximal element bound to the site, an
optional distal heavy atom, and an internal geometry type (`none`, `atom`, `XH`, `water`,
`XY`, `XYH`) with bond lengths, tilt list and metal distance. The builder places it with the
same scored, clash-checked orientation search as the built-ins, and the analysis recognises
it again in relaxed structures from its signature `(proximal, distal, H on proximal, H on
distal)`. Two adsorbates may not share a signature.

The relaxed-structure classifier tracks O, Cl and H bound to O. An adsorbate bound through
another element (such as \*H on a metal) is followed by its connectivity: while it keeps its
bonds the state keeps its node; if its connectivity changes, the structure is listed for
review instead of being reassigned automatically.

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

`mechanet pathways` and `mechanet pourbaix` are registered this way. Every subcommand gets
`--json` and `--traceback` automatically (a plugin that defines them itself keeps its own).
The handler returns a result (printed readably, or as JSON), or `(result, text)`, or
`(result, text, exit_code)`; a `ValueError` is printed as `ERROR: ...` with exit code 2.
