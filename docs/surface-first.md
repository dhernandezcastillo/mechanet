# Surface first: coverage, then the mechanism

A mechanism runs on a surface, and which surface a catalyst shows depends on the potential.
The slab file you start from carries one termination; under reaction conditions the sites
may hold other adsorbates. The usual procedure in computational electrocatalysis therefore
has two stages: a surface Pourbaix diagram first, built from many coverages of the sites, and
the mechanism second, on the surfaces the diagram selects. Usama, Razzaq, Hättig, Steinmann
and Exner (*Nat. Commun.* **16**, 6137 (2025)) do this for IrO2(110) and carry four surfaces
of comparable stability into the mechanistic analysis, not one.

This page follows that order.

## 1. Probe adsorbates on the sites

List in `project.yaml` the states to try, per kind of site or per single site:

```yaml
scope:
  families: [aem, bifunctional]        # the mechanism, for later
  coverage:
    states:
      terminal: [O, OH, H2O, OOH]      # every terminal site is tried in these states
      bridge: [O, OH]
      a0065: [O, OH]                   # one site can have its own list
    max_changed_sites: null            # null: every combination; 2 (the default): up to two sites at once
```

Any adsorbate the campaign knows can be listed on any site: `empty` (a vacancy, or a bare
metal), O, OH, H2O, OOH and OO are built in, Cl and OCl come with the chlorine evolution
reaction, and a plugin adds others ([plugins](plugins.md)). A site keeps its ID from
`mechanet init`; `sites:` restricts the probe to some of them.

Count before you build:

```bash
mechanet generate --coverage --count
```

```text
coverage: 256 surfaces on 8 sites, 136 distinct after symmetry (bridge: O, OH; terminal: O, OH; every combination)
  nothing built (--count)
```

The first number is every combination, the slab as given included. The second leaves out
surfaces that a lattice translation of the slab maps onto another one: they are the same
calculation, so only one of them is built and run. The example is the edge of the NiFe
layered double hydroxide with O or OH on its four terminal and four bridging sites.

Then build the campaign:

```bash
mechanet generate --coverage
```

It goes to `<project>/coverage`, next to the mechanism campaign that `mechanet generate`
writes to `<project>/campaign`, and is run, followed and checked like any campaign
(`plan`, `submit`, `status`, `review`). It holds surfaces only: no steps and no cycles.

Three things keep the count affordable:

* **Fewer states.** Four states on four sites and two on four others are 4,096 surfaces.
* **`max_changed_sites`.** With 2, only surfaces that differ from the slab at one or two sites
  are built: a look at the neighbourhood of the termination you have.
* **Two passes.** O and OH everywhere first; the other adsorbates in a second campaign, on
  the surfaces the first diagram brings out.

`generate --coverage` warns above 400 distinct surfaces.

A surface with many adsorbates that need an orientation (OH, OOH, water) is placed with a
shorter search than an intermediate of a mechanism: two placement orders instead of all of
them, which would be 40,320 for eight sites ([science §3](science.md#3-starting-structures)).
The build report says for how many surfaces this was done.

After the relaxations, `mechanet review` tells which structures still are the surface they
were built as. This matters more here than anywhere else: with many adsorbates next to each
other, protons move, water forms and O–O bonds close during relaxation, and a structure that
became another surface must not be counted under the name of the one it started as.
`mechanet pourbaix` uses each structure for the surface it has become
([checking results](checking-results.md)).
