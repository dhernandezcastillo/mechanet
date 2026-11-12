"""Coverage family: the surface states a surface Pourbaix diagram compares.

It enumerates occupancy combinations of the reactive sites, up to ``max_changed_sites``
sites away from the native surface (default 2; ``null`` for every combination), and builds
each as a node. In a campaign built on a surface (``mechanet generate --coverage --on ID``)
the combinations are counted from that surface instead: a look at its neighbourhood. No
edges or cycles are written: ``mechanet pourbaix`` compares the formation free energies of
these nodes as a function of potential and pH.

Configured in project.yaml under ``scope.coverage``::

    coverage:
      sites: [m0038, m0046, a0016, a0019]   # default: all reactive sites
      max_changed_sites: 2
      states:                                # per site kind or site ID; default: the reaction's
        open_metal: [empty, OH, O, H2O]
        bridge: [O, OH]

The defaults come from the campaign reaction (``Reaction.coverage_states``): O, OH, H2O and
the vacancy for OER, plus Cl and OCl for CER; a plugin reaction declares its own (H on metal
top sites for HER, say) and falls back to the oxygen states otherwise.
"""

from __future__ import annotations

from itertools import combinations, product

from ..plugins import Family
from ..reactions import BUILTIN_REACTIONS, OXYGEN_COVERAGE_STATES

# The tables of 0.1.4, kept for code that reads them. The states now come from the reaction
# (Reaction.coverage_states); changing these tables changes nothing.
DEFAULT_STATES = {kind: list(states) for kind, states in OXYGEN_COVERAGE_STATES.items()}
CER_EXTRA = {"open_metal": ["Cl", "OCl"], "terminal": ["OCl"], "bridge": []}


def default_states(kind, reaction):
    """Coverage states of a site kind for ``reaction`` (a Reaction, or the name of a built-in one)."""
    if isinstance(reaction, str):
        reaction = BUILTIN_REACTIONS.get(reaction)
    declared = getattr(reaction, "coverage_states", None) or OXYGEN_COVERAGE_STATES
    return list(declared.get(kind, ("O", "OH")))


def _options(ctx):
    """(site IDs, states each may take besides the one it has, most sites changed at once)."""
    cfg = ctx.scope.get("coverage") or {}
    site_ids = list(cfg.get("sites") or ctx.sites)
    unknown = sorted(set(site_ids) - set(ctx.sites))
    if unknown:
        raise ValueError(f"scope.coverage.sites names unknown sites {unknown}")
    states_cfg = cfg.get("states") or {}
    options = {}
    for aid in site_ids:
        kind = ctx.sites[aid]["kind"]
        states = list(states_cfg.get(aid) or states_cfg.get(kind) or default_states(kind, ctx.reaction))
        for status in states:
            ctx.adsorbate(status)
        options[aid] = [s for s in states if s != ctx.surface_state(aid)]
    max_changed = cfg.get("max_changed_sites", 2)
    limit = len(site_ids) if max_changed is None else min(int(max_changed), len(site_ids))
    return site_ids, options, limit


def patterns(ctx):
    """Every surface the family builds, as (changed sites -> state, label), in building order."""
    site_ids, options, limit = _options(ctx)
    for k in range(1, limit + 1):
        for chosen in combinations(site_ids, k):
            for statuses in product(*(options[aid] for aid in chosen)):
                yield dict(zip(chosen, statuses)), "+".join(chosen)


def count(ctx):
    """How many surfaces the coverage scope gives, without building them.

    ``surfaces`` counts every combination and the slab as given; ``distinct`` leaves out those
    that a lattice translation of the slab maps onto another one (they are the same calculation).
    ``states`` lists, per site, the states it is probed in.
    """
    site_ids, options, limit = _options(ctx)
    total, distinct = 0, set()
    for occupancy, _ in patterns(ctx):
        total += 1
        distinct.add(ctx.canonical_key(ctx.on_surface(occupancy)))
    return {
        "sites": len(site_ids),
        "max_changed_sites": limit,
        "surfaces": total + 1,
        "distinct": len(distinct) + 1,
        "states": {aid: [ctx.surface_state(aid), *options[aid]] for aid in site_ids},
    }


def coverage(ctx):
    for occupancy, label in patterns(ctx):
        ctx.node(occupancy, "coverage", label)


FAMILY = Family(
    "coverage",
    coverage,
    None,
    "Surface coverage states for surface Pourbaix diagrams (nodes only, no cycles)",
    ["hansen2008", "rossmeisl2007"],
    core=False,
    order=200,
    scope_key="coverage",
    scope_fields=("sites", "max_changed_sites", "states"),
)


def register(registry):
    registry.add_family(FAMILY)
