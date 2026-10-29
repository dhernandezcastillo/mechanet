"""Coverage family: the surface states a surface Pourbaix diagram compares.

It enumerates occupancy combinations of the reactive sites, up to ``max_changed_sites``
sites away from the native surface (default 2; ``null`` for every combination), and builds
each as a node. No edges or cycles are written: ``mechanet pourbaix`` compares the
formation free energies of these nodes as a function of potential and pH.

Configured in project.yaml under ``scope.coverage``::

    coverage:
      sites: [m0038, m0046, a0016, a0019]   # default: all reactive sites
      max_changed_sites: 2
      states:                                # per site kind; defaults below
        open_metal: [empty, OH, O, H2O]
        bridge: [O, OH]
"""

from __future__ import annotations

from itertools import combinations, product

from ..plugins import Family

DEFAULT_STATES = {
    "open_metal": ["empty", "OH", "O", "H2O"],
    "terminal": ["empty", "OH", "O", "H2O"],
    "bridge": ["empty", "O", "OH"],
    "mu3": ["O", "OH"],
    "framework": ["O", "OH"],
}
CER_EXTRA = {"open_metal": ["Cl", "OCl"], "terminal": ["OCl"], "bridge": []}


def default_states(kind, reaction):
    states = list(DEFAULT_STATES.get(kind, ["O", "OH"]))
    if reaction == "cer":
        states += [s for s in CER_EXTRA.get(kind, []) if s not in states]
    return states


def coverage(ctx):
    cfg = ctx.scope.get("coverage") or {}
    site_ids = list(cfg.get("sites") or ctx.sites)
    unknown = sorted(set(site_ids) - set(ctx.sites))
    if unknown:
        raise ValueError(f"scope.coverage.sites names unknown sites {unknown}")
    states_cfg = cfg.get("states") or {}
    options = {}
    for aid in site_ids:
        kind = ctx.sites[aid]["kind"]
        states = list(states_cfg.get(aid) or states_cfg.get(kind) or default_states(kind, ctx.reaction.name))
        for status in states:
            ctx.adsorbate(status)
        options[aid] = [s for s in states if s != ctx.native[aid]]
    max_changed = cfg.get("max_changed_sites", 2)
    limit = len(site_ids) if max_changed is None else min(int(max_changed), len(site_ids))
    for k in range(1, limit + 1):
        for chosen in combinations(site_ids, k):
            for statuses in product(*(options[aid] for aid in chosen)):
                ctx.node(dict(zip(chosen, statuses)), "coverage", "+".join(chosen))


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
