"""Built-in mechanism families.

A family is a function ``build(ctx)`` that writes nodes, edges and cycles through the
builder context (:class:`mechanet.builder.Builder`). The context offers:

* ``ctx.sites`` (site ID -> record with ``kind``, ``native``, ``metal_neighbors``),
  ``ctx.site_class(id)`` ('terminal' or 'bridge'), ``ctx.nearby_pairs()``,
  ``ctx.shared_metals(a, b)``;
* ``ctx.node(occupancy, family, site_label, coupling=None)`` -> node ID;
* ``ctx.step(start, end, family, note, given={...}, reaction=None)`` -> edge ID, with the
  reaction's balance species completing ``given``; ``ctx.edge(start, end, water, oxygen,
  family, note)`` is the OER shorthand;
* ``ctx.cycle(label, edge_ids, family, evidence_scope, reaction=None)`` and
  ``ctx.oer_cycle(label, links, family, evidence_scope)``.

Families are registered with :class:`mechanet.plugins.Family` metadata (reaction,
literature sources, default scope, build order).
"""
