"""Pathway discovery on the reaction network with networkx.

The literature families write curated cycles. This module treats the network as a directed
graph and looks for *every* balanced catalytic cycle it contains, including mixed routes
that no single family writes (for example adsorbate evolution up to *O followed by a
bifunctional proton transfer):

1. **Elementary steps.** Between any two intermediates of the network, a step is proposed
   when it is local (at most two sites change, and two only if they are a site pair), when
   at most one site changes its heavy atoms (a second site may only gain or lose one proton,
   or both sites form/break an O-O coupling or give off the product together), and when its
   reservoir vector is one of the reaction's elementary vectors (one PCET, water attack or
   adsorption, an internal transfer, an O2/Cl2/H2 release, a Cl- adsorption; see
   ``Reaction.elementary``). Only steps in the reaction's direction (and chemical steps) are
   proposed unless ``allow_reverse``.
2. **Cycles.** Every simple cycle of at most ``max_length`` steps whose vectors sum to one
   turnover (OER: 2 H2O -> O2 + 4 H+ + 4 e-) is enumerated. A turnover moves the reaction's
   marker (``Reaction.release``: O2 released in OER, O2 consumed in ORR) exactly once, so the
   search runs from each marker step back to its start; every cycle found is checked to sum
   to exactly one turnover.
3. **Ranking.** With energies available, every cycle is evaluated at each potential (Gmax(U),
   thermodynamic overpotential, cyclic span) and the best routes are reported, curated or not.

Discovery reuses the computed intermediates and costs no new DFT. The ``enumerated``
family goes one step further at build time: it creates every occupancy of the chosen sites
(in the spirit of Govind Rajan and Carter, J. Phys. Chem. C 2020) before the same search.
"""

from __future__ import annotations

import json
from itertools import combinations, product
from pathlib import Path

import networkx as nx
import numpy as np

from .fileio import dump_json, read_json, short_hash
from .plugins import Family, campaign_registry
from .reactions import add as add_vectors, clean, electrons as vector_electrons, element_change
from .thermo import DOWNHILL_TOLERANCE_EV, release_last_rotation

EVIDENCE = (
    "Graph-enumerated balanced cycle through intermediates of this network; it combines proposed elementary "
    "steps and is not taken from the literature"
)


# ---------------------------------------------------------------------------
# Elementary steps
# ---------------------------------------------------------------------------


class StepRules:
    """Locality and stoichiometry rules for proposing elementary steps between two nodes."""

    def __init__(self, native, adsorbates, reaction, pair_ok, allow_reverse=False):
        self.native = dict(native)
        self.adsorbates = adsorbates
        self.reaction = reaction
        self.pair_ok = pair_ok
        vectors = [clean(v) for v in reaction.elementary]
        if allow_reverse:
            vectors += [clean({k: -x for k, x in v.items()}) for v in vectors]
        unique = []
        for v in vectors:
            if v not in unique:
                unique.append(v)
        self.vectors = unique
        self.release = reaction.release

    def full(self, node):
        occupancy = dict(self.native)
        occupancy.update(node.get("occupancy") or {})
        return occupancy

    def heavy(self, status):
        comp = self.adsorbates[status].composition
        return sum(n for element, n in comp.items() if element != "H")

    def hydrogens(self, status):
        return self.adsorbates[status].composition.get("H", 0)

    def composition_change(self, a, b):
        delta = {}
        for site in self.native:
            for element, n in self.adsorbates[b[site]].composition.items():
                delta[element] = delta.get(element, 0) + n
            for element, n in self.adsorbates[a[site]].composition.items():
                delta[element] = delta.get(element, 0) - n
        return {k: v for k, v in delta.items() if v}

    def step(self, node_a, node_b):
        """(vector, rule) of an elementary step node_a -> node_b, or None."""
        a, b = self.full(node_a), self.full(node_b)
        couple_a, couple_b = set(node_a.get("coupling") or ()), set(node_b.get("coupling") or ())
        changed = [s for s in self.native if a[s] != b[s]]
        involved = set(changed) | (couple_a ^ couple_b)
        if not involved or len(involved) > 2:
            return None
        if len(involved) == 2 and not self.pair_ok(*sorted(involved)):
            return None
        if couple_a != couple_b:
            sites = sorted(couple_a | couple_b)
            if couple_a and couple_b:
                return None
            if not changed and all(a[s] == "O" for s in sites):
                rule = "O-O coupling" if couple_b else "O-O decoupling"
            elif couple_a and all(a[s] == "O" and b[s] == "empty" for s in sites):
                rule = "release from a coupled pair"
            else:
                return None
        elif len(changed) == 2:
            heavy_change = [s for s in changed if self.heavy(a[s]) != self.heavy(b[s])]
            if len(heavy_change) == 2:
                if not all(self.heavy(a[s]) == 1 and b[s] == "empty" for s in changed):
                    return None
                rule = "pair release"
            else:
                partner = [s for s in changed if s not in heavy_change]
                if heavy_change:
                    if abs(self.hydrogens(a[partner[0]]) - self.hydrogens(b[partner[0]])) != 1:
                        return None
                    rule = "step with proton transfer to/from a neighbour"
                else:
                    if any(abs(self.hydrogens(a[s]) - self.hydrogens(b[s])) != 1 for s in changed):
                        return None
                    rule = "proton transfer between neighbours"
        else:
            rule = "single-site step"
        delta = self.composition_change(a, b)
        for vector in self.vectors:
            implied = element_change(vector)
            if implied != delta:
                continue
            # a step that gives off (or takes up) the turnover marker is a release step whatever
            # the sites do: two *H -> H2 (HER Tafel) changes no heavy atom at all
            marker = vector.get(self.release, 0)
            if rule in ("pair release", "release from a coupled pair") and not marker:
                continue
            if rule.startswith("proton transfer between") and marker:
                return vector, "pair release" if self.reaction.advances(vector) else "pair uptake"
            if rule.startswith(("O-O", "proton transfer between")) and vector:
                continue
            return vector, rule
        return None


def _pair_checker(sites, cell, cutoff):
    positions = {s["id"]: np.asarray(s["position_A"], dtype=float) for s in sites}
    inv = np.linalg.inv(cell)

    def ok(a, b):
        d = positions[b] - positions[a]
        frac = d @ inv
        frac -= np.rint(frac)
        best = min(np.linalg.norm((frac + np.array(o)) @ cell) for o in product((-1, 0, 1), repeat=3))
        return best <= cutoff + 1e-9

    return ok


def propose_steps(nodes, rules, existing=None):
    """Elementary steps between every ordered pair of nodes; returns [(from, to, vector, rule)]."""
    existing = existing or set()
    out = []
    node_list = [n for n in nodes if not n.get("computed_by")] + [n for n in nodes if n.get("computed_by")]
    for a in node_list:
        for b in node_list:
            if a["id"] == b["id"]:
                continue
            found = rules.step(a, b)
            if found is None:
                continue
            vector, rule = found
            if (a["id"], b["id"], json.dumps(vector, sort_keys=True)) in existing:
                continue
            out.append((a["id"], b["id"], vector, rule))
    return out


def product_release_steps(campaign, network, registry, reactions):
    """Steps that give a reaction's product off (its turnover marker, when produced).

    The network's own edges, plus the elementary steps :class:`StepRules` proposes between
    its nodes: a family may write a node that holds the product bound without writing its
    release (bound O2 next to bound water in the ``hydrated`` family). Proposals need the
    campaign's sites.json and surface_analysis.json; without ``campaign`` only edges count.
    Returns [{"edge_id", "from", "to", "vector", "reaction", "species", "proposed"}].
    """
    from .builder import edge_id

    producing = [rx for rx in reactions if rx.marker_sign > 0]
    out, seen = [], set()
    for edge in network.get("edges", []):
        vector = _edge_vector(edge)
        rx = next((r for r in producing if r.advances(vector)), None)
        seen.add((edge["from"], edge["to"], json.dumps(vector, sort_keys=True)))
        if rx is not None:
            out.append(
                {
                    "edge_id": edge["id"],
                    "from": edge["from"],
                    "to": edge["to"],
                    "vector": vector,
                    "reaction": rx.name,
                    "species": rx.release,
                    "proposed": False,
                }
            )
    if campaign is None or not producing:
        return out
    campaign = Path(campaign)
    try:
        sites = read_json(campaign / "sites.json")["sites"]
        analysis = read_json(campaign / "surface_analysis.json")
        config = read_json(campaign / "config.json")
    except (OSError, KeyError, ValueError):
        return out
    native = {s["id"]: s["native"] for s in sites}
    cell = np.asarray(analysis["cell_A"], dtype=float)
    pair_ok = _pair_checker(sites, cell, float(config.get("scope", {}).get("pair_max_distance_A", 3.3)))
    for rx in producing:
        rules = StepRules(native, registry.adsorbates, rx, pair_ok)
        for start, end, vector, _ in propose_steps(network["nodes"], rules, seen):
            if rx.advances(vector):
                out.append(
                    {
                        "edge_id": edge_id(start, end, vector),
                        "from": start,
                        "to": end,
                        "vector": vector,
                        "reaction": rx.name,
                        "species": rx.release,
                        "proposed": True,
                    }
                )
    return out


# ---------------------------------------------------------------------------
# Cycles
# ---------------------------------------------------------------------------


def build_graph(edges):
    """networkx MultiDiGraph of the network edges (key = edge ID)."""
    graph = nx.MultiDiGraph()
    for edge in edges:
        vector = clean(
            edge.get("reservoirs")
            if edge.get("reservoirs") is not None
            else {"H2O": -edge["water"], "H+": edge["pairs"], "O2": edge["oxygen"]}
        )
        graph.add_edge(
            edge["from"],
            edge["to"],
            key=edge["id"],
            reservoirs=vector,
            family=edge.get("family"),
            electrons=vector_electrons(vector),
        )
    return graph


def canonical(edge_ids, graph_edges, release):
    """Rotate an edge sequence into the turnover convention: a released product's step last,
    a consumed reactant's step first (see :func:`mechanet.thermo.release_last_rotation`)."""
    edge_ids = list(edge_ids)
    shift = release_last_rotation([graph_edges[e] for e in edge_ids], release)
    return tuple(edge_ids[shift:] + edge_ids[:shift])


def touched_sites(edge, nodes, native):
    """Sites whose occupancy (or coupling) differs between the two ends of an edge."""
    a, b = nodes[edge["from"]], nodes[edge["to"]]
    occ_a = {**native, **(a.get("occupancy") or {})}
    occ_b = {**native, **(b.get("occupancy") or {})}
    sites = {s for s in native if occ_a[s] != occ_b[s]}
    sites |= set(a.get("coupling") or ()) ^ set(b.get("coupling") or ())
    return frozenset(sites)


def enumerate_cycles(graph, reaction, max_length=8, max_cycles=200000, touched=None, max_sites=None):
    """Every simple cycle of at most ``max_length`` edges that is one turnover of ``reaction``.

    Only edges whose reservoir species belong to the reaction (``Reaction.species``) and that
    do not move its turnover marker backwards take part (an OER cycle never takes O2 up, an
    ORR cycle never gives it off). A turnover moves the marker exactly once, so a depth-first
    search runs from the end of each marker edge back to its start over the other edges, and
    every closed walk found is kept only if its vectors sum to one turnover. Branches are
    pruned when they move more electrons than one turnover (when every step runs in the
    reaction's direction) or touch more than ``max_sites`` sites (``touched``: edge ID ->
    sites). Returns (cycles, truncated) with cycles as tuples of edge IDs in the turnover
    convention (:func:`canonical`).
    """
    release = reaction.release
    target = abs(reaction.electrons)
    net = clean(reaction.net)
    allowed_species = set(reaction.species)
    edges, outgoing, releasing = {}, {}, []
    monotonic = True
    for u, v, key, data in graph.edges(keys=True, data=True):
        if set(data["reservoirs"]) - allowed_species or reaction.opposes(data["reservoirs"]):
            continue
        edges[key] = data
        if data["electrons"] * (1 if reaction.electrons >= 0 else -1) < 0:
            monotonic = False
        if reaction.advances(data["reservoirs"]):
            releasing.append((u, v, key))
        else:
            outgoing.setdefault(u, []).append((v, key))
    sign = 1 if reaction.electrons >= 0 else -1
    found, truncated = set(), False

    def sites_of(key):
        return touched.get(key, frozenset()) if touched is not None else frozenset()

    for u, v, key in releasing:
        first_sites = sites_of(key)
        if max_sites is not None and len(first_sites) > max_sites:
            continue
        first_e = sign * edges[key]["electrons"]
        if u == v:
            if clean(edges[key]["reservoirs"]) == net:
                found.add((key,))
            continue
        stack = [(v, [], frozenset((u, v)), first_sites, first_e)]
        while stack:
            node, path, visited, sites, n_e = stack.pop()
            if len(path) + 1 >= max_length:
                continue
            for nxt, k in outgoing.get(node, ()):
                e = n_e + sign * edges[k]["electrons"]
                if monotonic and e > target:
                    continue
                new_sites = sites | sites_of(k) if touched is not None else sites
                if max_sites is not None and len(new_sites) > max_sites:
                    continue
                if nxt == u:
                    sequence = path + [k, key]
                    if add_vectors(*(edges[x]["reservoirs"] for x in sequence)) == net:
                        found.add(canonical(sequence, edges, release))
                        if len(found) >= max_cycles:
                            return sorted(found), True
                    continue
                if nxt in visited:
                    continue
                stack.append((nxt, path + [k], visited | {nxt}, new_sites, e))
    return sorted(found), truncated


# ---------------------------------------------------------------------------
# Symmetry of discovered cycles
# ---------------------------------------------------------------------------


def _node_lookup(nodes, native):
    lookup = {}
    for node in nodes:
        changes = {k: v for k, v in (node.get("occupancy") or {}).items() if native.get(k) != v}
        lookup[
            json.dumps(
                {"occupancy": changes, "coupling": sorted(node["coupling"]) if node.get("coupling") else None},
                sort_keys=True,
            )
        ] = node["id"]
    return lookup


def _image(node, site_map, native, lookup):
    changes = {site_map[k]: v for k, v in (node.get("occupancy") or {}).items() if native.get(k) != v}
    coupling = sorted(site_map[x] for x in node["coupling"]) if node.get("coupling") else None
    return lookup.get(json.dumps({"occupancy": changes, "coupling": coupling}, sort_keys=True))


def symmetry_classes(cycles, network, native, edges_by_id):
    """Group cycles that a lattice translation of the slab maps onto each other."""
    from .builder import edge_id

    nodes = {n["id"]: n for n in network["nodes"]}
    lookup = _node_lookup(network["nodes"], native)
    translations = (network.get("symmetry") or {}).get("translations") or []
    index = {c: i for i, c in enumerate(cycles)}
    parent = list(range(len(cycles)))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for t in translations:
        site_map = t["site_map"]
        for c in cycles:
            mapped = []
            for e in c:
                edge = edges_by_id[e]
                start = _image(nodes[edge["from"]], site_map, native, lookup)
                end = _image(nodes[edge["to"]], site_map, native, lookup)
                if start is None or end is None:
                    mapped = None
                    break
                mapped.append(edge_id(start, end, clean(edge["reservoirs"])))
            if mapped is None:
                continue
            key = tuple(mapped)
            if key in index:
                a, b = find(index[c]), find(index[key])
                if a != b:
                    parent[max(a, b)] = min(a, b)
    return [find(i) for i in range(len(cycles))]


# ---------------------------------------------------------------------------
# Discovery on a campaign
# ---------------------------------------------------------------------------


def _campaign_inputs(campaign):
    campaign = Path(campaign)
    network = read_json(campaign / "network.json")
    sites = read_json(campaign / "sites.json")["sites"]
    analysis = read_json(campaign / "surface_analysis.json")
    config = read_json(campaign / "config.json")
    return network, sites, analysis, config


def discover(
    campaign,
    max_length=6,
    auto_steps=True,
    allow_reverse=False,
    max_cycles=200000,
    max_sites=2,
    registry=None,
    out=None,
):
    """Find all balanced cycles of a campaign's network (see the module docstring).

    Writes ``pathways/pathways.json`` (summary, every cycle, the proposed edges) and
    ``pathways/pathways.csv``; network.json is left alone (see :func:`adopt`).
    """
    campaign = Path(campaign).resolve()
    network, sites, analysis, config = _campaign_inputs(campaign)
    registry = campaign_registry(campaign, registry)
    native = {s["id"]: s["native"] for s in sites}
    cell = np.asarray(analysis["cell_A"], dtype=float)
    cutoff = float(config.get("scope", {}).get("pair_max_distance_A", 3.3))
    pair_ok = _pair_checker(sites, cell, cutoff)
    reactions = sorted({network.get("reaction", "oer")} | {c.get("reaction", "oer") for c in network["cycles"]})
    edges = {e["id"]: e for e in network["edges"]}
    new_edges = {}
    if auto_steps:
        from .builder import edge_id, oer_trio

        for name in reactions:
            rx = registry.reaction(name)
            rules = StepRules(native, registry.adsorbates, rx, pair_ok, allow_reverse)
            existing = {
                (e["from"], e["to"], json.dumps(clean(e.get("reservoirs") or {}), sort_keys=True))
                for e in edges.values()
            }
            for start, end, vector, rule in propose_steps(network["nodes"], rules, existing):
                eid = edge_id(start, end, vector)
                if eid in edges or eid in new_edges:
                    continue
                record = {"id": eid, "from": start, "to": end}
                trio = oer_trio(vector)
                if trio is not None:
                    record.update(water=trio[0], pairs=trio[1], oxygen=trio[2])
                record.update(
                    reservoirs=vector,
                    electrons=vector_electrons(vector),
                    family="discovered",
                    families=["discovered"],
                    source_ids=["govindrajan2020"],
                    status="proposed_auto",
                    notes=[f"Auto-proposed elementary step ({rule})."],
                )
                new_edges[eid] = record
    all_edges = {**edges, **new_edges}
    graph = build_graph(all_edges.values())
    curated = {}
    for c in network["cycles"]:
        rx = registry.reaction(c.get("reaction", network.get("reaction", "oer")))
        curated[
            canonical(
                c["edge_ids"],
                {
                    k: {
                        "reservoirs": clean(
                            v.get("reservoirs") or {"H2O": -v["water"], "H+": v["pairs"], "O2": v["oxygen"]}
                        )
                    }
                    for k, v in all_edges.items()
                },
                rx.release,
            )
        ] = c["id"]
    node_map = {n["id"]: n for n in network["nodes"]}
    touched = {key: touched_sites(edge, node_map, native) for key, edge in all_edges.items()}
    results, truncated_any = [], False
    for name in reactions:
        rx = registry.reaction(name)
        cycles, truncated = enumerate_cycles(graph, rx, max_length, max_cycles, touched, max_sites)
        truncated_any |= truncated
        classes = symmetry_classes(cycles, network, native, all_edges)
        for cycle, cls in zip(cycles, classes):
            families = sorted({all_edges[e]["family"] for e in cycle})
            results.append(
                {
                    "id": curated.get(cycle) or f"discovered_{short_hash(list(cycle), 10)}",
                    "reaction": name,
                    "edge_ids": list(cycle),
                    "node_ids": [all_edges[e]["from"] for e in cycle],
                    "length": len(cycle),
                    "curated": cycle in curated,
                    "families": families,
                    "uses_new_edges": [e for e in cycle if e in new_edges],
                    "symmetry_class": cls,
                }
            )
    seen_class = set()
    for r in sorted(results, key=lambda r: (not r["curated"], r["length"], r["id"])):
        r["symmetry_representative"] = (r["reaction"], r["symmetry_class"]) not in seen_class
        seen_class.add((r["reaction"], r["symmetry_class"]))
    discovered = [r for r in results if not r["curated"]]
    recovered = {r["id"] for r in results if r["curated"]}
    missing = sorted(c["id"] for c in network["cycles"] if c["id"] not in recovered)
    lengths = {}
    for r in discovered:
        if r["symmetry_representative"]:
            lengths[r["length"]] = lengths.get(r["length"], 0) + 1
    summary = {
        "max_length": max_length,
        "max_sites": max_sites,
        "auto_steps": auto_steps,
        "allow_reverse": allow_reverse,
        "reactions": reactions,
        "nodes": len(network["nodes"]),
        "network_edges": len(edges),
        "new_edges": len(new_edges),
        "cycles_found": len(results),
        "curated_cycles_recovered": sum(r["curated"] for r in results),
        "curated_cycles_total": len(network["cycles"]),
        "discovered_cycles": len(discovered),
        "discovered_symmetry_unique": sum(r["symmetry_representative"] for r in discovered),
        "discovered_unique_by_length": {str(k): lengths[k] for k in sorted(lengths)},
        "curated_not_recovered": missing,
        "truncated": truncated_any,
        "note": (
            "A curated cycle is not recovered when it is longer than max_length, touches more than "
            "max_sites sites, or uses a step outside the elementary rules."
        ),
    }
    report = {"schema_version": 1, "summary": summary, "cycles": results, "new_edges": list(new_edges.values())}
    out = Path(out) if out else campaign / "pathways"
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "pathways.json", report)
    with (out / "pathways.csv").open("w") as stream:
        stream.write("cycle_id,reaction,curated,length,symmetry_representative,families,new_edges,node_ids\n")
        for r in results:
            stream.write(
                f"{r['id']},{r['reaction']},{r['curated']},{r['length']},{r['symmetry_representative']},"
                f"{';'.join(r['families'])},{len(r['uses_new_edges'])},{' '.join(r['node_ids'])}\n"
            )
    return report


def _rotation_key(edge_ids):
    """The same key for every rotation of a cycle's edge sequence."""
    edge_ids = list(edge_ids)
    return min(tuple(edge_ids[k:] + edge_ids[:k]) for k in range(len(edge_ids))) if edge_ids else ()


def adopt(campaign, cycle_ids):
    """Copy discovered cycles (and the proposed edges they use) from pathways.json into network.json.

    ``thermo``, ``show-cycle`` and ``plot-profile`` then treat them like curated cycles. The
    adopted cycles reuse the computed intermediates, so nothing new is calculated. The
    first adoption keeps a backup, ``network.before_pathways.json``. Returns the adopted IDs.
    """
    campaign = Path(campaign)
    report = read_json(campaign / "pathways" / "pathways.json")
    network = read_json(campaign / "network.json")
    by_id = {r["id"]: r for r in report["cycles"]}
    unknown = sorted(set(cycle_ids) - set(by_id))
    if unknown:
        raise ValueError(f"Not in pathways/pathways.json: {unknown}")
    chosen = [by_id[i] for i in dict.fromkeys(cycle_ids) if not by_id[i]["curated"]]
    backup = campaign / "network.before_pathways.json"
    if chosen and not backup.exists():
        dump_json(backup, network)
    proposed = {e["id"]: e for e in report.get("new_edges", [])}
    edges = {e["id"]: e for e in network["edges"]}
    for r in chosen:
        for key in r["edge_ids"]:
            if key not in edges:
                edges[key] = proposed[key]
                network["edges"].append(proposed[key])
    have = {c["id"] for c in network["cycles"]}
    have_keys = {_rotation_key(c["edge_ids"]) for c in network["cycles"]}
    adopted = []
    for r in chosen:
        if r["id"] in have or _rotation_key(r["edge_ids"]) in have_keys:
            continue  # already in network.json (under its own ID, or as a curated cycle)
        total = add_vectors(*(edges[x]["reservoirs"] for x in r["edge_ids"]))
        network["cycles"].append(
            {
                "id": r["id"],
                "edge_ids": r["edge_ids"],
                "family": "discovered",
                "reaction": r["reaction"],
                "evidence_scope": EVIDENCE,
                "source_ids": ["govindrajan2020"],
                "reservoirs": total,
                "discovered": True,
                "geometry_complete": None,
                "symmetry_representative": True,
                "symmetry_equivalent_cycles": [],
            }
        )
        adopted.append(r["id"])
    if adopted and "discovered" not in network.get("families", []):
        network["families"] = list(network.get("families", [])) + ["discovered"]
    if adopted:
        dump_json(campaign / "network.json", network)
    return adopted


# ---------------------------------------------------------------------------
# Ranking with energies
# ---------------------------------------------------------------------------


def _edge_vector(edge):
    if edge.get("reservoirs") is not None:
        return clean(edge["reservoirs"])
    return clean({"H2O": -edge["water"], "H+": edge["pairs"], "O2": edge["oxygen"]})


def minimum_limiting_cycle(edges, delta_zero, reaction, tolerance=DOWNHILL_TOLERANCE_EV):
    """Lowest limiting potential over *all* turnover cycles of the graph, of any length.

    A cycle runs downhill at U when every step does: DeltaG(U) = DeltaG(0) - n U <= 0 for a
    step releasing n electrons (n = 0: a chemical step, which must be downhill on its own).
    In s = sign * U (sign = +1 for anodic, -1 for cathodic reactions) a step in the reaction's
    direction is downhill for s >= t and a step against it for s <= t, t = sign DeltaG(0)/n.
    The first threshold s at which a marker edge u -> v (``Reaction.release``) is closed by a
    path v -> u of steps downhill at s is the minimum limiting potential of the network: the
    bottleneck-path form of the minimum-overpotential search of Govind Rajan and Carter (2020),
    which needs no cycle enumeration. Feasibility grows with s when every step runs in the
    reaction's direction (binary search); steps against it (``pathways --allow-reverse``) stop
    being downhill as s grows, and the thresholds are then scanned in increasing order.

    Only the edges of ``delta_zero`` (edge ID -> DeltaG at 0 V on the analysis scale, i.e. the
    steps whose two ends have energies) take part; ``edges``: edge ID -> edge record. Returns
    None when no cycle can run downhill (or when a closed walk over the reaction's species need
    not be one turnover, see ``Reaction.closed_walks_are_turnovers``), else
    {"limiting_potential_V", "edge_ids"} with the edges in the turnover convention.
    """
    if not reaction.closed_walks_are_turnovers:
        return None
    release = reaction.release
    sign = 1 if reaction.electrons >= 0 else -1
    allowed_species = set(reaction.species)
    vectors, chemical, forward, reverse = {}, [], [], []
    for key, dg in delta_zero.items():
        vector = _edge_vector(edges[key])
        if set(vector) - allowed_species or reaction.opposes(vector):
            continue
        vectors[key] = vector
        n = vector_electrons(vector)
        if n == 0:
            if dg <= tolerance:
                chemical.append(key)
        elif sign * n > 0:
            forward.append((sign * dg / n, key))
        else:
            reverse.append((sign * dg / n, key))
    thresholds = sorted({x for x, _ in forward})

    def cycle_at(limit):
        allowed = (
            list(chemical)
            + [k for x, k in forward if x <= limit + tolerance]
            + [k for x, k in reverse if x >= limit - tolerance]
        )
        graph = nx.DiGraph()
        parallel = {}
        marker_edges = []
        for key in sorted(allowed):
            edge = edges[key]
            if reaction.advances(vectors[key]):
                marker_edges.append(key)
                continue
            graph.add_edge(edge["from"], edge["to"])
            parallel.setdefault((edge["from"], edge["to"]), key)
        best = None
        for key in marker_edges:
            u, v = edges[key]["from"], edges[key]["to"]
            if u == v:
                return [key]
            if u in graph and v in graph and nx.has_path(graph, v, u):
                path = nx.shortest_path(graph, v, u)
                cycle = [parallel[h] for h in zip(path[:-1], path[1:])] + [key]
                if best is None or len(cycle) < len(best):
                    best = cycle
        return best

    if not thresholds:
        return None
    if reverse:
        found = next(((x, c) for x in thresholds for c in [cycle_at(x)] if c is not None), None)
        if found is None:
            return None
        threshold, cycle = found
    else:
        if cycle_at(thresholds[-1]) is None:
            return None
        lo, hi = 0, len(thresholds) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cycle_at(thresholds[mid]) is not None:
                hi = mid
            else:
                lo = mid + 1
        threshold, cycle = thresholds[lo], cycle_at(thresholds[lo])
    records = {k: {"reservoirs": vectors[k]} for k in cycle}
    if add_vectors(*(vectors[k] for k in cycle)) != clean(reaction.net):
        return None
    return {"limiting_potential_V": sign * threshold, "edge_ids": list(canonical(cycle, records, release))}


def rank(
    campaign,
    references,
    potentials,
    pathways_report=None,
    include_vibrations=True,
    energy_source=None,
    scale=None,
    pH=None,
    top=10,
    registry=None,
):
    """Evaluate every enumerated cycle at each potential and find the best route of each reaction.

    Cycles are ranked per reaction, each against its own equilibrium potential; the top lists
    show symmetry representatives. For each reaction :func:`minimum_limiting_cycle` gives the
    exact lowest limiting potential over every cycle, of any length, formed by the steps whose
    two ends have energies; it is reported by its symmetry representative. Raises ValueError
    when the references are incomplete.
    """
    from .analysis import analyze_thermodynamics, edge_vector
    from .thermo import Conditions, cycle_metrics, vector_free_energy

    campaign = Path(campaign).resolve()
    registry = campaign_registry(campaign, registry)
    report = pathways_report or read_json(campaign / "pathways" / "pathways.json")
    thermo = analyze_thermodynamics(
        campaign,
        references,
        out_dir=campaign / "pathways" / "analysis",  # leaves the `thermo` results in analysis/ alone
        potentials_V=potentials,
        include_vibrations=include_vibrations,
        energy_source=energy_source,
        scale=scale,
        pH=pH,
        registry=registry,
    )
    reservoirs = thermo["reservoirs"]
    if not reservoirs["valid"]:
        raise ValueError(f"References incomplete ({references}): " + "; ".join(reservoirs["errors"]))
    energies = thermo["node_energies_eV"]
    conditions = Conditions(
        thermo["conditions"]["temperature_K"], thermo["conditions"]["pH"], thermo["conditions"]["activities"]
    )
    scale = thermo["scale"]
    network = read_json(campaign / "network.json")
    edges = {e["id"]: e for e in network["edges"]}
    new_edge_ids = {e["id"] for e in report.get("new_edges", [])} - set(edges)
    edges.update({e["id"]: e for e in report.get("new_edges", []) if e["id"] in new_edge_ids})
    delta_zero = {}
    for key, edge in edges.items():
        if edge["from"] in energies and edge["to"] in energies:
            delta_zero[key] = (
                energies[edge["to"]]
                - energies[edge["from"]]
                + vector_free_energy(edge_vector(edge), reservoirs, conditions, conditions.to_she(0.0, scale))
            )
    known = {_rotation_key(c["edge_ids"]): c for c in report["cycles"]}
    representative = {
        (c["reaction"], c["symmetry_class"]): c
        for c in report["cycles"]
        if c.get("symmetry_class") is not None and c.get("symmetry_representative")
    }
    # The network's own cycles, also when enumeration did not reach them (longer than
    # --max-length, more sites than --max-sites): the sweep's route may be one of them.
    main_reaction = network.get("reaction", "oer")
    curated = {}
    image_of = {}
    for c in network["cycles"]:
        curated[_rotation_key(c["edge_ids"])] = {
            "id": c["id"],
            "reaction": c.get("reaction") or main_reaction,
            "edge_ids": list(c["edge_ids"]),
            "node_ids": [edges[e]["from"] for e in c["edge_ids"] if e in edges],
            "length": len(c["edge_ids"]),
            "curated": not c.get("discovered", False),
            "in_network": True,
            "symmetry_representative": c.get("symmetry_representative", True),
        }
        for image in c.get("symmetry_equivalent_cycles") or []:
            image_of[image] = c["id"]
    curated_by_id = {c["id"]: c for c in curated.values()}
    names = sorted(
        {main_reaction}
        | {c.get("reaction") or main_reaction for c in network["cycles"]}
        | {c["reaction"] for c in report["cycles"]}
    )
    known_by_id = {c["id"]: c for c in report["cycles"]}
    rows, global_best, info = [], {}, {}
    for name in names:
        rx = registry.reaction(name)
        net_zero = vector_free_energy(rx.net, reservoirs, conditions, conditions.to_she(0.0, scale))
        equilibrium = net_zero / rx.electrons
        sign = 1 if rx.electrons >= 0 else -1
        ranked = 0
        for cycle in report["cycles"]:
            if cycle["reaction"] != name or any(e not in delta_zero for e in cycle["edge_ids"]):
                continue
            ranked += 1
            vectors = [edge_vector(edges[e]) for e in cycle["edge_ids"]]
            zero = [delta_zero[e] for e in cycle["edge_ids"]]
            for potential in thermo["potentials_V"]:
                m = cycle_metrics(
                    [{"reservoirs": v} for v in vectors],
                    zero,
                    potential,
                    [vector_electrons(v) for v in vectors],
                    equilibrium,
                    rx.release,
                    scale,
                )
                rows.append(
                    {
                        "cycle_id": cycle["id"],
                        "reaction": name,
                        "curated": cycle["curated"],
                        "symmetry_representative": cycle.get("symmetry_representative", True),
                        "potential_V": potential,
                        "scale": scale,
                        "Gmax_U_eV": m["Gmax_U_eV"],
                        "thermodynamic_overpotential_V": m["thermodynamic_overpotential_V"],
                        "limiting_potential_V": m["limiting_potential_V"],
                        "cyclic_span_eV": m["cyclic_span_eV"],
                        "largest_step_deltaG_eV": m["largest_step_deltaG_eV"],
                        "length": cycle["length"],
                        "node_ids": cycle["node_ids"],
                    }
                )
        usable = [
            k
            for k, e in edges.items()
            if not (set(edge_vector(e)) - set(rx.species)) and not rx.opposes(edge_vector(e))
        ]
        with_energy = sum(k in delta_zero for k in usable)
        against = any(sign * vector_electrons(edge_vector(edges[k])) < 0 for k in usable if k in delta_zero)
        info[name] = {
            "equilibrium_potential_V": equilibrium,
            "cycles_ranked": ranked,
            "cycles_without_energies": sum(c["reaction"] == name for c in report["cycles"]) - ranked,
            "steps": len(usable),
            "steps_with_energies": with_energy,
        }
        if not rx.closed_walks_are_turnovers:
            # the species admit closed walks that are not one turnover: report the best ranked cycle
            info[name]["sweep"] = (
                "skipped: the reaction's species allow closed walks that are not one turnover, so only "
                "enumerated cycles are ranked"
            )
            limits = [r for r in rows if r["reaction"] == name and r["limiting_potential_V"] is not None]
            if limits:
                top_row = min(limits, key=lambda r: (sign * r["limiting_potential_V"], r["cycle_id"]))
                global_best[name] = {
                    "limiting_potential_V": top_row["limiting_potential_V"],
                    "thermodynamic_overpotential_V": sign * (top_row["limiting_potential_V"] - equilibrium),
                    "equilibrium_potential_V": equilibrium,
                    "scale": scale,
                    "cycle_id": top_row["cycle_id"],
                    "edge_ids": list(known_by_id[top_row["cycle_id"]]["edge_ids"]),
                    "node_ids": list(top_row["node_ids"]),
                    "curated": bool(top_row["curated"]),
                    "symmetry_image_found": None,
                    "note": "Best enumerated cycle; the exact potential sweep was skipped (see reactions.sweep).",
                }
            continue
        best = minimum_limiting_cycle(edges, delta_zero, rx)
        if best is None:
            info[name]["sweep"] = "no cycle of steps with energies runs downhill at any potential"
            continue
        info[name]["sweep"] = "exact"
        key = _rotation_key(best["edge_ids"])
        match = known.get(key) or curated.get(key)
        if match is not None and match["id"] in image_of and match["id"] not in known_by_id:
            match = curated_by_id.get(image_of[match["id"]], match)  # the class representative
        if match is None:
            # longer than --max-length or beyond --max-sites: record it so that it can be adopted
            match = {
                "id": f"discovered_{short_hash(best['edge_ids'], 10)}",
                "reaction": name,
                "edge_ids": best["edge_ids"],
                "node_ids": [edges[e]["from"] for e in best["edge_ids"]],
                "length": len(best["edge_ids"]),
                "curated": False,
                "families": sorted({edges[e].get("family", "") for e in best["edge_ids"]}),
                "uses_new_edges": [e for e in best["edge_ids"] if e in new_edge_ids],
                "symmetry_class": None,
                "symmetry_representative": True,
                "found_by": "potential sweep",
            }
            report["cycles"].append(match)
            known[key] = match
            known_by_id[match["id"]] = match
            dump_json(campaign / "pathways" / "pathways.json", report)
        chosen = match
        if not match.get("symmetry_representative", True):
            # a translation image has the same energies: report the class representative
            chosen = representative.get((name, match.get("symmetry_class")), match)
        global_best[name] = {
            "limiting_potential_V": best["limiting_potential_V"],
            "thermodynamic_overpotential_V": sign * (best["limiting_potential_V"] - equilibrium),
            "equilibrium_potential_V": equilibrium,
            "scale": scale,
            "cycle_id": chosen["id"],
            "edge_ids": list(chosen["edge_ids"]),
            "node_ids": list(chosen["node_ids"]),
            "curated": bool(chosen["curated"]),
            "in_network": bool(chosen.get("in_network")) or _rotation_key(chosen["edge_ids"]) in curated,
            "symmetry_image_found": match["id"] if chosen is not match else None,
            "note": (
                f"Exact minimum over every cycle, of any length, formed by the {with_energy} of {len(usable)} "
                f"{name} steps whose two ends have energies"
                + (", including steps against the reaction's direction" if against else "")
                + " (potential sweep)."
            ),
        }

    def best_rows(subset, field):
        chosen = [r for r in subset if r[field] is not None and r["symmetry_representative"]]
        return sorted(chosen, key=lambda r: (r[field], r["cycle_id"]))[:top]

    by_gmax, by_eta = {}, {}
    for name in info:
        for potential in thermo["potentials_V"]:
            subset = [r for r in rows if r["reaction"] == name and r["potential_V"] == potential]
            by_gmax.setdefault(name, {})[str(potential)] = best_rows(subset, "Gmax_U_eV")
            by_eta.setdefault(name, {})[str(potential)] = best_rows(subset, "thermodynamic_overpotential_V")
    out = {
        "schema_version": 2,
        "scale": scale,
        "potentials_V": thermo["potentials_V"],
        "conditions": thermo["conditions"],
        "analysis_level": thermo["analysis_level"],
        "n_cycles_ranked": len({r["cycle_id"] for r in rows}),
        "reactions": info,
        "best_by_Gmax": by_gmax,
        "best_by_overpotential": by_eta,
        "global_minimum_limiting_potential": global_best,
        "warnings": thermo.get("warnings", []),
        "notes": thermo.get("notes", []),
        "rows": rows,
    }
    dump_json(campaign / "pathways" / "ranking.json", out)
    return out


# ---------------------------------------------------------------------------
# Build-time enumeration family
# ---------------------------------------------------------------------------

DEFAULT_ENUMERATION_STATES = {
    "oer": {"terminal": ["empty", "OH", "O", "OOH", "OO", "H2O"], "bridge": ["empty", "OH", "O"]},
    "cer": {"terminal": ["empty", "O", "Cl", "OCl"], "bridge": ["O", "OCl"]},
}


def enumerated(ctx):
    """Every occupancy of the chosen sites, every elementary step between them, every balanced cycle.

    Configured under ``scope.enumerate`` in project.yaml::

        enumerate:
          sites: [m0038, a0016]          # required; keep it to one or two sites
          states: {terminal: [empty, OH, O, OOH, OO], bridge: [O, OH]}
          max_length: 8
    """
    cfg = ctx.scope.get("enumerate") or {}
    site_ids = list(cfg.get("sites") or [])
    if not site_ids:
        raise ValueError("The enumerated family needs scope.enumerate.sites (one or two site IDs)")
    unknown = sorted(set(site_ids) - set(ctx.sites))
    if unknown:
        raise ValueError(f"scope.enumerate.sites names unknown sites {unknown}")
    if len(site_ids) > 3:
        raise ValueError("scope.enumerate.sites: at most three sites (the number of structures grows exponentially)")
    defaults = DEFAULT_ENUMERATION_STATES.get(ctx.reaction.name, DEFAULT_ENUMERATION_STATES["oer"])
    states_cfg = cfg.get("states") or {}
    options = {}
    for aid in site_ids:
        cls = ctx.site_class(aid)
        states = list(
            states_cfg.get(aid) or states_cfg.get(ctx.sites[aid]["kind"]) or states_cfg.get(cls) or defaults[cls]
        )
        if ctx.native[aid] not in states:
            states.insert(0, ctx.native[aid])
        options[aid] = states
    label = "+".join(site_ids)
    pairs = set(ctx.nearby_pairs())

    def pair_ok(a, b):
        return (a, b) in pairs or (b, a) in pairs

    ids = []
    for combo in product(*(options[aid] for aid in site_ids)):
        ids.append(ctx.node(dict(zip(site_ids, combo)), "enumerated", label))
    for a, b in combinations(site_ids, 2):
        if "O" in options[a] and "O" in options[b] and pair_ok(a, b):
            occupancy = {aid: ctx.native[aid] for aid in site_ids}
            occupancy.update({a: "O", b: "O"})
            ids.append(ctx.node(occupancy, "enumerated", label, coupling=(a, b)))
    nodes = [ctx.nodes[i] for i in dict.fromkeys(ids)]
    rules = StepRules({aid: ctx.native[aid] for aid in ctx.sites}, ctx.registry.adsorbates, ctx.reaction, pair_ok)
    for start, end, vector, rule in propose_steps(nodes, rules):
        ctx.step(start, end, "enumerated", f"Enumerated elementary step ({rule}).", given=vector)
    enumerated_edges = [e for e in ctx.edges.values() if "enumerated" in e["families"]]
    graph = build_graph(enumerated_edges)
    touched = {e["id"]: touched_sites(e, ctx.nodes, ctx.native) for e in enumerated_edges}
    cycles, truncated = enumerate_cycles(
        graph, ctx.reaction, int(cfg.get("max_length", 8)), int(cfg.get("max_cycles", 20000)), touched, None
    )
    for cycle in cycles:
        ctx.cycle(f"enumerated_{short_hash(list(cycle), 10)}", list(cycle), "enumerated", EVIDENCE)


ENUMERATED_FAMILY = Family(
    "enumerated",
    enumerated,
    None,
    "All occupancies of one to three chosen sites, all elementary steps between them and all "
    "balanced cycles (graph enumeration)",
    ["govindrajan2020"],
    core=False,
    order=150,
    scope_key="enumerate",
    scope_fields=("sites", "states", "max_length", "max_cycles"),
)


def register_family(registry):
    registry.add_family(ENUMERATED_FAMILY)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def register_command(subparsers, registry):
    p = subparsers.add_parser(
        "pathways",
        help="Find every balanced cycle of the network (networkx), rank them with energies",
        description=(
            "Enumerate the catalytic cycles of a campaign's network, curated or not, after proposing the "
            "elementary steps the families did not write; with --rank, evaluate every cycle and find the "
            "route of lowest limiting potential over all cycles of any length."
        ),
    )
    p.add_argument("campaign", nargs="?")
    p.add_argument("--max-length", type=int, default=6, help="Longest cycle enumerated, in steps (default 6)")
    p.add_argument(
        "--max-sites",
        type=int,
        default=2,
        help="Most sites one cycle may involve (default 2, as the curated families; 0 = no limit)",
    )
    p.add_argument("--no-auto-steps", action="store_true", help="Use only the network's own edges")
    p.add_argument("--allow-reverse", action="store_true", help="Also propose steps against the reaction direction")
    p.add_argument("--max-cycles", type=int, default=200000, help="Stop enumerating after this many cycles")
    p.add_argument("--rank", action="store_true", help="Rank every cycle that has energies, per reaction")
    p.add_argument(
        "--references", help="References file for --rank (default: thermodynamics.references of the campaign)"
    )
    p.add_argument("--potentials", help="Comma-separated potentials for --rank")
    p.add_argument("--scale", choices=["RHE", "SHE"])
    p.add_argument("--pH", type=float)
    p.add_argument("--electronic-only", action="store_true", help="Rank without 05_freq (screening estimate)")
    p.add_argument("--top", type=int, default=10, help="Cycles listed per potential in ranking.json (default 10)")
    p.add_argument(
        "--adopt",
        nargs="+",
        metavar="CYCLE_ID",
        help="Copy these discovered cycles into network.json; 'best' (with --rank) adopts, for each reaction, the "
        "discovered cycle of lowest overpotential",
    )
    p.set_defaults(handler=_cli)


def _text(result):
    s = result["summary"]
    sites = s["max_sites"] or "any number of"
    lines = [
        f"{s['cycles_found']} cycles of up to {s['max_length']} steps on {sites} sites: "
        f"{s['curated_cycles_recovered']} of {s['curated_cycles_total']} curated cycles recovered, "
        f"{s['discovered_cycles']} discovered ({s['discovered_symmetry_unique']} symmetry-unique), "
        f"{s['new_edges']} steps proposed"
    ]
    if s["curated_not_recovered"]:
        lines.append(f"  not recovered: {', '.join(s['curated_not_recovered'][:10])} ({s['note']})")
    if s["truncated"]:
        lines.append("  enumeration stopped at --max-cycles")
    ranking = result.get("ranking")
    if ranking:
        lines.append(f"ranking: {ranking['analysis_level']}; potentials vs {ranking['scale']}")
        for name, info in ranking["reactions"].items():
            best = ranking["global_minimum_limiting_potential"].get(name)
            if best:
                scope = (
                    f"exact over the {info['steps_with_energies']} of {info['steps']} steps with energies"
                    if info.get("sweep") == "exact"
                    else f"best enumerated cycle; {info.get('sweep')}"
                )
                lines.append(
                    f"  {name}: lowest limiting potential {best['limiting_potential_V']:.3f} V (eta_TD "
                    f"{best['thermodynamic_overpotential_V']:.3f} V), {best['cycle_id']}, {len(best['edge_ids'])} "
                    f"steps{', curated' if best['curated'] else ''}; {scope}"
                )
            else:
                lines.append(f"  {name}: {info.get('sweep') or 'no cycle with energies'}")
            for potential, rows in ranking["best_by_Gmax"].get(name, {}).items():
                if rows:
                    shown = "; ".join(f"{r['cycle_id']} {r['Gmax_U_eV']:.3f} eV" for r in rows[:3])
                    lines.append(f"    lowest Gmax at {potential} V: {shown}")
        lines += [f"  warning: {w}" for w in ranking.get("warnings", [])]
        lines += [f"  note: {n}" for n in ranking.get("notes", [])]
    if "adopted" in result:
        lines.append(f"adopted into network.json: {', '.join(result['adopted']) or 'nothing new'}")
    lines.append("files: " + ", ".join(result["files"]))
    return "\n".join(lines)


def _cli(args, registry):
    from .campaign import find_campaign_dir

    args.campaign = str(find_campaign_dir(args.campaign))
    if args.adopt and args.adopt != ["best"]:
        adopted = adopt(args.campaign, args.adopt)
        return {"adopted": adopted, "network": str(Path(args.campaign) / "network.json")}
    if args.adopt == ["best"] and not args.rank:
        raise ValueError("--adopt best needs --rank")
    if args.references and not args.rank:
        raise ValueError("--references is used by --rank; add --rank")
    references = None
    if args.rank:
        from .cli import default_references

        references = args.references or default_references(args.campaign)
    report = discover(
        args.campaign,
        max_length=args.max_length,
        auto_steps=not args.no_auto_steps,
        allow_reverse=args.allow_reverse,
        max_cycles=args.max_cycles,
        max_sites=args.max_sites or None,
        registry=registry,
    )
    result = {
        "summary": report["summary"],
        "files": [
            str(Path(args.campaign) / "pathways" / "pathways.json"),
            str(Path(args.campaign) / "pathways" / "pathways.csv"),
        ],
    }
    if references is not None:
        potentials = [float(x) for x in args.potentials.split(",")] if args.potentials else None
        ranking = rank(
            args.campaign,
            references,
            potentials,
            report,
            include_vibrations=not args.electronic_only,
            scale=args.scale,
            pH=args.pH,
            top=args.top,
            registry=registry,
        )
        result["files"].append(str(Path(args.campaign) / "pathways" / "ranking.json"))
        result["ranking"] = {
            "analysis_level": ranking["analysis_level"],
            "scale": ranking["scale"],
            "reactions": ranking["reactions"],
            "global_minimum_limiting_potential": ranking["global_minimum_limiting_potential"],
            "best_by_Gmax": {
                name: {potential: rows[:5] for potential, rows in by_potential.items()}
                for name, by_potential in ranking["best_by_Gmax"].items()
            },
            "warnings": ranking["warnings"],
            "notes": ranking["notes"],
        }
        if args.adopt == ["best"]:
            best = sorted(
                g["cycle_id"]
                for g in ranking["global_minimum_limiting_potential"].values()
                if not g["curated"] and not g.get("in_network")
            )
            result["adopted"] = adopt(args.campaign, best)
    return result, _text(result)
