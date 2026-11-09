"""Native shared protons: their two positions are tautomers of one intermediate.

A proton of the seed that lies within ``topology.shared_proton_max_OH_A`` of two oxygens is
shared in a short, strong hydrogen bond (the inspector and the builder list them). In such a
bond the proton is not owned by either oxygen in any chemically meaningful sense: in
water-hydroxyl layers on metals path-integral simulations find the distinction between the
covalent and the hydrogen bond partly or almost entirely lost (Li, Probert, Alavi and
Michaelides, Phys. Rev. Lett. 104, 066102, 2010), and protons hop quickly between surface
oxygens of wet oxides (Merte et al., Science 336, 889, 2012; Tocci and Michaelides, J. Phys.
Chem. Lett. 5, 474, 2014). Which oxygen a geometry optimization leaves it on is therefore a
configuration of the same intermediate, not another intermediate, as long as the
intermediate does not act on either oxygen.

Rules used here:

* A native shared proton is *background* for an intermediate when both of its oxygens are
  reactive sites that the intermediate leaves at their native occupancy.
* Where it sits is read from the H atoms around its two oxygens, not from atom labels
  (protons are indistinguishable): *seed* when the two oxygens hold as many H as in the seed,
  *partner* when one H went from the seed oxygen to the other, *centred* when an H lies
  within the upper O-H edge of both, otherwise unknown. An H that is bonded to neither
  oxygen, or also close to a third one, leaves the pair unknown.
* A relaxed structure that differs from an intermediate only by background protons in the
  partner or centred position is that intermediate in another tautomer, provided the
  occupancy read with them on their seed oxygens leaves both oxygens native.
* A shared proton next to a site the intermediate changes is chemistry: the builder already
  localizes it on its unchanged donor, and where it ends up decides the intermediate.
* The free energy of an intermediate is that of its lowest tautomer (another one is
  Boltzmann-weighted by exp(-dE/kT): 14 % at 0.05 eV, 2 % at 0.1 eV at 298 K). A cycle whose
  intermediates use different tautomers is fine when each was computed in the others'; when
  some were not, its step energies may carry the energy between two tautomers and the cycle
  is flagged. "Computed" means an accepted 02_final structure, where select compares the
  structures of one intermediate.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .fileio import read_json
from .magnetism import FrameMaps

SIDES = ("seed", "partner", "centred")


def _site_pairs(shared, ids, d_initial, native, gone=()):
    """{h: (seed_oxygen, partner_oxygen)} of the shared protons between two reactive sites that can
    be read on their own: pairs that share an oxygen with another such pair are left out (their H
    cannot be told apart), and so are pairs with an atom in ``gone`` (a molecule that left)."""
    pairs = {}
    for h, partners in sorted(shared.items()):
        partners = sorted(partners)
        if len(partners) != 2 or not all(ids[o] in native for o in partners):
            continue
        seed = min(partners, key=lambda o: d_initial[h, o])
        pairs[h] = (seed, partners[1] if seed == partners[0] else partners[0])
    count = defaultdict(int)
    for seed, partner in pairs.values():
        count[seed] += 1
        count[partner] += 1
    return {
        h: (seed, partner)
        for h, (seed, partner) in pairs.items()
        if count[seed] == 1 and count[partner] == 1 and not {h, seed, partner} & set(gone)
    }


def background_protons(shared, ids, d_initial, native, occupancy, coupling, gone=()):
    """{h: (seed_oxygen, partner_oxygen)} of the native shared protons that are background.

    ``shared`` maps an H index to the indices of its two oxygens (``analysis._shared_protons``:
    native protons still at their seed position); ``occupancy`` and ``coupling`` are the
    intermediate's (``state.json``, or the node's). Both oxygens must be reactive sites left at
    their native occupancy by the intermediate; without an occupancy nothing is background.
    Pairs that share an oxygen with another pair between reactive sites, or that lost an atom to
    a released molecule (``gone``), are left out.
    """
    if not shared or not native or occupancy is None:
        return {}
    changed = {s for s, v in (occupancy or {}).items() if native.get(s) != v} | set(coupling or ())
    return {
        h: pair
        for h, pair in _site_pairs(shared, ids, d_initial, native, gone).items()
        if not {ids[pair[0]], ids[pair[1]]} & changed
    }


def pair_hydrogens(seed, partner, hydrogen, oxygen, distances, lo, hi):
    """The H atoms around two oxygens: (on the seed, on the partner, between both), or None.

    An H whose nearest oxygen is one of the two is *on* it below the lower O-H hysteresis edge
    ``lo`` while the other is beyond that edge, and *between* them within the upper edge ``hi``
    of both. An H close to one of them and to a third oxygen, stretched into the band towards
    neither, or bonded to no oxygen at all makes the pair unreadable (None).
    """
    pair = (seed, partner)
    on_seed, on_partner, between = [], [], []
    for h in hydrogen:
        near = sorted((float(distances[h, o]), o) for o in oxygen)
        if not near or near[0][1] not in pair:
            continue
        d0, o0 = near[0]
        other = partner if o0 == seed else seed
        d_other = float(distances[h, other])
        if any(d <= hi for d, o in near[1:] if o not in pair):
            return None
        if d0 < lo and d_other >= lo:
            (on_seed if o0 == seed else on_partner).append(h)
        elif d0 <= hi and d_other <= hi:
            between.append(h)
        else:
            return None
    return on_seed, on_partner, between


def pair_side(initial, final):
    """'seed', 'partner', 'centred' or None from :func:`pair_hydrogens` of the start and of the relaxed structure."""
    if initial is None or final is None or initial[2]:
        return None
    s0, p0 = len(initial[0]), len(initial[1])
    s, p, b = len(final[0]), len(final[1]), len(final[2])
    if s + p + b != s0 + p0:
        return None
    if not b:
        return "seed" if (s, p) == (s0, p0) else "partner" if (s, p) == (s0 - 1, p0 + 1) else None
    # An H between both: centred when it is the step between the seed and the transferred
    # arrangement (one way the seed's, the other way the transferred counts).
    return "centred" if s <= s0 - 1 and s + b >= s0 else None


def normalized_owners(seed, partner, initial, final, distances):
    """The H around a pair given back to the two oxygens as in the seed: {o: [h, ...]}.

    The H nearest to the seed oxygen go to it, as many as it held in the seed; the others go to
    the partner. Used only when :func:`pair_side` is 'partner' or 'centred'.
    """
    around = sorted(final[0] + final[1] + final[2], key=lambda h: float(distances[h, seed]))
    n_seed = len(initial[0])
    return {seed: around[:n_seed], partner: around[n_seed:]}


def tautomer_record(shared, ids, symbols, d_initial, d_final, native, background, hysteresis, gone=()):
    """Where the H of each native shared proton between two reactive sites sit in the relaxed structure.

    {H id: {"seed", "partner", "side", "on", "moved", "background", "distances_A"}}: ``side`` as
    in :func:`pair_side` (None for a pair that cannot be read on its own, see
    :func:`background_protons`), ``on`` the oxygen that holds the extra H ('seed' or 'partner'
    side; None when centred or unknown), ``distances_A`` those of the labelled H to the two
    oxygens.
    """
    record = {}
    if not native:
        return record
    lo, hi = hysteresis["O-H"]
    hydrogen = [i for i, s in enumerate(symbols) if s == "H"]
    oxygen = [i for i, s in enumerate(symbols) if s == "O"]
    pairs = {}
    for h, partners in sorted(shared.items()):
        partners = sorted(partners)
        if len(partners) != 2 or not all(ids[o] in native for o in partners):
            continue
        seed = min(partners, key=lambda o: d_initial[h, o])
        pairs[h] = (seed, partners[1] if seed == partners[0] else partners[0])
    readable = _site_pairs(shared, ids, d_initial, native, gone)
    for h, (seed, other) in pairs.items():
        side = None
        if h in readable:
            side = pair_side(
                pair_hydrogens(seed, other, hydrogen, oxygen, d_initial, lo, hi),
                pair_hydrogens(seed, other, hydrogen, oxygen, d_final, lo, hi),
            )
        record[ids[h]] = {
            "seed": ids[seed],
            "partner": ids[other],
            "side": side,
            "on": ids[seed] if side == "seed" else ids[other] if side == "partner" else None,
            "moved": side in ("partner", "centred"),
            "background": h in background,
            "distances_A": {ids[o]: round(float(d_final[h, o]), 3) for o in (seed, other)},
        }
    return record


def moved_lines(record):
    """Sentences for the background shared protons that are not where the seed had them."""
    out = []
    for h, entry in sorted((record or {}).items()):
        if not entry.get("background") or not entry.get("moved"):
            continue
        seed, partner = entry["seed"], entry["partner"]
        if entry.get("side") == "centred":
            out.append(f"the shared H {h} sits between O {seed} and O {partner}")
        else:
            distances = entry.get("distances_A") or {}
            labelled = distances.get(partner, 9) < distances.get(seed, 9)
            out.append(
                f"the shared H {h} moved from O {seed} to O {partner}"
                if labelled
                else f"a shared H moved from O {seed} to O {partner}"
            )
    return out


class TautomerCheck:
    """Which tautomer each intermediate of a cycle uses, and whether the others were computed.

    ``rows`` are collect's state rows (``final_valid``, ``final_assigned_node_id``,
    ``final_topology``, and ``assigned_node_id`` with ``topology`` for the analysis geometry);
    ``selected`` maps a node to the structure its energy uses, whose tautomer is read from
    ``topology_key``. Which tautomers of a node were computed is judged on its accepted 02_final
    structures (where select compares the structures of one intermediate), together with the
    analysis geometry of those structures. Symmetry images are relabelled into their own frame
    as for the local moments.
    """

    def __init__(self, network, native, rows, selected, topology_key="topology"):
        self.maps = FrameMaps(network)
        self.native = dict(native or {})
        self.rows = {row["state_id"]: row for row in rows}
        self.selected = {k: v for k, v in dict(selected).items() if v}
        self.topology_key = topology_key
        self._assigned = defaultdict(list)
        for row in rows:
            if row.get("final_valid") and row.get("final_assigned_node_id"):
                self._assigned[row["final_assigned_node_id"]].append(row["state_id"])

    @classmethod
    def from_analysis(cls, campaign, analysis_dir, network, thermo):
        """The check behind a thermodynamics result (``thermo``) whose collect tables are in ``analysis_dir``."""
        path = Path(analysis_dir) / "states.json"
        states = read_json(path).get("states", []) if path.is_file() else []
        selected = {n["node_id"]: n.get("selected_state_id") for n in thermo.get("nodes", [])}
        sites = Path(campaign) / "sites.json"
        native = {s["id"]: s["native"] for s in read_json(sites)["sites"]} if sites.is_file() else None
        key = (thermo.get("tautomers") or {}).get("topology_key") or (thermo.get("magnetism") or {}).get("topology_key")
        return cls(network, native, states, selected, key or "topology")

    def _changed(self, node_id):
        node = self.maps.nodes.get(node_id) or {}
        changed = {s for s, v in (node.get("occupancy") or {}).items() if self.native.get(s) != v}
        return changed | set(node.get("coupling") or ())

    def _record(self, state_id, target, key=None):
        """{H: O holding it, '*' when it sits between both} in ``target``'s frame, for the shared
        protons whose two oxygens ``target`` leaves native; None when a translation is unknown."""
        row = self.rows.get(state_id) or {}
        topology = row.get(key or self.topology_key) or {}
        record = topology.get("tautomer") or {}
        if not record:
            return {}
        matched = (topology.get("microstate") or {}).get("matched_node_id")
        changed = self._changed(target)
        out = {}
        for h, entry in record.items():
            side = entry.get("side")
            if side not in SIDES:
                continue
            mapped = self.maps.relabel([h, entry["seed"], entry["partner"]], target, matched)
            if mapped is None:
                return None
            if {mapped[1], mapped[2]} & changed:
                continue
            out[mapped[0]] = mapped[1] if side == "seed" else mapped[2] if side == "partner" else "*"
        return out

    def _keys(self, node_id):
        """Tautomers {H: O} of every accepted structure of ``node_id``."""
        owner = self.maps.representative(node_id)
        keys = []
        for sid in self._assigned.get(owner, []):
            row = self.rows.get(sid) or {}
            sources = ["final_topology"]
            if self.topology_key != "final_topology" and row.get("assigned_node_id") == owner:
                sources.append(self.topology_key)
            for key in sources:
                record = self._record(sid, node_id, key)
                if record:
                    keys.append(record)
        return keys

    @staticmethod
    def _matches(key, wanted):
        return all(h in key and key[h] in (o, "*") for h, o in wanted.items())

    def variants(self, node_id):
        """Tautomers among the accepted 02_final structures of a node: [{"on": {H: O}, "state_ids": [...]}]."""
        groups = {}
        for sid in self._assigned.get(self.maps.representative(node_id), []):
            record = self._record(sid, node_id, "final_topology") or {}
            groups.setdefault(tuple(sorted(record.items())), []).append(sid)
        return [{"on": dict(key), "state_ids": sids} for key, sids in groups.items()]

    def cycle(self, node_ids):
        """Tautomer check of one cycle (its intermediates in order).

        A shared proton is compared between the intermediates that leave both its oxygens
        native; tautomers are compared as a whole (every such proton at once). status:
        ``not_available`` (an intermediate without a structure, or no shared proton common to
        two of them), ``consistent`` (they agree), ``mixed_both_computed`` (they differ, and each
        was computed in the tautomer of every other) or ``needs_review`` (some were not: the
        step energies between them may contain the energy between two tautomers).
        """
        selected = {}
        for node_id in node_ids:
            sid = self.selected.get(node_id)
            if not sid:
                return {"status": "not_available", "reason": f"{node_id} has no structure with an accepted energy"}
            record = self._record(sid, node_id)
            if record is None:
                return {"status": "not_available", "reason": f"{node_id}: symmetry translation not recorded"}
            selected[node_id] = record
        nodes = list(dict.fromkeys(node_ids))
        protons = sorted({h for n in nodes for h in selected[n] if sum(h in selected[m] for m in nodes) > 1})
        if not protons:
            return {
                "status": "not_available",
                "reason": "no shared proton between two sites that two or more intermediates leave native",
            }
        differ, missing = False, defaultdict(list)
        for n in nodes:
            keys = None
            for m in nodes:
                shared = [h for h in protons if h in selected[n] and h in selected[m]]
                if m == n or not shared:
                    continue
                theirs = {h: selected[m][h] for h in shared}
                if all(theirs[h] == selected[n][h] or "*" in (theirs[h], selected[n][h]) for h in shared):
                    continue
                differ = True
                keys = self._keys(n) if keys is None else keys
                wanted = {h: o for h, o in theirs.items() if o != "*"}
                if not any(self._matches(key, wanted) for key in keys) and wanted not in missing[n]:
                    missing[n].append(wanted)
        rows = []
        for h in protons:
            on = {n: selected[n][h] for n in nodes if h in selected[n]}
            if len({o for o in on.values() if o != "*"}) > 1:
                rows.append({"H": h, "on": on})
        status = "consistent" if not differ else "needs_review" if missing else "mixed_both_computed"
        return {"status": status, "protons": rows, "not_computed": dict(missing)}


CYCLE_STATUS_TEXT = {
    "consistent": "every intermediate uses the same tautomer",
    "mixed_both_computed": "the intermediates use different tautomers, each computed in the others' (the lowest "
    "counts)",
    "needs_review": "the intermediates use different tautomers and some were not computed in the others'",
}


def cycle_text(check, label):
    """Lines for show-cycle: which intermediates hold each shared H where, and what was not computed."""
    lines = []
    for entry in check.get("protons") or []:
        by_side = defaultdict(list)
        for node_id, side in entry["on"].items():
            by_side[side].append(label(node_id))
        parts = [
            ("between its two O" if side == "*" else f"on O {side}") + f" in {', '.join(dict.fromkeys(names))}"
            for side, names in sorted(by_side.items())
        ]
        lines.append(f"shared H {entry['H']}: {'; '.join(parts)}")
    for node_id, wanted in (check.get("not_computed") or {}).items():
        for key in wanted:
            where = " and ".join(f"H {h} on O {o}" for h, o in sorted(key.items()))
            lines.append(f"  {label(node_id)} was not computed with {where}")
    return lines


NOTE = (
    "Accepted structures of an intermediate that differ only in where a native shared proton sits, both its "
    "oxygens being sites the intermediate leaves native, are tautomers of the same intermediate (read from "
    "the H counts on the two oxygens, not from atom labels). Its energy is that of its lowest tautomer "
    "(another tautomer weighs exp(-dE/kT): 14 % at 0.05 eV, 2 % at 0.1 eV, 298 K). A cycle whose "
    "intermediates use different tautomers is flagged (needs_review) when some of them were not computed "
    "(an accepted 02_final structure) in the tautomer of another: the step energies between them may then "
    "contain the energy between two tautomers. A proton next to a site the intermediate changes is part of "
    "its chemistry and is not counted here."
)
