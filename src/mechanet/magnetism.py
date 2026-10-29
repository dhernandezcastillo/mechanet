"""Local magnetic moments across intermediates: cycle moments, spin checks, variant solutions.

Diagnostics built from the projected local moments that VASP prints with LORBIT (the numbers
of analysis/magnetic_moments.csv). None of them changes an energy.

* Cycle moments: for each intermediate of a cycle, the moments of the active-site atoms (the
  bound and outer atoms of every site the cycle changes, and the metals the sites bind to),
  plus the other atoms whose moment changes most. ``plot-profile`` draws them under the
  free-energy profile; ``show-cycle`` and the report print them.
* Spin check of a cycle: between consecutive intermediates, a moment of at least
  ``analysis.moment_change_muB`` that changes sign anywhere, or a moment that changes by more
  than that away from the active site (farther than ``analysis.moment_site_radius_A`` from
  every site metal), is flagged. The two intermediates of such a step have their spectator
  atoms in different magnetic solutions (or the oxidation went away from the site), and the
  step's free energy contains that difference.
* Magnetic solutions of the variants of one intermediate: accepted 02_final variants whose
  moments differ in the same way did not converge to the same solution, so part of their
  energy gap is a spin-state gap.

Atoms are matched by their stable IDs. A symmetry image is computed by its representative's
structure, so its atoms are relabelled through the lattice translation recorded for it.
Moments are not oxidation states, and a consistent cycle does not prove the magnetic ground
state: every structure starts from the same MAGMOM.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .fileio import read_json
from .structure import distance_matrix, read_poscar
from .vasp import STAGE_FINAL, STAGE_STATIC

SITE_LABEL = re.compile(r"[am]\d{4}")
NATIVE_LABEL = re.compile(r"a\d{4}$")
DEFAULT_CHANGE_MUB = 0.5
DEFAULT_SITE_RADIUS_A = 3.6
# Atoms outside the active site are shown when their moment spans at least this much.
MIN_SHOWN_RANGE_MUB = 0.25
MAX_EXTRA_ATOMS = 2
ROLES = ("site metal", "bound", "outer", "flagged", "largest change")
NOTE = (
    "Projected local moments (LORBIT) of the structure each intermediate's energy comes from. A flagged "
    "step changes a moment's sign, or changes a moment away from the active site, between two "
    "intermediates: their spectator atoms are in different magnetic solutions (or the oxidation went "
    "elsewhere), and that difference is in the step's free energy. Moments are not oxidation states; a "
    "consistent cycle does not prove the magnetic ground state."
)


def moment_settings(config):
    """Thresholds of the moment diagnostics (``analysis.moment_change_muB``, ``moment_site_radius_A``)."""
    analysis = (config or {}).get("analysis") or {}
    return {
        "moment_change_muB": float(analysis.get("moment_change_muB", DEFAULT_CHANGE_MUB)),
        "site_radius_A": float(analysis.get("moment_site_radius_A", DEFAULT_SITE_RADIUS_A)),
    }


def _relabel(label, mapping):
    return SITE_LABEL.sub(lambda m: mapping.get(m.group(0), m.group(0)), str(label))


def differs(a, b, threshold):
    """'sign' when moments of at least ``threshold`` have opposite signs, 'change' when two moments
    differ by more than ``threshold``, otherwise None."""
    if a * b < 0 and min(abs(a), abs(b)) >= threshold:
        return "sign"
    if abs(b - a) > threshold:
        return "change"
    return None


class FrameMaps:
    """Atom labels between a node's own frame and its symmetry representative's.

    Every node is described with the slab's labels. A symmetry image has no structure of its
    own: its representative's is used, relabelled by the lattice translation recorded for the
    image (``node.symmetry.translation_fractional``, one of ``network.symmetry.translations``).
    """

    def __init__(self, network):
        self.nodes = {n["id"]: n for n in network.get("nodes", [])}
        translations = []
        for t in (network.get("symmetry") or {}).get("translations") or []:
            mapping = {**(t.get("atom_id_map") or {}), **(t.get("site_map") or {})}
            translations.append((np.asarray(t.get("fractional") or [], dtype=float), mapping))
        self.to_rep, self.from_rep = {}, {}
        for node_id, node in self.nodes.items():
            if not node.get("computed_by"):
                continue
            shift = (node.get("symmetry") or {}).get("translation_fractional")
            mapping = None
            if shift is not None:
                shift = np.asarray(shift, dtype=float)
                for fractional, candidate in translations:
                    if fractional.shape == shift.shape and np.allclose(fractional, shift, atol=1e-5):
                        mapping = candidate
                        break
            self.to_rep[node_id] = mapping
            self.from_rep[node_id] = {v: k for k, v in mapping.items()} if mapping is not None else None

    def representative(self, node_id):
        return (self.nodes.get(node_id) or {}).get("computed_by") or node_id

    def relabel(self, labels, target, matched=None):
        """Labels of a structure used for ``target``, expressed in ``target``'s frame.

        The structure was computed for ``target``'s representative; ``matched`` is the node
        whose site occupancy the relaxed structure has (the topology check's
        ``matched_node_id``), which may be an image of that representative. Returns None when
        a translation needed for the relabelling is unknown.
        """
        rep = self.representative(target)
        steps = []
        if matched and matched != rep and self.representative(matched) == rep:
            steps.append(self.to_rep.get(matched))
        if target != rep:
            steps.append(self.from_rep.get(target))
        if any(step is None for step in steps):
            return None
        out = list(labels)
        for step in steps:
            out = [_relabel(label, step) for label in out]
        return out


def _solution_differences(first, second, threshold):
    out = []
    for label in first.keys() & second.keys():
        element, a = first[label]
        if element == "H":
            continue
        b = second[label][1]
        if differs(a, b, threshold):
            out.append((label, element, a, b))
    return sorted(out, key=lambda x: (-abs(x[3] - x[2]), x[0]))


def magnetic_solutions(variants, threshold=DEFAULT_CHANGE_MUB):
    """Group the variants of one intermediate by magnetic solution.

    ``variants``: [(state_id, energy_eV, {label: (element, moment)})]. Variants are taken from
    the lowest energy up; each joins the first solution whose lowest variant has, on every
    heavy atom both have, moments that neither change sign (when both are at least
    ``threshold``) nor differ by more than ``threshold``; otherwise it starts a solution.
    Returns the solutions, lowest first, each with its energy above the lowest and, for the
    others, the atoms in which it differs from the lowest.
    """
    ordered = sorted((v for v in variants if v[2]), key=lambda v: (v[1], v[0]))
    groups = []
    for state_id, energy, moments in ordered:
        for group in groups:
            if not _solution_differences(group["moments"], moments, threshold):
                group["state_ids"].append(state_id)
                break
        else:
            groups.append({"state_ids": [state_id], "energy": energy, "moments": moments})
    out = []
    for group in groups:
        entry = {"state_ids": group["state_ids"], "relative_eV": round(group["energy"] - groups[0]["energy"], 4)}
        if group is not groups[0]:
            diffs = _solution_differences(groups[0]["moments"], group["moments"], threshold)
            entry["n_atoms_differing"] = len(diffs)
            entry["differs_from_lowest"] = [
                {"atom_id": label, "element": element, "lowest_muB": round(a, 3), "this_muB": round(b, 3)}
                for label, element, a, b in diffs[:5]
            ]
        out.append(entry)
    return out


class MomentTracker:
    """Local moments of the structures an analysis uses, per node and along cycles.

    ``state_rows``: rows of :func:`mechanet.analysis.collect` (``stages`` with the parsed
    moments, ``topology``/``final_topology``, ``final_valid``, ``final_assigned_node_id``,
    ``energy_final_e0_eV``). ``selected``: node ID -> state ID whose energy the node uses
    (symmetry images included). ``stage``: the stage whose moments belong to that energy;
    ``topology_key``: the topology record that assigned the state to its node.
    """

    def __init__(self, campaign, network, state_rows, selected, stage, topology_key="topology", config=None):
        self.campaign = Path(campaign)
        self.maps = FrameMaps(network)
        self.rows = {row["state_id"]: row for row in state_rows}
        self.selected = {k: v for k, v in dict(selected).items() if v}
        self.stage = stage
        self.topology_key = topology_key
        self.settings = moment_settings(config if config is not None else self._config())
        self.threshold = self.settings["moment_change_muB"]
        sites_path = self.campaign / "sites.json"
        sites = read_json(sites_path)["sites"] if sites_path.is_file() else []
        self.native = {s["id"]: s["native"] for s in sites} if sites else None
        self.site_metals = {s["id"]: [m["id"] for m in s.get("metal_neighbors") or []] for s in sites}
        self.seed_index, self.seed_distances = {}, None
        seed_path = self.campaign / "inputs" / "seed.vasp"
        if seed_path.is_file():
            try:
                seed = read_poscar(seed_path)
                self.seed_index = {aid: k for k, aid in enumerate(seed.ids)}
                self.seed_distances = distance_matrix(seed)
            except (OSError, ValueError):
                self.seed_index, self.seed_distances = {}, None
        self._labels_cache, self._node_cache, self._steps, self._regions = {}, {}, {}, {}

    def _config(self):
        path = self.campaign / "config.json"
        return read_json(path) if path.is_file() else {}

    @classmethod
    def from_analysis(cls, campaign, analysis_dir, network, thermo):
        """Tracker for a thermodynamics result (``thermo``) whose collect tables are in ``analysis_dir``."""
        info = thermo.get("magnetism") or {}
        path = Path(analysis_dir) / "states.json"
        states = read_json(path).get("states", []) if path.is_file() else []
        selected = {n["node_id"]: n.get("selected_state_id") for n in thermo.get("nodes", [])}
        stage = info.get("stage") or (STAGE_FINAL if thermo.get("energy_source") == "final" else STAGE_STATIC)
        return cls(campaign, network, states, selected, stage, info.get("topology_key") or "topology")

    # ------------------------------------------------------------------ moments per state and node
    def _labels(self, state_id):
        if state_id not in self._labels_cache:
            try:
                meta = read_json(self.campaign / "states" / state_id / "state.json")
                self._labels_cache[state_id] = (list(meta["atom_ids"]), list(meta["symbols"]))
            except (OSError, ValueError, KeyError):
                self._labels_cache[state_id] = None
        return self._labels_cache[state_id]

    def state_moments(self, state_id, target, stage=None, topology_key=None):
        """({label: (element, moment)} in ``target``'s frame, None) or (None, reason)."""
        stage = stage or self.stage
        row = self.rows.get(state_id)
        labels = self._labels(state_id)
        if row is None or labels is None:
            return None, f"no collected record for {state_id}"
        values = ((row.get("stages") or {}).get(stage) or {}).get("magmoms")
        ids, symbols = labels
        if values is None or len(values) != len(ids):
            return None, f"no local moments in {stage} of {state_id} (they need ISPIN = 2 and LORBIT)"
        microstate = (row.get(topology_key or self.topology_key) or {}).get("microstate") or {}
        mapped = self.maps.relabel(ids, target, microstate.get("matched_node_id"))
        if mapped is None:
            return None, f"{target}: the symmetry translation to its representative is not recorded"
        return {label: (symbol, float(value)) for label, symbol, value in zip(mapped, symbols, values)}, None

    def node_moments(self, node_id):
        """(moments, reason) of the structure whose energy ``node_id`` uses, in its own frame."""
        if node_id not in self._node_cache:
            state_id = self.selected.get(node_id)
            self._node_cache[node_id] = (
                self.state_moments(state_id, node_id) if state_id else (None, "no structure with an accepted energy")
            )
        return self._node_cache[node_id]

    # ------------------------------------------------------------------ cycles
    def _sites(self, node_ids):
        """(sites whose occupancy changes along the cycle, every site some intermediate changes)."""
        native = self.native or {}
        occupancies, couplings = [], []
        for node_id in node_ids:
            node = self.maps.nodes.get(node_id) or {}
            occupancies.append({**native, **(node.get("occupancy") or {})})
            couplings.append(set(node.get("coupling") or ()))
        keys = set().union(*occupancies) if occupancies else set()
        changing = {s for s in keys if len({occ.get(s) for occ in occupancies}) > 1}
        coupled = set().union(*couplings) if couplings else set()
        changing |= coupled - (set.intersection(*couplings) if couplings else set())
        present = {s for s in keys if any(occ.get(s) != native.get(s) for occ in occupancies)} | coupled
        return sorted(changing), sorted(present | changing)

    def _region(self, sites):
        """Labels near the active site: the sites, their metals and every slab atom within the radius."""
        key = frozenset(sites)
        if key not in self._regions:
            metals = sorted({m for s in sites for m in self.site_metals.get(s, ())})
            labels = set(sites) | set(metals)
            nearest = {}
            index = [self.seed_index[m] for m in metals if m in self.seed_index]
            if self.seed_distances is not None and index:
                closest = self.seed_distances[index].min(axis=0)
                for aid, k in self.seed_index.items():
                    nearest[aid] = float(closest[k])
                    if closest[k] <= self.settings["site_radius_A"] + 1e-9:
                        labels.add(aid)
            self._regions[key] = (labels, nearest, key)
        return self._regions[key]

    @staticmethod
    def _in_region(label, region):
        labels, _, sites = region
        if label in labels:
            return True
        if NATIVE_LABEL.match(label):
            return False
        # an adsorbate atom (h_/d_/p_/w_ + site) belongs to its site
        return any(m.group(0) in sites for m in SITE_LABEL.finditer(label))

    def _step(self, a, b):
        """Heavy atoms whose moment changes sign or by more than the threshold from node a to b."""
        key = (a, b)
        if key not in self._steps:
            first, second = self.node_moments(a)[0], self.node_moments(b)[0]
            if first is None or second is None:
                self._steps[key] = None
            else:
                out = []
                for label in first.keys() & second.keys():
                    element, x = first[label]
                    if element == "H":
                        continue
                    y = second[label][1]
                    kind = differs(x, y, self.threshold)
                    if kind:
                        out.append((label, element, x, y, kind))
                self._steps[key] = sorted(out)
        return self._steps[key]

    def check(self, node_ids):
        """Spin check of a cycle: its intermediates in order, the last step returning to the first."""
        node_ids = list(node_ids)
        _, present = self._sites(node_ids)
        region = self._region(present)
        flags, unchecked = [], []
        for k, a in enumerate(node_ids):
            b = node_ids[(k + 1) % len(node_ids)]
            changes = self._step(a, b)
            if changes is None:
                unchecked.append(k + 1)
                continue
            for label, element, x, y, kind in changes:
                if kind == "sign":
                    what = "sign_flip"
                elif not self._in_region(label, region):
                    what = "away_from_site"
                else:
                    continue
                flags.append(
                    {
                        "step": k + 1,
                        "from_node": a,
                        "to_node": b,
                        "atom_id": label,
                        "element": element,
                        "from_muB": round(x, 3),
                        "to_muB": round(y, 3),
                        "kind": what,
                        "distance_to_site_metal_A": round(region[1][label], 2) if label in region[1] else None,
                    }
                )
        with_moments = sum(self.node_moments(n)[0] is not None for n in node_ids)
        status = (
            "not_available"
            if with_moments == 0
            else "needs_review"
            if flags
            else "incomplete"
            if unchecked
            else "consistent"
        )
        return {"status": status, "flags": flags, "unchecked_steps": unchecked}

    def cycle(self, node_ids):
        """Moments of the active-site atoms (and the largest other changes) along a cycle, with its spin check."""
        node_ids = list(node_ids)
        result = self.check(node_ids)
        changing, present = self._sites(node_ids)
        per = [self.node_moments(n) for n in node_ids]
        moments = [m for m, _ in per]
        tracked = [
            (metal, "site metal") for metal in sorted({m for s in changing for m in self.site_metals.get(s, ())})
        ]
        for site in changing:
            tracked += [(site, "bound"), (f"d_{site}", "outer")]
        seen = {label for label, _ in tracked}
        flagged = {}
        for flag in result["flags"]:
            if flag["atom_id"] not in seen:
                size = abs(flag["to_muB"] - flag["from_muB"])
                flagged[flag["atom_id"]] = max(flagged.get(flag["atom_id"], 0.0), size)
        extra = [(label, "flagged") for label in sorted(flagged, key=lambda x: (-flagged[x], x))]
        spans = {}
        for label in set().union(*(m.keys() for m in moments if m)) - seen - set(flagged):
            values = [m[label][1] for m in moments if m and label in m]
            element = next(m[label][0] for m in moments if m and label in m)
            if element != "H" and len(values) >= 2 and max(values) - min(values) >= MIN_SHOWN_RANGE_MUB:
                spans[label] = max(values) - min(values)
        extra += [(label, "largest change") for label in sorted(spans, key=lambda x: (-spans[x], x))]
        tracked += extra[:MAX_EXTRA_ATOMS]
        atoms = []
        for label, role in tracked:
            values = [m[label][1] if m and label in m else None for m in moments]
            if all(v is None for v in values):
                continue
            element = next(m[label][0] for m in moments if m and label in m)
            atoms.append(
                {
                    "atom_id": label,
                    "element": element,
                    "role": role,
                    "moments_muB": [None if v is None else round(v, 3) for v in values],
                }
            )
        result.update(
            {
                "node_ids": node_ids,
                "state_ids": [self.selected.get(n) for n in node_ids],
                "active_sites": changing,
                "atoms": atoms,
                "missing": [{"node_id": n, "reason": reason} for n, (m, reason) in zip(node_ids, per) if m is None],
            }
        )
        return result

    # ------------------------------------------------------------------ variants of one intermediate
    def variants(self, node_id):
        """Magnetic solutions of the accepted 02_final variants of a computed node.

        None with fewer than two variants that have moments.
        """
        variants = []
        for row in self.rows.values():
            if (
                not row.get("final_valid")
                or row.get("final_assigned_node_id") != node_id
                or row.get("energy_final_e0_eV") is None
            ):
                continue
            moments, _ = self.state_moments(row["state_id"], node_id, STAGE_FINAL, "final_topology")
            if moments is not None:
                variants.append((row["state_id"], float(row["energy_final_e0_eV"]), moments))
        if len(variants) < 2:
            return None
        return magnetic_solutions(variants, self.threshold)


def short_label(node_id):
    """Compact name of a node: 'native' for the reference, else its occupancy tag (n_0065O_36a161 -> 0065O)."""
    if node_id == "ref":
        return "native"
    core = node_id[2:] if node_id.startswith("n_") else node_id
    return core.rsplit("_", 1)[0] if "_" in core else core


def flag_text(flag, label=None):
    """One line for a spin-check flag."""
    label = label or (lambda node_id: node_id)
    where = (
        "changes sign"
        if flag["kind"] == "sign_flip"
        else "changes away from the active site"
        + (
            f" ({flag['distance_to_site_metal_A']:.1f} A from a site metal)"
            if flag.get("distance_to_site_metal_A")
            else ""
        )
    )
    return (
        f"{flag['element']} {flag['atom_id']} {where}: {flag['from_muB']:+.2f} -> {flag['to_muB']:+.2f} muB "
        f"from {label(flag['from_node'])} to {label(flag['to_node'])}"
    )


def moments_table(cycle, label=None):
    """Plain-text table of a cycle's tracked moments (rows: atoms; columns: intermediates)."""
    label = label or (lambda node_id: node_id)
    heads = [label(n) for n in cycle["node_ids"]]
    width = max([8] + [len(h) for h in heads])
    names = [f"{a['element']} {a['atom_id']} ({a['role']})" for a in cycle["atoms"]]
    first = max([4] + [len(n) for n in names])
    lines = [" " * first + "  " + "  ".join(h.rjust(width) for h in heads)]
    for name, atom in zip(names, cycle["atoms"]):
        cells = ["-" if v is None else f"{v:+.2f}" for v in atom["moments_muB"]]
        lines.append(name.ljust(first) + "  " + "  ".join(c.rjust(width) for c in cells))
    return lines
