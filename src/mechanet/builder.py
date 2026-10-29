"""Deterministic construction of intermediates, starting geometries and balanced cycles.

The builder turns one slab and its reactive sites into a reaction network:

* a **node** is a site-occupancy map (each reactive site holds one adsorbate from the
  registry, e.g. ``{"a0065": "OOH"}``) plus optional O-O couplings between sites;
* an **edge** joins two nodes with a balanced *reservoir vector* (``mechanet.reactions``);
* a **cycle** is an ordered list of edges whose vectors sum to one turnover of a reaction.

Mechanism families (``mechanet.families``) decide which nodes, edges and cycles to write;
this module supplies the machinery they share. These are initial minimum guesses and
balanced thermodynamic connections, not identified minima or verified elementary paths.

Placement policy
----------------
* New H, O-O, O-Cl and water orientations are chosen by a scored grid search around each
  reacting site. Candidates must respect non-bonded distance floors
  (``geometry.clash_floors_A``) and a minimum M-O-X angle; among feasible candidates the
  builder keeps a small, geometrically diverse set, preferring clearance and H-bond donation.
* Variants that differ only by the labels of identical atoms are merged.
* Lattice translations of the seed (with its masks and any explicit MAGMOM map) are
  detected within ``geometry.symmetry_tolerance_A``; a node that is the image of another
  node under such a translation is recorded with ``computed_by`` and no structures.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from itertools import combinations, permutations, product
from pathlib import Path

import numpy as np

from . import __version__, elements
from .fileio import dump_json, short_hash
from .plugins import default_registry
from .reactions import add as add_vectors, clean, electrons, oer_trio
from .structure import (
    Structure,
    distance_matrix,
    minimum_image,
    pbc_distances,
    read_structure,
    reorder,
    slab_normal,
    species_grouped,
    species_permutation,
    unit,
    write_poscar,
    write_xyz,
)
from .symmetry import DEFAULT_SYMMETRY_TOLERANCE_A, lattice_translations

DEFAULT_FLOORS = {
    "H-H": 1.5,
    "O-H": 1.5,
    "O-O": 2.2,
    "M-H": 1.8,
    "M-O": 1.6,
    "S-any": 2.0,
    "X-any": 2.0,
    "default": 1.35,
}
FLOOR_ALIASES = {"K-any": "S-any"}
DEFAULT_ANCHOR_ANGLE_DEG = 100.0
# Tolerance below the best M-O-X angle a site geometry allows (see Builder._anchor_angle).
ANCHOR_ANGLE_MARGIN_DEG = 5.0
DEFAULT_SEARCH = {
    "azimuth_step_deg": 15,
    "OH_tilts_deg": [20, 35, 50],
    "OO_tilts_deg": [35, 55, 70],
    "OOH_H_torsion_step_deg": 30,
    "H2O_tilts_deg": [0, 25, 50],
    "H2O_roll_step_deg": 15,
    "variants_per_site": 2,
    "hbond_bonus": 0.15,
    "clearance_cap_A": 0.6,
    "beam_width": 6,
    "diversity_min_displacement_A": 0.75,
}
COUPLED_LIFT_SCALES = (1.0, 1.5, 2.0)
SIDE_ON_METAL_O_A = 1.95
OPEN_SITE_REFERENCE_A = 1.75
TERMINAL_KINDS = ("terminal", "open_metal")
BRIDGE_KINDS = ("bridge", "mu3", "framework")
SITE_KINDS = TERMINAL_KINDS + BRIDGE_KINDS
# Intended covalent bonds among adsorbate-type atoms in a starting structure.
BASE_BOND_CUTOFFS = {frozenset({"O", "H"}): 1.25, frozenset({"O"}): 1.85, frozenset({"O", "Cl"}): 2.05}
NATIVE_ID = re.compile(r"a\d{4}$")
SITE_LABEL = re.compile(r"[am]\d{4}")


def outward_direction(rays, normal, min_norm=0.25):
    """Direction away from the surface at a ligand site, from its metal-to-ligand unit ``rays``.

    The surface normal is used when the site has no metal neighbours, when the rays nearly
    cancel (an O bridging two metals in their plane, as in a CoO2-terminated perovskite: an
    M-O-M angle above about 166 degrees leaves a sum shorter than ``min_norm``), and when
    blending a sideways direction with the normal leaves nothing. A direction less than
    about 75 degrees from the normal is blended with it.
    """
    normal = np.asarray(normal, dtype=float)
    total = np.sum(rays, axis=0) if len(rays) else np.zeros(3)
    if np.linalg.norm(total) < min_norm:
        return normal.copy()
    direction = unit(total)
    if direction @ normal < 0.25:
        blended = direction + normal
        direction = unit(blended) if np.linalg.norm(blended) >= min_norm else normal.copy()
    return direction


def site_class(kind):
    """'terminal' for single-metal sites (incl. open metal sites), 'bridge' for the others."""
    return "terminal" if kind in TERMINAL_KINDS else "bridge"


def edge_id(start, end, vector):
    """Deterministic edge ID; OER-only edges are keyed by their (water, pairs, oxygen) counts."""
    trio = oer_trio(vector)
    key = (start, end, *trio) if trio is not None else (start, end, sorted(vector.items()))
    return "e_" + short_hash(key)


def _neighbors(structure, i, allowed, cutoff):
    """Neighbours of atom i among ``allowed`` elements; ``cutoff`` is a number or element -> number."""
    out = []
    for j in range(len(structure)):
        if j == i or structure.symbols[j] not in allowed:
            continue
        limit = cutoff[structure.symbols[j]] if isinstance(cutoff, dict) else cutoff
        d = float(np.linalg.norm(minimum_image(structure.positions[j] - structure.positions[i], structure.cell)))
        if d < limit:
            out.append({"id": structure.ids[j], "index": j + 1, "element": structure.symbols[j], "distance_A": d})
    return out


def _two_site_event_orders(a, b):
    """All six interleavings of two ordered events on each of two sites."""
    events = [(a, 0), (a, 1), (b, 0), (b, 1)]
    return [
        order
        for order in permutations(events)
        if order.index((a, 0)) < order.index((a, 1)) and order.index((b, 0)) < order.index((b, 1))
    ]


class Builder:
    """Network and starting-structure builder; it is also the context handed to families."""

    def __init__(self, config, output, base_dir=".", registry=None):
        self.config = config
        self.registry = registry or default_registry()
        self.out = Path(output).resolve()
        self.base_dir = Path(base_dir)
        inp = config["inputs"]
        self.input_paths = {
            k: (Path(v) if Path(v).is_absolute() else self.base_dir / v).resolve()
            for k, v in inp.items()
            if isinstance(v, str)
        }
        self.seed = read_structure(self.input_paths["seed"])
        if not species_grouped(self.seed.symbols):
            raise ValueError(
                f"The slab file {self.input_paths['seed']} lists an element in more than one block; the "
                "campaign's structure files need one block per element. Run `mechanet init` on the slab file in "
                "a new folder: it writes a copy grouped by element and proposes the sites on it. Or group the "
                "atoms yourself (ASE: atoms[atoms.numbers.argsort(kind='stable')]). Either way the atom IDs follow "
                "the new order: site IDs written for the old order point to other atoms"
            )
        self.slab_cfg = config.get("slab", {})
        self.geo = config.get("geometry", {})
        self.scope = config.get("scope", {})
        self.roles = elements.roles_for(self.seed.symbols, self.slab_cfg.get("roles"))
        self.metal_elements = {e for e, r in self.roles.items() if r == "metal"}
        self.species_order = list(self.slab_cfg.get("species_order") or self.seed.species_order)
        self.reaction = self.registry.reaction(config.get("reaction", "oer"))
        self.vacuum_axis = int(self.slab_cfg.get("vacuum_axis", 0))
        self.surface_sign = int(self.slab_cfg.get("surface_sign", 1))
        self.normal = slab_normal(self.seed.cell, self.vacuum_axis, self.surface_sign)
        self.inplane_axes = [i for i in range(3) if i != self.vacuum_axis]
        self.symmetry_tolerance_A = float(self.geo.get("symmetry_tolerance_A", DEFAULT_SYMMETRY_TOLERANCE_A))
        self.seed_index = {aid: i for i, aid in enumerate(self.seed.ids)}
        self._apply_fixed_region()
        self.sites = self._site_records(config.get("sites") or [])
        if not self.sites:
            raise ValueError("No reactive sites are configured; run `mechanet init` or list sites in project.yaml")
        self.freq_mobile, self.frequency_source = self._frequency_mask()
        for aid, site in self.sites.items():
            if site["kind"] != "framework":
                continue
            # An explicitly requested framework site changes the active region in EVERY state,
            # including the reference, for comparable partial Hessians.
            i = self.seed_index[aid]
            self.seed.mobile[i] = True
            self.freq_mobile[i] = True
            for h in site["hydrogen_neighbors"]:
                j = self.seed_index[h["id"]]
                self.seed.mobile[j] = True
                self.freq_mobile[j] = True
        self.native = {aid: s["native"] for aid, s in self.sites.items()}
        # Metals coordinated to each reactive site: an atom X bonded to the site atom must keep an
        # M-O-X angle >= geometry.anchor_min_M_O_X_angle_deg, enforced as the equivalent M...X
        # distance floor; side-on OO and coupled O/O are exempt.
        self.anchor_metals = {aid: [m["id"] for m in s["metal_neighbors"]] for aid, s in self.sites.items()}
        self.anchor_angle_deg = float(self.geo.get("anchor_min_M_O_X_angle_deg", DEFAULT_ANCHOR_ANGLE_DEG))
        self._anchor_angles = {}
        self.inv_cell = np.linalg.inv(self.seed.cell)
        self.seed_distances = distance_matrix(self.seed)
        self.base_bonds = []
        for i, sym in enumerate(self.seed.symbols):
            for j in range(i):
                cutoff = BASE_BOND_CUTOFFS.get(frozenset((sym, self.seed.symbols[j])))
                if cutoff and self.seed_distances[i, j] <= cutoff:
                    self.base_bonds.append([self.seed.ids[i], self.seed.ids[j]])
        self.floor_table, self.floors = self._floor_table(self.geo.get("clash_floors_A"))
        search_in = self.geo.get("orientation_search", {})
        self.ignored_search_keys = [
            f"orientation_search.{k}" for k in search_in if k not in DEFAULT_SEARCH and k != "note"
        ]
        self.search_cfg = {**DEFAULT_SEARCH, **{k: v for k, v in search_in.items() if k in DEFAULT_SEARCH}}
        step = float(self.search_cfg["azimuth_step_deg"])
        if step <= 0 or int(self.search_cfg["variants_per_site"]) < 1:
            raise ValueError("orientation_search needs a positive azimuth step and variants_per_site >= 1")
        self.azimuths = [float(a) for a in np.arange(0.0, 360.0, step)]
        # Strong, low-barrier H-bonds of the seed (a proton shared between two O).
        shared_max = float(config.get("topology", {}).get("shared_proton_max_OH_A", 1.6))
        oxygen = [i for i, s in enumerate(self.seed.symbols) if s == "O"]
        self.shared_protons = []
        for i, sym in enumerate(self.seed.symbols):
            if sym != "H":
                continue
            close = sorted((self.seed_distances[i, j], j) for j in oxygen if self.seed_distances[i, j] <= shared_max)
            if len(close) >= 2 and close[0][0] <= 1.25:
                self.shared_protons.append(
                    {
                        "H": self.seed.ids[i],
                        "donor": self.seed.ids[close[0][1]],
                        "acceptor": self.seed.ids[close[1][1]],
                        "donor_distance_A": float(close[0][0]),
                        "acceptor_distance_A": float(close[1][0]),
                    }
                )
        self.translations = []
        if self.geo.get("symmetry_reduction", True):
            magmom = config.get("vasp", {}).get("magmom_by_id") or None
            values = [magmom.get(aid) for aid in self.seed.ids] if magmom else None
            for t in lattice_translations(
                self.seed, masks=(self.seed.mobile, self.freq_mobile), atom_values=values, tol=self.symmetry_tolerance_A
            ):
                idmap = {self.seed.ids[i]: self.seed.ids[j] for i, j in enumerate(t["permutation"])}
                site_map = self._site_map(idmap)
                closed = all(
                    site_map.get(s) in self.sites
                    and self.native[site_map[s]] == self.native[s]
                    and self.sites[site_map[s]]["kind"] == self.sites[s]["kind"]
                    for s in self.sites
                )
                # Explicit moments on adsorbate role IDs (e.g. d_a0067) must also map onto equal values.
                label_map = {**idmap, **site_map}
                if closed and magmom:
                    closed = all(
                        magmom.get(self._map_label(k, label_map)) == v
                        for k, v in magmom.items()
                        if k not in self.seed_index
                    )
                if closed:
                    self.translations.append(
                        {
                            "fractional": t["fractional"],
                            "id_map": idmap,
                            "site_map": site_map,
                            "label_map": label_map,
                            "max_residual_A": t["max_residual_A"],
                            "rms_residual_A": t["rms_residual_A"],
                        }
                    )
        self.nodes, self.edges, self.cycles = {}, {}, []
        self.recipes, self.node_keys, self.rejected = {}, {}, []
        self.states, self.signature_to_state = {}, {}
        self.state_structures = {}
        self.motifs = []
        self.attempted = 0
        self.duplicates_merged = []
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "states").mkdir(exist_ok=True)
        (self.out / "inputs").mkdir(exist_ok=True)
        self.reference = self.node({}, "reference", "whole_surface")

    # ------------------------------------------------------------------ set-up
    def _apply_fixed_region(self):
        spec = self.slab_cfg.get("fixed") or {"source": "file"}
        if spec.get("source", "file") == "file":
            self.fixed_source = "file"
            return
        ids = spec.get("fixed_ids")
        if ids is None:
            raise ValueError("slab.fixed needs fixed_ids (written by `mechanet init`) or source: file")
        unknown = sorted(set(ids) - set(self.seed_index))
        if unknown:
            raise ValueError(f"slab.fixed.fixed_ids names unknown atoms {unknown[:5]}")
        fixed = set(ids)
        self.seed.mobile = np.array([[aid not in fixed] * 3 for aid in self.seed.ids])
        self.fixed_source = "fixed_ids"

    def _site_records(self, entries):
        metal_cutoffs = {}

        def cutoff_for(ligand):
            if ligand not in metal_cutoffs:
                metal_cutoffs[ligand] = {m: elements.metal_ligand_cutoff(m, ligand) for m in self.metal_elements}
            return metal_cutoffs[ligand]

        sites = {}
        for entry in entries:
            sid, kind = str(entry["id"]), str(entry["kind"])
            native = str(entry.get("native", "empty" if kind == "open_metal" else "O"))
            if kind not in SITE_KINDS:
                raise ValueError(f"Site {sid}: kind must be one of {SITE_KINDS}, not {kind!r}")
            if sid in sites:
                raise ValueError(f"Site {sid} is listed twice")
            self.registry.adsorbate(native)
            if kind == "open_metal":
                metal_id = str(entry.get("metal") or ("a" + sid[1:]))
                if metal_id not in self.seed_index:
                    raise ValueError(f"Open metal site {sid}: unknown metal atom {metal_id}")
                mi = self.seed_index[metal_id]
                if self.seed.symbols[mi] not in self.metal_elements:
                    raise ValueError(f"Open metal site {sid}: {metal_id} is {self.seed.symbols[mi]}, not a metal")
                if native != "empty":
                    raise ValueError(
                        f"Open metal site {sid}: native must be 'empty' (adsorbates on a bare metal are "
                        "built as nodes, not as the reference)"
                    )
                if entry.get("direction"):
                    direction = unit(np.asarray(entry["direction"], dtype=float))
                else:
                    from .inspector import vacancy_direction

                    ligands = [
                        j
                        for j in range(len(self.seed))
                        if self.roles[self.seed.symbols[j]] == "ligand"
                        and j != mi
                        and float(
                            np.linalg.norm(
                                minimum_image(self.seed.positions[j] - self.seed.positions[mi], self.seed.cell)
                            )
                        )
                        < elements.metal_ligand_cutoff(self.seed.symbols[mi], self.seed.symbols[j])
                    ]
                    direction = vacancy_direction(self.seed, mi, ligands, self.normal)
                position = self.seed.positions[mi] + OPEN_SITE_REFERENCE_A * direction
                sites[sid] = {
                    "id": sid,
                    "index": None,
                    "element": None,
                    "kind": kind,
                    "native": native,
                    "metal_id": metal_id,
                    "metal_neighbors": [
                        {
                            "id": metal_id,
                            "index": mi + 1,
                            "element": self.seed.symbols[mi],
                            "distance_A": OPEN_SITE_REFERENCE_A,
                        }
                    ],
                    "hydrogen_neighbors": [],
                    "position_A": position.tolist(),
                    "direction": direction.tolist(),
                }
                continue
            if sid not in self.seed_index:
                raise ValueError(f"Site {sid}: no atom with this ID in the slab")
            i = self.seed_index[sid]
            element = self.seed.symbols[i]
            if element != "O":
                raise ValueError(f"Site {sid} is {element}; the built-in adsorbates transform O ligand sites only")
            metals = _neighbors(self.seed, i, self.metal_elements, cutoff_for(element))
            hydrogens = _neighbors(self.seed, i, {"H"}, elements.OH_BOND_MAX_A)
            protonation = {0: "O", 1: "OH", 2: "H2O"}.get(len(hydrogens))
            if protonation != native:
                raise ValueError(f"Site {sid}: native state {native!r} disagrees with the slab ({len(hydrogens)} H)")
            expected = {"terminal": 1, "bridge": 2}.get(kind)
            if expected is not None and len(metals) != expected:
                raise ValueError(f"Site {sid}: kind {kind!r} needs {expected} metal neighbour(s), found {len(metals)}")
            if kind == "mu3" and len(metals) < 3:
                raise ValueError(f"Site {sid}: kind 'mu3' needs at least 3 metal neighbours, found {len(metals)}")
            sites[sid] = {
                "id": sid,
                "index": i + 1,
                "element": element,
                "kind": kind,
                "native": native,
                "protonation": native,
                "metal_neighbors": metals,
                "hydrogen_neighbors": hydrogens,
                "position_A": self.seed.positions[i].tolist(),
            }
        return sites

    def _frequency_mask(self):
        """Frequency (partial-Hessian) region of the reference: a mask file, explicit IDs or the sites."""
        path = self.input_paths.get("frequency_mask")
        if path is not None:
            freq = read_structure(path)
            if freq.symbols != self.seed.symbols or not np.allclose(freq.cell, self.seed.cell, atol=1e-8):
                raise ValueError("The frequency mask must have the same cell, element order and atom count as the seed")
            displacement = np.array(
                [minimum_image(b - a, self.seed.cell) for a, b in zip(self.seed.positions, freq.positions)]
            )
            self.frequency_mask_max_difference_A = float(np.linalg.norm(displacement, axis=1).max())
            if self.frequency_mask_max_difference_A > 0.15:
                raise ValueError("Frequency-mask coordinates differ by >0.15 A; atom correspondence needs review")
            return freq.mobile.copy(), "file"
        ids = self.slab_cfg.get("frequency_atoms")
        source = "frequency_atoms"
        if ids is None:
            ids = set()
            for aid, site in self.sites.items():
                if site["kind"] == "open_metal":
                    continue
                ids.add(aid)
                ids.update(h["id"] for h in site["hydrogen_neighbors"])
            source = "sites"
        unknown = sorted(set(ids) - set(self.seed_index))
        if unknown:
            raise ValueError(f"slab.frequency_atoms names unknown atoms {unknown[:5]}")
        wanted = set(ids)
        return np.array([[aid in wanted] * 3 for aid in self.seed.ids]), source

    def _site_map(self, idmap):
        """Site ID -> image site ID under an atom permutation (open sites follow their metal)."""
        out = {}
        for sid, site in self.sites.items():
            if site["kind"] == "open_metal":
                out[sid] = "m" + idmap[site["metal_id"]][1:]
            else:
                out[sid] = idmap[sid]
        return out

    def _floor_table(self, floors):
        given = {
            FLOOR_ALIASES.get(k, k): float(v)
            for k, v in (floors or {}).items()
            if FLOOR_ALIASES.get(k, k) in DEFAULT_FLOORS
        }
        floors = {**DEFAULT_FLOORS, **given}
        table = np.full((6, 6), floors["default"])

        def put(a, b, value):
            table[a, b] = table[b, a] = value

        put(0, 0, floors["H-H"])
        put(0, 1, floors["O-H"])
        put(1, 1, floors["O-O"])
        put(2, 0, floors["M-H"])
        put(2, 1, floors["M-O"])
        for c in range(6):
            put(3, c, floors["S-any"])
        for c in range(6):
            put(5, c, floors["X-any"])
        return table, floors

    def _category(self, symbol):
        """Floor category: 0 H, 1 O, 2 metal, 3 spectator cation, 4 other, 5 halogen."""
        if symbol == "H":
            return 0
        if symbol == "O":
            return 1
        role = self.roles.get(symbol) or elements.default_role(symbol)
        if role == "metal":
            return 2
        if role == "spectator":
            return 3
        if symbol in elements.HALOGENS:
            return 5
        return 4

    def adsorbate(self, status):
        return self.registry.adsorbate(status)

    def site_position(self, aid):
        site = self.sites[aid]
        if site["kind"] == "open_metal":
            return np.asarray(site["position_A"], dtype=float)
        return self.seed.positions[self.seed_index[aid]]

    # ------------------------------------------------------------------ geometry
    def _site_frame(self, aid):
        s = self.sites[aid]
        origin = self.site_position(aid)
        rays = [
            -unit(minimum_image(self.seed.positions[self.seed_index[m["id"]]] - origin, self.seed.cell))
            for m in s["metal_neighbors"]
        ]
        outward = outward_direction(rays, self.normal)
        reference = self.seed.cell[self.inplane_axes[0]]
        t = reference - np.dot(reference, outward) * outward
        if np.linalg.norm(t) < 1e-8:
            reference = self.seed.cell[self.inplane_axes[1]]
            t = reference - np.dot(reference, outward) * outward
        t = unit(t)
        return outward, t, unit(np.cross(outward, t))

    @staticmethod
    def _direction(outward, t, v, tilt_deg, azimuth_deg):
        tilt, az = math.radians(tilt_deg), math.radians(azimuth_deg)
        lateral = math.cos(az) * t + math.sin(az) * v
        return unit(math.cos(tilt) * outward + math.sin(tilt) * lateral)

    def _record(self, symbol, position, mobile=None, frequency=None):
        return {
            "symbol": symbol,
            "position": np.asarray(position, dtype=float),
            "mobile": np.ones(3, dtype=bool) if mobile is None else mobile,
            "frequency": np.ones(3, dtype=bool) if frequency is None else frequency,
        }

    def _base(self, occupancy, coupling, lift_scale=1.0):
        """Apply removals/deprotonations/new proximal atoms/coupling; return sites still needing orientation."""
        records = {
            aid: self._record(sy, p.copy(), flags.copy(), ff.copy())
            for aid, sy, p, flags, ff in zip(
                self.seed.ids, self.seed.symbols, self.seed.positions, self.seed.mobile, self.freq_mobile
            )
        }
        bonds = [b.copy() for b in self.base_bonds]
        oriented, repositioned, localized = [], set(), []
        changed_sites = set()
        for aid, status in sorted(occupancy.items()):
            if aid not in self.sites:
                raise ValueError(f"Unknown reactive site {aid}")
            ads = self.adsorbate(status)
            # Exact native state is kept without perturbation.
            if status == self.native[aid]:
                continue
            changed_sites.add(aid)
            site = self.sites[aid]
            for hid in (h["id"] for h in site["hydrogen_neighbors"]):
                records.pop(hid, None)
            if status == "empty":
                records.pop(aid, None)
                continue
            if aid in records:
                records[aid]["mobile"] = np.ones(3, dtype=bool)
                records[aid]["frequency"] = np.ones(3, dtype=bool)
                if records[aid]["symbol"] != ads.proximal:
                    records[aid]["symbol"] = ads.proximal
            else:
                metal = self.seed.positions[self.seed_index[site["metal_id"]]]
                records[aid] = self._record(ads.proximal, metal + ads.metal_distance_A * np.asarray(site["direction"]))
            if ads.oriented:
                oriented.append(aid)
        if coupling:
            changed_sites.update(coupling)
        # A strongly shared proton whose acceptor oxygen is modified is localized
        # on its (unchanged) donor at a normal O-H length; the relaxation decides.
        for sp in self.shared_protons:
            hid, donor, acceptor = sp["H"], sp["donor"], sp["acceptor"]
            if hid in records and donor in records and acceptor in changed_sites and donor not in changed_sites:
                vec = minimum_image(records[hid]["position"] - records[donor]["position"], self.seed.cell)
                records[hid]["position"] = records[donor]["position"] + 0.98 * unit(vec)
                localized.append(hid)
        if coupling:
            a, b = coupling
            pa, pb = records[a]["position"], records[b]["position"]
            vec = minimum_image(pb - pa, self.seed.cell)
            center = pa + 0.5 * vec
            # A bidentate peroxo-like starting guess. This may relax away.
            axis = unit(vec)
            lift = float(self.geo.get("coupled_pair_lift_A", 0.50)) * float(lift_scale)
            metals = [p for p, sy in zip(self.seed.positions, self.seed.symbols) if sy in self.metal_elements]
            for _ in range(21):
                trial_a = center - 0.725 * axis + lift * self.normal
                trial_b = center + 0.725 * axis + lift * self.normal
                closest = min(
                    np.linalg.norm(minimum_image(p - q, self.seed.cell)) for q in (trial_a, trial_b) for p in metals
                )
                if closest >= 1.65:
                    break
                lift += 0.05
            records[a]["position"], records[b]["position"] = trial_a, trial_b
            for aid in (a, b):
                records[aid]["mobile"] = np.ones(3, dtype=bool)
                records[aid]["frequency"] = np.ones(3, dtype=bool)
                repositioned.add(aid)
            bonds.append([a, b])
        return records, bonds, oriented, repositioned, localized

    def _site_grid(self, aid, status, binding_modes):
        """Every orientation parameter set searched for one reacting site."""
        cfg = self.search_cfg
        ads = self.adsorbate(status)
        if ads.geometry == "XH":
            return [{"tilt_deg": float(t), "azimuth_deg": a} for t in cfg[ads.tilts] for a in self.azimuths]
        if ads.geometry == "water":
            rolls = [float(r) for r in np.arange(0.0, 180.0, float(cfg["H2O_roll_step_deg"]))]
            grid = []
            for tilt in cfg[ads.tilts]:
                for az in [0.0] if float(tilt) == 0.0 else self.azimuths:
                    grid += [{"tilt_deg": float(tilt), "azimuth_deg": az, "roll_deg": r} for r in rolls]
            return grid
        if ads.geometry == "XY" and binding_modes.get(aid) == "side_on":
            return [{"binding": "side_on", "azimuth_deg": a} for a in self.azimuths if a < 180.0]
        if ads.geometry == "XY":
            return [{"tilt_deg": float(t), "azimuth_deg": a} for t in cfg[ads.tilts] for a in self.azimuths]
        if ads.geometry == "XYH":
            torsions = [float(x) for x in np.arange(0.0, 360.0, float(cfg["OOH_H_torsion_step_deg"]))]
            return [
                {"tilt_deg": float(t), "azimuth_deg": a, "torsion_deg": x}
                for t in cfg[ads.tilts]
                for a in self.azimuths
                for x in torsions
            ]
        raise ValueError(f"No orientation grid for {status}")

    def _site_atoms(self, aid, status, params, binding_modes, records):
        """Atoms placed for one site: [(id, symbol, position, bonded_parent_id or None)]."""
        outward, t, v = self._site_frame(aid)
        origin = records[aid]["position"]
        ads = self.adsorbate(status)
        initial_h = [h["id"] for h in self.sites[aid]["hydrogen_neighbors"]]
        if ads.geometry == "XH":
            d = self._direction(outward, t, v, params["tilt_deg"], params["azimuth_deg"])
            return [(f"h_{aid}", "H", origin + ads.xh_A * d, aid)]
        if ads.geometry == "water":
            # Molecular water occupying this site; a native H role is kept.
            b = self._direction(outward, t, v, params["tilt_deg"], params["azimuth_deg"])
            e1 = t - np.dot(t, b) * b
            if np.linalg.norm(e1) < 1e-6:
                e1 = v - np.dot(v, b) * b
            e1 = unit(e1)
            e2 = unit(np.cross(b, e1))
            roll = math.radians(params["roll_deg"])
            u = math.cos(roll) * e1 + math.sin(roll) * e2
            half = math.radians(104.5 / 2)
            hids = [initial_h[0] if initial_h else f"h_{aid}", f"w_{aid}"]
            return [
                (hids[0], "H", origin + ads.xh_A * (math.cos(half) * b + math.sin(half) * u), aid),
                (hids[1], "H", origin + ads.xh_A * (math.cos(half) * b - math.sin(half) * u), aid),
            ]
        if ads.geometry == "XY" and params.get("binding") == "side_on":
            site = self.sites[aid]
            if site["kind"] not in TERMINAL_KINDS or len(site["metal_neighbors"]) != 1:
                raise ValueError("Side-on binding requires one single-metal anchor")
            metal_id = site["metal_neighbors"][0]["id"]
            metal = self.seed.positions[self.seed_index[metal_id]]
            metal = origin + minimum_image(metal - origin, self.seed.cell)
            az = math.radians(params["azimuth_deg"])
            lateral = math.cos(az) * t + math.sin(az) * v
            half = ads.bond_A / 2
            # Two equivalent M-O distances of 1.95 A.
            center = metal + math.sqrt(SIDE_ON_METAL_O_A**2 - half**2) * outward
            return [
                (aid, ads.proximal, center - half * lateral, None),
                (f"d_{aid}", ads.distal, center + half * lateral, aid),
            ]
        if ads.geometry in ("XY", "XYH"):
            d = self._direction(outward, t, v, params["tilt_deg"], params["azimuth_deg"])
            distal = origin + ads.bond_A * d
            atoms = [(f"d_{aid}", ads.distal, distal, aid)]
            if ads.geometry == "XYH":
                # proximal--distal--H = 105 degrees; rotation about the bond sampled by torsion.
                q = outward - np.dot(outward, d) * d
                q = unit(q if np.linalg.norm(q) > 1e-6 else t - np.dot(t, d) * d)
                q2 = unit(np.cross(d, q))
                phi = math.radians(params["torsion_deg"])
                hd = math.cos(math.radians(75)) * d + math.sin(math.radians(75)) * (
                    math.cos(phi) * q + math.sin(phi) * q2
                )
                atoms.append((f"p_{aid}", "H", distal + ads.xh_A * hd, f"d_{aid}"))
            return atoms
        raise ValueError(f"Unsupported oriented status {status}")

    def _environment(self, records, exclude, added):
        ids, symbols, positions = [], [], []
        for aid, rec in records.items():
            if aid in exclude or aid in added:
                continue
            ids.append(aid)
            symbols.append(rec["symbol"])
            positions.append(rec["position"])
        for aid, (symbol, position) in added.items():
            ids.append(aid)
            symbols.append(symbol)
            positions.append(position)
        return {
            "ids": ids,
            "symbols": symbols,
            "pos": np.array(positions),
            "cat": np.array([self._category(s) for s in symbols]),
            "index": {aid: k for k, aid in enumerate(ids)},
        }

    def _score(self, candidates, env):
        """Clearance (A above the floor), H-bond count and worst contact per candidate."""
        rows, row_cand = [], []
        side_on = [params.get("binding") == "side_on" for params, _ in candidates]
        for c, (_, atoms) in enumerate(candidates):
            local = {a[0]: a for a in atoms}
            for aid, sym, pos, parent in atoms:
                grand = local[parent][3] if parent in local else None
                rows.append((aid, sym, pos, parent, grand, local))
                row_cand.append(c)
        positions = np.array([r[2] for r in rows])
        dist = pbc_distances(positions, env["pos"], self.seed.cell, self.inv_cell)
        floors = self.floor_table[np.array([self._category(r[1]) for r in rows])[:, None], env["cat"][None, :]]
        margin = dist - floors
        for k, (_aid, _sym, pos, parent, grand, _local) in enumerate(rows):
            for partner in (parent, grand):
                if partner is not None and partner in env["index"]:
                    margin[k, env["index"][partner]] = np.inf
            if parent in self.anchor_metals and parent in env["index"] and not side_on[row_cand[k]]:
                parent_pos = env["pos"][env["index"][parent]]
                r_ox = float(np.linalg.norm(minimum_image(pos - parent_pos, self.seed.cell)))
                for metal in self.anchor_metals[parent]:
                    q = env["index"].get(metal)
                    if q is not None:
                        r_mo = float(np.linalg.norm(minimum_image(env["pos"][q] - parent_pos, self.seed.cell)))
                        floors[k, q] = max(floors[k, q], self._anchor_floor(r_mo, r_ox, self._anchor_angle(parent)))
                        margin[k, q] = dist[k, q] - floors[k, q]
        row_clear = margin.min(axis=1) if margin.shape[1] else np.full(len(rows), np.inf)
        row_worst = margin.argmin(axis=1) if margin.shape[1] else np.zeros(len(rows), dtype=int)
        n = len(candidates)
        clearance = np.full(n, np.inf)
        worst = [None] * n
        hbonds = np.zeros(n, dtype=int)
        oxygen_env = np.where(env["cat"] == 1)[0]
        for k, (aid, sym, pos, parent, grand, local) in enumerate(rows):
            c = row_cand[k]
            if row_clear[k] < clearance[c]:
                clearance[c] = row_clear[k]
                q = int(row_worst[k])
                worst[c] = {
                    "a": aid,
                    "b": env["ids"][q],
                    "distance_A": float(dist[k, q]),
                    "minimum_A": float(floors[k, q]),
                }
            if sym != "H" or parent is None:
                continue
            parent_pos = (
                local[parent][2]
                if parent in local
                else (env["pos"][env["index"][parent]] if parent in env["index"] else None)
            )
            if parent_pos is None:
                continue
            to_parent = minimum_image(parent_pos - pos, self.seed.cell)
            for q in oxygen_env:
                if env["ids"][q] in (parent, grand) or not (1.55 <= dist[k, q] <= 2.25):
                    continue
                to_acceptor = minimum_image(env["pos"][q] - pos, self.seed.cell)
                cosine = np.dot(to_parent, to_acceptor) / (np.linalg.norm(to_parent) * np.linalg.norm(to_acceptor))
                if cosine <= math.cos(math.radians(130)):
                    hbonds[c] += 1
                    break
        cap = float(self.search_cfg["clearance_cap_A"])
        bonus = float(self.search_cfg["hbond_bonus"])
        score = np.minimum(clearance, cap) + bonus * hbonds
        return clearance, score, hbonds, worst

    def _anchor_floor(self, r_mo, r_ox, angle_deg=None):
        """M...X distance that corresponds to the minimum M-O-X angle for given bond lengths."""
        angle = math.radians(self.anchor_angle_deg if angle_deg is None else angle_deg)
        return math.sqrt(max(r_mo * r_mo + r_ox * r_ox - 2.0 * r_mo * r_ox * math.cos(angle), 0.0))

    def _anchor_angle(self, site_id):
        """Minimum M-O-X angle at a ligand site: the global floor, or less where the site cannot reach it.

        An atom on a site points best along the site's outward direction; at an O bridging two
        metals in their plane (a CoO2-terminated perovskite) that direction makes 90 degrees
        with both M-O bonds, so no orientation could meet a 100-degree floor. Such a site gets
        the best angle it allows minus ANCHOR_ANGLE_MARGIN_DEG; every other site keeps the floor.
        """
        cached = self._anchor_angles.get(site_id)
        if cached is not None:
            return cached
        angle = self.anchor_angle_deg
        site = self.sites.get(site_id)
        if site is not None and site["kind"] != "open_metal" and site["metal_neighbors"]:
            outward = self._site_frame(site_id)[0]
            origin = self.site_position(site_id)
            angles = []
            for metal in site["metal_neighbors"]:
                bond = minimum_image(self.seed.positions[self.seed_index[metal["id"]]] - origin, self.seed.cell)
                angles.append(math.degrees(math.acos(float(np.clip(unit(bond) @ outward, -1.0, 1.0)))))
            angle = min(angle, min(angles) - ANCHOR_ANGLE_MARGIN_DEG)
        self._anchor_angles[site_id] = angle
        return angle

    def _label_free_distance(self, first, second):
        """Largest nearest-same-element distance between two atom sets [(symbol, position)].

        Identical elements are interchangeable, so a water whose two H labels are
        swapped is at distance 0 from the original. Different compositions -> inf.
        """
        element_set = sorted({s for s, _ in first})
        if sorted(s for s, _ in first) != sorted(s for s, _ in second):
            return np.inf
        worst = 0.0
        for element in element_set:
            a = np.array([p for s, p in first if s == element])
            b = np.array([p for s, p in second if s == element])
            d = pbc_distances(a, b, self.seed.cell, self.inv_cell)
            worst = max(worst, float(d.min(axis=1).max()), float(d.min(axis=0).max()))
        return worst

    def _diverse(self, ranked, atoms_of, limit):
        """Greedy pick of up to ``limit`` items whose placed atoms differ substantially."""
        threshold = float(self.search_cfg["diversity_min_displacement_A"])
        picked = []
        for item in ranked:
            atoms = atoms_of(item)
            if all(self._label_free_distance(atoms, atoms_of(other)) >= threshold for other in picked):
                picked.append(item)
            if len(picked) >= limit:
                break
        return picked

    def _changed_atoms(self, st):
        """[(symbol, position)] of atoms that are new or moved relative to the seed."""
        out = []
        for aid, symbol, position in zip(st.ids, st.symbols, st.positions):
            k = self.seed_index.get(aid)
            if k is None or np.linalg.norm(minimum_image(position - self.seed.positions[k], st.cell)) > 1e-6:
                out.append((symbol, position))
        return out

    def _search(self, occupancy, coupling, binding_modes, lift_scale=1.0):
        """Beam search over the orientation grids of all reacting sites of one node.

        Sites are placed sequentially; every placement order is tried and the results are
        pooled before the final diverse selection, so an early greedy choice cannot block a
        later site.
        """
        records, bonds, oriented, repositioned, localized = self._base(occupancy, coupling, lift_scale)
        if not oriented:
            return [{"placement": {}, "score": None, "min_clearance_A": None, "hbonds": 0}], None
        per_site = int(self.search_cfg["variants_per_site"])
        cache, pooled, failures = {}, [], []
        for order in permutations(oriented):
            finals, failure = self._beam(order, occupancy, binding_modes, records, cache)
            pooled.extend(finals)
            if failure:
                failures.append(failure)
        if not pooled:
            return [], max(failures, key=lambda f: f["min_clearance_A"]) if failures else None
        unique = {}
        for item in pooled:
            key = json.dumps(item[0], sort_keys=True)
            if key not in unique or item[2] > unique[key][2]:
                unique[key] = item
        ranked = sorted(unique.values(), key=lambda x: -x[2])

        def atoms_of(item):
            return [(v[0], v[1]) for v in item[1].values()]

        chosen = self._diverse(ranked, atoms_of, per_site)
        return [{"placement": p[0], "score": p[2], "min_clearance_A": p[3], "hbonds": p[4]} for p in chosen], None

    def _beam(self, order, occupancy, binding_modes, records, cache):
        beam = int(self.search_cfg["beam_width"])
        per_site = int(self.search_cfg["variants_per_site"])
        partials = [({}, {}, 0.0, np.inf, 0)]  # placement, added atoms, score sum, min clearance, hbonds
        best_failure = None
        for aid in order:
            status = occupancy[aid]
            moves_site = self.adsorbate(status).geometry == "XY" and binding_modes.get(aid) == "side_on"
            if aid not in cache:
                grid = self._site_grid(aid, status, binding_modes)
                cache[aid] = [(p, self._site_atoms(aid, status, p, binding_modes, records)) for p in grid]
            cand = cache[aid]
            extended = []
            for placement, added, total, clear_min, hb in partials:
                env = self._environment(records, {aid} if moves_site else set(), added)
                clearance, score, hbonds, worst = self._score(cand, env)
                feasible = [k for k in np.argsort(-score, kind="stable") if clearance[k] >= 0]
                if not feasible:
                    k = int(np.argmax(clearance))
                    if best_failure is None or clearance[k] > best_failure["min_clearance_A"]:
                        best_failure = {
                            "site": aid,
                            "status": status,
                            "params": cand[k][0],
                            "min_clearance_A": float(clearance[k]),
                            "worst_contact": worst[k],
                        }
                    continue

                def atoms_of(k, cand=cand):
                    return [(a[1], a[2]) for a in cand[k][1]]

                for k in self._diverse(feasible, atoms_of, max(per_site, 3)):
                    new_added = dict(added)
                    new_added.update({a[0]: (a[1], a[2]) for a in cand[k][1]})
                    extended.append(
                        (
                            {**placement, aid: cand[k][0]},
                            new_added,
                            total + float(score[k]),
                            min(clear_min, float(clearance[k])),
                            hb + int(hbonds[k]),
                        )
                    )
            if not extended:
                return [], best_failure
            extended.sort(key=lambda x: -x[2])
            partials = extended[:beam]
        return partials, None

    def default_placement(self, occupancy, coupling, binding_modes):
        records, _, oriented, _, _ = self._base(occupancy, coupling)
        return {aid: self._site_grid(aid, occupancy[aid], binding_modes)[0] for aid in oriented}

    def structure(self, occupancy, coupling, placement=None, binding_modes=None, lift_scale=1.0):
        """(structure, frequency mask, intended bonds, info) of one placement of a node."""
        binding_modes = binding_modes or {}
        records, bonds, oriented, repositioned, localized = self._base(occupancy, coupling, lift_scale)
        if placement is None:
            placement = self.default_placement(occupancy, coupling, binding_modes)
        for aid in oriented:
            if aid not in placement:
                raise ValueError(f"Placement for {aid} is missing")
            for atom_id, symbol, position, parent in self._site_atoms(
                aid, occupancy[aid], placement[aid], binding_modes, records
            ):
                if atom_id in records:  # a repositioned site atom (side-on)
                    records[atom_id]["position"] = np.asarray(position, dtype=float)
                else:
                    records[atom_id] = self._record(symbol, position)
                if atom_id in self.seed_index:
                    repositioned.add(atom_id)
                if parent is not None:
                    bonds.append([parent, atom_id])
        ids = list(records)
        st = Structure(
            [records[x]["symbol"] for x in ids],
            np.array([records[x]["position"] for x in ids]),
            self.seed.cell.copy(),
            np.array([records[x]["mobile"] for x in ids]),
            ids,
        )
        freq = np.array([records[x]["frequency"] for x in ids])
        permutation = species_permutation(st.symbols, self.species_order)
        st = reorder(st, permutation)
        freq = freq[permutation]
        # Keep the intended chemical graph. Do not silently redefine it from a
        # bad placement where a proton is already bonded to the wrong oxygen.
        unique_bonds = sorted({tuple(sorted(b)) for b in bonds if all(aid in records for aid in b)})
        return (
            st,
            freq,
            [list(b) for b in unique_bonds],
            {"repositioned": sorted(repositioned), "localized_shared_protons": localized},
        )

    def _topology_issues(self, st, bonds, distances):
        intended = {tuple(sorted(b)) for b in bonds}
        observed = set()
        for i, sym in enumerate(st.symbols):
            for j in range(i):
                cutoff = BASE_BOND_CUTOFFS.get(frozenset((sym, st.symbols[j])))
                if cutoff and distances[i, j] <= cutoff:
                    observed.add(tuple(sorted((st.ids[i], st.ids[j]))))
        if observed != intended:
            return {
                "missing_expected_bonds": [list(b) for b in sorted(intended - observed)],
                "unexpected_short_contacts": [list(b) for b in sorted(observed - intended)],
            }
        return None

    def _clashes(self, st, distances, bonds, repositioned=(), side_on_sites=(), coupling=None):
        """Non-bonded contacts below the floors for every pair involving a new/moved atom.

        Intended bonds and 1-3 pairs across them are excluded (the topology check covers
        them). A native pair that moved only slightly (not repositioned by an orientation
        choice) may keep a contact that was already shorter in the seed, minus 0.05 A.
        """
        repositioned = set(repositioned)
        index = {aid: i for i, aid in enumerate(st.ids)}
        changed = set()
        for aid, p in zip(st.ids, st.positions):
            k = self.seed_index.get(aid)
            if k is None or np.linalg.norm(minimum_image(p - self.seed.positions[k], st.cell)) > 1e-6:
                changed.add(aid)
        neighbors = {aid: set() for aid in st.ids}
        for a, b in bonds:
            if a in neighbors and b in neighbors:
                neighbors[a].add(b)
                neighbors[b].add(a)
        cats = np.array([self._category(s) for s in st.symbols])
        # Side-on OO and coupled O/O bind or bridge metals by construction: no angle floor.
        side_on_atoms = {x for site in side_on_sites for x in (site, f"d_{site}")} | set(coupling or ())
        anchored = {}  # atom -> {metal: minimum M...X distance from the M-O-X angle floor}
        for aid in st.ids:
            if aid in side_on_atoms or aid in self.seed_index and aid not in changed:
                continue
            floors_for = {}
            for o in neighbors[aid]:
                if o not in self.anchor_metals or o not in index:
                    continue
                r_ox = float(distances[index[aid], index[o]])
                for metal in self.anchor_metals[o]:
                    if metal in index:
                        r_mo = float(distances[index[metal], index[o]])
                        floor = self._anchor_floor(r_mo, r_ox, self._anchor_angle(o))
                        floors_for[metal] = max(floors_for.get(metal, 0.0), floor)
            if floors_for:
                anchored[aid] = floors_for
        # The atom bound to an open metal site sits at its own M-X distance by construction.
        site_bonds = {
            frozenset((sid, site["metal_id"])) for sid, site in self.sites.items() if site["kind"] == "open_metal"
        }
        problems = []
        for aid in sorted(changed):
            i = index[aid]
            for j, other in enumerate(st.ids):
                if j == i or (other in changed and other < aid):
                    continue
                if other in neighbors[aid] or neighbors[aid] & neighbors[other]:
                    continue
                if frozenset((aid, other)) in site_bonds:
                    continue
                floor = float(self.floor_table[cats[i], cats[j]])
                if other in anchored.get(aid, {}):
                    floor = max(floor, anchored[aid][other])
                elif aid in anchored.get(other, {}):
                    floor = max(floor, anchored[other][aid])
                if (
                    aid in self.seed_index
                    and other in self.seed_index
                    and aid not in repositioned
                    and other not in repositioned
                ):
                    floor = min(floor, float(self.seed_distances[self.seed_index[aid], self.seed_index[other]]) - 0.05)
                d = float(distances[i, j])
                if d < floor:
                    problems.append({"a": aid, "b": other, "distance_A": d, "minimum_A": floor})
        return problems

    def _signature(self, st, freq):
        # Exact duplicate starts, not a claim to discover equivalent relaxed basins.
        frac = (st.positions @ np.linalg.inv(st.cell)) % 1.0
        return short_hash(
            {
                "symbols": st.symbols,
                "ids": st.ids,
                "fractional": np.round(frac, 9).tolist(),
                "mobile": st.mobile.tolist(),
                "frequency": freq.tolist(),
            },
            24,
        )

    # ------------------------------------------------------------------ network
    @staticmethod
    def _node_key(changes, coupling):
        return json.dumps({"occupancy": changes, "coupling": sorted(coupling) if coupling else None}, sort_keys=True)

    def delta_atoms(self, changes):
        """Composition change of a set of site changes relative to the native surface."""
        delta = {"O": 0, "H": 0}
        for aid, status in changes.items():
            new, old = self.adsorbate(status).composition, self.adsorbate(self.native[aid]).composition
            for element in set(new) | set(old):
                delta[element] = delta.get(element, 0) + new.get(element, 0) - old.get(element, 0)
        return {k: v for k, v in delta.items() if k in ("O", "H") or v}

    def _family_sources(self, family):
        if family == "reference":
            return []
        return list(self.registry.family(family).sources)

    def _representative(self, changes, coupling):
        """Lexicographically smallest image of a node under the detected translations."""
        own = self._node_key(changes, coupling)
        best = None
        for tr in self.translations:
            site_map = tr["site_map"]
            mapped = {site_map[k]: v for k, v in changes.items()}
            mapped_coupling = tuple(site_map[x] for x in coupling) if coupling else None
            key = self._node_key(mapped, mapped_coupling)
            if key < own and (best is None or key < best[0]):
                best = (key, mapped, mapped_coupling, tr)
        return best

    @staticmethod
    def _map_label(label, label_map):
        return SITE_LABEL.sub(lambda m: label_map.get(m.group(0), m.group(0)), str(label))

    def node(self, occupancy, family, site, coupling=None):
        """Node ID for an occupancy map (created with its starting structures on first use)."""
        occupancy = dict(occupancy)
        for aid, status in occupancy.items():
            if aid not in self.sites:
                raise ValueError(f"Unknown reactive site {aid}")
            self.adsorbate(status)
        changes = {k: v for k, v in occupancy.items() if v != self.native[k]}
        key = self._node_key(changes, coupling)
        sources = self._family_sources(family)
        if key in self.node_keys:
            node = self.nodes[self.node_keys[key]]
            if family not in node["families"]:
                node["families"].append(family)
            node["source_ids"] = sorted(set(node["source_ids"] + sources))
            node["aliases"].append({"family": family, "site": site, "occupancy": occupancy})
            if node.get("computed_by"):
                rep = self.nodes[node["computed_by"]]
                if family not in rep["families"]:
                    rep["families"].append(family)
                rep["source_ids"] = sorted(set(rep["source_ids"] + sources))
            return node["id"]
        if not changes and not coupling:
            node_id = "ref"
        else:
            # Ligand sites are tagged by their atom number; open metal sites keep their 'm' prefix.
            tag = "_".join(f"{x[1:] if x.startswith('a') else x}{v}" for x, v in sorted(changes.items()))
            if coupling:
                tag += "_coupled"
            node_id = "n_" + (tag or "pair") + "_" + short_hash(key, 6)
        node = {
            "id": node_id,
            "state_ids": [],
            "family": family,
            "families": [family],
            "site": site,
            "occupancy": occupancy,
            "coupling": list(coupling) if coupling else None,
            "source_ids": sources.copy(),
            "aliases": [],
            "delta_atoms": self.delta_atoms(changes),
        }
        self.nodes[node_id] = node
        self.node_keys[key] = node_id
        self.recipes[node_id] = {"occupancy": occupancy, "coupling": coupling}
        representative = self._representative(changes, coupling)
        if representative is not None:
            _, mapped, mapped_coupling, tr = representative
            rep_id = self.node(mapped, family, self._map_label(site, tr["label_map"]), mapped_coupling)
            rep = self.nodes[rep_id]
            if rep["delta_atoms"] != node["delta_atoms"]:
                raise AssertionError(f"Symmetry image {rep_id} changes composition of {node_id}")
            node.update(
                computed_by=rep_id,
                geometry_status="symmetry_equivalent",
                symmetry={
                    "translation_fractional": [round(x, 6) for x in tr["fractional"]],
                    "max_residual_A": round(tr["max_residual_A"], 5),
                    "site_map": {k: tr["site_map"][k] for k in sorted(changes) + sorted(coupling or [])},
                    "note": "Lattice-translation image of the representative node (the seed matches its "
                    "image to within max_residual_A); treated as the same calculation, not recomputed.",
                },
            )
            rep.setdefault("symmetry_twins", []).append(node_id)
            if family not in rep["families"]:
                rep["families"].append(family)
            rep["source_ids"] = sorted(set(rep["source_ids"] + node["source_ids"]))
            return node_id
        side_on_sites = sorted(
            k
            for k, v in changes.items()
            if self.adsorbate(v).side_on
            and self.sites[k]["kind"] in TERMINAL_KINDS
            and len(self.sites[k]["metal_neighbors"]) == 1
        )
        if not self.geo.get("side_on_OO", True):
            side_on_sites = []
        binding_patterns = [{}]
        for switches in product((False, True), repeat=len(side_on_sites)):
            if any(switches):
                binding_patterns.append({k: "side_on" for k, flag in zip(side_on_sites, switches) if flag})
        built = []
        for binding in binding_patterns:
            kept = 0
            for lift in COUPLED_LIFT_SCALES if coupling else (1.0,):
                if coupling and kept >= int(self.search_cfg["variants_per_site"]):
                    break
                choices, failure = self._search(occupancy, coupling, binding, lift)
                if not choices:
                    self.rejected.append(
                        {
                            "node_id": node_id,
                            "binding_modes": binding,
                            "lift_scale": lift,
                            "reason": "no_orientation_meets_distance_floors",
                            "best_infeasible_candidate": failure,
                            "chemical_candidate_rejected": False,
                        }
                    )
                    continue
                for choice in choices:
                    if coupling and kept >= int(self.search_cfg["variants_per_site"]):
                        break
                    self.attempted += 1
                    st, freq, bonds, info = self.structure(occupancy, coupling, choice["placement"], binding, lift)
                    distances = distance_matrix(st)
                    clash = self._clashes(st, distances, bonds, info["repositioned"], sorted(binding), coupling)
                    topology_issue = self._topology_issues(st, bonds, distances)
                    if clash or topology_issue:
                        self.rejected.append(
                            {
                                "node_id": node_id,
                                "binding_modes": binding,
                                "lift_scale": lift,
                                "placement": choice["placement"],
                                "reason": "initial_topology_mismatch" if topology_issue else "initial_geometry_clash",
                                "clashes": clash,
                                "topology_issue": topology_issue,
                                "chemical_candidate_rejected": False,
                            }
                        )
                        continue
                    threshold = float(self.search_cfg["diversity_min_displacement_A"])
                    moved = self._changed_atoms(st)
                    duplicate = next(
                        (
                            (sid, d)
                            for sid, other in built
                            for d in [self._label_free_distance(moved, self._changed_atoms(other))]
                            if d < threshold
                        ),
                        None,
                    )
                    if duplicate:
                        self.duplicates_merged.append(
                            {
                                "node_id": node_id,
                                "kept": duplicate[0],
                                "placement": choice["placement"],
                                "coupled_lift_scale": lift if coupling else None,
                                "label_free_distance_A": round(duplicate[1], 4),
                                "reason": f"new/moved atoms within {threshold:g} A of a kept variant, "
                                "ignoring labels of identical atoms",
                            }
                        )
                        continue
                    signature = self._signature(st, freq)
                    if signature in self.signature_to_state:
                        state_id = self.signature_to_state[signature]
                        meta = self.states[state_id]
                        if node_id not in meta["node_aliases"]:
                            meta["node_aliases"].append(node_id)
                        if state_id not in node["state_ids"]:
                            node["state_ids"].append(state_id)
                        continue
                    state_id = node_id if node_id == "ref" else f"{node_id}__v{len(built) + 1}"
                    self._write_state(
                        state_id,
                        node,
                        family,
                        site,
                        occupancy,
                        coupling,
                        binding,
                        lift,
                        choice,
                        st,
                        freq,
                        bonds,
                        info,
                        distances,
                        signature,
                    )
                    built.append((state_id, st))
                    kept += 1
        delta_seen = {json.dumps(self.states[s]["delta_atoms"], sort_keys=True) for s in node["state_ids"]}
        if delta_seen and delta_seen != {json.dumps(node["delta_atoms"], sort_keys=True)}:
            raise AssertionError(f"Built composition differs from occupancy bookkeeping for {node_id}")
        node["geometry_status"] = "generated" if node["state_ids"] else "all_initial_placements_rejected"
        return node_id

    def _write_state(
        self,
        state_id,
        node,
        family,
        site,
        occupancy,
        coupling,
        binding,
        lift,
        choice,
        st,
        freq,
        bonds,
        info,
        distances,
        signature,
    ):
        tracked = sorted(set(node["delta_atoms"]) | {"O", "H"})
        delta = {s: st.symbols.count(s) - self.seed.symbols.count(s) for s in tracked}
        delta = {k: v for k, v in delta.items() if k in ("O", "H") or v}
        meta = {
            "schema_version": 3,
            "id": state_id,
            "node_id": node["id"],
            "node_aliases": [node["id"]],
            "family": family,
            "site": site,
            "variant": {
                "index": len(node["state_ids"]) + 1,
                "placement": choice["placement"],
                "binding_modes": binding,
                "coupled_lift_scale": lift if coupling else None,
                "search_score": choice["score"],
                "min_clearance_above_floor_A": choice["min_clearance_A"],
                "donated_hbonds": choice["hbonds"],
                "localized_shared_protons": info["localized_shared_protons"],
                "repositioned_native_ids": info["repositioned"],
            },
            "atom_ids": st.ids,
            "symbols": st.symbols,
            "relax_mobile": st.mobile.tolist(),
            "frequency_mobile": freq.tolist(),
            "delta_atoms": delta,
            "added_ids": [x for x in st.ids if x not in self.seed_index],
            "removed_ids": [x for x in self.seed.ids if x not in st.ids],
            "source_ids": self._family_sources(family),
            "expected_bonds": bonds,
            "occupancy": occupancy,
            "coupling": list(coupling) if coupling else None,
            "geometry_hash": signature,
            "status": "initial_guess",
            "notes": [
                "All coordinates are initial guesses; optimize and inspect actual relaxed topology.",
                "An expected bond does not assign peroxide/superoxide charge or metal oxidation state.",
                "IDs name canonical structural roles; vacancy refill may replace the original isotope. "
                "See edge atom_bookkeeping.",
            ],
        }
        if info["localized_shared_protons"]:
            meta["notes"].append(
                "A strongly shared seed proton next to a modified acceptor site was moved to 0.98 A "
                "from its donor oxygen: " + ", ".join(info["localized_shared_protons"])
            )
        if binding:
            meta["source_ids"] = sorted(set(meta["source_ids"] + ["lee2023"]))
            node["source_ids"] = sorted(set(node["source_ids"] + ["lee2023"]))
            meta["binding_annotations"] = [
                {
                    "site_id": aid,
                    "mode": "side_on_eta2",
                    "metal_id": self.sites[aid]["metal_neighbors"][0]["id"],
                    "atom_ids": [aid, f"d_{aid}"],
                    "evidence_scope": "Side-on O-O binding motif (shown on NiOOH); final binding must be inspected",
                }
                for aid in sorted(binding)
            ]
            meta["notes"].append(
                "Side-on starts move both O atoms; no coordination constraint is imposed during optimization."
            )
        if coupling:
            meta["notes"].append(
                "Coupled O/O is a putative O2-like adsorbate minimum, not an O4 transition state; O-O "
                "starts at 1.45 A and the outward lift avoids M-O distances below 1.65 A."
            )
            metal_contacts = {}
            for aid in coupling:
                io = st.ids.index(aid)
                metal_contacts[aid] = [
                    {"id": st.ids[j], "element": sy, "distance_A": float(distances[io, j])}
                    for j, sy in enumerate(st.symbols)
                    if sy in self.metal_elements and distances[io, j] <= 2.6
                ]
            meta["coupled_O_metal_contacts"] = metal_contacts
            if any(not value for value in metal_contacts.values()):
                meta["notes"].append(
                    "One coupled O has no metal within 2.6 A in this start; binding assignment must be "
                    "reviewed after optimization."
                )
        meta["closest_nonbonded_contacts_A"] = self._closest_contacts(st, distances, bonds)
        state_dir = self.out / "states" / state_id
        state_dir.mkdir(parents=True, exist_ok=True)
        write_poscar(state_dir / "POSCAR", st, comment=f"mechanet {state_id}; atom IDs in state.json")
        write_xyz(state_dir / "initial.extxyz", st)
        dump_json(state_dir / "state.json", meta)
        self.states[state_id] = meta
        self.state_structures[state_id] = st
        self.signature_to_state[signature] = state_id
        node["state_ids"].append(state_id)

    def _closest_contacts(self, st, distances, bonds):
        """Shortest non-bonded contact per element pair involving a new or moved atom."""
        intended = {tuple(sorted(b)) for b in bonds}
        neighbors = {aid: set() for aid in st.ids}
        for a, b in bonds:
            if a in neighbors and b in neighbors:
                neighbors[a].add(b)
                neighbors[b].add(a)
        result = {}
        changed = [
            i
            for i, (aid, p) in enumerate(zip(st.ids, st.positions))
            if aid not in self.seed_index
            or np.linalg.norm(minimum_image(p - self.seed.positions[self.seed_index[aid]], st.cell)) > 1e-6
        ]
        for i in changed:
            for j in range(len(st)):
                if j == i:
                    continue
                a, b = st.ids[i], st.ids[j]
                if tuple(sorted((a, b))) in intended or neighbors[a] & neighbors[b]:
                    continue
                label = "-".join(
                    sorted(("M" if s in self.metal_elements else s) for s in (st.symbols[i], st.symbols[j]))
                )
                if label not in result or distances[i, j] < result[label]:
                    result[label] = round(float(distances[i, j]), 4)
        return dict(sorted(result.items()))

    # ------------------------------------------------------------------ edges and cycles
    def step(self, start, end, family, note="", given=None, reaction=None):
        """Edge between two nodes; the reaction's balance species complete ``given`` coefficients."""
        rx = self.registry.reaction(reaction) if reaction else self.reaction
        a, b = self.nodes[start]["delta_atoms"], self.nodes[end]["delta_atoms"]
        delta = {el: b.get(el, 0) - a.get(el, 0) for el in set(a) | set(b)}
        vector = rx.balance(delta, clean(given or {}))
        eid = edge_id(start, end, vector)
        sources = self._family_sources(family)
        if eid not in self.edges:
            record = {"id": eid, "from": start, "to": end}
            trio = oer_trio(vector)
            if trio is not None:
                record.update(water=trio[0], pairs=trio[1], oxygen=trio[2])
            record.update(
                reservoirs=vector,
                electrons=electrons(vector),
                family=family,
                families=[family],
                source_ids=sources.copy(),
                status="proposed",
                notes=[note] if note else [],
            )
            self.edges[eid] = record
        else:
            ed = self.edges[eid]
            ed["families"] = sorted(set(ed["families"] + [family]))
            ed["source_ids"] = sorted(set(ed["source_ids"] + sources))
            if note and note not in ed["notes"]:
                ed["notes"].append(note)
        return eid

    def edge(self, start, end, water, oxygen, family, note=""):
        """OER edge from its counts: i + water H2O -> j + pairs (H+ + e-) + oxygen O2."""
        return self.step(start, end, family, note, given={"H2O": -water, "O2": oxygen}, reaction="oer")

    def realizable(self, node_id):
        node = self.nodes[node_id]
        return bool(node["state_ids"] or (node.get("computed_by") and self.nodes[node["computed_by"]]["state_ids"]))

    def cycle(self, label, edge_ids, family, evidence_scope="proposed_on_this_surface", reaction=None):
        rx = self.registry.reaction(reaction) if reaction else self.reaction
        for j, e in enumerate(edge_ids):
            if self.edges[e]["to"] != self.edges[edge_ids[(j + 1) % len(edge_ids)]]["from"]:
                raise AssertionError(f"Cycle {label} has nonmatching consecutive states")
        total = add_vectors(*(self.edges[e]["reservoirs"] for e in edge_ids))
        if not rx.is_turnover(total):
            raise AssertionError(f"{rx.name.upper()} cycle {label} has wrong reservoir totals: {total}")
        if any(c["edge_ids"] == edge_ids for c in self.cycles):
            return None
        record = {
            "id": label,
            "edge_ids": list(edge_ids),
            "family": family,
            "reaction": rx.name,
            "evidence_scope": evidence_scope,
            "source_ids": self._family_sources(family),
            "reservoirs": total,
            "geometry_complete": None,
        }
        trio = oer_trio(total)
        if trio is not None:
            record["reservoir_totals"] = dict(zip(("water", "pairs", "oxygen"), trio))
        self.cycles.append(record)
        return label

    def oer_cycle(self, label, links, family, evidence_scope="proposed_on_this_surface"):
        """Cycle from (from, to, water, oxygen, note) links, the form the OER families use."""
        eids = [self.edge(a, b, w, g, family, note) for a, b, w, g, note in links]
        return self.cycle(label, eids, family, evidence_scope, reaction="oer")

    def site_class(self, aid):
        return site_class(self.sites[aid]["kind"])

    def shared_metals(self, a, b):
        return sorted(
            set(m["id"] for m in self.sites[a]["metal_neighbors"])
            & set(m["id"] for m in self.sites[b]["metal_neighbors"])
        )

    def nearby_pairs(self):
        cutoff = float(self.scope.get("pair_max_distance_A", self.geo.get("pair_max_distance_A", 3.3)))
        result = []
        for a, b in combinations(self.sites, 2):
            dist = float(np.linalg.norm(minimum_image(self.site_position(b) - self.site_position(a), self.seed.cell)))
            if dist <= cutoff:
                result.append((a, b))
        return result

    def write_motif(self, motif_id, structure, meta):
        """A separate search seed (e.g. a TS guess), never inserted as a normal minimum."""
        d = self.out / "search_motifs" / motif_id
        d.mkdir(parents=True, exist_ok=True)
        write_poscar(d / "POSCAR", structure, comment=f"mechanet search motif {motif_id}")
        dump_json(d / "motif.json", meta)
        self.motifs.append(meta)

    def _node_image(self, node_id, site_map):
        node = self.nodes[node_id]
        changes = {k: v for k, v in node["occupancy"].items() if v != self.native[k]}
        mapped = {site_map[k]: v for k, v in changes.items()}
        coupling = tuple(site_map[x] for x in node["coupling"]) if node.get("coupling") else None
        return self.node_keys.get(self._node_key(mapped, coupling))

    def _annotate_cycle_symmetry(self):
        by_edges = {tuple(c["edge_ids"]): c for c in self.cycles}
        for c in self.cycles:
            c["symmetry_equivalent_cycles"] = []
        for tr in self.translations:
            for c in self.cycles:
                mapped = []
                for eid in c["edge_ids"]:
                    e = self.edges[eid]
                    start, end = self._node_image(e["from"], tr["site_map"]), self._node_image(e["to"], tr["site_map"])
                    mapped.append(edge_id(start, end, e["reservoirs"]))
                twin = by_edges.get(tuple(mapped))
                if twin is not None and twin["id"] != c["id"] and twin["id"] not in c["symmetry_equivalent_cycles"]:
                    c["symmetry_equivalent_cycles"].append(twin["id"])
        for c in self.cycles:
            orbit = [c] + [x for x in self.cycles if x["id"] in c["symmetry_equivalent_cycles"]]

            def rank(cycle):
                twins = sum(bool(self.nodes[self.edges[e]["from"]].get("computed_by")) for e in cycle["edge_ids"])
                return (twins, cycle["id"])

            c["symmetry_representative"] = min(orbit, key=rank)["id"] == c["id"]

    # ------------------------------------------------------------------ finish
    def finish(self, families=None):
        names = list(families if families is not None else self.scope.get("families", []))
        if not names:
            names = self.registry.core_families(self.reaction.name)
        for family in self.registry.ordered_families(names):
            family.build(self)
        for c in self.cycles:
            c["geometry_complete"] = all(self.realizable(self.edges[e]["from"]) for e in c["edge_ids"])
        self._annotate_cycle_symmetry()
        self._atom_bookkeeping()
        # Persist final aliases and source unions after all families have been registered.
        for node in self.nodes.values():
            for sid in node["state_ids"]:
                meta = self.states[sid]
                meta["source_ids"] = sorted(set(meta["source_ids"] + node["source_ids"]))
                meta["family_aliases"] = sorted(set(meta.get("family_aliases", []) + node["families"]))
                if node.get("symmetry_twins"):
                    meta["symmetry_twin_nodes"] = sorted(node["symmetry_twins"])
                dump_json(self.out / "states" / sid / "state.json", meta)
        for node in self.nodes.values():
            if not node.get("computed_by"):
                node["symmetry_multiplicity"] = 1 + len(node.get("symmetry_twins", []))
        return self._write_outputs(names)

    def _atom_bookkeeping(self):
        id_cache = {}
        tracked = ("O", "H", "Cl")
        for edge in self.edges.values():
            endpoints = []
            for node_id in (edge["from"], edge["to"]):
                if node_id not in id_cache:
                    node = self.nodes[node_id]
                    if node["state_ids"]:
                        meta = self.states[node["state_ids"][0]]
                        id_cache[node_id] = dict(zip(meta["atom_ids"], meta["symbols"]))
                    else:
                        recipe = self.recipes[node_id]
                        st, _, _, _ = self.structure(recipe["occupancy"], recipe["coupling"])
                        id_cache[node_id] = dict(zip(st.ids, st.symbols))
                endpoints.append(id_cache[node_id])
            left, right = endpoints
            removed = {
                symbol: sorted(aid for aid, sy in left.items() if sy == symbol and aid not in right)
                for symbol in tracked
            }
            added = {
                symbol: sorted(aid for aid, sy in right.items() if sy == symbol and aid not in left)
                for symbol in tracked
            }
            removed = {k: v for k, v in removed.items() if v or k in ("O", "H")}
            added = {k: v for k, v in added.items() if v or k in ("O", "H")}
            ledger = {
                "removed_surface_role_ids": removed,
                "added_surface_role_ids": added,
                "reservoirs": edge["reservoirs"],
                "note": "Role IDs are structural positions, not persistent isotope identities across vacancy "
                "refill. Reservoir atom matching is a hypothesis.",
            }
            if not edge["reservoirs"] and len(removed.get("H", [])) == len(added.get("H", [])) == 1:
                ledger["proposed_internal_H_transfer"] = {"from_role": removed["H"][0], "to_role": added["H"][0]}
            edge["atom_bookkeeping"] = ledger

    def _write_outputs(self, families):
        symmetry = {
            "enabled": bool(self.geo.get("symmetry_reduction", True)),
            "tolerance_A": self.symmetry_tolerance_A,
            "translations": [
                {
                    "fractional": [round(x, 6) for x in tr["fractional"]],
                    "max_residual_A": tr["max_residual_A"],
                    "rms_residual_A": tr["rms_residual_A"],
                    "site_map": tr["site_map"],
                    "atom_id_map": tr["id_map"],
                }
                for tr in self.translations
            ],
            "condition": (
                "Approximate translation (residuals above) mapping every atom, both selective-dynamics "
                "masks and any explicit MAGMOM map onto themselves; VASP's default MAGMOM is uniform. "
                "Image nodes are treated as the same calculation; this assumes the same electronic "
                "solution, which a structural match cannot prove."
            ),
        }
        reactions = sorted({c["reaction"] for c in self.cycles} | {self.reaction.name})
        network = {
            "schema_version": 3,
            "generator": f"mechanet {__version__}",
            "reference": "ref",
            "reaction": self.reaction.name,
            "reactions": {name: self.registry.reaction(name).describe() for name in reactions},
            "nodes": list(self.nodes.values()),
            "edges": list(self.edges.values()),
            "cycles": self.cycles,
            "convention": "edge reservoirs: amount of each reservoir species produced (negative: consumed); "
            "electrons = sum(reservoirs * charge), positive when released (anodic)",
            "identity_convention": "Canonical structural-role IDs; isotope lineage is edge-specific, especially "
            "after vacancy refill.",
            "scope": "Proposed thermodynamic network; endpoints are unoptimized initial guesses.",
            "symmetry": symmetry,
            "families": families,
            "TS_search_motifs": [m["id"] for m in self.motifs],
        }
        dump_json(self.out / "network.json", network)
        sites_out = [{k: v for k, v in s.items()} for s in self.sites.values()]
        analysis = {
            "schema_version": 3,
            "seed_file": str(self.input_paths["seed"]),
            "atom_count": len(self.seed),
            "formula_counts": dict(Counter(self.seed.symbols)),
            "cell_A": self.seed.cell.tolist(),
            "vacuum_axis": self.vacuum_axis,
            "surface_sign": self.surface_sign,
            "surface_normal": self.normal.tolist(),
            "roles": self.roles,
            "species_order": self.species_order,
            "fixed_region_source": self.fixed_source,
            "frequency_region_source": self.frequency_source,
            "optimization_mobile_atoms": [i + 1 for i in range(len(self.seed)) if self.seed.mobile[i].any()],
            "frequency_mobile_atoms": [i + 1 for i in range(len(self.seed)) if self.freq_mobile[i].any()],
            "frequency_active_DOF": int(self.freq_mobile.sum()),
            "sites": sites_out,
            "shared_protons": self.shared_protons,
            "translation_symmetry": {k: v for k, v in symmetry.items() if k != "condition"},
        }
        dump_json(self.out / "surface_analysis.json", analysis)
        dump_json(self.out / "sites.json", {"schema_version": 1, "sites": sites_out})
        dump_json(self.out / "rejected_placements.json", self.rejected)
        hashes = self._copy_inputs()
        with (self.out / "sites.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["site_id", "kind", "native", "atom_index_1based", "metal_neighbors", "hydrogen_neighbors"])
            for s in self.sites.values():
                writer.writerow(
                    [
                        s["id"],
                        s["kind"],
                        s["native"],
                        s["index"] or "",
                        ";".join(f"{m['element']}{m['index']}:{m['distance_A']:.4f}A" for m in s["metal_neighbors"]),
                        ";".join(f"H{h['index']}:{h['distance_A']:.4f}A" for h in s["hydrogen_neighbors"]),
                    ]
                )
        with (self.out / "states.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "state_id",
                    "node_id",
                    "family",
                    "site",
                    "N",
                    "delta_atoms",
                    "frequency_atoms",
                    "frequency_DOF",
                    "variant_index",
                    "placement",
                    "binding_modes",
                    "coupled_lift_scale",
                    "min_clearance_above_floor_A",
                ]
            )
            for s in self.states.values():
                ff = np.array(s["frequency_mobile"])
                v = s["variant"]
                writer.writerow(
                    [
                        s["id"],
                        s["node_id"],
                        s["family"],
                        s["site"],
                        len(s["symbols"]),
                        json.dumps(s["delta_atoms"], sort_keys=True),
                        int(ff.any(axis=1).sum()),
                        int(ff.sum()),
                        v["index"],
                        json.dumps(v["placement"], sort_keys=True),
                        json.dumps(v.get("binding_modes", {}), sort_keys=True),
                        v.get("coupled_lift_scale"),
                        v.get("min_clearance_above_floor_A"),
                    ]
                )
        contacts = {}
        for s in self.states.values():
            for label, value in s["closest_nonbonded_contacts_A"].items():
                contacts[label] = min(value, contacts.get(label, np.inf))
        twins = [n for n in self.nodes.values() if n.get("computed_by")]
        report = {
            "schema_version": 3,
            "mechanet_version": __version__,
            "reaction": self.reaction.name,
            "atom_count_reference": len(self.seed),
            "families": families,
            "candidate_nodes": len(self.nodes),
            "computed_nodes": len(self.nodes) - len(twins),
            "symmetry_equivalent_nodes": len(twins),
            "unique_initial_structures": len(self.states),
            "proposed_edges": len(self.edges),
            "complete_stoichiometric_cycles": len(self.cycles),
            "symmetry_unique_cycles": sum(c["symmetry_representative"] for c in self.cycles),
            "cycles_with_generated_geometry_for_every_node": sum(c["geometry_complete"] for c in self.cycles),
            "initial_placements_built": self.attempted,
            "initial_placements_rejected": len(self.rejected),
            "label_permutation_duplicates_merged": len(self.duplicates_merged),
            "nodes_without_viable_start": [
                n["id"] for n in self.nodes.values() if not n["state_ids"] and not n.get("computed_by")
            ],
            "closest_nonbonded_contacts_over_all_starts_A": dict(sorted(contacts.items())),
            "clash_floors_A": self.floors,
            "orientation_search": self.search_cfg,
            "ignored_orientation_search_keys": self.ignored_search_keys,
            "translation_symmetry": symmetry,
            "shared_protons": self.shared_protons,
            "separate_TS_search_motifs": len(self.motifs),
            "input_provenance": hashes,
            "structures_with_side_on_binding": sum(
                bool(s["variant"].get("binding_modes")) for s in self.states.values()
            ),
            "frequency_dof_total": int(sum(np.array(s["frequency_mobile"]).sum() for s in self.states.values())),
            "no_DFT_performed": True,
            "limitations": [
                "Candidate orientations and electronic states are not exhaustive.",
                "Geometry-clash rejection applies to an initial placement, not to the chemical hypothesis.",
                "The pair cutoff is a declared geometric scope, not proof that more distant chemistry is impossible.",
                "Symmetry reduction assumes the same electronic state for exact translational images; "
                "give an explicit MAGMOM map if you intend to break it.",
                "All graph edges remain proposed until endpoint and appropriate path validation.",
            ],
        }
        dump_json(self.out / "build_report.json", report)
        dump_json(self.out / "label_duplicates_merged.json", self.duplicates_merged)
        return report

    def _copy_inputs(self):
        """Copy every input into <campaign>/inputs and point the campaign config at the copies."""
        names = {
            "seed": "seed.vasp",
            "frequency_mask": "frequency_mask.vasp",
            "incar_loose": "INCAR_loose",
            "incar_final": "INCAR_final",
            "kpoints": "KPOINTS",
        }
        hashes, copied = {}, {}
        originals = self.out / "inputs" / "original"
        for key, src in self.input_paths.items():
            if key not in names or not src.is_file():
                continue
            dest = self.out / "inputs" / names[key]
            if key in ("seed", "frequency_mask"):
                originals.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, originals / src.name)
                structure = read_structure(src)
                if key == "seed":
                    structure.mobile = self.seed.mobile.copy()
                write_poscar(dest, structure, comment=f"mechanet {key}; original in inputs/original/{src.name}")
            else:
                shutil.copy2(src, dest)
            hashes[key] = {
                "original_path": str(src),
                "copied_to": str(dest.relative_to(self.out)),
                "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            }
            copied[key] = f"inputs/{names[key]}"
        config = json.loads(json.dumps(self.config))
        config["inputs"] = {**{k: v for k, v in config.get("inputs", {}).items() if k not in names}, **copied}
        # Plugin files travel with the campaign (a campaign copied to a cluster keeps its
        # reactions and families); config.json lists them relative to the campaign.
        plugins, taken = [], set()
        for entry in self.config.get("plugins") or ():
            src = Path(entry).expanduser()
            src = (src if src.is_absolute() else self.base_dir / src).resolve()
            name, k = src.name, 2
            while name.lower() in taken:  # case-insensitive, as on macOS and Windows file systems
                name, k = f"{src.stem}_{k}{src.suffix}", k + 1
            taken.add(name.lower())
            dest = self.out / "inputs" / "plugins" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            hashes[f"plugin:{name}"] = {
                "original_path": str(src),
                "copied_to": str(dest.relative_to(self.out)),
                "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            }
            plugins.append(f"inputs/plugins/{name}")
        config["plugins"] = plugins
        config.setdefault("slab", {})
        config["slab"]["species_order"] = self.species_order
        if self.fixed_source != "file":
            config["slab"]["fixed"] = {"source": "file", "note": "fixed_ids applied to inputs/seed.vasp"}
        dump_json(self.out / "config.json", config)
        return hashes


def build(config, out_dir, base_dir=".", families=None, registry=None):
    """Build a fresh campaign; refuse to write into a non-empty directory."""
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {out}; a campaign is never overwritten: choose another folder "
            "(`mechanet generate --out DIR`) or move the old one away"
        )
    if families is not None:
        config = json.loads(json.dumps(config))
        config.setdefault("scope", {})["families"] = list(families)
    return Builder(config, out, base_dir, registry).finish()
