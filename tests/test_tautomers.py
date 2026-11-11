"""Shared-proton tautomers, intermediates discovered by a relaxation, and adsorbates that leave the surface."""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import EXAMPLE, FAKE_VASP, FINAL_PROTOCOL, POMASS, write_potcars  # noqa: E402

from mechanet.analysis import (  # noqa: E402
    DEFAULT_HYSTERESIS,
    analyze_thermodynamics,
    classify_microstate,
    collect,
    fragment_formula,
    select_for_thermo,
)
from mechanet.builder import Builder  # noqa: E402
from mechanet.cli import main  # noqa: E402
from mechanet.discovered import discover, discover_text, node_id, undiscover, undiscover_text  # noqa: E402
from mechanet.fileio import file_hash, read_json, write_json  # noqa: E402
from mechanet.notation import Notation  # noqa: E402
from mechanet.project import load_project, resolve_config  # noqa: E402
from mechanet.review import review_report, review_text  # noqa: E402
from mechanet.runner import prepare, run_state  # noqa: E402
from mechanet.status import campaign_status, status_page, status_text  # noqa: E402
from mechanet.structure import minimum_image, read_poscar, write_poscar  # noqa: E402
from mechanet.tautomers import TautomerCheck, background_protons, cycle_text  # noqa: E402

REFERENCES = {
    "temperature_K": 298.15,
    "protocol_id": "synthetic test references",
    "H2": {"energy_e0_eV": -6.77398545, "free_energy_correction_eV": -0.121865, "phase": "gas"},
    "H2O": {
        "energy_e0_eV": -14.24029443,
        "free_energy_correction_eV": -0.017671,
        "phase": "liquid",
        "Gcorr_pressure_bar": 0.031,
    },
    "oxygen_reference": {"method": "water_formation", "delta_G_2H2O_to_O2_2H2_eV": 4.92, "temperature_K": 298.15},
}


def run_cli(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(list(argv))
    return code, out.getvalue()


class RuleTests(unittest.TestCase):
    """The rules on their own, on small synthetic inputs."""

    native = {"a1": "OH", "a2": "OH", "a3": "O"}

    def test_background_needs_both_oxygens_native_in_the_intermediate(self):
        ids = ["a1", "a2", "a3", "h1"]
        d = np.array([[0, 2.47, 3, 1.09], [2.47, 0, 3, 1.39], [3, 3, 0, 3], [1.09, 1.39, 3, 0]], dtype=float)
        shared = {3: {0, 1}}
        self.assertEqual(background_protons(shared, ids, d, self.native, {"a3": "OH"}, None), {3: (0, 1)})
        self.assertEqual(background_protons(shared, ids, d, self.native, {"a2": "O"}, None), {})
        self.assertEqual(background_protons(shared, ids, d, self.native, {"a1": "OH", "a3": "OH"}, None), {3: (0, 1)})
        self.assertEqual(background_protons(shared, ids, d, self.native, {}, ["a1", "a3"]), {})
        self.assertEqual(background_protons(shared, ids, d, self.native, None, None), {}, "no occupancy, no rule")
        self.assertEqual(background_protons(shared, ids, d, {"a1": "OH"}, {}, None), {}, "a2 is not a reactive site")

    def test_pairs_that_share_an_oxygen_are_not_read(self):
        # a1-h1...a2-h2...a3: a2 takes part in two shared bonds, so neither pair is read on its own.
        ids = ["a1", "a2", "a3", "h1", "h2"]
        d = np.full((5, 5), 3.0)
        d[3, 0] = d[0, 3] = d[4, 1] = d[1, 4] = 1.09
        d[3, 1] = d[1, 3] = d[4, 2] = d[2, 4] = 1.39
        native = {"a1": "OH", "a2": "OH", "a3": "OH"}
        shared = {3: {0, 1}, 4: {1, 2}}
        self.assertEqual(background_protons(shared, ids, d, native, {"a3": "O"}, None), {})
        self.assertEqual(background_protons({3: {0, 1}}, ids, d, native, {"a3": "O"}, None), {3: (0, 1)})
        self.assertEqual(background_protons({3: {0, 1}}, ids, d, native, {}, None, gone={1}), {})

    def test_centred_means_between_the_seed_and_the_transferred_arrangement(self):
        from mechanet.tautomers import pair_side

        start = ([10], [11], [])
        self.assertEqual(pair_side(start, ([], [11], [10])), "centred")
        self.assertIsNone(
            pair_side(start, ([10], [], [11])), "the partner's own H towards the seed is not the shared one"
        )
        self.assertEqual(pair_side(start, ([], [10, 11], [])), "partner")
        self.assertIsNone(pair_side(start, ([10, 11], [], [])))
        self.assertEqual(pair_side(start, ([11], [10], [])), "seed", "only the counts matter")

    @staticmethod
    def _distances(pairs, n):
        d = np.full((n, n), 5.0)
        np.fill_diagonal(d, 0.0)
        for (i, j), value in pairs.items():
            d[i, j] = d[j, i] = value
        return d

    def test_where_a_background_proton_sits_is_read_from_h_counts(self):
        # Site O a0001 with the shared H a0080 (towards a0002), site O a0002 with its own H a0081;
        # a0009 is a framework O.
        ids, symbols = ["a0001", "a0002", "a0080", "a0081", "a0009"], ["O", "O", "H", "H", "O"]
        structure = types.SimpleNamespace(symbols=symbols)
        context = {"native": {"a0001": "OH", "a0002": "OH"}}
        base = {(0, 1): 2.47, (2, 0): 1.09, (2, 1): 1.39, (3, 1): 0.98, (3, 0): 2.9}
        start = self._distances(base, 5)
        tautomers = {2: (0, 1)}

        def read(final, normalize=True):
            moved = []
            out = classify_microstate(
                structure,
                ids,
                self._distances({**base, **final}, 5),
                context,
                DEFAULT_HYSTERESIS,
                start,
                tautomers=tautomers if normalize else None,
                moved=moved,
            )
            return out[0], out[2], moved

        transferred = {(2, 1): 0.99, (2, 0): 1.48}
        self.assertEqual(read(transferred, normalize=False)[0], {"a0001": "O", "a0002": "H2O"})
        self.assertEqual(read(transferred), ({}, None, [2]))
        # The shared H to a0002 and a0002's own H to a0001: labels swapped, counts did not: nothing moved.
        swapped = {(2, 1): 0.99, (2, 0): 1.48, (3, 0): 0.98, (3, 1): 1.49}
        self.assertEqual(read(swapped), ({}, None, []))
        # Between both oxygens: unreadable literally, the same intermediate with the proton centred.
        centred = {(2, 0): 1.235, (2, 1): 1.235}
        self.assertIsNone(read(centred, normalize=False)[0])
        self.assertEqual(read(centred), ({}, None, [2]))
        # Bonded to no oxygen, or close to a third one: the rule does not apply.
        changes, reason, moved = read({(2, 0): 1.8, (2, 1): 2.6})
        self.assertIsNone(changes)
        self.assertIn("not bonded to any oxygen", reason)
        self.assertEqual(moved, [])
        self.assertEqual(read({(2, 1): 1.30, (2, 0): 2.2, (2, 4): 0.99})[2], [])
        # a0002's own H left for the framework O: the pair holds one H fewer, no tautomer.
        self.assertEqual(read({(3, 1): 2.5, (3, 0): 2.6, (3, 4): 0.98})[2], [])

    def test_cycle_check(self):
        network = {
            "nodes": [
                {"id": "A", "occupancy": {}, "coupling": None},
                {"id": "B", "occupancy": {"s9": "O"}, "coupling": None},
                {"id": "C", "occupancy": {"s2": "H2O"}, "coupling": None},
            ]
        }
        native = {"s1": "OH", "s2": "OH", "s3": "OH", "s4": "OH", "s9": "OH"}
        pairs = (("s1", "s2"), ("s3", "s4"))

        def row(sid, node, *sides):
            record = {}
            for k, on in enumerate(sides):
                seed, partner = pairs[k]
                side = "seed" if on == seed else "partner" if on == partner else "centred"
                record[f"h{k + 1}"] = {"seed": seed, "partner": partner, "side": side, "background": True}
            topology = {"tautomer": record}
            return {
                "state_id": sid,
                "final_valid": True,
                "final_assigned_node_id": node,
                "assigned_node_id": node,
                "final_topology": topology,
                "topology": topology,
            }

        rows = [row("a", "A", "s1"), row("b", "B", "s2"), row("c", "C", "s1")]
        check = TautomerCheck(network, native, rows, {"A": "a", "B": "b", "C": "c"})
        result = check.cycle(["A", "B", "C"])
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(
            result["protons"][0]["on"], {"A": "s1", "B": "s2"}, "C changes s2: the proton is its chemistry"
        )
        self.assertEqual(result["not_computed"], {"A": [{"h1": "s2"}], "B": [{"h1": "s1"}]})
        both = rows + [row("a2", "A", "s2"), row("b2", "B", "s1")]
        self.assertEqual(
            TautomerCheck(network, native, both, {"A": "a", "B": "b", "C": "c"}).cycle(["A", "B", "C"])["status"],
            "mixed_both_computed",
        )
        same = [row("a", "A", "s1"), row("b", "B", "s1"), row("c", "C", "s1")]
        self.assertEqual(
            TautomerCheck(network, native, same, {"A": "a", "B": "b", "C": "c"}).cycle(["A", "B", "C"])["status"],
            "consistent",
        )
        centred = [row("a", "A", "s1"), row("b", "B", "*")]
        self.assertEqual(
            TautomerCheck(network, native, centred, {"A": "a", "B": "b"}).cycle(["A", "B"])["status"], "consistent"
        )
        self.assertEqual(check.cycle(["A", "C"])["status"], "not_available", "one intermediate leaves both native")
        self.assertEqual(
            sorted(len(v["state_ids"]) for v in TautomerCheck(network, native, both, {}).variants("A")), [1, 1]
        )
        # Two shared protons: each side was computed for each proton, but not the other's combination.
        two = [
            row("a1", "A", "s1", "s3"),
            row("a2", "A", "s2", "s4"),
            row("b1", "B", "s2", "s3"),
            row("b2", "B", "s1", "s4"),
        ]
        joint = TautomerCheck(network, native, two, {"A": "a1", "B": "b1"}).cycle(["A", "B"])
        self.assertEqual(joint["status"], "needs_review")
        self.assertEqual(joint["not_computed"], {"A": [{"h1": "s2", "h2": "s3"}], "B": [{"h1": "s1", "h2": "s3"}]})
        self.assertIn("  A was not computed with H h1 on O s2 and H h2 on O s3", cycle_text(joint, str))

    def test_discovered_images_record_the_translation_back(self):
        from mechanet.discovered import images
        from mechanet.magnetism import FrameMaps

        forward = {"a0001": "a0002", "a0002": "a0003", "a0003": "a0001"}
        backward = {v: k for k, v in forward.items()}
        network = {
            "symmetry": {
                "translations": [
                    {"fractional": [1 / 3, 0, 0], "site_map": forward, "atom_id_map": forward},
                    {"fractional": [2 / 3, 0, 0], "site_map": backward, "atom_id_map": backward},
                ]
            },
            "nodes": [{"id": "D", "occupancy": {"a0001": "OO"}}],
        }
        for mapped, _coupling, back in images(network, {"a0001": "OO"}, None):
            network["nodes"].append(
                {
                    "id": "twin_" + next(iter(mapped)),
                    "occupancy": mapped,
                    "computed_by": "D",
                    "symmetry": {"translation_fractional": back["fractional"]},
                }
            )
        maps = FrameMaps(network)
        self.assertEqual(len(network["nodes"]), 3)
        for twin in network["nodes"][1:]:
            self.assertEqual(maps.relabel(["a0001"], twin["id"]), list(twin["occupancy"]))

    def test_formulas_and_labels(self):
        self.assertEqual(fragment_formula(["O", "O"]), "O₂")
        self.assertEqual(fragment_formula(["O", "O", "H"]), "OOH")
        self.assertEqual(fragment_formula(["H", "O", "H"]), "H₂O")
        self.assertEqual(fragment_formula(["Cl", "Cl"]), "Cl₂")
        self.assertEqual(fragment_formula(["O", "O", "O"]), "O₃")
        sites = [
            {"id": "a0065", "kind": "terminal", "native": "OH", "metal_neighbors": [{"id": "a0003"}]},
            {"id": "a0064", "kind": "bridge", "native": "O", "metal_neighbors": [{"id": "a1"}, {"id": "a2"}]},
        ]
        notation = Notation({"nodes": []}, sites)
        self.assertEqual(notation.occupancy_label({"a0065": "OO", "a0064": "OH"}), "*OO₆₅ · μ-OH₆₄")
        self.assertEqual(notation.occupancy_label({}), "native surface")
        # Discovered nodes are named like generated ones.
        self.assertEqual(node_id({"a0065": "O"}, None), "n_0065O_36a161")


def _edit(campaign, state_id, change):
    """Change a finished structure's 02_final CONTCAR (and the hash the runner recorded for it)."""
    state_dir = campaign / "states" / state_id
    ids = read_json(state_dir / "state.json")["atom_ids"]
    contcar = state_dir / "02_final" / "CONTCAR"
    st = read_poscar(contcar)
    change(st, {aid: k for k, aid in enumerate(ids)})
    write_poscar(contcar, st)
    attempt = read_json(state_dir / "02_final" / "attempt.json")
    hashes = attempt["output_hashes"]
    if isinstance(hashes.get("CONTCAR"), dict):
        hashes["CONTCAR"]["sha256"] = file_hash(contcar)
    else:
        hashes["CONTCAR"] = file_hash(contcar)
    write_json(state_dir / "02_final" / "attempt.json", attempt)


def onto(h, o, distance=0.98):
    """Move the H onto the oxygen ``o`` (along their line)."""

    def change(st, k):
        v = minimum_image(st.positions[k[h]] - st.positions[k[o]], st.cell)
        st.positions[k[h]] = st.positions[k[o]] + distance * v / np.linalg.norm(v)

    return change


def away(atoms, metal, shift=4.0):
    """Move ``atoms`` away from ``metal`` by ``shift`` along the metal->first atom direction."""

    def change(st, k):
        v = minimum_image(st.positions[k[atoms[0]]] - st.positions[k[metal]], st.cell)
        for a in atoms:
            st.positions[k[a]] += shift * v / np.linalg.norm(v)

    return change


def leave(atoms, offsets, metal, height=4.5):
    """Move ``atoms`` off the surface as one molecule: the first one ``height`` away from ``metal``
    along their line, the others at ``offsets`` from it."""

    def change(st, k):
        v = minimum_image(st.positions[k[atoms[0]]] - st.positions[k[metal]], st.cell)
        base = st.positions[k[atoms[0]]] + height * v / np.linalg.norm(v)
        for atom, offset in zip(atoms, offsets):
            st.positions[k[atom]] = base + np.asarray(offset, dtype=float)

    return change


@unittest.skipIf(os.environ.get("MECHANET_SKIP_INTEGRATION"), "integration test disabled by environment")
class CampaignTests(unittest.TestCase):
    """AEM on the four terminal sites of the LDH (shared protons a0085: O65/O68, a0087: O67/O66)."""

    O65, OOH65, OO65, O66 = "n_0065O_36a161", "n_0065OOH_cfb68d", "n_0065OO_7550cb", "n_0066O_cd3a07"

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cfg = resolve_config(load_project(EXAMPLE / "project.yaml"), cluster=False)
        cfg["frequency"].update(FINAL_PROTOCOL)
        cfg["scope"]["families"] = ["aem"]
        cfg["sites"] = [s for s in cfg["sites"] if s["id"] in ("a0065", "a0066", "a0067", "a0068")]
        fake = root / "fake_vasp.py"
        fake.write_text(FAKE_VASP)
        cfg["vasp"].update(potcars=write_potcars(root), command=[sys.executable, "-I", str(fake)])
        campaign = root / "base" / "campaign"
        Builder(cfg, campaign).finish()
        for state_dir in sorted((campaign / "states").glob("*")):
            st = read_poscar(state_dir / "POSCAR")
            (state_dir / "ctl").mkdir()
            masses = np.repeat([POMASS[s] for s in st.symbols], 3)
            write_json(
                state_dir / "harmonic.json",
                {"x0": st.positions.reshape(-1).tolist(), "K": np.diag(25.0 * np.sqrt(masses)).tolist()},
            )
            meta = read_json(state_dir / "state.json")
            write_json(state_dir / "moments.json", [{"Fe": 4.1, "Ni": 1.6}.get(s, 0.0) for s in meta["symbols"]])
        prepare(campaign, with_potcars=True)
        for state_dir in sorted((campaign / "states").glob("*")):
            assert run_state(state_dir, through="final", execute=True)["status"] == "complete"
        cls.refs = root / "references.json"
        cls.refs.write_text(json.dumps(REFERENCES))
        cls.base = campaign

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def copy(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        target = Path(tmp.name) / "campaign"
        shutil.copytree(self.base, target)
        return target

    @staticmethod
    def rows(campaign):
        return {row["state_id"]: row for row in collect(campaign)["states"]}

    def test_a_background_shared_proton_that_moved_gives_the_same_intermediate(self):
        campaign = self.copy()
        _edit(campaign, f"{self.O65}__v1", onto("a0087", "a0066"))
        row = self.rows(campaign)[f"{self.O65}__v1"]
        topology = row["final_topology"]
        self.assertEqual(topology["status"], "same_node_other_tautomer", topology.get("microstate"))
        self.assertTrue(row["final_valid"])
        self.assertEqual(row["final_assigned_node_id"], self.O65)
        self.assertEqual(topology["microstate"]["tautomer_moved"], ["a0087"])
        self.assertEqual(topology["microstate"]["exact_occupancy"]["status"], "unlisted_microstate")
        record = topology["tautomer"]["a0087"]
        self.assertEqual(
            (record["on"], record["seed"], record["moved"], record["background"]), ("a0066", "a0067", True, True)
        )
        report = review_report(campaign)
        self.assertEqual(report["not_accepted"], [])
        self.assertEqual(report["tautomers"][0]["moved"], ["the shared H a0087 moved from O a0067 to O a0066"])
        self.assertIn("Accepted in another tautomer", review_text(report))

    def test_a_shared_proton_next_to_a_changed_site_is_chemistry(self):
        campaign = self.copy()
        # *O66: the builder put the shared H a0087 on O67; moved onto O66 it gives *O67, the image of *O65.
        meta = read_json(campaign / "states" / f"{self.O66}__v1" / "state.json")
        self.assertEqual(meta["variant"]["localized_shared_protons"], ["a0087"])
        _edit(campaign, f"{self.O66}__v1", onto("a0087", "a0066"))
        row = self.rows(campaign)[f"{self.O66}__v1"]
        topology = row["final_topology"]
        self.assertEqual(topology["status"], "reassigned_to_existing_node")
        self.assertEqual(topology["microstate"]["matched_node_id"], "n_0067O_2cf563")
        self.assertEqual(row["final_assigned_node_id"], self.O65)
        self.assertNotIn("tautomer_moved", topology["microstate"])
        item = next(i for i in review_report(campaign)["not_accepted"] if i["state_id"] == f"{self.O66}__v1")
        self.assertEqual(item["counts_for_label"], "*O₆₇, image of *O₆₅")

    def test_select_and_thermo_report_tautomers(self):
        campaign = self.copy()
        _edit(campaign, f"{self.O65}__v1", onto("a0087", "a0066"))
        _edit(campaign, f"{self.OOH65}__v1", onto("a0087", "a0066"))
        _edit(campaign, f"{self.O66}__v1", onto("a0087", "a0066"))  # counts for *O65, in the other tautomer
        selection = select_for_thermo(campaign, include_partial=True)
        # *O65 from its own start and the *O66 one; *OOH65 from its two starts.
        self.assertEqual(set(selection["nodes_with_several_tautomers"]), {self.O65, self.OOH65})
        node = next(n for n in selection["nodes"] if n["node_id"] == self.O65)
        self.assertEqual(sorted(t["on"]["a0087"] for t in node["tautomers"]), ["a0066", "a0067"])
        result = analyze_thermodynamics(campaign, self.refs, include_vibrations=False, energy_source="final")
        tautomers = result["tautomers"]
        self.assertIn("aem_a0065_AEM", tautomers["cycles_needing_review"])
        check = tautomers["cycles"]["aem_a0065_AEM"]
        entry = check["protons"][0]
        self.assertEqual(entry["H"], "a0087")
        self.assertEqual(entry["on"]["ref"], "a0067")
        self.assertEqual(entry["on"][self.O65], "a0066")
        for node in (self.O65, self.OOH65):
            self.assertNotIn(node, check["not_computed"], "computed in both tautomers")
        self.assertEqual(check["not_computed"]["ref"], [{"a0087": "a0066"}])
        self.assertEqual(set(tautomers["nodes_with_several_tautomers"]), {self.O65, self.OOH65})
        rows = {r["cycle_id"]: r for r in result["cycles"]}
        self.assertEqual(rows["aem_a0065_AEM"]["tautomer_check"], "needs_review")
        from mechanet.campaign import report

        text = report(campaign)
        self.assertIn("## Shared protons", text)
        self.assertIn("aem_a0065_AEM: *OH₆₅ was not computed with H a0087 on O a0066", text)
        code, text = run_cli("show-cycle", "aem_a0065_AEM", str(campaign))
        self.assertIn("shared H a0087: on O a0066 in", text)
        self.assertIn("*OH₆₅ was not computed with H a0087 on O a0066", text)
        self.assertIn("may contain the energy between two tautomers", text)

    def test_discover_and_undiscover(self):
        campaign = self.copy()
        ooh = f"{self.OOH65}__v2"
        _edit(campaign, ooh, onto("p_a0065", "a0068"))
        report = review_report(campaign)
        item = next(i for i in report["not_accepted"] if i["state_id"] == ooh)
        self.assertIn(
            "the sites are now *OO₆₅ · *(H₂O)₆₈, which is not an intermediate of this campaign", item["changes"]
        )
        self.assertTrue(item["discoverable"])
        self.assertIn('mechanet review --discover STATE "why"', review_text(report))
        with self.assertRaisesRegex(ValueError, "Give a reason"):
            discover(campaign, ooh, " ")
        with self.assertRaisesRegex(ValueError, "accepted for its own intermediate"):
            discover(campaign, f"{self.O65}__v1", "why")
        result = discover(campaign, ooh, "the proton went to the neighbouring OH")
        new = result["node"]["id"]
        self.assertEqual(result["label"], "*OO₆₅ · *(H₂O)₆₈")
        self.assertEqual(result["node"]["occupancy"], {"a0065": "OO", "a0068": "H2O"})
        self.assertEqual(
            result["node"]["delta_atoms"], read_json(campaign / "states" / ooh / "state.json")["delta_atoms"]
        )
        self.assertEqual(result["twins"], ["n_0066H2O_0067OO_9db41d"])
        self.assertEqual(result["verdict"]["kind"], "reassigned")
        self.assertEqual(result["verdict"]["counts_for"], new)
        network = read_json(campaign / "network.json")
        twin = next(n for n in network["nodes"] if n["id"] == "n_0066H2O_0067OO_9db41d")
        self.assertEqual(twin["computed_by"], new)
        self.assertEqual(network["discovered_nodes"][0]["state_id"], ooh)
        with self.assertRaisesRegex(ValueError, "relaxed into"):
            discover(campaign, ooh, "again")
        rows = self.rows(campaign)
        self.assertEqual(rows[ooh]["final_assigned_node_id"], new)
        code, text = run_cli("validate", str(campaign))
        self.assertEqual(code, 0, text)
        self.assertNotIn("no realizable geometry", text)
        code, text = run_cli("pathways", str(campaign))
        self.assertEqual(code, 0, text)
        found = read_json(campaign / "pathways" / "pathways.json")["cycles"]
        through = [c for c in found if new in c["node_ids"]]
        self.assertTrue(through, "pathways proposes cycles through the discovered intermediate")
        code, text = run_cli("pathways", str(campaign), "--adopt", through[0]["id"])
        self.assertEqual(code, 0, text)
        status = campaign_status(campaign, slurm=False)
        self.assertEqual(status["summary"]["discovered_intermediates"], 1)
        cycle = next(c for c in status["cycles"] if c["cycle_id"] == through[0]["id"])
        self.assertEqual(cycle["state_ids"][cycle["node_ids"].index(new)], [ooh])
        with self.assertRaisesRegex(ValueError, "generated with the campaign"):
            undiscover(campaign, self.O65)
        code, text = run_cli("review", str(campaign), "--undiscover", "n_0066H2O_0067OO_9db41d")
        self.assertEqual(code, 0, text)
        self.assertIn(through[0]["id"], text)
        network = read_json(campaign / "network.json")
        self.assertFalse([n for n in network["nodes"] if n.get("discovered_from")])
        self.assertNotIn(through[0]["id"], {c["id"] for c in network["cycles"]})
        self.assertNotIn("discovered_nodes", network)
        self.assertEqual(self.rows(campaign)[ooh]["final_topology"]["microstate"]["status"], "unlisted_microstate")

    def test_what_the_tautomer_reading_does_not_hide(self):
        campaign = self.copy()
        normal = np.asarray(read_json(campaign / "surface_analysis.json")["surface_normal"], dtype=float)
        o65, ooh65 = f"{self.O65}__v1", f"{self.OOH65}__v1"
        # A background H bonded to no oxygen at all.
        _edit(
            campaign, o65, lambda st, k: st.positions.__setitem__(k["a0087"], st.positions[k["a0067"]] + 1.8 * normal)
        )
        # The OOH proton went to O67 while the shared H moved to O66: *OO65 with a water on 66, read literally.
        _edit(campaign, ooh65, onto("p_a0065", "a0067"))
        _edit(campaign, ooh65, onto("a0087", "a0066"))
        rows = self.rows(campaign)
        micro = rows[o65]["final_topology"]["microstate"]
        self.assertEqual(
            (micro["status"], micro["reason"]), ("outside_site_model", "a0087 is not bonded to any oxygen")
        )
        self.assertFalse(rows[o65]["final_valid"])
        self.assertIsNone(rows[o65]["final_topology"]["tautomer"]["a0087"]["side"])
        micro = rows[ooh65]["final_topology"]["microstate"]
        self.assertEqual(micro["status"], "unlisted_microstate")
        self.assertEqual(micro["occupancy_changes"], {"a0065": "OO", "a0066": "H2O"})
        self.assertNotIn("tautomer_moved", micro)
        report = review_report(campaign, rows=list(rows.values()))
        self.assertEqual(report["tautomers"], [])

    def test_ambiguous_centred_proton_and_swapped_labels(self):
        campaign = self.copy()
        o65, oooh = f"{self.O65}__v1", f"{self.OOH65}__v1"

        def midpoint(st, k):
            v = minimum_image(st.positions[k["a0066"]] - st.positions[k["a0067"]], st.cell)
            st.positions[k["a0087"]] = st.positions[k["a0067"]] + 0.5 * v

        # O66 gave its own H to O65 and the shared H sits midway: *O66 or the image of *O65, unreadable.
        _edit(campaign, o65, onto("a0086", "a0065"))
        _edit(campaign, o65, midpoint)
        # The shared H to O66 and O66's own H to O67: only labels changed, the same tautomer.
        _edit(campaign, oooh, onto("a0087", "a0066"))
        _edit(campaign, oooh, onto("a0086", "a0067"))
        rows = self.rows(campaign)
        self.assertEqual(rows[o65]["final_topology"]["microstate"]["status"], "outside_site_model")
        self.assertFalse(rows[o65]["final_valid"])
        topology = rows[oooh]["final_topology"]
        self.assertEqual(topology["status"], "same_occupancy_proton_labels_permuted")
        self.assertEqual(topology["tautomer"]["a0087"]["side"], "seed")
        result = analyze_thermodynamics(campaign, self.refs, include_vibrations=False, energy_source="final")
        self.assertEqual(result["tautomers"]["cycles_needing_review"], [])

    def test_a_spectator_that_left_still_needs_review(self):
        campaign = self.copy()
        sid = f"{self.O65}__v1"
        _edit(campaign, sid, onto("a0087", "a0066"))
        _edit(campaign, sid, away(["a0066", "a0086", "a0087"], "a0011"))
        row = self.rows(campaign)[sid]
        self.assertEqual(row["final_topology"]["released"][0]["formula"], "H₂O")
        self.assertEqual(row["final_topology"]["microstate"]["status"], "metal_attachment_changed")
        self.assertIsNone(row["final_topology"]["tautomer"]["a0087"]["side"], "its oxygen left the surface")
        item = next(i for i in review_report(campaign)["not_accepted"] if i["state_id"] == sid)
        self.assertEqual(item["kind"], "needs_review")
        self.assertEqual(item["changes"][0], "H₂O (O a0066) released from Ni a0011")
        # The OOH proton went to O66 and that water left: a spectator, though it carries an added H.
        ooh = f"{self.OOH65}__v1"
        _edit(campaign, ooh, onto("p_a0065", "a0066"))
        _edit(campaign, ooh, away(["a0066", "a0086", "p_a0065"], "a0011"))
        topology = self.rows(campaign)[ooh]["final_topology"]
        self.assertEqual([x["formula"] for x in topology["released"]], ["H₂O"])
        self.assertEqual(topology["microstate"]["status"], "metal_attachment_changed")

    def test_a_discovered_intermediate_through_select_thermo_and_pathways(self):
        campaign = self.copy()
        ooh = f"{self.OOH65}__v2"
        _edit(campaign, ooh, onto("p_a0065", "a0068"))
        new = discover(campaign, ooh, "the proton went to the neighbouring OH")["node"]["id"]
        # discover collected again: the status page knows at once.
        from mechanet.review import last_collect

        verdict = last_collect(campaign)["verdicts"][ooh]
        self.assertEqual((verdict["kind"], verdict["counts_for"]), ("reassigned", new))
        self.assertEqual(run_cli("pathways", str(campaign))[0], 0)
        through = [c for c in read_json(campaign / "pathways" / "pathways.json")["cycles"] if new in c["node_ids"]]
        self.assertEqual(run_cli("pathways", str(campaign), "--adopt", through[0]["id"])[0], 0)
        selection = select_for_thermo(campaign, include_partial=True)
        node = next(n for n in selection["nodes"] if n["node_id"] == new)
        self.assertEqual(node["selected"], [ooh])
        result = analyze_thermodynamics(campaign, self.refs, include_vibrations=False, energy_source="final")
        self.assertIn(new, result["node_energies_eV"])
        self.assertEqual({r["status"] for r in result["cycles"] if r["cycle_id"] == through[0]["id"]}, {"available"})
        undiscover(campaign, new)
        with self.assertRaisesRegex(ValueError, "Not in pathways/pathways.json"):
            from mechanet.pathways import adopt

            adopt(campaign, [through[1]["id"]])
        code, text = run_cli("validate", str(campaign))
        self.assertEqual(code, 0, text)

    def test_an_adsorbate_that_left_the_surface_has_no_bound_minimum(self):
        campaign = self.copy()
        starts = sorted(p.name for p in (campaign / "states").glob(f"{self.OO65}__v*"))
        self.assertEqual(len(starts), 3)
        for sid in starts:
            _edit(campaign, sid, away(["a0065", "d_a0065"], "a0003"))
        rows = self.rows(campaign)
        topology = rows[starts[0]]["final_topology"]
        self.assertEqual(topology["released"], [{"formula": "O₂", "atoms": ["a0065", "d_a0065"], "from": ["a0003"]}])
        self.assertEqual(topology["microstate"]["status"], "adsorbate_released")
        with self.assertRaisesRegex(ValueError, "not an intermediate"):
            discover(campaign, starts[0], "why")
        report = review_report(campaign, rows=list(rows.values()))
        self.assertEqual({i["kind"] for i in report["not_accepted"]}, {"no_bound_minimum"})
        item = report["not_accepted"][0]
        self.assertEqual(item["changes"][0], "O₂ (O a0065, O d_a0065) released from Fe a0003")
        self.assertNotIn("O a0065 is no longer bound to Fe a0003", item["changes"])
        self.assertEqual([n["node_id"] for n in report["no_bound_minimum"]], [self.OO65])
        self.assertIn("aem_a0065_bound_OO", {c["cycle_id"] for c in report["cycles"]["no_bound_minimum"]})
        self.assertNotIn("aem_a0065_bound_OO", {c["cycle_id"] for c in report["cycles"]["held_up"]})
        why = report["no_bound_minimum"][0]["why"]
        self.assertTrue(why.startswith("no bound minimum found from 3 starts: O₂ left the surface"), why)
        text = review_text(report)
        self.assertIn("No bound minimum from this start: a molecule left the surface", text)
        self.assertNotIn("What to do", text)
        status = campaign_status(campaign, slurm=False)
        cycle = next(c for c in status["cycles"] if c["cycle_id"] == "aem_a0065_bound_OO")
        self.assertTrue(cycle["no_bound_minimum"])
        self.assertFalse(cycle["held_up"])
        self.assertEqual(status["summary"]["intermediates_without_bound_minimum"], 1)
        line = next(x for x in status_text(status, show_cycles=True).splitlines() if x.endswith("aem_a0065_bound_OO"))
        self.assertIn("∅", line)
        self.assertIn("no bound minimum: every finished start lost its adsorbate", status_page(status))
        self.assertIn("0", status_text(status, show_cycles=True, ascii=True))

    def test_an_adsorbate_that_left_with_a_surface_atom_names_the_surface_left_behind(self):
        """The O of *O leaves as O2 with a lattice O (not a site), or with the OH of the next site:
        no energy for *O, no decision to take, and the surface left behind is named."""
        sid = f"{self.O65}__v1"
        cases = (
            # atoms that leave, where, what left the surface, the atom of the surface, the surface left
            (["a0065", "a0064"], [(0, 0, 0), (1.21, 0, 0)], "O₂", "a0064", None),
            (["a0065", "a0068", "a0088"], [(0, 0, 0), (1.45, 0, 0), (1.75, 0.92, 0)], "OOH", "a0068", "*₆₅ · *₆₈"),
        )
        for atoms, offsets, formula, taken, label in cases:
            with self.subTest(taken=taken):
                campaign = self.copy()
                _edit(campaign, sid, leave(atoms, offsets, "a0003"))
                rows = self.rows(campaign)
                topology = rows[sid]["final_topology"]
                micro = topology["microstate"]
                self.assertEqual(micro["status"], "surface_atom_released")
                self.assertEqual(micro["taken_surface_atoms"], [taken])
                self.assertEqual([m["formula"] for m in topology["released"]], [formula])
                self.assertFalse(rows[sid]["final_valid"])
                report = review_report(campaign, rows=list(rows.values()))
                item = next(i for i in report["not_accepted"] if i["state_id"] == sid)
                # no decision to take; a surface the sites can describe can be added to the campaign
                self.assertEqual((item["kind"], item["discoverable"]), ("no_bound_minimum", label is not None))
                self.assertEqual(item["surface_left"]["taken"], [taken])
                self.assertEqual(item["surface_left"]["label"], label)
                self.assertIsNone(item["surface_left"]["node_id"])
                self.assertEqual(
                    item["changes"][-2], f"it took O {taken} of the surface along: no bound minimum from this start"
                )
                text = review_text(report)
                self.assertIn("the surface left behind is a possible intermediate of a lattice route", text)
                self.assertNotIn("What to do", text)
                if label is None:
                    self.assertEqual(
                        item["changes"][-1],
                        "the surface left behind is a possible intermediate that the sites of this campaign cannot "
                        "describe (O a0064 is not part of a reactive site: declare it as a site to follow this route)",
                    )
                    self.assertNotIn("--discover", text)
                    with self.assertRaisesRegex(ValueError, "declare it as a site"):
                        discover(campaign, sid, "why")
                else:
                    self.assertEqual(
                        item["changes"][-1],
                        f"the surface left behind is {label}, a possible intermediate that this campaign does not "
                        "list: mechanet review --discover can add it, with a start made from this structure without "
                        "the molecule",
                    )
                    self.assertEqual(item["surface_left"]["occupancy"], {"a0065": "empty", "a0068": "empty"})
                    self.assertIn('mechanet review --discover STATE "why"       add that surface', text)
                # the intermediate has no accepted structure, and the status page copes
                self.assertEqual([n["node_id"] for n in report["no_bound_minimum"]], [self.O65])
                status = campaign_status(campaign, slurm=False)
                self.assertEqual(status["summary"]["intermediates_without_bound_minimum"], 1)
                code, text = run_cli("validate", str(campaign))
                self.assertEqual(code, 0, text)

    def gone_with_the_next_site(self):
        """A copy of the campaign in which the O of *O65 left with the OH of site a0068."""
        campaign = self.copy()
        sid = f"{self.O65}__v1"
        atoms, offsets = ["a0065", "a0068", "a0088"], [(0, 0, 0), (1.45, 0, 0), (1.75, 0.92, 0)]
        _edit(campaign, sid, leave(atoms, offsets, "a0003"))
        return campaign, sid, atoms

    def test_the_surface_left_behind_gets_a_start_of_its_own(self):
        """review --discover adds the surface a release left behind, with a start made from the
        relaxed structure without the molecule; the start is one of the campaign like any other."""
        campaign, sid, atoms = self.gone_with_the_next_site()
        result = discover(campaign, sid, "O2 left with the OH of the next site")
        node, start = result["node"], result["start_state_id"]
        self.assertEqual(result["label"], "*₆₅ · *₆₈")
        self.assertEqual(node["occupancy"], {"a0065": "empty", "a0068": "empty"})
        self.assertEqual((node["state_ids"], start), ([f"{node['id']}__v1"], f"{node['id']}__v1"))
        self.assertEqual(node["delta_atoms"], {"O": -2, "H": -2})
        origin = node["discovered_from"]
        self.assertEqual((origin["state_id"], origin["start_state_id"]), (sid, start))
        self.assertEqual(origin["released"], [{"formula": "OOH", "atoms": atoms, "composition": {"H": 1, "O": 2}}])
        self.assertEqual(origin["taken_surface_atoms"], ["a0068"])
        self.assertEqual(result["twins"], ["n_0066empty_0067empty_bc125e"])
        text = discover_text(result)
        self.assertIn(f"its start is states/{start}: the relaxed structure without the molecule", text)
        self.assertIn(f"mechanet submit --states {start} --execute", text)
        # the start: the relaxed positions, without the three atoms that left
        folder = campaign / "states" / start
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["POSCAR", "initial.extxyz", "state.json"])
        meta, old = read_json(folder / "state.json"), read_json(campaign / "states" / sid / "state.json")
        self.assertEqual(set(old["atom_ids"]) - set(meta["atom_ids"]), set(atoms))
        relaxed = read_poscar(campaign / "states" / sid / "02_final" / "CONTCAR")
        where = dict(zip(old["atom_ids"], relaxed.positions))
        written = read_poscar(folder / "POSCAR")
        for atom, position in zip(meta["atom_ids"], written.positions):
            self.assertLess(float(np.linalg.norm(position - where[atom])), 1e-8, atom)
        self.assertEqual(
            (meta["node_id"], meta["family"], meta["delta_atoms"]), (node["id"], "discovered", node["delta_atoms"])
        )
        self.assertEqual(
            meta["from_relaxed"],
            {
                "state_id": sid,
                "built_for_node_id": self.O65,
                "geometry": "02_final/CONTCAR",
                "removed_atom_ids": sorted(atoms),
                "released": ["OOH"],
            },
        )
        self.assertIn("without the released OOH", meta["notes"][-1])
        # everything else is what the builder gives that intermediate: atoms, order, masks, bonds
        builder = Builder(read_json(campaign / "config.json"), campaign.parent / "again", base_dir=campaign)
        st, freq, bonds, _info = builder.structure(dict(node["occupancy"]), None)
        self.assertEqual(meta["atom_ids"], st.ids)
        self.assertEqual(meta["frequency_mobile"], freq.tolist())
        self.assertEqual(meta["relax_mobile"], st.mobile.tolist())
        self.assertEqual(meta["expected_bonds"], bonds)
        code, text = run_cli("validate", str(campaign))
        self.assertEqual(code, 0, text)
        self.assertIn("0 error(s), 0 warning(s)", text)
        # the structure it was found in still gives no energy, and now names the intermediate
        item = next(i for i in review_report(campaign)["not_accepted"] if i["state_id"] == sid)
        self.assertEqual((item["kind"], item["discoverable"]), ("no_bound_minimum", False))
        self.assertEqual(
            item["changes"][-1],
            f"the surface left behind is *₆₅ · *₆₈ ({node['id']}), an intermediate of this campaign: "
            "a lattice route leads there from this one",
        )
        with self.assertRaisesRegex(ValueError, "already an intermediate of this campaign"):
            discover(campaign, sid, "again")
        # it runs and is judged like any start
        structure = read_poscar(folder / "POSCAR")
        (folder / "ctl").mkdir()
        masses = np.repeat([POMASS[s] for s in structure.symbols], 3)
        write_json(
            folder / "harmonic.json",
            {"x0": structure.positions.reshape(-1).tolist(), "K": np.diag(25.0 * np.sqrt(masses)).tolist()},
        )
        write_json(folder / "moments.json", [{"Fe": 4.1, "Ni": 1.6}.get(s, 0.0) for s in meta["symbols"]])
        self.assertEqual(
            [p["state_id"] for p in prepare(campaign, selection=[start], with_potcars=True)["prepared"]], [start]
        )
        self.assertEqual(run_state(folder, through="final", execute=True)["status"], "complete")
        row = self.rows(campaign)[start]
        self.assertTrue(row["final_valid"])
        self.assertEqual(row["final_assigned_node_id"], node["id"])
        status = campaign_status(campaign, slurm=False)
        self.assertEqual(status["summary"]["discovered_intermediates"], 1)
        # removing the intermediate keeps what was computed for it
        out = undiscover(campaign, node["id"])
        self.assertEqual(out["start"], {"state_id": start, "set_aside": f"undiscovered/{start}"})
        self.assertFalse(folder.exists())
        self.assertTrue((campaign / "undiscovered" / start / "02_final" / "CONTCAR").is_file())
        self.assertIn("moved to undiscovered/", undiscover_text(out))
        self.assertFalse([n for n in read_json(campaign / "network.json")["nodes"] if n.get("discovered_from")])
        code, text = run_cli("validate", str(campaign))
        self.assertEqual(code, 0, text)

    def test_a_start_that_never_ran_goes_with_its_intermediate(self):
        campaign, sid, _atoms = self.gone_with_the_next_site()
        result = discover(campaign, sid, "why")
        folder = campaign / "states" / result["start_state_id"]
        self.assertTrue(folder.is_dir())
        out = undiscover(campaign, result["node"]["id"])
        self.assertEqual(out["start"], {"state_id": result["start_state_id"], "set_aside": None})
        self.assertFalse(folder.exists())
        self.assertFalse((campaign / "undiscovered").exists())
        self.assertIn("which was never run, was removed", undiscover_text(out))
        code, text = run_cli("validate", str(campaign))
        self.assertEqual(code, 0, text)
        # and it can be found again
        self.assertEqual(discover(campaign, sid, "again")["start_state_id"], result["start_state_id"])

    def test_nothing_is_registered_when_the_start_cannot_be_written(self):
        campaign, sid, _atoms = self.gone_with_the_next_site()
        before = (campaign / "network.json").read_bytes()
        with patch("mechanet.builder.Builder.structure", side_effect=ValueError("no placement")):
            with self.assertRaisesRegex(ValueError, "the builder cannot write .* add it through the families"):
                discover(campaign, sid, "why")
        self.assertEqual((campaign / "network.json").read_bytes(), before)
        self.assertEqual(
            sorted(p.name for p in (campaign / "states").iterdir()),
            sorted(p.name for p in (self.base / "states").iterdir()),
        )


if __name__ == "__main__":
    unittest.main()
