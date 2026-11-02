"""`mechanet review`: finished structures collect did not accept, the cycles they hold up, decisions."""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import EXAMPLE, FAKE_VASP, FINAL_PROTOCOL, POMASS, write_potcars  # noqa: E402

from mechanet.analysis import collect  # noqa: E402
from mechanet.builder import Builder  # noqa: E402
from mechanet.cli import main  # noqa: E402
from mechanet.fileio import file_hash, read_json, write_json  # noqa: E402
from mechanet.project import load_project, resolve_config  # noqa: E402
from mechanet.review import classify, decide, describe_changes, last_collect, review_report  # noqa: E402
from mechanet.runner import prepare, run_state  # noqa: E402
from mechanet.status import campaign_status, status_page, status_text  # noqa: E402
from mechanet.structure import read_poscar, write_poscar  # noqa: E402

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
META = {
    "atom_ids": ["a0003", "a0011", "a0058", "a0065", "a0066", "a0079", "d_a0065", "p_a0065"],
    "symbols": ["Fe", "Ni", "O", "O", "O", "H", "O", "H"],
}


def run_cli(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(list(argv))
    return code, out.getvalue()


class DescribeTests(unittest.TestCase):
    def test_a_proton_relay_reads_in_order(self):
        topology = {
            "lost_bonds": [["a0058", "a0079"], ["d_a0065", "p_a0065"]],
            "new_bonds": [["a0058", "p_a0065"], ["a0066", "a0079"]],
            "lost_active_metal_contacts": [["a0003", "a0065"]],
            "microstate": {"status": "metal_attachment_changed", "reason": "active metal-ligand contacts changed"},
        }
        self.assertEqual(
            describe_changes(topology, META),
            [
                "H p_a0065 moved from O d_a0065 to O a0058",
                "H a0079 moved from O a0058 to O a0066",
                "O a0065 is no longer bound to Fe a0003",
                "a bond between the metal and an adsorbate or ligand changed",
            ],
        )

    def test_bonds_contacts_and_reasons(self):
        topology = {
            "lost_bonds": [["a0065", "d_a0065"]],
            "new_bonds": [["a0058", "a0066"]],
            "new_active_metal_contacts": [["a0011", "a0058"]],
            "missing_expected_bonds": [["a0065", "d_a0065"], ["d_a0065", "p_a0065"]],
            "shared_proton_transfers": [{"H": "a0079", "from": "a0058", "to": "a0066"}],
            "microstate": {"status": "outside_site_model", "reason": "p_a0065 is not bonded to any oxygen"},
        }
        self.assertEqual(
            describe_changes(topology, META),
            [
                "the O a0065–O d_a0065 bond broke",
                "a O a0058–O a0066 bond formed",
                "the shared H a0079 moved from O a0058 to O a0066",
                "the expected O d_a0065–H p_a0065 bond is missing",
                "O a0058 is now bound to Ni a0011",
                "outside the site model: p_a0065 is not bonded to any oxygen",
            ],
        )
        self.assertEqual(
            describe_changes({"microstate": {"status": "unlisted_microstate"}}, META),
            ["the new arrangement of the sites is not an intermediate of this campaign"],
        )

    def test_classify(self):
        def row(valid, assigned, status="connectivity_retained", usable=True):
            topology = {"status": status, "valid_for_assigned_node": usable, "assigned_node_id": assigned}
            return {
                "state_id": "s",
                "node_id": "n",
                "final_valid": valid,
                "final_assigned_node_id": assigned,
                "final_topology": topology,
            }

        self.assertEqual(classify(row(True, "n")), ("accepted", "n"))
        self.assertEqual(classify(row(True, "m", "reassigned_to_existing_node")), ("reassigned", "m"))
        self.assertEqual(classify(row(True, "n", "accepted_by_explicit_review")), ("reviewed", "n"))
        self.assertEqual(classify(row(True, "m", "assigned_by_explicit_review")), ("reviewed", "m"))
        self.assertEqual(
            classify(row(False, None, "changed_connectivity_needs_review", usable=False)), ("needs_review", None)
        )
        self.assertEqual(classify(row(False, "n")), ("rejected", None))


@unittest.skipIf(os.environ.get("MECHANET_SKIP_INTEGRATION"), "integration test disabled by environment")
class ReviewCampaignTests(unittest.TestCase):
    """AEM on the LDH site a0065, relaxed with the synthetic VASP; the *OOH start then loses its H."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cfg = resolve_config(load_project(EXAMPLE / "project.yaml"), cluster=False)
        cfg["frequency"].update(FINAL_PROTOCOL)
        cfg["scope"]["families"] = ["aem"]
        cfg["sites"] = [s for s in cfg["sites"] if s["id"] == "a0065"]
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
        ooh = sorted(p.name for p in (campaign / "states").glob("n_0065OOH_*"))
        cls.ooh, cls.waiting = ooh[0], ooh[1:]
        for state_dir in sorted((campaign / "states").glob("*")):
            if state_dir.name not in cls.waiting:
                assert run_state(state_dir, through="final", execute=True)["status"] == "complete"
        # The H of the first *OOH start ends on the neighbouring OH a0066 (not a site of this campaign).
        state_dir = campaign / "states" / cls.ooh
        ids = read_json(state_dir / "state.json")["atom_ids"]
        contcar = state_dir / "02_final" / "CONTCAR"
        st = read_poscar(contcar)
        h, o = ids.index("p_a0065"), ids.index("a0066")
        v = st.positions[h] - st.positions[o]
        st.positions[h] = st.positions[o] + 0.98 * v / np.linalg.norm(v)
        write_poscar(contcar, st)
        attempt = read_json(state_dir / "02_final" / "attempt.json")
        hashes = attempt["output_hashes"]
        if isinstance(hashes.get("CONTCAR"), dict):
            hashes["CONTCAR"]["sha256"] = file_hash(contcar)
        else:
            hashes["CONTCAR"] = file_hash(contcar)
        write_json(state_dir / "02_final" / "attempt.json", attempt)
        cls.refs = root / "references.json"
        cls.refs.write_text(json.dumps(REFERENCES))
        cls.base = campaign

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def copy(self):
        """A fresh copy of the campaign for a test that writes decisions."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        target = Path(tmp.name) / "campaign"
        shutil.copytree(self.base, target)
        return target

    def test_report_explains_the_structure_and_the_cycles_it_holds_up(self):
        campaign = self.copy()
        report = review_report(campaign)
        self.assertEqual([i["state_id"] for i in report["not_accepted"]], [self.ooh])
        item = report["not_accepted"][0]
        self.assertEqual(item["kind"], "needs_review")
        self.assertEqual(item["label"], "*OOH₆₅")
        self.assertEqual(item["changes"][0], "H p_a0065 moved from O d_a0065 to O a0066")
        self.assertEqual([o["run"] for o in item["other_starts"]], ["not run yet"] * len(self.waiting))
        held = {c["cycle_id"]: c for c in report["cycles"]["held_up"]}
        self.assertIn("aem_a0065_AEM", held)
        missing = held["aem_a0065_AEM"]["missing"]
        self.assertEqual([m["label"] for m in missing], ["*OOH₆₅"])
        self.assertIn("needs review", missing[0]["why"])
        self.assertIn("not finished", missing[0]["why"])
        self.assertEqual(report["finished"], report["accepted"] + 1)

    def test_accept_assign_and_clear(self):
        campaign = self.copy()
        with self.assertRaisesRegex(ValueError, "has not finished"):
            decide(campaign, self.waiting[0], accept="too early")
        with self.assertRaisesRegex(ValueError, "Give a reason"):
            decide(campaign, self.ooh, accept="  ")
        with self.assertRaisesRegex(ValueError, "Unknown intermediate"):
            decide(campaign, self.ooh, assign=("n_0065OOH_nothing", "why"))
        with self.assertRaisesRegex(ValueError, "another composition"):
            decide(campaign, self.ooh, assign=("ref", "why"))
        result = decide(campaign, self.ooh, accept="the H only leans towards a0066")
        self.assertEqual(result["verdict"]["kind"], "reviewed")
        node_id = read_json(campaign / "states" / self.ooh / "state.json")["node_id"]
        self.assertEqual(result["verdict"]["counts_for"], node_id)
        review = read_json(campaign / "states" / self.ooh / "review.json")
        self.assertTrue(review["accept_relaxed_topology"])
        self.assertEqual(review["written_by"], "mechanet review")
        report = review_report(campaign)
        self.assertEqual(report["not_accepted"], [])
        self.assertEqual([i["state_id"] for i in report["decided"]], [self.ooh])
        self.assertNotIn("aem_a0065_AEM", {c["cycle_id"] for c in report["cycles"]["held_up"]})
        cleared = decide(campaign, self.ooh, clear=True)
        self.assertTrue(cleared["cleared"])
        self.assertFalse((campaign / "states" / self.ooh / "review.json").exists())
        self.assertEqual(len(review_report(campaign)["not_accepted"]), 1)

    def test_commands_point_to_review(self):
        campaign = self.copy()
        code, text = run_cli("review", str(campaign))
        self.assertEqual(code, 0)
        self.assertIn("Needs your review: the relaxed structure changed", text)
        self.assertIn("H p_a0065 moved from O d_a0065 to O a0066", text)
        self.assertIn("aem_a0065_AEM", text)
        self.assertIn('mechanet review --accept STATE "why"', text)
        code, text = run_cli("collect", str(campaign))
        self.assertIn("1 finished structure gave no energy to its own intermediate; `mechanet review`", text)
        code, text = run_cli("thermo", str(campaign), "--electronic-only", "--references", str(self.refs))
        self.assertIn("`mechanet review` shows why", text)
        code, text = run_cli("show-cycle", "aem_a0065_AEM", str(campaign))
        self.assertIn("no structure with an accepted energy; `mechanet review` shows why", text)
        code, text = run_cli("review", str(campaign), "--accept", self.ooh, "H leans towards a0066")
        self.assertEqual(code, 0)
        self.assertIn("now counts for *OOH₆₅", text)
        stamp = (campaign / "analysis" / "thermodynamics.json").stat().st_mtime
        os.utime(campaign / "states" / self.ooh / "review.json", (stamp + 5, stamp + 5))
        code, text = run_cli("show-cycle", "aem_a0065_AEM", str(campaign))
        self.assertIn("a review.json changed after the last thermo; run mechanet thermo again", text)
        code, text = run_cli("--json", "review", str(campaign))
        self.assertEqual(json.loads(text)["not_accepted"], [])

    def test_status_marks_what_the_last_collect_did_not_accept(self):
        campaign = self.copy()
        self.assertIsNone(last_collect(campaign))
        collect(campaign)
        result = campaign_status(campaign, slurm=False)
        self.assertEqual([s["state_id"] for s in result["not_accepted"]], [self.ooh])
        aem = next(c for c in result["cycles"] if c["cycle_id"] == "aem_a0065_AEM")
        self.assertTrue(aem["complete"])
        self.assertTrue(aem["held_up"])
        self.assertGreaterEqual(result["summary"]["held_up"], 1)
        text = status_text(result, show_cycles=True)
        self.assertIn("Finished, but not accepted at the last collect", text)
        self.assertIn("held up", text)
        self.assertIn("?", next(line for line in text.splitlines() if line.rstrip().endswith("aem_a0065_AEM")))
        page = status_page(result)
        self.assertIn("<h2>Finished, but not accepted</h2>", page)
        self.assertIn('class="sq s-finished flag"', page)
        # A decision written after the collect is not judged by it: the mark goes away until the next collect.
        write_json(campaign / "states" / self.ooh / "review.json", {"accept_relaxed_topology": True, "comment": "x"})
        stamp = (campaign / "analysis" / "states.json").stat().st_mtime
        os.utime(campaign / "states" / self.ooh / "review.json", (stamp + 5, stamp + 5))
        again = campaign_status(campaign, slurm=False)
        self.assertEqual(again["not_accepted"], [])
        self.assertFalse(next(c for c in again["cycles"] if c["cycle_id"] == "aem_a0065_AEM")["held_up"])


if __name__ == "__main__":
    unittest.main()
