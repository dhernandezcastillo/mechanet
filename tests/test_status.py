"""Readable notation of intermediates, steps and cycles; `mechanet status` and its page."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import POSIX_ONLY, EXAMPLE, SRC  # noqa: E402,F401  (puts src on sys.path)

from mechanet.builder import Builder  # noqa: E402
from mechanet.cli import main  # noqa: E402
from mechanet.fileio import read_json, write_json  # noqa: E402
from mechanet.notation import Notation, citations, encodable, formula, plain, route, species  # noqa: E402
from mechanet.project import load_project, resolve_config  # noqa: E402
from mechanet.status import campaign_status, state_list, status_page, status_text  # noqa: E402

SITES = [
    {
        "id": "a0065",
        "element": "O",
        "kind": "terminal",
        "native": "OH",
        "metal_neighbors": [{"id": "a0003", "element": "Fe"}],
    },
    {
        "id": "a0066",
        "element": "O",
        "kind": "terminal",
        "native": "OH",
        "metal_neighbors": [{"id": "a0011", "element": "Ni"}],
    },
    {
        "id": "a0061",
        "element": "O",
        "kind": "bridge",
        "native": "OH",
        "metal_neighbors": [{"id": "a0003", "element": "Fe"}, {"id": "a0005", "element": "Ni"}],
    },
    {
        "id": "m0012",
        "element": None,
        "kind": "open_metal",
        "native": "empty",
        "metal_neighbors": [{"id": "a0012", "element": "Ru"}],
    },
]


def toy_network():
    """An AEM cycle on a0065, stored starting at the native surface (not in the turnover order)."""
    nodes = [
        {"id": "ref", "occupancy": {}, "state_ids": ["ref"]},
        {"id": "e", "occupancy": {"a0065": "empty"}, "state_ids": ["e_v1"]},
        {"id": "o", "occupancy": {"a0065": "O"}, "state_ids": ["o_v1", "o_v2"]},
        {"id": "ooh", "occupancy": {"a0065": "OOH"}, "state_ids": ["ooh_v1"]},
        {"id": "pair", "occupancy": {"a0065": "O", "a0066": "O"}, "coupling": ["a0065", "a0066"], "state_ids": []},
        {"id": "w", "occupancy": {"a0065": "H2O", "a0061": "empty"}, "state_ids": []},
    ]
    edges = [
        {"id": "e2", "from": "ref", "to": "o", "reservoirs": {"H+": 1}},
        {"id": "e3", "from": "o", "to": "ooh", "reservoirs": {"H2O": -1, "H+": 1}},
        {"id": "e4", "from": "ooh", "to": "e", "reservoirs": {"O2": 1, "H+": 1}},
        {"id": "e1", "from": "e", "to": "ref", "reservoirs": {"H2O": -1, "H+": 1}},
    ]
    cycles = [
        {
            "id": "aem_a0065_AEM",
            "family": "aem",
            "reaction": "oer",
            "edge_ids": ["e2", "e3", "e4", "e1"],
            "evidence_scope": "proposed_on_this_surface",
            "source_ids": ["rossmeisl2005", "rossmeisl2007", "man2011"],
            "symmetry_equivalent_cycles": ["aem_a0067_AEM"],
        }
    ]
    return {"reference": "ref", "reaction": "oer", "nodes": nodes, "edges": edges, "cycles": cycles}


class NotationTests(unittest.TestCase):
    def setUp(self):
        self.notation = Notation(toy_network(), SITES)

    def test_terms_of_each_site_kind(self):
        term = self.notation.term
        self.assertEqual(term("a0065", "OOH"), "*OOH₆₅")
        self.assertEqual(term("a0065", "empty"), "*₆₅")  # the bare metal
        self.assertEqual(term("a0065", "H2O"), "*(H₂O)₆₅")  # digits of a formula are bracketed
        self.assertEqual(term("a0061", "OH"), "μ-OH₆₁")
        self.assertEqual(term("a0061", "empty"), "μ-□₆₁")  # a bridge vacancy
        self.assertEqual(term("m0012", "empty"), "*₁₂")
        self.assertEqual(self.notation.symbol("a0061"), "μ₆₁")
        self.assertEqual(self.notation.site_legend("a0061"), "61 = a0061: O bridging Fe a0003 and Ni a0005 (native OH)")
        self.assertEqual(self.notation.site_legend("m0012"), "12 = m0012: open site on Ru a0012 (bare)")

    def test_labels_of_intermediates(self):
        label = self.notation.label
        self.assertEqual(label("ref"), "native surface")
        self.assertEqual(label("ref", ["a0065", "a0061"]), "*OH₆₅ · μ-OH₆₁")
        self.assertEqual(label("o"), "*O₆₅")
        self.assertEqual(label("pair"), "*O₆₅–*O₆₆")  # the two O bonded to each other
        self.assertEqual(label("w", ["a0065", "a0061"]), "*(H₂O)₆₅ · μ-□₆₁")

    def test_a_cycle_reads_in_the_turnover_order(self):
        view = self.notation.cycle("aem_a0065_AEM")
        self.assertEqual(view["title"], "Adsorbate evolution (AEM)")
        self.assertEqual(view["labels"], ["*₆₅", "*OH₆₅", "*O₆₅", "*OOH₆₅"])  # O2 release last
        self.assertEqual(
            [s["equation"] for s in view["steps"]],
            [
                "*₆₅ + H₂O → *OH₆₅ + H⁺ + e⁻",
                "*OH₆₅ → *O₆₅ + H⁺ + e⁻",
                "*O₆₅ + H₂O → *OOH₆₅ + H⁺ + e⁻",
                "*OOH₆₅ → *₆₅ + O₂ + H⁺ + e⁻",
            ],
        )
        self.assertEqual({s["kind"] for s in view["steps"]}, {"electrochemical"})
        self.assertEqual(
            view["sources"], "Proposed on this surface. Sources: Rossmeisl et al. 2005, 2007; Man et al. 2011"
        )
        self.assertEqual(view["site_legend"], ["65 = a0065: terminal O on Fe a0003 (native OH)"])
        self.assertEqual(view["release"], "O₂")

    def test_chemical_steps_and_charged_species(self):
        edge = {"id": "x", "from": "ooh", "to": "e", "reservoirs": {"O2": 1}}
        step = self.notation.step(edge)
        self.assertEqual(step["kind"], "chemical")
        self.assertEqual(step["equation"], "*OOH₆₅ → *₆₅ + O₂")
        heyrovsky = self.notation.step({"id": "y", "from": "ooh", "to": "e", "reservoirs": {"Cl-": -1, "Cl2": 1}})
        self.assertEqual(heyrovsky["equation"], "*OOH₆₅ + Cl⁻ → *₆₅ + Cl₂ + e⁻")
        self.assertEqual(species("H2O"), "H₂O")
        self.assertEqual(species("Fe3+"), "Fe³⁺")
        self.assertEqual(formula("OOH"), "OOH")

    def test_routes_of_the_built_in_families(self):
        cases = {
            "aem_a0065_bound_OO": ("aem", "AEM, bound O₂", None),
            "surface_bridge_a0061_AEM": ("surface_bridge", "bridge AEM", None),
            "bifunctional_a0065_a0061_BF2": ("bifunctional", "BF-II", None),
            "bifunctional_a0065_a0061_resolved": ("bifunctional", "stepwise *OOH", None),
            "pair_coupling_a0065_a0066_65R_66R_65D_66D_direct": (
                "pair_coupling",
                "direct O–O coupling",
                "refill 65, then refill 66, then deprotonate 65, then deprotonate 66",
            ),
            "terminal_bridge_coupling_a0065_a0061_65R_61D_61R_65D_bound": (
                "terminal_bridge_coupling",
                "bound terminal–bridge coupling",
                "refill 65, then deprotonate 61, then refill 61, then deprotonate 65",
            ),
            "hydrated_a0065_a0061_tautomer_via_OO_H2O": ("hydrated", "tautomer via *OO", None),
            "hydrated_a0065_a0061_via_OO_H2O": ("hydrated", "hydration via *OO", None),
            "oo_oo_a0065_a0066_65A_66A_65D_66D": (
                "oo_oo",
                "OO/OO (O₄)",
                "OOH on 65, then OOH on 66, then deprotonate 65, then deprotonate 66",
            ),
            "cer_volmer_heyrovsky_m0012": ("cer_volmer_heyrovsky", "Volmer–Heyrovsky", None),
        }
        for cycle_id, (family, short, variant) in cases.items():
            with self.subTest(cycle_id):
                names = route({"id": cycle_id, "family": family})
                self.assertEqual((names["short"], names["variant"]), (short, variant))
        plugin = route({"id": "myfam_a0001_X", "family": "myfam"}, {"myfam": "my own mechanism"})
        self.assertEqual((plugin["short"], plugin["title"]), ("X", "My own mechanism (X)"))

    def test_ascii_form_and_citations(self):
        self.assertEqual(
            plain("*OOH₆₅ · μ-O₆₁ + H₂O → *₆₅ · μ-□₆₁ + H⁺ + e⁻"), "*OOH65 & mu-O61 + H2O -> *65 & mu-[]61 + H+ + e-"
        )
        self.assertEqual(plain("Hernández"), "Hernandez")
        ascii_stream = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
        self.assertFalse(encodable(ascii_stream, "μ"))
        self.assertTrue(encodable(ascii_stream, "mu"))
        self.assertEqual(citations(["halck2014", "exner2023"]), "Halck et al. 2014; Exner 2023")


FAKE_SQUEUE = """#!/bin/sh
here="$(dirname "$0")"
for arg in "$@"; do
  case "$arg" in --user=*) cat "$here/jobs.txt"; exit 0 ;; esac
done
exit 0
"""


class StatusTests(unittest.TestCase):
    """A small LDH campaign (AEM and bifunctional cycles on a0065 with the bridge a0061) in every status."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cfg = resolve_config(load_project(EXAMPLE / "project.yaml"), cluster=False)
        cfg["scope"]["families"] = ["aem", "bifunctional"]
        cfg["sites"] = [s for s in cfg["sites"] if s["id"] in ("a0065", "a0061")]
        cls.campaign = root / "campaign"
        Builder(cfg, cls.campaign).finish()
        cls.bin = root / "bin"
        cls.bin.mkdir()
        (cls.bin / "squeue").write_text(FAKE_SQUEUE)
        (cls.bin / "squeue").chmod(0o755)
        network = read_json(cls.campaign / "network.json")
        nodes = {n["id"]: n for n in network["nodes"]}
        native = {s["id"]: s["native"] for s in read_json(cls.campaign / "sites.json")["sites"]}
        # a node keeps the occupancy its first family wrote (native values may be listed)
        occupancy = {
            n["id"]: {k: v for k, v in (n.get("occupancy") or {}).items() if v != native.get(k)}
            for n in network["nodes"]
        }

        def state_of(occ):
            node = next(n for n, o in occupancy.items() if o == occ and nodes[n].get("state_ids"))
            return nodes[node]["state_ids"]

        cls.ids = {
            "finished": "ref",
            "running": state_of({"a0065": "O"})[0],
            "queued": state_of({"a0065": "OOH"})[0],
            "attention": state_of({"a0065": "empty"})[0],
            "continue": state_of({"a0061": "O"})[0],
        }

        def attempt(sid, stage, status, **extra):
            folder = cls.campaign / "states" / sid / stage
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / "attempt.json", {"status": status, **extra})

        for stage in ("01_loose", "02_final"):
            attempt(cls.ids["finished"], stage, "complete")
        attempt(cls.ids["running"], "01_loose", "complete")
        attempt(cls.ids["running"], "02_final", "running")
        attempt(cls.ids["attention"], "01_loose", "complete")
        attempt(cls.ids["attention"], "02_final", "failed", reason="VASP did not write a normal completion footer")
        attempt(cls.ids["continue"], "01_loose", "interrupted", reason="SIGTERM while VASP was running")
        write_json(
            cls.campaign / "submissions.json",
            {
                "schema_version": 1,
                "entries": [
                    {"state_id": cls.ids["running"], "through": "final", "status": "submitted", "job_id": "71"},
                    {"state_id": cls.ids["queued"], "through": "final", "status": "submitted", "job_id": "72"},
                ],
            },
        )
        states = cls.campaign.resolve() / "states"
        (cls.bin / "jobs.txt").write_text(
            f"71|RUNNING|{states / cls.ids['running']}|1:02:03|22:57:57|mn_x|None\n"
            f"72|PENDING|{states / cls.ids['queued']}|0:00|1-00:00:00|mn_y|Priority\n"
            "73|RUNNING|/elsewhere/states/other|0:10|1:00|other|None\n"
        )
        cls.env = patch.dict(os.environ, {"PATH": f"{cls.bin}{os.pathsep}{os.environ.get('PATH', '')}"})
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        cls.tmp.cleanup()

    @POSIX_ONLY
    def test_every_state_lands_in_its_group(self):
        result = campaign_status(self.campaign)
        category = {s["state_id"]: s["category"] for s in result["states"]}
        for expected, sid in self.ids.items():
            self.assertEqual(category[sid], expected, sid)
        self.assertEqual(sum(result["counts"].values()), len(result["states"]))
        self.assertEqual(result["slurm"]["active_jobs"], 2)  # job 73 runs elsewhere
        running = next(s for s in result["states"] if s["state_id"] == self.ids["running"])
        self.assertIn("02_final, job 71 running 1:02:03 (22:57:57 left)", running["detail"])
        self.assertEqual(running["label"], "*O₆₅")
        attention = result["attention"][0]
        self.assertIn(
            "02_final failed: VASP did not write a normal completion footer; use --retry", attention["detail"]
        )
        self.assertIn(
            "interrupted", next(s for s in result["states"] if s["state_id"] == self.ids["continue"])["detail"]
        )
        self.assertIn("02_final (as last submitted; 02_final where never submitted)", result["target"])
        todo = state_list(result, "todo")
        self.assertIn(self.ids["continue"], todo)
        self.assertNotIn(self.ids["finished"], todo)
        self.assertEqual(state_list(result, "active"), sorted([self.ids["running"], self.ids["queued"]]))

    def test_cycles_show_one_entry_per_intermediate(self):
        result = campaign_status(self.campaign)
        aem = next(c for c in result["cycles"] if c["cycle_id"] == "aem_a0065_AEM")
        self.assertEqual(aem["labels"], ["*₆₅", "*OH₆₅", "*O₆₅", "*OOH₆₅"])
        self.assertEqual([("finished" in cats) for cats in aem["progress"]], [False, True, False, False])
        self.assertEqual((aem["n_finished"], aem["n_intermediates"], aem["complete"]), (1, 4, False))
        self.assertEqual(aem["progress"][2][0], "running")
        bf2 = next(c for c in result["cycles"] if c["cycle_id"] == "bifunctional_a0065_a0061_BF2")
        self.assertEqual(bf2["labels"][0], "*₆₅ · μ-O₆₁")
        self.assertEqual(bf2["short"], "BF-II")
        self.assertEqual(set(result["families"]), {"aem", "bifunctional"})

    def test_through_moves_the_target(self):
        result = campaign_status(self.campaign, through="loose", slurm=False)
        category = {s["state_id"]: s["category"] for s in result["states"]}
        self.assertEqual(category[self.ids["running"]], "finished")  # 01_loose is complete
        self.assertEqual(category[self.ids["attention"]], "finished")
        self.assertIn("--through loose", result["target"])

    def test_without_slurm_the_run_records_and_the_ledger_speak(self):
        result = campaign_status(self.campaign, slurm=False)
        category = {s["state_id"]: s for s in result["states"]}
        self.assertEqual(category[self.ids["running"]]["category"], "running")
        self.assertIn("SLURM was not asked", category[self.ids["running"]]["detail"])
        self.assertEqual(category[self.ids["queued"]]["category"], "queued")
        self.assertIn("job 72", category[self.ids["queued"]]["detail"])
        self.assertFalse(result["slurm"]["available"])
        self.assertIn("SLURM: not asked", status_text(result))

    def test_text_page_and_command(self):
        result = campaign_status(self.campaign)
        text = status_text(result, show_cycles=True)
        self.assertIn("✓ finished 1", text)
        self.assertIn("aem_a0065_AEM", text)
        self.assertTrue(plain(text).isascii())
        page = status_page(result)
        self.assertIn('id="c-aem_a0065_AEM"', page)
        self.assertIn("*OOH₆₅ · μ-O₆₁", page)
        self.assertNotIn("http", page.split("</style>")[0])  # nothing loaded from elsewhere
        self.assertNotIn("<script src", page)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["status", str(self.campaign)]), 0)
        self.assertIn("needs you 1", out.getvalue())
        self.assertTrue((self.campaign / "status.html").is_file())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["status", str(self.campaign), "--list", "queued"]), 0)
        self.assertEqual(out.getvalue().split(), [self.ids["queued"]])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["--json", "status", str(self.campaign), "--no-html"]), 0)
        self.assertEqual(json.loads(out.getvalue())["counts"]["queued"], 1)

    def test_terminals_without_unicode_get_ascii(self):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="ascii")
        with patch.object(sys, "stdout", stream):
            self.assertEqual(main(["status", str(self.campaign), "--cycles", "--no-html"]), 0)
            self.assertEqual(main(["show-cycle", "bifunctional_a0065_a0061_BF2", str(self.campaign)]), 0)
            stream.flush()
        text = raw.getvalue().decode("ascii")
        self.assertIn("# finished 1", text)
        self.assertIn("*O65 & mu-O61 + H2O -> *OOH65 & mu-OH61", text)

    def test_show_cycle_reads_as_equations(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["show-cycle", "bifunctional_a0065_a0061_BF2", str(self.campaign)]), 0)
        text = out.getvalue()
        self.assertIn("Bifunctional mechanism II (BF-II)", text)
        self.assertIn("1. *₆₅ · μ-O₆₁ + H₂O → *OH₆₅ · μ-O₆₁ + H⁺ + e⁻", text)
        self.assertIn("*O₆₅ · μ-O₆₁ + H₂O → *OOH₆₅ · μ-OH₆₁", text)
        self.assertIn("chemical", text)
        self.assertIn("Exner", text)


if __name__ == "__main__":
    unittest.main()
