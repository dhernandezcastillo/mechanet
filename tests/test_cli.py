"""Command line: every command runs, human and JSON output, errors exit with status 2."""

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from mechanet.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
JOB = """#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH -t 08:00:00
#SBATCH --mail-user=someone@example.org
module load vasp/6.4.2
srun vasp_std
"""


def run(*args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main([str(a) for a in args])
    return code, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env = patch.dict(os.environ, {"MECHANET_CONFIG_DIR": str(cls.tmp / "config")})
        cls.env.start()
        (cls.tmp / "job.sh").write_text(JOB)
        code, out, err = run(
            "init",
            EXAMPLES / "ruo2_110" / "slab" / "RuO2_110.vasp",
            cls.tmp / "ruo2",
            "--yes",
            "--scope",
            "aem",
            "--job-script",
            cls.tmp / "job.sh",
        )
        assert code == 0, err
        cls.init_output = out
        cls.project = cls.tmp / "ruo2"
        code, out, err = run("generate", cls.project / "project.yaml")
        assert code == 0, err
        cls.campaign = cls.project / "campaign"

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        shutil.rmtree(cls.tmp)

    def test_init_reports_next_steps_and_imports_the_cluster(self):
        self.assertIn("next:", self.init_output)
        self.assertIn("mechanet generate", self.init_output)
        project = yaml.safe_load((self.project / "project.yaml").read_text())
        self.assertEqual(project["cluster"], "job")
        self.assertEqual(
            json.loads((self.campaign / "config.json").read_text())["vasp"]["command"], ["srun", "vasp_std"]
        )

    def test_inspect_human_and_json(self):
        code, out, _ = run("inspect", EXAMPLES / "pt_111" / "slab" / "Pt_111.vasp")
        self.assertEqual(code, 0)
        self.assertIn("proposed reactive sites (9)", out)
        code, out, _ = run("--json", "inspect", EXAMPLES / "pt_111" / "slab" / "Pt_111.vasp")
        self.assertEqual(len(json.loads(out)["open_metal_sites"]), 9)

    def test_campaign_listing_commands(self):
        for args, expected in (
            (("plan", self.campaign), "relaxation_jobs"),
            (("list", self.campaign), "frequency_dof"),
            (("cycles", self.campaign), "aem_m0038_AEM"),
            (("show-cycle", "aem_m0038_AEM", self.campaign), "releases H+ + O2 + e-"),
            (("validate", self.campaign), "valid:"),
            (("plugins",), "cer_volmer_heyrovsky"),
            (("report", self.campaign), "## Network"),
        ):
            code, out, err = run(*args)
            self.assertEqual(code, 0, err)
            self.assertIn(expected, out, args)

    def test_campaign_is_found_from_the_project_directory(self):
        cwd = os.getcwd()
        try:
            os.chdir(self.project)
            code, out, err = run("--json", "cycles")
        finally:
            os.chdir(cwd)
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["total_matching"], 4)

    def test_submit_preview_writes_job_scripts_only(self):
        code, out, err = run("submit", self.campaign, "--pilot")
        self.assertEqual(code, 0, err)
        self.assertIn("Preview only", out)
        scripts = list((self.campaign / "states").glob("*/run_job.sh"))
        self.assertTrue(scripts)
        text = scripts[0].read_text()
        self.assertIn("#SBATCH --time=08:00:00", text)
        self.assertIn("module load vasp/6.4.2", text)
        self.assertNotIn("someone@example.org", text)
        self.assertFalse((self.campaign / "method_card.frozen.json").exists())

    def test_cluster_commands(self):
        code, out, _ = run("cluster", "list")
        self.assertEqual(code, 0)
        self.assertIn("job", out)
        code, out, _ = run("cluster", "show", "job")
        self.assertIn("#SBATCH --ntasks-per-node=48", out)
        code, out, err = run("cluster", "import", self.tmp / "job.sh", "--name", "second")
        self.assertEqual(code, 0, err)
        code, out, err = run("cluster", "use", "second", self.campaign)
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads((self.campaign / "config.json").read_text())["scheduler"]["profile"], "second")

    def test_cluster_show_prints_the_real_job_script_and_list_flags_problems(self):
        code, out, err = run("--json", "cluster", "show", "job")
        self.assertEqual(code, 0, err)
        script = json.loads(out)["job_script"]
        self.assertTrue(script.startswith("#!/bin/bash\n#SBATCH --job-name=mn_STATE"))
        self.assertIn("module load vasp/6.4.2", script)
        code, out2, _ = run("cluster", "show", "job", "--json")  # --json after the command too
        self.assertEqual(json.loads(out2)["job_script"], script)
        folder = self.tmp / "config" / "clusters"
        (folder / "tabbed.yaml").write_text("sbatch:\n\ttime: '1:00:00'\n")
        code, _, err = run("cluster", "show", "tabbed")
        self.assertEqual(code, 2)
        self.assertIn("line 2, column 1", err)
        self.assertIn("spaces, not tabs", err)
        code, out, _ = run("cluster", "list")
        self.assertIn("tabbed: unreadable", out)
        (folder / "tabbed.yaml").unlink()

    def test_examples_command_writes_the_packaged_slabs(self):
        code, out, err = run("examples")
        self.assertEqual(code, 0, err)
        for name, folder in (("RuO2_110", "ruo2_110"), ("IrO2_110", "iro2_110"), ("Pt_111", "pt_111")):
            self.assertIn(name, out)
            code, out2, err = run("examples", name.lower(), "--out", self.tmp / "slabs")
            self.assertEqual(code, 0, err)
            self.assertEqual(
                (self.tmp / "slabs" / f"{name}.vasp").read_text(),
                (EXAMPLES / folder / "slab" / f"{name}.vasp").read_text(),
            )
        code, _, err = run("examples", "RuO2_110", "--out", self.tmp / "slabs")
        self.assertEqual(code, 2)
        self.assertIn("--force", err)

    def test_submit_without_a_cluster_profile_says_how_to_set_one(self):
        config = json.loads((self.campaign / "config.json").read_text())
        saved = json.dumps(config)
        config.pop("scheduler")
        (self.campaign / "config.json").write_text(json.dumps(config))
        try:
            code, _, err = run("submit", self.campaign, "--pilot")
        finally:
            (self.campaign / "config.json").write_text(saved)
        self.assertEqual(code, 2)
        self.assertIn("no cluster profile", err)
        self.assertIn("mechanet cluster use NAME", err)

    def test_attention_reasons_are_printed_in_full(self):
        ledger = self.campaign / "submissions.json"
        saved = ledger.read_text() if ledger.exists() else None
        ledger.write_text(json.dumps({"schema_version": 1, "entries": [{"state_id": "ref", "status": "submitting"}]}))
        try:
            code, out, err = run("submit", self.campaign, "--pilot")
        finally:
            if saved is None:
                ledger.unlink()
            else:
                ledger.write_text(saved)
        self.assertEqual(code, 0, err)
        self.assertIn("ref (skipped_submission_unknown):", out)
        self.assertIn("mechanet resolve-submission ref", out)
        self.assertIn("--job-id ID", out)

    def test_plugin_commands_get_json_and_may_define_it_themselves(self):
        from mechanet import plugins
        from mechanet.cli import build_parser

        saved = plugins._DEFAULT
        self.addCleanup(setattr, plugins, "_DEFAULT", saved)
        registry = plugins.default_registry(fresh=True)

        def register(subparsers, registry):
            p = subparsers.add_parser("hello")
            p.add_argument("--json", action="store_true")
            p.set_defaults(handler=lambda args, registry: {"hello": True})

        registry.add_command("hello", register)
        parser = build_parser(registry)
        args = parser.parse_args(["hello", "--json"])
        self.assertTrue(args.json)
        self.assertEqual(args.handler(args, registry), {"hello": True})
        self.assertFalse(registry.errors)
        # the plugin's own --json must not change the other commands, and the global flag stays separate
        self.assertFalse(hasattr(parser.parse_args(["plan"]), "json"))
        self.assertTrue(parser.parse_args(["plan", "--json"]).json)
        self.assertTrue(parser.parse_args(["--json", "hello"]).json_global)

    def test_references_template_and_molecules(self):
        code, out, err = run("references", "--reaction", "cer", "--out", self.tmp / "refs.yaml")
        self.assertEqual(code, 0, err)
        refs = yaml.safe_load((self.tmp / "refs.yaml").read_text())
        self.assertIn("Cl2", refs)
        self.assertIsNone(refs["H2"]["energy_e0_eV"])
        code, out, err = run("references", "--molecules", self.tmp / "molecules", "--campaign", self.campaign)
        self.assertEqual(code, 0, err)
        incar = (self.tmp / "molecules" / "H2O" / "freq" / "INCAR").read_text()
        self.assertIn("IBRION = 5", incar)
        self.assertNotIn("LDIPOL", incar)
        self.assertTrue((self.tmp / "molecules" / "H2" / "relax" / "POSCAR").is_file())

    def test_pathways_command(self):
        code, out, err = run("--json", "pathways", self.campaign, "--max-length", "5")
        self.assertEqual(code, 0, err)
        summary = json.loads(out)["summary"]
        self.assertEqual(summary["curated_cycles_recovered"], summary["curated_cycles_total"])

    def test_errors_exit_with_status_two(self):
        code, out, err = run("show-cycle", "no_such_cycle", self.campaign)
        self.assertEqual(code, 2)
        self.assertIn("Unknown cycle", err)
        code, _, err = run("plan", self.tmp / "nowhere")
        self.assertEqual(code, 2)
        self.assertIn("not a campaign", err)
        code, _, err = run("thermo", self.campaign, "--references", self.tmp / "missing.yaml")
        self.assertEqual(code, 2)
        self.assertIn("References file not found", err)


if __name__ == "__main__":
    unittest.main()
