"""Project files: init (scripted and non-interactive), validation, resolution and generation."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from mechanet.fileio import read_json
from mechanet.project import generate, init_project, kpoints_text, load_project, resolve_config, validate_project

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
RUO2 = EXAMPLES / "ruo2_110" / "slab" / "RuO2_110.vasp"
JOB = "#!/bin/bash\n#SBATCH -N 1\n#SBATCH -n 64\n#SBATCH -t 12:00:00\nmodule load vasp\nsrun vasp_std\n"


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {"MECHANET_CONFIG_DIR": str(self.root / "config")})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_non_interactive_init_writes_a_complete_project(self):
        result = init_project(RUO2, self.root / "p", interactive=False, output=lambda *a: None)
        project = load_project(self.root / "p")
        self.assertEqual(result["sites"], 4)
        self.assertEqual(project["reaction"], "oer")
        self.assertEqual([s["id"] for s in project["sites"]], ["a0016", "a0019", "m0038", "m0046"])
        self.assertEqual(project["slab"]["fixed"], {"source": "file"})
        self.assertEqual(project["vasp"]["potcar_variants"]["Ru"], "Ru_pv")
        self.assertIsNone(project["vasp"]["dft_u"])  # Ru has no Materials Project U
        self.assertEqual(project["origins"]["sites"], "auto")
        for name in (
            "slab/RuO2_110.vasp",
            "slab/inspection.json",
            "slab/site_map.html",
            "slab/site_map.svg",
            "inputs/INCAR_loose",
            "inputs/INCAR_final",
            "inputs/KPOINTS",
            "inputs/references.yaml",
        ):
            self.assertTrue((self.root / "p" / name).is_file(), name)
        self.assertIn("IDIPOL = 3", (self.root / "p" / "inputs" / "INCAR_final").read_text())
        self.assertEqual(read_json(self.root / "p" / "slab" / "inspection.json")["file"], "slab/RuO2_110.vasp")
        with self.assertRaisesRegex(FileExistsError, "--force"):
            init_project(RUO2, self.root / "p", interactive=False, output=lambda *a: None)

    def test_scripted_answers_are_recorded_as_origins(self):
        job = self.root / "job.sh"
        job.write_text(JOB)
        answers = iter(["n", "a0016,m0038", "2", "aem", "", "hpc"])  # sites? keep; reaction; scope; freq; name
        printed = []
        result = init_project(
            RUO2,
            self.root / "q",
            interactive=True,
            job_script=job,
            input_func=lambda prompt: next(answers),
            output=printed.append,
        )
        project = load_project(self.root / "q")
        self.assertEqual([s["id"] for s in project["sites"]], ["a0016", "m0038"])
        self.assertEqual(project["reaction"], "oer")
        self.assertEqual(project["scope"]["families"], ["aem"])
        self.assertEqual(project["cluster"], "hpc")
        self.assertEqual(project["origins"]["sites"], "overridden")
        self.assertEqual(project["origins"]["scope.families"], "overridden")
        self.assertEqual(result["cluster"], "hpc")
        self.assertTrue((self.root / "config" / "clusters" / "hpc.yaml").is_file())

    def test_own_incar_templates_keep_their_u_and_moments(self):
        ldh = EXAMPLES / "nife_ldh"
        init_project(
            ldh / "slab" / "CONTCAR_seed",
            self.root / "ldh",
            interactive=False,
            output=lambda *a: None,
            incar_loose=ldh / "inputs" / "INCAR_loose",
            incar_final=ldh / "inputs" / "INCAR_final",
            kpoints=ldh / "inputs" / "KPOINTS",
        )
        project = load_project(self.root / "ldh")
        self.assertIsNone(project["vasp"]["dft_u"])  # the templates' LDAU arrays are used
        self.assertIsNone(project["vasp"]["magmom_by_element"])
        self.assertEqual(project["origins"]["vasp.dft_u"], "INCAR templates")
        self.assertEqual(
            (self.root / "ldh" / "inputs" / "INCAR_final").read_text(), (ldh / "inputs" / "INCAR_final").read_text()
        )
        # without templates, an oxide with Fe and Ni gets the Materials Project U proposal
        init_project(ldh / "slab" / "CONTCAR_seed", self.root / "ldh2", interactive=False, output=lambda *a: None)
        self.assertEqual(load_project(self.root / "ldh2")["vasp"]["dft_u"]["Ni"]["U"], 6.2)

    def test_validation_messages(self):
        errors = validate_project(
            {
                "mechanet": 2,
                "name": "",
                "slab": {"file": "x", "vacuum_axis": 5, "surface_sign": 1, "bogus": 1},
                "sites": "a0001",
                "scope": {"families": []},
                "inputs": {},
                "extra": 1,
            }
        )
        text = "\n".join(errors)
        for part in (
            "schema version",
            "unknown top-level keys: 'extra'",
            "`name`",
            "`reaction`",
            "unknown slab keys: 'bogus'",
            "vacuum_axis",
            "`sites`",
            "scope.families",
            "`inputs`",
        ):
            self.assertIn(part, text)

    def test_typos_inside_sections_are_refused_with_a_suggestion(self):
        shutil.copytree(EXAMPLES / "ruo2_110", self.root / "ruo2")
        path = self.root / "ruo2" / "project.yaml"
        data = yaml.safe_load(path.read_text())
        data["thermodynamics"]["ph"] = 13
        data["vasp"]["comon_overrides"] = {"NCORE": 16}
        data["frequency"]["centrl"] = "refine"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        with self.assertRaises(ValueError) as caught:
            load_project(path)
        message = str(caught.exception)
        for hint in ("'ph' (did you mean 'pH'?)", "'comon_overrides' (did you mean", "'centrl' (did you mean"):
            self.assertIn(hint, message)

    def test_values_that_would_fail_later_are_refused_now(self):
        shutil.copytree(EXAMPLES / "ruo2_110", self.root / "ruo2")
        path = self.root / "ruo2" / "project.yaml"
        data = yaml.safe_load(path.read_text())
        data["vasp"]["stage_overrides"] = {"01_loos": {"NSW": 80}}
        data["vasp"]["command"] = "srun vasp_std > vasp.out"
        data["frequency"]["central"] = "finall"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        with self.assertRaises(ValueError) as caught:
            load_project(path)
        message = str(caught.exception)
        for part in ("'01_loos' (did you mean '01_loose'?)", "vasp.command: '>'", "'finall' (did you mean 'final'?)"):
            self.assertIn(part, message)
        data["vasp"].pop("stage_overrides")
        data["frequency"]["central"] = "final"
        data["vasp"]["command"] = "srun -n 4 vasp_gam"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        config = resolve_config(load_project(path), cluster=False)
        self.assertEqual(config["vasp"]["command"], ["srun", "-n", "4", "vasp_gam"])
        data["vasp"]["command"] = ["srun", "-n", 4, "vasp_gam"]  # a number in the list becomes text
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        config = resolve_config(load_project(path), cluster=False)
        self.assertEqual(config["vasp"]["command"], ["srun", "-n", "4", "vasp_gam"])
        data["vasp"]["command"] = 5
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        with self.assertRaisesRegex(ValueError, "vasp.command: give a list of words"):
            load_project(path)

    def test_a_broken_job_script_fails_before_any_question(self):
        job = self.root / "job.sh"
        job.write_text("#!/bin/bash\n#SBATCH -t 1:00:00\nmpirun -np $(nproc) vasp_std\n")

        def no_questions(prompt):
            raise AssertionError(f"asked {prompt!r}")

        with self.assertRaisesRegex(ValueError, "command substitution"):
            init_project(RUO2, self.root / "b", job_script=job, interactive=True, input_func=no_questions)
        self.assertFalse((self.root / "b").exists())

    def test_every_default_setting_is_a_known_key(self):
        from mechanet.project import DEFAULT_SECTIONS, SECTION_KEYS

        for section, defaults in DEFAULT_SECTIONS.items():
            self.assertLessEqual(set(defaults), SECTION_KEYS[section], section)

    def test_init_checks_options_before_asking_and_never_clobbers_inputs(self):
        quiet = {"interactive": False, "output": lambda *a: None}
        with self.assertRaisesRegex(ValueError, "Give both --incar-loose and --incar-final"):
            init_project(RUO2, self.root / "a", incar_final=RUO2, **quiet)
        with self.assertRaisesRegex(ValueError, "available: pbe-d3"):
            init_project(RUO2, self.root / "a", preset="pbe", **quiet)
        with self.assertRaisesRegex(FileNotFoundError, "No cluster profile 'nope'"):
            init_project(RUO2, self.root / "a", cluster="nope", **quiet)
        self.assertFalse((self.root / "a").exists())  # nothing written before the checks pass
        # `mechanet init POSCAR`: the default folder name is taken by the slab file itself.
        work = self.root / "work"
        work.mkdir()
        shutil.copy(RUO2, work / "POSCAR")
        cwd = os.getcwd()
        os.chdir(work)
        try:
            result = init_project("POSCAR", **quiet)
        finally:
            os.chdir(cwd)
        self.assertEqual(Path(result["project"]), Path("POSCAR_project") / "project.yaml")
        self.assertTrue((work / "POSCAR_project" / "project.yaml").is_file())
        # --force rewrites project.yaml but keeps edited inputs and computed references.
        references = work / "POSCAR_project" / "inputs" / "references.yaml"
        references.write_text("H2: {energy_e0_eV: -6.77}\n")
        incar = work / "POSCAR_project" / "inputs" / "INCAR_final"
        incar.write_text(incar.read_text() + "NCORE = 16\n")
        printed = []
        init_project(work / "POSCAR", work / "POSCAR_project", force=True, interactive=False, output=printed.append)
        self.assertEqual(references.read_text(), "H2: {energy_e0_eV: -6.77}\n")
        self.assertIn("NCORE = 16", incar.read_text())
        self.assertTrue(any("kept the existing inputs/INCAR_loose" in str(line) for line in printed))

    def test_init_picks_a_cluster_profile_only_when_it_is_unambiguous(self):
        from mechanet import clusters

        quiet = {"interactive": False, "output": lambda *a: None}
        for name in ("alpha", "beta"):
            job = self.root / f"{name}.sh"
            job.write_text(JOB.replace("-n 64", "-n 32" if name == "beta" else "-n 64"))
            clusters.import_job_script(job, name)
        init_project(RUO2, self.root / "two", **quiet)
        self.assertIsNone(load_project(self.root / "two")["cluster"])
        (self.root / "config" / "clusters" / "beta.yaml").unlink()
        init_project(RUO2, self.root / "one", **quiet)
        project = load_project(self.root / "one")
        self.assertEqual(project["cluster"], "alpha")
        self.assertEqual(project["origins"]["cluster"], "auto")
        self.assertIn("cluster: alpha", (self.root / "one" / "project.yaml").read_text())
        self.assertIn("[auto] profile name", (self.root / "one" / "project.yaml").read_text())
        # A job script whose name is taken by a different profile is refused before the interview.
        other = self.root / "alpha_job.sh"
        other.write_text(JOB.replace("12:00:00", "6:00:00"))
        with self.assertRaisesRegex(ValueError, "differs from"):
            init_project(RUO2, self.root / "three", job_script=other, cluster="alpha", **quiet)
        # The same script again reuses its profile.
        same = self.root / "again.sh"
        same.write_text(JOB)
        init_project(RUO2, self.root / "four", job_script=same, cluster="alpha", **quiet)
        self.assertEqual(load_project(self.root / "four")["cluster"], "alpha")

    def test_interactive_init_asks_again_for_unknown_site_ids(self):
        answers = iter(["n", "a0016,zz99", "a0016", "", "", ""])
        printed = []
        init_project(
            RUO2,
            self.root / "s",
            interactive=True,
            input_func=lambda prompt: next(answers),
            output=printed.append,
        )
        self.assertEqual([s["id"] for s in load_project(self.root / "s")["sites"]], ["a0016"])
        self.assertTrue(any("unknown site IDs ['zz99']" in str(line) for line in printed))

    def test_generate_finds_the_project_and_relative_cluster_files(self):
        shutil.copytree(EXAMPLES / "ruo2_110", self.root / "ruo2")
        shutil.rmtree(self.root / "ruo2" / "campaign", ignore_errors=True)
        (self.root / "ruo2" / "hpc.yaml").write_text(
            "sbatch: {time: '24:00:00', ntasks: 64}\nvasp_command: srun vasp_std\n"
        )
        path = self.root / "ruo2" / "project.yaml"
        data = yaml.safe_load(path.read_text())
        data["cluster"] = "hpc.yaml"
        data["vasp"]["command"] = ["srun", "vasp_gam"]
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        cwd = os.getcwd()
        os.chdir(self.root / "ruo2" / "slab")
        try:
            report = generate(families=["aem"])
        finally:
            os.chdir(cwd)
        config = read_json(Path(report["campaign"]) / "config.json")
        self.assertEqual(config["scheduler"]["sbatch"]["ntasks"], 64)
        self.assertEqual(config["vasp"]["command"], ["srun", "vasp_gam"])  # project.yaml wins
        self.assertTrue(any("vasp.command of project.yaml" in w for w in report["warnings"]))

    def test_resolve_and_generate_the_example(self):
        shutil.copytree(EXAMPLES / "ruo2_110", self.root / "ruo2")
        project = load_project(self.root / "ruo2" / "project.yaml")
        config = resolve_config(project, cluster=False)
        self.assertEqual(config["thermodynamics"]["potential_scale"], "RHE")
        self.assertEqual(config["frequency"]["central"], "final")
        self.assertTrue(Path(config["inputs"]["seed"]).is_absolute())
        self.assertNotIn("scheduler", config)
        report = generate(self.root / "ruo2" / "project.yaml", families=["aem"])
        campaign = self.root / "ruo2" / "campaign"
        self.assertEqual(report["campaign"], str(campaign))
        self.assertTrue((campaign / "project.snapshot.yaml").is_file())
        self.assertIn("No cluster profile", report["warnings"][0])
        self.assertEqual(read_json(campaign / "config.json")["inputs"]["seed"], "inputs/seed.vasp")
        self.assertEqual(report["complete_stoichiometric_cycles"], 4)  # AEM and bound-OO on two CUS sites

    def test_missing_cluster_and_unknown_family(self):
        shutil.copytree(EXAMPLES / "ruo2_110", self.root / "ruo2")
        path = self.root / "ruo2" / "project.yaml"
        data = yaml.safe_load(path.read_text())
        data["cluster"] = "missing"
        path.write_text(yaml.safe_dump(data))
        with self.assertRaisesRegex(FileNotFoundError, "No cluster profile 'missing'"):
            resolve_config(load_project(path))
        data["cluster"], data["scope"]["families"] = None, ["aem", "no_such_family"]
        path.write_text(yaml.safe_dump(data))
        with self.assertRaisesRegex(ValueError, "no_such_family"):
            resolve_config(load_project(path))

    def test_kpoints_mesh(self):
        text, mesh = kpoints_text([[6.0, 0, 0], [0, 12.0, 0], [0, 0, 30.0]], 2, spacing=0.25)
        self.assertEqual(mesh, [5, 3, 1])
        self.assertTrue(text.splitlines()[2].startswith("Gamma"))


if __name__ == "__main__":
    unittest.main()
