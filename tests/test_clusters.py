"""Cluster profiles: importing job scripts, validation, applying a profile to a campaign."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mechanet import clusters
from mechanet.slurm import job_script_text, sbatch_options, setup_lines

SCRIPT_MPIRUN = """#!/bin/bash
#SBATCH -J NiFeLDH
#SBATCH -N 1
#SBATCH --ntasks-per-node=112
#SBATCH -t 23:59:00
#SBATCH --mail-type=FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=someone@example.org

module use /opt/site/modulefiles
module load vasp/6.4.1
export OMP_NUM_THREADS=1

mpirun -n 112 vasp_std
"""

SCRIPT_SRUN = """#!/bin/bash -l
#SBATCH --job-name=relax
#SBATCH --nodes=2
#SBATCH --ntasks 256
#SBATCH --time=2-00:00:00
#SBATCH --partition=standard --account=proj123
#SBATCH --exclusive
#SBATCH -o slurm.%j.out
module purge
module load intel/2023 intel-mpi vasp/6.4.2
source /home/me/venvs/dft/bin/activate
cd $SLURM_SUBMIT_DIR
ulimit -s unlimited
time srun --cpu-bind=cores /apps/vasp/bin/vasp_std > vasp.out 2>&1
cp CONTCAR CONTCAR.bak
"""


class ClusterProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {"MECHANET_CONFIG_DIR": str(self.root / "config")})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def script(self, text, name="job.sh"):
        path = self.root / name
        path.write_text(text)
        return path

    def test_import_mpirun_script_drops_personal_options(self):
        path, profile, notes = clusters.import_job_script(self.script(SCRIPT_MPIRUN), "lab")
        self.assertEqual(path, self.root / "config" / "clusters" / "lab.yaml")
        self.assertEqual(profile["sbatch"], {"nodes": 1, "ntasks-per-node": 112, "time": "23:59:00"})
        self.assertEqual(
            profile["setup"],
            ["module use /opt/site/modulefiles", "module load vasp/6.4.1", "export OMP_NUM_THREADS=1"],
        )
        self.assertNotIn("environment", profile)
        self.assertEqual(profile["vasp_command"], ["mpirun", "-n", "112", "vasp_std"])
        text = path.read_text()
        self.assertNotIn("someone@example.org", text)
        self.assertNotIn("mail-user=", text)
        self.assertTrue(any("ignored_lines" in note for note in notes))

    def test_import_srun_script_keeps_order_and_ignores_job_steps(self):
        _, profile, _ = clusters.import_job_script(self.script(SCRIPT_SRUN), "big")
        self.assertEqual(
            profile["sbatch"],
            {
                "nodes": 2,
                "ntasks": 256,
                "time": "2-00:00:00",
                "partition": "standard",
                "account": "proj123",
                "exclusive": True,
            },
        )
        self.assertEqual(profile["setup"][:2], ["module purge", "module load intel/2023 intel-mpi vasp/6.4.2"])
        self.assertEqual(profile["shell"], "/bin/bash -l")
        self.assertEqual(profile["vasp_command"], ["srun", "--cpu-bind=cores", "/apps/vasp/bin/vasp_std"])
        self.assertIn("cp CONTCAR CONTCAR.bak", profile["ignored_lines"])
        self.assertIn("cd $SLURM_SUBMIT_DIR", profile["ignored_lines"])

    def test_import_needs_a_vasp_line_and_a_walltime(self):
        with self.assertRaisesRegex(ValueError, "No VASP launch line"):
            clusters.import_job_script(self.script("#SBATCH -t 1:00:00\necho hi\n"), "x")
        with self.assertRaisesRegex(ValueError, "walltime"):
            clusters.import_job_script(self.script("srun vasp_gam\n"), "y")
        with self.assertRaisesRegex(FileExistsError, "--force"):
            clusters.import_job_script(self.script(SCRIPT_MPIRUN), "lab")
            clusters.import_job_script(self.script(SCRIPT_MPIRUN), "lab")

    def test_validation_reports_every_problem(self):
        with self.assertRaises(ValueError) as caught:
            clusters.validate_profile(
                {"sbatch": {"job-name": "x", "BAD KEY": 1}, "vasp_command": [], "max_active_jobs": -2, "colour": "blue"}
            )
        message = str(caught.exception)
        for part in (
            "unknown keys ['colour']",
            "--job-name",
            "invalid sbatch option",
            "sbatch.time",
            "vasp_command is required",
            "max_active_jobs",
        ):
            self.assertIn(part, message)
        with self.assertRaisesRegex(ValueError, "quoted string"):
            clusters.validate_profile({"sbatch": {"time": 86340}, "vasp_command": "srun vasp_std"})

    def test_listing_loading_and_applying(self):
        clusters.import_job_script(self.script(SCRIPT_SRUN), "big")
        self.assertEqual([p["name"] for p in clusters.list_profiles()], ["big"])
        profile = clusters.load_profile("big")
        config = {"vasp": {}, "scheduler": {"min_remaining_s": {"relaxation": 600}}}
        clusters.apply_profile(config, profile)
        self.assertEqual(config["vasp"]["command"], ["srun", "--cpu-bind=cores", "/apps/vasp/bin/vasp_std"])
        self.assertEqual(config["scheduler"]["profile"], "big")
        self.assertEqual(config["scheduler"]["min_remaining_s"], {"relaxation": 600})
        self.assertEqual(sbatch_options(config["scheduler"])["time"], "2-00:00:00")
        self.assertEqual(setup_lines(config["scheduler"])[0], "module purge")
        with self.assertRaisesRegex(FileNotFoundError, "cluster import"):
            clusters.load_profile("nope")

    def test_exports_stay_verbatim_and_in_order_and_the_job_environment_is_right(self):
        script = (
            "#!/bin/bash\n#SBATCH -t 12:00:00\n#SBATCH -n 8\n"
            "export MODULEPATH=/opt/mods:$MODULEPATH\n"
            "module load vasp\n"
            "export PATH=$HOME/bin:$PATH\n"
            "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n"
            "export MKL_NUM_THREADS=1   # pure MPI\n"
            "export A=2 B=3\n"
            "VASP_BIN=/opt/vasp/bin/vasp_std\n"
            "mpirun -np $SLURM_NTASKS $VASP_BIN\n"
        )
        _, profile, _ = clusters.import_job_script(self.script(script), "exports")
        self.assertEqual(
            profile["setup"],
            [
                "export MODULEPATH=/opt/mods:$MODULEPATH",
                "module load vasp",
                "export PATH=$HOME/bin:$PATH",
                "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK",
                "export MKL_NUM_THREADS=1   # pure MPI",
                "export A=2 B=3",
                "export VASP_BIN=/opt/vasp/bin/vasp_std",
            ],
        )
        self.assertEqual(profile["vasp_command"], ["mpirun", "-np", "$SLURM_NTASKS", "$VASP_BIN"])
        state = self.root / "campaign" / "states" / "s1"
        state.mkdir(parents=True)
        text = job_script_text(clusters.scheduler_section(profile), state)
        self.assertNotIn('export OMP_NUM_THREADS="1"', text)  # the script's own setting wins
        self.assertIn('export OPENBLAS_NUM_THREADS="1"', text)
        # Run the script's environment part in bash (module stubbed, exec replaced by a printout).
        lines = text.replace("#!/bin/bash\n", "#!/bin/bash\nmodule() { :; }\n").splitlines()
        lines[-1] = 'echo "$PATH|$OMP_NUM_THREADS|$MKL_NUM_THREADS|$A$B|$VASP_BIN|$MODULEPATH"'
        env = {"PATH": "/usr/bin:/bin", "HOME": "/home/u", "SLURM_CPUS_PER_TASK": "4", "MODULEPATH": "/m"}
        out = subprocess.run(["bash", "-c", "\n".join(lines)], env=env, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "/home/u/bin:/usr/bin:/bin|4|1|23|/opt/vasp/bin/vasp_std|/opt/mods:/m")
        self.assertEqual(
            clusters.expand_command(profile["vasp_command"], {"SLURM_NTASKS": "8", "VASP_BIN": "/v/vasp_std"}),
            ["mpirun", "-np", "8", "/v/vasp_std"],
        )

    def test_launch_line_variants(self):
        cases = {
            "srun vasp_std &\nwait\n": ["srun", "vasp_std"],
            "srun vasp_std && echo done\n": ["srun", "vasp_std"],
            "mpirun -np ${SLURM_NTASKS:-1} vasp_gam > out.log 2>&1\n": [
                "mpirun",
                "-np",
                "${SLURM_NTASKS:-1}",
                "vasp_gam",
            ],
            "OMP_NUM_THREADS=1 exec srun vasp_ncl | tee log\n": ["srun", "vasp_ncl"],
            "mpirun -np 4 \\\n    /opt/vasp.6.4/bin/vasp_std\n": ["mpirun", "-np", "4", "/opt/vasp.6.4/bin/vasp_std"],
            "srun $VASP_EXE\n": ["srun", "$VASP_EXE"],
        }
        for body, command in cases.items():
            with self.subTest(body=body):
                profile, _, notes = clusters.parse_job_script("#SBATCH -t 60\n" + body)
                self.assertEqual(profile["vasp_command"], command)
                self.assertEqual(profile["sbatch"]["time"], "60")
        profile, _, notes = clusters.parse_job_script("#SBATCH -t 60\nOMP_NUM_THREADS=1 srun vasp_std > o 2>&1\n")
        self.assertEqual(profile["setup"], ["export OMP_NUM_THREADS=1"])
        self.assertTrue(any("'> o'" in n and "'2>&1'" in n for n in notes))
        with self.assertRaisesRegex(ValueError, "command substitution"):
            clusters.parse_job_script("#SBATCH -t 60\nmpirun -np $(nproc) vasp_std\n")
        conda = 'eval "$(conda shell.bash hook)"\nconda activate mn\nsrun vasp_std\n'
        self.assertEqual(
            clusters.parse_job_script("#SBATCH -t 60\n" + conda)[0]["setup"],
            ['eval "$(conda shell.bash hook)"', "conda activate mn"],
        )
        with self.assertRaisesRegex(ValueError, "only \\$VAR"):
            clusters.validate_profile({"sbatch": {"time": "1:00:00"}, "vasp_command": "srun -n ${N:=4} vasp_std"})
        with self.assertRaisesRegex(ValueError, "shell operator"):
            clusters.validate_profile({"sbatch": {"time": "1:00:00"}, "vasp_command": ["srun", "vasp_std", "&&"]})

    def test_sbatch_comments_times_and_per_submission_options(self):
        profile, ignored, _ = clusters.parse_job_script(
            "#!/bin/bash\n#SBATCH --time=1440  # don't change this\n#SBATCH --begin=now+1hour\n"
            "#SBATCH --hold\n#SBATCH --deadline=2030-01-01\n#SBATCH --qos=normal\nsrun vasp_std\n"
        )
        self.assertEqual(profile["sbatch"], {"time": "1440", "qos": "normal"})
        self.assertEqual(len([line for line in ignored if "per submission" in line]), 3)
        with self.assertRaisesRegex(ValueError, "line 2"):
            clusters.parse_job_script("#!/bin/bash\n#SBATCH --partition='open\nsrun vasp_std\n")
        path, profile, _ = clusters.import_job_script(self.script("#SBATCH -t 90\nsrun vasp_std\n"), "short")
        self.assertIn("time: '90'", path.read_text())
        self.assertEqual(clusters.load_profile("short")["sbatch"]["time"], "90")
        self.assertIsNone(clusters.load_profile("short").get("shell"))

    def test_walltimes_stop_signal_and_minimum_times(self):
        for text, seconds in {
            "30": 1800,
            "30:15": 1815,
            "2:00:00": 7200,
            "1-12": 129600,
            "1-0:30": 88200,
            "2-00:00:00": 172800,
            "UNLIMITED": None,
        }.items():
            self.assertEqual(clusters.walltime_seconds(text), seconds)
        with self.assertRaisesRegex(ValueError, "SLURM time"):
            clusters.walltime_seconds("1h")
        self.assertEqual(clusters.stop_lead({"sbatch": {"time": "00:30:00"}}), 450)
        self.assertEqual(clusters.stop_lead({"sbatch": {"time": "24:00:00"}}), 1800)
        self.assertEqual(clusters.stop_lead({"sbatch": {"time": "1:00:00"}, "stop_signal_lead_s": 300}), 300)
        errors, warnings = clusters.timing_report({"sbatch": {"time": "00:30:00"}, "stop_signal_lead_s": 1800})
        self.assertTrue(errors and "every job" in errors[0])
        errors, warnings = clusters.timing_report({"sbatch": {"time": "1:00:00"}})
        self.assertEqual(errors, [])
        self.assertTrue(any("single_point" in w and "one per job" in w for w in warnings))
        with self.assertRaisesRegex(ValueError, "every job"):
            clusters.validate_profile(
                {"sbatch": {"time": "00:20:00"}, "vasp_command": "srun vasp_std", "stop_signal_lead_s": 1800}
            )
        for bad, message in (
            (600, "must be a mapping"),
            ({"relax": 10}, "unknown kinds"),
            ({"relaxation": "ten"}, "numbers of seconds"),
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, message):
                clusters.validate_profile(
                    {"sbatch": {"time": "24:00:00"}, "vasp_command": "srun vasp_std", "min_remaining_s": bad}
                )

    def test_listing_marks_broken_profiles(self):
        clusters.import_job_script(self.script(SCRIPT_SRUN), "good")
        folder = clusters.clusters_dir()
        (folder / "tabs.yaml").write_text("sbatch:\n\ttime: '1:00:00'\n")
        (folder / "numeric.yaml").write_text("sbatch: {time: 86340}\nvasp_command: srun vasp_std\n")
        rows = {row["name"]: row for row in clusters.list_profiles()}
        self.assertEqual(rows["good"]["status"], "ok")
        self.assertEqual(rows["tabs"]["status"], "unreadable")
        self.assertEqual(rows["numeric"]["status"], "invalid")
        self.assertIn("quoted string", rows["numeric"]["problem"])

    def test_variables_with_several_words_blocks_and_wrappers(self):
        script = (
            "#!/bin/bash\n#SBATCH -t 0\n#SBATCH --account=00123\n#SBATCH -n 8\n"
            'export VASP_CMD="mpirun -np 4 vasp_std"\n'
            'if [ "$SLURM_NNODES" -gt 1 ]; then\n  module load vasp/multi\nelse\n  module load vasp/single\nfi\n'
            "srun --ntasks-per-node=1 mkdir -p /scratch/$USER/VASP\n"
            "timeout 11h $VASP_CMD > vasp.out 2>&1\n"
        )
        profile, ignored, notes = clusters.parse_job_script(script)
        self.assertEqual(profile["vasp_command"], ["mpirun", "-np", "4", "vasp_std"])
        self.assertEqual(profile["sbatch"], {"time": "0", "account": "00123", "ntasks": 8})
        self.assertIn('if [ "$SLURM_NNODES" -gt 1 ]; then', profile["setup"])
        self.assertIn("  module load vasp/single", profile["setup"])
        self.assertTrue(any("block with environment lines was kept" in n for n in notes))
        self.assertTrue(any("'timeout 11h'" in n for n in notes))
        self.assertTrue(any("mkdir" in line for line in ignored))
        self.assertIsNone(clusters.walltime_seconds("0"))
        profile, _, _ = clusters.parse_job_script(
            '#SBATCH -t 60\nMPI_OPTS="--bind-to core --map-by socket"\nmpirun $MPI_OPTS -np $SLURM_NTASKS vasp_std\n'
        )
        self.assertEqual(
            profile["vasp_command"],
            ["mpirun", "--bind-to", "core", "--map-by", "socket", "-np", "$SLURM_NTASKS", "vasp_std"],
        )
        with self.assertRaisesRegex(ValueError, "inside an `if` block"):
            clusters.parse_job_script("#SBATCH -t 60\nif true; then\n  srun vasp_std\nelse\n  srun vasp_gam\nfi\n")
        _, _, notes = clusters.parse_job_script('#SBATCH -t 60\n[ -z "$NP" ] && NP=4\nmpirun -np $NP vasp_std\n')
        self.assertTrue(any("$NP" in n and "was not copied" in n for n in notes))
        # A bare variable is split like an unquoted shell word when the job runs.
        self.assertEqual(
            clusters.expand_command(
                ["$LAUNCH", "-n", "${NP:-$SLURM_NTASKS}", "vasp_std"],
                {"LAUNCH": "srun --mpi=pmix", "SLURM_NTASKS": "64"},
            ),
            ["srun", "--mpi=pmix", "-n", "64", "vasp_std"],
        )

    def test_one_line_forms_names_and_defaults(self):
        cases = {
            'MPI_OPTS="--bind-to core"; mpirun $MPI_OPTS -np 4 vasp_std\n': [
                "mpirun",
                "--bind-to",
                "core",
                "-np",
                "4",
                "vasp_std",
            ],
            'export CMD="srun vasp_std"; $CMD\n': ["srun", "vasp_std"],
            "srun -n $SLURM_NNODES mkdir -p /scratch/$USER/vasp_tmp\nsrun vasp_gam\n": ["srun", "vasp_gam"],
            "mpirun /opt/VASP/bin/VASP_STD\n": ["mpirun", "/opt/VASP/bin/VASP_STD"],
            "srun -n ${NP:-${SLURM_NTASKS}} vasp_std\n": ["srun", "-n", "${NP:-${SLURM_NTASKS}}", "vasp_std"],
        }
        for body, command in cases.items():
            with self.subTest(body=body):
                profile, _, notes = clusters.parse_job_script("#SBATCH -t 60\n" + body)
                self.assertEqual(profile["vasp_command"], command)
                self.assertFalse([n for n in notes if n.startswith("warning:")])  # a default needs no setter
        self.assertEqual(clusters.expand_command(["-n", "${NP:-${SLURM_NTASKS}}"], {"SLURM_NTASKS": "8"}), ["-n", "8"])
        with self.assertRaisesRegex(ValueError, "stop_signal_lead_s must be a positive integer"):
            clusters.validate_profile(
                {"sbatch": {"time": "1:00:00"}, "vasp_command": "srun vasp_std", "stop_signal_lead_s": "30min"}
            )

    def test_malformed_profiles_are_listed_not_fatal(self):
        clusters.import_job_script(self.script(SCRIPT_SRUN), "good")
        folder = clusters.clusters_dir()
        (folder / "listy.yaml").write_text(
            "mechanet_cluster: 1\nsbatch:\n  - --time=1:00:00\nvasp_command: [srun, vasp_std]\n"
        )
        (folder / "number.yaml").write_text("sbatch: {time: '1:00:00'}\nvasp_command: 5\n")
        rows = {row["name"]: row for row in clusters.list_profiles()}
        self.assertEqual(rows["listy"]["status"], "invalid")
        self.assertIn("sbatch must be a mapping", rows["listy"]["problem"])
        self.assertIn("vasp_command must be text or a list", rows["number"]["problem"])
        with self.assertRaisesRegex(FileNotFoundError, "usable: good; 2 invalid"):
            clusters.load_profile("missing")
        with self.assertRaisesRegex(FileNotFoundError, "Cluster profile file .* not found"):
            clusters.load_profile(str(self.root / "profiles" / "missing.yaml"))

    def test_setup_lines_run_without_errexit(self):
        _, profile, _ = clusters.import_job_script(
            self.script("#!/bin/bash\n#SBATCH -t 1:00:00\nmodule load no_such_module_here\nsrun vasp_std\n"),
            "lenient",
        )
        state = self.root / "campaign" / "states" / "s1"
        state.mkdir(parents=True)
        lines = job_script_text(clusters.scheduler_section(profile), state).splitlines()
        self.assertLess(lines.index("module load no_such_module_here"), lines.index("set -euo pipefail"))
        lines[-1] = "echo reached"
        out = subprocess.run(["bash", "-c", "\n".join(lines)], capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "reached")

    def test_expand_command(self):
        self.assertEqual(
            clusters.expand_command(["mpirun", "-n", "${SLURM_NTASKS}", "vasp_std"], {"SLURM_NTASKS": "64"}),
            ["mpirun", "-n", "64", "vasp_std"],
        )
        with self.assertRaisesRegex(ValueError, "unset environment variable"):
            clusters.expand_command(["mpirun", "-n", "$NOPE"], {})
        self.assertEqual(
            clusters.expand_command(["srun", "-n", "${N:-4}", "${EXTRA-}", "vasp_std"], {}),
            ["srun", "-n", "4", "vasp_std"],
        )


if __name__ == "__main__":
    unittest.main()
