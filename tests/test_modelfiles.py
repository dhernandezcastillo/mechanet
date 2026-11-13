"""Model files of the ML-potential presets: the models folder, the packages' own copies and `mechanet models`.

Model files here are small zip archives standing in for the real ones (PyTorch saves a model as
a zip archive); no package that runs a model is needed, and nothing is downloaded.
"""

import contextlib
import io
import json
import math
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ase.calculators.emt import EMT

from mechanet import modelfiles
from mechanet.cli import main
from mechanet.modelfiles import (
    MODEL_FILES,
    check,
    check_file,
    check_text,
    find_model_file,
    model_file,
    model_path,
    models_folder,
    official_filename,
    package_copy,
)


def write_model(path, payload=b"weights" * 100):
    """A small zip archive in the place of a model file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model/data.pkl", b"pickle")
        archive.writestr("model/data/0", payload)
    return path


class Isolated(unittest.TestCase):
    """Every test sees only its own folders: no models folder, cache or home of the computer running it."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.folder = self.tmp / "models"
        self.home = self.tmp / "home"
        self.home.mkdir()
        environ = patch.dict(os.environ, {"MECHANET_MODELS": str(self.folder)})
        environ.start()
        self.addCleanup(environ.stop)
        for name in ("XDG_DATA_HOME", "XDG_CACHE_HOME", "FAIRCHEM_CACHE_DIR"):
            os.environ.pop(name, None)
        home = patch.object(Path, "home", return_value=self.home)
        home.start()
        self.addCleanup(home.stop)


class FolderTests(Isolated):
    def test_the_folder_is_set_by_mechanet_models_then_xdg_then_home(self):
        self.assertEqual(models_folder({"MECHANET_MODELS": "/group/models"}), Path("/group/models"))
        self.assertEqual(models_folder({"XDG_DATA_HOME": "/data"}), Path("/data/mechanet/models"))
        self.assertEqual(models_folder({}), self.home / ".local" / "share" / "mechanet" / "models")
        self.assertEqual(models_folder({"MECHANET_MODELS": "/m", "XDG_DATA_HOME": "/data"}), Path("/m"))

    def test_official_names(self):
        self.assertEqual(official_filename("mace-mh1", "mh-1"), "mace-mh-1.model")
        self.assertEqual(official_filename("mace-mh1", "mh-0"), "mace-mh-0.model")
        self.assertEqual(official_filename("uma", "uma-s-1p2p1"), "uma-s-1p2p1.pt")
        self.assertEqual(official_filename("uma", "uma-m-1p1"), "uma-m-1p1.pt")  # fairchem's <name>.pt
        self.assertEqual(official_filename("sevennet", "7net-omni"), "checkpoint_sevennet_omni.pth")
        self.assertIsNone(official_filename("sevennet", "7net-0"))
        self.assertIsNone(official_filename("uma", "/somewhere/file.pt"))
        with self.assertRaisesRegex(ValueError, "known: mace-mh1, uma, sevennet"):
            model_file("emt")

    def test_a_file_directly_in_the_folder_or_up_to_two_levels_down(self):
        self.assertIsNone(find_model_file("uma-s-1p2p1.pt"))  # no folder yet
        nested = write_model(self.folder / "UMA" / "checkpoints" / "uma-s-1p2p1.pt")
        self.assertEqual(find_model_file("uma-s-1p2p1.pt"), nested)  # as `hf download --local-dir UMA` lays it out
        direct = write_model(self.folder / "uma-s-1p2p1.pt")
        self.assertEqual(find_model_file("uma-s-1p2p1.pt"), direct)  # the file in the folder itself comes first
        write_model(self.folder / "a" / "b" / "c" / "mace-mh-1.model")
        self.assertIsNone(find_model_file("mace-mh-1.model"))  # three levels down is not looked at

    def test_two_copies_inside_the_folder_are_refused(self):
        write_model(self.folder / "old" / "mace-mh-1.model")
        write_model(self.folder / "new" / "mace-mh-1.model")
        with self.assertRaisesRegex(ValueError, "2 files named mace-mh-1.model .* keep one"):
            find_model_file("mace-mh-1.model")

    def test_a_path_first_then_the_folder_then_the_package(self):
        given = write_model(self.tmp / "my-finetune.model")
        self.assertEqual(model_path("mace-mh1", str(given)), (given, "path"))
        self.assertEqual(model_path("mace-mh1", "mh-1"), (None, None))
        copy = write_model(self.folder / "mace-mh-1.model")
        self.assertEqual(model_path("mace-mh1", "mh-1"), (copy, "models folder"))
        self.assertEqual(model_path("mace-mh1", "medium", filename="mace-mh-1.model"), (copy, "models folder"))
        self.assertEqual(model_path("sevennet", "7net-0"), (None, None))  # no official name known: the package


class FileCheckTests(Isolated):
    def test_complete_cut_short_or_not_a_model(self):
        good = write_model(self.tmp / "good.pt")
        self.assertEqual(check_file(good), {"size": good.stat().st_size, "complete": True, "problem": None})
        cut = self.tmp / "cut.pt"
        cut.write_bytes(good.read_bytes()[: good.stat().st_size // 2])  # a download interrupted halfway
        self.assertRegex(check_file(cut)["problem"], "download cut short")
        page = self.tmp / "page.pt"
        page.write_text("<html>Access to model facebook/UMA is restricted</html>")  # an error page saved instead
        self.assertFalse(check_file(page)["complete"])
        empty = self.tmp / "empty.pt"
        zipfile.ZipFile(empty, "w").close()
        self.assertRegex(check_file(empty)["problem"], "empty archive")

    def test_the_full_check_reads_every_member_and_gives_the_sha256(self):
        good = write_model(self.tmp / "good.pt", payload=b"A" * 4000)
        full = check_file(good, full=True)
        self.assertTrue(full["complete"])
        self.assertRegex(full["sha256"], r"^[0-9a-f]{64}$")
        data = bytearray(good.read_bytes())
        start = data.find(b"A" * 4000)
        data[start + 10] = ord("B")  # one byte changed inside a member: its checksum no longer matches
        bad = self.tmp / "bad.pt"
        bad.write_bytes(bytes(data))
        self.assertTrue(check_file(bad)["complete"])  # the quick check sees an intact table of contents
        self.assertRegex(check_file(bad, full=True)["problem"], "model/data/0 fails its checksum")


class PackageCopyTests(Isolated):
    def test_where_the_packages_keep_their_downloads(self):
        mace, uma, sevennet = (model_file(name) for name in ("mace-mh1", "uma", "sevennet"))
        for entry in (mace, uma, sevennet):
            self.assertIsNone(package_copy(entry))
        squeezed = write_model(self.home / ".cache" / "mace" / "macemh1model")  # mace-torch drops - and .
        self.assertEqual(package_copy(mace), squeezed)
        repo = self.tmp / "fc" / "models--facebook--UMA"
        write_model(repo / "snapshots" / "0ld" / "checkpoints" / "uma-s-1p2p1.pt")
        current = write_model(repo / "snapshots" / "c0ffee" / "checkpoints" / "uma-s-1p2p1.pt")
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text("c0ffee")
        self.assertIsNone(package_copy(uma))  # not in ~/.cache/fairchem
        self.assertEqual(package_copy(uma, {"FAIRCHEM_CACHE_DIR": str(self.tmp / "fc")}), current)
        with patch("importlib.util.find_spec", return_value=None):
            cached = write_model(self.home / ".cache" / "sevennet" / "SevenNet_omni" / "checkpoint_sevennet_omni.pth")
            self.assertEqual(package_copy(sevennet), cached)


def installed(versions):
    return patch.object(modelfiles, "_installed", side_effect=lambda dist: versions.get(dist))


class CheckTests(Isolated):
    def test_a_file_in_the_folder_with_its_package_is_ready_without_internet(self):
        write_model(self.folder / "uma-s-1p2p1.pt")
        with installed({"fairchem-core": "2.25.0"}):
            result = check(["uma"])
        self.assertEqual((result["folder"], result["folder_set_by"]), (str(self.folder), "MECHANET_MODELS"))
        (report,) = result["models"]
        self.assertEqual((report["found_in"], report["complete"], report["package_ok"]), ("models folder", True, True))
        self.assertTrue(report["ready"])
        self.assertIn("another version of the model", report["notes"][0])  # not the size of the tested file
        text = check_text(result)
        self.assertIn("ready    yes, without internet", text)
        self.assertNotIn("get it", text)
        self.assertIn("FAIR Chemistry License", text)

    def test_what_is_missing_and_where_to_get_it(self):
        with installed({"fairchem-core": "2.22.0", "mace-torch": "0.3.16"}):
            result = check()
        reports = {r["calculator"]: r for r in result["models"]}
        self.assertEqual(set(reports), {e.calculator for e in MODEL_FILES})
        self.assertFalse(any(r["ready"] for r in reports.values()))
        self.assertFalse(reports["uma"]["package_ok"])  # 2.22 lacks the high-precision preset
        self.assertTrue(reports["mace-mh1"]["package_ok"])
        text = check_text(result)
        self.assertIn("fairchem-core 2.22.0 is too old: needs fairchem-core>=2.24", text)
        self.assertIn("https://huggingface.co/facebook/UMA/resolve/main/checkpoints/uma-s-1p2p1.pt", text)
        self.assertIn("granted access to facebook/UMA", text)
        self.assertIn('sevenn is not installed here (pip install "sevenn>=0.13")', text)
        self.assertIn("academic, non-commercial use only", text)  # MACE-MH-1's licence, always shown
        self.assertIn("(it does not exist yet)", text)

    def test_fairchem_s_own_copy_needs_hf_hub_offline_on_nodes_without_internet(self):
        cache = self.tmp / "fc"
        write_model(cache / "models--facebook--UMA" / "snapshots" / "c0ffee" / "checkpoints" / "uma-s-1p2p1.pt")
        with installed({"fairchem-core": "2.25.0"}), patch.dict(os.environ, {"FAIRCHEM_CACHE_DIR": str(cache)}):
            (report,) = check(["uma"])["models"]
        self.assertEqual(report["found_in"], "fairchem-core's own copy")
        self.assertTrue(report["ready"])
        self.assertIn("HF_HUB_OFFLINE=1", report["notes"][-1])

    def test_a_broken_file_is_not_ready(self):
        self.folder.mkdir()
        (self.folder / "mace-mh-1.model").write_bytes(b"PK\x03\x04 cut short")
        with installed({"mace-torch": "0.3.16"}):
            result = check(["mace-mh1"])
        self.assertFalse(result["models"][0]["ready"])
        self.assertIn("NOT USABLE", check_text(result))
        self.assertIn("the model file is not usable", check_text(result))

    def test_load_computes_one_structure_as_a_job_would(self):
        write_model(self.folder / "mace-mh-1.model")
        made = []

        def fake_load(spec, options=None, registry=None, optimizer=None):
            made.append((spec, dict(options), os.environ.get("HF_HUB_OFFLINE")))
            return SimpleNamespace(calc=EMT(), describe=lambda: {"model": {"device": "cpu", "sha256": "f" * 64}})

        with (
            installed({"mace-torch": "0.3.16"}),
            patch.dict(os.environ),
            patch("mechanet.calculators.load_calculator", fake_load),
        ):
            os.environ.pop("HF_HUB_OFFLINE", None)
            (report,) = check(["mace-mh1"], load=True, device="cpu")["models"]
        self.assertEqual(made, [("mace-mh1", {"device": "cpu"}, "1")])  # loaded offline, as on a compute node
        test = report["test"]
        self.assertIsNone(test["error"])
        self.assertEqual((test["device"], test["structure"]), ("cpu", "Pt(111) 2x2x3 with O"))
        self.assertTrue(math.isfinite(test["energy_eV"]))
        self.assertGreater(test["max_force_eVA"], 0)  # O placed by hand is not relaxed
        self.assertRegex(report["sha256"], r"^[0-9a-f]{64}$")  # --load checks the whole file
        self.assertTrue(report["ready"])

        def broken(*args, **kwargs):
            raise RuntimeError("CUDA error: no kernel image is available")

        with installed({"mace-torch": "0.3.16"}), patch("mechanet.calculators.load_calculator", broken):
            result = check(["mace-mh1"], load=True)
        (report,) = result["models"]
        self.assertEqual(report["test"]["error"], "RuntimeError: CUDA error: no kernel image is available")
        self.assertFalse(report["ready"])
        self.assertIn("test     FAILED: RuntimeError", check_text(result))
        self.assertIn("the test calculation failed", check_text(result))
        with installed({}):
            (report,) = check(["mace-mh1"], load=True)["models"]
        self.assertIsNone(report["test"])  # no package here: nothing to load


class CommandTests(Isolated):
    def run_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = main(list(argv))
        return code, out.getvalue()

    def test_the_command_reports_and_its_exit_status_suits_job_scripts(self):
        with installed({"mace-torch": "0.3.16"}):
            code, out = self.run_cli("models", "--json")
            self.assertEqual(code, 0)  # a report of everything is not a failure
            self.assertEqual([r["calculator"] for r in json.loads(out)["models"]], ["mace-mh1", "uma", "sevennet"])
            code, out = self.run_cli("models", "mace-mh1")
            self.assertEqual(code, 1)  # named and not ready: a job script can stop here
            self.assertIn("mace-mh1: MACE-MH-1, file mace-mh-1.model", out)
            write_model(self.folder / "mace-mh-1.model")
            code, _ = self.run_cli("models", "mace-mh1")
            self.assertEqual(code, 0)
            code, _ = self.run_cli("models", "emt")
            self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
