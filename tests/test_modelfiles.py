"""Model files of the ML-potential presets: the models folder, the packages' own copies and `mechanet models`.

Model files here are small zip archives standing in for the real ones (PyTorch saves a model as
a zip archive); no package that runs a model is needed, and nothing comes from the internet:
downloads come from a web server on this computer, or from a stand-in for huggingface_hub. No
Hugging Face token of the computer running the tests is seen.
"""

import contextlib
import http.server
import io
import json
import math
import os
import shutil
import socket
import sys
import tempfile
import threading
import types
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ase.calculators.emt import EMT

from mechanet import modelfiles
from mechanet.cli import main
from mechanet.modelfiles import (
    MODEL_FILES,
    TOKEN_PERMISSION,
    check,
    check_file,
    check_text,
    download,
    find_model_file,
    hub_access_hint,
    hugging_face_token,
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


def refusal(name, status, message):
    """An error as huggingface_hub raises it when Hugging Face refuses a download: an OSError
    carrying the response, here only its status."""
    error = type(name, (OSError,), {})(message)
    error.response = SimpleNamespace(status_code=status)
    return error


def fake_hub(calls, fail=None):
    """huggingface_hub's hf_hub_download, laying the files out in local_dir as it does."""

    def hf_hub_download(repo_id, filename, revision=None, local_dir=None, **kwargs):
        calls.append(dict(repo_id=repo_id, filename=filename, revision=revision, local_dir=local_dir, **kwargs))
        if fail is not None:
            raise fail
        target = Path(local_dir) / filename
        if target.suffix == ".pt":
            write_model(target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("licence text")
        return str(target)

    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = hf_hub_download
    return patch.dict(sys.modules, {"huggingface_hub": hub})


class Served:
    """A web server on this computer that answers every request with ``body``, announcing ``length``
    bytes (by default the body's), then closes the connection."""

    def __init__(self, body, length=None):
        self.body, self.length = body, len(body) if length is None else length

    def __enter__(self):
        body, length = self.body, self.length

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(length))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def links(urls):
    """The table of model files with the download links of some presets replaced."""
    table = tuple(replace(e, url=urls.get(e.calculator, e.url)) for e in MODEL_FILES)
    return patch.object(modelfiles, "MODEL_FILES", table)


class Isolated(unittest.TestCase):
    """Every test sees only its own folders: no models folder, cache, home or Hugging Face token of
    the computer running it."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.folder = self.tmp / "models"
        self.home = self.tmp / "home"
        self.home.mkdir()
        environ = patch.dict(os.environ, {"MECHANET_MODELS": str(self.folder)})
        environ.start()
        self.addCleanup(environ.stop)
        for name in ("XDG_DATA_HOME", "XDG_CACHE_HOME", "FAIRCHEM_CACHE_DIR", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
                     "HF_TOKEN_PATH", "HF_HOME", "HF_HUB_OFFLINE"):  # fmt: skip
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


class TokenTests(Isolated):
    def test_where_huggingface_hub_finds_the_token_that_uma_s_documentation_sets_up(self):
        self.assertIsNone(hugging_face_token({}))
        self.assertEqual(hugging_face_token({"HF_TOKEN": "hf_x"}), "HF_TOKEN")
        self.assertEqual(hugging_face_token({"HUGGING_FACE_HUB_TOKEN": "hf_x"}), "HUGGING_FACE_HUB_TOKEN")
        self.assertIsNone(hugging_face_token({"HF_TOKEN": " \n"}))  # blank is unset, as huggingface_hub reads it
        saved = self.home / ".cache" / "huggingface" / "token"  # where `hf auth login` saves it
        saved.parent.mkdir(parents=True)
        saved.write_text("")
        self.assertIsNone(hugging_face_token({}))
        saved.write_text("hf_x")
        self.assertEqual(hugging_face_token({}), str(saved))
        self.assertEqual(hugging_face_token({"HF_TOKEN": "hf_x"}), "HF_TOKEN")  # the variable comes first
        elsewhere = self.tmp / "hf" / "token"
        elsewhere.parent.mkdir()
        elsewhere.write_text("hf_x")
        self.assertEqual(hugging_face_token({"HF_HOME": str(elsewhere.parent)}), str(elsewhere))
        self.assertEqual(hugging_face_token({"HF_TOKEN_PATH": str(elsewhere)}), str(elsewhere))
        cache = self.tmp / "xdg" / "huggingface" / "token"
        cache.parent.mkdir(parents=True)
        cache.write_text("hf_x")
        self.assertEqual(hugging_face_token({"XDG_CACHE_HOME": str(self.tmp / "xdg")}), str(cache))

    def test_the_report_says_where_the_token_is_and_never_shows_it(self):
        secret = "hf_StandInTokenNeverToBeShown"
        with installed({"fairchem-core": "2.25.0"}), patch.dict(os.environ, {"HF_TOKEN": secret}):
            result = check(["uma"])
        text = check_text(result)
        self.assertEqual(result["hugging_face_token"], "HF_TOKEN")
        self.assertIn("token    found in HF_TOKEN", text)
        self.assertIn("get it   mechanet models uma --download (with your Hugging Face token)", text)
        self.assertNotIn(secret, text + json.dumps(result))
        with installed({"fairchem-core": "2.25.0"}):
            text = check_text(check(["uma"]))
        self.assertIn("token    none found here", text)
        write_model(self.folder / "uma-s-1p2p1.pt")
        with installed({"fairchem-core": "2.25.0"}):
            text = check_text(check(["uma"]))
        self.assertNotIn("token", text)  # the file is here: no token is needed

    def test_a_refusal_says_which_step_is_missing(self):
        anonymous = refusal(
            "GatedRepoError",
            401,
            "401 Client Error. Cannot access gated repo for url https://huggingface.co/facebook/UMA/resolve/main/"
            "checkpoints/uma-s-1p2p1.pt. Access to model facebook/UMA is restricted. You must have access to it "
            "and be authenticated to access it. Please log in.",  # what Hugging Face answers without a token
        )
        hint = hub_access_hint(anonymous, {})
        self.assertIn("No Hugging Face token was found", hint)
        self.assertIn(f'"{TOKEN_PERMISSION}"', hint)
        self.assertIn("request access at https://huggingface.co/facebook/UMA", hint)
        self.assertIn("hf auth login", hint)
        self.assertEqual(hub_access_hint(OSError("401 Client Error: gated repo facebook/UMA"), {}), hint)
        mistyped = refusal("HfHubHTTPError", 401, "401 Client Error. Invalid credentials in Authorization header")
        self.assertIn("did not accept the token in HF_TOKEN", hub_access_hint(mistyped, {"HF_TOKEN": "hf_x"}))
        waiting = refusal("GatedRepoError", 403, "403 Client Error. Cannot access gated repo for url ...")
        hint = hub_access_hint(waiting, {"HF_TOKEN": "hf_x"})
        self.assertIn("The token in HF_TOKEN gives no access to facebook/UMA yet", hint)
        self.assertIn(TOKEN_PERMISSION, hint)
        saved = self.home / ".cache" / "huggingface" / "token"
        saved.parent.mkdir(parents=True)
        saved.write_text("hf_x")
        self.assertIn("The token that `hf auth login` saved", hub_access_hint(waiting, {}))
        self.assertIsNone(hub_access_hint(ConnectionError("Max retries exceeded with url"), {}))
        self.assertIsNone(hub_access_hint(KeyError("uma-x"), {}))


class DownloadTests(Isolated):
    def setUp(self):
        super().setUp()
        self.log = []

    def test_uma_comes_from_hugging_face_with_the_token_and_its_licence_beside_it(self):
        calls = []
        with fake_hub(calls), patch.dict(os.environ, {"HF_TOKEN": "hf_StandInTokenNeverToBeShown"}):
            done = download("uma", log=self.log.append)
        model = self.folder / "UMA" / "checkpoints" / "uma-s-1p2p1.pt"  # as `hf download --local-dir UMA` puts it
        self.assertEqual((done["status"], Path(done["path"])), ("downloaded", model))
        self.assertEqual(
            calls[0],
            {
                "repo_id": "facebook/UMA",
                "filename": "checkpoints/uma-s-1p2p1.pt",
                "revision": "main",
                "local_dir": str(self.folder / "UMA"),
            },
        )
        self.assertEqual([c["filename"] for c in calls[1:]], ["LICENSE", "USE_POLICY.md"])
        self.assertEqual([Path(p).name for p in done["licence_files"]], ["LICENSE", "USE_POLICY.md"])
        self.assertEqual(find_model_file("uma-s-1p2p1.pt"), model)
        self.assertIn("uma: downloading uma-s-1p2p1.pt (2.3 GB) from huggingface.co/facebook/UMA", self.log[0])
        self.assertIn("with the token in HF_TOKEN", self.log[0])
        self.assertIn("FAIR Chemistry License", self.log[0])
        self.assertNotIn("hf_StandInTokenNeverToBeShown", "\n".join(self.log))
        calls.clear()
        with fake_hub(calls):
            again = download("uma", log=self.log.append)
        self.assertEqual((again["status"], calls), ("already in the models folder", []))  # never twice

    def test_a_refused_download_says_why_and_leaves_no_model(self):
        refused = refusal("GatedRepoError", 403, "403 Client Error. Cannot access gated repo")
        with fake_hub([], fail=refused), patch.dict(os.environ, {"HF_TOKEN": "hf_x"}):
            with self.assertRaisesRegex(
                RuntimeError, r"could not be downloaded: The token in HF_TOKEN gives no access .* \(GatedRepoError\)"
            ):
                download("uma", log=self.log.append)
        self.assertIsNone(find_model_file("uma-s-1p2p1.pt"))

    def test_offline_or_without_huggingface_hub_it_says_so_before_trying(self):
        calls = []
        with fake_hub(calls), patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            with self.assertRaisesRegex(RuntimeError, "unset HF_HUB_OFFLINE"):
                download("uma", log=self.log.append)
        self.assertEqual(calls, [])
        with patch.dict(sys.modules, {"huggingface_hub": None}):  # not installed in this environment
            with self.assertRaisesRegex(RuntimeError, "needs huggingface_hub, which comes with fairchem-core"):
                download("uma", log=self.log.append)
        self.assertFalse(self.folder.exists())  # nothing started: no folder made

    def test_a_broken_copy_in_the_folder_is_left_for_you_to_delete(self):
        self.folder.mkdir()
        (self.folder / "uma-s-1p2p1.pt").write_text("<html>Access to model facebook/UMA is restricted</html>")
        calls = []
        with fake_hub(calls), self.assertRaisesRegex(RuntimeError, "is not usable .*: delete it, then try again"):
            download("uma", log=self.log.append)
        self.assertEqual(calls, [])

    def test_mace_and_sevennet_come_from_their_links_without_an_account(self):
        model = write_model(self.tmp / "served.model", payload=os.urandom(3_000_000)).read_bytes()
        with Served(model) as url, links({"mace-mh1": f"{url}/mace-mh-1.model"}):
            done = download("mace-mh1", log=self.log.append)
        target = self.folder / "mace-mh-1.model"
        self.assertEqual((done["status"], Path(done["path"]), done["licence_files"]), ("downloaded", target, []))
        self.assertEqual(target.read_bytes(), model)
        self.assertEqual([p.name for p in self.folder.iterdir()], ["mace-mh-1.model"])  # no .part left
        self.assertIn("Academic Software License", self.log[0])
        self.assertEqual(self.log[-1], "mace-mh1: 3 of 3 MB")

    def test_a_download_cut_short_or_not_a_model_leaves_nothing(self):
        model = write_model(self.tmp / "served.model", payload=os.urandom(500_000)).read_bytes()
        with Served(model[:1000], length=len(model)) as url, links({"mace-mh1": f"{url}/mace-mh-1.model"}):
            with self.assertRaisesRegex(RuntimeError, "stopped after 1,000 of|broke off"):
                download("mace-mh1", log=self.log.append)
        with Served(b"<html>Not Found</html>") as url, links({"sevennet": f"{url}/checkpoint_sevennet_omni.pth"}):
            with self.assertRaisesRegex(RuntimeError, "is not usable: not a complete model file"):
                download("sevennet", log=self.log.append)
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_without_a_connection_it_says_to_download_elsewhere(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]  # closed again: nothing answers there
        with links({"mace-mh1": f"http://127.0.0.1:{port}/mace-mh-1.model"}):
            with self.assertRaisesRegex(RuntimeError, "no connection to 127.0.0.1.* download it on one that has it"):
                download("mace-mh1", log=self.log.append)


class CommandTests(Isolated):
    def run_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = main(list(argv))
        return code, out.getvalue()

    def run_cli_with_stderr(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_download_only_what_is_named_and_say_how_it_went(self):
        code, _, err = self.run_cli_with_stderr("models", "--download")
        self.assertEqual(code, 2)  # each model comes under its own licence: none is downloaded unasked
        self.assertIn("name the models to download", err)
        with installed({"fairchem-core": "2.25.0"}), fake_hub([]), patch.dict(os.environ, {"HF_TOKEN": "hf_x"}):
            code, out, err = self.run_cli_with_stderr("models", "uma", "--download")
            self.assertEqual(code, 0)
            self.assertIn("download saved", out)
            self.assertIn("with LICENSE and USE_POLICY.md next to it", out)
            self.assertIn("ready    yes, without internet", out)
            self.assertNotIn("Nothing was downloaded", out)
            self.assertIn("uma: downloading uma-s-1p2p1.pt", err)  # progress goes to stderr
            code, out, _ = self.run_cli_with_stderr("models", "uma", "--download", "--json")
            self.assertEqual(json.loads(out)["models"][0]["download"]["status"], "already in the models folder")
        shutil.rmtree(self.folder)
        model = write_model(self.tmp / "served.model").read_bytes()
        anonymous = refusal("GatedRepoError", 401, "401 Client Error. Cannot access gated repo")
        with (
            installed({"fairchem-core": "2.25.0", "mace-torch": "0.3.16"}),
            fake_hub([], fail=anonymous),
            Served(model) as url,
            links({"mace-mh1": f"{url}/mace-mh-1.model"}),
        ):
            code, out, _ = self.run_cli_with_stderr("models", "mace-mh1", "uma", "--download")
        self.assertEqual(code, 1)  # UMA did not come: a job script stops here
        self.assertIn("download FAILED: UMA uma-s-1p2p1 could not be downloaded: No Hugging Face token was found", out)
        self.assertIn(f"download saved {self.folder / 'mace-mh-1.model'}", out)  # one failure leaves the others

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
