"""The foundation-potential presets (mace-mh1, uma, sevennet) and recommended relaxation settings.

mace-torch, fairchem-core and sevenn are not installed where the tests run (they need three
environments of their own), so stand-in modules play them: each records how it was called and
answers as the real package does, including what goes wrong (mace-torch replaces an unknown
head by the model's last head with only a warning; gated UMA weights refuse the download).
EMT gives the energies, so relaxations are real. One test loads the real MACE-MH-1 when asked
(MECHANET_TEST_MACE_MH1=1; it downloads 59 MB).
"""

import hashlib
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from ase.calculators.calculator import Calculator as ASEBase
from ase.calculators.emt import EMT

from mechanet import calculators
from mechanet.calculators import (
    RECOMMENDED_ML_RELAXATION,
    check_relax_settings,
    fingerprint,
    load_calculator,
    relax_settings_text,
    resolve_device,
)
from mechanet.plugins import Registry, default_registry
from mechanet.prescreen import relaxation_settings
from mechanet.structure import Structure

PRESETS = ("mace-mh1", "uma", "sevennet")


def slab():
    """Pt(111) 2x2x3 with O in an fcc hollow, bottom layer fixed: something EMT can relax."""
    from ase.build import add_adsorbate, fcc111

    atoms = fcc111("Pt", (2, 2, 3), vacuum=6.0)
    add_adsorbate(atoms, "O", 1.9, "fcc")
    mobile = np.ones((len(atoms), 3), dtype=bool)
    mobile[atoms.positions[:, 2] < atoms.positions[:, 2].min() + 0.1] = False
    ids = [f"a{i + 1:04d}" for i in range(len(atoms))]
    return Structure(atoms.get_chemical_symbols(), atoms.positions, atoms.cell.array, mobile, ids)


def module(name, **attributes):
    m = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(m, key, value)
    return m


def fake_mace(model_file, heads=("omat_pbe", "omol", "oc20_usemppbe"), calls=None):
    """mace-torch's mace_mp and checkpoint download, as 0.3.16 behaves."""
    calls = [] if calls is None else calls

    class FakeMACE(EMT):
        def __init__(self, head=None):
            super().__init__()
            self.available_heads = list(heads)
            # mace-torch: an unknown head becomes the last head, with a warning only
            self.head = head if head in heads else heads[-1]

    def mace_mp(model=None, device="", default_dtype="float32", dispersion=False, **kwargs):
        calls.append(dict(model=model, device=device, default_dtype=default_dtype, dispersion=dispersion, **kwargs))
        return FakeMACE(head=kwargs.get("head"))

    def download(name):
        calls.append({"download": name})
        return str(model_file)

    foundations = module(
        "mace.calculators.foundations_models",
        mace_mp_names=["small", "medium", "mh-0", "mh-1"],
        download_mace_mp_checkpoint=download,
    )
    calcs = module("mace.calculators", mace_mp=mace_mp, foundations_models=foundations)
    return {
        "mace": module("mace", calculators=calcs),
        "mace.calculators": calcs,
        "mace.calculators.foundations_models": foundations,
    }, calls


def fake_fairchem(checkpoint, tasks=("oc20", "oc22", "omat", "omol"), fail=None, calls=None):
    """fairchem-core 2.x: get_predict_unit and FAIRChemCalculator."""
    calls = [] if calls is None else calls

    def get_predict_unit(model_name, inference_settings="default", device=None, **kwargs):
        calls.append(dict(model_name=model_name, inference_settings=inference_settings, device=device, **kwargs))
        if fail is not None:
            raise fail
        return SimpleNamespace(dataset_to_tasks={t: [] for t in tasks})

    class FAIRChemCalculator(EMT):
        def __init__(self, predict_unit, task_name=None):
            super().__init__()
            self.task_name = task_name

    pretrained = module(
        "fairchem.core.calculate.pretrained_mlip",
        get_predict_unit=get_predict_unit,
        pretrained_checkpoint_path_from_name=lambda name: str(checkpoint),
    )
    core = module("fairchem.core", FAIRChemCalculator=FAIRChemCalculator, pretrained_mlip=pretrained)
    return {"fairchem": module("fairchem", core=core), "fairchem.core": core}, calls


def fake_sevenn(checkpoint, modals=("mpa", "oc22", "omat24"), calls=None):
    """sevenn 0.13: SevenNetCalculator with modal channels, matched exactly."""
    calls = [] if calls is None else calls

    class SevenNetCalculator(EMT):
        def __init__(self, model="7net-0", file_type="checkpoint", device="auto", modal=None, **kwargs):
            calls.append(dict(model=model, device=device, modal=modal, **kwargs))
            if modal not in modals:
                raise ValueError(f"unknown modal {modal} (not in {list(modals)})")
            super().__init__()
            self.modal = modal

    calc_module = module("sevenn.calculator", SevenNetCalculator=SevenNetCalculator)
    util = module("sevenn.util", pretrained_name_to_path=lambda name: str(checkpoint))
    return {"sevenn": module("sevenn", calculator=calc_module, util=util), "sevenn.calculator": calc_module,
            "sevenn.util": util}, calls  # fmt: skip


class FakeD3(ASEBase):
    """torch-dftd's TorchDFTD3Calculator: a constant -0.5 eV, no forces."""

    implemented_properties = ["energy", "forces", "free_energy"]
    made = []

    def __init__(self, **kwargs):
        super().__init__()
        FakeD3.made.append(kwargs)

    def calculate(self, atoms=None, properties=("energy",), system_changes=()):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": -0.5, "free_energy": -0.5, "forces": np.zeros((len(atoms), 3))}


def fake_torch(cuda=False):
    return {"torch": module("torch", float64="float64", cuda=SimpleNamespace(is_available=lambda: cuda))}


def fake_d3():
    inner = module("torch_dftd.torch_dftd3_calculator", TorchDFTD3Calculator=FakeD3)
    return {
        "torch_dftd": module("torch_dftd", torch_dftd3_calculator=inner),
        "torch_dftd.torch_dftd3_calculator": inner,
    }


class PresetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.model_file = self.tmp / "macemh1model"
        self.model_file.write_bytes(b"stand-in MACE-MH-1 weights")
        calculators._SHA256.clear()

    def mace(self, **kwargs):
        modules, calls = fake_mace(self.model_file, **kwargs)
        return patch.dict(sys.modules, modules), calls

    def test_mace_mh1_is_mh1_on_the_omat_head_in_float64(self):
        context, calls = self.mace()
        with context:
            calc = load_calculator("mace-mh1", {"device": "cpu"})
            relaxed = calc.relax(slab(), fmax=0.05, steps=40)
        self.assertEqual(calls[0], {"download": "mh-1"})
        self.assertEqual(
            calls[1], {"model": str(self.model_file), "device": "cpu", "default_dtype": "float64", "dispersion": False,
                       "head": "omat_pbe"},
        )  # fmt: skip
        model = calc.describe()["model"]
        self.assertEqual(model["name"], "MACE-MH-1")
        self.assertEqual(model["head"], "omat_pbe")
        self.assertEqual(model["sha256"], hashlib.sha256(b"stand-in MACE-MH-1 weights").hexdigest())
        self.assertEqual((model["dtype"], model["device"], model["dispersion"]), ("float64", "cpu", None))
        # the settings the preset is meant to relax with, unless the caller chooses
        self.assertEqual((calc.optimizer, calc.optimizer_source), ("FIRE", "calculator"))
        self.assertEqual(calc.recommended, {"fmax_eVA": 0.03, "steps": 1500})
        self.assertTrue(np.isfinite(relaxed.energy_eV))
        with context:
            given = load_calculator("mace-mh1", {"device": "cpu"}, optimizer="BFGS")
        self.assertEqual((given.optimizer, given.optimizer_source), ("BFGS", "given"))

    def test_an_unknown_head_is_refused_not_replaced(self):
        context, _ = self.mace()
        with (
            context,
            self.assertRaisesRegex(ValueError, r"has no head 'omat'.*omat_pbe.*would have used 'oc20_usemppbe'"),
        ):
            load_calculator("mace-mh1", {"device": "cpu", "head": "omat"})
        with context:
            other = load_calculator("mace-mh1", {"device": "cpu", "head": "oc20_usemppbe"})
        self.assertEqual(other.describe()["model"]["head"], "oc20_usemppbe")

    def test_a_model_file_or_a_known_name(self):
        tuned = self.tmp / "my-mh1-finetune.model"
        tuned.write_bytes(b"fine-tuned weights")
        context, calls = self.mace()
        with context:
            calc = load_calculator("mace-mh1", {"device": "cpu", "model": str(tuned)})
            with self.assertRaisesRegex(ValueError, "neither a file nor a name"):
                load_calculator("mace-mh1", {"device": "cpu", "model": "mh-9"})
        self.assertNotIn({"download": str(tuned)}, calls)  # a file is used as it is
        self.assertEqual(calc.describe()["model"]["sha256"], hashlib.sha256(b"fine-tuned weights").hexdigest())

    def test_a_changed_model_file_is_not_mistaken_for_the_old_one(self):
        context, _ = self.mace()
        settings = {"fmax_eVA": 0.03, "steps": 1500}
        with context:
            before = fingerprint(load_calculator("mace-mh1", {"device": "cpu"}).describe(), settings)
            self.model_file.write_bytes(b"retrained MACE-MH-1 weights, another size")
            after = fingerprint(load_calculator("mace-mh1", {"device": "cpu"}).describe(), settings)
        self.assertNotEqual(before, after)

    def test_dispersion_adds_d3_with_bj_damping_and_pbe_parameters(self):
        context, calls = self.mace()
        FakeD3.made.clear()
        with context, patch.dict(sys.modules, {**fake_torch(), **fake_d3()}):
            calc = load_calculator("mace-mh1", {"device": "cpu", "dispersion": True})
            plain = load_calculator("mace-mh1", {"device": "cpu"})
            start = slab()
            with_d3 = calc.relax(start, steps=0).energy_eV
            without = plain.relax(start, steps=0).energy_eV
        self.assertFalse(calls[1]["dispersion"])  # mace_mp adds none: mechanet adds one D3 for every preset
        self.assertEqual(FakeD3.made[0]["damping"], "bj")
        self.assertEqual(FakeD3.made[0]["xc"], "pbe")
        record = calc.describe()["model"]["dispersion"]
        self.assertEqual((record["method"], record["xc"]), ("D3(BJ)", "pbe"))
        self.assertAlmostEqual(record["cutoff_A"], 50.2718, places=3)  # 95 bohr, VASP's VDW_RADIUS is 50.2 A
        self.assertEqual(calc.describe()["class"], "ase.calculators.mixing.SumCalculator")
        self.assertAlmostEqual(with_d3 - without, -0.5)
        with context, patch.dict(sys.modules, {"torch_dftd": None, "torch_dftd.torch_dftd3_calculator": None}):
            with self.assertRaisesRegex(RuntimeError, "needs torch-dftd"):
                load_calculator("mace-mh1", {"device": "cpu", "dispersion": True})

    def test_uma_on_the_oc22_task_with_high_precision(self):
        checkpoint = self.tmp / "uma-s-1p2p1.pt"
        checkpoint.write_bytes(b"stand-in UMA weights")
        modules, calls = fake_fairchem(checkpoint)
        with patch.dict(sys.modules, modules):
            calc = load_calculator("uma", {"device": "cpu"})
            with self.assertRaisesRegex(ValueError, r"has no task 'oc23'; it has oc20, oc22, omat, omol"):
                load_calculator("uma", {"device": "cpu", "task": "oc23"})
        self.assertEqual(
            calls[0], {"model_name": "uma-s-1p2p1", "inference_settings": "high-precision", "device": "cpu"}
        )
        self.assertEqual(calc.calc.task_name, "oc22")
        model = calc.describe()["model"]
        self.assertEqual((model["name"], model["task"], model["inference_settings"]), ("UMA", "oc22", "high-precision"))
        self.assertEqual(model["sha256"], hashlib.sha256(b"stand-in UMA weights").hexdigest())
        self.assertEqual(calc.optimizer, "FIRE")
        old, _ = fake_fairchem(checkpoint, tasks=("oc20", "omat", "omol", "odac", "omc"))
        with patch.dict(sys.modules, old), self.assertRaisesRegex(ValueError, "needs a UMA 1.2 model"):
            load_calculator("uma", {"device": "cpu", "model": "uma-s-1p1"})

    def test_uma_says_how_to_get_the_gated_weights(self):
        gated, _ = fake_fairchem(self.tmp / "none.pt", fail=OSError("401 Client Error: gated repo facebook/UMA"))
        with patch.dict(sys.modules, gated), self.assertRaisesRegex(RuntimeError, "request access at"):
            load_calculator("uma", {"device": "cpu"})
        other, _ = fake_fairchem(self.tmp / "none.pt", fail=KeyError("uma-x"))
        with patch.dict(sys.modules, other), self.assertRaises(KeyError):  # anything else passes unchanged
            load_calculator("uma", {"device": "cpu", "model": "uma-x"})

    def test_sevennet_omni_on_the_oc22_modal(self):
        checkpoint = self.tmp / "checkpoint_sevennet_omni.pth"
        checkpoint.write_bytes(b"stand-in SevenNet-Omni weights")
        modules, calls = fake_sevenn(checkpoint)
        with patch.dict(sys.modules, modules):
            calc = load_calculator("sevennet", {"device": "cpu"})
            with self.assertRaisesRegex(ValueError, "unknown modal OC22"):  # matched exactly: lower case
                load_calculator("sevennet", {"device": "cpu", "modal": "OC22"})
        self.assertEqual(calls[0], {"model": "7net-omni", "device": "cpu", "modal": "oc22"})
        model = calc.describe()["model"]
        self.assertEqual((model["name"], model["modal"]), ("SevenNet-Omni", "oc22"))
        self.assertEqual(model["sha256"], hashlib.sha256(b"stand-in SevenNet-Omni weights").hexdigest())

    def test_missing_packages_say_what_to_install(self):
        absent = {
            "mace-mh1": {"mace": None, "mace.calculators": None},
            "uma": {"fairchem": None, "fairchem.core": None},
            "sevennet": {"sevenn": None, "sevenn.calculator": None},
        }
        wanted = {"mace-mh1": "mace-torch>=0.3.14", "uma": "fairchem-core>=2.22", "sevennet": "sevenn>=0.13"}
        for name in PRESETS:
            with patch.dict(sys.modules, absent[name]), self.assertRaisesRegex(RuntimeError, wanted[name]):
                load_calculator(name, {"device": "cpu"})

    def test_device_auto(self):
        with patch.dict(sys.modules, fake_torch(cuda=True)):
            self.assertEqual(resolve_device("auto"), "cuda")
        with patch.dict(sys.modules, fake_torch(cuda=False)):
            self.assertEqual(resolve_device(""), "cpu")
        with patch.dict(sys.modules, {"torch": None}):
            self.assertEqual(resolve_device("auto"), "cpu")
        self.assertEqual(resolve_device("cuda:1"), "cuda:1")


class RelaxSettingsTests(unittest.TestCase):
    def test_settings_are_checked(self):
        self.assertEqual(check_relax_settings(None), {})
        self.assertEqual(check_relax_settings(RECOMMENDED_ML_RELAXATION), {"optimizer": "FIRE", "fmax_eVA": 0.03,
                                                                            "steps": 1500})  # fmt: skip
        for wrong, message in (
            ({"optimizer": "CG"}, "not one of"),
            ({"fmax_eVA": 0}, "positive"),
            ({"steps": 1.5}, "nonnegative integer"),
            ({"steps": True}, "nonnegative integer"),
            ({"temperature": 300}, "unknown relax_settings"),
            (["FIRE"], "must be a dict"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                check_relax_settings(wrong)
        self.assertEqual(relax_settings_text(RECOMMENDED_ML_RELAXATION), "FIRE, 0.03 eV/A, at most 1500 steps")
        self.assertEqual(relax_settings_text({}), "-")

    def test_registry_and_entry_point_factories_can_recommend(self):
        registry = Registry()

        def make(**options):
            return EMT()

        make.relax_settings = {"optimizer": "LBFGS", "steps": 50}  # as an entry point would carry it
        registry.add_calculator("tuned-emt", make, "EMT with its own settings")
        registry.add_calculator("plain-emt", lambda **options: EMT(), "EMT with mechanet's defaults")
        registry.add_calculator(
            "given-emt", lambda **options: EMT(), "settings given", relax_settings={"fmax_eVA": 0.01}
        )
        with self.assertRaisesRegex(ValueError, "Calculator 'bad'"):
            registry.add_calculator("bad", make, relax_settings={"optimizer": "Newton"})
        tuned = load_calculator("tuned-emt", registry=registry)
        self.assertEqual((tuned.optimizer, tuned.recommended), ("LBFGS", {"steps": 50}))
        plain = load_calculator("plain-emt", registry=registry)
        self.assertEqual((plain.optimizer, plain.optimizer_source, plain.recommended), ("BFGS", "default", {}))
        self.assertEqual(load_calculator("given-emt", registry=registry).recommended, {"fmax_eVA": 0.01})
        listed = {c.name: c.relax_settings for c in default_registry().calculators.values()}
        for name in PRESETS:
            self.assertEqual(listed[name], RECOMMENDED_ML_RELAXATION)
        self.assertEqual((listed["emt"], listed["mace-mp"]), ({}, {}))

    def test_given_values_win_then_the_calculator_then_the_defaults(self):
        recommending = SimpleNamespace(recommended={"fmax_eVA": 0.03, "steps": 1500}, optimizer_source="calculator")
        self.assertEqual(
            relaxation_settings(recommending),
            ({"fmax_eVA": 0.03, "steps": 1500}, {"fmax_eVA": "calculator", "steps": "calculator",
                                                  "optimizer": "calculator"}),
        )  # fmt: skip
        self.assertEqual(
            relaxation_settings(recommending, fmax_eVA=0.05),
            (
                {"fmax_eVA": 0.05, "steps": 1500},
                {"fmax_eVA": "given", "steps": "calculator", "optimizer": "calculator"},
            ),
        )
        self.assertEqual(
            relaxation_settings(object(), steps=10),
            ({"fmax_eVA": 0.05, "steps": 10}, {"fmax_eVA": "default", "steps": "given"}),
        )


@unittest.skipUnless(
    os.environ.get("MECHANET_TEST_MACE_MH1") and importlib.util.find_spec("mace"),
    "set MECHANET_TEST_MACE_MH1=1 with mace-torch installed to load the real MACE-MH-1 (downloads 59 MB)",
)
class RealMaceMH1Tests(unittest.TestCase):
    def test_the_real_model(self):
        calculators._SHA256.clear()
        calc = load_calculator("mace-mh1", {"device": "cpu"})
        model = calc.describe()["model"]
        self.assertEqual(model["head"], "omat_pbe")
        self.assertIn("oc20_usemppbe", model["heads"])
        self.assertRegex(model["sha256"], r"^[0-9a-f]{64}$")
        relaxed = calc.relax(slab(), fmax=0.03, steps=1500)
        self.assertTrue(relaxed.converged)
        self.assertTrue(np.isfinite(relaxed.energy_eV))
        with self.assertRaisesRegex(ValueError, "has no head"):
            load_calculator("mace-mh1", {"device": "cpu", "head": "omat"})


if __name__ == "__main__":
    unittest.main()
