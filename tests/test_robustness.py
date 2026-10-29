"""Slab types, project files, plugins and reference files at the edges of what mechanet handles.

- scope blocks of families (``scope.coverage``, ``scope.enumerate``, plugin blocks);
- an O bridging two metals in their plane (CoO2-terminated perovskite);
- slab files listing an element in more than one block;
- single-atom catalysts in 2D hosts (Fe-N4 in graphene), stepped surfaces and buried dopants;
- plugin reactions in ``init``, in campaigns and in copied campaigns;
- molecular references: what ``references --collect`` checks and records;
- readable messages and summaries.
"""

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from support import EXAMPLE, FAKE_VASP

from mechanet.builder import outward_direction
from mechanet.cli import build_parser, main
from mechanet.fileio import read_json, write_json
from mechanet.inspector import inspect_slab, summary
from mechanet.plugins import Family, Registry, default_registry
from mechanet.project import generate, init_project, load_project, validate_project
from mechanet.reactions import SPECIES
from mechanet.references import collect_references, molecule_inputs, write_template
from mechanet.structure import minimum_image, read_poscar, species_grouped
from mechanet.thermo import reservoir_chemical_potentials

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
RUO2 = EXAMPLES / "ruo2_110"
HER_PLUGIN = EXAMPLES / "plugins" / "her.py"


def quiet(*_args):
    return None


def run(*args):
    """mechanet with text output: (exit code, stdout)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = main([str(a) for a in args])
    return code, out.getvalue()


def perovskite_slab(path, a=3.829, repeat=2, layers=5, vacuum=15.0):
    """SrCoO3(001) with CoO2 on both faces (O bridging two Co in their plane); bottom two layers fixed."""
    from ase import Atoms
    from ase.constraints import FixAtoms
    from ase.io import write

    symbols, positions = [], []
    for k in range(layers):
        motif = [("Co", 0.5, 0.5), ("O", 0.5, 0.0), ("O", 0.0, 0.5)] if k % 2 == 0 else [("Sr", 0, 0), ("O", 0.5, 0.5)]
        for i in range(repeat):
            for j in range(repeat):
                for symbol, fx, fy in motif:
                    symbols.append(symbol)
                    positions.append([(fx + i) * a, (fy + j) * a, vacuum / 2 + k * a / 2])
    atoms = Atoms(symbols, positions=positions, cell=[repeat * a, repeat * a, (layers - 1) * a / 2 + vacuum], pbc=True)
    atoms = atoms[atoms.numbers.argsort(kind="stable")]
    z = atoms.positions[:, 2]
    atoms.set_constraint(FixAtoms(indices=[i for i in range(len(atoms)) if z[i] < z.min() + 0.3 * np.ptp(z)]))
    write(path, atoms, format="vasp", direct=True)
    return path


def fen4_graphene(path, ripple_A=0.0):
    """Fe-N4 single-atom catalyst in a 5x5 graphene sheet (Fe in the divacancy, pyridinic N).

    ``ripple_A``: amplitude of a periodic corrugation of the sheet (Fe stays at mid-height).
    """
    from ase import Atom
    from ase.build import graphene
    from ase.io import write

    sheet = graphene(a=2.46, vacuum=7.5).repeat((5, 5, 1))
    centre = 0.5 * (sheet.cell[0] + sheet.cell[1])
    removed = np.argsort(np.linalg.norm(sheet.positions[:, :2] - centre[:2], axis=1))[:2]
    middle = sheet.positions[removed].mean(axis=0)
    sheet = sheet[[i for i in range(len(sheet)) if i not in removed]]
    for i in np.argsort(np.linalg.norm(sheet.positions - middle, axis=1))[:4]:
        sheet[i].symbol = "N"
    sheet.positions[:, 2] += ripple_A * np.sin(2 * np.pi * sheet.get_scaled_positions()[:, 0])
    sheet.append(Atom("Fe", middle))
    sheet = sheet[sheet.numbers.argsort(kind="stable")]
    write(path, sheet, format="vasp", direct=True)
    return path


def ungrouped_ruo2(path):
    """The RuO2(110) example with its atoms ordered by height: O and Ru alternate in blocks."""
    structure = read_poscar(RUO2 / "slab" / "RuO2_110.vasp")
    order = np.argsort(structure.positions[:, 2], kind="stable")
    symbols = [structure.symbols[i] for i in order]
    blocks = []
    for symbol in symbols:
        if blocks and blocks[-1][0] == symbol:
            blocks[-1][1] += 1
        else:
            blocks.append([symbol, 1])
    frac = structure.positions[order] @ np.linalg.inv(structure.cell)
    lines = ["RuO2(110), atoms ordered by height", "1.0"]
    lines += [" ".join(f"{x:.10f}" for x in row) for row in structure.cell]
    lines += [" ".join(b[0] for b in blocks), " ".join(str(b[1]) for b in blocks), "Selective dynamics", "Direct"]
    for f, flags in zip(frac, structure.mobile[order]):
        lines.append(" ".join(f"{x:.12f}" for x in f) + "   " + " ".join("T" if m else "F" for m in flags))
    Path(path).write_text("\n".join(lines) + "\n")
    return path


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {"MECHANET_CONFIG_DIR": str(self.root / "config")})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def project_copy(self, example=RUO2, name="p"):
        shutil.copytree(example, self.root / name)
        return self.root / name

    def edit(self, project, **changes):
        path = project / "project.yaml"
        data = yaml.safe_load(path.read_text())
        for key, value in changes.items():
            data[key] = value
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return data


class ScopeBlockTests(TempCase):
    def test_coverage_and_enumerate_blocks_configure_their_families(self):
        project = self.project_copy()
        self.edit(project, scope={"families": ["aem", "coverage"], "coverage": {"sites": ["m0038"]}})
        restricted = generate(project / "project.yaml", self.root / "c1")
        self.edit(project, scope={"families": ["aem", "coverage"]})
        every_site = generate(project / "project.yaml", self.root / "c2")
        self.assertLess(restricted["candidate_nodes"], every_site["candidate_nodes"])
        self.edit(project, scope={"families": ["enumerated"], "enumerate": {"sites": ["m0038"], "max_length": 6}})
        report = generate(project / "project.yaml", self.root / "c3")
        self.assertGreater(report["complete_stoichiometric_cycles"], 0)

    def test_typos_in_family_blocks_are_refused_with_a_suggestion(self):
        project = self.project_copy()
        cases = (
            ({"families": ["aem"], "coverge": {}}, "unknown scope keys: 'coverge' (did you mean 'coverage'?)"),
            ({"families": ["aem"], "coverage": {"max_changed": 1}}, "did you mean 'max_changed_sites'?"),
            ({"families": ["aem"], "coverage": [1]}, "scope.coverage configures the coverage family"),
        )
        for scope, message in cases:
            self.edit(project, scope=scope)
            with self.assertRaises(ValueError) as caught:
                load_project(project)
            self.assertIn(message, str(caught.exception))
        self.assertEqual(validate_project({**yaml.safe_load((RUO2 / "project.yaml").read_text())}), [])

    def test_a_plugin_family_declares_its_own_block(self):
        plugin = self.root / "fam.py"
        plugin.write_text(
            "from mechanet.plugins import Family\n"
            "def build(ctx):\n"
            "    for site in ctx.scope['tagged']['sites']:\n"
            "        ctx.node({site: 'OH'}, 'tagged', site)\n"
            "def register(registry):\n"
            "    registry.add_family(Family('tagged', build, None, 'one OH per listed site', "
            "scope_key='tagged', scope_fields=('sites',)))\n"
        )
        project = self.project_copy()
        self.edit(project, plugins=[str(plugin)], scope={"families": ["aem", "tagged"], "tagged": {"sites": ["a0016"]}})
        self.assertTrue(load_project(project)["scope"]["tagged"])
        report = generate(project / "project.yaml", self.root / "c")
        self.assertIn("tagged", report["families"])
        self.edit(project, scope={"families": ["aem", "tagged"], "tagged": {"site": ["a0016"]}})
        with self.assertRaisesRegex(ValueError, "did you mean 'sites'"):
            load_project(project)
        plugin.write_text("raise RuntimeError('broken')\n")
        with self.assertRaisesRegex(ValueError, "plugins: Plugin file .* failed to load"):
            load_project(project)

    def test_scope_keys_cannot_collide(self):
        registry = Registry()
        registry.add_family(Family("a", None, None, "", scope_key="block"))
        with self.assertRaisesRegex(ValueError, "already belongs to family 'a'"):
            registry.add_family(Family("b", None, None, "", scope_key="block"))
        with self.assertRaisesRegex(ValueError, "reserved"):
            registry.add_family(Family("c", None, None, "", scope_key="families"))


class InPlaneBridgeTests(TempCase):
    def test_outward_direction_falls_back_to_the_normal(self):
        normal = np.array([0.0, 0.0, 1.0])
        x = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(outward_direction([x, -x], normal), normal)  # rays cancel
        np.testing.assert_allclose(outward_direction([], normal), normal)
        np.testing.assert_allclose(outward_direction([-normal], normal), normal)  # blend would vanish
        tilted = outward_direction([np.array([0.6, 0.0, 0.8])], normal)
        np.testing.assert_allclose(tilted, [0.6, 0.0, 0.8])  # an ordinary site is unchanged

    def test_cooo2_terminated_perovskite_builds_every_bridge_state(self):
        slab = perovskite_slab(self.root / "SrCoO3_001.vasp")
        init_project(slab, self.root / "p", scope="surface_bridge,hydrated", interactive=False, output=quiet)
        report = generate(self.root / "p" / "project.yaml")
        campaign = Path(report["campaign"])
        self.assertGreater(report["complete_stoichiometric_cycles"], 0)
        self.assertEqual(report["nodes_without_viable_start"], [])
        # An H on an in-plane bridge O sits at a right angle to both M-O bonds, never toward a metal.
        checked = 0
        for state in sorted((campaign / "states").iterdir()):
            meta = read_json(state / "state.json")
            structure = read_poscar(state / "POSCAR")
            ids = meta["atom_ids"]
            for i, symbol in enumerate(structure.symbols):
                if symbol != "H":
                    continue
                o = min(
                    (j for j, s in enumerate(structure.symbols) if s == "O"),
                    key=lambda j: np.linalg.norm(
                        minimum_image(structure.positions[j] - structure.positions[i], structure.cell)
                    ),
                )
                if not ids[o].startswith("a"):
                    continue
                for m, s in enumerate(structure.symbols):
                    bond = minimum_image(structure.positions[m] - structure.positions[o], structure.cell)
                    if s != "Co" or np.linalg.norm(bond) > 2.2:
                        continue
                    oh = minimum_image(structure.positions[i] - structure.positions[o], structure.cell)
                    angle = np.degrees(np.arccos(bond @ oh / np.linalg.norm(bond) / np.linalg.norm(oh)))
                    self.assertGreaterEqual(angle, 85.0 - 1e-6)
                    checked += 1
        self.assertGreater(checked, 0)


class UngroupedSlabTests(TempCase):
    def test_init_writes_a_grouped_copy_and_generate_runs(self):
        slab = ungrouped_ruo2(self.root / "RuO2_mixed.vasp")
        self.assertFalse(species_grouped(read_poscar(slab).symbols))
        lines = []
        init_project(slab, self.root / "p", scope="aem", interactive=False, output=lines.append)
        self.assertTrue(any("more than one block" in line for line in lines))
        project = load_project(self.root / "p")
        copy = read_poscar(self.root / "p" / project["slab"]["file"])
        self.assertTrue(species_grouped(copy.symbols))
        self.assertEqual(sorted(copy.symbols), sorted(read_poscar(slab).symbols))
        self.assertEqual(slab.read_text()[:30], "RuO2(110), atoms ordered by he")  # the given file is untouched
        for site in project["sites"]:  # site IDs refer to the grouped copy
            atom = site.get("metal") or site["id"]
            self.assertEqual(copy.symbols[int(atom[1:]) - 1], "Ru" if site["kind"] == "open_metal" else "O")
        report = generate(self.root / "p" / "project.yaml")
        self.assertEqual(report["candidate_nodes"], 9)

    def test_a_hand_made_project_gets_a_clear_message(self):
        project = self.project_copy()
        ungrouped_ruo2(project / "slab" / "mixed.vasp")
        data = self.edit(project)
        data["slab"]["file"] = "slab/mixed.vasp"
        (project / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        with self.assertRaisesRegex(
            ValueError, "more than one block.*`mechanet init` on the slab file in a new folder"
        ):
            generate(project / "project.yaml")


class SingleAtomCatalystTests(TempCase):
    def test_fe_n4_in_graphene_gets_an_open_metal_site(self):
        slab = fen4_graphene(self.root / "FeN4.vasp")
        report = inspect_slab(slab)
        sites = report["proposal"]["sites"]
        self.assertEqual([(s["kind"], s["native"]) for s in sites], [("open_metal", "empty")])
        fe = next(i for i, s in enumerate(read_poscar(slab).symbols) if s == "Fe")
        self.assertEqual(sites[0]["metal"], f"a{fe + 1:04d}")
        self.assertTrue(report["open_metal_sites"][0]["rule"].startswith("free axial"))
        self.assertIsNone(report["formal_count"]["average_metal_oxidation_state"])
        self.assertIn("covalent framework (C)", report["formal_count"]["note"])
        text = summary(report)
        self.assertNotIn("[('", text)
        self.assertNotIn("{'", text)
        init_project(slab, self.root / "p", scope="aem", interactive=False, output=quiet)
        built = generate(self.root / "p" / "project.yaml")
        self.assertEqual(built["complete_stoichiometric_cycles"], 2)
        oh = next(
            state
            for state in (Path(built["campaign"]) / "states").iterdir()
            if {k: v for k, v in read_json(state / "state.json")["delta_atoms"].items() if v} == {"O": 1, "H": 1}
        )
        structure = read_poscar(oh / "POSCAR")
        fe_i = structure.symbols.index("Fe")
        o_i = next(i for i, s in enumerate(structure.symbols) if s == "O")
        bond = minimum_image(structure.positions[o_i] - structure.positions[fe_i], structure.cell)
        self.assertAlmostEqual(abs(bond[2]) / np.linalg.norm(bond), 1.0, places=3)  # axial, along the normal

    def test_a_corrugated_sheet_has_no_interior_either(self):
        report = inspect_slab(fen4_graphene(self.root / "FeN4.vasp", ripple_A=0.35))
        self.assertGreater(report["slab_thickness_A"], 0.6)  # thicker than the layer gap
        self.assertEqual([s["kind"] for s in report["proposal"]["sites"]], ["open_metal"])

    def test_buried_dopants_are_not_opened(self):
        from ase.build import fcc100
        from ase.io import write

        slab = fcc100("Pt", (2, 2, 6), vacuum=8.0)
        z = slab.positions[:, 2]
        second = sorted(set(np.round(z, 2)))[-2]
        slab[next(i for i in range(len(slab)) if abs(z[i] - second) < 0.05)].symbol = "Au"
        slab = slab[slab.numbers.argsort(kind="stable")]
        write(self.root / "Pt100_Au.vasp", slab, format="vasp", direct=True)
        report = inspect_slab(self.root / "Pt100_Au.vasp")
        # Au has no interior atom to compare with, but it sits below the outermost layer
        self.assertEqual({s["element"] for s in report["open_metal_sites"]}, {"Pt"})
        self.assertTrue(all(s["rule"].startswith("below") for s in report["open_metal_sites"]))

    def test_a_relaxed_single_atom_site_stays_axial(self):
        from ase.io import read, write

        atoms = read(fen4_graphene(self.root / "FeN4.vasp"))
        fe = next(a.index for a in atoms if a.symbol == "Fe")
        # the noise of a relaxed structure: four in-plane Fe-N rays that nearly cancel
        for shift in ([0.005, 0.0, 0.0], [0.002, 0.0, -0.05]):
            moved = atoms.copy()
            moved.positions[fe] += shift
            write(self.root / "relaxed.vasp", moved, format="vasp", direct=True)
            sites = inspect_slab(self.root / "relaxed.vasp")["open_metal_sites"]
            self.assertEqual(len(sites), 1, shift)
            self.assertEqual(sites[0]["direction"], [0.0, 0.0, 1.0])
        init_project(self.root / "relaxed.vasp", self.root / "p", scope="aem", interactive=False, output=quiet)
        built = generate(self.root / "p" / "project.yaml")
        for state in (Path(built["campaign"]) / "states").iterdir():
            if {k: v for k, v in read_json(state / "state.json")["delta_atoms"].items() if v} == {"O": 1, "H": 1}:
                structure = read_poscar(state / "POSCAR")
                fe_i, o_i = structure.symbols.index("Fe"), structure.symbols.index("O")
                bond = minimum_image(structure.positions[o_i] - structure.positions[fe_i], structure.cell)
                self.assertAlmostEqual(abs(bond[2]) / np.linalg.norm(bond), 1.0, places=3)

    def test_a_stepped_slab_keeps_its_interior(self):
        from ase.build import bulk, surface
        from ase.io import write

        slab = surface(bulk("Pt", "fcc", a=3.92, cubic=True), (3, 2, 1), 6, vacuum=8.0)  # planes < 0.6 A apart
        write(self.root / "Pt321.vasp", slab[slab.numbers.argsort(kind="stable")], format="vasp", direct=True)
        report = inspect_slab(self.root / "Pt321.vasp")
        self.assertEqual(report["layers"]["count"], 1)
        self.assertTrue(report["bulk_like_coordination"]["Pt"]["from"].startswith("interior"))
        self.assertTrue(all(s["rule"].startswith("below") for s in report["open_metal_sites"]))
        self.assertIn("1 layer\n", summary(report))

    def test_a_few_framework_bonds_keep_the_formal_count(self):
        from mechanet.structure import Structure, write_poscar

        rutile = read_poscar(RUO2 / "slab" / "RuO2_110.vasp")
        carbon = rutile.positions[rutile.ids.index("a0016")] + [0.0, 0.0, 1.25]  # a C on one bridging O
        write_poscar(
            self.root / "RuO2_C.vasp",
            Structure(
                rutile.symbols + ["C"],
                np.vstack([rutile.positions, carbon]),
                rutile.cell,
                np.vstack([rutile.mobile, [[True] * 3]]),
                rutile.ids + [f"a{len(rutile.ids) + 1:04d}"],
            ),
        )
        self.assertEqual(inspect_slab(self.root / "RuO2_C.vasp")["formal_count"]["average_metal_oxidation_state"], 4.0)

    def test_rutile_detection_is_unchanged(self):
        report = inspect_slab(RUO2 / "slab" / "RuO2_110.vasp")
        self.assertEqual(sorted(s["id"] for s in report["proposal"]["sites"]), ["a0016", "a0019", "m0038", "m0046"])
        self.assertTrue(all(s["rule"].startswith("below") for s in report["open_metal_sites"]))
        self.assertEqual(report["formal_count"]["average_metal_oxidation_state"], 4.0)


class PluginReactionTests(TempCase):
    def make_her_project(self):
        slab = self.root / "Pt_111.vasp"
        shutil.copy(EXAMPLES / "pt_111" / "slab" / "Pt_111.vasp", slab)
        lines = []
        init_project(
            slab, self.root / "pt", reaction="her", plugins=[HER_PLUGIN], interactive=False, output=lines.append
        )
        return self.root / "pt", lines

    def test_init_takes_a_plugin_reaction(self):
        project, lines = self.make_her_project()
        data = load_project(project)
        self.assertEqual(data["reaction"], "her")
        self.assertEqual(data["plugins"], ["plugins/her.py"])
        self.assertEqual(data["scope"]["families"], ["her_volmer_heyrovsky"])
        self.assertEqual(data["thermodynamics"]["potentials_V"], [0.0, -0.1, -0.2])
        self.assertTrue((project / "plugins" / "her.py").is_file())
        self.assertFalse((project / "plugins" / "__pycache__").exists())
        with self.assertRaisesRegex(ValueError, "needs --plugin FILE"):
            init_project(
                project / "slab" / "Pt_111.vasp", self.root / "q", reaction="her", interactive=False, output=quiet
            )

    def test_campaign_commands_load_the_campaign_plugins(self):
        project, _ = self.make_her_project()
        generate(project / "project.yaml")
        with patch.dict(SPECIES):
            SPECIES.pop("H2", None)  # as in a fresh process: only the campaign's plugin defines H2
            code, text = run("show-cycle", "her_volmer_heyrovsky_m0010", project / "campaign")
        self.assertEqual(code, 0, text)
        self.assertIn("takes up H+ + e-", text)

    def test_reference_molecules_follow_the_reaction(self):
        project, _ = self.make_her_project()
        generate(project / "project.yaml")
        code, text = run("references", "--molecules", self.root / "refs", "--campaign", project / "campaign")
        self.assertEqual(code, 0, text)
        self.assertEqual(sorted(p.name for p in (self.root / "refs").iterdir()), ["H2"])

    def test_a_campaign_keeps_its_plugins_when_moved(self):
        project, _ = self.make_her_project()
        built = generate(project / "project.yaml")
        config = read_json(project / "campaign" / "config.json")
        self.assertEqual(config["plugins"], ["inputs/plugins/her.py"])
        self.assertIn("plugin:her.py", read_json(project / "campaign" / "build_report.json")["input_provenance"])
        moved = self.root / "elsewhere" / "campaign"
        shutil.copytree(built["campaign"], moved)
        shutil.rmtree(project)  # the project and its plugin files are gone (a campaign copied to a cluster)
        with patch.dict(SPECIES):
            SPECIES.pop("H2", None)
            code, text = run("show-cycle", "her_volmer_heyrovsky_m0010", moved)
        self.assertEqual(code, 0, text)
        self.assertIn("takes up H+ + e-", text)

    def test_reference_molecules_follow_the_slab_overrides(self):
        project, _ = self.make_her_project()
        generate(project / "project.yaml")
        config = read_json(project / "campaign" / "config.json")
        config["vasp"]["common_overrides"] = {"ENCUT": 550, "NCORE": 8}
        write_json(project / "campaign" / "config.json", config)
        code, text = run("references", "--molecules", self.root / "refs", "--campaign", project / "campaign")
        self.assertEqual(code, 0, text)
        self.assertIn("INCAR_final with the slab's overrides: ENCUT=550", text)
        incar = (self.root / "refs" / "H2" / "relax" / "INCAR").read_text()
        self.assertIn("ENCUT = 550", incar)
        self.assertNotIn("NCORE", incar)  # parallelisation is not part of the protocol

    def test_a_campaign_whose_plugin_is_gone_still_lists_and_cleans(self):
        project, _ = self.make_her_project()
        generate(project / "project.yaml")
        config = read_json(project / "campaign" / "config.json")
        config["plugins"] = [str(self.root / "gone" / "her.py")]  # an older campaign copied elsewhere
        write_json(project / "campaign" / "config.json", config)
        code, text = run("list", project / "campaign")
        self.assertEqual(code, 0, text)

    def test_reference_molecules_without_a_geometry_are_named(self):
        plugin = self.root / "her_n2.py"
        plugin.write_text(
            HER_PLUGIN.read_text().replace('required_references = ("H2",)', 'required_references = ("H2", "N2")')
        )
        slab = self.root / "Pt_111.vasp"
        shutil.copy(EXAMPLES / "pt_111" / "slab" / "Pt_111.vasp", slab)
        init_project(slab, self.root / "pt", reaction="her", plugins=[plugin], interactive=False, output=quiet)
        generate(self.root / "pt" / "project.yaml")
        code, text = run("references", "--molecules", self.root / "refs", "--campaign", self.root / "pt" / "campaign")
        self.assertEqual(code, 0, text)
        self.assertIn("her also needs N2, which has no built-in geometry here", text)

    def test_potentials_on_another_scale_are_compared_on_the_reaction_scale(self):
        from mechanet.project import _stray_potentials

        oer = default_registry().reaction("oer")
        thermo = {"potential_scale": "SHE", "pH": 14, "potentials_V": [0.4, 0.6]}  # 1.23 and 1.43 V vs RHE
        self.assertIsNone(_stray_potentials({"thermodynamics": thermo}, oer))
        thermo["potentials_V"] = [-1.5, -1.4]
        warning = _stray_potentials({"thermodynamics": thermo}, oer)
        self.assertIn("V vs RHE at pH 14", warning)

    def test_potentials_left_over_from_another_reaction_are_flagged(self):
        project = self.project_copy(EXAMPLES / "pt_111", "pt")
        data = self.edit(project, reaction="her", plugins=[str(HER_PLUGIN)])
        data["scope"]["families"] = ["her_volmer_heyrovsky"]
        (project / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        report = generate(project / "project.yaml")
        self.assertTrue(any("left over from another reaction" in w for w in report["warnings"]))


class PluginLoadingTests(TempCase):
    CHARGED = (
        "from mechanet.reactions import Species\n\n\n"
        "def mu_oh(reservoirs, conditions):\n    return 0.0\n\n\n"
        'OH = Species("OH-", (("O", 1), ("H", 1)), -1, "aqueous", ("H2O", "H2"), chemical_potential=mu_oh)\n\n\n'
        "def register(registry):\n    registry.add_species(OH)\n"
    )

    def test_a_plugin_file_and_its_copies_load_more_than_once(self):
        first, copy = self.root / "a" / "ohminus.py", self.root / "b" / "ohminus.py"
        for path in (first, copy):
            path.parent.mkdir()
            path.write_text(self.CHARGED)
        with patch.dict(SPECIES):
            default_registry([str(first)])
            registry = default_registry([str(copy)])  # the campaign's copy, in the same process
            default_registry([str(first)])
            self.assertIn(("species", "OH-", str(copy.resolve())), registry.loaded)

    def test_the_plugins_command_reports_a_broken_plugin(self):
        project = self.root / "p"
        project.mkdir()
        (project / "broken.py").write_text("raise RuntimeError('half-written plugin')\n")
        (project / "notes.txt").write_text("not a plugin\n")
        shutil.copy(HER_PLUGIN, project / "her.py")
        (project / "project.yaml").write_text("name: p\nplugins: [broken.py, notes.txt, her.py]\n")
        here = os.getcwd()
        os.chdir(project)
        try:
            code, text = run("plugins")
        finally:
            os.chdir(here)
        self.assertEqual(code, 0, text)
        errors = [line for line in text.splitlines() if line.startswith("PLUGIN ERROR")]
        self.assertEqual(len(errors), 2, text)  # every broken file is reported
        self.assertIn("half-written plugin", errors[0])
        self.assertIn("not a Python file", errors[1])
        self.assertIn("her_volmer_heyrovsky", text)  # what does load is still listed
        self.assertEqual(default_registry().errors, [])  # the shared registry is not modified

    def test_a_plugin_must_be_a_python_file(self):
        notes = self.root / "her.txt"
        notes.write_text("def register(registry): pass\n")
        with self.assertRaisesRegex(ValueError, "not a Python file"):
            Registry().load_file(notes)

    def test_compiled_files_are_not_plugins(self):
        compiled = self.root / "her.pyc"
        compiled.write_bytes(b"\x00")
        with self.assertRaisesRegex(ValueError, "not a Python file"):
            Registry().load_file(compiled)

    def test_each_registry_keeps_its_own_family_origin(self):
        first, copy = self.root / "a" / "her.py", self.root / "b" / "her.py"
        for path in (first, copy):
            path.parent.mkdir()
            shutil.copy(HER_PLUGIN, path)
        one, two = default_registry([str(first)]), default_registry([str(copy)])
        self.assertEqual(one.families["her_volmer_heyrovsky"].origin, str(first.resolve()))
        self.assertEqual(two.families["her_volmer_heyrovsky"].origin, str(copy.resolve()))

    def test_plugin_names_differing_in_case_are_refused(self):
        for folder, name in (("a", "Her.py"), ("b", "her.py")):
            (self.root / folder).mkdir()
            shutil.copy(HER_PLUGIN, self.root / folder / name)
        slab = self.root / "Pt_111.vasp"
        shutil.copy(EXAMPLES / "pt_111" / "slab" / "Pt_111.vasp", slab)
        with self.assertRaisesRegex(ValueError, "share a name"):
            init_project(
                slab,
                self.root / "p",
                reaction="her",
                plugins=[self.root / "a" / "Her.py", self.root / "b" / "her.py"],
                interactive=False,
                output=quiet,
            )

    def test_a_plugin_list_that_is_not_a_list_is_reported(self):
        project = self.root / "p"
        project.mkdir()
        (project / "project.yaml").write_text("name: p\nplugins: her.py\n")
        here = os.getcwd()
        os.chdir(project)
        try:
            code, text = run("plugins")
        finally:
            os.chdir(here)
        self.assertEqual(code, 0, text)
        self.assertIn("`plugins` must be a list of plugin files", text)

    def test_scope_fields_must_be_a_tuple_of_names(self):
        for fields in ("sites", ("sites", 3)):
            with self.assertRaisesRegex(ValueError, "scope_fields must be a tuple"):
                Registry().add_family(Family("x", quiet, "oer", "x", [], scope_key="x", scope_fields=fields))

    def test_a_scope_block_of_a_family_that_is_not_built_is_flagged(self):
        project = self.project_copy()
        self.edit(project, scope={"families": ["aem"], "coverage": {"sites": ["m0038"]}})
        report = generate(project / "project.yaml")
        self.assertTrue(any(w.startswith("scope.coverage configures the coverage family") for w in report["warnings"]))


class MessageTests(TempCase):
    def test_unused_reference_entries_are_no_error(self):
        refs = {
            "temperature_K": 298.15,
            "protocol_id": "test",
            "H2": {"energy_e0_eV": -6.77, "free_energy_correction_eV": 0.0, "phase": "gas"},
            "H2O": {"energy_e0_eV": None, "free_energy_correction_eV": None, "phase": "liquid"},
            "oxygen_reference": {
                "method": "water_formation",
                "delta_G_2H2O_to_O2_2H2_eV": 4.92,
                "temperature_K": 298.15,
            },
        }
        self.assertEqual(reservoir_chemical_potentials(refs, needed=("H2",))["errors"], [])
        self.assertTrue(reservoir_chemical_potentials(refs, needed=("H2", "H2O"))["errors"])

    def test_water_pressure_move_is_a_note_only_when_something_moves(self):
        refs = {
            "temperature_K": 298.15,
            "protocol_id": "test",
            "H2": {"energy_e0_eV": -6.77, "free_energy_correction_eV": 0.0, "phase": "gas"},
            "H2O": {
                "energy_e0_eV": -14.22,
                "free_energy_correction_eV": 0.0,
                "phase": "liquid",
                "Gcorr_pressure_bar": 0.035,
            },
            "oxygen_reference": {
                "method": "water_formation",
                "delta_G_2H2O_to_O2_2H2_eV": 4.92,
                "temperature_K": 298.15,
            },
        }
        result = reservoir_chemical_potentials(refs)
        self.assertFalse(any("moved from" in w for w in result["warnings"] + result["notes"]))
        refs["H2O"]["Gcorr_pressure_bar"] = 1.0
        result = reservoir_chemical_potentials(refs)
        # moving the 1 bar correction to the liquid-water pressure is the convention, not a problem
        self.assertTrue(any("moved from 1 bar" in n for n in result["notes"]))
        self.assertFalse(any("moved from" in w for w in result["warnings"]))

    def test_command_list_follows_a_study_and_hides_run_state(self):
        text = build_parser(default_registry()).format_help()
        listed = [line.split()[0] for line in text.splitlines() if line.startswith("    ") and line[4] != " "]
        order = ["examples", "inspect", "init", "generate", "plan", "submit", "collect", "thermo", "pathways", "report"]
        self.assertEqual([c for c in listed if c in order], order)
        self.assertNotIn("run-state", listed)
        with self.assertRaises(SystemExit) as caught, contextlib.redirect_stdout(io.StringIO()):
            main(["run-state", "--help"])
        self.assertEqual(caught.exception.code, 0)

    def test_pourbaix_summary_is_text(self):
        from mechanet.pourbaix import _text

        out = {
            "phases": [
                {"node_id": "n1", "label": "m1:OOH", "area_fraction": 0.5, "releases_product_downhill": "in part"},
                {"node_id": "ref", "label": "native", "area_fraction": 0.5, "releases_product_downhill": None},
            ],
            "transitions": [
                {"pH": 0, "from_V": -0.5, "to_V": 0.8, "label": "m1:OH", "releases_product_downhill": "throughout"},
                {"pH": 0, "from_V": 0.8, "to_V": 2.5, "label": "m1:OOH", "releases_product_downhill": "in part"},
                {"pH": 7, "from_V": -0.5, "to_V": 2.5, "label": "m1:OOH", "releases_product_downhill": "throughout"},
            ],
            "left_out": [{"node_id": "n2", "label": "m1:OO", "reason": "bound O2 is given off downhill"}],
            "nodes_without_energy": 0,
            "analysis_level": "level",
            "notes": [
                "Releases the product downhill in its region: m1:OOH (in part).",
                "mu(H2O) moved from 1 bar (the declared Gcorr_pressure_bar) to the liquid-water pressure 0.035 bar.",
            ],
            "warnings": [],
            "files": ["/c/pourbaix/pourbaix.json"],
        }
        text = _text(out, (0, 14), (-0.5, 2.5), "SHE")
        self.assertIn("in part of its region", text)
        # every segment carries its flag, the first and a single one included
        self.assertIn(
            "at pH 0: m1:OH up to 0.80 V (gives the product off downhill), "
            "m1:OOH above 0.80 V (gives the product off downhill in part of its range)",
            text,
        )
        self.assertIn("at pH 7: m1:OOH throughout (gives the product off downhill)", text)
        self.assertIn("note: mu(H2O) moved from 1 bar", text)
        self.assertNotIn("note: Releases the product downhill", text)  # the flags already say it
        self.assertIn("left out: m1:OO", text)
        self.assertNotIn("node_id=", text)

    def test_collect_summary_is_text(self):
        from mechanet import cli

        summary_ = {
            "n_states": 1,
            "n_not_run": 0,
            "n_reassigned_to_other_node": 1,
            "reassigned_states": {"n_a__v1": "n_b"},
            "n_final_valid": 1,
            "n_electronic_valid": 1,
            "n_free_energy_valid": 0,
            "protocol_notes": [],
            "energy_source": "02_final E0",
            "limitations": ["A limit."],
        }
        args = SimpleNamespace(campaign=None, out=None, temperature=None, cutoff=None, imaginary_tolerance=None)
        with (
            patch("mechanet.cli._campaign", return_value=self.root),
            patch("mechanet.analysis.collect", return_value={"out_dir": "/c/analysis", "summary": summary_}),
        ):
            _, text = cli.cmd_collect(args, None)
        self.assertIn("collected 1 state into /c/analysis", text)
        self.assertIn("n_a__v1 -> n_b", text)
        self.assertNotIn("summary:", text)

    def test_report_counts_one_intermediate_in_the_singular(self):
        from mechanet.campaign import report as write_report

        project = self.project_copy()
        self.edit(project, scope={"families": ["aem"]})
        campaign = Path(generate(project / "project.yaml")["campaign"])
        write_json(
            campaign / "analysis" / "thermodynamics.json",
            {
                "analysis_level": "level",
                "scale": "RHE",
                "conditions": {"pH": 0.0},
                "cycles": [],
                "best_cycles": {},
                "network_resting_states": {
                    "1.23": {"node_id": "n1", "formation_free_energy_eV": -0.4, "transient_nodes": {"n2": {}}}
                },
                "warnings": [],
            },
        )
        text = write_report(campaign)
        self.assertIn("1 intermediate that gives the product off downhill", text)
        self.assertNotIn("1 intermediates", text)


class ReferencesProvenanceTests(TempCase):
    """`references --collect` vouches only for what it checked, every molecule, on every collect."""

    def potcars(self, o_label="O"):
        root = self.root / f"paw_{o_label}"
        root.mkdir(exist_ok=True)
        paths = {}
        for symbol, label in (("H", "H"), ("O", o_label)):
            path = root / f"{symbol}_POTCAR"
            path.write_text(f"  PAW_PBE {label} 01Jan2000\n   TITEL  = PAW_PBE {label} 01Jan2000\n End of Dataset\n")
            paths[symbol] = str(path)
        return paths

    def run_molecules(self, folder, paw=True, names=("H2", "H2O")):
        fake = self.root / "fake_vasp.py"
        fake.write_text(FAKE_VASP)
        molecule_inputs(EXAMPLE / "inputs" / "INCAR_final", folder, names, potcars=self.potcars() if paw else None)
        for name in names:
            relax, freq = folder / name / "relax", folder / name / "freq"
            subprocess.run([sys.executable, "-I", str(fake)], cwd=relax, check=True, capture_output=True)
            shutil.copy(relax / "CONTCAR", freq / "POSCAR")
            subprocess.run([sys.executable, "-I", str(fake)], cwd=freq, check=True, capture_output=True)
        return folder

    def campaign(self, config=None, paw=True):
        campaign = self.root / "campaign"
        shutil.rmtree(campaign, ignore_errors=True)
        (campaign / "inputs").mkdir(parents=True)
        shutil.copy(EXAMPLE / "inputs" / "INCAR_final", campaign / "inputs" / "INCAR_final")
        if config is not None:
            write_json(campaign / "config.json", config)
        if paw:
            datasets = [
                {"element": "O", "titel": "PAW_PBE O 01Jan2000"},
                {"element": "H", "titel": "PAW_PBE H 01Jan2000"},
            ]
            write_json(campaign / "potcars" / "abc" / "provenance.json", {"datasets": datasets})
        return campaign

    @staticmethod
    def edit_incar(path, tag, value):
        lines = [line for line in path.read_text().splitlines() if not line.split("=")[0].strip() == tag]
        path.write_text("\n".join(lines + ([f"{tag} = {value}"] if value is not None else [])) + "\n")

    def flags(self, refs):
        return yaml.safe_load(refs.read_text())["provenance"]

    def test_collect_documents_the_protocol_and_what_it_checked(self):
        mol = self.run_molecules(self.root / "mol")
        refs = write_template(self.root / "references.yaml")
        result = collect_references(mol, refs, campaign=self.campaign())
        data = yaml.safe_load(refs.read_text())
        self.assertIn("IVDW=12", data["protocol_id"])
        self.assertIn("references --collect", data["protocol_id"])
        for flag in ("temperature_verified", "water_standard_state_verified", "DFT_protocol_compatibility_verified"):
            self.assertIs(data["provenance"][flag], True, flag)
        self.assertEqual(result["warnings"], [])
        self.assertEqual(data["H2"]["Gcorr_temperature_K"], 298.15)
        self.assertNotIn("H2:", data["protocol_id"])  # one protocol for both molecules
        for run in ("relax", "freq"):
            self.assertEqual(data["H2O"]["protocol"][run]["paw"], ["PAW_PBE O 01Jan2000", "PAW_PBE H 01Jan2000"])
        thermo = reservoir_chemical_potentials(data)
        self.assertFalse(any("not been documented" in w or "unverified" in w for w in thermo["warnings"]), thermo)

    def test_every_molecule_is_compared(self):
        mol = self.run_molecules(self.root / "mol")
        self.edit_incar(mol / "H2O" / "relax" / "INCAR", "ENCUT", 400)  # the second molecule only
        refs = write_template(self.root / "references.yaml")
        result = collect_references(mol, refs, campaign=self.campaign())
        self.assertIs(self.flags(refs)["DFT_protocol_compatibility_verified"], False)
        self.assertTrue(any(w.startswith("protocol differs") and "H2O ENCUT" in w for w in result["warnings"]))
        self.assertTrue(any("different ENCUT" in w for w in result["warnings"]), result["warnings"])
        self.assertTrue(any("relax and freq INCARs differ in ENCUT" in w for w in result["warnings"]))

    def test_the_slab_incar_includes_the_overrides(self):
        mol = self.run_molecules(self.root / "mol")
        refs = write_template(self.root / "references.yaml")
        config = {"vasp": {"common_overrides": {"ENCUT": 600}, "stage_overrides": {}}}
        result = collect_references(mol, refs, campaign=self.campaign(config))
        self.assertIs(self.flags(refs)["DFT_protocol_compatibility_verified"], False)
        self.assertTrue(any("ENCUT: molecule 500, slab 600" in w for w in result["warnings"]), result["warnings"])
        # the energies of frequency.central = refine come from 04_static, whose override wins
        config["vasp"]["stage_overrides"] = {"04_static": {"ENCUT": 500}}
        result = collect_references(mol, refs, campaign=self.campaign(config))
        self.assertIs(self.flags(refs)["DFT_protocol_compatibility_verified"], True, result["warnings"])
        self.assertIn("04_static", self.flags(refs)["checked"])

    def test_paw_datasets_of_every_molecule_are_compared(self):
        mol = self.run_molecules(self.root / "mol")
        o_s = self.potcars("O_s")
        for stage in ("relax", "freq"):
            potcar = mol / "H2O" / stage / "POTCAR"
            potcar.write_text(Path(o_s["O"]).read_text() + Path(o_s["H"]).read_text())
        refs = write_template(self.root / "references.yaml")
        result = collect_references(mol, refs, campaign=self.campaign())
        self.assertIs(self.flags(refs)["DFT_protocol_compatibility_verified"], False)
        self.assertTrue(any("H2O PAW O: molecule PAW_PBE O_s 01Jan2000" in w for w in result["warnings"]))

    def test_nothing_is_claimed_without_paw_information(self):
        mol = self.run_molecules(self.root / "mol", paw=False)
        refs = write_template(self.root / "references.yaml")
        result = collect_references(mol, refs, campaign=self.campaign())
        self.assertIs(self.flags(refs)["DFT_protocol_compatibility_verified"], False)
        self.assertTrue(any(w.startswith("PAW datasets of H2, H2O not compared") for w in result["warnings"]))
        mol = self.run_molecules(self.root / "mol2")
        result = collect_references(mol, refs, campaign=self.campaign(paw=False))
        self.assertIs(self.flags(refs)["DFT_protocol_compatibility_verified"], False)
        self.assertTrue(any("no POTCAR record in the campaign" in w for w in result["warnings"]))
        # a molecule element the campaign has no dataset of yet (H before any OH state ran) is not compared
        campaign = self.campaign()
        write_json(
            campaign / "potcars" / "abc" / "provenance.json",
            {"datasets": [{"element": "O", "titel": "PAW_PBE O 01Jan2000"}]},
        )
        result = collect_references(mol, refs, campaign=campaign)
        self.assertIs(self.flags(refs)["DFT_protocol_compatibility_verified"], False)
        self.assertIn(
            "PAW datasets of H2, H2O not compared: no slab dataset of H recorded or readable here", result["warnings"]
        )

    def test_flags_are_set_again_on_every_collect(self):
        mol = self.run_molecules(self.root / "mol")
        refs = write_template(self.root / "references.yaml")
        collect_references(mol, refs, campaign=self.campaign())
        self.assertIs(self.flags(refs)["DFT_protocol_compatibility_verified"], True)
        self.edit_incar(mol / "H2" / "relax" / "INCAR", "ENCUT", 300)
        result = collect_references(mol, refs)  # no campaign this time: nothing was compared
        data = yaml.safe_load(refs.read_text())
        self.assertIs(data["provenance"]["DFT_protocol_compatibility_verified"], False)
        self.assertNotIn("checked", data["provenance"])
        self.assertIn("ENCUT=300", data["protocol_id"])
        self.assertTrue(any("no campaign here" in w for w in result["warnings"]))
        # a correction computed at another temperature is not vouched for, and the analysis refuses it
        data["H2O"]["Gcorr_temperature_K"] = 320.0
        refs.write_text(yaml.safe_dump(data))
        collect_references(self.run_molecules(self.root / "h2", names=("H2",)), refs)
        data = yaml.safe_load(refs.read_text())
        self.assertIs(data["provenance"]["temperature_verified"], False)
        errors = reservoir_chemical_potentials(data)["errors"]
        self.assertTrue(any("computed at 320 K" in e for e in errors), errors)

    def test_hand_entered_values_are_kept_but_not_vouched_for(self):
        refs = write_template(self.root / "references.yaml")
        data = yaml.safe_load(refs.read_text())
        data["H2O"].update(energy_e0_eV=-14.22, free_energy_correction_eV=0.0, Gcorr_pressure_bar=0.035)
        refs.write_text(yaml.safe_dump(data))
        result = collect_references(
            self.run_molecules(self.root / "mol", names=("H2",)), refs, campaign=self.campaign()
        )
        data = yaml.safe_load(refs.read_text())
        self.assertEqual(data["H2O"]["energy_e0_eV"], -14.22)
        for flag in ("temperature_verified", "water_standard_state_verified", "DFT_protocol_compatibility_verified"):
            self.assertIs(data["provenance"][flag], False, flag)
        self.assertTrue(any(w.startswith("H2O: free-energy correction entered by hand") for w in result["warnings"]))
        self.assertTrue(any(w.startswith("H2O: entered by hand; its protocol") for w in result["warnings"]))

    def test_a_protocol_written_by_the_user_is_kept(self):
        mol = self.run_molecules(self.root / "mol")
        refs = write_template(self.root / "references.yaml")
        data = yaml.safe_load(refs.read_text())
        for mine in ("PBE+U(Ni 6.2)-D3(BJ), 520 eV, PAW PBE 54", "RPBE-D3, 450 eV, O_s PAW (replaces O)"):
            data["protocol_id"] = mine
            refs.write_text(yaml.safe_dump(data))
            collect_references(mol, refs)
            self.assertEqual(yaml.safe_load(refs.read_text())["protocol_id"], mine)
        for placeholder in (None, "user", "REPLACE with the protocol"):
            data["protocol_id"] = placeholder
            refs.write_text(yaml.safe_dump(data))
            collect_references(mol, refs)
            self.assertIn("references --collect", yaml.safe_load(refs.read_text())["protocol_id"])

    def test_the_frequency_runs_are_compared_too(self):
        mol = self.run_molecules(self.root / "mol")
        self.edit_incar(mol / "H2O" / "freq" / "INCAR", "ENCUT", 300)
        o_s = self.potcars("O_s")
        (mol / "H2O" / "freq" / "POTCAR").write_text(Path(o_s["O"]).read_text() + Path(o_s["H"]).read_text())
        refs = write_template(self.root / "references.yaml")
        result = collect_references(mol, refs, campaign=self.campaign())
        self.assertIs(self.flags(refs)["DFT_protocol_compatibility_verified"], False)
        self.assertTrue(any("H2O (freq) ENCUT: molecule 300, slab 500" in w for w in result["warnings"]))
        self.assertTrue(any("H2O (freq) PAW O: molecule PAW_PBE O_s 01Jan2000" in w for w in result["warnings"]))

    def test_a_hand_entered_g_ev_is_kept_until_the_frequencies_run(self):
        mol = self.run_molecules(self.root / "mol")
        shutil.rmtree(mol / "H2O" / "freq")
        refs = write_template(self.root / "references.yaml")
        data = yaml.safe_load(refs.read_text())
        data["H2O"]["G_eV"] = -14.5
        refs.write_text(yaml.safe_dump(data))
        result = collect_references(mol, refs)
        data = yaml.safe_load(refs.read_text())
        self.assertEqual(data["H2O"]["G_eV"], -14.5)
        self.assertTrue(any("G_eV -14.5 eV entered by hand is kept" in w for w in result["warnings"]))
        mu = reservoir_chemical_potentials(data)
        self.assertEqual(mu["errors"], [])
        self.assertAlmostEqual(mu["mu_H2O_eV"], -14.5 + mu["H2O_pressure_correction_eV"], places=9)
        self.assertIs(data["provenance"]["temperature_verified"], False)

    def test_tags_that_are_switched_off_are_not_compared(self):
        from mechanet.references import _differences

        self.assertEqual(_differences({"LHFCALC": ".TRUE."}, {"LHFCALC": "T", "AEXX": "0.25", "HFSCREEN": "0"}), [])
        self.assertEqual(_differences({"LHFCALC": "T", "HFSCREEN": "0.2"}, {"LHFCALC": "T"}), ["HFSCREEN"])
        self.assertEqual(_differences({}, {"EB_K": "80"}), [])  # no solvation on either side
        self.assertEqual(_differences({"LSOL": "T"}, {"LSOL": ".TRUE.", "EB_K": "80"}), ["EB_K"])

    def test_a_named_campaign_must_exist(self):
        mol = self.run_molecules(self.root / "mol")
        refs = write_template(self.root / "references.yaml")
        code, _ = run("references", "--collect", mol, "--out", refs, "--campaign", self.root / "nowhere")
        self.assertNotEqual(code, 0)

    def test_logical_and_numeric_tags_compare_by_value(self):
        from mechanet.references import _tag_value

        self.assertEqual({_tag_value(v, "LASPH") for v in (".TRUE.", "T", ".T.", "true")}, {"TRUE"})
        self.assertEqual(_tag_value(None, "LASPH"), _tag_value(".FALSE.", "LASPH"))
        self.assertEqual(_tag_value("500", "ENCUT"), _tag_value("500.0", "ENCUT"))
        self.assertEqual(_tag_value(None, "IVDW"), _tag_value("0", "IVDW"))
        self.assertEqual(_tag_value("TPSS", "METAGGA"), "TPSS")  # not a logical tag

    def test_compact_files_are_updated_in_place(self):
        mol = self.run_molecules(self.root / "mol")
        refs = self.root / "compact.yaml"
        refs.write_text(
            yaml.safe_dump(
                {
                    "references": {
                        "H2": {"E": -1.0, "Gcorr": 0.1},
                        "H2O": {"E": -2.0, "Gcorr": 0.2, "phase": "liquid"},
                    },
                    "protocol_id": "user",
                }
            )
        )
        result = collect_references(mol, refs, campaign=self.campaign())
        data = yaml.safe_load(refs.read_text())
        self.assertNotIn("H2", data)
        h2 = data["references"]["H2"]
        self.assertEqual(h2["E"], result["molecules"]["H2"]["energy_e0_eV"])
        self.assertNotIn("energy_e0_eV", h2)
        self.assertIn("references --collect", data["protocol_id"])
        self.assertIs(data["provenance"]["DFT_protocol_compatibility_verified"], True, result["warnings"])
        mu = reservoir_chemical_potentials(data)
        self.assertAlmostEqual(mu["mu_H2_eV"], h2["E"] + h2["Gcorr"], places=9)
        self.assertFalse(any("unverified" in w or "not been documented" in w for w in mu["warnings"]), mu)
        # the compact-format warning names only what was assumed (here: the water-formation convention)
        assumed = [w for w in mu["warnings"] if w.startswith("Compact")]
        self.assertEqual(len(assumed), 1, mu["warnings"])
        self.assertIn("assumed the 4.92 eV water-formation convention at 298.15 K.", assumed[0])
        self.assertFalse(any(n.startswith("Compact") for n in mu["notes"]), mu["notes"])  # once, as a warning

    def test_no_water_warnings_without_water(self):
        refs = {
            "temperature_K": 298.15,
            "protocol_id": "test",
            "H2": {"energy_e0_eV": -6.77, "free_energy_correction_eV": 0.0, "phase": "gas"},
            "provenance": {"water_standard_state_verified": False},
        }
        warnings = reservoir_chemical_potentials(refs, needed=("H2",))["warnings"]
        self.assertFalse(any("water" in w.lower() for w in warnings), warnings)


if __name__ == "__main__":
    unittest.main()
