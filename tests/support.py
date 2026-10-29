"""Test helpers: a synthetic VASP executable, fake SLURM commands and a tiny campaign.

The fake executable writes VASP-formatted OUTCAR text (iteration headers, EDIFF markers,
TOTAL-FORCE, FREE ENERGIE and magnetization blocks, timing footer) so the real parser,
runner and frequency assembly are exercised. It never pretends to be DFT: forces come from
a harmonic model F = -K (x - x0) when harmonic.json exists, otherwise from fixed synthetic
values. Behaviour is switched by marker files in <state>/ctl (one-shot markers are consumed
when used); eshift_<stage> and moment_<stage> (values in the file) imitate another
electronic solution in that stage.
"""

import json  # noqa: F401 - kept for modules that import it from here
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
EXAMPLE = ROOT / "examples" / "nife_ldh"

from mechanet.fileio import read_json, write_json  # noqa: E402
from mechanet.structure import Structure, write_poscar  # noqa: E402

POMASS = {
    "Fe": 55.847,
    "Ni": 58.69,
    "K": 39.098,
    "O": 16.0,
    "H": 1.0,
    "Ru": 101.07,
    "Ir": 192.22,
    "Pt": 195.08,
    "Cl": 35.453,
}
# frequency settings of the two protocols (frequency.central)
REFINE_PROTOCOL = {"central": "refine", "ediff": 1e-7, "max_force_eVA": 0.03, "warn_force_eVA": 0.02}
FINAL_PROTOCOL = {"central": "final", "ediff": 1e-6, "max_force_eVA": 0.035, "warn_force_eVA": 0.03}

FAKE_VASP = r"""
import json, os, signal, sys, time
from pathlib import Path

cwd = Path.cwd()
if cwd.parent.name == "units":
    unit, stage, state = cwd.name, cwd.parent.parent.name, cwd.parent.parent.parent
else:
    unit, stage, state = None, cwd.name, cwd.parent
ctl = state / "ctl"

def flag(name):
    return (ctl / name).exists()

def take(name):
    path = ctl / name
    if path.exists():
        path.unlink()
        return True
    return False

inc = {}
for line in Path("INCAR").read_text().splitlines():
    line = line.split("#")[0]
    if "=" in line:
        key, value = line.split("=", 1)
        inc[key.strip().upper()] = value.strip()
lines = Path("POSCAR").read_text().splitlines()
species, counts = lines[5].split(), [int(x) for x in lines[6].split()]
n = sum(counts)
cell = [[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]
rows = lines[9:9 + n]
frac = [[float(x) for x in r.split()[:3]] for r in rows]
mobile = [r.split()[3:6] == ["T", "T", "T"] for r in rows]
pos = [[sum(f[k] * cell[k][j] for k in range(3)) for j in range(3)] for f in frac]
ibrion, nsw = int(inc.get("IBRION", "-1")), int(inc.get("NSW", "0"))
istart, icharg = int(inc.get("ISTART", "0")), int(inc.get("ICHARG", "2"))
with (state / "fake_calls.txt").open("a") as f:
    f.write((f"{stage}/{unit}" if unit else stage) + "\n")

model = json.loads((state / "harmonic.json").read_text()) if (state / "harmonic.json").exists() else None
# another electronic solution: shifted energies and a different Fe moment in this stage
eshift = float((ctl / f"eshift_{stage}").read_text()) if flag(f"eshift_{stage}") else 0.0
fe_moment = float((ctl / f"moment_{stage}").read_text()) if flag(f"moment_{stage}") else 4.0

def forces(p):
    if model is not None:
        x = [c for row in p for c in row]
        d = [a - b for a, b in zip(x, model["x0"])]
        F = [-sum(k * dj for k, dj in zip(row, d)) for row in model["K"]]
        return [F[3 * a:3 * a + 3] for a in range(n)]
    big = flag(f"bigforce_{stage}")
    small = float((ctl / f"force_{stage}").read_text()) if flag(f"force_{stage}") else 0.001
    out = []
    for a in range(n):
        if a == 0:
            out.append([0.4, 0.0, 0.0])          # frozen atom: must be ignored by force checks
        elif big and a == 1:
            out.append([0.2, 0.0, 0.0])
        else:
            out.append([small, 0.0, 0.0])
    return out

def scf(ionic, niter, converged):
    out = []
    for it in range(1, niter + 1):
        e = -100.0 + eshift - 0.001 * ionic + 0.3 / it
        out.append(f"----------------------------------------- Iteration {ionic:4d}({it:4d})  ---------------------------------------\n\n")
        out.append("    POTLOK:  cpu time    0.10: real time    0.10\n     EDDAV:  cpu time    0.10: real time    0.10\n")
        out.append(" Free energy of the ion-electron system (eV)\n  ---------------------------------------------------\n")
        out.append(f"  entropy T*S    EENTRO =        {-0.0001 * it - 0.00001 * ionic:.8f}\n")
        out.append(f"  free energy    TOTEN  =     {e:.8f} eV\n\n")
        out.append(f"  energy without entropy =     {e - 0.002:.8f}  energy(sigma->0) =     {e - 0.001:.8f}\n\n")
    if converged:
        out.append("------------------------ aborting loop because EDIFF is reached ----------------------------------------\n\n")
    else:
        out.append(" ------------------------ aborting loop EDIFF was not reached (unconverged)  ----------------------------\n\n")
    return out

def tail(ionic, p):
    e = -100.0 + eshift - 0.001 * ionic
    out = [" POSITION                                       TOTAL-FORCE (eV/Angst)\n",
           " -----------------------------------------------------------------------------------\n"]
    for xyz, f in zip(p, forces(p)):
        out.append("  %12.5f %12.5f %12.5f   %13.6f %13.6f %13.6f\n" % tuple(xyz + f))
    out += [" -----------------------------------------------------------------------------------\n",
            "    total drift:                                0.000000      0.000000      0.000000\n\n",
            "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n  ---------------------------------------------------\n",
            f"  free  energy   TOTEN  =     {e:.8f} eV\n\n",
            f"  energy  without entropy=     {e - 0.002:.8f}  energy(sigma->0) =     {e - 0.001:.8f}\n\n",
            " magnetization (x)\n\n# of ion       s       p       d       tot\n------------------------------------------\n"]
    for a in range(n):
        moment = fe_moment if a == 0 else 0.0
        out.append(f"    {a + 1}        0.000   0.000   {moment:.3f}   {moment:.3f}\n")
    out += ["--------------------------------------------------\n", "tot          0.000   0.000   4.000   4.000\n\n"]
    return out

def write_contcar(p):
    out = list(lines[:9])
    # Cartesian -> direct for a general cell (3x3 inverse by cofactors)
    a, b, c = cell
    det = (a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0]) + a[2]*(b[0]*c[1]-b[1]*c[0]))
    inv = [[(b[1]*c[2]-b[2]*c[1])/det, (a[2]*c[1]-a[1]*c[2])/det, (a[1]*b[2]-a[2]*b[1])/det],
           [(b[2]*c[0]-b[0]*c[2])/det, (a[0]*c[2]-a[2]*c[0])/det, (a[2]*b[0]-a[0]*b[2])/det],
           [(b[0]*c[1]-b[1]*c[0])/det, (a[1]*c[0]-a[0]*c[1])/det, (a[0]*b[1]-a[1]*b[0])/det]]
    for xyz, r in zip(p, rows):
        f = [sum(xyz[k] * inv[k][j] for k in range(3)) for j in range(3)]
        out.append("  %.16f  %.16f  %.16f   %s" % (f[0], f[1], f[2], " ".join(r.split()[3:6])))
    if flag("hdf5_species"):
        out[5] = "   " + "   ".join({"K": "K_sv"}.get(s, s) + "/1a2b3c4d" for s in species)
    Path("CONTCAR").write_text("\n".join(out) + "\n")

def footer():
    return ["\n General timing and accounting informations for this job:\n ========================================================\n\n",
            "                  Total CPU time used (sec):        1.234\n",
            "                         Elapsed time (sec):        1.234\n"]

def kill_runner():
    os.kill(os.getppid(), signal.SIGKILL)
    sys.exit(137)

head = [f"   number of dos      NEDOS =    301   number of ions     NIONS = {n:6d}\n",
        f"   NELM   = {int(inc.get('NELM', '60')):6d};   NELMIN=  2; NELMDL= -5     # of ELM steps\n",
        f"   EDIFF  = {float(inc.get('EDIFF', '1e-4')):.1E}   stopping-criterion for ELM\n"]
if icharg == 1:
    if not Path("CHGCAR").is_file() or flag(f"chg_atomic_{stage}"):
        head.append(" initial charge density was supplied:\n charge density of overlapping atoms calculated\n")
    else:
        head.append(" initial charge density was supplied:\n charge-density read from file: NiFeLDH\n magnetization density read from file 1\n")
if istart >= 1:
    if Path("WAVECAR").is_file() and not flag("wavefallback"):
        print(" reading WAVECAR\n the WAVECAR file was read successfully")
    else:
        print(" WAVECAR not read")
print(" entering main loop")

chunks = list(head)
tag = unit or stage
if ibrion in (1, 2, 3):
    if take(f"kill_{stage}"):
        for k in range(1, 4):
            p = [[x + (0.01 * k if mobile[a] and j == 0 else 0.0) for j, x in enumerate(xyz)] for a, xyz in enumerate(pos)]
            chunks += scf(k, 3, True) + tail(k, p)
        chunks += scf(4, 2, False)[:-1]
        Path("OUTCAR").write_text("".join(chunks))
        write_contcar(p)
        kill_runner()
    fail = take(f"fail_{stage}")
    slow = flag(f"slow_{stage}")
    steps = 40 if slow else 2
    if slow:
        (ctl / "vasp_started").write_text(str(os.getpid()))
    p = pos
    stopped = False
    for k in range(1, steps + 1):
        p = [[x + (0.001 * k if mobile[a] and j == 0 and slow else 0.0) for j, x in enumerate(xyz)] for a, xyz in enumerate(pos)]
        chunks += scf(k, 2, True) + tail(k, p)
        Path("OUTCAR").write_text("".join(chunks))
        write_contcar(p)
        if slow:
            time.sleep(0.15)
            stopcar = Path("STOPCAR")
            if stopcar.exists() and "LSTOP" in stopcar.read_text() and ".TRUE." in stopcar.read_text().upper():
                stopped = True
                break
    if fail:
        chunks = [c.replace(" 0.001000 ", " 0.200000 ") for c in chunks]
    elif not stopped:
        chunks.append(" reached required accuracy - stopping structural energy minimisation\n")
elif ibrion == 5:
    dof = [(a, j) for a in range(n) if mobile[a] for j in range(3)]
    h = float(inc.get("POTIM", "0.015"))
    chunks += scf(1, 3, True) + tail(1, pos)
    k = 1
    for a, j in dof:
        for s in (1, -1):
            k += 1
            p = [list(xyz) for xyz in pos]
            p[a][j] += s * h
            chunks += scf(k, 2, True) + tail(k, p)
    for block in ("Eigenvectors and eigenvalues of the dynamical matrix", "Eigenvectors after division by SQRT(mass)"):
        chunks.append(f"\n {block}\n ----------------------------------------------------\n\n")
        for i in range(len(dof)):
            chunks.append("%4d f  = %11.6f THz %11.6f 2PiTHz %12.6f cm-1 %11.6f meV\n" % (i + 1, 1.0, 6.28, 100.0 + i, 1.0))
            chunks.append("             X         Y         Z           dx          dy          dz\n")
            for a in range(n):
                chunks.append("  %8.6f  %8.6f  %8.6f    %10.6f  %10.6f  %10.6f\n" % (0, 0, 0, 0, 0, 0.5 if mobile[a] else 0))
    write_contcar(pos)
else:
    if unit and take(f"kill_unit_{unit.split('_')[0]}"):
        chunks += scf(1, 2, False)[:-1]
        Path("OUTCAR").write_text("".join(chunks))
        kill_runner()
    if unit and flag(f"slow_unit_{unit.split('_')[0]}") and not (ctl / "vasp_started").exists():
        (ctl / "vasp_started").write_text(str(os.getpid()))
        time.sleep(1.5)
    chunks += scf(1, 3, not flag(f"nelm_{tag}")) + tail(1, pos)
    write_contcar(pos)
chunks += footer()
Path("OUTCAR").write_text("".join(chunks))
filler = ("synthetic test output, not VASP data\n" * 64)
for name in ("OSZICAR", "PROCAR", "DOSCAR", "EIGENVAL", "PCDAT", "XDATCAR", "REPORT"):
    Path(name).write_text(name + "\n" + filler)
Path("vasprun.xml").write_text(
    '<?xml version="1.0" encoding="ISO-8859-1"?>\n<modeling>\n <calculation>\n'
    '  <scstep><energy><i name="e_fr_energy"> -100.0 </i><i name="e_0_energy"> -100.001 </i></energy></scstep>\n'
    '  <energy><i name="e_fr_energy"> -100.0 </i><i name="e_0_energy"> -100.001 </i></energy>\n'
    ' </calculation>\n <!-- ' + filler.replace("--", "-") + ' -->\n</modeling>\n')
if inc.get("LCHARG", "").upper() == ".TRUE.":
    Path("CHG").write_text("CHG\n" + filler)
if inc.get("LWAVE", "").upper() == ".TRUE.":
    Path("WAVECAR").write_bytes(b"SYNTHETIC TEST WAVECAR " + stage.encode() * 64)
elif istart >= 1 and flag("vasp_deletes_wavecar"):
    Path("WAVECAR").unlink()
if inc.get("LCHARG", "").upper() == ".TRUE.":
    Path("CHGCAR").write_bytes(b"SYNTHETIC TEST CHGCAR " + stage.encode() * 64)
if take(f"postkill_{stage}"):
    kill_runner()
"""


def potcar_text(symbol):
    label = "K_sv" if symbol == "K" else symbol
    return (
        f"  PAW_PBE {label} 01Jan2000\n   TITEL  = PAW_PBE {label} 01Jan2000\n"
        f"   POMASS =   {POMASS[symbol]:.3f}; ZVAL   =    1.000    mass and valenz\n"
        f"   VRHFIN ={symbol}: synthetic\n End of Dataset\n"
    )


def base_config(potcars, fake):
    """A campaign config.json as `mechanet generate` writes it for the NiFe LDH example."""
    return {
        "schema_version": 1,
        "project": "synthetic",
        "reaction": "oer",
        "inputs": {
            "seed": "inputs/seed.vasp",
            "incar_loose": "inputs/INCAR_loose",
            "incar_final": "inputs/INCAR_final",
            "kpoints": "inputs/KPOINTS",
        },
        "slab": {"species_order": ["Fe", "Ni", "K", "O", "H"]},
        "vasp": {
            "command": [sys.executable, "-I", str(fake)],
            "potcars": potcars,
            "magmom_by_element": None,
            "template_species_order": ["Fe", "Ni", "K", "O", "H"],
            "common_overrides": {"NCORE": 8},
            "stage_overrides": {},
            "max_attempts": 2,
            "max_continuations": 8,
        },
        "scheduler": {
            "profile": "test",
            "sbatch": {"nodes": 1, "ntasks": 112, "time": "23:59:00"},
            "setup": [],
            "stop_signal_lead_s": 1800,
            "min_remaining_s": {"relaxation": 1800, "single_point": 5400, "displacement": 2700},
        },
        "frequency": {
            "central": "final",
            "ediff": 1e-06,
            "potim": 0.015,
            "nfree": 2,
            "max_static_e0_shift_eV": 0.01,
            "max_moment_change_muB": 0.5,
            "method": "displacements",
            "nelm": 120,
            "delete_wavecar_after_use": True,
            "max_unit_failures_per_job": 3,
            "prune_unit_outputs": True,
            "unit_prune_files": [
                "PROCAR",
                "DOSCAR",
                "vasprun.xml",
                "EIGENVAL",
                "CHG",
                "PCDAT",
                "XDATCAR",
                "REPORT",
                "WAVECAR",
            ],
        },
        "thermodynamics": {
            "temperature_K": 298.15,
            "potential_scale": "RHE",
            "potentials_V": [1.23, 1.43, 1.53, 1.63],
            "low_frequency_cutoff_cm1": 100.0,
            "imaginary_tolerance_cm1": 80.0,
            "water_equilibrium_pressure_bar": 0.035,
            "electronic_mode_corrections": {"per_H_eV": 0.3, "per_O_eV": 0.05},
        },
        "analysis": {"require_input_hashes": True, "max_eentro_eV": 0.02},
        "refine": {"ediff": 1e-06, "ediffg": -0.015, "ibrion": 2, "nsw": 60},
        "thermo_selection": {"energy_window_eV": 0.1, "max_variants_per_node": 2},
        "topology": {
            "shared_proton_max_OH_A": 1.6,
            "hysteresis_A": {"O-H": [1.2, 1.35], "O-O": [1.75, 1.95], "M-O": [2.2, 2.5]},
        },
    }


def write_potcars(root, symbols=("Fe", "Ni", "K", "O", "H")):
    potcars = {}
    for symbol in symbols:
        path = Path(root) / f"{symbol}_POTCAR"
        path.write_text(potcar_text(symbol))
        potcars[symbol] = str(path)
    return potcars


def make_campaign(root, config_updates=None, harmonic=None, name="test_state"):
    """Campaign with the NiFe example INCAR templates/KPOINTS and a 3-atom Fe-O-H state."""
    root = Path(root)
    campaign = root / "campaign"
    inputs = campaign / "inputs"
    inputs.mkdir(parents=True)
    for item in ("INCAR_loose", "INCAR_final", "KPOINTS"):
        shutil.copy2(EXAMPLE / "inputs" / item, inputs / item)
    state = campaign / "states" / name
    state.mkdir(parents=True)
    (state / "ctl").mkdir()
    atoms = Structure(
        ["Fe", "O", "H"],
        np.array([[3.0, 3.0, 3.0], [4.8, 3.0, 3.0], [5.5, 3.7, 3.0]]),
        np.eye(3) * 15,
        np.array([[False] * 3, [True] * 3, [True] * 3]),
        ["a0001", "a0002", "a0003"],
    )
    write_poscar(state / "POSCAR", atoms)
    write_poscar(inputs / "seed.vasp", atoms)
    meta = {
        "id": name,
        "node_id": "test",
        "symbols": atoms.symbols,
        "atom_ids": atoms.ids,
        "relax_mobile": atoms.mobile.tolist(),
        "frequency_mobile": atoms.mobile.tolist(),
    }
    write_json(state / "state.json", meta)
    if harmonic is not None:
        write_json(
            state / "harmonic.json",
            {"x0": atoms.positions.reshape(-1).tolist(), "K": np.asarray(harmonic, dtype=float).tolist()},
        )
    potcars = write_potcars(root)
    fake = root / "fake_vasp.py"
    fake.write_text(FAKE_VASP)
    config = base_config(potcars, fake)
    # The protocol of the code default (frequency.central = "refine") with its thresholds;
    # tests of the "final" protocol pass FINAL_PROTOCOL in config_updates.
    config["frequency"].update(REFINE_PROTOCOL)
    for section, values in (config_updates or {}).items():
        if isinstance(values, dict):
            config.setdefault(section, {}).update(values)
        else:
            config[section] = values
    write_json(campaign / "config.json", config)
    return campaign, state, meta


def calls(state):
    path = Path(state) / "fake_calls.txt"
    return path.read_text().splitlines() if path.exists() else []


def marker(state, name, text=""):
    (Path(state) / "ctl" / name).write_text(text)


def clean_env(extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_") and not k.startswith("PYTHON")}
    env.update(extra or {})
    return env


def run_cli(args, env=None, wait=True, json_output=True):
    """Run `mechanet [--json] ...` in a separate interpreter (needed when the fake VASP kills its parent)."""
    argv = (["--json"] if json_output else []) + list(args)
    code = "import sys; sys.path.insert(0, %r)\nfrom mechanet.cli import main\nsys.exit(main(%r))" % (str(SRC), argv)
    command = [sys.executable, "-I", "-c", code]
    if not wait:
        return subprocess.Popen(command, env=clean_env(env), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.run(command, env=clean_env(env), capture_output=True, text=True, timeout=300)


def attempt(folder):
    return read_json(Path(folder) / "attempt.json")
