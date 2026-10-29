"""Build the example slabs: RuO2(110), IrO2(110) and Pt(111).

    python build_slabs.py OUT_DIR        (needs pymatgen and ase)

Rutile cells are the neutron-diffraction structures of Bolzan, Fong, Kennedy and Howard,
Acta Cryst. B 53, 373 (1997); Pt uses ASE's reference lattice constant. They are
starting points: relax the bulk with your own functional and rebuild before production.
Each oxide slab is the stoichiometric (110) termination with bridging O on top and
coordinatively unsaturated metal sites, as a 2 x 1 cell; the lower half is fixed.
"""
import sys
from pathlib import Path

import numpy as np
from ase.build import fcc111
from ase.constraints import FixAtoms
from ase.io import write
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor

RUTILE = {"RuO2": ("Ru", 4.4968, 3.1049, 0.3053), "IrO2": ("Ir", 4.5051, 3.1586, 0.3077)}


def rutile_110(metal, a, c, x):
    bulk = Structure(Lattice.tetragonal(a, c), [metal, metal, "O", "O", "O", "O"],
                     [[0, 0, 0], [0.5, 0.5, 0.5], [x, x, 0], [-x, -x, 0], [0.5 + x, 0.5 - x, 0.5],
                      [0.5 - x, 0.5 + x, 0.5]])
    generator = SlabGenerator(bulk, (1, 1, 0), min_slab_size=12, min_vacuum_size=15, center_slab=True,
                              lll_reduce=True)
    for slab in generator.get_slabs():
        slab = slab.get_orthogonal_c_slab()
        atoms = AseAtomsAdaptor.get_atoms(slab)
        top = atoms.get_chemical_symbols()[int(np.argmax(atoms.positions[:, 2]))]
        if slab.is_symmetric() and top == "O" and abs(slab.composition["O"] / slab.composition[metal] - 2) < 1e-6:
            break
    else:
        raise RuntimeError(f"no stoichiometric O-terminated {metal}O2(110) slab found")
    lengths = atoms.cell.lengths()
    atoms = atoms.repeat((2, 1, 1)) if lengths[0] < lengths[1] else atoms.repeat((1, 2, 1))
    z = atoms.positions[:, 2]
    atoms.set_constraint(FixAtoms(indices=[i for i in range(len(atoms)) if z[i] < z.min() + 0.45 * (z.max() - z.min())]))
    return atoms


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, (metal, a, c, x) in RUTILE.items():
        write(out / f"{name}_110.vasp", rutile_110(metal, a, c, x), format="vasp", direct=True, sort=True)
    pt = fcc111("Pt", size=(3, 3, 4), vacuum=8.0)
    pt.set_constraint(FixAtoms(indices=[atom.index for atom in pt if atom.tag >= 3]))
    write(out / "Pt_111.vasp", pt, format="vasp", direct=True, sort=True)
    print("wrote", ", ".join(sorted(p.name for p in out.glob("*.vasp"))))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
