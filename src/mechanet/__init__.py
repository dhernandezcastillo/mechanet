"""mechanet: mechanism networks for electrocatalysis on surface slabs.

The package turns one relaxed slab into a balanced network of reaction
intermediates, writes starting geometries for every intermediate, runs staged
VASP calculations on any SLURM cluster, and evaluates the results with the
computational hydrogen electrode (CHE).

Typical use goes through the command line (``mechanet --help``); the same
functions are importable for notebooks and scripts, for example::

    from mechanet.project import generate
    report = generate("my_project/project.yaml")     # = mechanet generate my_project
    print(report["complete_stoichiometric_cycles"])

    from mechanet.cli import main
    main(["--json", "thermo", "my_project/campaign"])  # any command, as on the command line
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
