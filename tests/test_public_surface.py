"""Nothing that a version of mechanet offered may disappear by accident.

``tests/data/public_surface_X.Y.Z.json`` lists what version X.Y.Z offered: every command
with its options and choices, the built-in reactions, adsorbates, families and literature
keys, and every public name each module defined (functions with their parameters, classes with
their public members and dataclass fields, constants). There is one list per version, from
0.1.4 on, and the test checks that all of each is still there: commands and options may be
added, never dropped; a function may gain parameters after the ones it had; a name may move to
another module as long as the old module still exports it. A name is listed under the module
that defined it in that version, so it is the list of 0.1.4 that keeps the names which have
since moved out of ``mechanet.analysis`` importable from there.

Removing something on purpose means removing it from every list that has it, in the same
commit, so the removal is visible in review.

A new version needs a list of its own, or the test fails. In the commit that sets the version,
``python tests/test_public_surface.py --write`` writes the list of this checkout under the
version it carries (it does not replace a list that exists) and prints the line to add to
``VERSIONS``. ``--write FILE`` writes the list of whatever mechanet is importable to FILE.
"""

import argparse
import ast
import dataclasses
import importlib
import inspect
import json
import pkgutil
import re
import sys
import unittest
from pathlib import Path

DATA = Path(__file__).with_name("data")
PREFIX = "public_surface_"
# Every version made since the first list, with the number of its commands.
VERSIONS = {"0.1.4": 27, "0.1.5": 28}
PLAIN = (str, int, float, bool, tuple, list, dict, set, frozenset, type(None))


def _public(name):
    return not name.startswith("_")


def _parameters(function):
    try:
        return list(inspect.signature(function).parameters)
    except (TypeError, ValueError):
        return None


def _describe(value):
    """What a public name is, as the snapshot stores it."""
    if inspect.isclass(value):
        entry = {"members": sorted(n for n in vars(value) if _public(n))}
        if dataclasses.is_dataclass(value):
            entry["fields"] = [f.name for f in dataclasses.fields(value)]
        return entry
    if inspect.isfunction(value):
        return _parameters(value)
    return None  # a constant: its presence is what counts


def surface():
    """The commands, registry content and public names of the importable mechanet."""
    import mechanet
    from mechanet.cli import build_parser
    from mechanet.plugins import default_registry

    registry = default_registry()
    parser = build_parser(registry)
    commands = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                commands[name] = {
                    ("/".join(a.option_strings) if a.option_strings else f"<{a.dest}>"): (
                        sorted(map(str, a.choices)) if a.choices else None
                    )
                    for a in sub._actions
                }
    modules = {}
    for info in pkgutil.walk_packages(mechanet.__path__, "mechanet."):
        if info.name.endswith("__main__"):
            continue
        module = importlib.import_module(info.name)
        assigned = set()
        for node in ast.parse(inspect.getsource(module)).body:
            targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
            for target in targets:
                assigned |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)} if target is not None else set()
        names = {}
        for name, value in vars(module).items():
            if not _public(name) or inspect.ismodule(value):
                continue
            if inspect.isclass(value) or inspect.isfunction(value):
                if value.__module__ == module.__name__:
                    names[name] = _describe(value)
            elif name in assigned and isinstance(value, PLAIN):
                names[name] = None
        modules[info.name] = names
    return {
        "commands": commands,
        "global_options": sorted("/".join(a.option_strings) for a in parser._actions if a.option_strings),
        "registry": {
            "reactions": sorted(registry.reactions),
            "adsorbates": sorted(registry.adsorbates),
            "families": sorted(registry.families),
            "commands": sorted(registry.commands),
            "sources": sorted(registry.sources),
        },
        "modules": modules,
    }


def missing(old, new_commands, new_globals, new_registry):
    """Everything in the snapshot ``old`` that the importable mechanet no longer offers."""
    gone = []
    for command, options in old["commands"].items():
        if command not in new_commands:
            gone.append(f"command {command}")
            continue
        for option, choices in options.items():
            if option not in new_commands[command]:
                gone.append(f"option {command} {option}")
            elif choices and set(choices) - set(new_commands[command][option] or ()):
                gone.append(
                    f"choices of {command} {option}: {sorted(set(choices) - set(new_commands[command][option] or ()))}"
                )
    gone += [f"global option {o}" for o in old["global_options"] if o not in new_globals]
    for kind, items in old["registry"].items():
        gone += [f"{kind[:-1]} {item}" for item in items if item not in new_registry.get(kind, ())]
    for module_name, names in old["modules"].items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            gone.append(f"module {module_name}")
            continue
        for name, entry in names.items():
            if not hasattr(module, name):
                gone.append(f"{module_name}.{name}")
                continue
            value = getattr(module, name)
            if isinstance(entry, list):  # a function: the parameters it had, in their order
                now = _parameters(value) or []
                if now[: len(entry)] != entry:
                    gone.append(f"{module_name}.{name}({', '.join(entry)}) is now ({', '.join(now)})")
            elif isinstance(entry, dict):  # a class
                lost = sorted(set(entry["members"]) - {n for n in dir(value) if _public(n)})
                if lost:
                    gone.append(f"{module_name}.{name}: members {lost}")
                if "fields" in entry:
                    fields = [f.name for f in dataclasses.fields(value)] if dataclasses.is_dataclass(value) else []
                    if fields[: len(entry["fields"])] != entry["fields"]:
                        gone.append(f"{module_name}.{name}: fields {entry['fields']} are now {fields}")
    return gone


def snapshots():
    """The lists in tests/data as {version: file}, oldest version first."""
    found = {path.stem[len(PREFIX) :]: path for path in DATA.glob(f"{PREFIX}*.json")}
    return dict(sorted(found.items(), key=lambda item: [int(n) for n in re.findall(r"\d+", item[0])]))


class PublicSurfaceTests(unittest.TestCase):
    def test_everything_a_version_offered_is_still_there(self):
        lists = snapshots()
        self.assertEqual([v for v in VERSIONS if v not in lists], [], "lists that are gone from tests/data")
        now = surface()
        for version, path in lists.items():
            with self.subTest(version=version):
                old = json.loads(path.read_text(encoding="utf-8"))
                gone = missing(old, now["commands"], now["global_options"], now["registry"])
                self.assertEqual(
                    gone, [], f"offered by {version} and no longer (see the module docstring):\n  " + "\n  ".join(gone)
                )
                # the list is a real one: the size of that release, and no version offered less than 0.1.4
                self.assertEqual(len(old["commands"]), VERSIONS.get(version, len(old["commands"])))
                self.assertGreaterEqual(len(old["commands"]), 27)
                self.assertGreater(sum(len(v) for v in old["commands"].values()), 200)
                self.assertGreater(sum(len(v) for v in old["modules"].values()), 500)

    def test_the_version_has_a_list_of_its_own(self):
        import mechanet

        self.assertIn(
            mechanet.__version__,
            VERSIONS,
            f"mechanet {mechanet.__version__} has no list of what it offers. In the commit that sets the version, "
            "run `python tests/test_public_surface.py --write` and add the line it prints to VERSIONS",
        )


if __name__ == "__main__":
    if sys.argv[1:2] == ["--write"] and len(sys.argv) in (2, 3):
        if len(sys.argv) == 3:
            target = Path(sys.argv[2])  # of whatever mechanet is importable
        else:  # of this checkout, under the version it carries; the list of a version is written once
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
            import mechanet

            target = DATA / f"{PREFIX}{mechanet.__version__}.json"
            if target.exists():
                sys.exit(f"{target} exists: the list of a version is written once, when the version is made")
        offered = surface()
        target.write_text(json.dumps(offered, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        print("wrote", target)
        if len(sys.argv) == 2 and mechanet.__version__ not in VERSIONS:
            print(f'add to VERSIONS in {Path(__file__).name}:  "{mechanet.__version__}": {len(offered["commands"])}')
    else:
        unittest.main()
