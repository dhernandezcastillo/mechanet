"""Nothing that mechanet 0.1.4 offered may disappear by accident.

``tests/data/public_surface_0.1.4.json`` lists what the 0.1.4 release offered: every command
with its options and choices, the built-in reactions, adsorbates, families and literature
keys, and every public name each module defined (functions with their parameters, classes with
their public members and dataclass fields, constants). The test checks that all of it is still
there: commands and options may be added, never dropped; a function may gain parameters after
the ones it had; a name may move to another module as long as the old module still exports it.

Removing something on purpose means removing it from the snapshot too, in the same commit, so
the removal is visible in review. ``python tests/test_public_surface.py --write FILE`` writes
the snapshot of whatever mechanet is importable (used once, with the 0.1.4 sources).
"""

import argparse
import ast
import dataclasses
import importlib
import inspect
import json
import pkgutil
import sys
import unittest
from pathlib import Path

SNAPSHOT = Path(__file__).with_name("data") / "public_surface_0.1.4.json"
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


class PublicSurfaceTests(unittest.TestCase):
    def test_everything_of_0_1_4_is_still_there(self):
        old = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        now = surface()
        gone = missing(old, now["commands"], now["global_options"], now["registry"])
        self.assertEqual(gone, [], "no longer offered (see the module docstring):\n  " + "\n  ".join(gone))
        # the snapshot is a real one: the size of the 0.1.4 release
        self.assertEqual(len(old["commands"]), 27)
        self.assertGreater(sum(len(v) for v in old["commands"].values()), 200)
        self.assertGreater(sum(len(v) for v in old["modules"].values()), 500)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--write":
        Path(sys.argv[2]).write_text(json.dumps(surface(), sort_keys=True, separators=(",", ":")), encoding="utf-8")
        print("wrote", sys.argv[2])
    else:
        unittest.main()
