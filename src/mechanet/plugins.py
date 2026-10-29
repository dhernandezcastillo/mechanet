"""Plugin registry.

mechanet is extended through Python entry-point groups; the built-in OER/CER chemistry is
registered the same way an external package would register its own:

=========================  ==========================================================
``mechanet.reactions``     a :class:`mechanet.reactions.Reaction` subclass
``mechanet.adsorbates``    ``register(registry)`` adding :class:`Adsorbate` objects
``mechanet.families``      ``register(registry)`` adding :class:`Family` objects
``mechanet.commands``      ``register_command(subparsers, registry)`` adding CLI subcommands
=========================  ==========================================================

A project can also load local plugin files (``plugins: [./my_families.py]`` in
project.yaml); each file defines ``register(registry)``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Callable

from .adsorbates import Adsorbate

GROUPS = ("mechanet.reactions", "mechanet.adsorbates", "mechanet.families", "mechanet.commands")
# Keys of the scope section of project.yaml that belong to mechanet itself (not to a family).
SCOPE_BASE_KEYS = ("families", "pair_max_distance_A")


@dataclass
class Family:
    """A mechanism family: builds nodes, edges and cycles from the sites of a slab.

    ``build(ctx)`` receives the :class:`mechanet.builder.Builder` of the campaign (sites,
    native states, ``node``, ``step``, ``cycle``, ``oer_cycle``, ``nearby_pairs``, ...).
    ``reaction`` names the reaction whose cycles the family writes (None for families that
    only add nodes, such as the Pourbaix coverage family). ``core`` marks families in the
    default scope. A family configured from project.yaml declares its block with
    ``scope_key`` (it reads ``ctx.scope[scope_key]``) and, optionally, the keys that block
    accepts with ``scope_fields``; ``mechanet`` then accepts the block and checks its keys.
    """

    name: str
    build: Callable
    reaction: str | None
    description: str
    sources: list = field(default_factory=list)
    core: bool = False
    order: int = 100
    origin: str = "builtin"
    scope_key: str | None = None
    scope_fields: tuple = ()


class Registry:
    def __init__(self):
        self.reactions = {}
        self.adsorbates = {}
        self.families = {}
        self.commands = {}
        self.sources = {}  # literature key -> record
        self.loaded = []  # (group, name, origin)
        self.errors = []

    # ------------------------------------------------------------------ adding
    def add_reaction(self, cls, origin="builtin"):
        from .reactions import validate_reaction

        name = cls.name
        if name in self.reactions and self.reactions[name] is not cls:
            raise ValueError(f"Reaction {name!r} is registered twice")
        validate_reaction(cls())
        self.reactions[name] = cls
        self.loaded.append(("reaction", name, origin))

    def add_species(self, species, origin="builtin"):
        """A new reservoir species (mechanet.reactions.Species), e.g. H2 for HER or N2/NH3 for NRR."""
        from .reactions import register_species

        register_species(species)
        self.loaded.append(("species", species.name, origin))

    def add_adsorbate(self, adsorbate: Adsorbate, origin="builtin"):
        if adsorbate.name in self.adsorbates and self.adsorbates[adsorbate.name] != adsorbate:
            raise ValueError(f"Adsorbate {adsorbate.name!r} is registered twice with different data")
        clash = next(
            (
                a.name
                for a in self.adsorbates.values()
                if a.name != adsorbate.name and a.signature == adsorbate.signature
            ),
            None,
        )
        if clash:
            raise ValueError(
                f"Adsorbate {adsorbate.name!r} has the same signature as {clash!r}; relaxed structures "
                "could not be told apart"
            )
        self.adsorbates[adsorbate.name] = adsorbate
        self.loaded.append(("adsorbate", adsorbate.name, origin))

    def add_family(self, family: Family, origin=None):
        if origin:
            family.origin = origin
        if family.name in self.families:
            raise ValueError(f"Mechanism family {family.name!r} is registered twice")
        if family.scope_key is not None:
            if not isinstance(family.scope_key, str) or not family.scope_key:
                raise ValueError(f"Mechanism family {family.name!r}: scope_key must be a non-empty string")
            if family.scope_key in SCOPE_BASE_KEYS:
                raise ValueError(
                    f"Mechanism family {family.name!r}: scope_key {family.scope_key!r} is a reserved scope key"
                )
            owner = self.scope_blocks().get(family.scope_key)
            if owner is not None:
                raise ValueError(
                    f"Mechanism family {family.name!r}: scope block {family.scope_key!r} already belongs to "
                    f"family {owner.name!r}"
                )
        fields = family.scope_fields if family.scope_fields is not None else ()
        if not isinstance(fields, (tuple, list)) or not all(isinstance(f, str) and f for f in fields):
            raise ValueError(
                f"Mechanism family {family.name!r}: scope_fields must be a tuple of field names, "
                f"e.g. ('sites',), not {fields!r}"
            )
        family.scope_fields = tuple(fields)
        if family.scope_fields and not family.scope_key:
            raise ValueError(f"Mechanism family {family.name!r}: scope_fields need a scope_key (the block's name)")
        self.families[family.name] = family
        self.loaded.append(("family", family.name, family.origin))

    def add_command(self, name, register, origin="builtin"):
        self.commands[name] = register
        self.loaded.append(("command", name, origin))

    def add_sources(self, records: dict):
        self.sources.update(records)

    # ------------------------------------------------------------------ lookup
    def reaction(self, name):
        try:
            return self.reactions[name]()
        except KeyError:
            raise ValueError(f"Unknown reaction {name!r}; available: {sorted(self.reactions)}") from None

    def adsorbate(self, name):
        try:
            return self.adsorbates[name]
        except KeyError:
            raise ValueError(f"Unknown adsorbate {name!r}; available: {sorted(self.adsorbates)}") from None

    def family(self, name):
        try:
            return self.families[name]
        except KeyError:
            raise ValueError(f"Unknown mechanism family {name!r}; available: {sorted(self.families)}") from None

    def ordered_families(self, names):
        unknown = sorted(set(names) - set(self.families))
        if unknown:
            raise ValueError(f"Unknown mechanism families {unknown}; available: {sorted(self.families)}")
        return sorted((self.families[n] for n in set(names)), key=lambda f: (f.order, f.name))

    def core_families(self, reaction):
        return [
            f.name
            for f in sorted(self.families.values(), key=lambda f: (f.order, f.name))
            if f.core and f.reaction == reaction
        ]

    def scope_blocks(self):
        """Scope blocks of project.yaml that configure a family: block name -> family."""
        return {f.scope_key: f for f in self.families.values() if f.scope_key}

    # ------------------------------------------------------------------ loading
    def load_file(self, path):
        """Import a local plugin file and call its ``register(registry)``.

        A file is executed once per process for each content: loading it again (or a copy of
        it, such as the one ``generate`` puts in the campaign) reuses the module, so the
        species, reactions and functions it defines are the same objects in every registry.
        """
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Plugin file not found: {path}")
        if path.suffix != ".py":
            raise ValueError(f"Plugin file {path} is not a Python file (a plugin is a .py file defining register)")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        module = _PLUGIN_MODULES.get(digest)
        if module is None:
            module = self._execute(path, f"mechanet_local_plugin_{digest[:16]}")
            _PLUGIN_MODULES[digest] = module
        if not hasattr(module, "register"):
            raise ValueError(f"Plugin file {path} defines no register(registry) function")
        before = len(self.loaded)
        try:
            module.register(self)
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Plugin file {path}: register() failed: {type(exc).__name__}: {exc}") from exc
        for i in range(before, len(self.loaded)):
            kind, item, _ = self.loaded[i]
            self.loaded[i] = (kind, item, str(path))
            if kind == "family":  # a copy: the module, and a family it defines, may serve other registries
                self.families[item] = dataclasses.replace(self.families[item], origin=str(path))
        return module

    @staticmethod
    def _execute(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Plugin file {path} is not a Python file (a plugin is a .py file defining register)")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True  # no __pycache__ folder next to the project's plugin files
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - any error in a plugin file is reported with its path
            sys.modules.pop(name, None)
            raise ValueError(f"Plugin file {path} failed to load: {type(exc).__name__}: {exc}") from exc
        finally:
            sys.dont_write_bytecode = write_bytecode
        return module


def _register_builtins(registry):
    from . import reactions, adsorbates
    from .families import oer, cer, coverage

    for cls in reactions.BUILTIN_REACTIONS.values():
        registry.add_reaction(cls)
    adsorbates.register(registry)
    oer.register(registry)
    cer.register(registry)
    coverage.register(registry)
    from . import pourbaix, pathways

    pathways.register_family(registry)
    registry.add_command("pourbaix", pourbaix.register_command)
    registry.add_command("pathways", pathways.register_command)
    from .literature import LITERATURE

    registry.add_sources(LITERATURE)


def _external_entry_points():
    """Entry points of other distributions (the built-ins are registered explicitly)."""
    found = []
    for group in GROUPS:
        try:
            eps = entry_points(group=group)
        except TypeError:  # pragma: no cover - Python < 3.10 API
            eps = entry_points().get(group, [])
        for ep in eps:
            if ep.value.split(":")[0].split(".")[0] == "mechanet":
                continue
            found.append((group, ep))
    return found


_DEFAULT = None
_PLUGIN_MODULES = {}  # sha256 of a plugin file -> its module: each content is executed once per process


def campaign_plugin_files(campaign, config=None):
    """The plugin files of a campaign, as absolute paths.

    ``generate`` copies a project's plugin files into ``<campaign>/inputs/plugins`` and stores
    them relative to the campaign, so a campaign copied to a cluster keeps its plugins. An
    absolute path (a campaign written before) is kept as it is.
    """
    import json

    campaign = Path(campaign)
    if config is None:
        path = campaign / "config.json"
        config = json.loads(path.read_text()) if path.is_file() else {}
    files = []
    for entry in config.get("plugins") or ():
        path = Path(entry).expanduser()
        files.append(str(path if path.is_absolute() else (campaign / path).resolve()))
    return files


def campaign_registry(campaign, registry=None, config=None):
    """The registry a campaign needs: the built-ins plus the plugin files of its config.json."""
    plugins = campaign_plugin_files(campaign, config)
    if plugins:
        return default_registry(plugins)
    return registry or default_registry()


def default_registry(plugin_files=(), fresh=False):
    """Built-ins, installed third-party plugins and the given local plugin files."""
    global _DEFAULT
    if _DEFAULT is not None and not plugin_files and not fresh:
        return _DEFAULT
    registry = Registry()
    _register_builtins(registry)
    for group, ep in _external_entry_points():
        try:
            obj = ep.load()
            origin = f"{ep.dist.name if getattr(ep, 'dist', None) else 'entry point'}:{ep.name}"
            if group == "mechanet.reactions":
                registry.add_reaction(obj, origin)
            elif group == "mechanet.commands":
                registry.add_command(ep.name, obj, origin)
            else:
                before = len(registry.loaded)
                obj(registry)
                for i in range(before, len(registry.loaded)):
                    kind, item, _ = registry.loaded[i]
                    registry.loaded[i] = (kind, item, origin)
        except Exception as exc:  # noqa: BLE001 - a broken third-party plugin must not break the core
            registry.errors.append(f"{group}:{ep.name}: {type(exc).__name__}: {exc}")
    for path in plugin_files:
        registry.load_file(path)
    if not plugin_files:
        _DEFAULT = registry
    return registry
