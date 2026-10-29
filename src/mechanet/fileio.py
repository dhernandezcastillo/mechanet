"""Small file helpers shared by every module: atomic writes, JSON/YAML, hashes and timestamps.

Writes are atomic (write to a unique temporary sibling, fsync, rename), so a reader on a
shared cluster file system never sees a partial file.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import socket
import uuid
from pathlib import Path

import yaml


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def unique_tmp(path) -> Path:
    """A temporary sibling name that cannot collide between hosts, processes or threads."""
    path = Path(path)
    return path.with_name(f".{path.name}.tmp.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex}")


def atomic_bytes(path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = unique_tmp(path)
    try:
        with tmp.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_text(path, text: str) -> None:
    atomic_bytes(path, text.encode("utf-8"))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, sort_keys=True) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=sort_keys, allow_nan=False, default=_json_default) + "\n")


def dump_json(path, value) -> None:
    """JSON in insertion order (reports meant to be read by people)."""
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False, default=_json_default) + "\n")


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_yaml(path):
    """A YAML file; a syntax error becomes a ValueError naming the file, line and column."""
    try:
        data = yaml.safe_load(Path(path).read_text())
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f"{path}, line {mark.line + 1}, column {mark.column + 1}" if mark is not None else str(path)
        problem = getattr(exc, "problem", None) or str(exc).splitlines()[0]
        hint = " (YAML indentation must use spaces, not tabs)" if "\t" in str(exc) or "tab" in str(problem) else ""
        raise ValueError(f"{where}: not valid YAML: {problem}{hint}") from None
    return {} if data is None else data


def write_yaml(path, value, header="") -> None:
    """YAML in insertion order; ``header`` (comment lines) goes on top."""
    text = yaml.safe_dump(value, sort_keys=False, default_flow_style=False, allow_unicode=True, width=100)
    atomic_text(path, header + text)


def file_hash(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def short_hash(value, size=12) -> str:
    """Compact deterministic identifier of a JSON-serializable value (node and edge IDs)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:size]
