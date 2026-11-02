"""Sphinx configuration of the mechanet documentation.

The pages are the Markdown files of this folder (MyST); the landing page is the README and the
changelog is CHANGELOG.md, both included from the repository root. The command reference
(commands.md) is written from mechanet's own argument parser at every build, so it always
matches the installed version.

Build locally:  pip install -r docs/requirements.txt -e .  then
                python -m sphinx -W -b html docs docs/_build/html
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/dhernandezcastillo/mechanet"
sys.path.insert(0, str(ROOT / "src"))


def _release():
    try:
        from mechanet import __version__

        return __version__
    except ImportError:  # the package's dependencies are missing: read pyproject.toml
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        return re.search(r'^version = "([^"]+)"', text, re.MULTILINE).group(1)


project = "mechanet"
author = "David Hernández Castillo"
copyright = "2026, David Hernández Castillo"  # noqa: A001 - the name Sphinx reads
release = version = _release()

extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "requirements.txt"]
myst_heading_anchors = 3
myst_enable_extensions = ["colon_fence"]

html_theme = "furo"
html_title = f"mechanet {release}"
html_theme_options = {
    "source_repository": f"{REPOSITORY}/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# ---------------------------------------------------------------------------
# Command reference, from the CLI itself
# ---------------------------------------------------------------------------

COMMANDS_INTRO = """# Command reference

Every command of `mechanet {release}`, as `mechanet COMMAND --help` prints it, in the order of
a study. Every command also takes `--json` (machine-readable output) and `--traceback`.
Nothing is submitted to SLURM without `--execute`.
"""


def write_command_reference(path):
    """docs/commands.md: one section per command with its --help text."""
    os.environ["COLUMNS"] = "88"  # the width argparse wraps the help text at (fits the page)
    try:
        from mechanet.cli import build_parser
        from mechanet.plugins import default_registry
    except ImportError as exc:  # mechanet's dependencies are not installed
        Path(path).write_text(
            COMMANDS_INTRO.format(release=release)
            + f"\nThe reference could not be generated here ({exc}); install mechanet to build it.\n",
            encoding="utf-8",
        )
        return
    parser = build_parser(default_registry())
    choices = next(a for a in parser._actions if getattr(a, "choices", None) and a.dest == "command").choices
    helps = {a.dest: a.help for a in next(a for a in parser._actions if a.dest == "command")._choices_actions}
    lines = [COMMANDS_INTRO.format(release=release)]
    for name, sub in choices.items():
        if name not in helps:  # run-state: what every job runs, never called by hand
            continue
        lines += [f"## {name}", "", f"{helps[name]}.", "", "```text", sub.format_help().rstrip(), "```", ""]
        actions = [a for a in sub._actions if getattr(a, "choices", None) and isinstance(a.choices, dict)]
        for action in actions:  # cluster import / list / show / use
            for child, subsub in action.choices.items():
                lines += [f"### {name} {child}", "", "```text", subsub.format_help().rstrip(), "```", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


write_command_reference(Path(__file__).with_name("commands.md"))

# ---------------------------------------------------------------------------
# Links of the included README: its links to docs/*.md on GitHub (they have to work on PyPI
# too) become links between the pages of this site.
# ---------------------------------------------------------------------------

DOC_LINK = re.compile(re.escape(REPOSITORY) + r"/blob/main/docs/([\w.-]+\.md)")


def _local_doc_links(app, relative_path, parent_docname, content):
    content[0] = DOC_LINK.sub(r"\1", content[0])


def setup(app):
    app.connect("include-read", _local_doc_links)
