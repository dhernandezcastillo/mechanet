"""Open design questions stay visible: docs/pending.md and the PENDING(id) comments agree.

A question that was left open on purpose is listed in docs/pending.md with what is known about
it, and marked in the code where a fix would go. Settling one means removing both; this test
fails when an entry has no comment left, or a comment has no entry.
"""

import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from mechanet.project import X86_MACHINES, generate

ROOT = Path(__file__).resolve().parents[1]
MARK = re.compile(r"PENDING\(([a-z0-9-]+)\)")
ENTRY = re.compile(r"^## `([a-z0-9-]+)`\s*$", re.MULTILINE)


class PendingTests(unittest.TestCase):
    def test_entries_and_comments_agree(self):
        marked = {}
        for path in sorted((ROOT / "src" / "mechanet").rglob("*.py")):
            for found in MARK.findall(path.read_text(encoding="utf-8")):
                marked.setdefault(found, []).append(path.name)
        listed = ENTRY.findall((ROOT / "docs" / "pending.md").read_text(encoding="utf-8"))
        self.assertEqual(len(listed), len(set(listed)), "an ID is listed twice in docs/pending.md")
        self.assertEqual(sorted(marked), sorted(listed), f"comments in the code: {marked}")
        # the three known today, each marked where a fix would go
        self.assertEqual(sorted(listed), ["ml-coverage-shortlist", "tie-break", "two-site-adsorbates"])
        self.assertIn("builder.py", marked["tie-break"])
        self.assertIn("adsorbates.py", marked["two-site-adsorbates"])
        self.assertEqual(sorted(marked["ml-coverage-shortlist"]), ["pourbaix.py", "prescreen.py"])

    def test_generate_says_when_the_processor_may_order_ties_differently(self):
        self.assertIn("amd64", X86_MACHINES)
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "p"
            shutil.copytree(ROOT / "examples" / "pt_111", project)
            data = yaml.safe_load((project / "project.yaml").read_text())
            data["sites"] = data["sites"][:1]
            (project / "project.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
            for machine, warned in (("arm64", True), ("AMD64", False), ("x86_64", False), ("", False)):
                with patch("mechanet.project.platform.machine", return_value=machine):
                    report = generate(project / "project.yaml", Path(tmp) / f"c_{machine or 'unknown'}")
                found = [w for w in report["warnings"] if "equally scored orientations" in w]
                self.assertEqual(bool(found), warned, (machine, report["warnings"]))
                if warned:
                    self.assertIn("(arm64)", found[0])
                    self.assertIn("docs/pending.md", found[0])


if __name__ == "__main__":
    unittest.main()
