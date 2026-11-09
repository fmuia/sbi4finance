"""The tutorial stands alone: no code reaches outside tutorial/, and every asset matches MANIFEST.md."""

import ast
import hashlib
from pathlib import Path
import re
import sys
import unittest

TUTORIAL = Path(__file__).resolve().parents[1]
ASSETS = TUTORIAL / "assets"
CODE = sorted(p for folder in ("notebooks", "src", "tests", "curriculum") for p in (TUTORIAL / folder).rglob("*.py"))
RUNTIME = "sbi4f_tutorial"


def imported_modules(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.module


class IndependenceTests(unittest.TestCase):
    def test_code_imports_nothing_from_the_rest_of_the_repository(self):
        for path in CODE:
            with self.subTest(path=str(path.relative_to(TUTORIAL))):
                top = {name.split(".")[0] for name in imported_modules(path)}
                self.assertNotIn("sbi4finance", top)
                if path.is_relative_to(TUTORIAL / "src"):
                    # Numba is a lazy, optional acceleration for lab generation; cached
                    # viewing and the explicit NumPy simulator require no Numba install.
                    # Keras is lazy and limited to explicitly requested pricer
                    # training, just like BayesFlow's posterior training.
                    allowed = set(sys.stdlib_module_names) | {RUNTIME, "numpy", "scipy", "matplotlib", "bayesflow", "numba", "keras"}
                    self.assertLessEqual(top, allowed)

    def test_code_neither_edits_sys_path_nor_climbs_out_of_tutorial(self):
        for path in CODE:
            text = path.read_text()
            levels = len(path.relative_to(TUTORIAL).parts) - 1  # parents[levels] is tutorial/ itself
            with self.subTest(path=str(path.relative_to(TUTORIAL))):
                self.assertNotRegex(text, r"sys\.path\.(insert|append)")
                self.assertNotRegex(text, r"ROOT\.parent|\.parent\.parent")
                self.assertNotRegex(text, r"""["']\.\.(["'/\\]|$)""", "a parent-directory path component")
                for level in re.findall(r"parents\[(\d+)\]", text):
                    self.assertLessEqual(int(level), levels, f"parents[{level}] is above tutorial/")

    def test_every_asset_matches_the_manifest(self):
        rows = re.findall(r"^\| `([^`]+)` \| ([\d,]+) \| `([0-9a-f]{64})` \|", (TUTORIAL / "MANIFEST.md").read_text(), re.M)
        listed = {name: (int(size.replace(",", "")), digest) for name, size, digest in rows}
        present = {str(p.relative_to(ASSETS)) for p in ASSETS.rglob("*") if p.is_file()
                   and not p.is_relative_to(ASSETS / "local-checkpoints")}
        self.assertEqual(present, set(listed), "tutorial/assets/ and MANIFEST.md list different files")
        for name, (size, digest) in listed.items():
            with self.subTest(asset=name):
                data = (ASSETS / name).read_bytes()
                self.assertEqual((len(data), hashlib.sha256(data).hexdigest()), (size, digest))


if __name__ == "__main__":
    unittest.main()
