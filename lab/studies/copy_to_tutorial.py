"""Refresh the tutorial's copies of lab code: the functions in tutorial/src/sbi4f_tutorial/ that the lab defines.

    python studies/copy_to_tutorial.py [--check] [--add MODULE.NAME ...]

Each runtime module copies from one lab file (COPIED_FROM). A definition the
lab file also has is the lab's, unchanged: when its code differs, this script
replaces it with the lab's text. Code may differ only in docstrings, comments
and import paths. --add copies a lab definition the module does not have yet,
for example a helper that a refreshed function now calls; the parity test
names such helpers. Definitions listed in WRITTEN_FOR_THE_TUTORIAL are the
tutorial's own and are left alone. With --check nothing changes: the script
lists the copies that differ and exits non-zero. lab/tests/test_tutorial_parity.py
runs the same comparison.
"""

import argparse
import ast
import copy
from pathlib import Path
import re
import sys

LAB = Path(__file__).resolve().parents[1]
RUNTIME = LAB.parent / "tutorial" / "src" / "sbi4f_tutorial"
# Runtime module -> the lab file it copies from, relative to lab/.
COPIED_FROM = {
    "bergomi": "src/sbi4finance/bergomi.py",
    "bs_hierarchy": "checks/bs_hierarchy.py",
    "gbm": "src/sbi4finance/gbm.py",
    "hkde": "src/sbi4finance/hkde.py",
    "inference": "src/sbi4finance/calibration/inference.py",
    "long_end": "checks/long_end/long_end.py",
    "plots": "src/sbi4finance/svj/plots.py",
    "pricing": "src/sbi4finance/calibration/pricing.py",
    "rough_bergomi": "src/sbi4finance/rough_bergomi.py",
    "surface": "src/sbi4finance/calibration/surface.py",
    "tasks": "src/sbi4finance/calibration/tasks.py",
    "teaching": "src/sbi4finance/calibration/teaching.py",
}
# Modules and definitions written for the tutorial, and why.
WRITTEN_FOR_THE_TUTORIAL = {
    "paths": "where tutorial/assets/ is",
    "why_bayes": "slot 1's toy example; the lab no longer uses it (archived at 574b940)",
    "tasks.rough_task": "reads the task the lab stored, instead of building it",
    "inference._checked_task": "checks a task's keys; the lab validated the shipped tasks (test_the_shipped_tasks_are_the_labs)",
    "long_end.ROUGH": "reads tutorial/assets/",
    "long_end.SPY": "reads tutorial/assets/",
    "long_end.TASK": "reads tutorial/assets/",
}


def definitions(path):
    """Top-level functions, classes and single-name assignments of a file: name -> (node, first line, last line)."""
    found = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        else:
            continue
        first = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
        found[name] = (node, first, node.end_lineno)
    return found


def code_of(node):
    """The node without docstrings, with relative and package imports written alike."""
    node = copy.deepcopy(node)
    for inner in ast.walk(node):
        if isinstance(inner, (ast.FunctionDef, ast.ClassDef)) and isinstance(inner.body[0], ast.Expr) \
                and isinstance(inner.body[0].value, ast.Constant) and isinstance(inner.body[0].value.value, str):
            inner.body = inner.body[1:] or [ast.Pass()]
        elif isinstance(inner, ast.ImportFrom):
            inner.module, inner.level = inner.module and inner.module.rsplit(".", 1)[-1], 0
    return ast.dump(node)


def lab_text(module, name):
    """The lab's source of a definition, its function-level imports made relative to the flat runtime package."""
    path = LAB / COPIED_FROM[module]
    _, first, last = definitions(path)[name]
    text = "\n".join(path.read_text().splitlines()[first - 1:last])
    text = re.sub(r"from \.\.(\w)", r"from .\1", text)
    return text.replace("from sbi4finance.calibration.", "from .").replace("from sbi4finance.", "from .")


def differences():
    """(module, name, reason) for every copy that differs from the lab, or that the lab no longer has."""
    found = []
    for path in sorted(RUNTIME.glob("*.py")):
        module = path.stem
        if module == "__init__" or module in WRITTEN_FOR_THE_TUTORIAL:
            continue
        if module not in COPIED_FROM:
            found.append((module, None, f"say in COPIED_FROM which lab file sbi4f_tutorial/{module}.py copies"))
            continue
        lab = definitions(LAB / COPIED_FROM[module])
        for name, (node, _, _) in definitions(path).items():
            if f"{module}.{name}" in WRITTEN_FOR_THE_TUTORIAL:
                continue
            if name not in lab:
                found.append((module, name, f"not in lab/{COPIED_FROM[module]}: delete it, or list it in "
                                            "WRITTEN_FOR_THE_TUTORIAL"))
            elif code_of(node) != code_of(lab[name][0]):
                found.append((module, name, f"differs from lab/{COPIED_FROM[module]}"))
    return found


def refresh(module, name):
    """Replace the runtime's copy of one definition with the lab's text."""
    path = RUNTIME / f"{module}.py"
    _, first, last = definitions(path)[name]
    lines = path.read_text().splitlines()
    lines[first - 1:last] = lab_text(module, name).splitlines()
    path.write_text("\n".join(lines) + "\n")


def add(module, name):
    """Append a lab definition the runtime module does not have yet."""
    path = RUNTIME / f"{module}.py"
    if name in definitions(path):
        raise SystemExit(f"sbi4f_tutorial/{module}.py already defines {name}")
    path.write_text(path.read_text().rstrip("\n") + "\n\n\n" + lab_text(module, name) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--add", nargs="+", default=[], metavar="MODULE.NAME")
    args = parser.parse_args()
    if not args.check:
        for item in args.add:
            add(*item.split(".", 1))
            print(f"added {item}")
        for module, name, reason in differences():
            if reason.startswith("differs"):
                refresh(module, name)
                print(f"copied {module}.{name} from lab/{COPIED_FROM[module]}")
    remaining = differences()
    for module, name, reason in remaining:
        print(f"{module}.{name}: {reason}" if name else reason)
    print("current" if not remaining else "stale")
    sys.exit(1 if remaining else 0)


if __name__ == "__main__":
    main()
