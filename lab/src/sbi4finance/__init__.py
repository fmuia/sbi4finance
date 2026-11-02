"""Lab workers reuse the numerical runtime from the same source checkout."""

from importlib.util import find_spec
from pathlib import Path


def _verify_runtime_checkout():
    """Reject an editable runtime left pointing at a different worktree.

    A deployed source checkout must include its sibling tutorial. Installed
    distributions without a source-tree layout use their declared dependency.
    """
    package = Path(__file__).resolve().parent
    lab = package.parent.parent
    if package.parent.name != "src" or not (lab / "pyproject.toml").is_file():
        return
    expected = lab.parent / "tutorial" / "src" / "sbi4f_tutorial" / "__init__.py"
    spec = find_spec("sbi4f_tutorial")
    actual = None if spec is None or spec.origin is None else Path(spec.origin).resolve()
    if not expected.is_file() or actual != expected.resolve():
        raise ImportError(
            "The lab requires the tutorial runtime from the same complete checkout. "
            "Run 'pixi install --locked -e core' from this checkout's lab/ directory; "
            "when deploying, include sibling tutorial/. "
            f"Expected {expected}; found {actual}."
        )


_verify_runtime_checkout()
