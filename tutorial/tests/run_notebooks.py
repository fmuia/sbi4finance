"""Run every notebook headless, as marimo export does, and fail if any cell raises.

    python tests/run_notebooks.py [NOTEBOOK.py ...]     (all of notebooks/ by default; a few minutes)

Each notebook runs in a fresh process; its rendered HTML goes to a temporary directory. The check reads the
session that marimo embeds in the export and counts the cells whose output is an error.
"""

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def errors_in(html):
    match = re.search(r'"session":\s*', html)
    session = json.JSONDecoder().raw_decode(html[match.end():])[0]
    outputs = [o for cell in session["cells"] for o in cell.get("outputs", [])]
    return len(session["cells"]), [o for o in outputs if o.get("type") == "error"]


def main(paths):
    failed = []
    with tempfile.TemporaryDirectory() as out:
        for notebook in paths:
            target = Path(out) / (notebook.stem + ".html")
            started = time.perf_counter()
            run = subprocess.run([sys.executable, "-m", "marimo", "export", "html", str(notebook), "-o", str(target)],
                                 capture_output=True, text=True, cwd=notebook.parent)
            seconds = time.perf_counter() - started
            if run.returncode or not target.exists():
                failed.append(notebook.name)
                print(f"FAIL {notebook.name}: export exited {run.returncode}\n{run.stderr[-2000:]}")
                continue
            cells, errors = errors_in(target.read_text())
            if errors:
                failed.append(notebook.name)
            print(f"{'FAIL' if errors else 'ok  '} {notebook.name}: {cells} cells, {len(errors)} errors, {seconds:.0f} s")
            for error in errors:
                print("     ", json.dumps(error)[:500])
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main([Path(p).resolve() for p in sys.argv[1:]] or sorted(NOTEBOOKS.glob("*.py")))
