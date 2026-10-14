"""Validate the SVJ Studio lesson in the current, pinned Python environment.

Use `pixi run --locked -e collaboration studio-check`. The public authoring API
runs notebook validation in an isolated child of this interpreter. Studio 0.3's
CLI runtime validator instead re-enters Python projects through uv, which does
not preserve this project's Pixi environment.

This executes the notebook's default cached lesson and checks its projections;
it does not verify browser layout or the optional live inference control.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/svj_var_pipeline.py"


async def check(args):
    # Studio is optional and belongs to the collaboration environment.
    from marimo_studio.authoring import open_workspace

    # The isolated worker inherits this environment, including notebook flags.
    os.environ["SBI4FINANCE_SVJ_DEMO"] = "0"
    view = open_workspace(NOTEBOOK).view("lesson")
    # Building permits paths that Studio 0.3's source editor cannot read.
    for source in ("index.html", "view.toml", "AGENTS.md"):
        await view.read(source)
    report = await view.validate(level=args.level, runtime_timeout=args.runtime_timeout)
    checks = [
        {"stage": stage, "name": item["name"], "status": item["status"],
         "message": item["message"]}
        for stage, evidence in report.evidence.items()
        for item in evidence.get("checks", [])
    ]
    return {
        "ok": report.ok,
        "notebook": str(report.notebook.relative_to(ROOT)),
        "view": report.view,
        "level": report.level,
        "python": sys.executable,
        "checks": checks,
        "source_reads": "passed",
        "issues": [issue.to_dict() for issue in report.issues],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("static", "runtime"), default="runtime")
    parser.add_argument("--runtime-timeout", type=float, default=60,
                        help="isolated notebook deadline in seconds (default: 60)")
    args = parser.parse_args()
    print(f"Validating the SVJ lesson ({args.level}) in {sys.executable}",
          file=sys.stderr, flush=True)
    try:
        report = asyncio.run(check(args))
    except Exception as error:
        print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}),
              file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
