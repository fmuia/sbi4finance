"""Check notebook 05 in an isolated copy using the existing pinned Python.

Run with the project's existing Pixi Python and a new --out directory. This
checks offline execution, not dependency installation or scientific acceptance.
Only the four historical root-43021 illustrations execute; saved root-43022
validation results are read without training or evaluating its reserved test.
"""

import argparse
import base64
import hashlib
import html
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


NOTEBOOK = "notebooks/05_heston_joint_inference.py"
CHECKPOINT = "checkpoints/heston-theta-state-provisional"
GUARD = '''import ipaddress
import socket

def local_host(host):
    if host in (None, "localhost", b"localhost"):
        return True
    try:
        return ipaddress.ip_address(host.decode() if isinstance(host, bytes) else host).is_loopback
    except ValueError:
        return False

def guarded_connect(original):
    def call(self, address, *args):
        if self.family != socket.AF_UNIX and not local_host(address[0]):
            raise RuntimeError("External Python socket connections disabled")
        return original(self, address, *args)
    return call

socket.socket.connect = guarded_connect(socket.socket.connect)
socket.socket.connect_ex = guarded_connect(socket.socket.connect_ex)

def guarded_dns(original):
    def call(host, *args, **kwargs):
        if not local_host(host):
            raise RuntimeError("External Python DNS lookups disabled")
        return original(host, *args, **kwargs)
    return call

for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
    setattr(socket, name, guarded_dns(getattr(socket, name)))
original_getnameinfo = socket.getnameinfo
def getnameinfo(address, *args):
    if not local_host(address[0]):
        raise RuntimeError("External Python reverse DNS lookups disabled")
    return original_getnameinfo(address, *args)
socket.getnameinfo = getnameinfo
'''


def sha(content):
    return hashlib.sha256(content).hexdigest()


def inputs(root):
    """Copy historical evidence, never raw banks or this review's own reports."""
    paths = [root / NOTEBOOK, root / "pyproject.toml", root / "pixi.lock"]
    paths.extend(p for directory in ("src", "configs", CHECKPOINT)
                 for p in (root / directory).rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts)
    paths.extend(p for p in (root / "docs").glob("heston-*.md")
                 if p.name != "heston-validation-decision.md")
    for entry in (root / "docs/results").glob("heston-*"):
        if entry.name != "heston-validation-decision":
            paths.extend([entry] if entry.is_file() else
                         (p for p in entry.rglob("*") if p.is_file()))
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.repo.resolve(), args.out.resolve()
    if out.exists():
        parser.error("--out must be new; existing outputs are never overwritten")
    source_paths = inputs(root)
    out.mkdir(parents=True)
    isolated = out / "isolated"
    isolated.mkdir()
    copied = {}
    for path in source_paths:
        relative = path.relative_to(root)
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Input escapes the source repository: {relative}")
        content = path.read_bytes()
        target = isolated / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        copied[str(relative)] = sha(content)
    manifest = json.loads((isolated / "docs/results/heston-inference/bank-manifest.json").read_text())
    if manifest["configuration"]["root_seed"] != 43021:
        raise ValueError("The executable illustration must retain historical root seed 43021")
    guard_dir = isolated / "offline-guard"
    guard_dir.mkdir()
    (guard_dir / "sitecustomize.py").write_text(GUARD)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(guard_dir), str(isolated / "src"))),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", KERAS_BACKEND="jax",
               MPLBACKEND="Agg", MPLCONFIGDIR=str(isolated / "mpl-cache"),
               NUMBA_CACHE_DIR=str(isolated / "numba-cache"))
    timings = {}

    def run(label, arguments):
        started = time.perf_counter()
        result = subprocess.run([sys.executable, *arguments], cwd=isolated, env=env,
                                text=True, capture_output=True, timeout=600)
        timings[label] = time.perf_counter() - started
        (out / f"{label}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {out / (label + '.log')}")

    run("guard-preflight", ["-c", """import pathlib, socket, sitecustomize, sbi4finance
assert pathlib.Path(sbi4finance.__file__).resolve().is_relative_to(pathlib.Path.cwd() / 'src')
for method in ('connect', 'connect_ex'):
    with socket.socket() as connection:
        try: getattr(connection, method)(('192.0.2.1', 9))
        except RuntimeError: pass
        else: raise AssertionError(method + ' guard inactive')
for method, arguments in (
    ('getaddrinfo', ('example.com', 443)), ('gethostbyname', ('example.com',)),
    ('gethostbyname_ex', ('example.com',)), ('gethostbyaddr', ('192.0.2.1',)),
    ('getnameinfo', (('192.0.2.1', 443), 0))):
    try: getattr(socket, method)(*arguments)
    except RuntimeError: pass
    else: raise AssertionError(method + ' guard inactive')
assert socket.getaddrinfo('127.0.0.1', 0)
print('Isolated src imports and external Python socket/DNS guards verified; local IPC allowed.')
"""])
    run("marimo-check", ["-m", "marimo", "check", "--strict", NOTEBOOK])
    run("export", ["-m", "marimo", "export", "html", NOTEBOOK, "-o", "review.html", "--force"])
    exported = (isolated / "review.html").read_bytes()
    match = re.search(r'"session":\s*', exported.decode())
    if match is None:
        raise ValueError("Export has no inspectable marimo session")
    session = json.JSONDecoder().raw_decode(exported.decode()[match.end():])[0]
    outputs = [output for cell in session["cells"] for output in cell["outputs"]]
    errors = [output for output in outputs if output["type"] == "error"]
    if errors:
        raise ValueError(f"Export contains marimo cell errors: {errors}")
    encoded = json.dumps(outputs)
    visible = re.sub(r"<[^>]+>", " ", html.unescape(encoded)).lower()
    expected = ["provisional", "root 43021", "root 43022", "reserved test"]
    if any(term not in visible for term in expected):
        raise ValueError("Export is missing historical/provisional/test-status labels")
    figures = out / "figures"
    figures.mkdir()
    displayed = []
    for cell_index, cell in enumerate(session["cells"]):
        encoded_cell = json.dumps(cell["outputs"])
        for encoded_png in re.findall(r"data:image/png;base64,([A-Za-z0-9+/=]+)", encoded_cell):
            content = base64.b64decode(encoded_png, validate=True)
            name = f"figure-{len(displayed) + 1:02d}-cell-{cell_index + 1:02d}.png"
            (figures / name).write_bytes(content)
            displayed.append({"file": f"figures/{name}", "cell_index": cell_index, "sha256": sha(content)})
    if not displayed:
        raise ValueError("Export did not contain any displayed PNG figures")
    record = {
        "scope": "Offline notebook execution and historical regression illustrations; no scientific acceptance",
        "notebook": NOTEBOOK, "strict_marimo_check": "passed", "error_outputs": 0,
        "output_cells": len(session["cells"]), "visible_status_checks": expected,
        "displayed_figures": displayed,
        "visual_inspection": {"status": "pending", "note": "Inspect every extracted figure and the HTML separately."},
        "execution_modes": {
            "historical_root_seed": 43021, "regenerated_cases": ["test:0", "test:1", "test:2", "test:3"],
            "neural_draws_per_case": 2048, "default_reference_case": "test:0",
            "checkpoint": CHECKPOINT, "root_43022_validation": "saved results only",
            "reserved_root_43022_test_evaluated": False, "training_executed": False,
            "raw_banks_or_market_data_copied": False, "new_holdout_evaluation": False,
        },
        "network_guard_scope": "Python socket connect/connect_ex and DNS; loopback/Unix IPC permitted; not an OS sandbox",
        "network_guard_sha256": sha(GUARD.encode()), "source_and_artifact_sha256": copied,
        "verifier_sha256": sha(Path(__file__).read_bytes()), "python_executable": sys.executable,
        "python_version": platform.python_version(), "platform": platform.platform(),
        "package_versions": {name: version(name) for name in
                             ("marimo", "numpy", "pandas", "matplotlib", "bayesflow", "keras", "jax", "numba")},
        "environment_scope": "Existing pinned environment, fresh isolated input copy/process; no dependency installation certification",
        "isolated_copy": str(isolated), "html_path": str(isolated / "review.html"),
        "html_sha256": sha(exported), "elapsed_seconds": timings,
    }
    (out / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"verification": str(out / "verification.json"), "html": record["html_path"],
                      "figures": str(figures)}, indent=2))


if __name__ == "__main__":
    main()
