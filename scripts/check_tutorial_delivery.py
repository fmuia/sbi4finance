"""Export the current 01/10/11 lessons in isolation as a delivery preflight.

Use the locked Pixi Python. This is not the integrated participant route, a
clean-install check, scientific acceptance, or formal I4 verification. The
current notebook10 runs its existing fixed provisional live example once.
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


NOTEBOOKS = {
    "01_gbm_foundations": (2, ("exercises and next steps", "separate")),
    "10_hkde_corrected_workflow": (6, ("historical scientific acceptance: not passed",
                                      "provisional live demonstration", "review trail and exercises")),
    "11_hkde_calibration_audit": (3, ("historical failed flag", "review trail, reproduction and exercises")),
}
GUARD = '''import ipaddress, socket, sys
from importlib.machinery import PathFinder
from pathlib import Path

def local(host):
    if host in (None, "localhost", b"localhost"):
        return True
    try:
        return ipaddress.ip_address(host.decode() if isinstance(host, bytes) else host).is_loopback
    except ValueError:
        return False

def audit(event, args):
    if event in {"socket.connect", "socket.sendto"}:
        sock, address = args[0], args[-1]
        if sock.family != socket.AF_UNIX and not local(address[0]):
            raise RuntimeError("External Python socket access disabled")
sys.addaudithook(audit)

def dns(original):
    def guarded(host, *args, **kwargs):
        if not local(host):
            raise RuntimeError("External Python DNS access disabled")
        return original(host, *args, **kwargs)
    return guarded
for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
    setattr(socket, name, dns(getattr(socket, name)))
original_getnameinfo = socket.getnameinfo
def getnameinfo(address, *args):
    if not local(address[0]):
        raise RuntimeError("External Python reverse DNS access disabled")
    return original_getnameinfo(address, *args)
socket.getnameinfo = getnameinfo

class IsolatedSource:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "sbi4finance":
            spec = PathFinder.find_spec(fullname, path)
            if spec is None or not Path(spec.origin).resolve().is_relative_to(Path.cwd() / "src"):
                raise ImportError("Scientific source escaped the isolated copy")
            return spec
sys.meta_path.insert(0, IsolatedSource())
'''


def sha(content):
    return hashlib.sha256(content).hexdigest()


def inputs(root):
    names = {"pyproject.toml", "pixi.lock", "docs/gbm-cluster-smoke.md",
             "configs/tasks/gbm_drift.json", "configs/tasks/gbm_bank.json",
             "scripts/plot_hkde_correction.py"}
    names.update(f"notebooks/{name}.py" for name in NOTEBOOKS)
    names.update(str(path.relative_to(root)) for path in (root / "src").rglob("*.py"))
    directory = "docs/results/hkde-correction"
    for split in ("validation", "test"):
        name = f"{directory}/{split}.json"
        report = json.loads((root / name).read_text())
        names.update((name, f"{directory}/{split}-checks.npz"))
        names.update(report["input_sha256"])
        names.update(str(Path("scripts" if (root / "scripts" / name).exists()
                                   else "src/sbi4finance") / name) for name in report["source_sha256"])
    names.update(f"{directory}/{name}" for name in ("test-reference-pit.npz", "test-independent-audit.json"))
    names.update(f"docs/results/hkde-gate-audit/{name}" for name in
                 ("report.json", "checks.npz", "oracle-rank-control.png",
                  "paired-rank-errors.png", "conditional-cdf-errors.png"))
    return sorted(names)


def inspect_export(out, name):
    expected_images, expected_text = NOTEBOOKS[name]
    html_path = out / f"{name}.html"
    match = re.search(r'"session":\s*', exported := html_path.read_text())
    if match is None:
        raise ValueError(f"{name}: no inspectable marimo session")
    session = json.JSONDecoder().raw_decode(exported[match.end():])[0]
    outputs = [item for cell in session["cells"] for item in cell["outputs"]]
    if errors := [item for item in outputs if item["type"] == "error"]:
        raise ValueError(f"{name}: marimo cell errors: {errors}")
    markup = "\n".join(str(value) for item in outputs for value in item.get("data", {}).values())
    # Callouts store their visible content inside JSON-encoded HTML attributes.
    callouts = [json.loads(html.unescape(value)) for value in re.findall(r"data-html='([^']*)'", markup)]
    visible = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(markup + "\n" + "\n".join(callouts))).lower())
    if any(term not in visible for term in expected_text):
        raise ValueError(f"{name}: missing exercise/provisional status text")
    stage_seconds, live_checks = {}, []
    for value in re.findall(r"<marimo-table\b[^>]*data-data='([^']*)'", markup):
        rows = json.loads(html.unescape(value))
        rows = json.loads(rows) if isinstance(rows, str) else rows
        if isinstance(rows, list):
            stage_seconds.update({row["Stage"]: row["Current-machine seconds"] for row in rows
                                  if "Stage" in row and "Current-machine seconds" in row})
            live_checks.extend(row for row in rows if "Live value" in row and "Passed" in row)
    figures = []
    for cell_index, cell in enumerate(session["cells"]):
        for png in re.findall(r"data:image/png;base64,([A-Za-z0-9+/=]+)", json.dumps(cell["outputs"])):
            content = base64.b64decode(png, validate=True)
            filename = f"{name}-figure-{len(figures) + 1:02d}.png"
            (out / filename).write_bytes(content)
            figures.append({"file": filename, "cell_index": cell_index, "sha256": sha(content)})
    if len(figures) != expected_images:
        raise ValueError(f"{name}: expected {expected_images} displayed figures, got {len(figures)}")
    external_assets = sorted(set(re.findall(r"<(?:script|link)\b[^>]*(?:src|href)=[\"'](https?://[^\"']+)", exported)))
    return {"notebook": f"notebooks/{name}.py", "output_cells": len(session["cells"]), "error_outputs": 0,
            "html": html_path.name, "html_sha256": sha(html_path.read_bytes()), "figures": figures,
            "visible_status_checks": expected_text, "stage_seconds": stage_seconds,
            "live_case_checks": live_checks,
            "browser_offline_verified": False, "html_external_assets": external_assets,
            "console": [{"cell_index": index, "messages": cell["console"]}
                        for index, cell in enumerate(session["cells"]) if cell["console"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("preflight",), required=True)
    parser.add_argument("--out", type=Path, required=True, help="New directory for logs, HTML, figures and report")
    args = parser.parse_args()
    root, out = Path(__file__).resolve().parents[1], args.out.resolve()
    if out.exists():
        parser.error("--out must be new; existing results are never overwritten")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).splitlines()
    diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root)
    isolated = out / "isolated"
    isolated.mkdir(parents=True)
    (out / "working-tree.patch").write_bytes(diff)
    (out / "verifier.py").write_bytes(Path(__file__).read_bytes())
    copied = {}
    for name in inputs(root):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or name.startswith(("data/", "tests/")):
            raise ValueError(f"Not an allowed isolated input: {name}")
        content = path.read_bytes()
        target = isolated / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        copied[name] = sha(content)
    guard = isolated / "offline-guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(GUARD)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(guard), str(isolated / "src"))),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", KERAS_BACKEND="jax",
               JAX_PLATFORMS="cpu", MPLBACKEND="Agg", MPLCONFIGDIR=str(isolated / "mpl-cache"),
               NUMBA_CACHE_DIR=str(isolated / "numba-cache"))

    def run(label, arguments):
        started = time.perf_counter()
        result = subprocess.run([sys.executable, *arguments], cwd=isolated, env=env,
                                text=True, capture_output=True, timeout=600)
        (out / f"{label}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {out / (label + '.log')}")
        return time.perf_counter() - started

    run("guard-preflight", ["-c", """import socket, sitecustomize, sbi4finance
for method in ('connect', 'connect_ex'):
    with socket.socket() as sock:
        try: getattr(sock, method)(('192.0.2.1', 9))
        except RuntimeError: pass
        else: raise AssertionError(method + ' guard inactive')
with socket.socket(type=socket.SOCK_DGRAM) as sock:
    try: sock.sendto(b'probe', ('192.0.2.1', 9))
    except RuntimeError: pass
    else: raise AssertionError('UDP guard inactive')
try: socket.getaddrinfo('example.com', 443)
except RuntimeError: pass
else: raise AssertionError('DNS guard inactive')
with socket.socket() as server:
    server.bind(('127.0.0.1', 0)); server.listen()
    with socket.create_connection(server.getsockname()) as client:
        connection, _ = server.accept(); connection.close()
assert socket.getaddrinfo('127.0.0.1', 0)
print('External TCP/UDP/DNS blocked; loopback connection and isolated source imports passed.')
"""])
    runs = []
    for name in NOTEBOOKS:
        notebook = f"notebooks/{name}.py"
        check_seconds = run(f"{name}-check", ["-m", "marimo", "check", "--strict", notebook])
        html_path = out / f"{name}.html"
        export_seconds = run(f"{name}-export", ["-m", "marimo", "export", "html", notebook, "-o", str(html_path)])
        runs.append({**inspect_export(out, name), "strict_marimo_check": "passed",
                     "check_seconds": check_seconds, "export_seconds": export_seconds})
    if any(sha((isolated / name).read_bytes()) != digest for name, digest in copied.items()):
        raise RuntimeError("Execution changed a copied source or artifact")
    record = {
        "scope": "preflight", "execution_passed": True,
        "full_selected_route_verified": False, "scientific_acceptance": False, "formal_i4_verified": False,
        "source_revision": revision, "working_tree_status": status,
        "working_tree_diff_sha256": sha(diff),
        "source_and_artifact_sha256": copied, "verifier_sha256": sha(Path(__file__).read_bytes()),
        "notebooks": runs, "visual_inspection": {"status": "pending", "note": "Inspect every PNG, HTML table and exercise separately."},
        "isolation": {"raw_training_banks_copied": False, "market_fixtures_copied": False,
                      "training_executed": False, "new_scientific_study": False,
                      "gbm_tiny_temporary_bank": "64/16/8 synthetic examples generated by the existing foundations lesson",
                      "notebook10": "existing fixed 2,048-proposal provisional live example; historical failed flag retained",
                      "network_guard": "Python socket/DNS guard; loopback and Unix IPC permitted; not an OS network sandbox",
                      "network_guard_sha256": sha(GUARD.encode())},
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "platform": platform.platform(), "cpu_backend_requested": True,
        "package_versions": {name: version(name) for name in
                             ("marimo", "numpy", "pandas", "matplotlib", "bayesflow", "keras", "jax", "numba", "scipy")},
        "clean_install_verified_by_this_script": False,
        "remaining_acceptance": ["Record #27's pinned combined revision and retained notebook10 mode",
                                 "Verify clean locked setup separately on that revision",
                                 "Run the complete selected route including the local-data walkthrough",
                                 "Inspect every exported graph and exercise after final presentation edits"],
    }
    (out / "verification.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"verification": str(out / "verification.json"), "exports": str(out)}, indent=2))


if __name__ == "__main__":
    main()
