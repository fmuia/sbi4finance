"""Export the selected synthetic lessons in fresh isolated Python processes.

Preflight covers the original 01/10/11 tree. Participant scope requires an
explicit integrated Git revision and covers foundations, the local walkthrough,
cached and provisional-live notebook10, and the five-figure decision notebook11.
Neither scope certifies clean installation, browser review, scientific acceptance
or formal I4. Use the locked environment belonging to the supplied repository.
"""

import argparse
import base64
import hashlib
import html
from importlib.metadata import version
from importlib.util import find_spec
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import tomllib


NOTEBOOKS = {
    "01_gbm_foundations": (2, ("exercises and next steps", "separate")),
    "10_hkde_corrected_workflow": (6, ("historical scientific acceptance: not passed",
                                      "provisional live demonstration", "review trail and exercises")),
    "11_hkde_calibration_audit": (3, ("historical failed flag", "review trail, reproduction and exercises")),
}
PARTICIPANT = [
    ("01_gbm_foundations", "01_gbm_foundations", "analytic-live", NOTEBOOKS["01_gbm_foundations"]),
    ("hkde_data_walkthrough", "hkde_data_walkthrough", "cached",
     (2, ("provisional example", "the inference boundary", "what would real data add?"))),
    ("10_hkde_corrected_workflow-cached", "10_hkde_corrected_workflow", "cached",
     (4, ("cached review mode", "the local observation entering this lesson",
          "same observation, same target, different estimators", "review trail and exercises"))),
    ("10_hkde_corrected_workflow-live", "10_hkde_corrected_workflow", "provisional-live",
     (6, ("provisional live demonstration", "formal scientific acceptance is false",
          "the local observation entering this lesson", "review trail and exercises"))),
    ("11_hkde_calibration_audit", "11_hkde_calibration_audit", "cached",
     (5, ("scientific status: provisional", "participant budget: 2,048 proposals", "i4: deferred",
          "review trail, reproduction and exercises"))),
]
SYNTHETIC = "data/synthetic/hkde-walkthrough/"
LOCAL_INPUTS = {SYNTHETIC + name for name in
                ("bundle.json", "task.json", "posterior.npz", "panel/manifest.json", "panel/observations.csv")}
DECISION = "docs/results/hkde-acceptance/decision.json"
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
        if fullname.split(".")[0] == "bayesflow" and __import__("os").environ.get("SBI4FINANCE_CACHED_ONLY") == "1":
            raise ImportError("Cached route must not import BayesFlow")
        if fullname.split(".")[0] == "sbi4finance":
            spec = PathFinder.find_spec(fullname, path)
            if spec is None or not Path(spec.origin).resolve().is_relative_to(Path.cwd() / "src"):
                raise ImportError("Scientific source escaped the isolated copy")
            return spec
sys.meta_path.insert(0, IsolatedSource())
'''


def sha(content):
    return hashlib.sha256(content).hexdigest()


def inputs(root, scope="preflight"):
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
    if scope == "participant":
        names.update(LOCAL_INPUTS)
        names.update(("notebooks/hkde_data_walkthrough.py", "docs/results/hkde-inference/validation-checks.npz", DECISION))
        decision = json.loads((root / DECISION).read_text())
        names.add(decision["artifact_manifest"])
        names.update("docs/results/hkde-correction/plots/" + name for name in ("price-errors.png", "reference-pit.png"))
        audit = json.loads((root / "docs/results/hkde-gate-audit/report.json").read_text())
        names.update(audit["input_sha256"])
        names.update(audit["source_sha256"])
        names.update(str(path.relative_to(root)) for path in (root / "docs").rglob("*.md"))
    return sorted(names)


def verify_participant(root, names):
    """Bind the retained #22 entry point and #29 decision without copying banks."""
    tasks = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["pixi"]["tasks"]
    expected_tasks = {"review-foundations": "01_gbm_foundations", "hkde-data-walkthrough": "hkde_data_walkthrough",
                      "review-hkde-corrected": "10_hkde_corrected_workflow", "demo-hkde-corrected": "10_hkde_corrected_workflow",
                      "review-hkde-calibration": "11_hkde_calibration_audit"}
    for task, notebook in expected_tasks.items():
        entry = tasks[task]
        command = entry["cmd"] if isinstance(entry, dict) else entry
        mode = entry.get("env", {}).get("SBI4FINANCE_HKDE_DEMO", "0") if isinstance(entry, dict) else "0"
        if command != f"marimo edit notebooks/{notebook}.py --no-token" or mode != ("1" if task == "demo-hkde-corrected" else "0"):
            raise ValueError(f"Selected participant task/mode changed: {task}")
    if 'os.environ.get("SBI4FINANCE_HKDE_DEMO", "0")' not in (root / "notebooks/10_hkde_corrected_workflow.py").read_text():
        raise ValueError("Expected #22's cached-default, explicit opt-in notebook10")
    panel = json.loads((root / (SYNTHETIC + "panel/manifest.json")).read_text())
    if (panel["data_origin"], panel["purpose"], panel["provenance"]["case_id"]) != ("synthetic", "validation", "validation:257"):
        raise ValueError("The participant observation must be synthetic validation:257")
    decision = json.loads((root / DECISION).read_text())
    manifest_bytes = (root / decision["artifact_manifest"]).read_bytes()
    if sha(manifest_bytes) != decision["artifact_manifest_sha256"]:
        raise ValueError("Scientific decision manifest changed")
    if (decision["scientific_status"], decision["scientific_acceptance"], decision["i4_status"],
            decision["participant_proposals"], decision["evidence_scope"], decision["measure"]) != ("provisional", False, "deferred", 2048, "S", "Q"):
        raise ValueError("Expected the provisional Q decision with formal I4 deferred")
    frozen = json.loads(manifest_bytes)["files"]
    # These integrated interfaces are bound to --revision, not the older
    # scientific inventory. Copy/verify only needed scientific dependencies;
    # the full historical inventory also names raw surrogate training banks.
    interfaces = {"notebooks/10_hkde_corrected_workflow.py", "pyproject.toml", "pixi.lock"}
    checked = sorted(set(names) & frozen.keys() - interfaces)
    changed = [name for name in checked if sha((root / name).read_bytes()) != frozen[name]]
    if changed:
        raise ValueError(f"Frozen scientific source/artifact changed: {changed}")
    return {"decision_id": decision["decision_id"], "scientific_status": "provisional", "i4": "deferred",
            "mode_tasks_verified": sorted(expected_tasks), "frozen_manifest_dependencies_verified": checked,
            "integrated_interfaces_bound_to_git_revision": sorted(interfaces)}


def require_pin(root, revision, names):
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != revision:
        raise ValueError("--revision must equal the selected repository HEAD")
    tracked = set(subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision], cwd=root, text=True).splitlines())
    if set(names) - tracked or subprocess.run(["git", "diff", "--quiet", revision, "--", *names], cwd=root).returncode:
        raise ValueError("Copied participant inputs must be tracked and unchanged from --revision")


def inspect_export(out, name, contract=None, label=None):
    expected_images, expected_text = contract or NOTEBOOKS[name]
    label = label or name
    html_path = out / f"{label}.html"
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
            filename = f"{label}-figure-{len(figures) + 1:02d}.png"
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


def browser_mirror(out, isolated, copied, runs):
    """Bundle the installed matching frontend; preserve original HTML evidence."""
    static = Path(find_spec("marimo").origin).parent / "_static"
    browser = out / "browser"
    shutil.copytree(static, browser)
    asset_hashes = {str(path.relative_to(static)): sha(path.read_bytes()) for path in static.rglob("*") if path.is_file()}
    for name in copied:
        target = browser / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(isolated / name, target)
    prefix = f"https://cdn.jsdelivr.net/npm/@marimo-team/frontend@{version('marimo')}/dist"
    derived = {}
    for run in runs:
        original = (out / run["html"]).read_text()
        transformed = original.replace(prefix, ".").replace(str(isolated), "./source")
        (browser / run["html"]).write_text(transformed)
        derived[run["html"]] = sha(transformed.encode())
    manifest = {"installed_marimo": version("marimo"), "frontend_asset_sha256": asset_hashes,
                "source_snapshot_sha256": copied, "derived_html_sha256": derived,
                "transformation": f"Replace {prefix} with . and isolated repository path with ./source; no notebook recomputation"}
    content = json.dumps(manifest, indent=2) + "\n"
    (browser / "delivery-manifest.json").write_text(content)
    return {"directory": "browser", "manifest": "browser/delivery-manifest.json", "manifest_sha256": sha(content.encode()),
            "asset_count": len(asset_hashes), "derived_html_sha256": derived,
            "browser_offline_verified": False, "review": "Serve browser/ over loopback with external requests blocked; inspect separately."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("preflight", "participant"), required=True)
    parser.add_argument("--repo", type=Path, help="Selected source repository; required for participant scope")
    parser.add_argument("--revision", help="Full 40-character pinned Git revision; required for participant scope")
    parser.add_argument("--out", type=Path, required=True, help="New directory for logs, HTML, figures and report")
    args = parser.parse_args()
    participant = args.scope == "participant"
    if participant and (args.repo is None or not re.fullmatch(r"[0-9a-f]{40}", args.revision or "")):
        parser.error("participant scope requires --repo and a full --revision")
    root, out = (args.repo or Path(__file__).resolve().parents[1]).resolve(), args.out.resolve()
    if out.exists():
        parser.error("--out must be new; existing results are never overwritten")
    started_all = time.perf_counter()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).splitlines()
    diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root)
    names = inputs(root, args.scope)
    if participant:
        require_pin(root, args.revision, names)
    elif args.revision and args.revision != revision:
        parser.error("--revision differs from repository HEAD")
    contract_record = verify_participant(root, names) if participant else None
    isolated = out / "isolated"
    isolated.mkdir(parents=True)
    (out / "working-tree.patch").write_bytes(diff)
    (out / "verifier.py").write_bytes(Path(__file__).read_bytes())
    copied = {}
    for name in names:
        path = (root / name).resolve()
        if (not path.is_relative_to(root) or name.startswith("tests/")
                or name.startswith("data/") and (not participant or name not in LOCAL_INPUTS)
                or re.fullmatch(r"bank-(train|validation|test)\.npz", path.name)):
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
    steps = []

    def run(label, arguments, mode="cached"):
        started = time.perf_counter()
        child_env = dict(env, SBI4FINANCE_HKDE_DEMO="1" if mode == "provisional-live" else "0",
                         SBI4FINANCE_CACHED_ONLY="1" if participant and mode != "provisional-live" else "0")
        result = None
        try:
            with (out / f"{label}.log").open("w") as stream:
                result = subprocess.run([sys.executable, *arguments], cwd=isolated, env=child_env,
                                        text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
        finally:
            elapsed = time.perf_counter() - started
            steps.append({"label": label, "arguments": arguments, "mode": mode, "seconds": elapsed,
                          "returncode": result.returncode if result else None})
            (out / "execution-steps.json").write_text(json.dumps({"steps": steps,
                "elapsed_since_start_seconds": time.perf_counter() - started_all}, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {out / (label + '.log')}")
        return elapsed

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
    route = PARTICIPANT if participant else [(name, name, "default", contract) for name, contract in NOTEBOOKS.items()]
    for label, name, mode, contract in route:
        notebook = f"notebooks/{name}.py"
        check_seconds = run(f"{label}-check", ["-m", "marimo", "check", "--strict", notebook], mode)
        html_path = out / f"{label}.html"
        export_seconds = run(f"{label}-export", ["-m", "marimo", "export", "html", notebook, "-o", str(html_path)], mode)
        parsed = inspect_export(out, name, contract, label)
        if participant and name == "10_hkde_corrected_workflow":
            checks = parsed["live_case_checks"]
            if mode == "cached" and (checks or parsed["stage_seconds"]):
                raise ValueError("Cached notebook10 unexpectedly contains a live result")
            if mode == "provisional-live":
                protocol = json.loads((isolated / "configs/hkde_correction_boundary_protocol.json").read_text())
                metrics = {row["Metric"]: row for row in checks if "Metric" in row}
                diagnostics = {row["Diagnostic"]: row for row in checks if "Diagnostic" in row}
                if (len(checks) != 8 or set(metrics) != set(protocol["gates"])
                        or set(diagnostics) != {"ESS / proposal count", "Largest normalized weight"}):
                    raise ValueError("Live export must retain all eight case checks")
                for key, row in metrics.items():
                    limit = protocol["gates"][key]
                    if row["Threshold (≤)"] != limit or row["Passed"] != (math.isfinite(row["Live value"]) and row["Live value"] <= limit):
                        raise ValueError(f"Live metric/threshold/flag disagree: {key}")
                for key, limit, greater in (("ESS / proposal count", protocol["conditional"]["min_ess_fraction"], True),
                                            ("Largest normalized weight", protocol["conditional"]["max_weight"], False)):
                    row = diagnostics[key]
                    passed = math.isfinite(row["Live value"]) and (row["Live value"] >= limit if greater else row["Live value"] <= limit)
                    if row["Passed"] != passed:
                        raise ValueError(f"Live weight metric/flag disagree: {key}")
                parsed["live_case_checks_passed"] = all(row["Passed"] for row in checks)
        runs.append({**parsed, "mode": mode, "strict_marimo_check": "passed",
                     "check_seconds": check_seconds, "export_seconds": export_seconds})
        (out / "notebook-results.json").write_text(json.dumps(runs, indent=2) + "\n")
    if any(sha((isolated / name).read_bytes()) != digest for name, digest in copied.items()):
        raise RuntimeError("Execution changed a copied source or artifact")
    if participant:
        require_pin(root, args.revision, names)
    browser = browser_mirror(out, isolated, copied, runs) if participant else None
    record = {
        "scope": args.scope, "execution_passed": True, "selected_route_execution_passed": participant,
        "full_selected_route_verified": False, "scientific_acceptance": False, "formal_i4_verified": False,
        "source_revision": revision, "working_tree_status": status,
        "working_tree_diff_sha256": sha(diff),
        "source_and_artifact_sha256": copied, "verifier_sha256": sha((out / "verifier.py").read_bytes()),
        "participant_contract": contract_record, "browser_mirror": browser,
        "execution_steps": steps, "total_export_seconds": sum(run["export_seconds"] for run in runs),
        "total_isolation_verification_export_seconds": time.perf_counter() - started_all,
        "notebooks": runs, "visual_inspection": {"status": "pending", "note": "Inspect every PNG, HTML table and exercise separately."},
        "isolation": {"raw_training_banks_copied": False, "market_fixtures_copied": False,
                      "training_executed": False, "new_scientific_study": False,
                      "gbm_tiny_temporary_bank": "64/16/8 synthetic examples generated by the existing foundations lesson",
                      "notebook10": "cached plus one opt-in fixed 2,048-proposal provisional live example" if participant else "existing fixed 2,048-proposal provisional live example",
                      "historical_failed_flag_retained": True, "raw_live_rows_saved": False,
                      "live_evidence": "Exported tables, figures, all eight case flags and stage timings; no second inference run",
                      "network_guard": "Python socket/DNS guard; loopback and Unix IPC permitted; not an OS network sandbox",
                      "network_guard_sha256": sha(GUARD.encode())},
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "platform": platform.platform(), "cpu_backend_requested": True,
        "package_versions": {name: version(name) for name in
                             ("marimo", "numpy", "pandas", "matplotlib", "bayesflow", "keras", "jax", "numba", "scipy")},
        "clean_install_verified_by_this_script": False,
        "remaining_acceptance": (["Verify clean locked setup separately on the pinned revision",
                                  "Inspect every graph, table and exercise and verify the browser mirror offline"] if participant else
                                 ["Record #27's pinned combined revision and retained notebook10 mode",
                                  "Verify clean locked setup separately on that revision",
                                  "Run the complete selected route including the local-data walkthrough",
                                  "Inspect every exported graph and exercise after final presentation edits"]),
    }
    (out / "verification.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"verification": str(out / "verification.json"), "exports": str(out)}, indent=2))


if __name__ == "__main__":
    main()
