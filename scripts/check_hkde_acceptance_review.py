"""Verify the cached HKDE decision lesson; no inference or fresh scientific data.

Use the existing pinned Python. --out must name a new directory. The isolated
copy and HTML are retained for separate visual inspection; no install is tested.
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
import tempfile
import time
import tomllib


NOTEBOOK = "notebooks/11_hkde_calibration_audit.py"
DECISION = "docs/results/hkde-acceptance/decision.json"
# These presentation bindings also identify the original delivery evidence.
# Reconciled delivery is checked separately; scientific inputs stay byte-exact.
HISTORICAL_TEACHING_PATHS = {
    "notebooks/10_hkde_corrected_workflow.py",
    "pyproject.toml",
}
CACHED_INPUTS = [
    "docs/results/hkde-gate-audit/report.json",
    "docs/results/hkde-gate-audit/checks.npz",
    "docs/results/hkde-gate-audit/oracle-rank-control.png",
    "docs/results/hkde-gate-audit/paired-rank-errors.png",
    "docs/results/hkde-gate-audit/conditional-cdf-errors.png",
    "docs/results/hkde-correction/test.json",
    "docs/results/hkde-correction/plots/price-errors.png",
    "docs/results/hkde-correction/plots/reference-pit.png",
    "configs/hkde_correction_boundary_protocol.json",
    "configs/hkde_inference_protocol.json",
]
GUARD = '''import importlib.abc
import ipaddress
import socket
import sys

def local(address):
    if isinstance(address, (str, bytes)):
        return True  # AF_UNIX kernel IPC
    host = address[0]
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False

def guarded(original):
    def call(self, address, *args):
        if not local(address):
            raise RuntimeError("External Python socket connections disabled")
        return original(self, address, *args)
    return call

socket.socket.connect = guarded(socket.socket.connect)
socket.socket.connect_ex = guarded(socket.socket.connect_ex)
original_getaddrinfo = socket.getaddrinfo
def getaddrinfo(host, port, *args, **kwargs):
    if host is not None and not local((host, port)):
        raise RuntimeError("External Python DNS lookups disabled")
    return original_getaddrinfo(host, port, *args, **kwargs)
socket.getaddrinfo = getaddrinfo

class CachedOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"sbi4finance", "bayesflow", "keras", "jax"}:
            raise ModuleNotFoundError("Scientific inference imports disabled in cached review")
sys.meta_path.insert(0, CachedOnly())
'''


def sha(content):
    return hashlib.sha256(content).hexdigest()


def source(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Expected repository-relative input: {name}")
    return path


def verify_manifest(root, manifest, *, allow_integrated_teaching=False):
    """Verify current inputs and, explicitly, two historical delivery bindings.

    The historical manifest is never changed. Only notebook10 and the Pixi
    task table can be reconciled, using original bytes from local Git history.
    This does not verify the replacement lesson's execution or accept inference.
    """
    frozen = manifest["files"]
    if not frozen:
        raise ValueError("Frozen manifest is empty")
    current_sha256 = {name: sha(source(root, name).read_bytes()) for name in frozen}
    mismatches = [name for name, digest in frozen.items() if current_sha256[name] != digest]
    if mismatches and (not allow_integrated_teaching
                       or not set(mismatches) <= HISTORICAL_TEACHING_PATHS):
        raise ValueError(f"Frozen manifest hashes differ: {mismatches}")

    replacements = {}
    for name in mismatches:
        base = manifest["base_commit"]
        if not re.fullmatch(r"[0-9a-f]{40}", base):
            raise ValueError("Historical teaching verification requires a full frozen commit SHA")
        try:
            original = subprocess.run(
                ["git", "-C", str(root), "show", f"{base}:{name}"],
                check=True, capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError(f"Cannot verify historical teaching input {name} at local commit {base}") from error
        if sha(original) != frozen[name]:
            raise ValueError(f"Historical teaching input differs from frozen manifest: {name}")
        if name == "pyproject.toml":
            historical_config = tomllib.loads(original.decode())
            current_config = tomllib.loads(source(root, name).read_text())
            for config in (historical_config, current_config):
                config.get("tool", {}).get("pixi", {}).pop("tasks", None)
            if historical_config != current_config:
                raise ValueError("Integrated pyproject.toml may change only the Pixi task table")
        replacements[name] = {
            "historical_commit": base,
            "historical_sha256": frozen[name],
            "current_sha256": current_sha256[name],
            "scope": ("Pixi tasks only; dependency and environment configuration unchanged"
                      if name == "pyproject.toml" else
                      "Reconciled teaching entry point; cached/live execution requires separate integration verification"),
        }
    return {
        "manifest_files_verified": len(frozen),
        "manifest_current_files_verified": len(frozen) - len(replacements),
        "manifest_historical_teaching_files_verified": len(replacements),
        "all_frozen_files_match_current_tree": not replacements,
        "scientific_inputs_match_current_tree": True,
        "historical_teaching_bindings": replacements,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--allow-integrated-teaching", action="store_true",
        help="Verify original notebook10/Pixi-task bindings in local Git and record their replacements; scientific inputs remain exact",
    )
    args = parser.parse_args()
    root, out = args.repo.resolve(), args.out.resolve()
    if out.exists():
        parser.error("--out must name a new directory; existing results are never overwritten")
    decision = json.loads(source(root, DECISION).read_text())
    manifest_name = decision["artifact_manifest"]
    manifest_bytes = source(root, manifest_name).read_bytes()
    if sha(manifest_bytes) != decision["artifact_manifest_sha256"]:
        raise ValueError("Artifact manifest differs from the decision binding")
    manifest = json.loads(manifest_bytes)
    frozen = manifest["files"]
    if manifest["base_commit"] != decision["base_commit"]:
        raise ValueError("Manifest and decision must bind the same historical commit")
    provenance = verify_manifest(
        root, manifest, allow_integrated_teaching=args.allow_integrated_teaching,
    )
    out.mkdir(parents=True)
    isolated = Path(tempfile.mkdtemp(prefix="hkde-acceptance-review-", dir="/tmp"))
    copied = {}
    for name in [NOTEBOOK, DECISION, manifest_name, *CACHED_INPUTS]:
        content = source(root, name).read_bytes()
        if name in CACHED_INPUTS and frozen.get(name) != sha(content):
            raise ValueError(f"Cached review input is not bound by the manifest: {name}")
        target = isolated / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        copied[name] = sha(content)
    protocol = json.loads((isolated / "configs/hkde_correction_boundary_protocol.json").read_text())
    baseline = json.loads((isolated / "configs/hkde_inference_protocol.json").read_text())
    historical = json.loads((isolated / "docs/results/hkde-correction/test.json").read_text())
    decision_checks = {
        "conditional_thresholds_unchanged": decision["conditional_thresholds"] == protocol["gates"],
        "reference_thresholds_unchanged": decision["reference_thresholds"] == baseline["reference"]["gates"],
        "conditional_weights_unchanged": all(decision["conditional_weights"][key] == protocol["conditional"][key] for key in ("min_ess_fraction", "max_weight")),
        "calibration_limits_unchanged": all(decision["calibration_policy"][key] == protocol["calibration"][key] for key in ("family_alpha", "min_ess_fraction", "max_weight")),
        "budgets_unchanged": (decision["participant_proposals"] == protocol["conditional"]["participant_samples"] == protocol["calibration"]["proposal_samples"] == 2048
                              and decision["precision_proposals"] == protocol["conditional"]["samples"] == 8192
                              and decision["calibration_sir_draws"] == protocol["calibration"]["resampled_draws"] == 512),
        "status_provisional_i4_deferred": decision["scientific_status"] == "provisional" and decision["i4_status"] == "deferred",
        "failed_historical_acceptance_retained": decision["scientific_acceptance"] is False and decision["historical_test_passed"] is False and historical["passed"] is False,
        "no_fresh_experiment": decision["fresh_replication"]["executed"] is False and all(decision["fresh_replication"][key] == 0 for key in ("new_panels", "new_proposals", "training_runs")),
    }
    if not all(decision_checks.values()):
        raise ValueError(f"Decision/protocol mismatch: {decision_checks}")
    guard_dir = isolated / "offline-guard"
    guard_dir.mkdir()
    (guard_dir / "sitecustomize.py").write_text(GUARD)
    env = dict(os.environ, PYTHONPATH=str(guard_dir), PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
               MPLCONFIGDIR=str(isolated / "mpl-cache"))

    def run(label, arguments):
        result = subprocess.run([sys.executable, *arguments], cwd=isolated, env=env,
                                text=True, capture_output=True, timeout=180)
        (out / f"{label}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {out / (label + '.log')}")

    run("guard-preflight", ["-c", """import socket, sitecustomize
assert socket.socket.connect.__module__ == 'sitecustomize'
try: socket.create_connection(('192.0.2.1', 9))
except RuntimeError: pass
else: raise AssertionError('Socket guard inactive')
try: import sbi4finance
except ImportError as error: assert 'Scientific inference imports disabled' in str(error)
else: raise AssertionError('Cached-only import guard inactive')
print('External Python socket/DNS access and scientific imports guarded')
"""])
    run("marimo-check", ["-m", "marimo", "check", "--strict", NOTEBOOK])
    started = time.perf_counter()
    run("export", ["-m", "marimo", "export", "html", NOTEBOOK,
                   "-o", "review.html", "--force"])
    elapsed = time.perf_counter() - started
    exported = (isolated / "review.html").read_bytes()
    match = re.search(r'"session":\s*', exported.decode())
    if match is None:
        raise ValueError("Export has no inspectable marimo session")
    session = json.JSONDecoder().raw_decode(exported.decode()[match.end():])[0]
    outputs = [output for cell in session["cells"] for output in cell["outputs"]]
    if any(output["type"] == "error" for output in outputs):
        raise ValueError("Export contains marimo cell errors")
    encoded = json.dumps(outputs)
    visible = re.sub(r"<[^>]+>", " ", html.unescape(encoded)).lower()
    visible = re.sub(r"\s+", " ", visible.replace("\\n", " "))
    expected = ["scientific status: provisional", "participant budget: 2,048 proposals", "i4: deferred"]
    if any(term not in visible for term in expected):
        raise ValueError("Export is missing an explicit provisional/budget/I4 status")
    images = re.findall(r"data:image/png;base64,([A-Za-z0-9+/=]+)", encoded)
    image_hashes = sorted(sha(base64.b64decode(value)) for value in images)
    if len(images) != 5 or image_hashes != sorted(copied[p] for p in CACHED_INPUTS if p.endswith(".png")):
        raise ValueError("Export must display exactly the five bound cached figures")
    record = {
        "scope": "Cached scientific decision lesson; no new acceptance experiment",
        "notebook": NOTEBOOK, "strict_marimo_check": "passed",
        "output_cells": len(session["cells"]), "error_outputs": 0,
        "displayed_figures": len(images), "visible_status_checks": expected,
        "decision_checks": decision_checks,
        "visual_inspection": {"status": "pending", "note": "Inspect the five figures and HTML separately."},
        "new_scientific_panels": 0, "neural_inference_executed": False,
        "scientific_imports_blocked": ["sbi4finance", "bayesflow", "keras", "jax"],
        "copied_src_checkpoints_raw_banks_market_fixtures": False,
        "network_guard_scope": "Python socket connect/connect_ex and DNS; loopback/Unix IPC permitted; not an OS sandbox",
        "network_guard_sha256": sha(GUARD.encode()),
        **provenance, "source_and_artifact_sha256": copied,
        "verifier_sha256": sha(Path(__file__).read_bytes()),
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "package_versions": {name: version(name) for name in ("marimo", "numpy")},
        "environment_scope": "Existing environment, fresh isolated input copy/process; no dependency installation certification",
        "isolated_copy": str(isolated), "html_path": str(isolated / "review.html"),
        "html_sha256": sha(exported), "export_elapsed_seconds": elapsed,
    }
    (out / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"verification": str(out / "verification.json"), "html": record["html_path"]}, indent=2))


if __name__ == "__main__":
    main()
