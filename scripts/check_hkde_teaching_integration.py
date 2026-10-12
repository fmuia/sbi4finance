"""Run both HKDE lessons offline; optionally execute the provisional live demo.

Default cached review blocks BayesFlow. --live permits the frozen 2,048-proposal
demonstration and records execution separately from accuracy and scientific
acceptance. No market files or raw banks are copied; no training/test run occurs.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import socket
import subprocess
import sys
import tempfile
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkout_revision(root):
    """Identify HEAD and whether its committed tree describes the working copy."""
    def git(*arguments):
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        return {"commit": git("rev-parse", "HEAD"),
                "tree": git("rev-parse", "HEAD^{tree}"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=normal"))}
    except (OSError, subprocess.CalledProcessError):
        # A source archive has no Git metadata; the copied-file hashes still
        # identify exactly what ran. Never attribute it to a historical base.
        return {"commit": None, "tree": None, "dirty": None}


def isolation_status(isolated):
    """Inspect the copy, including banks stored below docs/results."""
    paths = [path.relative_to(isolated) for path in isolated.rglob("*")]
    status = {
        "market_files_present": any(
            path.parts[0] in {"data", "tests"}
            and not path.is_relative_to("data/synthetic/hkde-walkthrough")
            and path not in (Path("data"), Path("data/synthetic"))
            for path in paths),
        "raw_banks_present": any(
            {"bank", "banks"}.intersection(path.parts)
            or (path.suffix == ".npz" and (
                path.name.startswith("bank-") or path.stem in {"train", "validation", "test"}))
            for path in paths),
        "unused_checkpoints_present": any(
            path.parts[0] == "checkpoints" and path != Path("checkpoints")
            and not path.is_relative_to("checkpoints/hkde-theta-intensity-provisional")
            for path in paths),
    }
    if any(status.values()):
        raise RuntimeError(f"unexpected inputs in isolated review: {status}")
    return status


def copy_inputs(root, isolated, bindings):
    """Copy code and explicit lesson assets, never entire result directories."""
    paths = set(bindings)
    for directory in ("src", "scripts", "configs"):
        paths.update(str(path.relative_to(root)) for path in (root / directory).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts)
    hashes = {}
    for name in sorted(paths):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"review input must be relative to the checkout: {name}")
        target = isolated / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
        hashes[name] = sha(target)
    return hashes, isolation_status(isolated)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Run the explicitly provisional participant demonstration")
    parser.add_argument("--out", type=Path, help="Optional new JSON execution report")
    parser.add_argument("--artifacts", type=Path, help="Optional new directory for rendered PNGs and live weighted rows")
    args = parser.parse_args()
    args.out = args.out.resolve() if args.out else None
    args.artifacts = args.artifacts.resolve() if args.artifacts else None
    if args.out and args.out.exists():
        parser.error("--out must name a new file")
    if args.artifacts:
        args.artifacts.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    revision = checkout_revision(root)
    notebooks = ("hkde_data_walkthrough.py", "10_hkde_corrected_workflow.py")
    correction_dir = Path("docs/results/hkde-correction")
    bindings = set()
    for split in ("validation", "test"):
        saved = json.loads((root / correction_dir / f"{split}.json").read_text())
        bindings.update(saved["input_sha256"])
        bindings.update(str(correction_dir / name) for name in (f"{split}.json", f"{split}-checks.npz"))
        for name in saved["source_sha256"]:
            bindings.add(str(Path("scripts" if (root / "scripts" / name).exists()
                                  else "src/sbi4finance") / name))
    bindings.update(f"notebooks/{name}" for name in notebooks)
    bindings.update(("scripts/check_hkde_teaching_integration.py",
                     "scripts/plot_hkde_correction.py",
                     "src/sbi4finance/datasets/local.py", "src/sbi4finance/datasets/options.py",
                     "src/sbi4finance/datasets/hkde_teaching.py",
                     "data/synthetic/hkde-walkthrough/bundle.json",
                     "data/synthetic/hkde-walkthrough/task.json",
                     "data/synthetic/hkde-walkthrough/posterior.npz",
                     "data/synthetic/hkde-walkthrough/panel/manifest.json",
                     "data/synthetic/hkde-walkthrough/panel/observations.csv",
                     "docs/results/hkde-inference/validation-checks.npz",
                     "docs/results/hkde-correction/test-reference-pit.npz",
                     "docs/results/hkde-correction/test-independent-audit.json",
                     "docs/tutorial-participant-route.md",
                     "docs/hkde-correction-boundary-protocol.md",
                     "docs/hkde-correction-protocol.md",
                     "docs/hkde-joint-inference.md",
                     "docs/hkde-price-surrogate.md",
                     "docs/hkde-calibration-gate-audit.md",
                     "docs/results/hkde-gate-audit/report.json",
                     "docs/results/hkde-gate-audit/checks.npz",
                     "docs/results/hkde-gate-audit/input-consistency.json"))
    audit = json.loads((root / "docs/results/hkde-gate-audit/report.json").read_text())
    for key in ("input_sha256", "source_sha256"):
        for name, expected_sha in audit[key].items():
            if sha(root / name) != expected_sha:
                raise RuntimeError(f"gate audit binding changed: {name}")
            bindings.add(name)
    with tempfile.TemporaryDirectory(prefix="hkde-integration-offline-") as directory:
        isolated = Path(directory).resolve()
        source_hashes, isolated_status = copy_inputs(root, isolated, bindings)
        sys.path.insert(0, str(isolated / "src"))
        os.environ["MPLBACKEND"] = "Agg"
        os.environ["MPLCONFIGDIR"] = str(isolated / "mpl")
        os.environ["SBI4FINANCE_HKDE_DEMO"] = "1" if args.live else "0"
        os.environ.setdefault("KERAS_BACKEND", "jax")
        os.chdir(isolated)

        def offline(event, arguments):
            if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
                raise RuntimeError(f"network disabled: {event}")
            if not args.live and event == "import" and arguments[0].split(".")[0] == "bayesflow":
                raise RuntimeError("cached review must not import BayesFlow")

        sys.addaudithook(offline)
        with socket.socket() as probe:
            try:
                probe.connect(("127.0.0.1", 1))
            except RuntimeError as error:
                if str(error) != "network disabled: socket.connect":
                    raise
            else:
                raise RuntimeError("network guard did not reject the probe before connection")

        runs = []
        for name, expected_plots in zip(notebooks, (2, 6 if args.live else 4), strict=True):
            outputs, definitions = runpy.run_path(str(isolated / "notebooks" / name))["app"].run()
            images = [image for output in outputs for image in re.findall(
                r"data:image/png;base64,([A-Za-z0-9+/=]+)", getattr(output, "text", ""))]
            if len(images) != expected_plots:
                raise RuntimeError(f"{name}: expected {expected_plots} rendered plots, got {len(images)}")
            if args.artifacts:
                for index, image in enumerate(images, 1):
                    (args.artifacts / f"{Path(name).stem}-{index}.png").write_bytes(base64.b64decode(image))
            runs.append({"notebook": name, "outputs": len(outputs), "rendered_plots": len(images)})

        import numpy as np
        from sbi4finance.hkde_reference import weighted_quantiles
        for name, module in list(sys.modules.items()):
            if name.startswith("sbi4finance") and getattr(module, "__file__", None):
                if not Path(module.__file__).resolve().is_relative_to(isolated / "src"):
                    raise RuntimeError(f"{name}: module escaped the isolated checkout")
        if not args.live and "bayesflow" in sys.modules:
            raise RuntimeError("cached review imported BayesFlow")
        reports, checks = definitions["reports"], definitions["saved_checks"]["validation"]
        if (not reports["validation"]["passed"] or reports["test"]["passed"]
                or definitions["candidate_accepted"]):
            raise RuntimeError("the historical scientific gate must remain closed")
        index = list(checks["case_ids"]).index("validation:257")
        np.testing.assert_array_equal(definitions["local_observation"], checks["observations"][index])
        with np.load(isolated / "docs/results/hkde-inference/validation-checks.npz", allow_pickle=False) as raw:
            raw_index = list(raw["case_ids"]).index("validation:257")
            expected = [np.quantile(raw["target_draws"][raw_index, :, 0], [.05, .5, .95]),
                        weighted_quantiles(checks["target_prices"][index, :, 0], checks["weights"][index]),
                        checks["reference_target_quantiles"][index, 0]]
        actual = [[row[key] for key in ("5%", "Median", "95%")]
                  for row in definitions["comparison_rows"]]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        live, live_report = definitions["live_result"], None
        if bool(definitions["provisional_demo"]) != args.live:
            raise RuntimeError("notebook demonstration mode differs from the requested mode")
        if args.live:
            if (live is None or live["execution_mode"] != "provisional_demo"
                    or live["scientific_acceptance"] or "bayesflow" not in sys.modules):
                raise RuntimeError("expected an explicitly provisional live calculation")
            parameters, weights, prices = [np.asarray(live[key]) for key in
                                           ("parameters", "weights", "target_prices")]
            if (parameters.shape != (2048, 2) or weights.shape != (2048,)
                    or prices.shape != (2048, 2) or np.any(weights < 0)
                    or not all(np.isfinite(a).all() for a in (parameters, weights, prices))):
                raise RuntimeError("expected 2,048 finite paired parameter/weight/target rows")
            np.testing.assert_allclose(weights.sum(), 1., rtol=0, atol=1e-12)
            np.testing.assert_array_equal(live["observation"], definitions["local_observation"])
            for target in range(2):
                np.testing.assert_allclose(live["summary"]["target_quantiles"][target],
                                          weighted_quantiles(prices[:, target], weights), rtol=0, atol=1e-12)
            if bool(live["passed"]) != all(live["gates"].values()):
                raise RuntimeError("live accuracy flag differs from recorded gates")
            live_report = {key: live[key] for key in (
                "case_id", "execution_mode", "scientific_acceptance", "seed", "targets",
                "summary", "metrics", "gates", "ess", "ess_fraction", "max_weight",
                "prior_rows", "boundary_rows", "timing_seconds")}
            live_report.update(proposal_rows=len(weights), weights_sum=float(weights.sum()),
                               live_accuracy_passed=bool(live["passed"]),
                               reference_target_quantiles=np.asarray(
                                   live["reference_summary"]["target_quantiles"]).tolist())
            if args.artifacts:
                arrays_path = args.artifacts / "live-weighted-rows.npz"
                np.savez_compressed(arrays_path, parameters=parameters, weights=weights,
                                    target_prices=prices, observation=live["observation"],
                                    seed=np.asarray(live["seed"]),
                                    reference_target_quantiles=live_report["reference_target_quantiles"])
                live_report["weighted_rows_sha256"] = sha(arrays_path)
        elif live is not None:
            raise RuntimeError("cached review unexpectedly executed live inference")
        for name, expected_sha in source_hashes.items():
            if sha(isolated / name) != expected_sha:
                raise RuntimeError(f"teaching execution changed a bound artifact: {name}")
        isolation_status(isolated)
        report = {
            "passed": True, "execution_passed": True,
            "evidence_scope": ("S: offline provisional live demonstration on validation:257"
                               if args.live else "S: offline cached teaching integration only"),
            "source_revision": revision,
            "source_revision_scope": "Git tree identifies committed HEAD; source_sha256 identifies every copied execution input, including working-copy changes.",
            "case_id": "validation:257", "n_contracts": len(definitions["local_panel"].rows),
            "notebooks": runs, "network_probe_rejected": True, "bayesflow_imported": "bayesflow" in sys.modules,
            **isolated_status,
            "checkpoint_bytes_present_for_hashing": True, "checkpoint_loaded": args.live,
            "live_executed": args.live, "live_accuracy_passed": bool(live["passed"]) if args.live else None,
            "new_test_rerun": False, "bound_artifacts_unchanged": True,
            "corrected_validation_passed": True, "corrected_test_passed": False,
            "scientific_acceptance": False, "comparison": definitions["comparison_rows"],
            "live": live_report, "seconds_isolation_and_execution": time.perf_counter() - started,
            "source_sha256": source_hashes,
        }
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        with args.out.open("x") as handle:
            handle.write(output)
    print(output, end="")


if __name__ == "__main__":
    main()
