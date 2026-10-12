"""Run both cached HKDE lessons in isolation, with networking and BayesFlow blocked.

This verifies the provisional teaching handoff, not scientific acceptance or a
live posterior calculation. Published scientific caches and checkpoint bytes
remain available for provenance checks; no market files or raw banks are copied.
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
import sys
import tempfile
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional new JSON execution report")
    parser.add_argument("--artifacts", type=Path, help="Optional new directory for the six rendered PNGs")
    args = parser.parse_args()
    args.out = args.out.resolve() if args.out else None
    args.artifacts = args.artifacts.resolve() if args.artifacts else None
    if args.out and args.out.exists():
        parser.error("--out must name a new file")
    if args.artifacts:
        args.artifacts.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
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
                     "docs/results/hkde-correction/test-independent-audit.json"))
    source_hashes = {name: sha(root / name) for name in sorted(bindings)}

    with tempfile.TemporaryDirectory(prefix="hkde-integration-offline-") as directory:
        isolated = Path(directory).resolve()
        for relative in ("src", "scripts", "configs", "docs",
                         "checkpoints/hkde-theta-intensity-provisional",
                         "data/synthetic/hkde-walkthrough"):
            shutil.copytree(root / relative, isolated / relative,
                            ignore=shutil.ignore_patterns("__pycache__"))
        (isolated / "notebooks").mkdir()
        for name in notebooks:
            shutil.copyfile(root / "notebooks" / name, isolated / "notebooks" / name)
        shutil.copyfile(root / "README.md", isolated / "README.md")
        sys.path.insert(0, str(isolated / "src"))
        os.environ["MPLBACKEND"] = "Agg"
        os.environ["MPLCONFIGDIR"] = str(isolated / "mpl")
        os.chdir(isolated)

        def offline(event, arguments):
            if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
                raise RuntimeError(f"network disabled: {event}")
            if event == "import" and arguments[0].split(".")[0] == "bayesflow":
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
        for name, expected_plots in zip(notebooks, (2, 4), strict=True):
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
        if "bayesflow" in sys.modules:
            raise RuntimeError("cached review imported BayesFlow")
        reports, checks = definitions["reports"], definitions["saved_checks"]["validation"]
        if (not reports["validation"]["passed"] or reports["test"]["passed"]
                or definitions["candidate_accepted"] or definitions["live_result"] is not None):
            raise RuntimeError("this cached-only check requires the currently closed scientific gate")
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
        for name, expected_sha in source_hashes.items():
            if sha(isolated / name) != expected_sha:
                raise RuntimeError(f"cached review changed a bound artifact: {name}")
        report = {
            "passed": True, "evidence_scope": "S: offline cached teaching integration only",
            "integration_base_commit": "0ee09dd9a6b5a5333960cb800e28d1f6dbe7b13d",
            "baseline_parent_commits": ["160bccecea4e806e6efce1e422eabf6347b9fb1b",
                                        "b9b37877ca92521b287eb3bf2bd92e2ea7093215"],
            "case_id": "validation:257", "n_contracts": len(definitions["local_panel"].rows),
            "notebooks": runs, "network_probe_rejected": True, "bayesflow_imported": False,
            "market_files_present": False, "raw_banks_present": False,
            "checkpoint_bytes_present_for_hashing": True, "live_executed": False,
            "new_test_rerun": False, "bound_artifacts_unchanged": True,
            "corrected_validation_passed": True, "corrected_test_passed": False,
            "scientific_acceptance": False, "comparison": definitions["comparison_rows"],
            "seconds_isolation_and_cached_execution": time.perf_counter() - started,
            "source_sha256": source_hashes,
        }
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        with args.out.open("x") as handle:
            handle.write(output)
    print(output, end="")


if __name__ == "__main__":
    main()
