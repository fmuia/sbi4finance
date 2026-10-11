"""Run the cached teaching notebook with no market files, weights or networking.

Copies only the notebook, local data helpers and synthetic bundle into a fresh
temporary checkout. This checks offline execution, not posterior accuracy.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import time


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional new JSON execution report")
    args = parser.parse_args()
    if args.out:
        args.out = args.out.resolve()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="hkde-teaching-offline-") as directory:
        isolated = Path(directory).resolve()
        for relative in ("src/sbi4finance/datasets", "data/synthetic/hkde-walkthrough"):
            shutil.copytree(root / relative, isolated / relative,
                            ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(root / "src/sbi4finance/__init__.py", isolated / "src/sbi4finance/__init__.py")
        notebook = isolated / "notebooks/hkde_data_walkthrough.py"
        notebook.parent.mkdir()
        shutil.copyfile(root / "notebooks/hkde_data_walkthrough.py", notebook)
        sys.path.insert(0, str(isolated / "src"))
        os.environ["MPLBACKEND"] = "Agg"
        os.environ["MPLCONFIGDIR"] = str(isolated / "mpl")
        os.chdir(isolated)

        def offline(event, arguments):
            if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
                raise RuntimeError(f"network access is disabled in this offline check: {event}")

        sys.addaudithook(offline)
        outputs, definitions = runpy.run_path(str(notebook))["app"].run()
        plots = sum("data:image/png;base64," in getattr(output, "text", "") for output in outputs)
        if plots != 2:
            raise RuntimeError("expected both posterior plots in rendered notebook output")
        # These named results come from the notebook, not a separate replica.
        panel, bundle, arrays = [definitions[name] for name in ("panel", "report", "arrays")]
        import sbi4finance.datasets.hkde_teaching as reader
        if Path(reader.__file__).resolve() != isolated / "src/sbi4finance/datasets/hkde_teaching.py":
            raise RuntimeError("notebook used data code outside the isolated checkout")
        if any(name in sys.modules for name in ("bayesflow", "sbi4finance.hkde_inference", "sbi4finance.hkde")):
            raise RuntimeError("cached notebook unexpectedly imported training/inference/pricing code")
        report = {
            "passed": True, "evidence_scope": "offline cached execution only",
            "case_id": bundle["case_id"], "n_contracts": len(panel.rows),
            "cached_draw_count": len(arrays["parameter_draws"]),
            "notebook_outputs": len(outputs), "network_operations_blocked": True,
            "rendered_plots": plots,
            "market_files_present": False, "checkpoint_weights_present": False,
            "simulator_code_present": False, "reserved_test_evaluated": False,
            "checkpoint_status": bundle["checkpoint_status"],
            "scientific_validation_passed": bundle["scientific_validation_passed"],
            "seconds": time.perf_counter() - started,
            "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                              for name in ("notebooks/hkde_data_walkthrough.py",
                                           "src/sbi4finance/datasets/hkde_teaching.py",
                                           "scripts/check_hkde_teaching.py",
                                           "data/synthetic/hkde-walkthrough/bundle.json")},
        }
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out:
        with args.out.open("x") as handle:
            handle.write(output)
    print(output, end="")


if __name__ == "__main__":
    main()
