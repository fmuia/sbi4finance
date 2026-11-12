"""Run existing independent mathematical/support tests against research variants.

No builds, pricing banks or jobs are started. An optional Cython extension must
already have been built by benchmark_rough_cython.py in a separate directory.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from sbi4f_tutorial.rough_options import _pricing_acceleration as acceleration
from sbi4f_tutorial.rough_options._pricing_fft_experiments import PreparedHybridDriver
from scripts.benchmark_rough_pricing_speed import ROOT


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(extension=None):
    paths = [ROOT/"tutorial/tests/test_rough_pricing_oracles.py",
             ROOT/"tutorial/tests/test_rough_pricing_acceleration.py"]
    oracle = load_module("rough_mathematical_oracles", paths[0])
    parity = load_module("rough_acceleration_checks", paths[1])
    kernels = {"numba": None}
    if extension is not None:
        kernels["cython"] = load_module("_bergomi_pricing_cython", extension).terminal_moments
    checks = []
    for kernel_name, kernel in kernels.items():
        for driver_name in ("reference", "cached_exact", "fast_length"):
            suite = unittest.TestSuite()
            for cls in (oracle.RoughDriverOracleTests, oracle.RoughPricingOracleTests,
                        parity.RoughPricingAccelerationTests):
                suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
            with ExitStack() as stack:
                if driver_name != "reference":
                    driver = PreparedHybridDriver(driver_name)
                    stack.enter_context(patch.object(acceleration, "rough_driver", driver))
                    stack.enter_context(patch.object(oracle, "rough_driver", driver))
                if kernel is not None:
                    stack.enter_context(patch.object(acceleration, "_compiled_kernel", return_value=kernel))
                result = unittest.TestResult()
                suite.run(result)
            entry = {"kernel": kernel_name, "driver": driver_name, "tests": result.testsRun,
                "passed": result.wasSuccessful(), "skips": len(result.skipped),
                "failures": [{"test": str(test), "traceback": trace} for test, trace in result.failures+result.errors]}
            checks.append(entry)
            print(json.dumps(entry), flush=True)
    return {"format": "rough_pricing_variant_correctness_v1", "checks": checks,
        "passed": all(entry["passed"] and not entry["skips"] for entry in checks),
        "test_sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "cython_extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest() if extension else None,
        "scope": "Existing independent causal-sum/scalar-model/Black oracles and support/parity tests under research driver/kernel overrides; no bank or job changes. This does not certify prior-wide MC coverage or continuous-model bias."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cython-extension", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("new output path required")
    record = check(args.cython_extension)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(record, stream, indent=2); stream.write("\n")
    if not record["passed"]:
        raise SystemExit(1)
