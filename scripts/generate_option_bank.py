"""Generate an option bank, or verify an existing bank without changing it."""

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

from sbi4finance import option_bank


def _environment():
    return {
        "versions": {"numpy": np.__version__, "scipy": option_bank.scipy.__version__,
                     "python": platform.python_version()},
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "description": platform.platform()},
    }


def _differences(saved, current):
    return {key: {"saved": saved.get(key), "current": current.get(key)}
            for key in sorted(saved.keys() | current.keys())
            if saved.get(key) != current.get(key)}


def verify_bank(path, *, portable=False, rtol=1e-12, atol=1e-14):
    """Check file integrity/source identity, then regenerate selected panel IDs.

    Exact mode also requires identical recorded package versions and, when
    available, platform. Portable mode permits environment changes; floating
    arrays must satisfy abs(saved - regenerated) <= atol + rtol*abs(regenerated).
    IDs and scientific sources always match exactly. Neither mode writes files
    or certifies pricing bias, inference accuracy or market adequacy.
    """
    path = Path(path)
    if not np.isfinite([rtol, atol]).all() or rtol < 0 or atol < 0:
        raise ValueError("finite nonnegative verification tolerances required")
    manifest, splits = option_bank.load_bank(path)  # Includes stored NPZ hashes.
    expected_names = set(option_bank.source_files(manifest["configuration"]["task"]["model"]))
    saved_sources = manifest["source_sha256"]
    if set(saved_sources) != expected_names:
        raise ValueError("generator source list changed or incomplete")
    current_sources = {name: option_bank._sha256(Path(option_bank.__file__).with_name(name))
                       for name in expected_names}
    source_differences = _differences(saved_sources, current_sources)
    if source_differences:
        raise ValueError(f"generator source changed: {json.dumps(source_differences, sort_keys=True)}")

    environment = _environment()
    version_differences = _differences(manifest["versions"], environment["versions"])
    # Platform belongs to execution evidence, not the scientific task or bank
    # format. Older generation reports did not record it; do not invent it.
    saved_report_path = path / "verification.json"
    saved_report = json.loads(saved_report_path.read_text()) if saved_report_path.exists() else {}
    saved_platform = saved_report.get("environment", {}).get("platform")
    platform_differences = (_differences(saved_platform, environment["platform"])
                            if saved_platform is not None else None)
    if not portable and (version_differences or platform_differences):
        raise ValueError("exact regeneration requires the recorded environment; "
                         f"version differences={version_differences}; "
                         f"platform differences={platform_differences}; use --portable")

    checked, comparisons = {}, {}
    for split, arrays in splits.items():
        count = len(arrays["example_id"])
        ids = sorted({0, count // 2, count - 1}, reverse=True)
        regenerated = option_bank.generate_examples(manifest["configuration"], split, ids)
        comparisons[split] = {}
        if set(arrays) != set(regenerated):
            raise ValueError(f"{split}: regenerated array names changed")
        for key in arrays:
            stored, reference = arrays[key][ids], regenerated[key]
            if stored.shape != reference.shape or stored.dtype != reference.dtype:
                raise ValueError(f"{split}/{key}: regenerated shape or dtype changed")
            if key == "example_id":
                passed = np.array_equal(stored, reference)
                comparison = {"comparison": "exact IDs", "passed": bool(passed)}
            else:
                difference = np.abs(stored - reference)
                nonzero = reference != 0
                comparison = {
                    "comparison": "tolerance" if portable else "exact",
                    "max_absolute_difference": float(difference.max()),
                    "max_relative_difference_nonzero_reference": (
                        float(np.max(difference[nonzero] / np.abs(reference[nonzero])))
                        if np.any(nonzero) else 0.0),
                    "max_absolute_difference_at_zero_reference": (
                        float(difference[~nonzero].max()) if np.any(~nonzero) else 0.0),
                }
                passed = (np.allclose(stored, reference, rtol=rtol, atol=atol)
                          if portable else np.array_equal(stored, reference))
                comparison["passed"] = bool(passed)
            comparisons[split][key] = comparison
            if not passed:
                raise ValueError(f"{split}/{key}: regeneration mismatch: {json.dumps(comparison)}")
        checked[split] = ids
    sigma = manifest["configuration"]["task"]["context"]["quote_noise_std"]
    return {
        "passed": True,
        "scope": "S: storage, source identity and selected-ID numerical regeneration",
        "verification_mode": "portable" if portable else "exact",
        "tolerances": {"rtol": rtol if portable else 0.0, "atol": atol if portable else 0.0,
                       "relative_to": "regenerated values"},
        "environment": environment,
        "environment_comparison": {"saved_versions": manifest["versions"],
                                   "version_differences": version_differences,
                                   "saved_platform": saved_platform,
                                   "platform_differences": platform_differences,
                                   "platform_available": saved_platform is not None},
        "scientific_source_identity": {"matched": True, "sha256": current_sources,
                                       "differences": source_differences},
        "transfer_integrity": {"matched_manifest_split_hashes": True,
                               "split_sha256": {s: r["sha256"] for s, r in manifest["splits"].items()}},
        "array_comparisons": comparisons,
        "regenerated_ids": checked,
        "manifest_sha256": option_bank._sha256(path / "manifest.json"),
        "loaded_array_bytes": sum(a.nbytes for split in splits.values() for a in split.values()),
        "max_mc_se_over_quote_sigma": max(float(a["pricing_standard_errors"].max()) / sigma for a in splits.values()),
        "max_abs_fourier_refinement_over_quote_sigma": max(float(np.abs(a["refinement_difference"]).max()) / sigma for a in splits.values()),
        "numerical_bias_certified": False, "posterior_validated": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--out", type=Path, help="generate a new directory; never overwritten")
    action.add_argument("--verify-existing", type=Path, help="read-only verification of an existing bank")
    parser.add_argument("--config", type=Path, help="generation config (default: configs/tasks/hkde_option_bank.json)")
    parser.add_argument("--portable", action="store_true",
                        help="verification only: permit environment differences, rtol=1e-12 and atol=1e-14")
    parser.add_argument("--report", type=Path,
                        help="verification only: write a new report outside the bank (default: stdout only)")
    args = parser.parse_args()
    if args.verify_existing is not None:
        if args.config is not None:
            parser.error("--config is only for generation; verification uses the stored configuration")
        if args.report is not None:
            if args.report.resolve().is_relative_to(args.verify_existing.resolve()):
                parser.error("--report must be outside the existing bank")
            if args.report.exists():
                parser.error("--report must name a new file")
        start = time.perf_counter()
        report = verify_bank(args.verify_existing, portable=args.portable)
        report["verified_regeneration_seconds"] = time.perf_counter() - start
        if args.report is not None:
            with args.report.open("x") as file:
                json.dump(report, file, indent=2, allow_nan=False)
                file.write("\n")
    else:
        if args.portable or args.report is not None:
            parser.error("--portable and --report require --verify-existing")
        config_path = args.config or Path("configs/tasks/hkde_option_bank.json")
        config = json.loads(config_path.read_text())
        start = time.perf_counter()
        option_bank.write_bank(args.out, config)
        written = time.perf_counter()
        report = verify_bank(args.out)
        report.update(generation_write_seconds=written - start,
                      verified_regeneration_seconds=time.perf_counter() - written)
        option_bank._write_json(args.out / "verification.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
