"""Generate a new local/CPU-cluster option bank and regenerate selected IDs."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from sbi4finance import option_bank


def verify_bank(path):
    manifest, splits = option_bank.load_bank(path)
    for name, expected in manifest["source_sha256"].items():
        if option_bank._sha256(Path(option_bank.__file__).with_name(name)) != expected:
            raise ValueError(f"generator source changed: {name}")
    checked = {}
    for split, arrays in splits.items():
        count = len(arrays["example_id"])
        ids = sorted({0, count // 2, count - 1}, reverse=True)
        regenerated = option_bank.generate_examples(manifest["configuration"], split, ids)
        for key in arrays:
            np.testing.assert_array_equal(arrays[key][ids], regenerated[key])
        checked[split] = ids
    sigma = manifest["configuration"]["task"]["context"]["quote_noise_std"]
    return {"passed": True, "scope": "S: storage and same-environment selected-ID regeneration",
            "regenerated_ids": checked, "manifest_sha256": option_bank._sha256(Path(path) / "manifest.json"),
            "loaded_array_bytes": sum(a.nbytes for split in splits.values() for a in split.values()),
            "max_mc_se_over_quote_sigma": max(float(a["pricing_standard_errors"].max()) / sigma for a in splits.values()),
            "max_abs_fourier_refinement_over_quote_sigma": max(float(np.abs(a["refinement_difference"]).max()) / sigma for a in splits.values()),
            "numerical_bias_certified": False, "posterior_validated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tasks/hkde_option_bank.json"))
    parser.add_argument("--out", type=Path, required=True, help="new directory; never overwritten")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    start = time.perf_counter()
    option_bank.write_bank(args.out, config)
    written = time.perf_counter()
    report = verify_bank(args.out)
    report.update(generation_write_seconds=written - start,
                  verified_regeneration_seconds=time.perf_counter() - written)
    option_bank._write_json(args.out / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
