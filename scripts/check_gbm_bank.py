"""Verify a saved bank and regenerate selected IDs in a different order.

Default: exact comparison in the generating environment. --portable permits
tiny floating-point differences for a separately reported cross-platform check.
This is a small smoke verifier, not a cluster orchestration or benchmark layer.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket

import numpy as np

from sbi4finance import gbm_bank


def check_bank(path, *, portable=False):
    path = Path(path)
    manifest, splits = gbm_bank.load_bank(path)
    if set(manifest["source_sha256"]) != {"gbm.py", "gbm_bank.py"}:
        raise ValueError("incomplete generator source identity")
    for name, expected in manifest["source_sha256"].items():
        actual = hashlib.sha256(Path(gbm_bank.__file__).with_name(name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"generator source differs: {name}")
    current_versions = {"numpy": np.__version__, "python": platform.python_version()}
    if not portable and current_versions != manifest["versions"]:
        raise ValueError("environment versions differ; use --portable for a numerical comparison")
    checked = {}
    for split, arrays in splits.items():
        count = len(arrays["example_id"])
        ids = sorted({0, count // 2, count - 1}, reverse=True)
        regenerated = gbm_bank.generate_examples(manifest["configuration"], split, ids)
        maximum_difference = 0.0
        exact = True
        for key in ("observations", "parameters", "dt_years", "example_id"):
            saved = arrays[key] if key == "dt_years" else arrays[key][ids]
            actual = regenerated[key]
            equal = np.array_equal(saved, actual)
            exact = exact and equal
            if key == "example_id":
                np.testing.assert_array_equal(saved, actual)
            else:
                maximum_difference = max(maximum_difference, float(np.max(np.abs(saved - actual))))
                if portable:
                    np.testing.assert_allclose(saved, actual, rtol=1e-12, atol=1e-14)
                else:
                    np.testing.assert_array_equal(saved, actual)
        checked[split] = {"regenerated_ids": regenerated["example_id"].tolist(),
                          "exact_array_match": exact, "max_absolute_difference": maximum_difference}
    return {
        "passed": True, "mode": "portable" if portable else "same-environment exact",
        "tolerance": {"rtol": 1e-12, "atol": 1e-14} if portable else {"rtol": 0, "atol": 0},
        "manifest_sha256": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest(),
        "source_sha256": manifest["source_sha256"], "generating_versions": manifest["versions"],
        "checking_versions": current_versions, "platform": platform.platform(),
        "machine": platform.machine(), "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "selected_examples": checked,
        "evidence_scope": "S: bank integrity and selected example reproduction only",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--portable", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check_bank(args.bank, portable=args.portable), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
