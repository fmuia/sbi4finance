"""Verify bank integrity/provenance and regenerate selected split-qualified IDs.

Default comparison is exact in the generating environment. --portable uses the
predeclared tolerance from svj.bank for a separate cross-platform check. Neither
mode establishes posterior calibration or scientific acceptance.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket

import numpy as np

from sbi4finance.svj import bank


def check_bank(path, *, portable=False):
    path = Path(path)
    manifest, splits = bank.load_bank(path)
    if bank.source_hashes() != manifest["source_sha256"]:
        raise ValueError("generator source differs from recorded source identity")
    versions = bank.environment_versions()
    if not portable and versions != manifest["versions"]:
        raise ValueError("environment versions differ; use --portable for a numerical comparison")
    checked = {}
    for split, arrays in splits.items():
        count = len(arrays["example_id"])
        ids = sorted({0, count // 2, count - 1}, reverse=True)
        regenerated = bank.generate_examples(manifest["configuration"], split, ids)
        differences, exact = {}, True
        for key, actual in regenerated.items():
            saved = arrays[key] if key == "dt_years" else arrays[key][ids]
            exact = exact and np.array_equal(saved, actual)
            if key == "example_id":
                np.testing.assert_array_equal(saved, actual)
            else:
                differences[key] = float(np.max(np.abs(saved - actual)))
                if portable:
                    np.testing.assert_allclose(saved, actual, **bank.PORTABLE_TOLERANCE)
                else:
                    np.testing.assert_array_equal(saved, actual)
        checked[split] = {"regenerated_ids": regenerated["example_id"].tolist(),
                          "exact_array_match": bool(exact), "max_absolute_difference_by_array": differences}
    return {
        "passed": True, "mode": "portable" if portable else "same-environment exact",
        "tolerance": bank.PORTABLE_TOLERANCE if portable else {"rtol": 0, "atol": 0},
        "task_fingerprint": manifest["task_fingerprint"],
        "manifest_sha256": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest(),
        "source_sha256": manifest["source_sha256"], "generating_versions": manifest["versions"],
        "checking_versions": versions, "platform": platform.platform(), "machine": platform.machine(),
        "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "selected_examples": checked,
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
