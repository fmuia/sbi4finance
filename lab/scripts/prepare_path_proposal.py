"""Freeze a regularized PMMH covariance from existing chains' tune+draw z.

This builds a proposal only. Existing samples do not change the target/prior.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sbi4finance.path_inference.bank import task_fingerprint  # noqa: E402
from sbi4finance.path_inference.pmmh import regularized_covariance  # noqa: E402


def prepare(chains, *, diagonal_shrinkage=0.1, relative_eigenfloor=1e-4):
    chain_dirs = sorted(Path(chains).glob("chain-*"))
    if not chain_dirs:
        raise ValueError("no chain-* directories found")
    pooled, sources, identity = [], [], None
    for directory in chain_dirs:
        report = json.loads((directory / "pmmh-summary.json").read_text())
        current = {"task_sha256": task_fingerprint(report["task"]),
                   "parameter_names": report["task"]["parameter_names"],
                   "length": report["observation"]["length"],
                   "trajectory_id": report["observation"]["trajectory_id"],
                   "source_shard_sha256": report["observation"]["source_shard_sha256"]}
        if identity is not None and current != identity:
            raise ValueError("proposal source chains differ in task or observation identity")
        identity = current
        for phase in ("tune", "draws"):
            path = directory / f"{phase}.npz"
            with np.load(path, allow_pickle=False) as saved:
                z = np.asarray(saved["z"], dtype=float)
                if z.ndim != 3 or z.shape[0] != 1 or z.shape[2] != len(identity["parameter_names"]):
                    raise ValueError("source phase z dimensions differ")
                if z.shape[1]:
                    pooled.append(z[0])
            sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "chain_id": report["chain_id"], "phase": phase, "samples": z.shape[1]})
    covariance, details = regularized_covariance(np.concatenate(pooled),
        diagonal_shrinkage=diagonal_shrinkage, relative_eigenfloor=relative_eigenfloor)
    return {"schema": "path-pmmh-proposal-covariance-v1", "coordinates": "prior_cdf_logits",
            **identity, "covariance": covariance.tolist(), "regularization": details, "sources": sources,
            "scope": "fixed symmetric proposal covariance only; no target/prior change and no convergence claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diagonal-shrinkage", type=float, default=0.1)
    parser.add_argument("--relative-eigenfloor", type=float, default=1e-4)
    args = parser.parse_args()
    result = prepare(args.chains, diagonal_shrinkage=args.diagonal_shrinkage,
                     relative_eigenfloor=args.relative_eigenfloor)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), "pooled_samples": result["regularization"]["pooled_samples"],
                      "regularized_eigenvalues": result["regularization"]["regularized_eigenvalues"]}))


if __name__ == "__main__":
    main()
