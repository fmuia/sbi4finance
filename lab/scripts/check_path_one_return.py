"""Compare a joint rough checkpoint with its independent exact one-return posterior."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sbi4finance.path_inference import bank, inference, one_return  # noqa: E402


def run(args):
    started = time.monotonic()
    task, workflow, training = inference.load_checkpoint(args.checkpoint)
    manifest = bank.load_manifest(args.bank)
    if (inference.fingerprint(manifest["task"]) != inference.fingerprint(task)
            or training["bank_manifest_sha256"] != inference.sha256(args.bank / "manifest.json")):
        raise ValueError("exact one-return comparison requires the checkpoint's task and bank identity")
    split = bank.load_split(args.bank, "validation")
    offset = inference.integer(getattr(args, "offset", 0), "offset", minimum=0)
    stop = min(offset + inference.integer(args.cases, "cases"), len(split["observations"]))
    count = stop - offset
    if count < 1:
        raise ValueError("validation split is empty")
    inference.integer(args.batch_size, "batch size")
    split = {key: value[offset:stop] if isinstance(value, np.ndarray) else value
             for key, value in split.items()}
    batches = [inference.sample(workflow, task, split["observations"][start:min(count, start + args.batch_size), :1],
                               args.draws, seed=args.seed + offset + start)
               for start in range(0, count, args.batch_size)]
    draws = np.concatenate(batches)
    report, exact = one_return.compare(draws, split["parameters"][:count],
                                      split["observations"][:count, 0, 0], task,
                                      nodes=args.quadrature_nodes)
    report.update(format="path_exact_one_return_check_v1", task_sha256=inference.fingerprint(task),
                  checkpoint_sha256=training["model_sha256"], split="validation",
                  reserved_test_evaluated=False, seed=args.seed, offset=offset,
                  backend_seed_mapping=inference.BACKEND_SEED_MAPPING,
                  prior_checkpoint_development_pool_size=training.get("validation_cases_per_length"),
                  case_identities_sha256=inference.fingerprint(split["trajectory_id"].tolist()),
                  elapsed_seconds=time.monotonic() - started)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "one-return.npz", draws=draws, truth=split["parameters"][:count],
                        first_returns=split["observations"][:count, 0, 0],
                        trajectory_id=split["trajectory_id"][:count], **exact)
    report["arrays_sha256"] = inference.sha256(args.out / "one-return.npz")
    (args.out / "one-return.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=512)
    parser.add_argument("--draws", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--quadrature-nodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261030091)
    parser.add_argument("--offset", type=int, default=0)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
