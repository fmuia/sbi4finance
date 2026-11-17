"""Prepare exact extra posterior snapshots for the four preserved rough-known cases.

Use the current BayesFlow 2.1 environment and its installed sibling tutorial.
This script performs inference only: it never changes a simulator, prior, model
or old posterior frame. Publish its fresh output directory through studies/accept.py.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from sbi4f_tutorial.trajectories import evidence, evidence_assets, inference, lesson


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output, checkpoint=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    task, base_report, arrays = lesson.load_example("rough-known")
    live_task, workflow, training = lesson.load_live("rough-known", checkpoint)
    if live_task != task or training["model_sha256"] != base_report["checkpoint_sha256"]:
        raise ValueError("installed checkpoint differs from the preserved replay")
    if len(arrays["trajectory_id"]) != 4:
        raise ValueError("all four original cases are required")
    base_lengths = lesson.saved_lengths(arrays)
    days = sorted(set(evidence_assets.schedule()) - set(base_lengths))
    draws, histories = [], []
    for case in range(4):
        started = time.perf_counter()
        saved = evidence.saved_manager(task, arrays, case, base_report["checkpoint_sha256"])
        native = evidence.native_manager(task, arrays["observations"][case], base_report["checkpoint_sha256"],
                                         str(arrays["trajectory_id"][case]), draws=128, seed=20261113100+case)
        timing = evidence.prepare_frames(native, workflow, days, batch_size=8)
        frames = evidence.frame_list(native, task["n_observations"])
        draws.append(np.stack([frame["draws"] for frame in frames]))
        histories.append({"case": case, "trajectory_id": saved["metadata"]["history_id"],
                          "history_sha256": saved["metadata"]["history_sha256"],
                          "history_length": saved["metadata"]["history_length"],
                          "sampling_seed": native["metadata"]["seed"],
                          "query_batches": [{"days": frame["metadata"]["query_batch_days"],
                                              "seed": frame["metadata"]["query_seed"]}
                                             for frame in frames[::8]],
                          "inference_seconds": timing["inference_seconds"],
                          "original_frame_sha256": {str(day): evidence_assets._draws_sha(value["draws"])
                                                      for day, value in saved["frames"].items()}})
        print(f"case {case+1}: {len(frames)} exact new prefixes x 128 paired draws in {time.perf_counter()-started:.2f}s", flush=True)
    payload = output / "frames.npz"
    np.savez_compressed(payload, lengths=np.asarray(days, dtype=np.int64), draws=np.stack(draws),
                        trajectory_id=arrays["trajectory_id"])
    catalogue = lesson.catalogue()["artifacts"]["rough-known"]
    runtime = Path(evidence.__file__).resolve().parent
    bundle = {"format": "trajectory_dense_evidence_v1", "created": datetime.now(timezone.utc).isoformat(),
              "model": "rough-known", "measure": "P", "task": task,
              "task_sha256": base_report["task_sha256"], "checkpoint_sha256": base_report["checkpoint_sha256"],
              "source_replay_bundle_sha256": catalogue["bundle_sha256"],
              "source_split": "validation", "source_case_offset": 0,
              "scientific_acceptance": False, "interpolated": False, "parameter_prior_updated": False,
              "scope": "development posterior playback; exact prefix queries using unchanged supplied weights and original prior; no new calibration claim",
              "base_lengths": base_lengths, "all_lengths": evidence_assets.schedule(),
              "posterior_draws": 128, "query_batch_size": 8,
              "seed_mapping": inference.BACKEND_SEED_MAPPING, "histories": histories,
              "files": {"frames.npz": sha(payload)},
              "runtime_sha256": {name: sha(runtime/name) for name in ("evidence.py", "inference.py", "simulation.py")}}
    (output / "bundle.json").write_text(json.dumps(bundle, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(output), "bundle_sha256": sha(output/"bundle.json"),
                      "new_frames_per_case": len(days), "total_frames_per_case": len(evidence_assets.schedule()),
                      "payload_bytes": payload.stat().st_size}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="matching native cache inside tutorial/assets/local-checkpoints")
    args = parser.parse_args()
    prepare(args.output, args.checkpoint)
