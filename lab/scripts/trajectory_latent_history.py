"""Recover paired latent truth for all preserved tutorial histories, with replay proof.

Only the original task/split/case RNG streams are replayed. A numerical mismatch
stops publication. Frozen observations, parameter truths and weights are untouched.
Publish the output with studies/accept.py --run after inspecting the pairing proof.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from sbi4f_tutorial.trajectories import latent_assets, lesson, simulation


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    models, files = {}, {}
    for name in simulation.MODELS:
        task, report, arrays = lesson.load_example(name)
        ids = arrays["trajectory_id"].tolist()
        seed = int(ids[0].split(":")[1])
        expected_ids = [f"{simulation.task_fingerprint(task)[:16]}:{seed}:validation:{case}" for case in range(4)]
        if ids != expected_ids:
            raise ValueError("replay must preserve the original first four identified validation histories")
        replay = simulation.replay_details(task, "validation", 0, 4, seed=seed, backend="numpy")
        if not np.array_equal(replay["parameters"], arrays["truth"]):
            raise ValueError("replayed parameter truths are not identical to the original case truths")
        np.testing.assert_allclose(replay["observations"], arrays["observations"], rtol=2e-12, atol=2e-14,
                                   err_msg="unpaired latent path: original returns cannot be reproduced")
        payload = output / (name+".npz")
        np.savez_compressed(payload, trajectory_id=replay["trajectory_id"], parameters=replay["parameters"],
                            replayed_observations=replay["observations"], **replay["latent"])
        files[payload.name] = sha(payload)
        histories = [{"case": case, "case_id": ids[case],
                      "observation_sha256": latent_assets._array_sha(arrays["observations"][case]),
                      "parameter_sha256": latent_assets._array_sha(arrays["truth"][case]),
                      "parameters_byte_identical": True,
                      "returns_byte_identical": bool(np.array_equal(replay["observations"][case], arrays["observations"][case])),
                      "max_abs_return_error": float(np.max(np.abs(replay["observations"][case]-arrays["observations"][case])))}
                     for case in range(4)]
        models[name] = {"task": task, "task_sha256": report["task_sha256"], "case_ids": ids,
                        "source_replay_bundle_sha256": lesson.catalogue()["artifacts"][name]["bundle_sha256"],
                        "source_posterior_file_sha256": report["files"]["posterior.npz"],
                        "source_split": "validation", "source_case_offset": 0, "seed": seed,
                        "histories": histories}
        print(json.dumps({"model": name, "case_count": 4, "history_days": task["n_observations"],
                          "all_parameters_identical": True,
                          "max_abs_return_error": max(row["max_abs_return_error"] for row in histories)}), flush=True)
    runtime = Path(simulation.__file__).resolve().parent
    bundle = {"format": "trajectory_paired_latent_v1", "created": datetime.now(timezone.utc).isoformat(),
              "scientific_acceptance": False, "inference_input": False, "state_estimate": False,
              "preserves_observed_history": True, "observed_day_boundaries": True,
              "scope": "actual hidden generator paths paired with original synthetic histories; no filtering, posterior-state estimate or new calibration claim",
              "replay_tolerance": {"rtol": 2e-12, "atol": 2e-14}, "backend": "numpy",
              "models": models, "files": files,
              "runtime_sha256": {"trajectories/simulation.py": sha(runtime/"simulation.py"),
                                  "heston_options/simulator.py": sha(runtime.parent/"heston_options"/"simulator.py"),
                                  "rough_options/simulator.py": sha(runtime.parent/"rough_options"/"simulator.py")}}
    (output/"bundle.json").write_text(json.dumps(bundle, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(output), "bundle_sha256": sha(output/"bundle.json"),
                      "payload_bytes": sum((output/name).stat().st_size for name in files)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
