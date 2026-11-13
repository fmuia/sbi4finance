"""Verified extra posterior frames for the four preserved rough-known histories.

These are development examples from unchanged weights, not new calibration
evidence. Other model replays keep their original actual snapshots.
"""

import hashlib
import json

import numpy as np

from ..shared.paths import ASSETS
from . import evidence, lesson, simulation


BUNDLE_SHA256 = "cd4d9b1eb3097aed74c91e7e159ffcd9c0a89adc1d3f26ee9338b5d44a821f47"
ASSET_FOLDER = "evidence-rough-known-v1"


def schedule():
    """Daily early evidence, then weekly, monthly and quarterly exact prefixes."""
    original = {1, 2, 4, 8, 17, 21, 63, 126, 252, 504, 756, 1260}
    return sorted(original | set(range(1, 22)) | set(range(26, 127, 5)) |
                  set(range(147, 253, 21)) | set(range(315, 1261, 63)))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _draws_sha(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def load_dense(manager):
    """Attach bound extra days without replacing any original paired samples."""
    metadata = evidence._validate(manager)
    if manager["task"]["model"] != "rough-known" or metadata["source"] != "saved":
        return manager
    folder = (ASSETS / "trajectories" / ASSET_FOLDER).resolve()
    _require(folder.is_relative_to(ASSETS.resolve()), "dense evidence must remain under tutorial/assets")
    bundle_path = (folder / "bundle.json").resolve()
    _require(bundle_path.is_relative_to(ASSETS.resolve()), "dense evidence bundle must remain under tutorial/assets")
    _require(bundle_path.is_file() and _sha(bundle_path) == BUNDLE_SHA256,
             "dense evidence bundle is missing or differs from its pinned identity")
    bundle = json.loads(bundle_path.read_text())
    _require(bundle.get("format") == "trajectory_dense_evidence_v1"
             and bundle.get("model") == "rough-known" and bundle.get("measure") == "P"
             and bundle.get("scientific_acceptance") is False
             and bundle.get("source_split") == "validation" and bundle.get("source_case_offset") == 0
             and bundle.get("interpolated") is False and bundle.get("parameter_prior_updated") is False
             and bundle.get("files", {}).keys() == {"frames.npz"},
             "dense evidence format or scientific scope differs")
    _require(bundle.get("task") == manager["task"]
             and bundle.get("task_sha256") == metadata["task_sha256"]
             == simulation.task_fingerprint(bundle["task"])
             and bundle.get("checkpoint_sha256") == metadata["checkpoint_id"],
             "dense evidence task or checkpoint differs from this context")
    base = lesson.catalogue()["artifacts"]["rough-known"]
    _require(bundle.get("source_replay_bundle_sha256") == base["bundle_sha256"],
             "dense evidence was generated for a different original replay")
    payload = (folder / "frames.npz").resolve()
    _require(payload.is_relative_to(ASSETS.resolve()), "dense evidence arrays must remain under tutorial/assets")
    _require(payload.is_file() and _sha(payload) == bundle["files"]["frames.npz"],
             "dense evidence arrays differ from their bound file hash")
    with np.load(payload, allow_pickle=False) as saved:
        _require(set(saved.files) == {"lengths", "draws", "trajectory_id"}, "unexpected dense evidence array fields")
        lengths, draws, ids = saved["lengths"], saved["draws"], saved["trajectory_id"]
    histories = bundle.get("histories", [])
    _require(len(histories) == 4 and [row.get("case") for row in histories] == list(range(4))
             and ids.shape == (4,) and ids.dtype.kind in "US"
             and ids.tolist() == [row.get("trajectory_id") for row in histories]
             and len(set(ids.tolist())) == 4,
             "dense evidence must retain all four preserved validation histories in order")
    base_lengths = bundle.get("base_lengths", [])
    expected_lengths = sorted(set(schedule()) - set(base_lengths))
    _require(base_lengths == [1, 2, 4, 8, 17, 21, 63, 126, 252, 504, 756, 1260]
             and bundle.get("all_lengths") == schedule()
             and lengths.dtype.kind in "iu" and lengths.ndim == 1
             and lengths.tolist() == expected_lengths,
             "dense evidence prefix schedule differs; frames are never interpolated or relabelled")
    count = bundle.get("posterior_draws")
    _require(count == 128 and draws.shape == (4, len(lengths), count, len(manager["task"]["parameter_names"]))
             and draws.dtype.kind == "f" and np.isfinite(draws).all(),
             "dense evidence paired-draw axes or finite values differ")
    bounds = np.asarray(manager["task"]["prior_bounds"])
    _require(np.all(draws >= bounds[:, 0]) and np.all(draws <= bounds[:, 1]),
             "dense evidence draws lie outside the original physical prior support")
    selected = [i for i, value in enumerate(ids.tolist()) if value == metadata["history_id"]]
    _require(len(selected) == 1, "this history is not one of the dense replay's preserved cases")
    case = selected[0]
    history = histories[case]
    _require(history.get("history_sha256") == metadata["history_sha256"]
             and history.get("history_length") == metadata["history_length"],
             "dense evidence observed history bytes differ")
    _require(set(history.get("original_frame_sha256", {})) == {str(day) for day in base_lengths},
             "original paired-frame identities are incomplete")
    for day in base_lengths:
        _require(day in manager["frames"] and _draws_sha(manager["frames"][day]["draws"])
                 == history["original_frame_sha256"][str(day)],
                 "dense evidence cannot replace or relabel an original saved posterior")
    evidence.add_saved_frames(manager, {int(day): draws[case, index] for index, day in enumerate(lengths)},
                              task_sha256=bundle["task_sha256"], checkpoint_id=bundle["checkpoint_sha256"],
                              history_id=history["trajectory_id"], history_sha256=history["history_sha256"])
    return manager
