"""Actual posterior frames for one frozen history; animation performs no inference.

Saved frames are real cached snapshots. Native frames are prepared only through
an explicit action, always under the original task/prior and with future returns
masked. There is no interpolation, posterior-as-prior update or state forecast.
"""

import copy
import hashlib
import time

import numpy as np

from . import inference, simulation


FORMAT = "trajectory_evidence_manager_v1"
MAX_PREPARED_FRAMES = 61


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _history(values, maximum):
    _require(not np.iscomplexobj(values) and not np.ma.isMaskedArray(values),
             "provide one real unmasked return history")
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    _require(values.ndim == 2 and values.shape[1] == 1 and values.dtype.kind in "fiu"
             and 1 <= len(values) <= maximum, "provide one return history [time,1] within the task horizon")
    # A future suffix may be unavailable/nonfinite. It is never inspected by a
    # prefix query; any nonfinite value inside a revealed prefix is an error.
    result = np.asarray(values, dtype="<f8").copy()
    result.setflags(write=False)
    return result


def _history_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _manager(task, observed, checkpoint_id, history_id, source, draws, seed):
    simulation.validate_task(task)
    for name, value in (("checkpoint_id", checkpoint_id), ("history_id", history_id)):
        _require(isinstance(value, str) and bool(value.strip()), f"a nonempty {name} is required")
    history = _history(observed, task["n_observations"])
    metadata = {"format": FORMAT, "task_sha256": simulation.task_fingerprint(task),
                "checkpoint_id": checkpoint_id, "history_id": history_id,
                "history_sha256": _history_hash(history), "history_length": len(history), "source": source,
                "draw_count": draws, "seed": seed}
    return {"task": copy.deepcopy(task), "observations": history, "metadata": metadata,
            "context_id": inference.fingerprint(metadata), "frames": {}}


def _validate(manager):
    _require(isinstance(manager, dict) and set(manager) ==
             {"task", "observations", "metadata", "context_id", "frames"},
             "expected a complete evidence-frame manager")
    simulation.validate_task(manager["task"])
    metadata = manager["metadata"]
    history = np.asarray(manager["observations"])
    _require(history.ndim == 2 and history.shape[1] == 1
             and 1 <= len(history) <= manager["task"]["n_observations"]
             and metadata.get("history_length") == len(history), "frozen evidence history axes differ")
    _require(metadata.get("format") == FORMAT
             and metadata.get("task_sha256") == simulation.task_fingerprint(manager["task"])
             and metadata.get("history_sha256") == _history_hash(manager["observations"])
             and manager["context_id"] == inference.fingerprint(metadata),
             "evidence context changed: task, checkpoint, history, source or sampling design differs")
    return metadata


def _day(manager, day, *, check_values=True):
    day = simulation._integer(day, "revealed day", minimum=1)
    _require(day <= len(manager["observations"]), "revealed day exceeds the frozen observed history")
    if check_values:
        _require(np.isfinite(manager["observations"][:day]).all(), "revealed prefix contains nonfinite returns")
    return day


def _draws(task, values):
    _require(not np.iscomplexobj(values) and not np.ma.isMaskedArray(values),
             "posterior frames require real unmasked paired parameter rows")
    values = np.asarray(values)
    dimensions = len(task["parameter_names"])
    _require(values.ndim == 2 and values.shape[1] == dimensions and len(values) >= 2
             and values.dtype.kind in "fiu" and np.isfinite(values).all(),
             "posterior frame must contain finite complete paired rows [draws>=2,parameters]")
    bounds = np.asarray(task["prior_bounds"])
    _require(np.all(values >= bounds[:, 0]) and np.all(values <= bounds[:, 1]),
             "posterior frame lies outside the original task prior support")
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def _frame(manager, length, draws, details=None):
    day = _day(manager, length)
    rows = _draws(manager["task"], draws)
    metadata = manager["metadata"]
    identity = {"context_id": manager["context_id"], "length": day}
    return {"length": day, "draws": rows, "source": metadata["source"],
            "frame_id": inference.fingerprint(identity), "metadata": {
                **metadata, **identity, "actual_length": day, "posterior_draws": len(rows),
                "scope": "original prior conditioned on exactly this observed prefix; no interpolation or continuation",
                **({} if details is None else details)}}


def saved_manager(task, arrays, case, checkpoint_id):
    """Prepare the actual saved snapshots for one asset-validated replay case."""
    case = simulation._integer(case, "saved case", minimum=0)
    observations = np.asarray(arrays["observations"])
    ids = np.asarray(arrays["trajectory_id"])
    _require(observations.ndim == 3 and observations.shape[1:] == (task["n_observations"], 1)
             and ids.shape == (len(observations),) and case < len(observations),
             "saved history/case axes differ from the selected task")
    result = _manager(task, observations[case], checkpoint_id, str(ids[case]), "saved", None, None)
    for key in sorted((k for k in arrays if k.startswith("draws_")), key=lambda k: int(k.removeprefix("draws_"))):
        day = int(key.removeprefix("draws_"))
        values = np.asarray(arrays[key])
        _require(values.ndim == 3 and values.shape[0] == len(observations), "saved posterior case axes differ")
        result["frames"][day] = _frame(result, day, values[case], {"origin": "asset-validated saved snapshot"})
    _require(bool(result["frames"]), "the replay has no actual posterior frames")
    return result


def native_manager(task, observed, checkpoint_id, history_id, draws=128, seed=20261113):
    """Bind explicit native queries to one task, checkpoint and immutable history."""
    draws = simulation._integer(draws, "posterior frame draws", minimum=2)
    _require(draws <= 512, "posterior evidence frames are capped at 512 draws each")
    seed = simulation._integer(seed, "frame sampling seed")
    return _manager(task, observed, checkpoint_id, history_id, "native", draws, seed)


def exact_schedule(target, step=1, max_frames=MAX_PREPARED_FRAMES):
    """Propose exact query days, including one and target; never silently coarsen."""
    target = simulation._integer(target, "target day", minimum=1)
    step = simulation._integer(step, "query spacing", minimum=1)
    max_frames = simulation._integer(max_frames, "frame limit", minimum=1)
    _require(max_frames <= MAX_PREPARED_FRAMES, f"one preparation action is capped at {MAX_PREPARED_FRAMES} frames")
    needed = (target-1)//step + 1 + int((target-1) % step != 0)
    _require(needed <= max_frames,
             f"this schedule needs {needed} frames; choose a shorter reveal horizon or wider spacing (limit {max_frames})")
    days = sorted(set([*range(1, target + 1, step), target]))
    return days


def prepare_frames(manager, workflow, days, batch_size=8):
    """Explicitly query missing exact prefixes, then publish the successful batch.

    Future values are zeroed before even reaching the inference helper; its
    canonical mask and time conditions are constructed from the exact lengths.
    Existing samples are retained. Failure publishes no partial new frames.
    """
    metadata = _validate(manager)
    _require(metadata["source"] == "native", "fresh inference needs a separate native evidence context")
    batch_size = simulation._integer(batch_size, "query batch size", minimum=1)
    _require(batch_size <= 16, "query batches are capped at 16 prefixes")
    requested = [_day(manager, n) for n in days]
    _require(bool(requested) and len(requested) == len(set(requested)), "provide distinct exact query days")
    _require(len(requested) <= MAX_PREPARED_FRAMES, f"one preparation action is capped at {MAX_PREPARED_FRAMES} frames")
    requested = sorted(requested)
    missing = [n for n in requested if n not in manager["frames"]]
    staged = {}
    started = time.perf_counter()
    for index, start in enumerate(range(0, len(missing), batch_size)):
        selected = missing[start:start+batch_size]
        values = np.zeros((len(selected), manager["task"]["n_observations"], 1))
        for row, day in enumerate(selected):
            values[row, :day] = manager["observations"][:day]
        seed = (metadata["seed"] + index) % (2**64)
        draws = inference.sample(workflow, manager["task"], values, metadata["draw_count"],
                                 lengths=np.asarray(selected, dtype=int), seed=seed)
        expected = (len(selected), metadata["draw_count"], len(manager["task"]["parameter_names"]))
        _require(np.shape(draws) == expected, "native evidence query returned unexpected posterior axes")
        for row, day in enumerate(selected):
            staged[day] = _frame(manager, day, draws[row], {
                "origin": "explicit exact-prefix native query", "query_seed": seed,
                "query_batch_days": list(selected)})
    manager["frames"].update(staged)
    return {"new_lengths": missing, "reused_lengths": [n for n in requested if n not in missing],
            "inference_seconds": time.perf_counter()-started, "context_id": manager["context_id"]}


def frame_list(manager, target):
    """Copies of actual frames at or before the target, ready for local playback."""
    _validate(manager)
    target = _day(manager, target, check_values=False)
    frames = []
    for day in sorted(manager["frames"]):
        if day > target:
            continue
        value = manager["frames"][day]
        expected = inference.fingerprint({"context_id": manager["context_id"], "length": day})
        _require(value["frame_id"] == expected and value["metadata"]["context_id"] == manager["context_id"],
                 "posterior frame belongs to another task, checkpoint or history")
        frames.append({**value, "draws": value["draws"].copy(), "metadata": copy.deepcopy(value["metadata"])})
    return frames


def select_frame(manager, requested):
    """Use an exact frame or label the nearest actual past frame; never interpolate."""
    available = frame_list(manager, requested)
    _require(bool(available), "no actual posterior frame is available at or before this day")
    frame = available[-1]
    actual = frame["length"]
    exact = actual == requested
    return {**frame, "requested_length": requested, "exact": exact,
            "observed_prefix": manager["observations"][:actual].copy(),
            "label": (f"Actual posterior at {actual} observed days" if exact else
                      f"Requested {requested} days; nearest actual past posterior uses {actual} days")}


def add_saved_frames(manager, frames, *, task_sha256, checkpoint_id, history_id, history_sha256):
    """Add an externally validated denser replay only when every identity matches.

    ``frames`` maps actual observed days to paired parameter samples. Artifact
    manifest/hash validation belongs to its asset loader, before this function.
    """
    metadata = _validate(manager)
    _require(metadata["source"] == "saved", "cached replay frames need a saved evidence context")
    given = {"task_sha256": task_sha256, "checkpoint_id": checkpoint_id,
             "history_id": history_id, "history_sha256": history_sha256}
    _require(all(metadata[key] == value for key, value in given.items()),
             "dense replay task, checkpoint or observed history differs")
    staged = {_day(manager, day): _frame(manager, day, draws, {"origin": "asset-validated dense saved snapshot"})
              for day, draws in frames.items()}
    reused = sorted(set(staged) & set(manager["frames"]))
    missing = sorted(set(staged) - set(manager["frames"]))
    # Overlapping snapshots remain the original paired samples. A dense replay
    # supplies extra days, never silently replaces a previously displayed frame.
    manager["frames"].update({day: staged[day] for day in missing})
    return {"new_lengths": missing, "reused_lengths": reused}
