"""Read compact fixed-checkpoint market predictives without private payloads."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_digest(value):
    return sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def load_results(root, *, evaluation, task, observations, half_spread, reference_sha256):
    """Verify task, checkpoint, quote ordering/error profile and reference bindings."""
    record = json.loads((Path(root) / "results.json").read_text())
    target = task.get("target", task)
    sources = record["sources"]
    expected = {"evaluation_fingerprint": fingerprint(evaluation), "task_fingerprint": fingerprint(target),
                "conditional_draws_sha256": evaluation["conditional_draws"]["sha256"],
                "checkpoint_sha256": evaluation["checkpoint"]["sha256"],
                "training_run_sha256": evaluation["training_run"]["sha256"],
                "common_bank_record_sha256": evaluation["common_bank"]["sha256"],
                "observation_sha256": array_digest(observations), "half_spread_sha256": array_digest(half_spread),
                "reference_cache_sha256": reference_sha256}
    if (record.get("format") != "heston_actual63_market_predictives_v1"
            or record["checkpoint"] != "300k-300-final300" or record["seed"] != 20261113
            or record["draws_per_method"] != 256 or set(record["methods"]) != {"pymc", "amortizer"}
            or target != evaluation["task"] or record["parameter_names"] != target["parameter_names"]
            or record["prior_bounds"] != target["prior_bounds"]
            or record["quote_count"] != len(observations) or len(half_spread) != len(observations)
            or any(sources.get(key) != value for key, value in expected.items())
            or sources["observation_sha256"] != evaluation["cases"]["market"]["observation_sha256"]):
        raise ValueError("saved market predictives differ from their task, sources or observations")
    for name, method in record["methods"].items():
        for key in ("clean_q05_q50_q95_raw", "noisy_q05_q50_q95_raw"):
            q = np.asarray(method[key], dtype=float)
            if q.shape != (3, len(observations)) or not np.isfinite(q).all() or np.any(np.diff(q, axis=0) < 0):
                raise ValueError("three finite ordered price quantiles per contract required")
        q = np.asarray(method["c_q05_q50_q95"], dtype=float)
        density, edges = (np.asarray(method["c_histogram"][key], dtype=float) for key in ("density", "edges"))
        if (q.shape != (3,) or not np.isfinite(q).all() or np.any(q <= 0) or np.any(np.diff(q) < 0)
                or density.shape != (40,) or edges.shape != (41,) or np.any(np.diff(edges) <= 0)
                or not np.isfinite(density).all() or not np.isfinite(edges).all() or np.any(density < 0)
                or not np.isclose(np.sum(density * np.diff(edges)), 1.)
                or method["priced_draw_count"] != 256 or method["raw_draw_count"] < 256
                or method["outside_pricing_bounds_count"] != 0 or method["invalid_noise_scale_count"] != 0):
            raise ValueError("invalid raw noise marginal or pricing inclusion record")
    neural = record["methods"]["amortizer"]
    saved = evaluation["cases"]["market"]
    if (neural["raw_draw_count"] != evaluation["conditional_draws"]["draws_per_case"]
            or neural["outside_prior_count"] != saved["support"]["outside_prior"]["sample_count"]
            or neural["outside_prior_per_parameter"] != saved["support"]["outside_prior"]["per_dimension_count"]
            or neural["c_q05_q50_q95"] != np.asarray(saved["comparison"]["neural_quantiles_05_50_95"])[:, -1].tolist()
            or not record["methods"]["pymc"]["reference_chain_balanced"]):
        raise ValueError("saved raw neural support/quantiles or PyMC pairing differ")
    return record
