"""Read the small, cached HKDE data walkthrough; no inference or pricing here."""

import hashlib
import json
from pathlib import Path

from .local import ArtifactError, load_artifact
from .options import hkde_panel_prices


def load_teaching_example(path):
    """Return (panel, report, arrays) after checking the local replay bundle.

    Cached draws belong to the exact stored observation and scientific task.
    Missing or inconsistent files fail; there is no download/retraining fallback.
    This is a provisional validation example, not a statistically accepted fit.
    """
    import numpy as np

    path = Path(path)
    report = json.loads((path / "bundle.json").read_text())
    expected_files = {"task.json", "panel/manifest.json", "panel/observations.csv", "posterior.npz"}
    if report.get("format") != "hkde_teaching_v1" or set(report.get("files", {})) != expected_files:
        raise ArtifactError("expected the complete hkde_teaching_v1 replay bundle")
    for name, digest in report["files"].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != digest:
            raise ArtifactError(f"teaching bundle checksum mismatch: {name}")
    if (report["checkpoint_status"] != "provisional"
            or report["scientific_validation_passed"] is not False
            or report["reserved_test_evaluated"] is not False):
        raise ArtifactError("this replay must retain the published provisional validation status")
    task = json.loads((path / "task.json").read_text())
    panel = load_artifact(path / "panel")
    prices = np.asarray(hkde_panel_prices(panel, checkpoint_task=task))
    if (panel.metadata["data_origin"] != "synthetic" or panel.metadata["purpose"] != "validation"
            or panel.metadata["provenance"]["case_id"] != report["case_id"]
            or panel.metadata["provenance"]["bank_manifest_sha256"] != report["sources"]["bank_manifest_sha256"]
            or report["parameter_names"] != task["parameter_names"]):
        raise ArtifactError("cached example identity/task differs from the local panel")
    target = report["target_contract"]
    if (set(target) != {"strike", "maturity", "option_type"}
            or target["option_type"] != "put"
            or not np.isfinite([target["strike"], target["maturity"]]).all()
            or min(target["strike"], target["maturity"]) <= 0
            or target in task["context"]["contracts"]):
        raise ArtifactError("the downstream target must be a separate European put")
    with np.load(path / "posterior.npz", allow_pickle=False) as saved:
        arrays = dict(saved)
    count = report["cached_draw_count"]
    shapes = {"parameter_draws": (count, 2), "model_price_draws": (count,),
              "conditioning_prices": (9,)}
    if not isinstance(count, int) or isinstance(count, bool) or count < 1 or arrays.keys() != shapes.keys():
        raise ArtifactError("expected paired posterior draws and their conditioning prices")
    for name, shape in shapes.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ArtifactError(f"invalid cached {name}")
    if not np.array_equal(prices, arrays["conditioning_prices"]):
        raise ArtifactError("cached posterior belongs to a different observed panel")
    bounds = np.asarray(task["prior_bounds"])
    if (np.any(arrays["parameter_draws"] < bounds[:, 0])
            or np.any(arrays["parameter_draws"] > bounds[:, 1])):
        raise ArtifactError("cached parameter draws lie outside the checkpoint prior")
    reference = np.asarray(report["reference_price_quantiles"])
    if (report["quantiles"] != [0.05, 0.5, 0.95] or reference.shape != (3,)
            or not np.isfinite(reference).all() or np.any(np.diff(reference) < 0)):
        raise ArtifactError("invalid numerical-reference quantiles")
    return panel, report, arrays
