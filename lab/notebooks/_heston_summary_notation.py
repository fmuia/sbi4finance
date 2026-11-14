"""Validate display-only recovery figures derived from the frozen summary study."""

from hashlib import sha256
import json
from pathlib import Path


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_figures(root, source_root, record, labels):
    """Bind relabelled PNGs to the original study, images, draws and metrics."""
    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("format") != "heston_summary_notation_figures_v1"
            or manifest.get("display_only") is not True
            or manifest["labels"] != list(labels)
            or manifest["parameter_names"] != record["protocol"]["parameter_names"]
            or manifest["source_results_sha256"] != digest(source_root / "results.json")
            or manifest["source_recovery_manifest_sha256"] != record["producer"]["recovery_manifest_sha256"]
            or manifest["rows"] != 512 or manifest["draws_per_case"] != 512):
        raise ValueError("notation figures differ from the frozen study or current display labels")
    fits = {fit["dimension"]: fit for fit in record["fits"]}
    figures = {}
    for item in manifest["figures"]:
        dimension = item["dimension"]
        if dimension not in fits or dimension in figures:
            raise ValueError("each frozen final checkpoint needs one notation figure")
        fit = fits[dimension]
        original = fit["recovery_figure"]
        path = (root / item["file"]).resolve()
        if Path(item["file"]).is_absolute() or root not in path.parents:
            raise ValueError("notation figures must remain in their data directory")
        if (item["sha256"] != digest(path)
                or item["source_archived_image_sha256"] != original["sha256"]
                or digest(source_root / original["file"]) != original["sha256"]
                or item["source_draws_sha256"] != original["source_draws_sha256"]
                or item["truth_sha256"] != record["identities"]["truth_sha256"]
                or item["observations_sha256"] != record["identities"]["observations_sha256"]
                or any(item["metrics"][key] != fit["endpoints"]["primary_final"]["population"][key]
                       for key in ("rmse", "coverage90", "median_width90"))):
            raise ValueError("notation figure source, image checksum or recovery metric differs")
        figures[dimension] = path
    if set(figures) != set(fits):
        raise ValueError("all ten frozen final checkpoints need notation figures")
    return figures
