"""Export standalone Matplotlib figures from saved100k endpoint diagnostics.

Pure saved-data consumers from the archived10k exporter provide the scientific
plot layout. This version validates new endpoint identities and labels training
size correctly. No model loading, pricing, posterior queries or chain reads.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import export_heston_pr87_reproduction_figures as plotting
from scripts import render_heston_pr87_100k as saved

binding, read = saved.binding, saved.read


def build(protocol_path, evaluation_path, out, *, secondary_path=None):
    protocol_path, evaluation_path, out = map(lambda value: Path(value).resolve(), (protocol_path, evaluation_path, out))
    if out.exists(): raise FileExistsError("100k scientific figure identity must be new")
    protected = {}
    protocol = saved.checked(binding(protocol_path), protected)
    primary = saved.checked(binding(evaluation_path), protected)
    training = saved.validate_record(primary, protocol, protocol_path, protected, primary=True)
    records = [("primary", primary, evaluation_path)]
    if (training.get("best_validation_archive", {}).get("epoch") != 200) != (secondary_path is not None):
        raise ValueError("same full secondary panel required when predeclared minimum differs")
    if secondary_path is not None:
        secondary_path = Path(secondary_path).resolve()
        secondary = saved.checked(binding(secondary_path), protected)
        saved.validate_record(secondary, protocol, protocol_path, protected, primary=False)
        if secondary["training_run"] != primary["training_run"]:
            raise ValueError("same training identity required for both endpoints")
        records.append(("secondary", secondary, secondary_path))
    training_dir = Path(primary["training_run"]["path"]).parent
    curve_binding = binding(training_dir / "native-epoch-history.json")
    if training.get("native_epoch_history_sha256") != curve_binding["sha256"]:
        raise ValueError("native objective history differs from training")
    curve = saved.checked(curve_binding, protected)
    out.mkdir(parents=True)
    exports = {}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
            "axes.spines.right": False, "svg.fonttype": "none", "savefig.transparent": False}):
        fig = plotting.learning_figure(curve)
        fig.axes[0].set_xlabel("Native fit_offline epoch over the same 100,000 noisy datasets")
        exports["native-learning-curves"] = plotting.figure_files(fig, out, "native-learning-curves",
            "100k native training and validation objective", "20 million presentations; not equal compute with the 2-million-presentation 10k parent.")
        for endpoint, record, _ in records:
            for panel, label, interval_rows in (("native_validation", "native", np.arange(0, 100, 3)),
                    ("population", "historical", np.arange(0, 512, 16))):
                summary = record[panel]
                recovery = saved.load_recovery(summary, protected)
                if recovery is None:
                    exports[endpoint + "-" + label] = {"status": "undefined_nonfinite_raw_draws", "raw_draws_retained": True}
                    continue
                name = endpoint + "-" + label + "-recovery"
                title = f'100k {endpoint}, epoch{record["native_epoch"]}: {summary["rows"]} datasets × {summary["draws_per_case"]} raw draws'
                fig, details = plotting.recovery_figure(recovery, record["task"], title=title, interval_rows=interval_rows)
                exports[name] = {**plotting.figure_files(fig, out, name, title,
                    "Every saved median; fixed-index central90% intervals; displayed out-of-prior intervals expand the axis."), **details}
                name = endpoint + "-" + label + "-sbc"
                fig = plotting.sbc_figure(summary)
                fig._suptitle.set_text(f'100k {endpoint}, epoch{record["native_epoch"]}: {summary["rows"]}-case six-marginal SBC')
                exports[name] = plotting.figure_files(fig, out, name, "All six marginal SBC screens",
                    "Six-marginal DKW band within this panel; reused development data, no independent acceptance.")
    if any(saved.evaluation.fixed.digest(path) != sha for path, sha in protected.items()):
        raise ValueError("protected diagnostics changed during figure export")
    manifest = {"format": "heston_pr87_100k_scientific_figures_v1", "protocol": binding(protocol_path),
        "evaluation": binding(evaluation_path), "secondary_evaluation": binding(secondary_path) if secondary_path else None,
        "native_history": curve_binding, "producer": binding(__file__), "archived_plotting_helper": binding(plotting.__file__),
        "versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__}, "exports": exports,
        "protected_inputs": [{"path": path, "sha256": sha} for path, sha in sorted(protected.items())],
        "protected_inputs_unchanged": True, "model_loaded": False, "new_neural_sampling": False,
        "new_pricing": False, "new_reference_sampling": False, "scientific_acceptance": False}
    saved.evaluation.fixed.write_json(out / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("protocol", "evaluation", "out"): parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--secondary", type=Path)
    args = parser.parse_args()
    result = build(args.protocol, args.evaluation, args.out, secondary_path=args.secondary)
    print(json.dumps({"exports": len(result["exports"]), "manifest": str(args.out / "manifest.json")}))
