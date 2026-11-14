import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="How much of an option surface must a summary retain?",
)


@app.cell(hide_code=True)
def _():
    import json
    from hashlib import sha256
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    names = ("initial_variance", "kappa", "theta", "vol_of_vol", "rho", "c")
    labels = {"initial_variance": "Initial variance v₀", "kappa": "Mean reversion κ",
              "theta": "Long-run variance θ", "vol_of_vol": "Volatility of variance η",
              "rho": "Correlation ρ", "c": "Quote-noise multiplier c"}
    bounds = {"initial_variance": (.01, .16), "kappa": (.1, 10.), "theta": (.01, .16),
              "vol_of_vol": (.1, 3.), "rho": (-.95, -.1), "c": (.5, 2.)}
    dimensions = (64, 32, 16, 8, 7, 6, 5, 4, 2)
    colors = {64: "#84613f", 32: "#8660b2", 16: "#38946b", 8: "#2976ac",
              7: "#17a2a4", 6: "#b767a4", 5: "#a58d36", 4: "#d17b28", 2: "#c54f52"}
    case_labels = {"synthetic-broad": "Broad-prior synthetic case",
                   "synthetic-market-like": "Market-like synthetic case", "market": "Observed SPX quotes"}

    def load_results(root):
        """Validate the compact, read-only lesson data and rebuild the plotting view."""
        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate result key: " + key)
                result[key] = value
            return result

        def hash_value(value):
            payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            return sha256(payload).hexdigest()

        def check_hash(value):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("a source SHA-256 identity is malformed")

        def vector(value, label, *, low=None, high=None):
            array = np.asarray(value, dtype=float)
            if array.shape != (6,) or not np.isfinite(array).all():
                raise ValueError("six finite values required for " + label)
            if low is not None and np.any(array < low) or high is not None and np.any(array > high):
                raise ValueError("out-of-range values for " + label)
            return value

        root = Path(root).resolve()
        record = json.loads((root / "results.json").read_text(), object_pairs_hook=unique_keys)
        protocol, identities = record["protocol"], record["identities"]
        budget = {"rows": 300000, "validation_rows": 100, "epochs": 300, "batch_size": 256,
                  "optimizer_updates": 351600, "presentations": 90000000, "seed": 9011}
        expected_summary = {"input_features": 63, "widths": [128, 128], "activation": "silu", "output_activation": "linear"}
        expected_posterior = {"family": "FlowMatching", "widths": [128, 128, 128], "dropout": .1,
                              "target_transform": "raw_coordinates_native_inference_variable_standardization"}
        panel = protocol["population"]
        if (record.get("format") != "heston_summary_dimensions_v1" or record.get("scientific_acceptance") is not False
                or tuple(protocol["parameter_names"]) != names or tuple(protocol["summary_dimensions"]) != dimensions
                or protocol["prior_bounds"] != [list(bounds[name]) for name in names]
                or protocol["training"] != budget or protocol["summary"] != expected_summary
                or protocol["posterior"] != expected_posterior or tuple(protocol["conditional_cases"]) != tuple(case_labels)
                or panel["rows"] != 512 or panel["draws_per_case"] != 512):
            raise ValueError("the complete matched nine-size development study is required")
        vector(panel["prior_midpoint_rmse"], "prior midpoint RMSE", low=0)
        if panel["prior_width90"] != [.9 * (bounds[name][1] - bounds[name][0]) for name in names]:
            raise ValueError("prior interval widths differ")
        if not 0 < panel["rank_dkw_bound"] < 1:
            raise ValueError("invalid SBC screen boundary")
        for key, value in identities.items():
            for digest in value.values() if key == "reference_chains" else (value,):
                check_hash(digest)
        if set(identities["reference_chains"]) != set(case_labels):
            raise ValueError("three matched reference identities required")
        for component in ("runner", "inference"):
            check_hash(record["producer"][component]["sha256"])
        check_hash(record["archive"]["sha256"])
        check_hash(record["archive"]["manifest_sha256"])
        contract = hash_value({"protocol": protocol, "identities": identities})
        runs, figures, seen = [], {}, set()
        for fit in record["fits"]:
            dimension, training = fit["dimension"], fit["training"]
            if (dimension is not None and (type(dimension) is not int or dimension not in dimensions)) or dimension in seen:
                raise ValueError("each of the nine summary fits and direct baseline must occur once")
            seen.add(dimension)
            if fit["contract_sha256"] != contract:
                raise ValueError("fit and shared protocol/panel identities differ")
            if not np.isfinite(training["elapsed_seconds"]) or training["elapsed_seconds"] <= 0:
                raise ValueError("invalid recorded training time")
            for key, value in fit["source"].items():
                if key.endswith("sha256"):
                    check_hash(value)
            history = None
            roles = {"primary_final"}
            if dimension is not None:
                history = {"epoch": list(range(1, 301)), **fit["history"]}
                if any(len(history[key]) != 300 or not np.isfinite(history[key]).all() for key in ("loss", "val_loss")):
                    raise ValueError("one complete finite 300-epoch history is required per summary fit")
                payload = json.dumps(history, indent=2, sort_keys=True, allow_nan=False) + "\n"
                if sha256(payload.encode()).hexdigest() != fit["source"]["compact_history_sha256"]:
                    raise ValueError("learning-history checksum differs")
                if (training["best_epoch"] != int(np.argmin(history["val_loss"])) + 1
                        or training["best_validation_loss"] != min(history["val_loss"])):
                    raise ValueError("validation checkpoint and history differ")
                roles.add("secondary_validation_minimum")
            if set(fit["endpoints"]) != roles:
                raise ValueError("final and distinct validation-minimum endpoint roles are required")
            figure = fit["recovery_figure"]
            relative = Path(figure["file"])
            image_path = (root / relative).resolve()
            if relative.is_absolute() or root not in image_path.parents:
                raise ValueError("recovery images must stay inside the lesson data directory")
            check_hash(figure["sha256"])
            if sha256(image_path.read_bytes()).hexdigest() != figure["sha256"]:
                raise ValueError("recovery image checksum differs")
            figures[dimension] = image_path
            for role, endpoint in fit["endpoints"].items():
                expected_epoch = 300 if role == "primary_final" else training["best_epoch"]
                if type(endpoint["epoch"]) is not int or endpoint["epoch"] != expected_epoch:
                    raise ValueError("checkpoint role and epoch differ")
                for value in endpoint["source"].values():
                    check_hash(value)
                if dimension is not None and "model_sha256" not in endpoint["source"]:
                    raise ValueError("summary model identity required")
                if role == "primary_final" and endpoint["source"]["draws_sha256"] != figure["source_draws_sha256"]:
                    raise ValueError("recovery image and primary draw identities differ")
                metrics = endpoint["population"]
                for key in ("rmse", "median_width90"):
                    vector(metrics[key], key, low=0)
                for key in ("coverage90", "rank_ks"):
                    vector(metrics[key], key, low=0, high=1)
                if (len(metrics["rank_passed"]) != 6 or any(type(v) is not bool for v in metrics["rank_passed"])
                        or metrics["rank_passed"] != [value <= panel["rank_dkw_bound"] for value in metrics["rank_ks"]]):
                    raise ValueError("SBC screen flags and rank statistics differ")
                if set(endpoint["cases"]) != set(case_labels):
                    raise ValueError("all matched conditional cases are required")
                cases = {}
                for case, values in endpoint["cases"].items():
                    for key in ("mean_error_in_reference_sd", "width90_ratio"):
                        vector(values[key], key, low=0)
                    cases[case] = {"reference_ready": True, "reference_chain_sha256": identities["reference_chains"][case],
                                   "parameters": {name: {key: values[key][i] for key in values} for i, name in enumerate(names)}}
                run = {**training, "status": "complete", "smoke_test_only": False, "train_rows": budget["rows"],
                       "completed_epochs": budget["epochs"], "optimizer_updates": budget["optimizer_updates"],
                       "presentations": budget["presentations"], "seed": budget["seed"], "evaluated_epoch": endpoint["epoch"],
                       "source_hashes": endpoint["source"], "data_payload_sha256": identities["bank_sha256"]}
                evaluation = {"status": "complete", "scientific_acceptance": False, "endpoint": role, "epoch": endpoint["epoch"],
                    "source_evaluation_sha256": endpoint["source"]["evaluation_sha256"],
                    "parameters": {name: {key: metrics[key][i] for key in ("rmse", "coverage90", "median_width90")} for i, name in enumerate(names)},
                    "population": {"parameter_names": names, "rows": 512, "draws_per_case": 512,
                        "truth_sha256": identities["truth_sha256"], "observations_sha256": identities["observations_sha256"],
                        "recovery": {key: panel[key] for key in ("prior_midpoint_rmse", "prior_width90")},
                        "rank_dkw": {"rank_ks": metrics["rank_ks"], "passed": metrics["rank_passed"], "dkw_bound": panel["rank_dkw_bound"]}},
                    "cases": cases}
                label = "Direct" if dimension is None else f"MLP {dimension}"
                runs.append({"dimension": dimension, "endpoint": role, "label": label, "run": run, "evaluation": evaluation, "history": history})
        if seen != {None, *dimensions} or len(runs) != 19:
            raise ValueError("all nine fits, nineteen endpoints and ten recovery figures are required")
        return runs, figures, record

    def select_endpoints(all_runs, endpoint="primary_final", dimensions=None):
        selected = [item for item in all_runs if item["endpoint"] == endpoint
                    and (dimensions is None or item["dimension"] is None or item["dimension"] in dimensions)]
        return sorted(selected, key=lambda item: (item["dimension"] is not None, -(item["dimension"] or 0)))

    def training_backend(item):
        devices = item["run"].get("devices", [])
        descriptor = " ".join(devices if isinstance(devices, list) else [devices]).lower()
        if any(kind in descriptor for kind in ("cuda", "gpu", "rocm")):
            return "JAX GPU"
        if "cpu" in descriptor:
            return "JAX CPU"
        return "Historical CPU (device metadata not stored)" if item["dimension"] is None else "Not recorded"

    def tidy_axes(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)

    return (bounds, case_labels, colors, dimensions, labels, load_results, mo, names,
            np, plt, select_endpoints, tidy_axes, training_backend)


@app.cell(hide_code=True)
def _(load_results, mo, select_endpoints):
    evidence_root = mo.notebook_dir() / "data/heston_summary_dimensions"
    all_runs, recovery_figures, lesson_data = load_results(evidence_root)
    primary_runs = select_endpoints(all_runs)
    summary_runs = [item for item in primary_runs if item["dimension"] is not None]
    direct_run = next(item for item in primary_runs if item["dimension"] is None)
    return all_runs, direct_run, evidence_root, lesson_data, primary_runs, recovery_figures, summary_runs


@app.cell(hide_code=True)
def _(dimensions, mo):
    _result = (
        "**The completed refinement:** c recovery stays similar at dimensions 7 and 6, "
        "then deteriorates sharply at **5**. Five coordinates still retain similar pooled "
        "recovery for the five Heston parameters; physical recovery deteriorates at 4. "
        "This is a parameter-specific empirical boundary for these trained models, not "
        "an exact information-theoretic minimum."
        if {7, 6, 5}.issubset(dimensions) else
        "**The initial screen:** dimensions 8–64 retain roughly the direct model's pooled recovery, "
        "while 4 and 2 lose substantial information. The matched 7-, 6- and 5-dimensional "
        "extension refines this gap as completed results enter the study."
    )
    mo.md(rf"""
    # How much of an option surface must a summary retain?

    **A standalone rundown of {len(dimensions)} completed Heston learned-summary experiments.**

    We observe the prices of 63 options and want a joint posterior over six unknowns:
    a probability distribution for the parameters after seeing those quotes.
    Can we replace the full price vector by only 32 numbers? Eight? Two?

    This notebook explains what was changed, what was held fixed, and what the
    completed experiments show. It runs entirely from the small aggregate records
    included in the repository. Opening it does not train a network, load a checkpoint,
    price an option or require a simulation-bank download.

    {_result}

    Some weak summaries still pass every marginal calibration screen. That distinction
    between calibration and information retention is the main lesson.
    """)
    return


@app.cell(hide_code=True)
def _(bounds, labels, mo, names):
    _roles = ["Variance at the observation time", "Speed of variance mean reversion",
              "Long-run variance level", "Volatility of the variance process",
              "Correlation of return and variance shocks", "Multiplier of observed quote half-spreads"]
    mo.vstack([
        mo.md(r"""
        ## 1. The inference problem stays the same

        Heston describes a risk-neutral stock-price process with a stochastic variance
        that reverts toward a long-run level. Its five physical parameters are collected
        in $\phi=(v_0,\kappa,\theta,\eta,\rho)$. Here, “physical” distinguishes the model
        parameters from quote noise; these are still **risk-neutral** calibration parameters.

        A deterministic European-option pricer gives $P_i(\phi)$ at each of the 63 fixed
        SPXW contracts. We add independent Gaussian quote errors:

        $$Y_i=P_i(\phi)+c\,h_i\,\epsilon_i,\qquad\epsilon_i\sim\mathcal N(0,1).$$

        The half-spread $h_i$ is observed. The common multiplier $c$ is inferred together
        with the five Heston parameters. “One half-spread is one noise standard deviation”
        at $c=1$ is a modelling convention, not a universal market law.

        Every experiment uses the same independent Uniform priors:
        """),
        mo.ui.table([{"unknown": labels[name], "role": role, "prior": f"Uniform{bounds[name]}"}
                     for name, role in zip(names, _roles)], selection=None, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Insert one learned bottleneck

    The direct model receives all 63 prices after scaling by their half-spreads and
    applying a frozen ZCA transform fitted on the first 10,000 training observations.

    $$x_i=Y_i/h_i,\qquad\widetilde x=W(x-\mu).$$

    This is a full-rank, 63-dimensional change of coordinates. It rescales and mixes
    features while retaining the input information. **Whitening is not compression.**

    We insert a jointly trained multilayer perceptron (MLP):

    $$s_\psi:\mathbb R^{63}\longrightarrow\mathbb R^d,\qquad
      \widehat p_\omega\!\left(\phi,c\mid s_\psi(\widetilde x)\right).$$

    The summary has two hidden layers of 128 SiLU units (a smooth activation) and a linear output of size
    $d$. The completed sizes are listed in the schematic and controls below. The
    posterior network remains FlowMatching with three
    128-wide hidden layers and dropout 0.1. The summary and posterior network learn
    together through the native inference objective. FlowMatching learns to transform
    simple base noise into joint parameter draws conditioned on the inputs. There is no separate summary
    regression or reconstruction loss.

    A 64-dimensional output is a **learned-representation control**: 63→64 is not
    dimensional compression. For smaller outputs, the map can discard information.
    Six unknown parameters do not imply that six summary coordinates suffice to
    preserve their complete posterior.
    """)
    return


@app.cell(hide_code=True)
def _(dimensions, mo, plt):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    def summary_schematic():
        # Match the Heston lesson's rounded-box artwork and training/inference palette.
        # Reference: lab/evidence/illustrations/heston-workflow-20261101/render.py.
        ink, purple, green, gray = "#15153B", "#4E2A84", "#2E7D43", "#63637A"
        with plt.rc_context({"font.family": "sans-serif", "mathtext.fontset": "dejavusans"}):
            fig, ax = plt.subplots(figsize=(14.4, 7.1))
            fig.patch.set_facecolor("white")
            fig.subplots_adjust(0, 0, 1, 1)
            ax.set(xlim=(0, 14.4), ylim=(0, 7.1))
            ax.axis("off")

            def text(x, y, value, size=12, color=ink, weight="normal", ha="center"):
                ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                        ha=ha, va="center", linespacing=1.4)

            def box(x, y, w, h, value, color=purple, fill="#F0ECF6", size=13):
                ax.add_patch(FancyBboxPatch((x, y), w, h,
                    boxstyle="round,pad=0.015,rounding_size=0.14",
                    linewidth=1.5, edgecolor=color, facecolor=fill))
                text(x+w/2, y+h/2, value, size, color)

            def arrow(a, b, color=purple):
                ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>",
                    mutation_scale=15, linewidth=1.8, color=color, shrinkA=2, shrinkB=2))

            text(.4, 6.7, "One observation, two ways to condition a posterior", 21, weight="bold", ha="left")
            text(.4, 6.2, f"The {len(dimensions)} completed sizes keep the hidden-layer recipe fixed and vary the summary output dimension.",
                 13, gray, ha="left")
            box(.4, 3.2, 1.85, 1.15, "63 noisy\noption quotes", color=green, fill="#EFF7EF")
            box(2.8, 3.2, 2.45, 1.15, "Divide by half-spreads\n+ frozen ZCA\n63 features retained", color=green,
                fill="#EFF7EF", size=12.5)
            arrow((2.3, 3.775), (2.75, 3.775), green)
            # A shared preprocessing stage, with two independently trained routes.
            ax.plot([5.3, 5.65, 5.65], [3.775, 3.775, 4.95], color=purple, lw=1.8)
            arrow((5.65, 4.95), (6., 4.95))
            ax.plot([5.65, 5.65], [3.775, 2.15], color=green, lw=1.8)
            arrow((5.65, 2.15), (6., 2.15), green)

            ax.add_patch(FancyBboxPatch((5.82, 4.13), 6.08, 1.92,
                boxstyle="round,pad=0.015,rounding_size=0.16", linewidth=0, facecolor="#F0ECF6", zorder=-1))
            text(8.86, 5.72, "SUMMARY + POSTERIOR: JOINTLY TRAINED", 11, purple, "bold")
            box(6.05, 4.4, 2.55, 1.1, "MLP summary\n128 SiLU → 128 SiLU\nlinear output: d", fill="white", size=12.5)
            box(9.1, 4.4, 2.55, 1.1, "FlowMatching\nconditioned on d\n3 × 128; dropout 0.1", fill="white", size=12.5)
            arrow((8.65, 4.95), (9.05, 4.95))
            arrow((11.7, 4.95), (12.1, 4.95))
            box(12.15, 4.4, 1.85, 1.1, "Six-parameter\nposterior draws", size=12.5)

            # Three lines keep the intermediate dimensions clear in the narrow summary column.
            _sizes = ", ".join(str(d) for d in dimensions)
            if len(dimensions) > 6:
                _sizes = ", ".join(str(d) for d in dimensions[:4]) + "\n" + ", ".join(str(d) for d in dimensions[4:])
            text(7.325, 3.65, f"d = {_sizes}\n64: representation control", 10.1, purple)
            box(9.1, 3.05, 2.55, .55, "Gaussian base noise", color=gray, fill="white", size=11.5)
            arrow((10.375, 3.65), (10.375, 4.35), gray)
            arrow((10.375, 3.), (10.375, 2.75), gray)

            text(7.325, 3.03, "DIRECT BASELINE", 11, green, "bold")
            box(6.05, 1.6, 2.55, 1.1, "All 63 features\nNo learned summary", color=green, fill="#EFF7EF", size=12.5)
            box(9.1, 1.6, 2.55, 1.1, "FlowMatching\nconditioned on 63\n3 × 128; dropout 0.1", color=green, fill="#EFF7EF", size=12.5)
            arrow((8.65, 2.15), (9.05, 2.15), green)
            arrow((11.7, 2.15), (12.1, 2.15), green)
            box(12.15, 1.6, 1.85, 1.1, "Six-parameter\nposterior draws", color=green, fill="#EFF7EF", size=12.5)
            text(.4, .95, "Training: the generating parameters supervise the joint inference loss; no reconstruction objective.",
                 12, purple, ha="left")
            text(.4, .53, "Inference: freeze both networks; input quotes and base noise to sample (v₀, κ, θ, η, ρ, c).", 12, green, ha="left")
            text(.4, .17, "Same posterior architecture recipe, separately fitted weights. The same 63 contracts are observed in every fit.",
                 10.8, gray, ha="left")
        return fig

    summary_diagram = summary_schematic()
    mo.vstack([summary_diagram, mo.md(
        "**Read the upper route from left to right:** preprocessing retains 63 coordinates; "
        "the learned summary supplies d conditioning coordinates to the posterior sampler. "
        "The purple group is optimized jointly. The direct baseline bypasses the summary. "
        "During inference, all fitted weights are fixed and the true parameters are unknown.")])
    return


@app.cell(hide_code=True)
def _(all_runs, dimensions, mo, primary_runs, training_backend):
    _hardware_rows = []
    for _item in primary_runs:
        _run = _item["run"]
        _devices = _run.get("devices", [])
        if isinstance(_devices, str):
            _devices = [_devices]
        _hardware_rows.append({
            "fit": "Direct baseline" if _item["dimension"] is None else f"MLP {_item['dimension']}",
            "training backend": training_backend(_item),
            "recorded devices": ", ".join(map(str, _devices)) or "Not stored",
            "recorded platform": _run.get("platform", "Not stored"),
            "training minutes": round(_run["elapsed_seconds"] / 60, 1),
        })
    mo.vstack([
        mo.md("## 3. A matched comparison"),
        mo.ui.table([
            {"Held fixed": "Observation design and law", "Value": "Same 63 contracts, spreads, priors and inferred c"},
            {"Held fixed": "Training data", "Value": "Same 300,000 stored noisy observations; noise is not redrawn each epoch"},
            {"Held fixed": "Training budget", "Value": "300 epochs; batch size 256; 351,600 updates; 90 million presentations"},
            {"Held fixed": "Optimization", "Value": "Seed 9011; same native AdamW warmup/cosine schedule"},
            {"Held fixed": "Validation", "Value": "Same 100 observations; final epoch 300 is primary"},
            {"Held fixed": "Development evaluation", "Value": "Same 512 simulated cases, 512 draws per case; three conditional references"},
            {"Changed": "Learned representation", "Value": "Linear summary output: " + ", ".join(map(str, dimensions)) + " coordinates"},
        ], selection=None, pagination=False),
        mo.md(f"There are **{len(dimensions)} completed summary fits**, {len(all_runs)-1} evaluated summary checkpoints, and one historical direct-input "
              "endpoint. This matches data and optimizer updates, not parameter count or FLOPs. "
              "The same seed label also does not imply identical initial weights across different shapes. "
              "These are single-seed development comparisons with training hardware recorded separately below."),
        mo.md("**Training hardware and timing.** The original screen used local JAX CPUs. "
              "The intermediate extension uses local CPU and Oscar GPU training; "
              "this table includes only completed fits and reports their recorded devices. "
              "The historical direct baseline lacks device metadata in its compact record. "
              "Wall times are descriptive, not a hardware-matched speed comparison."),
        mo.ui.table(_hardware_rows, selection=None, pagination=False),
        mo.callout(mo.md(
            "The source code, data and optimizer-update budget are matched, but CPU and GPU kernels "
            "can produce different floating-point results and optimizer trajectories. "
            "A common seed does not guarantee the same trained weights across backends. "
            "A one-seed deterioration therefore does not locate a purely mathematical information limit."), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def _(dimensions, mo):
    recovery_model = mo.ui.dropdown(
        options={"Direct 63-input baseline": "direct", **{f"MLP {d}": f"mlp{d}" for d in dimensions}},
        value="MLP 8", label="Recovery plot: fitted representation")
    mo.vstack([mo.md("## 4. When does recovery deteriorate?"),
        mo.md(r"""
        ### True parameters versus posterior estimates

        Each point is one of the same **512 simulated observations**, with **512 saved
        posterior draws** per observation. The x-axis gives its generating parameter;
        the point is the posterior median and its vertical line is the central 90%
        interval. The dashed diagonal marks exact point recovery.

        Start with **MLP 8**, then compare **MLP 7 → 6 → 5 → 4** and **MLP 2**, or the direct
        baseline. An informative estimate should track the diagonal; a nearly horizontal
        band with broad intervals indicates weak recovery. Look at **c** and initial variance.
        Axes are fixed across all completed models so the visual comparison stays fair.
        """), recovery_model])
    return (recovery_model,)


@app.cell(hide_code=True)
def _(mo, recovery_figures, recovery_model):
    _key = recovery_model.value
    _dimension = None if _key == "direct" else int(_key.removeprefix("mlp"))
    _name = "Direct 63-input baseline" if _dimension is None else f"MLP {_dimension}"
    mo.vstack([
        mo.image(recovery_figures[_dimension], width="100%",
                 alt=f"{_name}: true values versus posterior medians and 90% intervals for all six parameters"),
        mo.md("**Final epoch 300 only.** All 512 cases and their central intervals are shown; "
              "the figure was exported from the original unfiltered saved draws. "
              "The horizontal and vertical dotted lines mark the prior boundaries. "
              "No model is sampled when you change the selection. These are reused development cases, "
              "not a fresh acceptance population."),
    ])
    return


@app.cell(hide_code=True)
def _(dimensions, labels, mo):
    target_selector = mo.ui.dropdown(options={label: key for key, label in labels.items()},
                                     value=labels["c"], label="Parameter to investigate")
    dimension_selector = mo.ui.multiselect(options={str(d): d for d in dimensions},
                                           value=[str(d) for d in dimensions], label="Summary dimensions")
    mo.vstack([mo.md("### Compare the summary dimensions quantitatively"),
               mo.md(r"""Start with **c**, then examine initial variance or volatility of variance.
               Before looking at the plots, predict which diagnostic should reveal information loss.

               | Panel | How to read it |
               |:---|:---|
               | Recovery | Root mean squared error (RMSE) of posterior-median estimates, divided by the error of always predicting the prior midpoint on these same truths. Lower is better. |
               | Interval width | Median central-90% posterior width divided by the prior's central-90% width. A value near one means little contraction, but narrow intervals are only useful if they are accurate. |
               | Coverage | Fraction of simulated truths inside the posterior's 5th-to-95th percentile interval. The nominal value is 90%. |
               | SBC | Simulation-based calibration ranks each simulated truth among its posterior draws. The plotted ratio measures departure from uniform ranks relative to the finite-sample DKW tolerance; values below one pass this screen. |
               """),
               mo.hstack([target_selector, dimension_selector], justify="start")])
    return dimension_selector, target_selector


@app.cell(hide_code=True)
def _(dimension_selector, primary_runs, select_endpoints, target_selector):
    selected_runs = select_endpoints(primary_runs, dimensions=dimension_selector.value)
    selected_target = target_selector.value
    return selected_runs, selected_target


@app.cell(hide_code=True)
def _(
    colors,
    dimensions,
    direct_run,
    labels,
    names,
    plt,
    selected_runs,
    selected_target,
    tidy_axes,
):
    _index = names.index(selected_target)
    _reference = direct_run["evaluation"]
    _prior_rmse = _reference["population"]["recovery"]["prior_midpoint_rmse"][_index]
    _prior_width = _reference["population"]["recovery"]["prior_width90"][_index]
    _fig, _axes = plt.subplots(2, 2, figsize=(10.7, 7.0), constrained_layout=True)
    _baseline = _reference["parameters"][selected_target]
    _direct_values = [_baseline["rmse"] / _prior_rmse, _baseline["median_width90"] / _prior_width,
                      _baseline["coverage90"],
                      _reference["population"]["rank_dkw"]["rank_ks"][_index] /
                      _reference["population"]["rank_dkw"]["dkw_bound"]]
    for _item in selected_runs:
        if _item["dimension"] is None:
            continue
        _d, _evaluation = _item["dimension"], _item["evaluation"]
        _metric = _evaluation["parameters"][selected_target]
        _rank = _evaluation["population"]["rank_dkw"]
        _values = [_metric["rmse"] / _prior_rmse, _metric["median_width90"] / _prior_width,
                   _metric["coverage90"], _rank["rank_ks"][_index] / _rank["dkw_bound"]]
        for _ax, _value in zip(_axes.flat, _values):
            _ax.scatter(_d, _value, color=colors[_d], s=58, zorder=3)
    _titles = ["Recovery: median-estimate RMSE", "Information: posterior interval width",
               "Calibration: nominal 90% coverage", "Calibration: marginal SBC screen"]
    _ylabels = ["RMSE / prior-midpoint RMSE (log scale)", "Median 90% width / prior 90% width",
                "Fraction of simulated truths covered", "Rank KS statistic / DKW bound"]
    for _ax, _title, _ylabel, _value in zip(_axes.flat, _titles, _ylabels, _direct_values):
        _ax.axhline(_value, color="#4c5964", linestyle="--", linewidth=1.2, label="Direct 63-input model")
        _ax.set_xscale("log", base=2)
        _ticks = sorted(dimensions)
        _ax.set_xticks(_ticks, [str(d) for d in _ticks])
        _ax.tick_params(axis="x", labelrotation=45 if len(_ticks) > 6 else 0)
        _ax.set(xlim=(1.6, 80), title=_title, xlabel="Tested summary dimension", ylabel=_ylabel)
        tidy_axes(_ax)
    _axes[0, 0].set_yscale("log")
    _axes[0, 0].axhline(1, color="#aa7777", linestyle=":", label="Always predict prior midpoint")
    _axes[0, 1].axhline(1, color="#aa7777", linestyle=":", label="Prior interval width")
    _axes[1, 0].axhline(.9, color="#aa7777", linestyle=":", label="Nominal coverage")
    _axes[1, 0].set_ylim(0, 1.04)
    _axes[1, 1].axhline(1, color="#aa7777", linestyle=":", label="Screen boundary")
    for _ax in _axes.flat:
        _ax.legend(fontsize=7, loc="best")
    _fig.suptitle(f"{labels[selected_target]} — final epoch 300 only", fontsize=13)
    _fig
    return


@app.cell(hide_code=True)
def _(labels, mo, selected_runs, selected_target):
    _rows = []
    for _item in selected_runs:
        _evaluation = _item["evaluation"]
        _metric = _evaluation["parameters"][selected_target]
        _rows.append({"representation": "Direct 63 inputs" if _item["dimension"] is None else f"MLP {_item['dimension']}",
                      "RMSE": f"{_metric['rmse']:.6g}", "90% coverage": f"{100 * _metric['coverage90']:.2f}%",
                      "median 90% width": f"{_metric['median_width90']:.6g}",
                      "marginal SBC screens passed": f"{sum(_evaluation['population']['rank_dkw']['passed'])}/6"})
    mo.vstack([mo.md(f"**{labels[selected_target]}: saved aggregate values.** Every row uses the same observations."),
               mo.ui.table(_rows, selection=None, pagination=False),
               mo.md("Markers show tested dimensions only; no transition curve is fitted between them. "
                     "The RMSE reference always predicts the prior midpoint on these same 512 truths. "
                     "The SBC ratio is a diagnostic screen, not a ranking of posterior quality. "
                     "Narrower intervals alone are not better; interpret them alongside coverage and reference agreement.")])
    return


@app.cell(hide_code=True)
def _(mo, summary_runs):
    _by_d = {item["dimension"]: item["evaluation"] for item in summary_runs}
    _c8 = _by_d[8]["parameters"]["c"]
    _c2 = _by_d[2]["parameters"]["c"]
    _v0_factor = _by_d[2]["parameters"]["initial_variance"]["rmse"] / _by_d[8]["parameters"]["initial_variance"]["rmse"]
    mo.vstack([mo.md(r"""
        ## 5. Why can a weak summary pass calibration?

        A method that returns the prior for every observation can have correct pooled
        coverage and uniform marginal SBC ranks under the prior-predictive experiment.
        It has honest uncertainty across simulations but learns nothing from the quotes.

        Similarly, an accurate posterior **given a lossy summary** can be calibrated
        while retaining less information than the posterior given all observations.
        Our intended target is the latter. Calibration therefore belongs alongside
        recovery, posterior contraction and matched conditional references.
        """),
        mo.callout(mo.md(
            f"At dimension 8, the median c interval is **{_c8['median_width90']:.3f}** wide. "
            f"At dimension 2 it is **{_c2['median_width90']:.3f}**, essentially the prior's **1.35**. "
            f"Initial-variance RMSE is about **{_v0_factor:.0f} times larger** at dimension 2. "
            "Yet the final models with 8, 4 and 2 summary coordinates all pass six marginal SBC screens."), kind="info"),
        mo.md("This is evidence of diminished information recovery in the fitted pipeline, rather than a training crash. "
              "It does not prove that a two-dimensional representation can never work under a different task or recipe."),
    ])
    return


@app.cell(hide_code=True)
def _(case_labels, mo):
    case_selector = mo.ui.dropdown(options={value: key for key, value in case_labels.items()},
                                   value=case_labels["synthetic-broad"], label="Fixed observation for the reference check")
    mo.vstack([mo.md("## 6. Does the posterior match MCMC for a particular surface?"),
               mo.md("The population diagnostics average over many simulated observations. A conditional check "
                     "instead fixes one quote vector and compares the neural posterior with its target-matched "
                     "Markov chain Monte Carlo (MCMC) reference using the same prior and likelihood. "
                     "The saved references passed their convergence and numerical-pricing checks. "
                     "Start with a synthetic case, then inspect the observed SPX quotes."), case_selector])
    return (case_selector,)


@app.cell(hide_code=True)
def _(case_labels, case_selector, names, np, plt, selected_runs):
    _items = selected_runs
    _case = case_selector.value
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    _cmap = plt.get_cmap("YlOrRd").with_extremes(bad="#edf4ec")
    for _ax, _metric, _title in zip(_axes, ["mean_error_in_reference_sd", "width90_ratio"],
                                  ["Absolute mean error / reference SD", "Central-90% interval / reference width"]):
        _raw = np.array([[item["evaluation"]["cases"][_case]["parameters"][name][_metric]
                         for name in names] for item in _items])
        if _metric == "mean_error_in_reference_sd":
            _deviation = np.maximum(_raw / .3, 1)
        else:
            _deviation = np.maximum(np.maximum(.75 / _raw, _raw / 1.25), 1)
        _severity = np.ma.masked_where(_deviation <= 1, np.log10(_deviation))
        _ax.imshow(_severity, cmap=_cmap, vmin=0, vmax=3, aspect="auto")
        for _row in range(len(_items)):
            for _col in range(len(names)):
                _ax.text(_col, _row, f"{_raw[_row, _col]:.2g}", ha="center", va="center", fontsize=8)
        _ax.set_xticks(range(len(names)), ["v₀", "κ", "θ", "η", "ρ", "c"])
        _ax.set_yticks(range(len(_items)), ["Direct" if item["dimension"] is None else f"MLP {item['dimension']}" for item in _items])
        _ax.set(title=_title, xlabel="Parameter")
        _ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    _fig.suptitle(case_labels[_case] + " — final epoch 300 only", fontsize=13)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The numbers are the recorded mean errors and width ratios. Pale green marks entries
    inside the development margin; yellow/red indicates increasing departure on a log scale.
    The margins are an absolute mean error of at most **0.3 reference posterior SD** and a
    central-90% width ratio in **[0.75, 1.25]**. They are comparison conventions, not a new
    acceptance test implemented by this notebook. The recorded comparisons include
    Monte Carlo diagnostics; these heatmaps simply display their aggregate values.

    **In the original six-size screen, every complete conditional panel fails, including
    the direct baseline.** Good pooled recovery at dimension 8 therefore does not establish a correct
    full-data posterior. The market reference is particularly concentrated near the upper
    bound of c; large standardized errors there need that context. A summary matching an
    imperfect baseline is a relative development result.

    The heatmaps use recorded aggregate diagnostics. The recovery scatterplots above
    are exported separately from the original saved posterior draws; opening this notebook
    does not sample new posteriors.
    """)
    return


@app.cell(hide_code=True)
def _(dimensions, mo):
    checkpoint_dimension = mo.ui.dropdown(options={str(d): d for d in dimensions}, value="8",
                                         label="Inspect one training trajectory")
    mo.vstack([mo.md("## 7. Final checkpoint versus validation minimum"),
               mo.md("Final epoch 300 was declared primary before fitting. The first minimum of the "
                     "native validation loss was archived as a secondary endpoint. It was not selected "
                     "by recovery plots or MCMC agreement. The validation population contains only 100 cases."),
               checkpoint_dimension])
    return (checkpoint_dimension,)


@app.cell(hide_code=True)
def _(all_runs, checkpoint_dimension, mo, plt, tidy_axes):
    _d = checkpoint_dimension.value
    _pair = [item for item in all_runs if item["dimension"] == _d]
    _final = next(item for item in _pair if item["endpoint"] == "primary_final")
    _history, _run = _final["history"], _final["run"]
    _fig, _ax = plt.subplots(figsize=(8.7, 3.3), constrained_layout=True)
    _ax.plot(_history["epoch"], _history["loss"], label="training", linewidth=1)
    _ax.plot(_history["epoch"], _history["val_loss"], label="validation", linewidth=1)
    _ax.axvline(_run["best_epoch"], color="#66717a", linestyle=":", label=f"validation minimum: {_run['best_epoch']}")
    _ax.set(title=f"MLP {_d}: one saved training trajectory", xlabel="epoch", ylabel="Native FlowMatching objective")
    tidy_axes(_ax)
    _ax.legend(fontsize=8)
    _table = []
    for _item in _pair:
        _metric = _item["evaluation"]["parameters"]["c"]
        _table.append({"endpoint": "final (primary)" if _item["endpoint"] == "primary_final" else "validation minimum (secondary)",
                       "epoch": _item["run"]["evaluated_epoch"], "c RMSE": f"{_metric['rmse']:.5f}",
                       "median c 90% width": f"{_metric['median_width90']:.5f}",
                       "SBC screens": f"{sum(_item['evaluation']['population']['rank_dkw']['passed'])}/6"})
    mo.vstack([_fig, mo.ui.table(_table, selection=None, pagination=False),
               mo.md("A lower validation objective is not a substitute for posterior checks. "
                     "For the original 4- and 2-dimensional fits, deterioration also persists at their validation minima. "
                     "All declared endpoints remain visible; we do not choose whichever tells the better story.")])
    return


@app.cell(hide_code=True)
def _(direct_run, mo, names, summary_runs):
    _by_d = {item["dimension"]: item["evaluation"] for item in summary_runs}
    _intermediates = [d for d in (7, 6, 5) if d in _by_d]
    _gap_rows = []
    for _d in (8, 7, 6, 5, 4):
        if _d not in _by_d:
            continue
        _evaluation = _by_d[_d]
        _row = {"summary dimension": _d}
        for _name, _symbol in zip(names, ("v₀", "κ", "θ", "η", "ρ", "c")):
            _row[_symbol + " RMSE ratio"] = round(
                _evaluation["parameters"][_name]["rmse"] / direct_run["evaluation"]["parameters"][_name]["rmse"], 2)
        _row["c 90% width / prior"] = round(_evaluation["parameters"]["c"]["median_width90"] / 1.35, 3)
        _gap_rows.append(_row)
    _refinement = []
    _extension_description = ("The completed 7-, 6- and 5-dimensional extension examines that gap one coordinate at a time."
                              if {7, 6, 5}.issubset(_by_d) else
                              "The 7-, 6- and 5-dimensional extension refines that gap as matched evaluations complete.")
    if {7, 6, 5}.issubset(_by_d):
        _c6, _c5 = (_by_d[d]["parameters"]["c"] for d in (6, 5))
        _physical_ratios = [_by_d[5]["parameters"][name]["rmse"] /
                            direct_run["evaluation"]["parameters"][name]["rmse"]
                            for name in names if name != "c"]
        _refinement = [mo.callout(mo.md(
            f"**The clearest new break is c at 6 → 5 coordinates.** "
            f"Its median-estimate RMSE increases from **{_c6['rmse']:.3f}** to **{_c5['rmse']:.3f}**, "
            f"and its median 90% interval widens from **{_c6['median_width90']:.3f}** to "
            f"**{_c5['median_width90']:.3f}**, close to the prior's **1.35**. "
            f"At dimension 5, all five Heston-parameter RMSEs remain **{min(_physical_ratios):.2f}–"
            f"{max(_physical_ratios):.2f}×** their direct-model values. "
            "At dimension 4, physical recovery deteriorates as well. "
            "Thus six is the smallest tested size retaining similar pooled recovery for all six targets "
            "in this single-seed comparison; it is not a proven sufficient statistic."), kind="info")]
    mo.vstack([
        mo.md(rf"""
        ## 8. What can we conclude?

        The original six-size screen found similar pooled recovery at dimensions 8–64,
        with substantial deterioration at 4 and 2. {_extension_description}
        The table below shows only completed
        fits: errors are divided by the direct model's errors, while the final column
        measures contraction of the c interval relative to its prior width.
        """),
        *_refinement,
        mo.ui.table(_gap_rows, selection=None, pagination=False),
        mo.md(("**Completed intermediate sizes:** " + ", ".join(map(str, _intermediates)) + ". "
               if _intermediates else "**Dimensions 7, 6 and 5 are pending completed exports.** ") +
              "Use the recovery panels and conditional checks alongside these ratios; no numerical pass/fail threshold "
              "for information retention is imposed after seeing the results."),
        mo.md(r"""
        - **Calibration and information retention answer different questions.** A lossy
          summary can return broad, calibrated uncertainty while recovering parameters poorly.
        - **Different parameters can deteriorate at different dimensions.** A single
          summary size need not mark the loss of every kind of information at once.
        - **This is an empirical comparison for one recipe and one seed per size.**
          It does not establish an exact universal cutoff or a sufficient statistic.
          Optimization, architecture, finite data and CPU/GPU numerical trajectories
          can all contribute. Repeat the relevant sizes on a common backend as well
          as with independent seeds to isolate the representation effect.

        Repeat the sizes bracketing a material deterioration with independent seeds
        before calling that boundary stable. Masking and moving-grid inference remain
        later investigations with different input contracts.

        **Discussion:** Which plot would you trust if recovery deteriorated but coverage
        stayed near 90%? What extra evidence would distinguish a bottleneck from an
        optimization failure? Would a minimal summary for these priors also be minimal
        for another observation design?
        """),
    ])
    return


@app.cell(hide_code=True)
def _(all_runs, lesson_data, mo, training_backend):
    _commit = lesson_data["producer"]["archive_commit"]
    _archive = lesson_data["archive"]
    _rows = [{"representation": "Direct" if item["dimension"] is None else f"MLP {item['dimension']}",
              "endpoint": item["endpoint"], "epoch": item["run"]["evaluated_epoch"],
              "training backend": training_backend(item),
              "model SHA-256": item["run"]["source_hashes"].get("model_sha256", "see archived historical baseline record")}
             for item in all_runs]
    mo.accordion({"Evidence and reproduction": mo.vstack([
        mo.md("This lesson reads one compact `results.json` and ten recovery PNGs in "
              "`data/heston_summary_dimensions/`. Training history is stored once per fit. "
              "The loader verifies the matched protocol, endpoint roles, history checksums and image identities. "
              "Metric arrays retain the recorded six-parameter ordering. Full diagnostics, raw posterior "
              "draws and training payloads stay in the archived experiment."),
        mo.md(f"""
        To inspect or reproduce the original experiment, use the
        [archived experiment download]({_archive["file_url"]})
        and its [manifest/checksums]({_archive["folder_url"]}), together with the
        [pinned analysis checkout and Git history](https://github.com/fmuia/sbi4finance/tree/{_commit}).
        The accompanying data-directory README gives the later notebook-integration route.

        Archive: `{_archive["file"]}`. SHA-256: `{_archive["sha256"]}`.
        Recorded training-code hashes refer to each fit's training commit, preserved
        in the archive's Git history; the archive checkout also contains later analysis tools.

        The archive preserves the original training setup. This notebook is a read-only
        teaching account: it performs no training, model sampling or option pricing.
        """),
        mo.ui.table(_rows, selection=None, pagination=True, page_size=10),
    ])})
    return


if __name__ == "__main__":
    app.run()
