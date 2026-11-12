import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="How much of an option surface must a summary retain?")


@app.cell(hide_code=True)
def _():
    import json
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
    dimensions = (64, 32, 16, 8, 4, 2)
    colors = {64: "#84613f", 32: "#8660b2", 16: "#38946b", 8: "#2976ac",
              4: "#d17b28", 2: "#c54f52"}
    case_labels = {"synthetic-broad": "Broad-prior synthetic case",
                   "synthetic-market-like": "Market-like synthetic case", "market": "Observed SPX quotes"}

    def read_study(root):
        """Read only the shipped aggregate records; enforce the matched study contract."""
        root = Path(root).resolve()
        study = json.loads((root / "study.json").read_text())
        if study.get("format") != "heston_flexible_study_v1" or study.get("scientific_acceptance") is not False:
            raise ValueError("the checked aggregate development study is required")
        runs, identities = [], set()
        for entry in study["runs"]:
            relative = Path(entry["directory"])
            directory = (root / relative).resolve()
            if relative.is_absolute() or root not in directory.parents:
                raise ValueError("evidence directories must stay inside the study")
            run = json.loads((directory / "run.json").read_text())
            evaluation = json.loads((directory / "evaluation.json").read_text())
            dimension, endpoint = entry["summary_dim"], entry["endpoint"]
            if dimension is not None and (type(dimension) is not int or dimension not in dimensions):
                raise ValueError("unexpected summary dimension in this six-size experiment")
            if endpoint not in ("primary_final", "secondary_validation_minimum") or (dimension, endpoint) in identities:
                raise ValueError("unique primary/secondary experiment endpoints are required")
            identities.add((dimension, endpoint))
            if (run.get("status") != "complete" or run.get("smoke_test_only") is not False
                    or run.get("train_rows") != 300000 or run.get("completed_epochs") != 300
                    or run.get("optimizer_updates") != 351600 or run.get("presentations") != 90000000
                    or evaluation.get("format") != "heston_flexible_compact_evaluation_v1"
                    or evaluation.get("status") != "complete" or evaluation.get("endpoint") != endpoint
                    or run.get("endpoint") != endpoint or run.get("evaluated_epoch") != evaluation.get("epoch")
                    or entry.get("epoch") != evaluation.get("epoch")
                    or tuple(evaluation["population"]["parameter_names"]) != names):
                raise ValueError("complete matched 300k/300-epoch aggregate endpoints are required")
            expected_epoch = 300 if endpoint == "primary_final" else run.get("best_epoch")
            if evaluation.get("epoch") != expected_epoch:
                raise ValueError("checkpoint role and evaluated epoch differ")
            if dimension is not None:
                config = run["configuration"]
                if (run.get("seed") != 9011 or config.get("summary_dim") != dimension
                        or config.get("summary_widths") != [128, 128] or config.get("summary_activation") != "silu"
                        or config.get("posterior_widths") != [128, 128, 128] or config.get("posterior_dropout") != .1):
                    raise ValueError("summary size must be the only declared recipe change")
            population = evaluation["population"]
            if population["rows"] != 512 or population["draws_per_case"] != 512:
                raise ValueError("the common 512-case/512-draw development population is required")
            expected_prior_widths = [.9 * (bounds[name][1] - bounds[name][0]) for name in names]
            if not np.allclose(population["recovery"]["prior_width90"], expected_prior_widths):
                raise ValueError("displayed priors differ from the experiment records")
            history_path = directory / "history.json"
            history = json.loads(history_path.read_text()) if history_path.exists() else None
            if dimension is not None and (history is None or history["epoch"] != list(range(1, 301))
                                          or not np.isfinite(history["loss"] + history["val_loss"]).all()):
                raise ValueError("complete finite learning history required")
            runs.append({"dimension": dimension, "endpoint": endpoint, "label": entry["label"],
                         "run": run, "evaluation": evaluation, "history": history})
        expected = {(None, "primary_final")} | {(d, role) for d in dimensions
                     for role in ("primary_final", "secondary_validation_minimum")}
        if identities != expected:
            raise ValueError("all six fits, twelve endpoints and the historical direct baseline are required")
        baseline = next(item for item in runs if item["dimension"] is None)
        reference = baseline["evaluation"]["population"]
        for item in runs:
            population = item["evaluation"]["population"]
            if (item["run"]["data_payload_sha256"] != baseline["run"]["data_payload_sha256"]
                    or population["truth_sha256"] != reference["truth_sha256"]
                    or population["observations_sha256"] != reference["observations_sha256"]):
                raise ValueError("bank or population pairing differs")
            for case in case_labels:
                current = item["evaluation"]["cases"][case]
                original = baseline["evaluation"]["cases"][case]
                if (current.get("reference_ready") is not True
                        or current["reference_chain_sha256"] != original["reference_chain_sha256"]):
                    raise ValueError("matched conditional references are required")
        return runs

    def select_endpoints(all_runs, endpoint="primary_final", dimensions=None):
        selected = [item for item in all_runs if item["endpoint"] == endpoint
                    and (dimensions is None or item["dimension"] is None or item["dimension"] in dimensions)]
        return sorted(selected, key=lambda item: (item["dimension"] is not None, -(item["dimension"] or 0)))

    def tidy_axes(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)

    return (Path, bounds, case_labels, colors, dimensions, json, labels, mo, names,
            np, plt, read_study, select_endpoints, tidy_axes)


@app.cell(hide_code=True)
def _(Path, mo, read_study, select_endpoints):
    evidence_root = mo.notebook_dir().parent / "evidence/pilot-results/2026-11-12/heston-flexible-inputs"
    all_runs = read_study(evidence_root)
    primary_runs = select_endpoints(all_runs)
    summary_runs = [item for item in primary_runs if item["dimension"] is not None]
    direct_run = next(item for item in primary_runs if item["dimension"] is None)
    return all_runs, direct_run, evidence_root, primary_runs, summary_runs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How much of an option surface must a summary retain?

    **A standalone rundown of six Heston learned-summary experiments.**

    We observe the prices of 63 options and want a joint posterior over six unknowns:
    a probability distribution for the parameters after seeing those quotes.
    Can we replace the full price vector by only 32 numbers? Eight? Two?

    This notebook explains what was changed, what was held fixed, and what the
    completed experiments show. It runs entirely from the small aggregate records
    included in the repository. Opening it does not train a network, load a checkpoint,
    price an option or require a simulation-bank download.

    **The result to investigate:** dimensions 8–64 retain roughly the direct model's
    pooled recovery; dimensions 4 and 2 lose substantial information. Some of those
    models still pass every marginal calibration screen. Understanding that apparent
    contradiction is the main lesson.
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
    $d\in\{64,32,16,8,4,2\}$. The posterior network remains FlowMatching with three
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
def _(mo):
    _boxes = [("63 noisy quotes", "fixed observed contracts"),
              ("Scale + frozen ZCA", "63 dimensions retained"),
              ("MLP summary", "128 → 128 → d"),
              ("FlowMatching", "same six-target posterior head")]
    _parts = []
    for _index, (_title, _detail) in enumerate(_boxes):
        if _index:
            _parts.append('<div style="align-self:center;color:#6c7b88">→</div>')
        _parts.append(f'<div style="flex:1;min-width:135px;padding:14px 10px;border:1px solid #cddae4;'
                      f'border-radius:8px;background:#f1f6fa;text-align:center"><b>{_title}</b>'
                      f'<div style="font-size:0.85em;color:#526579;margin-top:5px">{_detail}</div></div>')
    mo.Html('<div style="display:flex;flex-wrap:wrap;gap:8px">' + ''.join(_parts) + '</div>')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("## 3. A matched comparison"),
        mo.ui.table([
            {"Held fixed": "Observation design and law", "Value": "Same 63 contracts, spreads, priors and inferred c"},
            {"Held fixed": "Training data", "Value": "Same 300,000 stored noisy observations; noise is not redrawn each epoch"},
            {"Held fixed": "Training budget", "Value": "300 epochs; batch size 256; 351,600 updates; 90 million presentations"},
            {"Held fixed": "Optimization", "Value": "Seed 9011; same native AdamW warmup/cosine schedule"},
            {"Held fixed": "Validation", "Value": "Same 100 observations; final epoch 300 is primary"},
            {"Held fixed": "Development evaluation", "Value": "Same 512 simulated cases, 512 draws per case; three conditional references"},
            {"Changed": "Learned representation", "Value": "Linear summary output: 64, 32, 16, 8, 4 or 2 coordinates"},
        ], selection=None, pagination=False),
        mo.md("There are **six new fits**, twelve evaluated checkpoints, and one historical direct-input "
              "endpoint. This matches data and optimizer updates, not parameter count or FLOPs. "
              "The same seed label also does not imply identical initial weights across different shapes. "
              "These are single-seed development comparisons; wall times came from a shared local CPU."),
    ])
    return


@app.cell(hide_code=True)
def _(dimensions, labels, mo):
    target_selector = mo.ui.dropdown(options={label: key for key, label in labels.items()},
                                     value=labels["c"], label="Parameter to investigate")
    dimension_selector = mo.ui.multiselect(options={str(d): d for d in dimensions},
                                           value=[str(d) for d in dimensions], label="Summary dimensions")
    mo.vstack([mo.md("## 4. When does recovery deteriorate?"),
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
def _(colors, direct_run, labels, names, np, plt, selected_runs, selected_target, tidy_axes):
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
        _ax.set_xticks([2, 4, 8, 16, 32, 64], ["2", "4", "8", "16", "32", "64"])
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
def _(case_labels, dimensions, mo):
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
def _(case_labels, case_selector, labels, names, np, plt, selected_runs):
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

    **Every complete conditional panel still fails for every model, including the direct
    baseline.** Good pooled recovery at dimension 8 therefore does not establish a correct
    full-data posterior. The market reference is particularly concentrated near the upper
    bound of c; large standardized errors there need that context. A summary matching an
    imperfect baseline is a relative development result.

    The stored records contain aggregate diagnostics, not individual posterior draws.
    These displays do not manufacture posterior densities or recovery scatterplots.
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
                     "The small summaries' deterioration persists at their validation minima. "
                     "All declared endpoints remain visible; we do not choose whichever tells the better story.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. What can we conclude?

    - **Eight is the smallest tested size retaining roughly baseline pooled recovery.**
      Four deteriorates, and two further weakens physical-parameter recovery.
    - **Calibration and information retention answer different questions.** The small
      summaries can return nearly prior-width uncertainty for c and still pass SBC.
    - **This is an empirical breakdown for this recipe.** Dimension 6 is untested,
      there is one seed per size, and none of the full conditional checks passes.
      We have not established a universal phase transition or a sufficient statistic.

    Before claiming a reproducible threshold, repeat the informative 8-versus-4
    comparison with another seed and consider an intermediate dimension. Masking and
    moving-grid inference remain later investigations with different input contracts.

    **Discussion:** Which plot would you trust if recovery deteriorated but coverage
    stayed near 90%? What extra evidence would distinguish a bottleneck from an
    optimization failure? Would a minimal summary for these priors also be minimal
    for another observation design?
    """)
    return


@app.cell(hide_code=True)
def _(all_runs, evidence_root, mo):
    _rows = [{"representation": "Direct" if item["dimension"] is None else f"MLP {item['dimension']}",
              "endpoint": item["endpoint"], "epoch": item["run"]["evaluated_epoch"],
              "model SHA-256": item["run"]["source_hashes"].get("model_sha256", "historical baseline; see source record")}
             for item in all_runs]
    mo.accordion({"Evidence and reproduction": mo.vstack([
        mo.md(f"The notebook reads the checked-in aggregate study at `{evidence_root.name}`. "
              "Private simulation payloads and neural models are unnecessary for this rundown. "
              "The adjacent README, campaign record and independent audit document task, data and source identities. "
              "All raw neural draws were retained in the original evaluations, including support violations."),
        mo.md("""To retrain one candidate from the verified development bank, run from `lab/`:

```sh
pixi run -e core-reference python scripts/heston_flexible_experiment.py train \\
  --bundle "$HESTON_BUNDLE" --out "$NEW_RUN_DIRECTORY" \\
  --summary-dim 8 --rows 300000 --epochs 300 --seed 9011 --evaluate
```

This command is documentation, not an action executed by the notebook. New output
identities are required for each fit. The experiment dashboard remains available in
`heston_flexible_inputs.py`; this notebook is the explanatory fixed-grid rundown.
"""),
        mo.ui.table(_rows, selection=None, pagination=True, page_size=7),
    ])})
    return


if __name__ == "__main__":
    app.run()
