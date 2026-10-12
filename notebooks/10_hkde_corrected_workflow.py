import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import importlib.util
    import json
    import sys
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    ROOT = mo.notebook_dir().parent

    def load_script(name, path):
        specification = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(specification)
        # Runner scripts import neighboring helpers; restore search paths after
        # loading so notebook execution does not depend on its launch directory.
        previous_paths = sys.path.copy()
        try:
            sys.path.insert(0, str(path.parent))
            specification.loader.exec_module(module)
        finally:
            sys.path[:] = previous_paths
        return module

    plot_helpers = load_script("hkde_correction_plots", ROOT / "scripts/plot_hkde_correction.py")
    return ROOT, json, load_script, mo, np, plot_helpers, plt, time


@app.cell
def _(mo):
    mo.md(r"""
    # 10. An options candidate: neural proposal, likelihood correction, price uncertainty

    **A complete synthetic question:** which parameter pairs explain these
    nine option quotes, and what prices do those plausible pairs imply for
    two other contracts?

    This notebook assembles existing building blocks into an **M8 options
    candidate**. It does not select the final returns-versus-options curriculum.
    The model, observation and downstream prices all use the risk-neutral
    measure Q; nothing here forecasts tomorrow's chain or physical-measure
    loss probabilities.

    The raw HKDE network previously missed a downstream price gate. We keep
    that result and its weights unchanged, use it as a proposal distribution,
    and check a different estimator: likelihood-weighted joint parameter draws.
    An accepted forward surrogate then prices two fixed target puts.

    **Suggested 30-minute lesson:** 5 minutes for the financial task and data
    roles, 8 for correction, 7 for the live posterior/prices, 7 for diagnostics,
    and 3 for the exercises. Expensive training and reference studies are
    cached. Once the new scientific gates pass, default execution performs
    one bounded live correction with 2,048 proposals; it starts no bank
    generation, training or cluster job and needs no real-market fixtures.
    """)
    return


@app.cell
def _(ROOT, json, plot_helpers):
    protocol_path = ROOT / "configs/hkde_correction_boundary_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    task = json.loads((ROOT / protocol["task_config"]).read_text())["task"]
    baseline = json.loads((ROOT / protocol["baseline_results"] / "neural-validation.json").read_text())
    _predecessor_path = ROOT / protocol["predecessor_results"] / "validation.json"
    predecessor = json.loads(_predecessor_path.read_text()) if _predecessor_path.exists() else None
    reports, saved_checks, artifact_paths = plot_helpers.load_cache(ROOT / "docs/results/hkde-correction")
    if any(_report["experiment_id"] != protocol["experiment_id"] for _report in reports.values()):
        raise ValueError("cached correction evidence belongs to a different proposal experiment")
    cached_figures = plot_helpers.figures(reports, saved_checks, baseline, protocol)
    candidate_accepted = all(reports.get(_split, {}).get("passed", False) for _split in ("validation", "test"))
    return artifact_paths, baseline, cached_figures, candidate_accepted, predecessor, protocol, protocol_path, reports, saved_checks, task


@app.cell
def _(candidate_accepted, mo, protocol, reports):
    _rows = [{"Check": _split, "Saved status":
              ("PASS" if reports[_split]["passed"] else "FAIL") if _split in reports else "not recorded"}
             for _split in ("validation", "test")]
    _stages = []
    for _split, _report in reports.items():
        _conditional_names = [*protocol["gates"], "proposal_ess", "proposal_max_weight"]
        _stages.extend([
            {"Split": _split, "Stage": "Numerical reference", "Passed": _report.get("reference", {}).get("passed")},
            {"Split": _split, "Stage": "2,048-proposal conditional checks", "Passed":
             all(_report.get("gates", {}).get(_key, False) for _key in _conditional_names) if _report.get("cases") else None},
            {"Split": _split, "Stage": "8,192-proposal precision checks", "Passed":
             all(_report.get("precision", {}).get("gates", {}).values()) if _report.get("precision", {}).get("gates") else None},
            {"Split": _split, "Stage": "Finite-pool SIR rank check", "Passed": _report.get("calibration", {}).get("rank_diagnostic_passed")},
            {"Split": _split, "Stage": "Calibration pool efficiency", "Passed":
             all(_report.get("gates", {}).get(_key, False) for _key in ("calibration_ess", "calibration_max_weight"))
             if "calibration_ess" in _report.get("gates", {}) else None},
        ])
    mo.vstack([
        mo.md("## Candidate status\n\nThe corrected pipeline has its own frozen acceptance test. It does not retrospectively accept the raw neural posterior."),
        mo.ui.table(_rows, selection=None),
        mo.ui.table(_stages, selection=None) if _stages else mo.md("No corrected scientific stages are recorded yet."),
        mo.callout("Both recorded gates pass. The bounded synthetic live example below is enabled."
                   if candidate_accepted else
                   "The required corrected validation and new test are not both accepted. Live candidate execution is skipped; available development diagnostics remain visible.",
                   kind="success" if candidate_accepted else "info"),
    ])
    return


@app.cell
def _(mo, task):
    mo.vstack([
        mo.md(r"""
        ## Observation, known context and unknown targets

        The observation $y$ contains nine noisy **currency option prices**, in
        fixed strike/maturity order. The unknown pair is
        $a=(\theta,\lambda)$: Heston long-run variance and jump intensity per
        year. The prior is uniform over the declared rectangle below.

        Spot, interest/dividend rates, initial variance, all other HKDE
        parameters and the nine contracts are **known task context**. HKDE
        combines stochastic variance with double-exponential price jumps.
        We are not estimating every possible model parameter here.

        Synthetic generating parameters are evaluation truth, not conditioning
        inputs. Clean reference prices and numerical diagnostics are also
        auxiliary. The network and correction receive the noisy observation;
        the independent reference supplies a check after inference. Real
        market data would not reveal the true latent parameters.
        """),
        mo.ui.table([{"Unknown": _name, "Uniform lower": _bounds[0], "Uniform upper": _bounds[1]}
                     for _name, _bounds in zip(task["parameter_names"], task["prior_bounds"])], selection=None),
        mo.ui.table([{"Role": "Observed", "Contents": "Nine ordered noisy prices; [9,1] currency array"},
                     {"Role": "Known context", "Contents": "Contracts, spot/carry, fixed model parameters and quote-noise scale"},
                     {"Role": "Joint inference targets", "Contents": "Long-run variance θ and jump intensity λ"},
                     {"Role": "Auxiliary only", "Contents": "Synthetic truth, clean prices, reference weights and numerical diagnostics"}], selection=None),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why correct a neural posterior?

    Let $q_\phi(a\mid y)$ be the density returned by the frozen network.
    Synthetic training minimizes $-B^{-1}\sum_{b=1}^B\log q_\phi(a_b\mid y_b)$
    using independent parameter/surface pairs. This learns inverse calibration;
    it does not learn the forward pricing map or require true market parameters.
    Training makes the density an approximation to a posterior, not an exact
    one. Here an independently evaluable likelihood is available:

    $$L_F(y\mid a)\propto
      \exp\left[-\frac{1}{2(0.25)^2}
      \sum_{c=1}^9(y_c-C_{F,c}(a))^2\right].$$

    $C_F$ is the accepted numerical Fourier price map. The independent
    Gaussian quote-error model is part of the declared synthetic task.
    “Exact likelihood correction” here means using that specified numerical
    likelihood, not exact continuum pricing or exact finite-sample inference.

    Use a defensive proposal with a fixed boundary component,

    $$g(a\mid y)=0.8q_\phi(a\mid y)+0.1p(a)+0.1h(a),$$

    where $p$ remains the declared uniform prior. For bounds $(l_j,r_j)$,

    $$h(a)=\prod_{j=1}^{2}\frac{1}{\pi\sqrt{(a_j-l_j)(r_j-a_j)}}.$$

    To sample $h$, draw independent Beta(1/2,1/2) coordinates on $(0,1)$
    and scale them to the prior box. This arcsine density puts more proposal
    mass near **all** boundaries. Each row independently chooses one of the
    three components. The uniform term supplies positive proposal density
    throughout the box. For each joint draw,

    $$\widetilde w_i=
       \frac{L_F(y\mid a_i)p(a_i)}{g(a_i\mid y)},
       \qquad w_i=\frac{\widetilde w_i}{\sum_j\widetilde w_j}.$$

    Every row uses the **full mixture denominator**, even when it came from
    the prior or boundary component. The posterior prior in the numerator is
    still uniform; changing the proposal does not change the inferential model.
    Evaluate proposal and prior densities in the same raw parameter
    units, including the adapter and standardization Jacobians already supplied
    by the density implementation. Adding them again changes the target.
    Normalization uses log-sum-exp; no weights are clipped.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Finite particles remain an approximation

    A posterior expectation is estimated by $\sum_iw_if(a_i)$.
    Self-normalized importance sampling (**SNIS**) has finite-sample bias and
    Monte Carlo error. Increasing the sample can improve it under appropriate
    support/integrability conditions; a finite weighted cloud is not an exact
    posterior.

    Effective sample size $1/\sum_iw_i^2$ and the largest normalized weight
    diagnose weight concentration. High ESS cannot certify that a finite sample
    visited every important mode. The independent quadrature reference is
    therefore essential to the conditional checks in this example. Dividing
    proposals into eight independent blocks also makes Monte Carlo variability
    visible; it does not measure neural bias or Fourier discretization error.

    **Pricing still costs work.** Each proposal requires nine Fourier prices
    to evaluate $L_F$. The polynomial surrogate is accepted for only the two
    downstream target contracts and does not replace these nine likelihood
    prices. This corrected workflow is not a claim of purely amortized inference
    or a guaranteed speed advantage over the reusable grid reference.
    """)
    return


@app.cell
def _(baseline, mo):
    _error = baseline["metrics"]["max_abs_target_quantile_error_currency"]
    _gate = baseline["thresholds"]["max_abs_target_quantile_error_currency"]
    mo.callout(
        f"Preserved baseline: raw neural posterior target-price error {_error:.5f} currency exceeded {_gate:.2f}, "
        "despite passing marginal SBC. The raw checkpoint remains provisional. Correction changes the estimator, not the historical result.",
        kind="warn")
    return


@app.cell
def _(mo, predecessor):
    _items = [mo.md(r"""
        **Why the boundary component?** The first correction used 90% flow and
        10% uniform prior. Conditional accuracy improved, but its larger pool
        revealed a high-likelihood point close to the lower jump-intensity
        boundary that the logit-based flow represented poorly. Proposal
        reliability failed in the fixed `stress:2` case, so SBC and the new
        test were not run. The absolute ESS grew with more proposals; this
        failure did not mean the price estimates became less accurate.

        The one frozen follow-up adds symmetric boundary coverage while keeping
        the original prior, likelihood, checkpoint, budgets, seeds and gates.
        It retains every weighted row. This is an explicitly revised proposal,
        not retrospective acceptance of the first experiment. Its outcome is
        shown by the current candidate status, never assumed from the design.
        """)]
    if predecessor is not None:
        _diagnostics = predecessor["precision"]["proposal_diagnostics"]
        _items.append(mo.callout(
            f"Preserved v1 precision result: ESS/N {_diagnostics['min_ess_fraction']:.4f} < 0.5; "
            f"maximum weight {_diagnostics['max_weight']:.5f} > 0.01. All conditional accuracy gates passed; reliability gates stopped the run.",
            kind="warn"))
    mo.vstack(_items)
    return


@app.cell
def _(mo, protocol):
    mo.md(f"""
    ## One live synthetic panel

    We use **validation:257**, the fixed audit case with the largest earlier
    raw-neural price discrepancy. Its selection is explicit; the independent
    new test supplies the separate generalization check. This cell will load
    compatible immutable assets, draw **{protocol['conditional']['participant_samples']:,}**
    proposals, compute the numerical likelihood weights and evaluate the two
    accepted target-price outputs. It uses no generating truth in conditioning.

    The complete scientific audit separately checks both this participant
    budget and the {protocol['conditional']['samples']:,}-proposal precision
    budget. Cached validation/test results below are distinguished from this
    current machine's live computation.
    """)
    return


@app.cell
def _(ROOT, candidate_accepted, load_script, mo, np, protocol_path, reports, task, time):
    live_result = None
    if candidate_accepted:
        _started = time.perf_counter()
        _runner = load_script("hkde_correction_runner", ROOT / "scripts/evaluate_hkde_correction.py")
        _protocol, _baseline_protocol, _task, _grid, _saved, _model, _bindings = _runner.load_inputs(protocol_path)
        if _task != task:
            raise ValueError("displayed task differs from the bound correction task")
        for _split in ("validation", "test"):
            if reports[_split]["input_sha256"] != _bindings or reports[_split]["source_sha256"] != _runner.source_hashes():
                raise ValueError(f"{_split}: saved correction evidence does not match the live inputs/source")
        _index = list(_saved["case_ids"]).index("validation:257")
        _observation = _saved["observations"][_index]
        _seed = _runner.case_seed(_protocol, "validation", 0, _index)
        _workflow = _runner.inference.build_workflow(_task)
        _workflow.load_approximator(ROOT / _protocol["checkpoint"] / "model.keras")
        _loaded = time.perf_counter()
        _proposal = _runner.correction.propose(
            _workflow, _task, _observation,
            num_samples=_protocol["conditional"]["participant_samples"], seed=_seed,
            prior_fraction=_protocol["proposal"]["prior_fraction"],
            boundary_fraction=_protocol["proposal"]["boundary_fraction"])
        _proposed = time.perf_counter()
        _corrected = _runner.correction.correct(_proposal["parameters"], _proposal["log_proposal"], _observation, _task)
        _weighted = time.perf_counter()
        _target_prices = _runner.surrogate.predict(_model, _proposal["parameters"], task=_task, contracts=_baseline_protocol["targets"])
        _target_ready = time.perf_counter()
        _reference_weights = _runner.reference.posterior(_grid, _observation, _task["context"]["quote_noise_std"])
        _reference_summary = _runner.reference.summarize(_grid, _reference_weights, _protocol["quantiles"])
        _summary = _runner.correction.summarize(
            _proposal["parameters"], _corrected["weights"], _target_prices,
            reference_summary=_reference_summary, probabilities=_protocol["quantiles"],
            blocks=_protocol["conditional"]["uncertainty_blocks"])
        _metrics = _runner.aggregate([_summary])
        _gates = {_name: _value <= _protocol["gates"][_name] for _name, _value in _metrics.items()}
        _gates.update(proposal_ess=_corrected["ess_fraction"] >= _protocol["conditional"]["min_ess_fraction"],
                      proposal_max_weight=_corrected["max_weight"] <= _protocol["conditional"]["max_weight"])
        live_result = {
            "case_id": "validation:257", "seed": _seed, "observation": _observation,
            "parameters": _proposal["parameters"], "weights": _corrected["weights"],
            "target_prices": _target_prices, "targets": _baseline_protocol["targets"],
            "grid": _grid, "reference_weights": _reference_weights, "reference_summary": _reference_summary,
            "summary": _summary, "metrics": _metrics, "gates": _gates, "passed": all(_gates.values()),
            "ess": _corrected["ess"], "ess_fraction": _corrected["ess_fraction"],
            "max_weight": _corrected["max_weight"], "prior_rows": int(_proposal["is_prior"].sum()),
            "boundary_rows": int(_proposal["is_boundary"].sum()),
            "timing_seconds": {"imports_bindings_checkpoint_load": _loaded - _started,
                               "proposal_sample": _proposal["timing_seconds"]["sample"],
                               "proposal_density": _proposal["timing_seconds"]["density"],
                               "proposal_total": _proposed - _loaded,
                               "nine_price_likelihood_and_weights": _weighted - _proposed,
                               "two_target_surrogate": _target_ready - _weighted,
                               "reference_comparison_and_summary": time.perf_counter() - _target_ready,
                               "live_total": time.perf_counter() - _started},
        }
        _display = mo.md(f"Live correction completed with **{len(_proposal['parameters']):,} proposals** and ESS **{_corrected['ess']:.0f}**. "
                         "The tables below report this execution separately from the cached scientific study.")
    else:
        _display = mo.callout("Live correction is disabled until both frozen validation and the new independent test pass. No bank, model training or test run is started here.", kind="info")
    _display
    return (live_result,)


@app.cell
def _(live_result, mo, task):
    if live_result is not None:
        _view = mo.vstack([
            mo.md("### The actual conditioning panel"),
            mo.ui.table([{**_contract, "Observed synthetic price": float(live_result["observation"][_index, 0])}
                         for _index, _contract in enumerate(task["context"]["contracts"])], selection=None),
            mo.callout("This live case meets the same reported numerical thresholds."
                       if live_result["passed"] else
                       "This live finite-pool execution misses at least one threshold. Its result is retained; the notebook does not retry or change the seed.",
                       kind="success" if live_result["passed"] else "warn"),
            mo.ui.table([{"Metric": _name, "Live value": _value, "Passed": live_result["gates"][_name]}
                         for _name, _value in live_result["metrics"].items()], selection=None),
            mo.md(f"Prior-component rows: **{live_result['prior_rows']}**; boundary-component rows: **{live_result['boundary_rows']}**; "
                  f"ESS/N: **{live_result['ess_fraction']:.3f}**; "
                  f"largest normalized weight: **{live_result['max_weight']:.5f}**. "
                  "The live 2,048-draw API call need not be bitwise identical to the stored prefix of an 8,192-draw call; it uses the same proposal law and is checked independently."),
        ])
    else:
        _view = mo.md("The live observation/result table will appear after candidate acceptance.")
    _view
    return


@app.cell
def _(live_result, mo, plot_helpers):
    _items = [mo.md(r"""
        ## Keep the joint posterior when propagating prices

        Each row contains a paired $(\theta,\lambda)$ and one posterior weight.
        Mixing the two marginal coordinates independently would erase the
        trade-off between diffusion variance and jump intensity. The independent
        grid contours below describe the numerical reference density; weighted
        proposal points describe this live approximation.

        For each target contract, apply the accepted deterministic price map
        to every paired row and retain its weight. Weighted 5%, 50% and 95%
        price quantiles describe uncertainty induced by the parameter posterior.
        They are not quantiles of individual option payoffs, empirical future
        market prices or Monte Carlo uncertainty in the estimated quantiles.
        """)]
    if live_result is not None:
        _items.append(plot_helpers.live_joint_figure(live_result))
        _items.append(plot_helpers.live_target_figure(live_result))
        _items.append(mo.ui.table([
            {**_contract, "Corrected 5%": live_result["summary"]["target_quantiles"][_index][0],
             "Corrected median": live_result["summary"]["target_quantiles"][_index][1],
             "Corrected 95%": live_result["summary"]["target_quantiles"][_index][2],
             "Reference median": live_result["reference_summary"]["target_quantiles"][_index][1]}
            for _index, _contract in enumerate(live_result["targets"])], selection=None))
        _items.append(mo.ui.table([
            {**_contract, "5% quantile block MCSE": live_result["summary"]["uncertainty"]["target_quantile_block_mcse"][_index][0],
             "Median block MCSE": live_result["summary"]["uncertainty"]["target_quantile_block_mcse"][_index][1],
             "95% quantile block MCSE": live_result["summary"]["uncertainty"]["target_quantile_block_mcse"][_index][2]}
            for _index, _contract in enumerate(live_result["targets"])], selection=None))
        _items.append(mo.md("Block MCSEs describe estimated sampling variability from eight proposal blocks; they are not confidence bounds for total bias or numerical pricing error."))
    else:
        _items.append(mo.callout("No live weighted posterior has been computed.", kind="info"))
    mo.vstack(_items)
    return


@app.cell
def _(live_result, mo):
    _items = [mo.md(r"""
        ## What was cached, and what did this machine compute?

        Precomputed assets are the original training bank/checkpoint, the
        accepted Fourier reference grid, the target-price surrogate and the
        validation/new-test audit reports. Live work loads compatible assets,
        samples and evaluates the proposal density, computes nine Fourier
        likelihood prices per row, normalizes weights, predicts two target
        prices, and compares with the cached grid.

        Loading/framework compilation, proposal sampling, density evaluation,
        likelihood pricing and downstream prediction have separate timings.
        The total below belongs to this local execution. It is not a warm
        repeated-throughput benchmark or a speed comparison with a new grid
        calculation. Both the reference grid and the neural checkpoint are
        reusable assets; no purely amortized speed benefit is assumed.
        """)]
    if live_result is not None:
        _items.append(mo.ui.table([{"Stage": _stage, "Current-machine seconds": _seconds}
                                   for _stage, _seconds in live_result["timing_seconds"].items()], selection=None))
        _items.append(mo.md("`proposal_total` includes sampling and density; `live_total` includes all stages. These overlapping rows must not be summed."))
    mo.vstack(_items)
    return


@app.cell
def _(cached_figures, mo, protocol, reports):
    _items = [mo.md(r"""
        ## Conditional validation and the new independent test

        The original 16 validation cases and four stress cases were already
        inspected during development. Their improvement is useful, but not
        independent confirmation. Only after every new validation gate passes
        does the protocol generate a separate 256-panel test under a new seed.
        Its first 16 cases receive conditional reference checks, and all 256
        receive the finite-pool calibration audit. The old posterior final test
        remains unevaluated.

        Parameter means, widths, joint correlation, interval mass and target
        price quantiles have separate gates. Both 2,048 and 8,192 conditional
        proposal budgets must pass; a larger-budget result cannot certify a
        cheaper participant calculation by itself.
        """)]
    for _split in ("validation", "test"):
        if _split in reports:
            _report = reports[_split]
            _items.append(mo.md(f"**{_split.capitalize()} participant-budget gates**"))
            _items.append(mo.ui.table([
                {"Metric": _name, "Recorded value": _report.get("metrics", {}).get(_name),
                 "Threshold": _threshold, "Passed": _report.get("gates", {}).get(_name)}
                for _name, _threshold in _report["thresholds"].items()], selection=None))
    if "price-errors" in cached_figures:
        _items.append(cached_figures["price-errors"])
    if "proposal-efficiency" in cached_figures:
        _items.append(cached_figures["proposal-efficiency"])
    mo.vstack(_items)
    return


@app.cell
def _(cached_figures, mo, reports):
    _items = [mo.md(r"""
        ## SBC audits the actual finite-pool resampling algorithm

        Each calibration case uses 2,048 mixture proposals, followed by 512
        joint multinomial resamples from its normalized weights. Rank the known
        synthetic truth among those draws and compare the randomized marginal
        ranks with uniform. This is sampling-importance-resampling (**SIR**),
        not direct IID sampling from an exact posterior.

        Conditional on one finite pool, resampling draws from its empirical
        distribution. After integrating over that shared random pool, the draws
        are dependent. The exact-posterior rank-uniformity result is a diagnostic
        benchmark, not a theorem guaranteeing uniform ranks for this approximate
        algorithm. Duplicate resampled rows are expected and not silently removed.

        DKW limits control the two marginal rank checks within each reported
        split at family $\alpha=0.05$. Passing marginal SBC does not certify
        correct dependence or every conditional price distribution. The earlier
        raw-neural result demonstrates why the independent conditional checks
        are also required. No result here certifies real-market adequacy.
        """)]
    if "calibration" in cached_figures:
        _items.append(cached_figures["calibration"])
    else:
        _skipped = [f"{_split}: {_report['calibration']['skipped']}" for _split, _report in reports.items()
                    if _report.get("calibration", {}).get("skipped")]
        _items.append(mo.callout("Calibration was not run: " + "; ".join(_skipped) if _skipped else
                                "No corrected calibration result is recorded yet.", kind="info"))
    if "reference-pit" in cached_figures:
        _items.append(mo.md(r"""
            **Interpreting the held-out failure.** On the same 256 panels,
            the numerical-reference marginal CDF evaluated at each truth has
            nearly the same long-run-variance shift: KS 0.10349 versus 0.10325
            for the SIR ranks, both above 0.09251. The 65-node comparison gives
            0.10340. This post-hoc calculation uses saved observations and the
            existing quadrature grid; it generates no new test or proposal draws.

            The agreement suggests variation shared by this finite cohort,
            rather than demonstrating proposal-specific bias. A calibrated
            method can occasionally fail a finite-sample diagnostic. The
            numerical reference is still approximate, and this explanation
            does not override the frozen gate. The corrected pipeline remains
            provisional; this review skips the live participant example.
            """))
        _items.append(cached_figures["reference-pit"])
    mo.vstack(_items)
    return


@app.cell
def _(ROOT, artifact_paths, mo):
    _links = "\n    ".join(f"- [{_name}]({_path})" for _name, _path in artifact_paths.items())
    mo.md(f"""
    ## Review trail and exercises

    - [Boundary-correction protocol]({ROOT / 'docs/hkde-correction-boundary-protocol.md'}) and [frozen settings]({ROOT / 'configs/hkde_correction_boundary_protocol.json'}).
    - [Preserved first-correction protocol]({ROOT / 'docs/hkde-correction-protocol.md'}) and [failed v1 evidence]({ROOT / 'docs/results/hkde-correction-v1/validation.json'}).
    - [Original HKDE inference]({ROOT / 'docs/hkde-joint-inference.md'}).
    - [Independent forward-surrogate checks]({ROOT / 'docs/hkde-price-surrogate.md'}).
    {_links}

    Run from the repository root with `PYTHONPATH=src pixi run marimo edit notebooks/10_hkde_corrected_workflow.py --no-token`.

    1. Identify the observation, known context, inferred quantities and synthetic truth. Which may enter the conditioning input?
    2. Why does a proposal drawn from the prior or boundary component still use the full mixture density in its denominator? Why does the posterior prior stay uniform?
    3. Why can high ESS coexist with a missed mode? Which independent calculation checks conditional accuracy here?
    4. Why must θ and λ remain paired when evaluating target prices?
    5. Distinguish error from raw neural inference, finite-pool importance sampling, numerical Fourier pricing and the forward surrogate.
    6. Which parts are precomputed, which run live, and which would have to change for an American option or a returns-based P forecast?
    7. If a numerical reference and an approximate sampler show the same rank shift on one finite cohort, what does this establish—and what remains uncertain?

    This draft **synthetic options lesson** leaves the final curriculum
    choice open. Optional market application would need compatible observations
    and context; empirical claims need separately held-out market outcomes.
    Neither is required for this offline lesson, and neither is supplied by
    labelling synthetic fixtures as real.
    """)
    return


if __name__ == "__main__":
    app.run()
