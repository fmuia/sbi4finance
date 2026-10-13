import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import importlib.util
    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from sbi4finance import hkde_surrogate

    ROOT = mo.notebook_dir().parent
    _spec = importlib.util.spec_from_file_location("hkde_surrogate_plots", ROOT / "scripts/plot_hkde_surrogate.py")
    plot_helpers = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(plot_helpers)
    return ROOT, hkde_surrogate, json, mo, np, plot_helpers, plt


@app.cell
def _(mo):
    mo.md(r"""
    # 7. Learning a price map: a separate building block from calibration

    The preceding HKDE exercise learned an inverse map: **observed option
    panel → joint parameter posterior**. This milestone learns a forward
    map: **parameters → prices of two fixed European puts**. It makes repeated
    downstream pricing cheaper when the learned map is sufficiently accurate.

    A fitted price map does not return a posterior, infer parameters from
    observed quotes, or repair an inaccurate calibration network. Here it is a
    small polynomial regression, with no neural framework or uncertainty model
    for its coefficients. This is synthetic **S evidence under Q**, not a
    physical-measure forecast or a claim about actual market prices.

    Default execution reads saved results, fits a tiny **analytic teaching
    polynomial**, and optionally evaluates a saved surrogate. It performs no
    Fourier pricing, bank generation, neural inference or final-test evaluation.
    The expensive scientific experiment is run separately and reviewed here.
    """)
    return


@app.cell
def _(ROOT, json, plot_helpers):
    protocol = json.loads((ROOT / "configs/hkde_surrogate_protocol.json").read_text())
    task = json.loads((ROOT / protocol["task_config"]).read_text())["task"]
    targets = protocol["targets"]
    report, checks, report_paths = plot_helpers.load_cache(ROOT / "docs/results/hkde-surrogate")
    cached_figures = plot_helpers.figures(report, checks) if report is not None else {}
    baseline = json.loads((ROOT / protocol["baseline_results"] / "neural-validation.json").read_text())
    return baseline, cached_figures, checks, protocol, report, report_paths, targets, task


@app.cell
def _(mo, targets, task):
    mo.vstack([
        mo.md(r"""
        ## The task is deliberately small and explicit

        HKDE combines Heston stochastic variance with double-exponential price
        jumps. Only the long-run variance $\theta$ and jump intensity $\lambda$
        vary in this exercise. Mean reversion, volatility of variance, leverage,
        jump-size distribution, initial variance, spot and carry are known and
        fixed to the existing HKDE task. Time is in years and prices in currency.

        The two output coordinates correspond to the ordered contracts below.
        They are bound to the saved model. This is not yet a general pricer
        accepting arbitrary strike, maturity or payoff descriptions. Changed
        context/contracts and parameters outside the declared box are rejected.
        """),
        mo.ui.table([{"Parameter": _name, "Lower bound": _bounds[0], "Upper bound": _bounds[1]}
                     for _name, _bounds in zip(task["parameter_names"], task["prior_bounds"])], selection=None),
        mo.ui.table(targets, selection=None),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Where do supervised price labels come from?

    Draw a parameter pair $a=(\theta,\lambda)$ from the declared box. For
    contract $c$, the reference label approximates the discounted expectation

    $$P_c(a)=E_Q[e^{-rT_c}(K_c-S_{T_c})^+\mid a].$$

    The existing Fourier pricer evaluates that map. The frozen experiment uses
    1,024 quadrature nodes and cutoff 400, with a separate refinement check at
    2,048 nodes and cutoff 800. The label is not an exact continuum oracle;
    its numerical accuracy must be checked independently of regression error.

    **No market quote noise and no individual payoff enter these labels.**
    A single simulated payoff is one random outcome, not a price. Observed
    market prices would introduce a different measurement/calibration problem.
    Synthetic parameters are known training inputs here, whereas they were
    unknown posterior targets in the inverse-calibration task.

    Both contract labels for a parameter pair stay in one split. The protocol
    uses 512 independent training pairs and 256 separate validation pairs,
    plus an existing 129×129 grid as a predeclared stress set. No grid row is
    included in the regression fit. A new 256-pair surrogate test split is
    evaluated only after validation passes; it is separate from the still
    reserved posterior-inference test.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## A learned map need not start with a neural network

    First scale each known parameter interval $[l_d,u_d]$ to $[-1,1]$:

    $$z_d=2\frac{a_d-l_d}{u_d-l_d}-1.$$

    This uses declared task bounds, not statistics from validation or test
    data. Chebyshev polynomials satisfy $T_0(z)=1$, $T_1(z)=z$ and
    $T_{k+1}(z)=2zT_k(z)-T_{k-1}(z)$. Form their tensor products and fit

    $$\widehat P_c(a)=\sum_{i=0}^{6}\sum_{j=0}^{6}
       \beta_{ijc}T_i(z_1)T_j(z_2).$$

    There are $(6+1)^2=49$ coefficients per output. If $A$ is the 512×49
    basis matrix and $B$ contains the two coefficient columns, a single
    float64 least-squares solve minimizes $\|AB-P\|_F^2$. Full rank and the
    condition number are checked: a small training residual alone does not
    establish a reliable fit. Predictions are not clipped to put-price bounds,
    so violations remain visible in validation.

    This smooth, two-input problem lets us test the price-learning responsibility
    with a compact fit before introducing a larger architecture. It does not
    establish that this basis suffices for a larger parameter box, arbitrary
    contracts, rough-volatility models or American exercise.
    """)
    return


@app.cell
def _(mo, task):
    theta_control = mo.ui.slider(task["prior_bounds"][0][0], task["prior_bounds"][0][1],
                                step=.001, value=.045, show_value=True, label="Long-run variance θ")
    intensity_control = mo.ui.slider(task["prior_bounds"][1][0], task["prior_bounds"][1][1],
                                    step=.025, value=.625, show_value=True, label="Jump intensity λ per year")
    mo.vstack([
        mo.md(r"""
        ## Live mechanics: a known polynomial, not an option-pricing result

        We fit the same API at degree two to 32 independent parameter pairs.
        The two analytic teaching functions are

        $$f_1(z)=4+0.2z_1+0.3z_2+0.1z_1z_2+0.05z_1^2,$$
        $$f_2(z)=1.5+0.1z_1+0.2z_2-0.04z_2^2.$$

        They are exactly representable by this smaller basis. The held-out
        comparison checks fitting, scaling and prediction mechanics; these
        outputs are **not HKDE prices** and do not support the scientific gates.
        """),
        mo.hstack([theta_control, intensity_control]),
    ])
    return intensity_control, theta_control


@app.cell
def _(hkde_surrogate, np, targets, task):
    toy_bounds = np.asarray(task["prior_bounds"])

    def toy_values(parameters):
        z = 2 * (np.asarray(parameters) - toy_bounds[:, 0]) / (toy_bounds[:, 1] - toy_bounds[:, 0]) - 1
        return np.column_stack([4 + .2*z[:, 0] + .3*z[:, 1] + .1*z[:, 0]*z[:, 1] + .05*z[:, 0]**2,
                                1.5 + .1*z[:, 0] + .2*z[:, 1] - .04*z[:, 1]**2])

    _rng = np.random.default_rng(97001)
    _train = _rng.uniform(toy_bounds[:, 0], toy_bounds[:, 1], size=(32, 2))
    toy_model = hkde_surrogate.fit(task, targets, _train, toy_values(_train), degree=2)
    toy_heldout = _rng.uniform(toy_bounds[:, 0], toy_bounds[:, 1], size=(48, 2))
    toy_heldout_error = hkde_surrogate.predict(toy_model, toy_heldout, task=task, contracts=targets) - toy_values(toy_heldout)
    return toy_bounds, toy_heldout, toy_heldout_error, toy_model, toy_values


@app.cell
def _(hkde_surrogate, intensity_control, mo, np, targets, task, theta_control, toy_heldout_error, toy_model, toy_values):
    selected_parameters = np.array([[theta_control.value, intensity_control.value]])
    _predicted = hkde_surrogate.predict(toy_model, selected_parameters, task=task, contracts=targets)[0]
    _actual = toy_values(selected_parameters)[0]
    mo.vstack([
        mo.ui.table([{"Analytic teaching output": _i + 1, "Known polynomial": _actual[_i],
                      "Fitted value": _predicted[_i], "Difference": _predicted[_i] - _actual[_i]}
                     for _i in range(2)], selection=None),
        mo.md(f"Maximum absolute error on 48 separate toy pairs: **{np.max(np.abs(toy_heldout_error)):.3g}**. "
              "Recovering a polynomial already contained in the basis checks mechanics, not financial approximation accuracy."),
    ])
    return (selected_parameters,)


@app.cell
def _(ROOT, hkde_surrogate, mo, selected_parameters, targets, task):
    _model_path = ROOT / "docs/results/hkde-surrogate/model.npz"
    if _model_path.exists():
        _saved_model = hkde_surrogate.load_model(_model_path)
        _prices = hkde_surrogate.predict(_saved_model, selected_parameters, task=task, contracts=targets)[0]
        _view = mo.vstack([
            mo.md("### Saved HKDE surrogate at the same slider values\n\nThese are learned prices from the published fit. No reference Fourier call is made by this cell; consult its independent error checks below."),
            mo.ui.table([{**_contract, "Surrogate price (currency)": _price}
                         for _contract, _price in zip(targets, _prices)], selection=None),
        ])
    else:
        _view = mo.callout("No saved HKDE surrogate is present. Only the explicitly labelled analytic teaching example has been fitted here.", kind="info")
    _view
    return


@app.cell
def _(cached_figures, mo, report):
    _items = [mo.md("## Measured approximation error\n\nValidation, grid-boundary stress, Fourier-label refinement and put bounds address different possible failures. A finite set of successful comparisons does not prove a uniform error bound over the entire parameter box.")]
    if report is not None:
        _items.append(mo.ui.table([
            {"Stage": "Surrogate validation", "Recorded status": "PASS" if report["validation_passed"] else "FAIL"},
            {"Stage": "New surrogate test", "Recorded status":
             ("PASS" if report["test_passed"] else "FAIL") if report["test_evaluated"] else "not evaluated: validation gate"},
            {"Stage": "Old posterior final test", "Recorded status": "reserved; not evaluated by this component"},
        ], selection=None))
        _items.append(mo.ui.table([{"Fit quantity": _name, "Recorded value": _value}
                                   for _name, _value in report["fit"].items()], selection=None))
        _items.append(mo.ui.table([
            {"Price-check set": _name, "Maximum absolute error (currency)": _summary["max_abs_error_currency"],
             "Largest per-contract RMSE (currency)": max(_summary["rmse_by_contract"])}
            for _name, _summary in report["errors"].items()], selection=None))
        _items.append(mo.md("The gate metrics below aggregate validation, stress-grid and paired-propagation checks before opening the new surrogate test. They exclude that test; its measured errors and separate status are shown above."))
        _items.append(mo.ui.table([
            {"Gate": _name, "Recorded value": report["metrics"].get(_name),
             "Threshold": _limit, "Passed": report["gates"].get(_name)}
            for _name, _limit in report["thresholds"].items()], selection=None))
        if "parameter-errors" in cached_figures:
            _items.append(cached_figures["parameter-errors"])
    else:
        _items.append(mo.callout("No scientific surrogate report is present. The analytic toy is not a substitute for this result.", kind="info"))
    mo.vstack(_items)
    return


@app.cell
def _(baseline, cached_figures, mo):
    _old_error = baseline["metrics"]["max_abs_target_quantile_error_currency"]
    _old_gate = baseline["thresholds"]["max_abs_target_quantile_error_currency"]
    _items = [mo.md(r"""
        ## Substitute the pricer on the same joint parameter draws

        For saved paired posterior draws $a_k\sim q_\phi(a\mid y)$, compare
        $\widehat P(a_k)$ with the Fourier value $P_F(a_k)$ **using exactly
        the same draws**. This isolates forward-surrogate substitution error
        from randomness in a newly sampled posterior. Preserve the θ–λ pairing;
        independently sampling marginal coordinates changes the joint law.

        The old calibration diagnostic instead compares price quantiles from
        $P_F(a_k)$ with those obtained under the independently weighted reference
        posterior. That difference concerns the **inference approximation**.
        Means and quantiles from the new surrogate are also compared with the
        reference, so the combined result remains visible.

        A faster accurate map cannot repair a wrong distribution of its inputs.
        A small observed pointwise substitution error limits quantile shifts on
        those same finite draws, but the validation grid is not a proof of a
        continuum-uniform bound. There is no new SBC claim here: deterministic
        price regression needs held-out price checks, while calibration remains
        a separate Bayesian inference task.
        """),
        mo.callout(f"The existing HKDE neural inference result remains provisional: its worst target-price quantile error was {_old_error:.5f} currency, exceeding {_old_gate:.2f}. Accepting a separate surrogate does not change that recorded failure or unlock the old posterior final test.", kind="warn")]
    if "paired-errors" in cached_figures:
        _items.append(cached_figures["paired-errors"])
    mo.vstack(_items)
    return


@app.cell
def _(cached_figures, mo, report):
    _items = [mo.md(r"""
        ## Cost is part of the component check

        Creating labels, fitting coefficients, saving/loading a model and
        answering queries are different costs. The benchmark warms each route
        once, then repeats direct Fourier and surrogate pricing on the same
        4,096 parameter pairs. Reading cached labels is not counted as direct
        pricing. Fresh file load plus prediction is reported separately.

        Any break-even must pay back measured setup cost using measured query
        savings. These timings exclude the unchanged posterior-inference cost
        and do not imply that the whole tutorial pipeline is accepted or faster.
        The direct comparator is the current looped Fourier implementation;
        the measured ratio is specific to these inputs, code and hardware.
        A Fourier pricer remains useful for labels, audits, changed tasks and
        cases outside the surrogate's supported box.
        """)]
    if report is not None and "timing" in cached_figures:
        _items.append(cached_figures["timing"])
        _items.append(mo.ui.table([
            {"Measured stage": _name, "Seconds": _value}
            for _name, _value in report["timings_seconds"].items()], selection=None))
        _amortization = report["amortization"]
        _setup_break_even = _amortization["training_setup_break_even_4096_row_queries"]
        _full_break_even = _amortization["complete_run_break_even_4096_row_queries"]
        _items.append(mo.md(
            f"Measured fit-setup break-even: **{_setup_break_even:.3g}** batches of 4,096 parameter pairs; "
            f"including this complete surrogate experiment: **{_full_break_even:.3g}** batches. "
            "These fractional batch equivalents exclude the previously cached reference and inference studies. "
            "File-load timing is within the current process, not full process startup."
            if _setup_break_even is not None and _full_break_even is not None else
            "No positive measured query saving supports a break-even estimate for this run."))
    else:
        _items.append(mo.callout("Measured pricing-cost results are not available yet.", kind="info"))
    mo.vstack(_items)
    return


@app.cell
def _(ROOT, mo, report_paths):
    _links = "\n    ".join(f"- [{_name}]({_path})" for _name, _path in report_paths.items())
    mo.md(f"""
    ## Review and reproduce

    - [Frozen scientific protocol]({ROOT / 'docs/hkde-surrogate-protocol.md'}).
    - [Machine-readable settings]({ROOT / 'configs/hkde_surrogate_protocol.json'}).
    - [Earlier HKDE inference result]({ROOT / 'docs/results/hkde-inference/neural-validation.json'}).
    - [Forward-surrogate implementation]({ROOT / 'src/sbi4finance/hkde_surrogate.py'}).
    {_links}

    Open this offline review from the repository root:

    ```bash
    PYTHONPATH=src pixi run marimo edit notebooks/07_hkde_price_surrogate.py --no-token
    ```

    Render the saved scientific figures into a new directory:

    ```bash
    PYTHONPATH=src pixi run python scripts/plot_hkde_surrogate.py \\
      --results docs/results/hkde-surrogate --out /tmp/hkde-surrogate-plots
    ```

    This component does not require real quotes. Applying it to a market-calibrated
    posterior would require compatible measure/context and separate model-adequacy
    assessment. Extending the input to contracts, replacing Fourier labels with
    validated Monte Carlo labels, or learning American/exotic prices are new
    tasks with their own numerical and regression checks.

    **Exercises**

    1. Why is the training label an expectation rather than one payoff or a noisy quote?
    2. Why can known-bound scaling be reused on the test set without leaking test observations?
    3. Which error is isolated by passing identical joint draws through two pricers?
    4. If the surrogate is almost exact but its input posterior is wrong, which diagnostic will still fail?
    5. What must change before this fixed two-output checkpoint can price a different strike or an American put?
    """)
    return


if __name__ == "__main__":
    app.run()
