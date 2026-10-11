import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from sbi4finance import hkde, option_bank

    ROOT = mo.notebook_dir().parent
    return ROOT, hkde, json, mo, np, option_bank, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    # 6. HKDE: from a jump law to an option-panel observation

    The reusable building block here is a **risk-neutral simulator/pricer and
    synthetic option bank**. A small numerical example is not an accepted
    posterior. Heston returns inference remains provisional; this independent
    options branch can advance without silently passing its failed checks.

    This notebook uses hypothetical contracts, rates and parameters. It runs
    offline, with no market files, neural training or cluster submission.
    Recorded reports describe **S: synthetic numerical evidence** only. The
    eventual live tutorial may still choose returns, options, or a carefully
    separated pair of examples.

    ## Why pricing code appears in an SBI project

    A market option quote estimates an expectation, not one randomly realized
    future payoff. To simulate a quote panel, draw a Q parameter vector, compute
    model prices, then apply a declared observation-error law. Later SBI can
    reverse that experiment: noisy panel → joint parameter posterior.

    | Workflow | Inputs | Training target or output |
    |---|---|---|
    | Reference pricing, implemented building block | Parameters, contracts, context | Discounted expected payoff |
    | Parameter inference, M5.1 review below | Noisy panel and known context | Joint Q parameters |
    | Pricing surrogate, deferred | Parameters, contracts, context | Accurate reference-price labels |
    | Direct price posterior, deferred | Noisy panel and context | Target prices generated from the same parameters |

    A posterior can be pushed through the reference pricer before a surrogate
    exists. A surrogate approximates the pricing map; it does not eliminate the
    need for trustworthy training prices. No workflow here forecasts tomorrow's
    market option chain or supplies physical-measure risk probabilities.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Heston variance plus double-exponential price jumps

    Let $X=\log(S/S_0)$, and measure time in years. Under Q,

    $$dV_t=\kappa(\theta-V_t)dt+\xi\sqrt{V_t}\,dW_t^V,$$
    $$dX_t=(r-q-\lambda c_J-V_t/2)dt+\sqrt{V_t}\,dW_t^S+J\,dN_t,
    \qquad d\langle W^S,W^V\rangle_t=\rho\,dt.$$

    The jump count has intensity $\lambda$ per year. Each log jump is positive
    with probability $p_+$; its magnitude then has exponential rate $a_+$.
    Otherwise it is minus an exponential with rate $a_-$. Thus

    $$f_J(j)=p_+a_+e^{-a_+j}\mathbf1_{j\ge0}
       +(1-p_+)a_-e^{a_-j}\mathbf1_{j<0},$$
    $$c_J=\mathbb E[e^J-1]
       =\frac{p_+a_+}{a_+-1}+\frac{(1-p_+)a_-}{a_-+1}-1.$$

    Rates are inverse sizes: mean upward log-jump magnitude is $1/a_+$.
    With positive upward mass, $a_+>1$ is needed for the compensator.
    Zero-probability terms are omitted. Subtracting $\lambda\mathbb E[J]$
    would not compensate proportional price growth.

    HKDE and Gaussian-jump Heston are sibling choices of jump law; neither
    adds jumps to variance. The Q parameters are pricing parameters. Changing
    a fitted P drift to $r-q$ does not identify the other Q parameters or a
    measure-change model.

    For $F_T=S_0e^{(r-q)T}$ and $D_T=e^{-rT}$, European prices are
    $C=D_T\mathbb E_Q[(S_T-K)^+]$ and
    $P=D_T\mathbb E_Q[(K-S_T)^+]$, with $C-P=D_T(F_T-K)$.
    Fourier inversion evaluates these expectations from the model's
    characteristic function; it has its own integration truncation and
    quadrature errors. It is not the neural inference step. This implementation
    uses direct Fourier quadrature, not the paper's PROJ or barrier pricer.
    """)
    return


@app.cell
def _(mo):
    jump_control = mo.ui.slider(0.0, 1.5, step=0.1, value=0.5,
                                label="Jump intensity λ / year", show_value=True)
    noise_control = mo.ui.slider(0.0, 0.3, step=0.025, value=0.1,
                                 label="Quote-noise SD, price units", show_value=True)
    mo.vstack([
        mo.md("## A small synthetic panel\n\nSix contracts at two maturities; spot 100, annual rate 2%, dividend yield zero. Controls are illustrations, not a new validated prior."),
        mo.hstack([jump_control, noise_control]),
    ])
    return jump_control, noise_control


@app.cell
def _(hkde, jump_control, noise_control, np):
    parameters = dict(kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.6,
                      jump_intensity=jump_control.value, jump_up_probability=0.3,
                      jump_up_rate=12.0, jump_down_rate=8.0)
    maturity = np.repeat([0.25, 1.0], 3)
    strike = np.tile([90.0, 100.0, 110.0], 2)
    forward = 100 * np.exp(0.02 * maturity)
    discount = np.exp(-0.02 * maturity)
    clean_put = hkde.european_price(forward, strike, discount, maturity,
                                    parameters, 0.04, kind="put")
    clean_call = hkde.european_price(forward, strike, discount, maturity,
                                     parameters, 0.04, kind="call")
    quote_noise = np.random.default_rng(6401).normal(0, noise_control.value, size=6)
    noisy_put = clean_put + quote_noise
    return clean_call, clean_put, discount, forward, maturity, noisy_put, parameters, strike


@app.cell
def _(clean_put, maturity, noisy_put, np, plt, strike):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True, layout="constrained")
    for _ax, _time in zip(_axes, np.unique(maturity), strict=True):
        _mask = maturity == _time
        _ax.plot(strike[_mask], clean_put[_mask], "o-", label="Model expectation estimate")
        _ax.scatter(strike[_mask], noisy_put[_mask], marker="x", s=70,
                    label="Observed synthetic quote", color="#bb653f")
        _ax.set(xlabel="Strike, price units", title=f"Maturity {_time:g} years")
    _axes[0].set_ylabel("European put price, price units")
    _axes[1].legend(fontsize=8)
    _fig
    return


@app.cell
def _(clean_call, clean_put, discount, forward, hkde, maturity, mo, np, parameters, pd, strike):
    _refined = hkde.european_price(forward, strike, discount, maturity, parameters,
                                  0.04, kind="put", upper_limit=300, nodes=768)
    _parity = clean_call - clean_put - discount * (forward - strike)
    _table = pd.DataFrame({
        "Maturity (years)": maturity, "Strike": strike,
        "Clean put": clean_put, "Refined put": _refined,
        "Absolute refinement difference": np.abs(_refined - clean_put),
        "Put-call parity residual": _parity,
    })
    mo.vstack([
        mo.md("""
        ## Numerical error and observation noise answer different questions

        This local comparison changes the Fourier integration cutoff and node
        count. Small differences support this particular panel; two numerical
        settings agreeing is not an independent proof of accuracy. Put–call
        parity checks consistency, and may be enforced by the pricing route
        itself. The recorded validation below includes the separate checks.

        Gaussian quote errors in the illustration are intentional measurement
        noise, in price units. They are not estimates of pricing error. A
        Gaussian model can produce negative or arbitrage-inconsistent observed
        quotes; clipping them would change that observation law. The clean
        model prices must obey their own financial/numerical checks.
        """),
        mo.ui.table(_table, selection=None, show_column_summaries=False),
    ])
    return


@app.cell
def _(ROOT, json, mo, pd):
    hkde_report = json.loads((ROOT / "docs/results/hkde-validation.json").read_text())
    _rows = []
    for _case in hkde_report["pricing_fixtures"]:
        _rows.append({
            "Fixture": _case["name"],
            "Largest Fourier refinement difference / DF": max(c["cutoff_node_refinement_max_error"] for c in _case["contracts"]),
            "Independent inversion difference / DF": max(c["independent_payoff_inversion_max_error"] for c in _case["contracts"]),
            "Riccati reference CF error": _case["riccati_reference_max_error"],
            "Recorded result": "PASS" if _case["passed"] else "FAIL",
        })
    mo.vstack([
        mo.md("""
        ## Recorded evidence: read the scope as well as the numbers

        This is the saved numerical report, not a computation launched when
        opening the notebook. Its cases, refinement settings, independent
        references and tolerances delimit the evidence. A passed software
        check or a small pilot bank does not validate every point of a larger
        prior. No posterior, learned surrogate or empirical market assessment
        is accepted by this report.
        """),
        mo.ui.table(pd.DataFrame(_rows), selection=None, show_column_summaries=False),
        mo.md(f"""
        The Fourier refinement budget is
        **{hkde_report['fixed_budgets']['fourier_refinement']:g} of DF**,
        with forward and discount both one in the fixtures. These normalized
        units differ from the live spot-100 panel's currency units. The
        Riccati characteristic-function comparison budget is
        **{hkde_report['fixed_budgets']['riccati_reference']:g}**.
        Overall recorded numerical result:
        **{'PASS' if hkde_report['passed'] else 'FAIL'}** within the saved scope.
        Path Monte Carlo in the full report retains correlated price errors
        and a separate diffusion-refinement allowance; those allowances are
        not certified bounds on unknown continuous-model bias.
        """),
        mo.accordion({"Full report, path references and limitations": mo.json(hkde_report)}),
    ])
    return (hkde_report,)


@app.cell
def _(ROOT, json, mo, option_bank, pd):
    hkde_bank_config = json.loads((ROOT / "configs/tasks/hkde_option_bank.json").read_text())
    _example = option_bank.generate_examples(hkde_bank_config, "train", [0])
    _table = pd.DataFrame({
        "Contract index": range(_example["observations"].shape[1]),
        "Observed noisy quote": _example["observations"][0, :, 0],
        "Reference price (not observed)": _example["reference_prices"][0],
        "Refinement difference (not observed)": _example["refinement_difference"][0],
    })
    mo.vstack([
        mo.md("""
        ## One example from the shared bank generator

        This executes exactly one seeded training example from the committed
        configuration; it neither writes a bank nor trains a network. The
        recorded task fixes nine put contracts: strikes 90/100/110 at
        0.25/0.5/1 years, spot 100, rate 2%, dividend zero and Gaussian
        quote-noise SD 0.25 price units. It varies theta and jump intensity;
        the remaining HKDE parameters are fixed. The configuration below is
        the authority for prior bounds, target order and numerical settings.
        Its restricted prior is not the full supported model domain.
        """),
        mo.ui.table(_table, selection=None, show_column_summaries=False),
        mo.json(hkde_bank_config),
    ])
    return (hkde_bank_config,)


@app.cell
def _(mo):
    mo.md(r"""
    ## What is one observation, and what belongs in a bank?

    One calibration example is **one complete option panel** at one synthetic
    valuation date. Its contracts have known strikes, maturities, exercise
    type, forwards and discount factors. The target is a selected joint Q
    parameter vector. A clean price, simulated jump realization, exact
    parameter or numerical error estimate is not an extra observed quote.
    In this task, batch arrays are `observations[B, 9, 1]` and
    `parameters[B, 2]` ordered as theta and jump intensity. Initial variance
    0.04 is known context. `reference_prices[B, 9]` and numerical diagnostics
    remain separate from the observed channel.

    | Role | Stored content | May condition a later estimator? |
    |---|---|---|
    | Observations | Noisy quotes in the declared contract order | Yes |
    | Known context | Contracts, synthetic rates/spot or forward/discount inputs | Yes, within the task's declared support |
    | Truth labels | Drawn parameters and clean model-price labels | Targets or checks; not hidden input channels |
    | Numerical diagnostics | Refinement residuals, Monte Carlo covariance if used | No, unless a separate observed-error task declares them |
    | Identity and provenance | Base-surface ID, split, task/source hashes | Integrity and grouped splitting |

    Draw each base surface into exactly one training, validation or test split
    before creating quote-noise augmentations. A shared bank architecture can
    retain these roles for different models while keeping their parameter and
    numerical contracts distinct. An option panel is not a return sequence.

    **S** needs no real market observation. Optional **M** conditions a trained
    network on a compatible actual panel, normally with frozen weights and
    without parameter labels. Optional **E** compares to contemporaneous held-out
    quotes under a leakage-free selection/preprocessing design. Target market
    quotes are not required merely to generate a posterior distribution of
    model prices. A single market case does not establish general performance.

    ## Review exercises

    1. If $a_+$ decreases toward one, why does a small change in a tail
       parameter threaten pricing? Inspect the exponential moment, not just
       the mean log jump.
    2. Could one simulated terminal payoff replace each clean price label?
       Explain which expectation and numerical uncertainty would be lost.
    3. If two noisy versions of the same clean surface enter different splits,
       what does validation reuse? Keep the underlying parameter/context draw
       and all its quote augmentations together.

    Sources and reproduction: [task configuration](../configs/tasks/hkde_option_bank.json),
    [HKDE source](../src/sbi4finance/hkde.py),
    [shared bank source](../src/sbi4finance/option_bank.py),
    [numerical report](../docs/results/hkde-validation.json),
    [reference design and reproduction](../docs/hkde-reference.md),
    [M4 subplan](../docs/simulator-plan/milestones/04-hkde-and-options.md),
    [mathematical notes](../docs/simulator-plan/notes/02-models-and-numerics.md),
    [inference/pricing workflows](../docs/simulator-plan/notes/03-inference-and-pricing.md),
    [next analysis](../docs/simulator-analysis-next-steps.md), and the
    [HKDE paper](https://arxiv.org/html/2502.13824v1).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## M5.1: infer two parameters, then check the learned posterior

    Now fix all context above and let $z=(\theta,\lambda)$ be unknown.
    The prior is independent uniform on the saved rectangle $\mathcal B$:
    $\theta\in[0.025,0.065]$ and $\lambda\in[0.25,1]$ per year.
    Theta is an annualized variance level, not volatility; $\sqrt\theta$
    gives the corresponding volatility scale. These narrow bounds define
    this experiment, not universal knowledge about markets.

    For the nine ordered quotes, the declared synthetic observation model is

    $$y_j=P_j(z)+\epsilon_j,\qquad
      \epsilon_j\overset{\mathrm{iid}}\sim\mathcal N(0,\sigma_q^2),
      \quad \sigma_q=0.25\ \text{price units},$$
    $$p(z\mid y)\propto\mathbf1_{z\in\mathcal B}
       \exp\left[-\frac{1}{2\sigma_q^2}
                 \sum_{j=1}^{9}(y_j-P_j(z))^2\right].$$

    Here $P_j$ is the saved refined Fourier pricing map. A two-dimensional
    tensor grid makes an independently evaluated numerical posterior feasible.
    If $q_i$ are trapezoidal integration weights including the normalized
    uniform prior, posterior **masses** are
    $w_i=q_iL(y\mid z_i)/\sum_k q_kL(y\mid z_k)$.
    Endpoints get half weights along each axis. A mass is not a density:
    for density contours below, divide by $q_i$ and by the prior-box area.

    Grid refinement checks posterior means, spreads, dependence, target-price
    quantiles and log evidence. Fourier refinement checks a different numerical
    approximation. Neither check validates the neural network automatically.
    The network sees only the nine noisy prices; joint parameter labels are
    training targets. Global scaling is learned from training panels only.
    No simulator truths or clean prices become hidden conditioning channels.

    The frozen protocol permits one small spline-flow baseline. Selection
    validation chooses its checkpoint; separate audit panels check it.
    Reserved test data remain unopened unless all validation gates pass.
    This notebook reads cached reports and reweights cached prices; it does
    not train, load a checkpoint, acquire data or submit a cluster job.
    """)
    return


@app.cell
def _(ROOT, json, mo, np, pd):
    _protocol_path = ROOT / "configs/hkde_inference_protocol.json"
    inference_protocol = json.loads(_protocol_path.read_text()) if _protocol_path.is_file() else None
    _results = ROOT / "docs/results/hkde-inference"
    reference_result = json.loads((_results / "reference.json").read_text()) if (_results / "reference.json").is_file() else None
    neural_result = json.loads((_results / "neural-validation.json").read_text()) if (_results / "neural-validation.json").is_file() else None
    final_result = json.loads((_results / "neural-test.json").read_text()) if (_results / "neural-test.json").is_file() else None
    reference_data, neural_data = None, None
    if (_results / "reference-grid.npz").is_file():
        with np.load(_results / "reference-grid.npz", allow_pickle=False) as _saved:
            reference_data = dict(_saved)
    if (_results / "validation-checks.npz").is_file():
        with np.load(_results / "validation-checks.npz", allow_pickle=False) as _saved:
            neural_data = dict(_saved)
    _records = []
    for _name, _report in (("Numerical posterior reference", reference_result),
                           ("Neural validation audit", neural_result),
                           ("Reserved final test", final_result)):
        _status = "Not recorded" if _report is None else (
            "PASS within reported scope" if _report.get("passed") is True else
            "FAIL / provisional" if _report.get("passed") is False else "See recorded gates")
        if _name == "Reserved final test" and _report is None and neural_result is not None and not neural_result["passed"]:
            _status = "Held back: validation has not passed"
        _records.append({"Stage": _name, "Recorded status": _status})
    _failed = [] if neural_result is None else [_key for _key, _passed in neural_result["gates"].items() if not _passed]
    mo.vstack([
        mo.md("""
        ### Recorded state of the experiment

        Missing artifacts are shown as pending, never replaced by invented
        results. A software check, a numerical reference and a neural audit
        have separate acceptance meanings. All results here concern **S**, the
        tested synthetic experiment; they do not certify a real-market model.
        """),
        mo.ui.table(pd.DataFrame(_records), selection=None, show_column_summaries=False),
        mo.md("**Recorded failed neural gates:** " + (", ".join(f"`{_key}`" for _key in _failed) if _failed else "none recorded; inspect which stages are available.")),
        mo.accordion({"Frozen experiment protocol": mo.json(inference_protocol) if inference_protocol is not None else mo.md("Not recorded."),
                      "Numerical reference report": mo.json(reference_result) if reference_result is not None else mo.md("Not recorded."),
                      "Neural validation report": mo.json(neural_result) if neural_result is not None else mo.md("Not recorded."),
                      "Reserved test report": mo.json(final_result) if final_result is not None else mo.md("Not evaluated; no results are substituted.")}),
    ])
    return final_result, inference_protocol, neural_data, neural_result, reference_data, reference_result


@app.cell
def _(mo, reference_data):
    if reference_data is None:
        reference_case, quote_shift = None, None
        _content = mo.md("**Posterior-grid illustration pending:** no cached reference grid is present.")
    else:
        reference_case = mo.ui.dropdown(
            options={str(_case): _index for _index, _case in enumerate(reference_data["case_ids"])},
            value="validation:257" if "validation:257" in reference_data["case_ids"] else str(reference_data["case_ids"][0]),
            label="Saved conditional case",
        )
        quote_shift = mo.ui.slider(-1.0, 1.0, step=0.25, value=0.0, show_value=True,
                                   label="Shift the one-year ATM quote, in quote-noise SDs")
        _content = mo.vstack([
            mo.md("""
            ### A live Bayesian update using the cached price grid

            The controls reweight saved model prices, so this remains fast and
            offline. Zero shift reproduces the selected observed panel. Moving one
            quote is a sensitivity illustration, not a new recorded acceptance
            result. Later diagnostic plots always show the original saved cases.
            """),
            mo.hstack([reference_case, quote_shift]),
        ])
    _content
    return quote_shift, reference_case


@app.cell
def _(mo, np, pd, plt, quote_shift, reference_case, reference_data, reference_result):
    mo.stop(reference_result is None or reference_data is None,
            mo.md("The reference report and grid are needed for this posterior illustration."))
    _sigma = reference_result["task"]["context"]["quote_noise_std"]
    _case = reference_case.value
    _observed = reference_data["observations"][_case].reshape(-1).copy()
    _observed[7] += quote_shift.value * _sigma
    _q = reference_data["quadrature_weights"]
    _log_mass = np.log(_q) - 0.5 * np.sum(((reference_data["prices"] - _observed) / _sigma) ** 2, axis=1)
    _mass = np.exp(_log_mass - _log_mass.max())
    _mass /= _mass.sum()
    _axis0, _axis1 = reference_data["axis0"], reference_data["axis1"]
    _area = (_axis0[-1] - _axis0[0]) * (_axis1[-1] - _axis1[0])
    _shape = (len(_axis0), len(_axis1))
    _prior_density = np.full(_shape, 1.0 / _area)
    _posterior_density = (_mass / _q / _area).reshape(_shape)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for _ax, _density, _title in zip(_axes, (_prior_density, _posterior_density),
                                     ("Independent uniform prior", "Joint numerical posterior"), strict=True):
        _image = _ax.pcolormesh(_axis0, _axis1, _density.T, shading="auto", cmap="Blues", vmin=0)
        _ax.set(xlabel="Long-run variance θ", ylabel="Jump intensity λ / year", title=_title)
        _fig.colorbar(_image, ax=_ax, label="Density per parameter-area unit")
    _axes[1].contour(_axis0, _axis1, _posterior_density.T, levels=5, colors="#284b63", linewidths=0.7)
    _truth = reference_data["truth"][_case]
    _axes[1].plot(*_truth, marker="x", color="#b75d38", markersize=8, label="Synthetic truth (check only)")
    _axes[1].legend(fontsize=7)
    _parameters = reference_data["parameters"]
    _mean = _mass @ _parameters
    _std = np.sqrt(_mass @ ((_parameters - _mean) ** 2))
    _bounds = np.asarray(reference_result["task"]["prior_bounds"])
    _prior_std = np.diff(_bounds, axis=1).ravel() / np.sqrt(12)
    _table = pd.DataFrame({"Parameter": ["theta", "jump_intensity"],
                           "Prior mean": _bounds.mean(axis=1), "Posterior mean": _mean,
                           "Prior SD": _prior_std, "Posterior SD": _std,
                           "SD contraction (1 − posterior/prior)": 1 - _std / _prior_std})
    mo.vstack([
        _fig,
        mo.ui.table(_table, selection=None, show_column_summaries=False),
        mo.md("""
        Each panel has its own color scale, because a concentrated posterior
        can be much taller than its prior. The posterior's slope shows parameter
        dependence: similar option panels may trade variance against jumps.
        The truth marker is displayed only to interpret synthetic recovery.
        It is absent from the likelihood and unavailable for market data.
        Positive contraction means narrower uncertainty for this observation;
        contraction alone does not prove calibration or correct joint dependence.
        """),
    ])
    return


@app.cell
def _(mo, neural_data, neural_result, np, plt):
    _explanation = mo.md(r"""
    ### Is the estimator updating the prior?

    Compare means **and** spreads conditional on the same observed panel.
    A network that returns the prior ignores the data even if its marginal SBC
    ranks look uniform. A useful candidate should follow the numerical posterior
    as the observations change, including its dependence and uncertainty.
    Matching one mean alone is insufficient. The comparison below is a cached
    audit of the original panels; it does not follow the quote-shift control.

    A posterior need not be narrowly concentrated in every direction. Weakly
    identified jump intensity can remain broad while the quoted-price fit is
    informative. This is a substantive inference result, not automatically a
    neural failure; compare against the reference for that same observation.
    """)
    mo.stop(neural_data is None or neural_result is None,
            mo.vstack([_explanation, mo.md("Conditional mean/spread graph awaits the recorded neural audit.")]))
    _means = np.asarray([_case["neural_mean"] for _case in neural_result["cases"]])
    _stds = np.asarray([_case["neural_std"] for _case in neural_result["cases"]])
    _prior = neural_result["prior_baseline"]
    _draw_count = neural_result["conditional_draws_per_case"]
    _fig, _axes = plt.subplots(2, 2, figsize=(10, 6.7), layout="constrained")
    for _index, _label in enumerate(("Long-run variance θ", "Jump intensity λ / year")):
        for _row, _values, _ref_key, _prior_key in (
            (0, _means, "reference_mean", "mean"), (1, _stds, "reference_std", "std"),
        ):
            _ax = _axes[_row, _index]
            _reference = neural_data[_ref_key][:, _index]
            _candidate = _values[:, _index]
            _limits = [min(_reference.min(), _candidate.min(), _prior[_prior_key][_index]),
                       max(_reference.max(), _candidate.max(), _prior[_prior_key][_index])]
            _padding = max((_limits[1] - _limits[0]) * 0.07, 1e-6)
            _limits = [_limits[0] - _padding, _limits[1] + _padding]
            _ax.plot(_limits, _limits, color="#64748b", linestyle="--", label="Reference agreement")
            _ax.axhline(_prior[_prior_key][_index], color="#b75d38", linestyle=":", label="Prior-only output")
            if _row == 0:
                _ax.errorbar(_reference, _candidate, yerr=_stds[:, _index] / np.sqrt(_draw_count),
                             fmt="o", markersize=4, color="#276678", label="Candidate ± 1 mean MCSE")
            else:
                _ax.scatter(_reference, _candidate, s=23, color="#276678", label="Candidate")
            _quantity = "mean" if _row == 0 else "SD"
            _ax.set(xlabel=f"Reference posterior {_quantity}", ylabel=f"Candidate posterior {_quantity}",
                    title=f"{_label}: {_quantity}", xlim=_limits, ylim=_limits)
            _ax.legend(fontsize=7)
    mo.vstack([
        _explanation, _fig,
        mo.md(f"""
        Mean discrepancy across the saved conditional cases, in reference-SD
        units: **candidate RMSE {neural_result['metrics']['mean_rmse_in_ref_sd']:.3f}**;
        **prior-only RMSE {_prior['mean_rmse_in_ref_sd']:.3f}**.
        Each dot is one observed panel. Mean MCSE bars describe finite neural
        draws, not uncertainty about neural bias. Stress cases are included in
        this conditional audit; they are not additional prior-predictive SBC cases.
        """),
    ])
    return


@app.cell
def _(mo, neural_data, neural_result, np, plt):
    _explanation = mo.md(r"""
    ### This graph is simulation-based calibration (SBC)

    Independently draw a parameter from the prior, simulate its noisy panel,
    and sample the network's posterior for that panel. For each parameter,
    count how many of $M$ posterior draws fall below the known truth:
    $r=\sum_{m=1}^{M}\mathbf1\{z^{(m)}<z_{\rm true}\}$.
    Randomize placement among ties if any, then the discrete rank as $u=(r+U)/(M+1)$ with
    $U\sim\operatorname{Uniform}(0,1)$. Correct continuous posterior draws
    produce uniform ranks across repeated prior/simulator experiments.

    The displayed empirical CDF should follow the diagonal. Its Dvoretzky–
    Kiefer–Wolfowitz (DKW) band controls the maximum deviation over the whole
    CDF; the protocol also accounts for checking both parameter marginals.
    This is broader than checking coverage at one interval level.

    **Passing these marginal ranks is useful evidence, not a posterior
    certificate.** A prior-only estimator can pass SBC, and wrong conditional
    dependence can escape marginal checks. That is why the same milestone
    also checks conditional means/spreads, correlation and target-price
    distributions against the numerical posterior.
    """)
    mo.stop(neural_data is None or neural_result is None,
            mo.vstack([_explanation, mo.md("Rank ECDF awaits the recorded neural audit.")]))
    _sbc = neural_result["sbc"]
    _ranks = np.sort(neural_data["randomized_ranks"], axis=0)
    _count = len(_ranks)
    _bound = _sbc["rank_dkw_bound"]
    _line = np.linspace(0, 1, 201)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for _index, _ax in enumerate(_axes):
        _ax.fill_between(_line, np.maximum(0, _line - _bound), np.minimum(1, _line + _bound),
                         color="#dce5ee", label="Jointly adjusted DKW band")
        _ax.plot(_line, _line, "--", color="#64748b", linewidth=1)
        _ax.step(np.r_[0, _ranks[:, _index], 1], np.r_[0, np.arange(1, _count + 1) / _count, 1],
                 where="post", color="#276678", label="Randomized-rank ECDF")
        _ax.set(xlabel="Randomized posterior rank", ylabel="Empirical CDF", xlim=(0, 1), ylim=(0, 1),
                title=f"{('θ', 'λ')[_index]}: max deviation {_sbc['rank_ks'][_index]:.3f}")
        _ax.legend(fontsize=7)
    mo.vstack([
        _explanation, _fig,
        mo.md(f"""
        **{_count} independent audit panels**, {_sbc['draws_per_case']} posterior
        draws each; saved family-level alpha {_sbc['sbc_family_alpha']:g}.
        DKW bound **{_bound:.4f}**; recorded marginal-rank result
        **{'PASS' if _sbc['rank_diagnostic_passed'] else 'FAIL'}**.
        Nominal 90% marginal coverage: theta **{_sbc['coverage_90'][0]:.3f}**,
        jump intensity **{_sbc['coverage_90'][1]:.3f}**.
        These are prior-predictive aggregates, not conditional guarantees for
        every parameter value or observed surface.
        """),
    ])
    return


@app.cell
def _(mo, neural_result, np, plt):
    _explanation = mo.md(r"""
    ### Conditional reference checks: these are not SBC

    Fix one observed panel and compare the neural posterior to the numerical
    reference. The mean error is measured in that reference posterior's SD:
    $(\widehat m-m_{\rm ref})/s_{\rm ref}$. Width and correlation discrepancies
    probe other parts of the joint distribution. A few difficult panels can
    fail these checks even when aggregate SBC looks acceptable.

    Gates were fixed in the protocol before the run. They are operational
    accuracy choices for this tutorial experiment, not statistical p-values or
    universal standards. A ratio above one in the gate plot means the measured
    discrepancy exceeds its declared allowance. Repeated draws from the
    reference quantify finite-sample variation; they do not relax the gates
    after seeing the candidate. Failed gates leave the candidate provisional
    and the reserved test unopened.
    """)
    mo.stop(neural_result is None,
            mo.vstack([_explanation, mo.md("Conditional gate graph awaits the recorded neural audit.")]))
    _names = list(neural_result["thresholds"])
    _labels = {
        "mean_rmse_in_ref_sd": "Mean RMSE / reference SD",
        "max_abs_mean_error_in_ref_sd": "Largest mean error / reference SD",
        "max_abs_relative_std_error": "Largest relative SD error",
        "max_abs_correlation_error": "Largest correlation error",
        "max_abs_target_quantile_error_currency": "Largest target-price quantile error",
        "mean_reference_mass_of_90_interval_abs_error": "Mean 90% interval-mass error",
    }
    _ratios = np.asarray([neural_result["metrics"][_key] / neural_result["thresholds"][_key]
                          for _key in _names])
    _null = np.asarray([neural_result["reference_draw_variability"]["metric_025_50_975_percentiles"][_key]
                       for _key in _names])
    _null /= np.asarray([neural_result["thresholds"][_key] for _key in _names])[:, None]
    _locations = np.arange(len(_names))
    _fig, _ax = plt.subplots(figsize=(10, 4.3), layout="constrained")
    _ax.barh(_locations, _ratios, color=["#b75d38" if _ratio > 1 else "#276678" for _ratio in _ratios],
             alpha=0.85, label="Recorded candidate discrepancy")
    _ax.errorbar(_null[:, 1], _locations, xerr=np.stack([_null[:, 1] - _null[:, 0], _null[:, 2] - _null[:, 1]]),
                 fmt="o", markersize=4, color="#17202a", capsize=3,
                 label="Reference-draw variability: median and 2.5–97.5%")
    _ax.axvline(1, color="#17202a", linestyle="--", label="Fixed acceptance threshold")
    _ax.set(yticks=_locations, yticklabels=[_labels[_key] for _key in _names],
            xlabel="Discrepancy / predeclared allowance", title="Conditional audit gates (not SBC)")
    _ax.invert_yaxis()
    _ax.legend(fontsize=7, loc="lower right")
    mo.vstack([
        _explanation, _fig,
        mo.md(f"""
        Reference-draw variability uses
        **{neural_result['reference_draw_variability']['replicates']} repetitions**
        at the same draw count. Its dots and intervals are a finite-sampling
        comparison, not uncertainty intervals on the candidate's error.
        The status table also includes the separate marginal SBC gate.
        Overall recorded neural validation:
        **{'PASS within this protocol' if neural_result['passed'] else 'FAIL; candidate remains provisional'}**.
        """),
    ])
    return


@app.cell
def _(inference_protocol, mo, neural_data, neural_result, np, pd, plt, reference_case):
    _explanation = mo.md(r"""
    ### From a joint parameter posterior to another option's price

    For each matched draw $(\theta^{(m)},\lambda^{(m)})$, evaluate a target
    contract with the same known pricing context. The resulting distribution
    describes uncertainty in a **model expectation** induced by parameter
    uncertainty. It is not the distribution of a single realized option payoff
    and not a forecast of tomorrow's market quote. Target-contract quotes did
    not enter the nine-quote conditioning panel.

    Keep matched parameters together: independently resampling their marginals
    destroys dependence and can change the price distribution. Here Fourier
    pricing supplies the downstream map; a learned pricing surrogate remains
    a separate future milestone with its own approximation checks.

    The 5%–95% interval is posterior uncertainty. Finite-draw Monte Carlo
    uncertainty in its estimated endpoints is a separate, typically much
    smaller quantity. Order-statistic intervals quantify this sampling
    uncertainty for each candidate quantile; repeated reference draws
    contextualize the aggregate discrepancy. Neither measures Fourier bias
    or real-market model inadequacy.

    Review questions: which graph would expose a prior-only network? Which
    checks could detect a correct pair of marginals with incorrect dependence?
    What additional evidence would be required before using this experiment
    to make an empirical claim about actual option prices?
    """)
    mo.stop(neural_data is None or neural_result is None or inference_protocol is None or reference_case is None,
            mo.vstack([_explanation, mo.md("Target-price intervals and Monte Carlo uncertainty await the recorded audit.")]))
    _case = reference_case.value
    _candidate = neural_result["cases"][_case]
    _prior = np.asarray(neural_result["prior_baseline"]["target_quantiles"])
    _reference = neural_data["reference_target_quantiles"][_case]
    _quantiles = np.asarray(_candidate["target_quantiles"])
    _mc_intervals = np.asarray(_candidate["target_quantile_mc95_interval"], dtype=float)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    _rows = []
    for _index, (_ax, _contract) in enumerate(zip(_axes, inference_protocol["targets"], strict=True)):
        for _row, _values, _color, _name in (
            (0, _prior[_index], "#b75d38", "Prior"),
            (1, _reference[_index], "#64748b", "Numerical posterior"),
            (2, _quantiles[_index], "#276678", "Neural candidate"),
        ):
            _ax.errorbar(_values[1], _row, xerr=[[_values[1] - _values[0]], [_values[2] - _values[1]]],
                         fmt="o", color=_color, capsize=4, label=_name)
        for _q_index, _probability in enumerate(inference_protocol["evaluation"]["quantiles"]):
            _interval = _mc_intervals[_index, _q_index]
            if np.isfinite(_interval).all():
                _ax.plot(_interval, [2.18, 2.18], color="#17202a", linewidth=3)
            _rows.append({"Target": f"K={_contract['strike']:g}, T={_contract['maturity']:g}",
                          "Quantile": _probability, "Reference price": _reference[_index, _q_index],
                          "Candidate price": _quantiles[_index, _q_index],
                          "MC 95% lower": _interval[0], "MC 95% upper": _interval[1]})
        _ax.set(yticks=[0, 1, 2], yticklabels=["Prior", "Reference", "Candidate"],
                xlabel="Target model price, currency units", ylim=(-0.45, 2.5),
                title=f"{_contract['option_type'].title()}, K={_contract['strike']:g}, T={_contract['maturity']:g} y")
    mo.vstack([
        _explanation,
        mo.md(f"Saved case **{_candidate['case_id']}**. The quote-shift control does not alter this recorded comparison."),
        _fig,
        mo.ui.table(pd.DataFrame(_rows), selection=None, show_column_summaries=False),
        mo.md("""
        Colored intervals span posterior 5%–95% quantiles, with a median dot.
        Short black marks above the candidate show **individual 95%
        order-statistic intervals** for each estimated quantile under IID
        sampling. They are Monte Carlo uncertainty intervals, not posterior
        intervals or simultaneous guarantees. The full table retains their
        numeric values when the marks are too small to see.
        """),
    ])
    return


if __name__ == "__main__":
    app.run()
