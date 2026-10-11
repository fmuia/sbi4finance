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

    from sbi4finance import bergomi, option_bank

    ROOT = mo.notebook_dir().parent
    return ROOT, bergomi, json, mo, np, option_bank, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    # 8. One-factor Bergomi: a curve, a Gaussian factor and option prices

    Heston models variance through a square-root diffusion. This branch starts
    from a Gaussian factor and exponentiates it to construct positive variance
    with a specified mean curve. It supplies another reusable **Q simulator and
    option-bank route**. A working kernel or bank is not an accepted posterior.

    All curves, contracts and paths here are synthetic. Default execution uses
    three short illustrative paths, small Monte Carlo panels, and one bank
    example for each model; it runs
    offline without market fixtures, training or cluster jobs. Larger numerical
    checks are displayed from a saved report. This is **S evidence**, not a
    claim that the model explains a real market. Returns-versus-options remains
    an open choice for the eventual participant tutorial.

    ## What is the forward-variance curve?

    The positive function $\xi_0(t)$ is the initial expected instantaneous
    variance at future time $t$. A value $0.04$ corresponds to 20% annualized
    volatility when square-rooted. It is different from an asset forward
    $F_T=S_0e^{(r-q)T}$, which has price units.

    For a one-factor convention, start $Z_0=0$ and set

    $$dZ_t=-k Z_tdt+dW_t^V,\qquad
      A_k(t)=\operatorname{Var}(Z_t)=\frac{1-e^{-2kt}}{2k},$$
    $$V_t=\xi_0(t)\exp\!\left(\eta Z_t-\tfrac12\eta^2 A_k(t)\right).$$

    The mathematical $k=0$ limit has $A_0(t)=t$; this implementation requires
    $k>0$. The subtraction normalizes the exponential:
    for a centered Gaussian $Z$, $\mathbb E[e^{\eta Z}]=
    e^{\eta^2\operatorname{Var}(Z)/2}$, so $\mathbb E[V_t]=\xi_0(t)$.
    Time is in years; $k$ is an inverse-year rate and this convention gives
    $\eta$ units year$^{-1/2}$. Changing the driver normalization changes the
    meaning of $\eta$.

    Under Q, $d\log S_t=(r-q-V_t/2)dt+\sqrt{V_t}\,dW_t^S$, with
    $d\langle W^S,W^V\rangle_t=\rho\,dt$. Negative $\rho$ creates leverage.
    This notebook declares the whole curve **known synthetic context**. If a
    later task infers curve coefficients, those unknown truths must move to
    the target side rather than be supplied secretly as observed inputs.
    """)
    return


@app.cell
def _(mo):
    eta_control = mo.ui.slider(0.0, 2.0, step=0.1, value=1.0,
                               label="Factor amplitude η", show_value=True)
    seed_control = mo.ui.number(start=0, stop=999999, value=8602, step=1, label="Path seed")
    mo.vstack([
        mo.md("## Simulate three paths\n\nThe controls change an illustration, not the domain accepted by the recorded numerical study."),
        mo.hstack([eta_control, seed_control]),
    ])
    return eta_control, seed_control


@app.cell
def _(bergomi, eta_control, np, seed_control):
    factor_parameters = dict(kappa=2.0, eta=eta_control.value, rho=-0.6)
    variance_curve = {"times": [0.0, 0.5, 1.0], "values": [0.04, 0.045, 0.05]}
    dt = np.full(64, 1 / 64)
    innovations = np.random.default_rng(int(seed_control.value)).normal(size=(3, 64, 3))
    factor_paths = bergomi.simulate_one_factor(factor_parameters, variance_curve, dt,
                                               innovations, spot=100, rate=0.02)
    return dt, factor_parameters, factor_paths, innovations, variance_curve


@app.cell
def _(factor_paths, np, plt):
    _fig, _axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, layout="constrained")
    _time = np.linspace(0, 1, 65)
    for _i in range(3):
        _axes[0].plot(_time, factor_paths["factor_state"][_i], linewidth=1)
        _axes[1].plot(_time, factor_paths["variance"][_i], linewidth=1)
    _axes[0].set(ylabel="Gaussian factor Z", title="Latent factor: can be positive or negative")
    _axes[1].set(xlabel="Elapsed time (years)", ylabel="Annualized variance",
                 title="Positive latent variance; these paths are simulator truth")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exact factor transitions do not give exact prices

    Over an interval $h$, the exact factor transition is
    $Z_{t+h}=e^{-kh}Z_t+U$, with
    $\operatorname{Var}(U)=A_k(h)$. Its covariance with the variance-driver
    Brownian increment is

    $$\operatorname{Cov}(U,\Delta W^V)=\frac{1-e^{-kh}}{k}.$$

    Jointly generating this transition and the price Brownian increment
    requires that integrated covariance. Reusing one normal with an arbitrary
    instantaneous correlation would generally give the wrong joint law.
    The source uses three independent normals to construct the required
    Gaussian quantities. Asset integration uses pre-step variance and remains
    time-discretized, even when the OU step itself is exact.

    A continuation must carry $Z_t$, current spot, elapsed time, model
    parameters and the original curve. At a later maturity $u$, the past
    factor contribution is $e^{-k(u-t)}Z_t$. Resetting the factor or the curve
    time origin would discard information. The normalization $A_k(u)$ also
    refers to the original experiment time, not a freshly restarted clock.
    """)
    return


@app.cell
def _(bergomi, dt, factor_parameters, factor_paths, innovations, mo, np, pd, variance_curve):
    _cut = 32
    _first = bergomi.simulate_one_factor(factor_parameters, variance_curve, dt[:_cut],
                                         innovations[:, :_cut], spot=100, rate=0.02)
    _second = bergomi.simulate_one_factor(
        factor_parameters, variance_curve, dt[_cut:], innovations[:, _cut:],
        factor_state=_first["factor_state"][:, -1, 0], elapsed_time=0.5,
        spot=1.0, rate=0.02,
    )
    _second["log_spot"] += _first["log_spot"][:, -1, None]
    _rows = []
    for _name in ["factor_state", "variance", "log_spot"]:
        _joined = np.concatenate([_first[_name], _second[_name][:, 1:]], axis=1)
        _difference = float(np.max(np.abs(_joined - factor_paths[_name])))
        np.testing.assert_allclose(_joined, factor_paths[_name], rtol=1e-12, atol=1e-12)
        _rows.append({"State/output": _name, "Maximum split-path difference": _difference})
    mo.vstack([
        mo.md("""
        **Matched continuation check.** The two runs use slices of the same
        already-drawn innovations. The kernel takes a common scalar starting
        spot, so the second segment starts at one and its log prices are
        shifted by each path's saved log spot. Carrying the full state reproduces the
        uninterrupted discrete calculation. This checks bookkeeping and the
        numerical transition, not an inferred state or a P risk forecast.
        """),
        mo.ui.table(pd.DataFrame(_rows), selection=None, show_column_summaries=False),
    ])
    return


@app.cell
def _(bergomi, factor_parameters, variance_curve):
    contracts = [dict(strike=k, maturity=t, option_type="put")
                 for t in [0.5, 1.0] for k in [90.0, 100.0, 110.0]]
    panel_result = bergomi.option_panel(
        "one_factor", factor_parameters, variance_curve, contracts,
        spot=100, rate=0.02, dividend=0.0, n_paths=4096, n_steps=64, seed=8621,
    )
    return contracts, panel_result


@app.cell
def _(contracts, mo, np, panel_result, pd):
    _prices = np.asarray(panel_result["prices"])
    _se = np.asarray(panel_result["standard_errors"])
    _rows = []
    for _i, _contract in enumerate(contracts):
        _time = _contract["maturity"]
        _parity_addition = 100 - _contract["strike"] * np.exp(-0.02 * _time)
        _rows.append({"Maturity (years)": _time, "Strike": _contract["strike"],
                      "Estimated put": _prices[_i], "Sampling SE": _se[_i],
                      "Call through parity": _prices[_i] + _parity_addition})
    mo.vstack([
        mo.md(r"""
        ## Prices are estimates of expectations

        The panel uses 4,096 independent variance-driver paths on 64 time steps,
        shared across contracts. It analytically averages out the independent
        asset Brownian motion conditional on each driver path. This reduces
        sampling noise. For a put, $0\le D(K-S_T)^+\le DK$, so both the payoff
        and its conditional expectation have finite variance. The estimated standard error is
        $s/\sqrt{M}$. Within the supported martingale domain, obtain calls via
        $C=P+D(F-K)$; the added term is known and has no sampling uncertainty.
        The implementation restricts $-1\le\rho\le0$ for its pricing domain.

        A finite expected call price alone does not guarantee finite
        call-payoff variance. Bounded-put estimation avoids relying on that
        assumption. The reported SE concerns Monte Carlo sampling **at this
        numerical grid**. It does not include integration bias, intentional
        quote noise, model uncertainty or market error. These low-cost
        illustrative prices are not a high-precision training bank.
        """),
        mo.ui.table(pd.DataFrame(_rows), selection=None, show_column_summaries=False),
    ])
    return


@app.cell
def _(mo, np, panel_result, plt):
    _cov = np.asarray(panel_result["estimator_covariance"])
    _scale = np.sqrt(np.diag(_cov))
    _correlation = np.divide(_cov, _scale[:, None] * _scale[None, :],
                              out=np.zeros_like(_cov), where=(_scale[:, None] * _scale[None, :]) > 0)
    _fig, _ax = plt.subplots(figsize=(5.6, 4.4), layout="constrained")
    _image = _ax.imshow(_correlation, vmin=-1, vmax=1, cmap="coolwarm")
    _ax.set(xlabel="Contract index", ylabel="Contract index",
             title="Sampling-error correlation from shared pricing paths")
    _fig.colorbar(_image, ax=_ax, label="Correlation")
    mo.vstack([
        _fig,
        mo.md("""
        The returned covariance is for the **estimated mean prices**, not the
        covariance of single payoffs. Shared paths correlate numerical errors
        across contracts. Independent Gaussian quote noise, if declared by a
        bank task, is a different random mechanism. Per-contract SEs are not
        a simultaneous confidence band for an entire surface.
        """),
    ])
    return


@app.cell
def _(ROOT, json, mo, np, pd):
    bergomi_report = json.loads((ROOT / "docs/results/bergomi-validation.json").read_text())
    _rows = []
    for _case in bergomi_report["cases"]:
        for _index, _comparison in enumerate(_case["refinements"]):
            _rows.append({
                "Parameters": str(_case["parameters"]),
                "Grid comparison": f"{_case['grids'][_index]['metadata']['n_steps']} → {_case['grids'][_index+1]['metadata']['n_steps']}",
                "Largest absolute price difference / spot": float(np.max(np.abs(_comparison["price_difference_second_minus_first"]))),
                "Largest sampling SE of difference / spot": max(_comparison["mc_standard_error_of_difference"]),
                "Recorded check": "PASS" if _comparison["passed"] else "FAIL",
            })
    mo.vstack([
        mo.md("""
        ## Recorded numerical evidence and remaining gates

        The saved report contains the tested domains, grids, covariance and
        normalization checks, deterministic-volatility limits and refinement
        comparisons. It is displayed as recorded, without running a large
        study on notebook opening. Read each check's scope and tolerances;
        accepted numerical fixtures do not by themselves accept every proposed
        prior, a posterior network or a market model.
        """),
        mo.ui.table(pd.DataFrame(_rows), selection=None, show_column_summaries=False),
        mo.md(f"""
        The recorded comparison allowance is
        **{bergomi_report['predeclared_budgets']['absolute_price_in_spot_units']:g}
        of spot + {bergomi_report['predeclared_budgets']['mc_se_multiplier']}
        sampling SEs of the difference**. The absolute component is 0.20 price
        units at spot 100, comparable to the pilot quote-noise SD of 0.25.
        A passed refinement screen is not a bound proving remaining grid bias
        negligible for that likelihood. It is a reason to retain these kernels
        and investigate the numerical budget before larger inference banks.
        Overall saved result: **{'PASS' if bergomi_report['passed'] else 'FAIL'}**
        for the recorded numerical fixtures. Posterior acceptance is deferred.
        """),
        mo.accordion({"Full moment, pricing and limitation report": mo.json(bergomi_report)}),
    ])
    return (bergomi_report,)


@app.cell
def _(ROOT, json, mo, option_bank, pd):
    bergomi_bank_config = json.loads((ROOT / "configs/tasks/bergomi_option_bank.json").read_text())
    _example = option_bank.generate_examples(bergomi_bank_config, "train", [0])
    _table = pd.DataFrame({
        "Contract index": range(_example["observations"].shape[1]),
        "Observed noisy quote": _example["observations"][0, :, 0],
        "Reference estimate (not observed)": _example["reference_prices"][0],
        "Sampling SE (not observed)": _example["pricing_standard_errors"][0],
    })
    mo.vstack([
        mo.md("""
        ## One seeded example from the shared bank route

        This is one generated example, not a saved full bank or a training run.
        The task varies eta and rho on a restricted prior and fixes the known
        synthetic curve and other model inputs. Nine puts at strikes 90/100/110
        and maturities 0.25/0.5/1 years have Gaussian quote-noise SD 0.25 price
        units. The complete configuration records the exact bounds and budget.
        Monte Carlo SE describes sampling, not uncalculated grid bias.
        A stored zero in the common refinement-difference field is explicitly
        marked **not calculated** for Monte Carlo; it is not a zero-bias claim.
        """),
        mo.ui.table(_table, selection=None, show_column_summaries=False),
        mo.json(bergomi_bank_config),
    ])
    return (bergomi_bank_config,)


@app.cell
def _(ROOT, bergomi_bank_config, json, mo, np, pd):
    import hashlib as _hashlib

    bergomi_bank_numerics = json.loads((ROOT / "docs/results/bergomi-bank-numerics.json").read_text())
    if bergomi_bank_numerics["config_sha256"] != _hashlib.sha256((ROOT / "configs/tasks/bergomi_option_bank.json").read_bytes()).hexdigest():
        raise ValueError("Bank-specific report does not match the displayed configuration")
    if bergomi_bank_numerics["task"] != bergomi_bank_config["task"]:
        raise ValueError("Bank-specific report describes a different task")
    _rows = []
    for _case in bergomi_bank_numerics["cases"]:
        _comparison = _case["refinement"]
        _differences = np.abs(_comparison["price_difference_second_minus_first"])
        _index = int(np.argmax(_differences))
        _rows.append({
            "Case": _case["role"], "eta": _case["parameters"]["eta"],
            "rho": _case["parameters"]["rho"], "Contract with largest difference": _index,
            "Largest absolute difference (currency)": float(_differences[_index]),
            "SE of that difference (currency)": _comparison["mc_standard_error_of_difference"][_index],
            "Allowed difference at that contract": _comparison["allowance"][_index],
            "Recorded screen": "PASS" if _comparison["passed"] else "FAIL",
        })
    _worst = max(_rows, key=lambda row: row["Largest absolute difference (currency)"])
    mo.vstack([
        mo.md("""
        ## The bank's own corners and center

        The earlier fixture study had a different curve and contract domain.
        This additional report matches the exact saved bank configuration,
        including its one-year contracts, constant known curve and eta/rho
        prior. Four corners and the center compare **128 with 512 steps**,
        each using **262,144 paths** and independent seeds. This is cached
        evidence, not another computation launched by this notebook.
        """),
        mo.ui.table(pd.DataFrame(_rows), selection=None, show_column_summaries=False),
        mo.md(f"""
        Largest observed difference:
        **{_worst['Largest absolute difference (currency)']:.5f} currency**,
        with combined sampling SE
        **{_worst['SE of that difference (currency)']:.5f}** at that contract.
        The screen permits **0.025 currency + 4 SEs**. Its saved result is
        **{'PASS' if bergomi_bank_numerics['passed'] else 'FAIL'}**.
        A pass does **not** prove bias below 0.025: sampling uncertainty can
        hide smaller differences, the 512-step reference retains grid bias,
        and five cases do not cover every draw. The bank's separate SE limit
        of 0.1 times quote-noise SD controls sampling precision only.
        """),
        mo.accordion({"Exact bank configuration, all contracts and limits": mo.json(bergomi_bank_numerics)}),
        mo.md("[Bank-specific report](../docs/results/bergomi-bank-numerics.json) · [Reproduction script](../scripts/check_bergomi_bank_numerics.py)"),
    ])
    return (bergomi_bank_numerics,)


@app.cell
def _(mo):
    mo.md(r"""
    ## A shared bank, with model-specific meaning

    One observation is a complete fixed-contract **Q quote panel**. Known
    context includes contract coordinates, rates/spot and the declared
    synthetic forward-variance curve. Selected model parameters become latent
    labels; Brownian paths, factor states and parameter-dependent Monte Carlo
    diagnostics are not observed inputs. A price-label bank can instead use
    parameters/contracts as inputs, but that is a different supervised task.

    Numerical path count, time grid and method are reproducibility metadata.
    This bank has `observations[B, 9, 1]`, `parameters[B, 2]` ordered eta/rho,
    `reference_prices[B, 9]` and estimator covariance `[B, 9, 9]`. A kernel's
    factor/variance trajectory is a different object from the panel that the
    later estimator would observe.
    Draw independent base examples into splits before quote augmentation.
    Keep numerical uncertainty distinct from the quote-error model; otherwise
    a network may learn an undocumented noisy-pricer experiment. A pilot bank
    establishes the storage/generation route, not accepted conditional inference.

    **S:** generated curves and panels support all synthetic work offline.
    **M:** an actual compatible panel can later condition a trained network,
    with no fabricated parameter labels. **E:** excluded actual quotes permit
    a separately designed empirical assessment. None is a substitute for the
    others. A Q panel calibration does not yield P return probabilities or
    tomorrow's option-chain dynamics.

    Two-factor Bergomi remains a separate optional scientific extension.
    The next section replaces the exponential kernel with rough Bergomi's
    power-law memory and states the actual Gaussian field, approximation and
    memory requirements. Current variance alone cannot initialize a rough
    conditional forecast.

    ## Review exercises

    1. Set $\eta=0$. What should variance equal? Why can the asset integration
       still have grid error when the known curve varies with time?
    2. Why does exact simulation of $Z_{t+h}\mid Z_t$ not imply an exact
       terminal asset distribution? Identify the integral still approximated.
    3. Which continuation inputs disappear if you restart with $Z=0$ and
       elapsed time zero? Explain why that is a new experiment, not a forecast
       conditioned on the existing state.

    Sources and reproduction: [task configuration](../configs/tasks/bergomi_option_bank.json),
    [one-factor source](../src/sbi4finance/bergomi.py),
    [shared bank source](../src/sbi4finance/option_bank.py),
    [numerical evidence](../docs/results/bergomi-validation.json),
    [M6 subplan](../docs/simulator-plan/milestones/06-bergomi-factors.md),
    [M7 subplan](../docs/simulator-plan/milestones/07-rough-bergomi.md),
    [model and numerical notes](../docs/simulator-plan/notes/02-models-and-numerics.md),
    [next analysis](../docs/simulator-analysis-next-steps.md), and
    [Bergomi's original model](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1493302).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Rough Bergomi: replace a factor with a memory kernel

    One exponential factor compresses all past volatility-driver shocks into
    one state. Rough Bergomi uses a power kernel instead:

    $$G_t=\sqrt{2H}\int_0^t(t-s)^{H-1/2}\,dW_s^V,
      \qquad 0<H<1/2,$$
    $$\operatorname{Var}(G_t)=t^{2H},\qquad
      V_t=\xi_0(t)\exp\!\left(\eta G_t-\tfrac12\eta^2t^{2H}\right).$$

    This is the **Riemann–Liouville Volterra field starting at zero**, not the
    stationary-increment version of fractional Brownian motion. The
    $\sqrt{2H}$ normalization is part of the model; another convention changes
    the numerical meaning of $\eta$. Smaller $H$ gives rougher local paths.
    With time in years, this $\eta$ has units year$^{-H}$, unlike the OU
    factor's year$^{-1/2}$. The $H\to1/2$ limit has a constant kernel, matching
    zero-mean-reversion one-factor dynamics, not arbitrary OU mean reversion.

    The initial curve remains known synthetic Q context. The new bank targets
    $H$ and $\eta$ on its declared restricted prior. Its observations are noisy
    option panels, not rough variance paths. This increment supplies a driver,
    price generator and pilot-bank route; **no rough posterior is trained or
    accepted here**.

    ### Two different numerical constructions

    The small-grid Gaussian reference calculates
    $C_{ij}=\int_0^{\min(t_i,t_j)}K_H(t_i-s)K_H(t_j-s)ds$ together with
    $\operatorname{Cov}(G_{t_i},\Delta W_j^V)$. It samples that joint Gaussian
    vector. This is a grid reference with a dense matrix cost, not a scalable
    path generator or an exact asset-price integral.

    The one-cell hybrid construction treats the most recent singular kernel
    cell exactly, including its independent Gaussian residual, and approximates
    older cells by interval-average weights. FFT convolution applies those
    weights. Its covariance differs from the continuum field at finite grid
    size. The exponential correction uses the **actual approximate driver
    variance**, so the intended mean variance curve stays normalized. Merely
    substituting $t^{2H}$ into a different driver's variance would change that
    expectation.
    """)
    return


@app.cell
def _(np):
    from sbi4finance import rough_bergomi

    rough_parameters = dict(hurst=0.12, eta=1.0, rho=-0.6)
    rough_dt = 1 / 16
    rough_covariance, rough_cross_covariance = rough_bergomi.rough_covariance(0.12, rough_dt, 16)
    hybrid_covariance, hybrid_cross_covariance = rough_bergomi.hybrid_covariance(0.12, rough_dt, 16)
    rough_times = np.arange(1, 17) * rough_dt
    return hybrid_covariance, hybrid_cross_covariance, rough_bergomi, rough_covariance, rough_cross_covariance, rough_parameters, rough_times


@app.cell
def _(hybrid_covariance, hybrid_cross_covariance, mo, np, plt, rough_covariance, rough_cross_covariance, rough_times):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    _axes[0].plot(rough_times, np.diag(rough_covariance), "o-", label="Continuum Gaussian reference")
    _axes[0].plot(rough_times, np.diag(hybrid_covariance), "x--", label="Hybrid driver on 16 steps")
    _axes[0].set(xlabel="Time (years)", ylabel="Driver variance (years^(2H))",
                 title="Normalize with the variance actually simulated")
    _axes[0].legend(fontsize=8)
    _axes[1].plot(rough_times, hybrid_covariance[:, -1] - rough_covariance[:, -1],
                  "o-", color="#bb653f")
    _axes[1].axhline(0, color="0.4", linewidth=0.8)
    _axes[1].set(xlabel="Time (years)", ylabel="Hybrid − reference (years^(2H))",
                 title="Covariance error against G at 1 year")
    _cross_error = float(np.max(np.abs(rough_cross_covariance - hybrid_cross_covariance)))
    mo.vstack([
        _fig,
        mo.md(f"""
        This is a deterministic covariance comparison at H=0.12, not a
        posterior check or an empirical estimate from a few paths. The maximum
        difference in driver–Brownian-increment cross-covariance is
        **{_cross_error:.3g}**; interval-average kernel weights preserve this
        cross-covariance while the full driver covariance still differs.
        Normalization corrects the marginal mean variance; it does not repair
        the missing covariance or erase option-price discretization bias.
        """),
    ])
    return


@app.cell
def _(bergomi, contracts, mo, np, pd, rough_parameters, variance_curve):
    _hybrid_panel = bergomi.option_panel(
        "rough", rough_parameters, variance_curve, contracts, spot=100, rate=0.02,
        n_paths=2048, n_steps=32, seed=9621, method="hybrid",
    )
    _reference_panel = bergomi.option_panel(
        "rough", rough_parameters, variance_curve, contracts, spot=100, rate=0.02,
        n_paths=2048, n_steps=32, seed=9631, method="covariance",
    )
    _table = pd.DataFrame({
        "Maturity (years)": [c["maturity"] for c in contracts],
        "Strike": [c["strike"] for c in contracts],
        "Hybrid put estimate": _hybrid_panel["prices"],
        "Hybrid sampling SE": _hybrid_panel["standard_errors"],
        "Gaussian-reference put estimate": _reference_panel["prices"],
        "Reference sampling SE": _reference_panel["standard_errors"],
    })
    mo.vstack([
        mo.md("""
        ### A bounded pricing comparison, not an acceptance study

        Both methods use 32 asset steps and 2,048 independent driver paths,
        with distinct seeds. Both average bounded conditional put values and
        have finite sampling variance. The reference removes the hybrid-driver
        approximation at its grid points, but **both retain asset-integration
        bias**. Individual standard errors do not cover that bias. This small
        live illustration is complemented by the recorded refinement study;
        numerical disagreement must be resolved before accepting a larger bank.
        """),
        mo.ui.table(_table, selection=None, show_column_summaries=False),
    ])
    return


@app.cell
def _(ROOT, json, mo, option_bank, pd):
    rough_bank_config = json.loads((ROOT / "configs/tasks/rough_bergomi_option_bank.json").read_text())
    _example = option_bank.generate_examples(rough_bank_config, "train", [0])
    _rows = pd.DataFrame({
        "Contract index": range(_example["observations"].shape[1]),
        "Observed quote": _example["observations"][0, :, 0],
        "Reference estimate (not observed)": _example["reference_prices"][0],
        "Sampling SE (not observed)": _example["pricing_standard_errors"][0],
    })
    mo.vstack([
        mo.md("""
        ### The same storage roles, a different model task

        One seeded example uses the rough task's declared H/eta prior,
        known curve, contract panel and quote-error law. No bank is written.
        The generator's latent labels and pricing diagnostics stay outside
        observed conditioning. A zero refinement-difference placeholder for
        Monte Carlo is explicitly marked **not calculated** in bank metadata;
        it cannot be read as zero bias. The raw task below fixes the finite
        numerical experiment and is not a posterior-acceptance result.
        """),
        mo.ui.table(_rows, selection=None, show_column_summaries=False),
        mo.json(rough_bank_config),
    ])
    return (rough_bank_config,)


@app.cell
def _(ROOT, json, mo, np, pd):
    rough_report = json.loads((ROOT / "docs/results/rough-bergomi-validation.json").read_text())
    _rows = []
    for _case in rough_report["cases"]:
        _comparison = _case["same_grid_reference_comparison"]
        _rows.append({
            "H": _case["parameters"]["hurst"],
            "eta": _case["parameters"]["eta"],
            "rho": _case["parameters"]["rho"],
            "Largest grid-refinement difference / spot": max(float(np.max(np.abs(c["price_difference_second_minus_first"]))) for c in _case["refinements"]),
            "Largest same-grid Gaussian comparison / spot": float(np.max(np.abs(_comparison["price_difference_second_minus_first"]))),
            "Recorded comparisons": "PASS" if _comparison["passed"] and all(c["passed"] for c in _case["refinements"]) else "FAIL",
        })
    mo.vstack([
        mo.md("""
        ### Saved rough-driver and pricing evidence

        These are the recorded numerical cases and budgets. Passing kernel
        checks, generating a pilot bank and opening this notebook do not
        accept a rough posterior. Price refinement, Gaussian references and
        their finite sampling uncertainty must be judged separately. The full
        report preserves unresolved checks and the restricted evidence scope.
        """),
        mo.ui.table(pd.DataFrame(_rows), selection=None, show_column_summaries=False),
        mo.md(f"""
        The declared allowance is
        **{rough_report['predeclared_budgets']['absolute_price_in_spot_units']:g}
        of spot + {rough_report['predeclared_budgets']['mc_se_multiplier']} SEs**
        of the independent-run price difference. It is a refinement screen,
        not a continuum-bias bound. Overall recorded result:
        **{'PASS' if rough_report['passed'] else 'FAIL'}** for these numerical
        fixtures. Per-bank Monte Carlo precision is separately limited relative
        to quote noise, but the remaining kernel/grid bias still needs its own
        analysis before larger inference banks are accepted.
        """),
        mo.accordion({"Complete Gaussian, pricing and limitation report": mo.json(rough_report)}),
        mo.md("[Raw report](../docs/results/rough-bergomi-validation.json) · [Next analysis](../docs/simulator-analysis-next-steps.md)"),
    ])
    return (rough_report,)


@app.cell
def _(ROOT, rough_bank_config, json, mo, np, pd):
    from hashlib import sha256 as _rough_sha256

    rough_bank_numerics = json.loads((ROOT / "docs/results/rough-bergomi-bank-numerics.json").read_text())
    if rough_bank_numerics["config_sha256"] != _rough_sha256((ROOT / "configs/tasks/rough_bergomi_option_bank.json").read_bytes()).hexdigest():
        raise ValueError("Bank-specific report does not match the displayed configuration")
    if rough_bank_numerics["task"] != rough_bank_config["task"]:
        raise ValueError("Bank-specific report describes a different task")
    _rows = []
    for _case in rough_bank_numerics["cases"]:
        _comparison = _case["refinement"]
        _differences = np.abs(_comparison["price_difference_second_minus_first"])
        _index = int(np.argmax(_differences))
        _rows.append({
            "Case": _case["role"], "H": _case["parameters"]["hurst"],
            "eta": _case["parameters"]["eta"], "Contract with largest difference": _index,
            "Largest absolute difference (currency)": float(_differences[_index]),
            "SE of that difference (currency)": _comparison["mc_standard_error_of_difference"][_index],
            "Allowed difference at that contract": _comparison["allowance"][_index],
            "Recorded screen": "PASS" if _comparison["passed"] else "FAIL",
        })
    _worst = max(_rows, key=lambda row: row["Largest absolute difference (currency)"])
    mo.vstack([
        mo.md("""
        ### The rough bank's own corners and center

        The earlier fixture study had a different curve and contract domain.
        This additional report matches the exact saved bank configuration,
        including its one-year contracts, constant known curve and H/eta
        prior. Four corners and the center compare **128 with 512 steps**,
        each using **262,144 paths** and independent seeds. This is cached
        evidence, not another computation launched by this notebook.
        """),
        mo.ui.table(pd.DataFrame(_rows), selection=None, show_column_summaries=False),
        mo.md(f"""
        Largest observed difference:
        **{_worst['Largest absolute difference (currency)']:.5f} currency**,
        with combined sampling SE
        **{_worst['SE of that difference (currency)']:.5f}** at that contract.
        The screen permits **0.025 currency + 4 SEs**. Its saved result is
        **{'PASS' if rough_bank_numerics['passed'] else 'FAIL'}**.
        A pass does **not** prove bias below 0.025: sampling uncertainty can
        hide smaller differences, the 512-step reference retains grid bias,
        and five cases do not cover every draw. The bank's separate SE limit
        of 0.1 times quote-noise SD controls sampling precision only.
        """),
        mo.accordion({"Exact bank configuration, all contracts and limits": mo.json(rough_bank_numerics)}),
        mo.md("[Bank-specific report](../docs/results/rough-bergomi-bank-numerics.json) · [Reproduction script](../scripts/check_bergomi_bank_numerics.py)"),
    ])
    return (rough_bank_numerics,)



@app.cell
def _(mo):
    mo.md(r"""
    ### Why there is no rough P forecast button

    At time $t$, the past contribution to a later driver's value is the whole
    function $m_t(u)=\int_0^tK_H(u-s)dW_s^V$ for $u>t$. Current variance or
    $G_t$ alone does not determine it. The kernel exposes full synthetic
    histories, but it does not infer the hidden memory from observed returns.
    A rough P forecast needs separately declared physical dynamics,
    initialization/prehistory and conditional memory inference. A Q calibration
    does not provide those objects or predict tomorrow's market option chain.

    A finite-factor approximation of the rough kernel would be a versioned
    numerical approximation with its own weights and error targets. It is not
    automatically the same scientific model as an independently parameterized
    two-factor Bergomi model.

    **Review exercises.** Why is correct marginal variance normalization not
    enough to price an option correctly? Why is a Gaussian covariance reference
    still a discretized price reference? What information would a rough
    conditional continuation need beyond today's variance?

    Sources: [rough kernel implementation](../src/sbi4finance/rough_bergomi.py),
    [restricted task](../configs/tasks/rough_bergomi_option_bank.json),
    [model and reproduction guide](../docs/rough-bergomi.md),
    [M7 subplan](../docs/simulator-plan/milestones/07-rough-bergomi.md),
    [mathematical conventions](../docs/simulator-plan/notes/02-models-and-numerics.md),
    [hybrid simulation paper](https://arxiv.org/abs/1507.03004), and
    [martingale and moment analysis](https://arxiv.org/abs/1811.10935).
    """)
    return



if __name__ == "__main__":
    app.run()
