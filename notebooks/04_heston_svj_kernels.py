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

    from sbi4finance import heston

    ROOT = mo.notebook_dir().parent
    return ROOT, heston, json, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    # 4. Heston variance and Gaussian price jumps

    **Build and check a simulator before asking it to support inference.**
    This increment adds changing variance, correlated price/variance shocks,
    and optional jumps to the constant-volatility GBM building block.
    We simulate three small synthetic histories, compare NumPy with Numba,
    and restart a path from its complete numerical state.

    Everything here runs offline. The numerical and timing tables read committed
    reports; opening the notebook does not launch a large benchmark or access
    market data. These are **S: synthetic kernel checks**, not posterior
    validation, calibrated forecasts, or empirical market evidence. The live
    tutorial's returns-versus-options choice remains open.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What does the model describe?

    With $X_t=\log(S_t/S_0)$, the continuous model underlying the simulator is

    $$dV_t=\kappa(\theta-V_t)dt+\xi\sqrt{V_t}\,dW_t^V,$$
    $$dX_t=(b-\lambda c-V_t/2)dt+\sqrt{V_t}\,dW_t^S+J\,dN_t,
    \qquad d\langle W^S,W^V\rangle_t=\rho\,dt,$$
    $$J\sim\mathcal N(m_J,s_J^2),\qquad c=e^{m_J+s_J^2/2}-1.$$

    Time is measured in years. $V=0.04$ means annualized volatility
    $\sqrt{V}=20\%$. The long-run variance is $\theta$; $\kappa$ controls
    reversion toward it; $\xi$ controls fluctuations in variance. Negative
    $\rho$ makes negative price shocks tend to accompany increases in variance
    (the leverage effect). Jumps arrive independently at rate $\lambda$ per
    year. Setting $\lambda=0$ gives Heston. These are **price jumps only**;
    this is not an SVCJ model with variance jumps.

    | Task | Meaning of the parameters and drift $b$ | Observation or output |
    |---|---|---|
    | Physical measure $P$ | Historical dynamics, $b=\mu$ | Return histories and future outcomes |
    | Risk-neutral measure $Q$ | Pricing dynamics, $b=r-q$ with separately chosen $Q$ parameters | Expected discounted payoffs; potentially option-panel calibration |

    The same kernel can implement either declared task. Changing only the drift
    does **not** infer the change of measure or turn fitted $P$ parameters into
    calibrated $Q$ parameters. This notebook uses known illustrative $P$
    parameters. A simulated payoff is one random outcome; an accurate price
    label requires an expectation and a validated pricing calculation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## From continuous dynamics to an explicit numerical experiment

    An observation interval $\Delta$ is divided into $L$ updates of length
    $h=\Delta/L$. Full-truncation Euler stores a signed auxiliary state $a$,
    and uses $v^+=\max(a,0)$ in **both** update equations:

    $$\Delta X=(b-v^+/2)h+\sqrt{v^+h}
       \left(\rho Z_V+\sqrt{1-\rho^2}Z_\perp\right),$$
    $$a'=a+\kappa(\theta-v^+)h+\xi\sqrt{v^+h}\,Z_V.$$

    $Z_V$ and $Z_\perp$ are independent standard normals; the price shock
    constructed from them has correlation $\rho$ with the variance shock.
    Both equations use the variance **before** the update.

    For each observed interval, the independent jump contribution is

    $$N_\Delta m_J+s_J\sqrt{N_\Delta}\,Z_J-\lambda c\Delta,
       \qquad N_\Delta\sim\operatorname{Poisson}(\lambda\Delta).$$

    The full Poisson count allows several jumps in one interval. The conditional
    normal total exactly aggregates their log sizes. Compensation subtracts
    $\lambda\,\mathbb E[e^J-1]$, preserving the specified proportional price
    drift $b$; subtracting $\lambda\,\mathbb E[J]$ would be different.
    Diffusion discretization error remains. Individual jump times and
    within-interval extrema are absent, so this output alone cannot support
    a continuously monitored barrier price.

    One simulated observation for later returns inference is an **entire
    ordered history**, with shape `[T, 1]`; a batch has shape `[B, T, 1]`.
    `variance_state` has shape `[B, T+1, 1]`, including the starting state.
    That array and the jump inputs are simulator truth, not observed conditioning.
    """)
    return


@app.cell
def _(mo):
    theta_control = mo.ui.slider(0.02, 0.08, step=0.005, value=0.04,
                                 label="Long-run variance θ", show_value=True)
    rho_control = mo.ui.slider(-0.9, 0.9, step=0.1, value=-0.6,
                               label="Shock correlation ρ", show_value=True)
    jumps_control = mo.ui.checkbox(value=True, label="Add Gaussian log-price jumps")
    seed_control = mo.ui.number(start=0, stop=999999, step=1, value=7412, label="Seed")
    mo.vstack([
        mo.md("## Try a small synthetic batch\n\nThree histories, 252 observed days, eight internal updates per day. Controls change this illustration, not the recorded validation domain."),
        mo.hstack([theta_control, rho_control]),
        mo.hstack([jumps_control, seed_control]),
    ])
    return jumps_control, rho_control, seed_control, theta_control


@app.cell
def _(heston, jumps_control, np, rho_control, seed_control, theta_control):
    parameters = dict(drift=0.05, kappa=2.0, theta=theta_control.value,
                      vol_of_vol=0.3, rho=rho_control.value,
                      jump_intensity=0.7 if jumps_control.value else 0.0,
                      jump_mean=-0.07, jump_std=0.1)
    dt = np.full(252, 1 / 252)
    innovations = heston.draw_innovations(
        dt, 8, seed=int(seed_control.value), example_ids=[0, 1, 2],
        jump_intensity=parameters["jump_intensity"], stream_id=0,
    )
    paths = heston.simulate(parameters, dt, 0.04, innovations, backend="numba")
    return dt, innovations, parameters, paths


@app.cell
def _(np, paths, plt):
    _fig, _axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, layout="constrained")
    for _i, _color in enumerate(["#4d749b", "#c06b48", "#568665"]):
        _axes[0].plot(np.arange(1, 253), 100 * np.expm1(paths["log_returns"][_i, :, 0]),
                      color=_color, linewidth=0.9, alpha=0.8, label=f"example {_i}")
        _axes[1].plot(np.arange(253), np.maximum(paths["variance_state"][_i, :, 0], 0),
                      color=_color, linewidth=1.1)
    _axes[0].set(ylabel="Daily simple return (%)", title="Synthetic observations: exp(log return) − 1")
    _axes[0].legend(ncols=3)
    _axes[1].set(xlabel="Observed trading day", ylabel="Effective variance",
                 title="Latent simulator truth: max(signed auxiliary, 0)")
    _fig
    return


@app.cell
def _(dt, heston, innovations, mo, np, parameters, paths, pd):
    _numpy = heston.simulate(parameters, dt, 0.04, innovations, backend="numpy")
    _cut = 126
    _first = heston.simulate(parameters, dt[:_cut], 0.04,
                             {k: v[:, :_cut] for k, v in innovations.items()})
    _second = heston.simulate(
        parameters, dt[_cut:], _first["variance_state"][:, -1, 0],
        {k: v[:, _cut:] for k, v in innovations.items()},
    )
    _rejoined = {
        "log_returns": np.concatenate([_first["log_returns"], _second["log_returns"]], axis=1),
        "variance_state": np.concatenate([_first["variance_state"], _second["variance_state"][:, 1:]], axis=1),
    }
    _rows = []
    for _name in paths:
        for _label, _candidate in [("NumPy versus Numba", _numpy), ("Restart versus uninterrupted", _rejoined)]:
            _error = float(np.max(np.abs(paths[_name] - _candidate[_name])))
            np.testing.assert_allclose(paths[_name], _candidate[_name], rtol=0, atol=1e-13)
            _rows.append({"Matched-input check": _label, "Array": _name, "Max absolute difference": _error})
    mo.vstack([
        mo.md("""
        ## Same inputs, same discrete calculation

        NumPy supplies a readable reference; Numba compiles a separate serial
        float64 loop with `fastmath=False`. This small check allows absolute
        differences up to `1e-13`; it does not promise cross-platform bitwise
        equality. First use can incur compilation or cache-loading cost.

        For restart, we slice **one already-drawn set of innovations** at day126
        and carry the signed state forward. Drawing twice with the same seed
        is not this coupling. A genuinely new future uses a distinct stream ID,
        with the same parameters and the terminal signed state as its start.
        """),
        mo.ui.table(pd.DataFrame(_rows), selection=None, show_column_summaries=False),
    ])
    return


@app.cell
def _(heston, mo, np, pd):
    _params = dict(drift=0.08, kappa=1.0, theta=0.04, vol_of_vol=0.5, rho=1.0)
    _inputs = {"brownian": np.array([[[[-3.0, 2.0]], [[-3.0, 2.0]]]]),
               "jump_counts": np.zeros((1, 2), dtype=np.int64),
               "jump_normals": np.zeros((1, 2))}
    _whole = heston.simulate(_params, [0.5, 0.5], 0.01, _inputs)
    _signed = float(_whole["variance_state"][0, 1, 0])
    _clipped = heston.simulate(_params, [0.5], max(_signed, 0.0),
                               {k: v[:, 1:] for k, v in _inputs.items()})
    mo.vstack([
        mo.md(r"""
        ### Why retain a negative auxiliary state?

        This deliberately coarse, prescribed-shock example forces a negative
        intermediate $a$. Effective variance remains zero there, but retaining
        $a$ determines how far mean reversion must move it before it becomes
        positive again. Clipping on restart changes that next state.
        The economic variance is not negative: $a$ is a numerical auxiliary.
        This example illustrates the update, not a recommended simulation grid.
        """),
        mo.ui.table(pd.DataFrame([
            {"Quantity": "Signed state at split", "Value": _signed},
            {"Quantity": "Next state with signed restart", "Value": float(_whole["variance_state"][0, 2, 0])},
            {"Quantity": "Next state after clipping restart", "Value": float(_clipped["variance_state"][0, 1, 0])},
        ]), selection=None, show_column_summaries=False),
    ])
    return


@app.cell
def _(ROOT, json):
    numerical_report = json.loads((ROOT / "docs/results/heston-svj-validation.json").read_text())
    timing_report = json.loads((ROOT / "docs/results/heston-svj-benchmark.json").read_text())
    return numerical_report, timing_report


@app.cell
def _(mo, numerical_report, pd):
    _rows = [{"Case": c["case"], "Paths": c["paths"],
              "Finest-grid pilot passed": c["finest_grid_pilot_passed"],
              "Last paired refinement passed": c["paired_refinement"][-1]["passed"],
              "Negative terminal auxiliary (%)": round(100 * c["levels"][-1]["negative_terminal_auxiliary_fraction"], 2)}
             for c in numerical_report["cases"]]
    mo.vstack([
        mo.md(r"""
        ## Recorded numerical evidence

        The committed pilot used **8,192 paths per case**, 64 observation
        intervals over one year, and 1/4/16 internal substeps. It compared
        variance moments, expected log returns and selected joint characteristic
        functions of terminal log return and effective variance with independent
        continuous-model references. A characteristic function averages
        $e^{i(uX+wV)}$; varying $(u,w)$ probes distribution and dependence beyond
        the mean. The reference uses analytic moments and affine Riccati ODEs.

        Finest-grid and last paired-refinement discrepancies must fit a stated
        absolute budget **plus five Monte Carlo standard errors**. Passing is a
        bounded diagnostic result, not a proof of the remaining bias or an
        accuracy certificate for arbitrary priors, grids, pricing or inference.
        The near-zero case deliberately violates the usual Feller condition
        $2\kappa\theta\ge\xi^2$; negative auxiliary states are recorded, not
        rejected or hidden. Numerical validation and posterior validation
        require different checks.
        """),
        mo.ui.table(pd.DataFrame(_rows), selection=None, show_column_summaries=False),
    ])
    return


@app.cell
def _(mo, pd, timing_report):
    _rows = []
    for _label, _key in [("Warm simulation: 256 paths", "warm_simulation_one_batch"),
                          ("Inputs + simulation: 1,024 paths", "complete_simulation")]:
        _rows.append({"Measurement": _label,
                      **{b: round(1000 * timing_report[_key][b]["median_seconds"], 2)
                         for b in ("numpy", "numba")}})
    mo.vstack([
        mo.md(f"""
        ## Recorded timing evidence — milliseconds

        One local macOS ARM64 run, 252 observations and eight substeps per
        observation; medians of {timing_report['repeats']} repetitions.
        This configuration differs from the numerical-refinement study above.
        Timings include argument validation and output allocation. Complete
        simulation includes random inputs; both exclude bank IO and imports.
        """),
        mo.ui.table(pd.DataFrame(_rows), selection=None, show_column_summaries=False),
        mo.md(f"""
        Warm kernel gain: **{timing_report['warm_speedup_numpy_over_numba']:.1f}×**;
        including random inputs: **{timing_report['complete_speedup_numpy_over_numba']:.1f}×**.
        Input generation alone took {1000 * timing_report['rng_one_batch']['median_seconds']:.1f}ms
        per batch; the Numba first call with a fresh cache took
        {timing_report['first_numba_call_seconds']:.2f}s. Software versions:
        Python {timing_report['environment']['python']}, NumPy
        {timing_report['environment']['numpy']}, Numba
        {timing_report['environment']['numba']}.

        These fixed-parameter CPU measurements establish neither cluster
        throughput nor an inference or pricing speedup. No fresh timings are
        collected when this notebook opens. The reports retain seeds, source
        hashes, parameters, repeat timings and measurement boundaries.
        """),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Review and reproduce

    1. Set $\rho=0$ and disable jumps. Does variance become constant? Which
       parameter must change for that, and what initialization gives GBM?
    2. Why might a year with $\lambda=0.7$ contain no jumps? Why is a
       Bernoulli jump indicator insufficient for larger observation intervals?
    3. Which arrays could be observed from prices alone? Why would supplying
       the plotted variance to an inference network leak synthetic truth?
    4. Explain why backend agreement does not establish continuous-model
       accuracy, and why numerical accuracy does not establish posterior accuracy.

    **Review hints:** $\rho=0$ removes instantaneous shock correlation, while
    $\xi$ still drives variance. Set $\xi=0$ and $a_0=\theta$ for constant
    variance. The probability of no jumps over a year is $e^{-0.7}$; a Poisson
    count can exceed one. Prices give returns, not exact latent states or jump
    counts. Two backends can agree on the same discretization error.

    Reproduce the focused checks and larger recorded experiments explicitly
    from the repository root (use new output paths):

    ```bash
    pixi run test-heston
    pixi run heston-validate --out /tmp/heston-validation-new.json
    pixi run heston-benchmark --out /tmp/heston-benchmark-new.json
    ```

    The [kernel guide](../docs/heston-svj-kernels.md) explains timing boundaries
    and fresh-cache measurements. Inspect the [implementation](../src/sbi4finance/heston.py),
    [focused tests](../tests/test_heston.py), [refinement script](../scripts/validate_heston.py),
    [numerical report](../docs/results/heston-svj-validation.json), and
    [timing report](../docs/results/heston-svj-benchmark.json).
    The guide links the full-truncation and affine-transform references.

    The next [joint-inference notebook](05_heston_joint_inference.py) defines
    a separate restricted task with **one** substep per observed day and jointly
    infers a parameter and current state. Its acceptance does not follow from
    this kernel pilot. Synthetic training needs no market observations;
    optional real-data application and held-out empirical evaluation add
    separate evidence requirements.
    """)
    return


if __name__ == "__main__":
    app.run()
