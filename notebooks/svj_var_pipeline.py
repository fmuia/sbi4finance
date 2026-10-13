import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import logging
    import os
    import time
    import warnings

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    os.environ.setdefault("KERAS_BACKEND", "jax")  # must precede the bayesflow import
    warnings.filterwarnings("ignore")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

    from sbi4finance.svj import amortized, audit, checks, data, filtering, model, plots, risk

    ROOT = mo.notebook_dir().parent
    plots.style()
    return ROOT, amortized, audit, checks, data, filtering, mo, model, np, pd, plt, plots, risk, time


@app.cell
def _(mo):
    mo.md(r"""
    # Advanced example: from four years of returns to VaR and ES with parameter uncertainty

    A risk team reports the 10-day 99% VaR and 97.5% ES of thousands of names every morning. A stochastic-volatility
    model with jumps (SVJ) describes daily equity returns well, but four years of data hold only about ten jumps, so its
    tail parameters are uncertain, and that uncertainty moves the expected shortfall. The exact posterior needs particle
    MCMC, minutes per name. Amortized inference pays that cost once, in training, and then returns the joint posterior
    of the parameters and today's volatility in about a second; an importance-sampling audit with the particle filter
    checks it name by name.

    Every step below calls `src/sbi4finance/svj`; the scripts in `scripts/svj_*.py` run the same steps at full scale.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The model

    Daily log returns in percent follow the Bates model in the daily form of Eraker, Johannes and Polson (2003):

    $$y_t = \mu + \sqrt{v_{t-1}}\,\varepsilon^1_t + B_t\,\xi_t, \qquad B_t \sim \mathrm{Bernoulli}(\lambda),\ \ \xi_t \sim \mathcal N(\mu_J, \sigma_J^2),$$

    $$v_t = \max\!\big(v_{t-1} + \kappa(\theta - v_{t-1}) + \sigma_v\sqrt{v_{t-1}}\,(\rho\,\varepsilon^1_t + \sqrt{1-\rho^2}\,\varepsilon^2_t),\ v_{\min}\big).$$

    The variance is latent and mean-reverting, jumps are rare and large, and only the diffusive shock moves the variance.
    The eight parameters are drawn in coordinates a risk manager can judge:
    """)
    return


@app.cell
def _(model):
    model.prior_table().round(2)
    return


@app.cell
def _(model, np):
    example_coords = np.array([8.0, np.log(0.18), np.log(35), np.log(0.55), np.arctanh(-0.6), np.log(2.0), -2.5, np.log(3.5)])
    example = model.simulate_from_coordinates(example_coords[None], np.random.default_rng(3))
    example_y = example["returns"][0]
    return example, example_coords, example_y


@app.cell
def _(example, example_y, model, np, plots, plt):
    _fig, (_top, _bottom) = plt.subplots(2, 1, figsize=(9.6, 4.6), sharex=True, constrained_layout=True)
    _days = np.arange(1, len(example_y) + 1)
    _jumps = example["jumps"][0]
    _top.plot(_days, example_y, color=plots.SERIES[0], linewidth=0.7)
    _top.scatter(_days[_jumps], example_y[_jumps], s=28, color=plots.SERIES[1], edgecolor=plots.SURFACE, zorder=3, label="jump day")
    _top.set_ylabel("daily return (%)")
    _top.legend(loc="lower right")
    _bottom.plot(_days, model.spot_vol_from_variance(example["variance"][0]), color=plots.SERIES[0], linewidth=1.2)
    _bottom.set_ylabel("spot volatility (latent)")
    _bottom.set_xlabel("trading day")
    _fig.suptitle("One simulated four-year path: what a risk manager sees, and the volatility that drove it",
                  x=0.01, ha="left", fontweight="bold", fontsize=10.5)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Why the likelihood is intractable

    The likelihood integrates over the whole latent variance path, a 1,001-dimensional integral with no closed form.
    The fully adapted particle filter estimates it without bias: weight particles by the density of today's return,
    resample, and move each particle with the exact law of the next variance given that return. The estimate is noisy,
    and the noise falls with the number of particles; the deterministic grid filter gives the reference value.
    """)
    return


@app.cell
def _(example_coords, example_y, filtering, model, np, pd, time):
    _params = model.model_params(example_coords)
    _rows = []
    for _n in (50, 100, 400):
        _start = time.perf_counter()
        _ll, _ = filtering.loglik_batch(np.repeat(_params[None], 64, 0), example_y, seed=_n, n_particles=_n, chunk=64)
        _rows.append({"particles": _n, "mean log L-hat": _ll.mean(), "sd of log L-hat": _ll.std(),
                      "seconds for 64 runs": time.perf_counter() - _start})
    _grid = float(filtering.grid_loglik(_params, example_y, n_cells=200))
    pd.DataFrame(_rows).set_index("particles").round(3).assign(**{"grid log L": round(_grid, 3)})
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Amortized posterior

    The network was trained once on 2.56 million simulated series (`pixi run svj-train`; about an hour on a few CPU cores).
    A summary network turns the 1,000 days into 32 statistics, and 50 Euler steps of a flow-matching network carry noise to
    1,000 joint draws of the eight parameters and today's spot volatility. On the example path, the truth is known:
    """)
    return


@app.cell
def _(ROOT, amortized, checks):
    checkpoint = ROOT / "checkpoints" / "svj"
    posterior = amortized.AmortizedPosterior.load(checkpoint)
    misspecification = checks.MisspecificationTest.load(checkpoint / "misspecification.npz")
    return misspecification, posterior


@app.cell
def _(example, example_y, model, np, pd, posterior, time):
    _start = time.perf_counter()
    example_draws = posterior.sample(example_y, num_samples=1_000, seed=0)[0]
    _seconds = time.perf_counter() - _start
    _q = model.quantities(example_draws)
    _truth = model.quantities(example["targets"][0])
    pd.DataFrame(
        {"truth": _truth, "5%": np.quantile(_q, 0.05, 0), "median": np.median(_q, 0), "95%": np.quantile(_q, 0.95, 0)},
        index=model.QUANTITY_LABELS,
    ).round(3).rename_axis(f"posterior in {_seconds:.1f} s")
    return (example_draws,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. A real window: VOO, the last 1,000 trading days of the bundled CSV

    Before trusting the network on real data, two checks. The misspecification test asks whether the window looks like
    the training data in the space of the 32 learned summaries (a small p-value flags it). The audit draws proposals from
    a Student t fitted to the network's draws and reweights them with the particle filter's likelihood estimate; the share
    of effective draws is the audit score, and a low score sends the name to particle MCMC.
    """)
    return


@app.cell
def _(mo):
    ticker = mo.ui.dropdown(options=["VOO", "SPMO"], value="VOO", label="ticker")
    ticker
    return (ticker,)


@app.cell
def _(ROOT, audit, data, misspecification, posterior, ticker, time):
    window = data.window(data.load_returns(ROOT / "data" / "SPMO_VOO_2020-01-01_2026-09-04.csv", ticker.value))
    real_y = window.to_numpy()
    real_draws = posterior.sample(real_y, num_samples=1_000, seed=0)[0]
    pvalue = float(misspecification.pvalue(posterior.summaries(real_y))[0])
    _start = time.perf_counter()
    checked = audit.audit(real_draws, real_y, seed=0, n_proposals=1_000, n_particles=100)
    audit_seconds = time.perf_counter() - _start
    return audit_seconds, checked, pvalue, real_draws, real_y, window


@app.cell
def _(audit_seconds, checked, mo, pvalue, window):
    mo.md(f"""
    **{window.index[0].date()} to {window.index[-1].date()}**: misspecification p-value **{pvalue:.2f}**;
    audit efficiency **{checked.efficiency:.1%}** ({checked.ess:.0f} effective draws of 1,000 proposals,
    {checked.n_particles} particles, {audit_seconds:.0f} s). The full-scale audit uses 3,000 proposals and picks the particle
    count from the noise of the likelihood estimate (`pixi run svj-daily`).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. From posterior to VaR and ES

    From every posterior draw the daily model runs forward 10 days, 20 paths per draw; the pooled returns are the
    predictive distribution. The plug-in forecast fixes the parameters at the posterior median and keeps only the
    uncertainty about today's variance. The conditional VaR is the number an analyst would report if parameters and
    today's variance were known: its posterior shows how uncertain the risk number itself is.
    """)
    return


@app.cell
def _(checked, pd, real_y, risk):
    forecast = risk.forecast(checked.posterior, real_y, seed=0, split=True)
    pd.DataFrame(
        {
            (f"{h}-day", name): [forecast[key][h]["var"], forecast[key][h]["es"]]
            for h in (1, 10)
            for name, key in (("predictive", "predictive"), ("plug-in", "plugin_at_posterior_median"))
        },
        index=["99% VaR (%)", "97.5% ES (%)"],
    ).round(2)
    return (forecast,)


@app.cell
def _(checked, forecast, np, plots, plt, risk):
    _cvar = risk.conditional_var(checked.posterior)
    _c = forecast["conditional_var_1d"]
    _fig, _ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
    _ax.hist(_cvar, bins=40, color=plots.SERIES[0], alpha=0.85, edgecolor=plots.SURFACE, linewidth=0.6)
    _ax.axvline(forecast["predictive"][1]["var"], color=plots.INK, linewidth=1.2, label="predictive VaR")
    _ax.axvline(forecast["plugin_at_posterior_median"][1]["var"], color=plots.SERIES[1], linewidth=1.2, label="plug-in VaR")
    _ax.set_xlabel("1-day 99% conditional VaR (%)")
    _ax.legend(loc="upper right")
    _ax.set_title(
        f"Posterior of the risk number: today's variance carries {_c['state_share']:.0%} of its variance, "
        f"the parameters {_c['parameter_share']:.0%}", loc="left")
    _ax.grid(axis="y", visible=False)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What this establishes, and what runs at full scale

    One forward pass gives the joint posterior of the parameters and today's volatility; the audit checks it with the
    exact likelihood estimate; the forecast carries both into VaR and ES. The evidence behind the two claims of the
    advanced segment runs as scripts, with results under `docs/results/`:

    | Script | Question |
    |---|---|
    | `pixi run svj-validate` | Is the network calibrated (SBC), and what do four years of returns identify? |
    | `pixi run svj-agreement` | Does it agree with particle MCMC on single series, before and after the audit? |
    | `pixi run svj-backtest` | Do predictive and plug-in VaR / ES hold up against a known truth? |
    | `pixi run svj-rolling` | How do rolling posteriors and the misspecification test behave on real data? |

    Exercises: change the jump-frequency prior in `model.Prior` and retrain the laptop variant
    (`pixi run svj-train --preset laptop`, about ten minutes); compare its contraction with the reference network.
    """)
    return


if __name__ == "__main__":
    app.run()
