import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import logging
    import os
    import warnings

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    os.environ.setdefault("KERAS_BACKEND", "jax")  # must precede the bayesflow import
    warnings.filterwarnings("ignore")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

    import bayesflow as bf

    from sbi4finance.mmar import univariate as model

    ROOT = mo.notebook_dir().parent
    return ROOT, bf, mo, model, np, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Basic worked example: amortized inference for a multifractal volatility model

    Schedule item 4. One asset, 256 trading days, four parameters. We

    1. write down a simulator (no likelihood needed),
    2. train a BayesFlow approximator on simulated data, or load the pretrained one,
    3. check it on fresh simulations (parameter recovery, calibration),
    4. apply it to real returns.

    ## 1. The model

    Daily simple returns follow a multifractal model of asset returns (MMAR; Calvet, Fisher & Mandelbrot 1997):

    $$r_t = \mu + \bar\sigma \sqrt{w_t}\, \varepsilon_t, \qquad \varepsilon_t \sim t_\nu \text{ scaled to unit variance},$$

    where $w_t$ is a volatility cascade: eight 32-day blocks whose weights come from a binomial cascade with contrast $q$ (larger $q$ means burstier volatility), normalized to mean one, with a random circular phase. The cascade makes the likelihood intractable, but simulating is cheap: 100,000 paths take about a second.
    """)
    return


@app.cell
def _(model):
    model.prior_table()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What does $q$ do? Same shocks and phase, only the cascade contrast changes.
    """)
    return


@app.cell
def _(mo):
    q_slider = mo.ui.slider(0.52, 0.90, step=0.02, value=0.72, label="cascade contrast q")
    q_slider
    return (q_slider,)


@app.cell
def _(model, plt, q_slider):
    _paths = model.q_scenarios((q_slider.value,), n_paths=3)[q_slider.value]
    _fig, _ax = plt.subplots(figsize=(9, 3))
    _ax.plot(_paths.T, lw=0.8)
    _ax.set(xlabel="trading day", ylabel="return (%)", title=f"three simulated years at q = {q_slider.value:.2f}")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Data

    Adjusted-close daily simple returns for SPMO and VOO, cached in `data/` so the notebook never touches the network. We fit the last 256 trading days.
    """)
    return


@app.cell
def _(ROOT, model, pd):
    window = pd.read_csv(
        ROOT / "data" / "SPMO_VOO_2020-01-01_2026-09-04.csv", index_col=0, parse_dates=True
    ).tail(model.WINDOW)
    TICKERS = tuple(window.columns)
    window.describe().T
    return TICKERS, window


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Prior predictive check

    Before training anything: do prior draws produce series that look like markets? Simulate 3,000 paths and compare a few summaries of the observed windows with the prior's 5-95% band.
    """)
    return


@app.cell
def _(TICKERS, model, np, pd, window):
    def _summaries(x):
        wealth = np.cumprod(1 + x, axis=1)
        drawdown = (wealth / np.maximum.accumulate(wealth, axis=1) - 1).min(axis=1)
        return pd.DataFrame(
            {
                "volatility %": 100 * x.std(axis=1, ddof=1),
                "5% quantile %": 100 * np.quantile(x, 0.05, axis=1),
                "max drawdown %": 100 * drawdown,
            }
        )

    _prior = _summaries(model.simulate(3_000, seed=42)["returns"]).quantile([0.05, 0.95]).T
    _prior.columns = ["prior 5%", "prior 95%"]
    _observed = _summaries(window.to_numpy().T).T
    _observed.columns = list(TICKERS)
    prior_check = pd.concat([_prior, _observed], axis=1)
    prior_check
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Amortized inference

    The adapter turns simulator output into network input: bounded parameters are mapped to the real line, the four parameters are concatenated into the inference target, and each return series becomes a (256, 1) time series for the summary network.
    """)
    return


@app.cell
def _(bf, model):
    adapter = (
        bf.Adapter()
        .convert_dtype("float64", "float32")
        .constrain("sigma_bar", lower=model.PRIOR.sigma_low, upper=model.PRIOR.sigma_high)
        .constrain("q", lower=model.PRIOR.q_low, upper=model.PRIOR.q_high)
        .constrain("nu", lower=model.PRIOR.nu_low, upper=model.PRIOR.nu_high)
        .concatenate(model.PARAMETER_NAMES, into="inference_variables")
        .as_time_series("returns")
        .rename("returns", "summary_variables")
    )
    adapter
    return (adapter,)


@app.cell
def _(mo):
    mo.md(r"""
    A `FusionTransformer` summarizes each 256-day series into a vector; a spline `CouplingFlow` maps that vector to the posterior. The pretrained checkpoint used 100,000 simulations and 60 epochs (about 10 minutes on a 2022 GPU; `scripts/train_univariate.py` reproduces it). The toy option trains on 2,000 simulations for five epochs, under a minute on a laptop CPU: enough to see the mechanics, not enough for calibrated posteriors.
    """)
    return


@app.cell
def _(mo):
    mode = mo.ui.radio(
        options=["load pretrained checkpoint", "train a toy model on CPU"],
        value="load pretrained checkpoint",
        label="approximator",
    )
    mode
    return (mode,)


@app.cell
def _(ROOT, adapter, bf, mode, model):
    _summary_network = bf.networks.FusionTransformer(dropout=0.1)
    _inference_network = bf.networks.CouplingFlow(transform="spline", depth=4)
    if mode.value == "load pretrained checkpoint":
        workflow = bf.BasicWorkflow(
            adapter=adapter,
            summary_network=_summary_network,
            inference_network=_inference_network,
            standardize="all",
            checkpoint_filepath=str(ROOT / "checkpoints" / "univariate"),
        )
        workflow.load_approximator()
        history = None
    else:
        workflow = bf.BasicWorkflow(
            adapter=adapter,
            summary_network=_summary_network,
            inference_network=_inference_network,
            standardize="all",
        )
        history = workflow.fit_offline(
            data=model.simulate(2_000, seed=1),
            validation_data=model.simulate(200, seed=2),
            epochs=5,
            batch_size=64,
        )
    return history, workflow


@app.cell
def _(bf, history, mo):
    bf.diagnostics.loss(history) if history is not None else mo.md(
        "*Pretrained checkpoint loaded; no training curve to show.*"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Does it work? Recovery and calibration on fresh simulations

    300 new simulations with known parameters. Recovery plots posterior medians against the truth; the rank-ECDF plot checks simulation-based calibration (the curves should stay inside the band).
    """)
    return


@app.cell
def _(model, workflow):
    test_sims = model.simulate(300, seed=555)
    estimates = workflow.sample(conditions=test_sims, num_samples=500)
    return estimates, test_sims


@app.cell
def _(bf, estimates, model, test_sims):
    bf.diagnostics.recovery(estimates=estimates, targets=test_sims, variable_names=model.PARAMETER_LABELS)
    return


@app.cell
def _(bf, estimates, model, test_sims):
    bf.diagnostics.calibration_ecdf(
        estimates=estimates, targets=test_sims, variable_names=model.PARAMETER_LABELS, difference=True
    )
    return


@app.cell
def _(estimates, test_sims, workflow):
    workflow.compute_default_diagnostics(test_sims, samples=estimates)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Posterior for the observed windows

    Inference on real data is now a forward pass: 1,000 posterior draws per ticker in well under a second on CPU.
    """)
    return


@app.cell
def _(TICKERS, model, np, pd, window, workflow):
    posterior = workflow.sample(
        conditions={"returns": window.to_numpy().T[..., None]}, num_samples=1_000
    )
    pd.DataFrame(
        {
            (ticker, stat): [np.quantile(posterior[name][i], q) for name in model.PARAMETER_NAMES]
            for i, ticker in enumerate(TICKERS)
            for stat, q in (("5%", 0.05), ("median", 0.5), ("95%", 0.95))
        },
        index=model.PARAMETER_NAMES,
    )
    return (posterior,)


@app.cell
def _(TICKERS, mo):
    ticker = mo.ui.dropdown(options=list(TICKERS), value=TICKERS[-1], label="ticker")
    ticker
    return (ticker,)


@app.cell
def _(TICKERS, bf, model, posterior, ticker):
    bf.diagnostics.pairs_posterior(
        posterior, dataset_id=TICKERS.index(ticker.value), variable_names=model.PARAMETER_LABELS
    ).figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What this establishes

    One simulator, one adapter, two networks: after training once, every new 256-day window costs a forward pass. Items 5 and 6 build on exactly this approximator: rolling VaR/ES with parameter uncertainty, and what happens when real data does not look like the simulator.

    *Simulator and checkpoint come from Stefan Radev's [amortized-fractals](https://github.com/stefanradev93/amortized-fractals).*
    """)
    return


if __name__ == "__main__":
    app.run()
