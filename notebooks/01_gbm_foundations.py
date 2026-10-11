import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path
    from statistics import NormalDist
    from tempfile import TemporaryDirectory
    from time import perf_counter

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from sbi4finance import gbm, gbm_bank, gbm_inference

    ROOT = mo.notebook_dir().parent
    return NormalDist, Path, ROOT, TemporaryDirectory, gbm, gbm_bank, gbm_inference, json, mo, np, pd, perf_counter, plt


@app.cell
def _(ROOT, json):
    reference_config = json.loads((ROOT / "configs" / "tasks" / "gbm_drift.json").read_text())
    full_bank_config = json.loads((ROOT / "configs" / "tasks" / "gbm_bank.json").read_text())
    cluster_guide = (ROOT / "docs" / "gbm-cluster-smoke.md").read_text()
    return cluster_guide, full_bank_config, reference_config


@app.cell
def _(mo):
    mo.md(r"""
    # 1–3. A small, checkable foundation for simulation-based inference

    This is a **review walkthrough of three implemented increments**, not the
    final course or a commitment to one financial application. We start with a
    model whose answers can be calculated exactly, then show how to preserve the
    experiment when saving data and running the same generator on a cluster.

    | Increment | Building block | What this notebook does |
    |---|---|---|
    | 1: M0/M1 | Define the task; exact GBM simulation, inference and pricing | Recompute a small example with analytic checks |
    | 2: M2.1–M2.3 | Save a synthetic bank; train and reload a posterior | Write/load a tiny real bank; review recorded training results |
    | 3: M2.4 | Generate and verify a bank on Oscar | Inspect historical execution evidence |

    The default run is offline and uses NumPy, small temporary files and plots.
    It performs **no neural training, downloads, SSH or job submission**.
    Current calculations and historical measurements are labeled separately.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Increment 1: define the statistical experiment before writing kernels

    Geometric Brownian motion (GBM) uses constant annual volatility $\sigma$:

    $$dS_t/S_t=\mu\,dt+\sigma\,dW_t,\qquad
    r_i=\log(S_{t_i}/S_{t_{i-1}})
       \sim\mathcal N((\mu-\sigma^2/2)\Delta_i,\sigma^2\Delta_i).$$

    Here $\mu$ is proportional **price drift**, not expected log-return drift.
    Volatility 0.20 means 20% annually; a daily interval is $\Delta_i=1/252$
    trading years. These increments are exact: no Euler timestep error is
    introduced by the GBM simulator.

    | Role | Example | May inference condition on it? |
    |---|---|---|
    | Observation | A complete ordered history of decimal log returns | Yes |
    | Known context | Year fractions and known volatility | Yes |
    | Synthetic truth | The drift that generated that history | No; supervised target/evaluation only |
    | Future outcomes | Returns after the information cutoff | No; prediction/evaluation only |
    | Example ID | A stable bank identity | No |

    One complete history is one inference example: `[batch, time, 1]`.
    A market history would not come with a known generating drift. Missingness
    is not filled silently, and a compatible shape alone does not establish
    scientific compatibility.

    We first work under the **physical measure $P$**, asking what historical
    returns imply about drift and future losses. A later, separate **risk-neutral
    $Q$** calculation prices a European payoff using specified financing inputs.
    The $P$ drift posterior is not a $Q$ calibration.
    """)
    return


@app.cell
def _(mo):
    history_length = mo.ui.slider(63, 1008, step=63, value=252, label="Observed trading days")
    history_seed = mo.ui.slider(2026, 2035, step=1, value=2026, label="Synthetic history seed")
    mo.vstack([
        mo.md("Choose how much history to observe. At a fixed seed, longer windows extend the same path."),
        mo.hstack([history_length, history_seed]),
    ])
    return history_length, history_seed


@app.cell
def _(gbm, history_length, history_seed, np, reference_config):
    physical_context = reference_config["returns"]
    observed_dt = np.full(history_length.value, physical_context["dt_years"])
    _history_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([history_seed.value, 0])))
    observed_returns = gbm.simulate_log_returns(
        physical_context["true_drift"], physical_context["known_volatility"], observed_dt, rng=_history_rng
    )
    observed_prices = gbm.prices_from_log_returns(observed_returns, physical_context["initial_price"])
    posterior_mean, posterior_std = gbm.drift_posterior(
        observed_returns, physical_context["known_volatility"], observed_dt,
        prior_mean=physical_context["prior_mean"], prior_std=physical_context["prior_std"],
    )
    return observed_dt, observed_prices, observed_returns, physical_context, posterior_mean, posterior_std


@app.cell
def _(mo, observed_dt, observed_returns, posterior_mean, posterior_std):
    mo.md(rf"""
    ### An exact posterior to check later approximations

    Use the Gaussian prior $\mu\sim\mathcal N(m_0,s_0^2)$ with $m_0=0$ and
    $s_0=0.10$. Let $H=\sum_i\Delta_i$ and $R=\sum_i r_i$. Then

    $$s_1^2=\left(s_0^{{-2}}+H/\sigma^2\right)^{{-1}},\qquad
    m_1=s_1^2\left[m_0/s_0^2+(R+\sigma^2H/2)/\sigma^2\right].$$

    The current history has shape **`{observed_returns.shape}`**, spans
    **{observed_dt.sum():.2f} years**, and gives
    $\mu\mid r_{{1:T}}\sim\mathcal N({posterior_mean[0]:.5f},{posterior_std[0]:.5f}^2)$.
    Those are annual decimal drift units. The generating truth is not used by
    `drift_posterior`; it appears only as an evaluation marker below.

    Uncertainty depends on elapsed time $H$, not just the number of observations.
    Sampling the same year more finely does not create more information about
    constant drift when volatility is known. The total return $R$ is sufficient
    for this special task; that shortcut will not carry over to hidden-volatility
    models.
    """)
    return


@app.cell
def _(np, observed_dt, observed_prices, physical_context, plt, posterior_mean, posterior_std):
    _fig, _axes = plt.subplots(1, 2, figsize=(10.5, 3.5), layout="constrained")
    _axes[0].plot(np.r_[0, np.cumsum(observed_dt)], observed_prices[0, :, 0], color="#4d749b")
    _axes[0].set(xlabel="Elapsed trading years", ylabel="Price (arbitrary currency units)", title="Observed synthetic history")
    _grid = np.linspace(-0.4, 0.5, 500)
    for _mean, _std, _label, _color in [
        (physical_context["prior_mean"], physical_context["prior_std"], "prior", "#71894a"),
        (posterior_mean[0], posterior_std[0], "exact posterior", "#4d749b"),
    ]:
        _density = np.exp(-0.5 * ((_grid - _mean) / _std) ** 2) / (_std * np.sqrt(2 * np.pi))
        _axes[1].plot(_grid, _density, label=_label, color=_color)
    _axes[1].axvline(physical_context["true_drift"], color="#c06b48", linestyle="--", label="synthetic truth")
    _axes[1].set(xlabel="Annual proportional drift mu", ylabel="Density",
                 title=f"Drift after {observed_dt.sum():.2f} observed years")
    _axes[1].legend(fontsize=8)
    _fig
    return


@app.cell
def _(gbm, history_seed, np, physical_context, posterior_mean, posterior_std):
    future_dt = np.full(physical_context["forecast_steps"], physical_context["dt_years"])
    predictive_mean, predictive_std = gbm.predictive_log_return(
        posterior_mean, posterior_std, physical_context["known_volatility"], future_dt.sum()
    )
    _parameter_rng, _future_rng = [np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([history_seed.value, purpose]))) for purpose in (1, 2)]
    _drift_draws = _parameter_rng.normal(posterior_mean[0], posterior_std[0], size=20_000)
    # Each future path keeps ONE drift draw for all ten future increments.
    future_returns = gbm.simulate_log_returns(
        _drift_draws, physical_context["known_volatility"], future_dt, rng=_future_rng
    )
    total_future_returns = future_returns.sum(axis=1)[:, 0]
    future_losses = -np.expm1(total_future_returns)
    return future_dt, future_losses, predictive_mean, predictive_std, total_future_returns


@app.cell
def _(NormalDist, future_dt, future_losses, mo, np, predictive_mean, predictive_std, total_future_returns):
    _analytic_var = -np.expm1(predictive_mean[0] + predictive_std[0] * NormalDist().inv_cdf(0.01))
    _mc_var = np.quantile(future_losses, 0.99)
    _mean_se = predictive_std[0] / np.sqrt(len(total_future_returns))
    mo.md(rf"""
    ### From a posterior to future outcomes

    For a future horizon $h={future_dt.sum():.5f}$ years (ten trading days), the
    total log return has the exact posterior-predictive law

    $$R_h\mid r_{{1:T}}\sim\mathcal N\!\left((m_1-\sigma^2/2)h,
                            \underbrace{{\sigma^2h}}_{{\text{{new process shocks}}}}
                            +\underbrace{{s_1^2h^2}}_{{\text{{uncertain fixed drift}}}}\right).$$

    The simulation draws **one drift per future path** and independent process
    shocks thereafter. Redrawing drift every day would change this law.
    There is no hidden variance state in constant-volatility GBM.

    Current check: simulated mean log return **{total_future_returns.mean():.6f}**,
    analytic mean **{predictive_mean[0]:.6f}**, Monte Carlo standard error
    **{_mean_se:.6f}**. More simulation reduces that sampling error; it does not
    eliminate the posterior uncertainty represented by $s_1$.

    Define relative loss $L=1-e^{{R_h}}$: positive means a loss on a long position.
    The exact 99% predictive loss quantile is **{100*_analytic_var:.2f}%** of
    current value; the 20,000-draw estimate is **{100*_mc_var:.2f}%**.
    This is a synthetic model calculation, not a market backtest. It is a
    quantile of the predictive mixture, not an average of conditional VaRs.
    """)
    return


@app.cell
def _(np, plt, predictive_mean, predictive_std, total_future_returns):
    _fig, _ax = plt.subplots(figsize=(8.5, 3.2), layout="constrained")
    _ax.hist(total_future_returns, bins=55, density=True, color="#4d749b", alpha=0.5,
             label="20,000 posterior-predictive paths")
    _x = np.linspace(predictive_mean[0] - 4 * predictive_std[0], predictive_mean[0] + 4 * predictive_std[0], 300)
    _density = np.exp(-0.5 * ((_x - predictive_mean[0]) / predictive_std[0]) ** 2) / (predictive_std[0] * np.sqrt(2 * np.pi))
    _ax.plot(_x, _density, color="black", linewidth=1.7, label="exact Gaussian predictive law")
    _ax.set(xlabel="Ten-day total log return (decimal)", ylabel="Density", title="Simulation has an independent analytic check")
    _ax.legend(fontsize=8)
    _fig
    return


@app.cell
def _(gbm, np, pd, reference_config):
    _q = reference_config["pricing"]
    _maturity = _q["maturity_years"]
    _forward = _q["spot"] * np.exp((_q["rate"] - _q["dividend_yield"]) * _maturity)
    _discount = np.exp(-_q["rate"] * _maturity)
    _strikes = np.asarray(_q["strikes"])
    _prices = gbm.european_price(_forward, _strikes, _discount, _q["volatility"] ** 2 * _maturity)
    _q_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([reference_config["seed"], 3])))
    _q_returns = gbm.simulate_log_returns(np.full(50_000, _q["rate"] - _q["dividend_yield"]),
                                         _q["volatility"], [_maturity], rng=_q_rng)
    _terminal = _q["spot"] * np.exp(_q_returns[:, 0, 0])
    _payoffs = _discount * np.maximum(_terminal[:, None] - _strikes, 0)
    _means = _payoffs.mean(axis=0)
    _se = _payoffs.std(axis=0, ddof=1) / np.sqrt(len(_payoffs))
    pricing_check = pd.DataFrame({
        "Strike": _strikes, "Analytic call price": _prices,
        "50,000-payoff mean": _means, "Monte Carlo SE": _se,
        "Error / MC SE": (_means - _prices) / _se,
        "One discounted payoff": _payoffs[0],
    })
    return (pricing_check,)


@app.cell
def _(mo, pricing_check):
    mo.vstack([
        mo.md(r"""
        ### Separate $Q$ reference: a payoff is not a price

        Independently specify spot 100, rate 5%, dividend yield 2%, volatility
        20% and one-year expiry. Here $F=100e^{(0.05-0.02)T}$,
        $D=e^{-0.05T}$ and integrated variance $w=\sigma^2T=0.04$.
        The call price is $D\,\mathbb E_Q[(S_T-K)^+]$.

        An individual discounted payoff is random; its average estimates the
        price. The Monte Carlo standard error is payoff SD divided by
        $\sqrt{N}$. A later supervised pricing surrogate needs a reliable
        expectation label and its numerical error, not one unqualified payoff.
        This small table is recomputed now, with no use of the $P$ posterior.
        """),
        mo.ui.table(pricing_check.round(5), selection=None, show_column_summaries=False),
        mo.md("""
        **Increment 1 result:** exact simulation, an exact drift posterior,
        a predictive reference and a separate European pricing reference are
        available. This is not option-surface calibration, an American pricer,
        or validation of GBM on markets. See the linked guide for the original
        recorded run and the mathematical derivation.
        """),
    ])
    return


@app.cell
def _(mo, np, observed_dt, observed_returns, pd, physical_context, pricing_check, reference_config):
    # Two concrete records: context is known at inference time; truth is not.
    returns_role_record = {
        "conditioning": {
            "observations": observed_returns,
            "context": {"measure": "P", "dt_years": observed_dt,
                        "known_volatility": physical_context["known_volatility"]},
        },
        "truth": {"mu": np.array([[physical_context["true_drift"]]])},
    }
    _q = reference_config["pricing"]
    _strikes = np.asarray(_q["strikes"], dtype=float)
    _maturity = _q["maturity_years"]
    option_panel_role_record = {
        "conditioning": {
            # One accurately computed expectation per contract, not one payoff.
            "observations": pricing_check["Analytic call price"].to_numpy()[None, :, None],
            "context": {
                "measure": "Q", "option_type": "call", "strike": _strikes,
                "maturity_years": np.full(len(_strikes), _maturity),
                "forward": np.full(len(_strikes), _q["spot"] * np.exp((_q["rate"] - _q["dividend_yield"]) * _maturity)),
                "discount_factor": np.full(len(_strikes), np.exp(-_q["rate"] * _maturity)),
            },
        },
        "truth": {"volatility": np.array([[_q["volatility"]]])},
    }
    for _record in (returns_role_record, option_panel_role_record):
        assert set(_record["conditioning"]) == {"observations", "context"}
        assert "truth" not in _record["conditioning"]
        assert not set(_record["truth"]) & set(_record["conditioning"]["context"])
    assert option_panel_role_record["conditioning"]["observations"].shape == (1, 3, 1)

    _tables = []
    for _record in (returns_role_record, option_panel_role_record):
        _rows = [{"Field": "observations", "Role": "conditioning",
                  "Shape": str(_record["conditioning"]["observations"].shape),
                  "Values (first 3)": np.round(_record["conditioning"]["observations"].ravel()[:3], 6).tolist()}]
        for _name, _value in _record["conditioning"]["context"].items():
            _array = np.asarray(_value)
            _rows.append({"Field": _name, "Role": "known context", "Shape": str(_array.shape),
                          "Values (first 3)": _array.ravel()[:3].tolist()})
        for _name, _value in _record["truth"].items():
            _rows.append({"Field": _name, "Role": "synthetic truth; excluded",
                          "Shape": str(_value.shape), "Values (first 3)": _value.ravel().tolist()})
        # Stringify mixed numeric/text lists for a predictable display table.
        for _row in _rows:
            _row["Values (first 3)"] = str(_row["Values (first 3)"])
        _tables.append(mo.ui.table(pd.DataFrame(_rows), selection=None, show_column_summaries=False))
    mo.vstack([
        mo.md("""
        ### M0 role records: two different inference questions

        These records are constructed now with explicit array shapes. `()`
        denotes a scalar; the first axis is a batch of one. Assertions check
        that each record's synthetic truth is outside its conditioning mapping.

        **Return-history record:** drift is unknown; volatility is declared known.
        """),
        _tables[0],
        mo.md("""
        **Hypothetical option-panel/calibration schema:** one clean call-price
        panel across three contracts has shape `[1, 3, 1]`. Each value is the
        analytic expectation from the preceding pricing check. Contract strikes,
        maturities, forwards and discount factors are known context; the
        generating volatility is separate truth, not an input to calibration.
        """),
        _tables[1],
        mo.md("""
        This second record illustrates roles only. It is **not an implemented
        option-calibration posterior, an HKDE example or a full option bank**.
        These are hypothetical clean prices, with no quote-noise law. A complete
        calibration task must separately specify its inferred quantities, prior,
        observation/noise model and reference validation. Its contract axis is
        different from the return-history time axis above.
        """),
    ])
    return option_panel_role_record, returns_role_record


@app.cell
def _(mo):
    mo.md("""
    ## Increment 2: persist examples without changing their meaning

    The default bank task has 8,192 training, 1,024 validation and 256 test
    histories. This notebook instead writes **64 / 16 / 8 examples** using the
    same scientific task, then verifies and loads the three NPZ files and JSON
    manifest. It creates a new temporary directory and cleans it up after
    loading; no project data or earlier run is overwritten.

    Each example's random identity combines **task fingerprint, root seed,
    split and local ID**. Prior and process shocks use separate random streams.
    A requested example therefore survives reordering, batching and expansion
    of the bank in the recorded environment. Different split names produce
    independent examples. Shared latent histories or overlapping real windows
    would require additional grouping rules in later tasks.
    """)
    return


@app.cell
def _(Path, TemporaryDirectory, full_bank_config, gbm_bank, json, np, perf_counter):
    _small_config = json.loads(json.dumps(full_bank_config))
    _small_config["splits"] = {"train": 64, "validation": 16, "test": 8}
    _started = perf_counter()
    with TemporaryDirectory(prefix="sbi4finance-gbm-review-") as _temporary:
        _bank_path = Path(_temporary) / "bank"
        gbm_bank.write_bank(_bank_path, _small_config)
        small_manifest, small_splits = gbm_bank.load_bank(_bank_path)
        bank_file_bytes = sum(path.stat().st_size for path in _bank_path.iterdir())
    bank_seconds = perf_counter() - _started
    _ids = [63, 7, 0]
    _regenerated = gbm_bank.generate_examples(_small_config, "train", _ids)
    stable_id_check = all(
        np.array_equal(_regenerated[key], small_splits["train"][key] if key == "dt_years"
                       else small_splits["train"][key][_ids])
        for key in _regenerated
    )
    if not stable_id_check:
        raise RuntimeError("Selected examples changed on independent reordered regeneration.")
    return bank_file_bytes, bank_seconds, small_manifest, small_splits, stable_id_check


@app.cell
def _(gbm_inference, mo, pd, small_manifest, small_splits, stable_id_check, bank_file_bytes, bank_seconds):
    _task = small_manifest["configuration"]["task"]
    bank_conditions = gbm_inference.conditioning_data(
        small_splits["train"]["observations"], small_splits["train"]["dt_years"], _task
    )
    bank_training_data = gbm_inference.training_data(small_splits["train"], _task)
    _rows = [{"Split": name, "Observed array": str(arrays["observations"].shape),
              "Target array": str(arrays["parameters"].shape), "First ID": arrays["example_id"][0]}
             for name, arrays in small_splits.items()]
    mo.vstack([
        mo.md(f"""
        **Executed now:** write/load/check took **{bank_seconds:.3f} s** and wrote
        **{bank_file_bytes:,} bytes** including metadata. Reordered regeneration
        of `train:63`, `train:7`, `train:0` matched exactly: **{stable_id_check}**.
        These are local measurements, not cluster or training results.
        """),
        mo.ui.table(pd.DataFrame(_rows), selection=None, show_column_summaries=False),
        mo.md(f"""
        `conditioning_data` returned only **`{list(bank_conditions)}`** with shape
        **`{bank_conditions['sum_log_returns'].shape}`**. For known volatility
        and fixed duration, the sum of log returns is sufficient for GBM drift.
        `training_data` separately adds **`mu`** as the supervised target.
        Analytic posterior answers are used for evaluation, not training labels.

        Manifest fields record array roles, time cutoff, source/configuration
        hashes, RNG construction, versions and individual file hashes. Reading
        a bank checks integrity, shapes and identity. **A hash verifies identity,
        not scientific correctness**; regeneration and analytic checks supply
        different evidence. The complete manifest is available below for review.
        """),
        mo.accordion({"Current tiny-bank manifest": mo.json(small_manifest)}),
    ])
    return bank_conditions, bank_training_data


@app.cell
def _(mo):
    mo.md(r"""
    ### Training, saving and reloading are explicit steps

    The bank fits in memory, so the existing `fit_offline` interface suffices.
    The following is a **code example, not executed by this notebook**:

    ```python
    task = manifest["configuration"]["task"]
    workflow = gbm_inference.build_workflow(new_checkpoint_directory)
    workflow.fit_offline(
        data=gbm_inference.training_data(splits["train"], task),
        validation_data=gbm_inference.training_data(splits["validation"], task),
        epochs=40, batch_size=128,
    )
    workflow.load_approximator()  # best validation checkpoint, not last epoch
    conditions = gbm_inference.conditioning_data(
        splits["test"]["observations"], splits["test"]["dt_years"], task
    )
    draws = workflow.sample(num_samples=4096, conditions=conditions, seed=195)
    restored = gbm_inference.build_workflow()
    restored.load_approximator(new_checkpoint_directory / "model.keras")
    ```

    The checkpoint preserves its adapter and normalization, learned globally
    from training batches. Retain the run's `report.json` as well: a raw model
    file alone does not establish its prior, known volatility or observation
    design. Compare seeded draws after independent reload and compare the
    approximate posterior with the analytic reference before claiming accuracy.
    """)
    return


@app.cell
def _(mo, pd):
    _historical = pd.DataFrame([
        {"Historical metric": "Mean error RMS / exact posterior SD", "Observed": "0.045", "Pilot maximum": "0.10"},
        {"Historical metric": "Maximum mean error / exact posterior SD", "Observed": "0.162", "Pilot maximum": "0.30"},
        {"Historical metric": "Median absolute relative SD error", "Observed": "2.0%", "Pilot maximum": "10%"},
        {"Historical metric": "Nominal 90% interval coverage: neural", "Observed": "84.4%", "Pilot maximum": "descriptive"},
        {"Historical metric": "Nominal 90% interval coverage: exact, same cases", "Observed": "86.3%", "Pilot maximum": "descriptive"},
        {"Historical metric": "Mean exact posterior mass in neural 90% intervals", "Observed": "89.1%", "Pilot maximum": "descriptive"},
    ])
    mo.vstack([
        mo.md("""
        ### Historical result for increment 2

        The following rounded values are transcribed from the committed
        [local bank/training guide](../docs/gbm-bank-training.md). They describe
        its original full 8,192-history training run, not the tiny bank above
        or a new training experiment. The original checkpoint/report is not
        bundled with this GBM notebook.
        """),
        mo.ui.table(_historical, selection=None, show_column_summaries=False),
        mo.md("""
        The conditional-moment pilot passed and same-seed reload draws matched.
        Its 256-case coverage result is not a full calibration certificate:
        even the exact posterior covered only 86.3% on that finite set. Nominal
        binomial standard error at 90% is about 1.9 percentage points. Do not
        recast the observed 84.4% as a general 90%-coverage guarantee.

        The recorded 19.6 MB bank took about 0.36 s to generate/write and 0.012 s
        to load. Training took 2.1 s and sampling 256 × 4,096 draws took 16.6 s,
        including the respective JAX compilation but excluding import startup.
        These are historical single-machine timings, not measurements performed
        by this notebook or a portable performance promise.
        """),
    ])
    return


@app.cell
def _(cluster_guide, mo):
    _measured = cluster_guide.split("## Measured run\n", 1)[1]
    mo.vstack([
        mo.md("""
        ## Increment 3: the same generation task ran on a real cluster

        This is **historical evidence from 11 October 2026**, read from the
        committed [CPU smoke guide](../docs/gbm-cluster-smoke.md). This notebook
        neither connects to Oscar nor checks current queues/permissions. The
        source revision and archive hash below identify the submitted code;
        they are not claimed to be the current working tree.

        The one-CPU job generated a bank, verified checksums and regenerated
        selected IDs. Transferred-file integrity and cross-platform numerical
        agreement were checked separately. This completes a small execution
        path, not neural training on Linux or a scaling study.
        """),
        mo.md(_measured),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exercises and next steps

    1. Keep the seed fixed and increase observed days from 252 to 504. Does
       posterior uncertainty shrink? Must its mean move toward the generating
       truth on every additional piece of history?
    2. Instead hold elapsed time at one year and make $\Delta$ smaller. Use the
       posterior formula to predict whether drift uncertainty changes.
    3. Find the observation, context and target arrays in the manifest. Which
       keys could a real return series supply, and which would be unknown?
    4. Why can nine examples match exactly on both Mac and Linux without proving
       universal cross-platform bitwise reproducibility?
    5. In the pricing table, compare a single discounted payoff, its sample mean
       and the analytic expectation. Which one should label a pricing surrogate?

    **Review hints:** longer elapsed time reduces posterior variance, but a
    random additional segment can move the mean either way. Finer sampling at
    fixed elapsed time does not reduce this drift variance. Truth is not part
    of observed conditioning. Selected-ID checks establish a bounded result for
    specified implementations. A reliable expected payoff, with numerical error
    understood, is the pricing label.

    Reproduce the local increments explicitly, choosing new output directories:

    ```bash
    pixi run gbm-demo --out /tmp/gbm-reference-new
    pixi run gbm-bank --out /tmp/gbm-bank-new
    pixi run gbm-train --bank /tmp/gbm-bank-new --out /tmp/gbm-trained-new
    pixi run python scripts/check_gbm_bank.py --bank /tmp/gbm-bank-new
    ```

    Cluster submission remains an explicit action using the
    [CPU smoke instructions](../docs/gbm-cluster-smoke.md), after verifying
    current access and resource settings. Nothing in the notebook submits work.

    Continue with the [GBM reference guide](../docs/gbm-reference.md),
    [bank/training guide](../docs/gbm-bank-training.md),
    [mathematical foundations](../docs/simulator-plan/notes/01-foundations.md),
    [simulator plan](../docs/simulator-plan/README.md), and
    [joint Heston inference notebook](05_heston_joint_inference.py).
    """)
    return


if __name__ == "__main__":
    app.run()
