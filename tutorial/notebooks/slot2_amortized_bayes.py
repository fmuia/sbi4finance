import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Slot 2 · What is amortized Bayes?")


@app.cell(hide_code=True)
def _():
    import json
    import os
    import time

    os.environ.setdefault("KERAS_BACKEND", "jax")

    import bayesflow as bf
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from sbi4f_tutorial import inference, plots, surface, teaching

    ROOT = mo.notebook_dir().parent
    ASSETS = ROOT / "assets"
    RESULTS = ASSETS / "results" / "calibration"
    TOY_CHECKPOINT = ASSETS / "checkpoints" / "slot2-toy" / "model.keras"
    plots.style()
    return (
        ASSETS,
        RESULTS,
        TOY_CHECKPOINT,
        bf,
        inference,
        mo,
        np,
        plots,
        plt,
        surface,
        teaching,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Slot 2 · What is amortized Bayes?

    **A network that learns the posterior from simulations, once, for every data set**

    Slot 1 ended on a cost. The posterior needs the likelihood evaluated thousands of times for every
    smile, and for the models of slots 4 and 5 each evaluation is a Fourier integral or a Monte Carlo run. This
    notebook shows the idea that removes most of that cost for a validation, on slot 1's own example, where the exact answer is
    known.

    Three claims, each backed by a computation below:

    1. **A network trained on simulated pairs returns the posterior.** It never evaluates the likelihood;
       it only sees parameters and the data they produce.
    2. **The cost moves from each data set to one training run.** That pays when many posteriors are
       needed: a valuation across a book, for example at month-end, or the thousands that validation needs.
    3. **The network is an approximation, so the workflow checks it.** Simulation-based calibration (SBC)
       checks it on average; an exact posterior, where one exists, or an audit checks it data set by data set.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | | Infer data set by data set | Amortized inference |
    |:---|:---|:---|
    | Methods | MCMC, sequential Monte Carlo, ABC, particle MCMC | neural posterior estimation, here with flow matching |
    | Needs | the likelihood, or a simulator per proposal | a simulator, once, for the training bank |
    | Cost of the first posterior | one full run | training: the bank plus the network |
    | Cost of each further posterior | one full run | one pass through the network |
    | Validation by SBC (thousands of posteriors) | thousands of full runs | thousands of passes |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "**Why the trained network is the posterior** (click to open)": mo.md(r"""
        Draw pairs from the joint distribution $p(\theta, y) = p(\theta)\,p(y \mid \theta)$: a parameter
        from the prior, then data from the model. Train a conditional density $q_\phi(\theta \mid y)$ to
        put probability on the true parameter of each pair. Its expected loss splits in two:

        $$\mathbb E_{p(\theta, y)}\big[-\log q_\phi(\theta \mid y)\big]
        = \mathbb E_{p(y)}\Big[\mathrm{KL}\big(p(\theta \mid y)\,\big\|\,q_\phi(\theta \mid y)\big)\Big]
        + \mathbb E_{p(y)}\big[\mathcal H\big(p(\theta \mid y)\big)\big].$$

        The second term does not depend on $\phi$. The first is zero only when $q_\phi$ equals the
        posterior, for every $y$ the prior and the model can produce. So the minimizer is the posterior,
        and it is learned for all data sets at once. Flow matching (slot 3) reaches the same target with
        a regression loss on a velocity field.

        Three consequences:

        1. **The prior and the simulator are trained in.** Changing either needs retraining or a
           correction (slot 6).
        2. **The network is accurate only where the prior puts data.** A real data set unlike any
           simulation is an extrapolation; the prior predictive check detects it.
        3. **The minimum is reached only approximately.** The gap between the network and the posterior
           for one data set is the amortization gap. Where the likelihood can be computed or estimated,
           importance sampling with the network as proposal corrects it (slots 4 and 5).
        """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The toy: slot 1's posterior, learned

    Slot 1 computed the posterior of the Black–Scholes volatility from one quote, the at-the-money
    one-month SPY price, on a grid. Here a network learns the same posterior from simulations. The prior
    is uniform on 8% to 40%, and each simulated quote is the Black–Scholes price plus the quote noise of
    slot 1.
    """)
    return


@app.cell
def _(RESULTS, bf, np, surface, teaching):
    spy = teaching.load_json(RESULTS / "spy" / "surface.json")
    _k, _maturity, _kind, _delta = teaching.design_arrays()
    _i = int(np.flatnonzero((_maturity * surface.TRADING_DAYS == 21) & (np.abs(_delta) < 1e-9))[0])
    QUOTE = {"k": float(_k[_i]), "T": float(_maturity[_i]), "kind": str(_kind[_i]),
             "price": float(spy["prices_forward_units"][_i]), "sd": float(spy["noise_sd"][_i])}
    PRIOR_LOW, PRIOR_HIGH = 0.08, 0.40
    _rng = np.random.default_rng(2026)


    def prior():
        return {"sigma": _rng.uniform(PRIOR_LOW, PRIOR_HIGH)}


    def quote(sigma):
        return {"price": surface.bs_price(QUOTE["k"], QUOTE["T"], sigma, QUOTE["kind"]) + QUOTE["sd"] * _rng.standard_normal()}


    simulator = bf.make_simulator([prior, quote])
    return PRIOR_HIGH, PRIOR_LOW, QUOTE, simulator


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    BayesFlow needs only these two functions that **draw**: `prior()` returns one parameter, and
    `quote(sigma)` one noisy price for it. `bf.make_simulator` chains them. A bank of 40,000 simulated
    pairs takes about a second:
    """)
    return


@app.cell
def _(mo, simulator, time):
    _start = time.perf_counter()
    bank = simulator.sample(40_000)
    bank_seconds = time.perf_counter() - _start
    mo.md(f"40,000 pairs of (σ, price) in **{bank_seconds:.1f} s**.")
    return (bank,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The network is BayesFlow's flow matching, at a toy size. The adapter maps $\sigma$ from its prior
    interval to the real line and names what the network infers and what it conditions on. Sampling uses
    100 Euler steps on a grid that is dense where the flow ends (slot 3).
    """)
    return


@app.cell
def _(PRIOR_HIGH, PRIOR_LOW, bf, inference):
    adapter = (
        bf.Adapter()
        .constrain("sigma", lower=PRIOR_LOW, upper=PRIOR_HIGH)
        .convert_dtype("float64", "float32")
        .rename("sigma", "inference_variables")
        .rename("price", "inference_conditions")
    )


    def build_toy_workflow():
        network = bf.networks.FlowMatching(subnet_kwargs={"widths": [128, 128, 128]},
                                           integrate_kwargs={"method": "euler", "steps": inference.sampler_times(100)})
        return bf.BasicWorkflow(adapter=adapter, inference_network=network,
                                standardize=["inference_variables", "inference_conditions"])

    return (build_toy_workflow,)


@app.cell(hide_code=True)
def _(mo):
    train_button = mo.ui.run_button(label="Train the toy network live (about 30 s on 4 cores)")
    train_button
    return (train_button,)


@app.cell
def _(TOY_CHECKPOINT, bank, build_toy_workflow, mo, time, train_button):
    _start = time.perf_counter()
    toy = build_toy_workflow()
    if train_button.value or not TOY_CHECKPOINT.exists():
        toy.fit_offline(data=bank, epochs=30, batch_size=256, verbose=0)
        if not TOY_CHECKPOINT.exists():  # instructors create the shipped network once; a live run never replaces it
            TOY_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            toy.approximator.save(TOY_CHECKPOINT)
        toy_source = "trained live"
    else:
        toy.fit_offline(data={key: value[:256] for key, value in bank.items()}, epochs=1, batch_size=256, verbose=0)
        toy.load_approximator(TOY_CHECKPOINT)
        toy_source = "loaded from the checkpoint trained before the session"
    toy_seconds = time.perf_counter() - _start
    mo.md(f"The toy network was **{toy_source}** in {toy_seconds:.0f} s.")
    return (toy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · The network against the exact posterior

    The grid posterior is slot 1's computation with this prior. The network's posterior comes from one
    call with the SPY quote as condition.
    """)
    return


@app.cell
def _(PRIOR_HIGH, PRIOR_LOW, QUOTE, np, surface, time, toy):
    sigma_grid = np.linspace(PRIOR_LOW, PRIOR_HIGH, 4001)
    _miss = (QUOTE["price"] - surface.bs_price(QUOTE["k"], QUOTE["T"], sigma_grid, QUOTE["kind"])) / QUOTE["sd"]
    _weights = np.exp(-0.5 * (_miss**2 - (_miss**2).min()))
    exact_density = _weights / np.trapezoid(_weights, sigma_grid)
    exact_mean = np.trapezoid(sigma_grid * exact_density, sigma_grid)
    exact_sd = np.sqrt(np.trapezoid((sigma_grid - exact_mean) ** 2 * exact_density, sigma_grid))
    _start = time.perf_counter()
    network_draws = np.asarray(toy.sample(num_samples=4000, conditions={"price": np.array([[QUOTE["price"]]])})["sigma"])[0, :, 0]
    network_seconds = time.perf_counter() - _start
    return (
        exact_density,
        exact_mean,
        exact_sd,
        network_draws,
        network_seconds,
        sigma_grid,
    )


@app.cell(hide_code=True)
def _(
    exact_density,
    exact_mean,
    exact_sd,
    mo,
    network_draws,
    network_seconds,
    plots,
    plt,
    sigma_grid,
):
    _fig, _ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    _ax.hist(100 * network_draws, bins=60, density=True, color=plots.SERIES[0], alpha=0.6, label="network, 4,000 draws")
    _ax.plot(100 * sigma_grid, exact_density / 100, color=plots.INK, label="exact posterior (grid)")
    _ax.set_xlim(100 * (exact_mean - 5 * exact_sd), 100 * (exact_mean + 5 * exact_sd))
    _ax.set_yticks([])
    _ax.set_xlabel("Black–Scholes volatility σ (%)")
    _ax.legend(fontsize=8)
    _ax.set_title("the SPY at-the-money one-month quote", loc="left")
    mo.vstack([
        _fig,
        mo.md(f"""
        | | mean | standard deviation |
        |:---|---:|---:|
        | exact posterior (grid) | {100 * exact_mean:.2f}% | {100 * exact_sd:.2f} points |
        | network | {100 * network_draws.mean():.2f}% | {100 * network_draws.std():.2f} points |

        4,000 draws in {network_seconds:.1f} s, including the first call's compilation.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Amortization: many data sets, one call

    The same network answers any quote the prior can produce. Here it returns the posteriors of 500
    different quotes in one call, against the exact grid posterior of each.
    """)
    return


@app.cell
def _(QUOTE, np, sigma_grid, simulator, surface, time, toy):
    _quotes = simulator.sample(500)
    _start = time.perf_counter()
    _draws = np.asarray(toy.sample(num_samples=100, conditions={"price": _quotes["price"]})["sigma"])[:, :, 0]
    many_seconds = time.perf_counter() - _start
    many = {"truth": _quotes["sigma"][:, 0], "price": _quotes["price"][:, 0], "draws": _draws}
    _misses = (many["price"][:, None] - surface.bs_price(QUOTE["k"], QUOTE["T"], sigma_grid[None, :], QUOTE["kind"])) / QUOTE["sd"]
    _w = np.exp(-0.5 * (_misses**2 - (_misses**2).min(axis=1, keepdims=True)))
    many_exact_mean = (_w * sigma_grid).sum(axis=1) / _w.sum(axis=1)
    return many, many_exact_mean, many_seconds


@app.cell(hide_code=True)
def _(
    PRIOR_HIGH,
    PRIOR_LOW,
    many,
    many_exact_mean,
    many_seconds,
    mo,
    plots,
    plt,
):
    _fig, _ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    _ax.scatter(100 * many_exact_mean, 100 * many["draws"].mean(axis=1), s=8, color=plots.SERIES[0])
    _lims = [100 * PRIOR_LOW, 100 * PRIOR_HIGH]
    _ax.plot(_lims, _lims, color=plots.BASELINE, linestyle="--")
    _ax.set_xlabel("exact posterior mean of σ (%)")
    _ax.set_ylabel("network posterior mean of σ (%)")
    _ax.set_title("500 simulated quotes, one call", loc="left")
    mo.vstack([_fig, mo.md(f"500 posteriors of 100 draws each in **{many_seconds:.1f} s**: "
                           f"{1000 * many_seconds / 500:.0f} ms per data set.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · When it pays

    Training is paid once. Each further posterior costs one pass. The break-even is the training cost
    divided by the cost of one exact posterior. The numbers are the tutorial's, on one 4-core machine.
    """)
    return


@app.cell
def _(ASSETS, RESULTS, np, teaching):
    costs = teaching.load_json(RESULTS / "costs" / "costs.json")
    _bank = teaching.load_json(RESULTS / "heston" / "s1" / "bank-quality.json")["generation"]["cpu_seconds"]
    _training = teaching.load_json(ASSETS / "checkpoints" / "heston" / "training.json")["timing_seconds"]["training_including_compile"]
    _heston_exact = costs["exact_heston_posterior"]["cpu_seconds"]
    _heston_net = costs["heston_amortized_posterior_1000_draws_cpu_s"]["median"]
    _rough_exact = 3600 * costs["exact_rough_posterior"]["cpu_hours"]
    _rough_net = costs["rough_amortized_posterior_1000_draws_cpu_s"]["median"]
    _rough_bank = 112_000 * 11.9  # the pilot bank's average pricing cost per surface (study S9)
    smiles = np.arange(0, 201)
    cumulative = {
        "Heston, exact": smiles * _heston_exact,
        "Heston, amortized (bank and 32 minutes of training on 4 cores)": _bank + 4 * _training + smiles * _heston_net,
        "rough Bergomi, exact by random-walk Metropolis (lower bound)": smiles * _rough_exact,
        "rough Bergomi, amortized (bank at the pilot's rate, training not counted)": _rough_bank + smiles * _rough_net,
    }
    break_even = {"Heston": (_bank + 4 * _training) / (_heston_exact - _heston_net),
                  "rough Bergomi": _rough_bank / (_rough_exact - _rough_net)}
    heston_unit_costs = {"exact": _heston_exact, "network": _heston_net}
    return break_even, cumulative, heston_unit_costs, smiles


@app.cell(hide_code=True)
def _(break_even, cumulative, heston_unit_costs, mo, plots, plt, smiles):
    _fig, _ax = plt.subplots(figsize=(8.4, 3.8), constrained_layout=True)
    for (_name, _cost), _color, _style in zip(cumulative.items(), [plots.SERIES[0]] * 2 + [plots.SERIES[1]] * 2,
                                               ["-", "--", "-", "--"]):
        _ax.plot(smiles, _cost / 3600, color=_color, linestyle=_style, label=_name)
    _ax.set_yscale("log")
    _ax.set_xlabel("number of smiles")
    _ax.set_ylabel("cumulative CPU-hours")
    _ax.legend(fontsize=7.5, loc="lower right")
    _ax.set_title("exact against amortized, cumulative cost", loc="left")
    mo.vstack([_fig, mo.md(f"""
    Amortization pays after about **{break_even['Heston']:.0f} Heston smiles** (training counted on 4 cores)
    and about **{break_even['rough Bergomi']:.0f} rough-Bergomi smiles** (against the 17 CPU-hour lower bound of
    random-walk Metropolis; training not counted yet). By importance sampling from a Gauss–Newton proposal, the
    rough-Bergomi posterior of a fixed-seed pricer takes about 85 CPU-minutes on SPY's 63 quotes (1,024 evaluations at 5.0
    CPU-seconds); against that, the training bank pays after about 260 smiles. Where that proposal works, one smile and a book are affordable; a validation on thousands of
    simulated smiles is not. Validation alone needs thousands: SBC on 2,000 smiles is 2,000 posteriors,
    about {2000 * heston_unit_costs['exact'] / 3600:.0f} CPU-hours of exact Heston chains against
    {2000 * heston_unit_costs['network'] / 3600:.0f} for the network.
    """)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · The workflow, and SBC

    The amortized workflow is the checklist that slots 4 and 5 follow:

    1. State the prior in units a practitioner can judge, and draw data from it: the prior predictive check.
    2. Simulate the training bank, and train the network.
    3. Check it on held-out simulations: SBC, coverage, contraction, recovery.
    4. Where an exact posterior exists, compare with it (slot 4).
    5. Apply it to the real data, and re-price the posterior over the data: the posterior predictive check.
    6. Where the likelihood can be estimated, correct the posterior of the real data set by importance
       sampling, and flag it if the correction is unreliable (slot 5).
    7. Propagate the posterior into the quantity of interest.

    **SBC** follows from Bayes' rule. Draw $\theta^\ast$ from the prior and data from the model. Given the
    data, $\theta^\ast$ is itself a draw from the posterior, so its rank among $M$ posterior draws is
    uniform. A network that is biased or too narrow breaks the uniformity, and the shape says how. Here are
    the 500 posteriors of section 3:
    """)
    return


@app.cell(hide_code=True)
def _(many, mo, np, plots, plt):
    _ranks = (many["draws"] < many["truth"][:, None]).mean(axis=1)
    _grid = np.linspace(0, 1, 201)
    _ecdf = (np.sort(_ranks)[None, :] <= _grid[:, None]).mean(axis=1)
    _band = 1.36 / np.sqrt(len(_ranks))  # Kolmogorov-Smirnov 95% band for the whole curve
    _fig, _ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
    _ax.fill_between(_grid, -_band, _band, color=plots.BASELINE, alpha=0.4, label="95% band")
    _ax.plot(_grid, _ecdf - _grid, color=plots.SERIES[0], label="network")
    _ax.axhline(0, color=plots.INK, linewidth=0.8)
    _ax.set_xlabel("rank of the true σ among the posterior draws")
    _ax.set_ylabel("ECDF minus uniform")
    _ax.legend(fontsize=8)
    _ax.set_title("simulation-based calibration, 500 simulated quotes", loc="left")
    mo.vstack([_fig, mo.md(f"Largest deviation **{np.max(np.abs(_ecdf - _grid)):.3f}** against a 95% band of "
                           f"±{_band:.3f}: the toy network is calibrated on average.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Where it comes from

    - **Simulation-based inference:** approximate Bayesian computation (Pritchard et al. 1999; Beaumont,
      Zhang and Balding 2002), MCMC without likelihoods (Marjoram et al. 2003), synthetic likelihood (Wood
      2010), neural posterior estimation (Papamakarios and Murray 2016; Greenberg, Nonnenmacher and Macke
      2019), BayesFlow (Radev et al. 2022), flow matching for simulation-based inference (Wildberger et al.
      2023).
    - **What finance uses instead:** closed-form likelihood expansions (Aït-Sahalia 2002), indirect
      inference and the method of moments (Gouriéroux, Monfort and Renault 1993), particle MCMC (Andrieu,
      Doucet and Holenstein 2010). Each infers one data set at a time.

    ## Takeaways

    1. **The network learns the posterior from simulations alone**, and on slot 1's quote it matches the
       exact posterior.
    2. **One call answers many data sets.** The cost of inference moves into training.
    3. **Amortization pays after a few tens of smiles against a random-walk chain, and a few hundred against
       importance sampling**, and it is what makes validation by SBC affordable.
    4. **The network is checked, not trusted**: SBC on average, an exact posterior or an audit one data set
       at a time.

    **Next, slot 3:** which networks represent $q_\phi(\theta \mid y)$, and how they read the data.

    *Evidence and code:* the [curriculum](../curriculum/02-slot2-amortized-bayes.md),
    `costs/costs.json` (study S9) and `src/sbi4f_tutorial/`.
    """)
    return


if __name__ == "__main__":
    app.run()
