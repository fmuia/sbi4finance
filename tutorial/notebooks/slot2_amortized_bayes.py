import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Slot 2 · What is amortized Bayes?")


@app.cell(hide_code=True)
def _():
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import bayesflow as bf

    from sbi4f_tutorial import plots, surface, teaching

    ROOT = mo.notebook_dir().parent
    ASSETS = ROOT / "assets"
    RESULTS = ASSETS / "results" / "calibration"
    plots.style()
    return ASSETS, RESULTS, bf, mo, np, plots, plt, surface, teaching, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What is amortized Bayes?

    **A neural network that learns the posterior from simulations, once, for every data set**

    Slot 1 ended on a cost. The posterior is our ultimate object of interest. However, its estimation needs the likelihood evaluated thousands of times for every smile, and for the models of slots 4 and 5 each evaluation is a Fourier integral or a Monte Carlo run. This notebook shows the idea that removes most of that cost on a simple example.

    The slot in brief:

    1. **We will train a neural network on simulated pairs to estimate the posterior.** The network will never evaluate the likelihood; it only sees parameters and the data they produce.
    2. **The cost moves from each data set to one training run.** This type of inference amortization pays off when many posteriors are needed: a valuation across a book, for example at month-end or a screener that needs to processed thousands of assets.
    3. **The network is an approximation, so we need diagnostics.** We will discuss simulation-based calibration (SBC), a way to detect systematic biases in the approximate posterior before deployment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | | Infer data set by data set | Amortized inference |
    |:---|:---|:---|
    | Methods | MCMC, SMC, ABC | neural posterior estimation, here with a coupling flow |
    | Needs | the likelihood, or a simulator per proposal | a simulator, once, for the training bank |
    | Cost of the first posterior | one full run | training: the bank plus the network |
    | Cost of each further posterior | one full run | one pass through the network |
    | Validation by SBC (thousands of posteriors) | thousands of full runs | thousands of passes |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "**How does the network learn the posterior?** (click to open)": mo.md(r"""
    During training, wraw pairs from the joint distribution $p(\theta, y) = p(\theta)\,p(y \mid \theta)$: a parameter from the prior, then data from the model. We then train a conditional density $q_\phi(\theta \mid y)$ to put probability on the true parameter of each pair. Its expected loss splits in two:

    $$
    \begin{equation}
    \mathbb{E}_{p(y)}\left[\mathrm{KL}\left(p(\theta \mid y)\|q_\phi(\theta \mid y)\right)\right] = \mathbb{E}_{p(\theta,y)}\left[-\log q_\phi(\theta \mid y)\right] - \mathbb{E}_{p(y)}\left[\mathcal{H}(p(\theta \mid y))\right].
    \end{equation}
    $$

    The second term does not depend on $\phi$. The first is zero only when $q_\phi$ equals the posterior for $p(y)$-almost every $y$. So the minimizer is the posterior, and it is learned for all data sets at once. The one-layer coupling flow below learns this conditional density by maximum likelihood.

    But how does the network learn $\log q_\phi(\theta \mid y)$?

    The network is a special kind of generative model that learns an **invertible mapping** between parameters $\theta$ and noise $z$ from a simple base distribution, typically a standard Gaussian. Conditioning on $y$ makes this mapping depend on the observed data.

    There are two main approaches:

    **1. Normalizing flows (single-step transformations)**

    A normalizing flow learns an invertible map $z = f_\phi(\theta; y)$. The change-of-variables formula gives the exact conditional log density:

    $$
    \log q_\phi(\theta \mid y) = \log p_Z(f_\phi(\theta; y)) + \log \left|\det \frac{\partial f_\phi(\theta; y)}{\partial \theta}\right|.
    $$

    Both terms can be evaluated directly in a single forward pass, as the normalizing flow is designed to have a cheap Jacobian determinant and the base distribution $p(z)$ is Gaussian. Thus, *the negative log density is the training loss*.

    **2. Continuous normalizing flows (ODE transformations)**

    Instead of a finite sequence of transformations, continuous flows transform noise into parameters through an ODE with a learned velocity field $v_\phi(x,t;y)$. The instantaneous change-of-variables formula gives a similiar expression:

    $$
    \log q_\phi(\theta \mid y) = \log p_Z(z) - \int_0^1 \nabla_x \cdot v_\phi(x_t,t;y)\,dt,
    $$

    where $x_0=z$, $x_1=\theta$, and $\nabla_x \cdot v_\phi$ is the divergence of the velocity field. Evaluating this density requires solving an ODE and integrating the divergence, which is computationally expensive. Thus, *flow matching and diffusion models typically use different surrogate training losses* instead of directly maximizing the log posterior.

    **Three consequences:**

    1. **The prior and the simulator are trained in.** Changing either needs retraining or a correction.
    2. **The network is accurate only where the prior puts data.** A real data set unlike any simulation is an extrapolation; a prior predictive check can flag it.
    3. **The minimum is reached only approximately.** The gap between the network and the posterior for one data set is the amortization gap. Where the likelihood can be computed, importance sampling with the network as a proposal can correct it.
    """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Learning a simple posterior

    Slot 1 computed the posterior of the Black–Scholes volatility from one quote, the at-the-money
    one-month SPY price, on a grid. Here a network learns the same posterior from simulations. The prior
    is uniform on 8% to 40%, and each simulated quote is the Black–Scholes price plus the quote noise of
    slot 1.
    """)
    return


@app.cell(hide_code=True)
def _(RESULTS, np, surface, teaching):
    spy = teaching.load_json(RESULTS / "spy" / "surface.json")
    _k, _maturity, _kind, _delta = teaching.design_arrays()
    _i = int(np.flatnonzero((_maturity * surface.TRADING_DAYS == 21) & (np.abs(_delta) < 1e-9))[0])
    QUOTE = {"k": float(_k[_i]), "T": float(_maturity[_i]), "kind": str(_kind[_i]),
             "price": float(spy["prices_forward_units"][_i]), "sd": float(spy["noise_sd"][_i])}
    PRIOR_LOW, PRIOR_HIGH = 0.08, 0.40
    rng = np.random.default_rng(2026)
    return PRIOR_HIGH, PRIOR_LOW, QUOTE, rng


@app.cell
def _(PRIOR_HIGH, PRIOR_LOW, QUOTE, rng, surface):
    def prior():
        return {"sigma": rng.uniform(PRIOR_LOW, PRIOR_HIGH)}


    def quote(sigma):
        price = surface.bs_price(QUOTE["k"], QUOTE["T"], sigma, QUOTE["kind"])
        price += QUOTE["sd"] * rng.standard_normal()
        return {"price": price}

    return prior, quote


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    BayesFlow needs only functions that can simulate parameter and data. `prior()` returns one parameter, and
    `quote(sigma)` one noisy price for it.
    """)
    return


@app.cell(hide_code=True)
def _(bf, prior, quote):
    simulator = bf.make_simulator([prior, quote])
    return (simulator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The code below trains a small flow matching network from scratch when you press the button below it. The `Adapter` tells the network which quantities are considered observed and which latent. It can also be used to preprocess simulations before passing them to the network (e.g., to enforce bounds).
    """)
    return


@app.cell(hide_code=True)
def _(PRIOR_HIGH, PRIOR_LOW, bf, mo, simulator, time, train_button):
    mo.stop(not train_button.value, mo.md("Data generation and training are off. Press the training button to run a new training."))

    adapter = (
        bf.Adapter()
            .constrain("sigma", lower=PRIOR_LOW, upper=PRIOR_HIGH)
            .rename("sigma", "inference_variables")
            .rename("price", "inference_conditions")
    )

    workflow = bf.BasicWorkflow(
        adapter=adapter,
        initial_learning_rate=1e-3,
        inference_network=bf.networks.FlowMatching(subnet_kwargs={"widths": (32,)*3}),
        standardize="all"
    )

    _start = time.perf_counter()
    train_data = simulator.sample(40_000)
    bank_seconds = time.perf_counter() - _start

    _start = time.perf_counter()
    history = workflow.fit_offline(data=train_data, epochs=40, batch_size=256, verbose=0)
    toy_seconds = time.perf_counter() - _start

    _fig = bf.diagnostics.loss(history)

    mo.vstack([
        mo.md(f"40,000 simulated pairs generated in {bank_seconds:.1f}s."),
        mo.md(f"New BayesFlow 2.1 toy trained explicitly in {toy_seconds:.0f}s."),
        _fig
    ])
    return (workflow,)


@app.cell(hide_code=True)
def _(mo):
    train_button = mo.ui.run_button(label="Train a new BayesFlow 2.1 toy explicitly (40 epochs)")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 The network against the exact posterior

    The grid posterior is slot 1's computation with this prior. The network's posterior comes from one
    call with the SPY quote as condition.
    """)
    return


@app.cell(hide_code=True)
def _(PRIOR_HIGH, PRIOR_LOW, QUOTE, mo, np, plt, surface):
    sigma_grid = np.linspace(PRIOR_LOW, PRIOR_HIGH, 4001)
    _miss = (QUOTE["price"] - surface.bs_price(QUOTE["k"], QUOTE["T"], sigma_grid, QUOTE["kind"])) / QUOTE["sd"]
    _weights = np.exp(-0.5 * (_miss**2 - (_miss**2).min()))
    exact_density = _weights / np.trapezoid(_weights, sigma_grid)
    exact_mean = np.trapezoid(sigma_grid * exact_density, sigma_grid)
    exact_sd = np.sqrt(np.trapezoid((sigma_grid - exact_mean) ** 2 * exact_density, sigma_grid))
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(100 * sigma_grid, exact_density / 100)
    _ax.set_xlabel("Black–Scholes volatility (%)")
    _ax.set_ylabel("exact posterior density")
    mo.vstack([mo.md("**Exact toy posterior**, available before any optional network training."), _fig])
    return exact_density, exact_mean, exact_sd, sigma_grid


@app.cell(hide_code=True)
def _(QUOTE, np, time, workflow):
    new_price = np.array([[QUOTE["price"]]])

    _start = time.perf_counter()

    network_draws = workflow.sample(
        num_samples=2000,
        conditions={"price": new_price}
    )

    network_seconds = time.perf_counter() - _start
    return network_draws, network_seconds


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
    draws_flat = network_draws["sigma"].squeeze()

    _fig, _ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    _ax.hist(100 * draws_flat, bins=60, density=True, color=plots.SERIES[0], alpha=0.6, label="network, 2,000 draws")
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
        | network | {100 * draws_flat.mean():.2f}% | {100 * draws_flat.std():.2f} points |

        2,000 draws in {network_seconds:.1f} s, including the first call's compilation.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 Amortization: many data sets, one call

    The same network answers any quote the prior can produce. Here it returns the posteriors of 300 different quotes in one call, against the exact grid posterior of each.
    """)
    return


@app.cell(hide_code=True)
def _(QUOTE, np, sigma_grid, simulator, surface, time, workflow):
    simulations = simulator.sample(300)

    _start = time.perf_counter()
    posterior_draws = workflow.sample(
        num_samples=200, 
        conditions={"price": simulations["price"]}
    )
    many_seconds = time.perf_counter() - _start

    _misses = (simulations["price"] - surface.bs_price(QUOTE["k"], QUOTE["T"], sigma_grid[None, :], QUOTE["kind"])) / QUOTE["sd"]
    _w = np.exp(-0.5 * (_misses**2 - (_misses**2).min(axis=1, keepdims=True)))
    many_exact_mean = (_w * sigma_grid).sum(axis=1) / _w.sum(axis=1)
    return many_seconds, posterior_draws, simulations


@app.cell(hide_code=True)
def _(bf, many_seconds, mo, plots, posterior_draws, simulations):
    _fig = bf.diagnostics.recovery(
        estimates=posterior_draws,
        targets=simulations,
        color=plots.SERIES[0],
        variable_names=[r"Black–Scholes vVolatility $\sigma$"],
    )

    mo.vstack([_fig, mo.md(f"300 posteriors of 200 draws each in **{many_seconds:.1f} s**: "
                           f"{1000 * many_seconds / 500:.0f} ms per data set.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 To amortize or not to amortize?

    Training is paid once. Each further posterior costs one pass. The break-even is the training cost
    divided by the cost of one exact posterior. The numbers will be hardware-dependent.
    """)
    return


@app.cell(hide_code=True)
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
    ## 5 The workflow and simulation-based calibration (SBC)

    The amortized workflow is the checklist that slots 4 and 5 follow:

    1. State the prior in units a practitioner can judge, and draw data from it: the prior predictive check.
    2. Simulate the training data and train the network.
    3. Check it on held-out simulations: SBC, recovery, *etc*.
    4. Where an exact posterior exists, compare with it.
    5. Apply it to the real data, and re-price the posterior over the data: the posterior predictive check.
    6. Where the likelihood can be estimated, correct the posterior of the real data set by importance
       sampling.
    7. Propagate the posterior into the quantity of interest.

    **SBC** follows from Bayes' rule. Draw $\theta^\ast$ from the prior and data from the model. Given the
    data, $\theta^\ast$ is itself a draw from the posterior, so its rank among $M$ posterior draws is
    uniform. A network that is biased or too narrow breaks the uniformity, and the shape says how. Here are
    the 500 posteriors of section 3:
    """)
    return


@app.cell
def _(bf, plots, posterior_draws, simulations):
    calibration_figure = bf.diagnostics.calibration_ecdf(
        estimates=posterior_draws,
        targets=simulations,
        variable_keys=["sigma"],
        variable_names=[r"Black–Scholes Volatility $\sigma$"],
        difference=True,
        legend_location="upper right",
        rank_ecdf_color=plots.SERIES[0],
        fill_color=plots.BASELINE,
    )
    calibration_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Takeaways

    1. **The network learns the posterior from simulations alone**, and on slot 1's quote it matches the
       exact posterior.
    2. **One call answers many data sets.** The cost of inference moves into training.
    3. **Amortization pays after a few tens of smiles against a random-walk chain, and a few hundred against
       importance sampling**, and it is what makes validation by SBC affordable.
    4. **The network is checked, not trusted**: SBC on average, an exact posterior or an audit one data set
       at a time.

    **Next, slot 3:** which networks can represent $q_\phi(\theta \mid y)$, and how they process the data.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
