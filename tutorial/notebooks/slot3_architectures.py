import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Slot 3 · Network architectures")


@app.cell(hide_code=True)
def _():
    import inspect
    import os
    import time

    os.environ.setdefault("KERAS_BACKEND", "jax")

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from sbi4f_tutorial import inference, plots, teaching

    ROOT = mo.notebook_dir().parent
    ASSETS = ROOT / "assets"
    RESULTS = ASSETS / "results" / "calibration"
    CHECKPOINTS = ASSETS / "checkpoints"
    plots.style()
    return (
        ASSETS,
        CHECKPOINTS,
        RESULTS,
        inference,
        inspect,
        mo,
        np,
        plots,
        plt,
        teaching,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Slot 3 · Network architectures for amortized inference

    **Which networks represent the posterior, and how they read the data**

    Slot 2 ended on the object to build: a conditional distribution $q_\phi(\theta \mid y)$, trained on
    simulated pairs. This notebook shows the networks that build it, using the trained Heston network of
    slot 4 on the SPY smile.

    Three claims, each shown below:

    1. **An inference network represents the posterior.** Flow matching learns a velocity field that
       carries Gaussian noise to the posterior along an ordinary differential equation.
    2. **A summary network, when needed, follows the shape of the data.** None for a fixed grid of quotes,
       a set network for a variable set of quotes, a time-series network for returns.
    3. **A few details decide whether a network works on financial data:** bounded parameters, scales,
       fresh noise in every batch, and the sampler's steps.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid("""
    flowchart LR
      Y["data y: 63 prices, or a set of quotes, or a return series"] --> H["summary network h (optional)"]
      H --> Q["inference network q(θ | h(y))"]
      Y -. "a fixed grid needs no summary" .-> Q
      Z["Gaussian noise z"] --> Q
      Q --> D["posterior draws of θ"]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Flow matching: noise carried to the posterior

    For a simulated pair $(\theta, y)$, Gaussian noise $z$ and a time $\tau \in [0, 1]$, the network
    $u_\phi$ is trained by least squares on straight paths from noise to parameter:

    $$\mathcal L(\phi) = \mathbb E\,\big\| u_\phi\big((1 - \tau)\,z + \tau\,\theta,\ \tau,\ y\big) - (\theta - z) \big\|^2 .$$

    Sampling integrates $dz/d\tau = u_\phi(z, \tau, y)$ from $\tau = 0$ to $1$. A normalizing flow does
    the same transport in one invertible map with an exact density; flow matching trades that for a
    regression loss that scales to many parameters and odd shapes, and the tutorial's networks use it.

    Below, the trained Heston network of slot 4 carries 2,000 noise points to SPY's posterior. Each panel
    stops the integration at a time $\tau$ and maps the points to parameter units.
    """)
    return


@app.cell
def _(CHECKPOINTS, RESULTS, inference, np, teaching):
    heston_task, heston_workflow = teaching.load_network(CHECKPOINTS / "heston")
    spy = teaching.load_json(RESULTS / "spy" / "surface.json")
    spy_conditions = inference.conditioning_data(np.asarray(spy["prices_forward_units"])[None], None, heston_task)
    _network = heston_workflow.approximator.inference_network
    _full = inference.sampler_times()
    snapshots = {}
    for _tau in (0.0, 0.25, 0.5, 0.75, 1.0):
        _times = _full[_full <= _tau + 1e-9] if _tau > 0 else np.array([0.0, 1e-4], dtype=np.float32)
        _network.integrate_kwargs = {**_network.integrate_kwargs, "method": "euler", "steps": _times}
        _draws = heston_workflow.sample(num_samples=2000, conditions=spy_conditions, seed=3)
        snapshots[_tau] = np.column_stack([np.asarray(_draws[n])[0, :, 0] for n in ("vol_of_vol", "rho")])
    inference.use_sampler(heston_workflow)  # back to the full 100-step sampler
    return heston_task, heston_workflow, snapshots, spy_conditions


@app.cell(hide_code=True)
def _(plots, plt, snapshots):
    _fig, _axes = plt.subplots(1, 5, figsize=(12.0, 2.9), sharex=True, sharey=True, constrained_layout=True)
    for _ax, (_tau, _xy) in zip(_axes, snapshots.items()):
        _ax.scatter(_xy[:, 0], _xy[:, 1], s=2, alpha=0.35, color=plots.SERIES[0])
        _ax.set_title(f"τ = {_tau:g}", loc="left")
        _ax.set_xlabel("vol-of-vol σ_v")
    _axes[0].set_ylabel("correlation ρ")
    _axes[0].set_xlim(0.1, 1.5)
    _axes[0].set_ylim(-0.95, -0.1)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At $\tau = 0$ the points fill the prior box; by $\tau = 1$ they sit on SPY's posterior,
    $\sigma_v \approx 1.13$ and $\rho \approx -0.64$. The path is not a straight line in parameter units,
    because the network works in standardized, unbounded coordinates.

    ## 2 · Reading the data

    | Data | Shape | Network | Where |
    |:---|:---|:---|:---|
    | A smile on the design grid | a fixed vector of 63 prices (and, in slot 5, the forward-variance curve as context) | no summary network: the standardized prices enter as conditions | slots 4 and 5 |
    | A real option chain | a set of (log-moneyness, maturity, price, noise) tuples, of varying size and order | a set network (DeepSet, set transformer), invariant to the order of the quotes | the extension that removes the design grid |
    | Daily returns | a series of 1,000 values | a time-series network: convolutions, then a recurrent or attention layer | not in the tutorial |
    | Any data, for misspecification tests | any | a summary network trained with a distance term in its output space (Schmitt et al. 2024) | slot 6 |

    In BayesFlow each is one line, and the rest of the workflow does not change:

    ```python
    bf.networks.DeepSet(summary_dim=16)                    # a set of quotes, any order, any number
    bf.networks.SetTransformer(summary_dim=16)             # the same, with attention between quotes
    bf.networks.TimeSeriesNetwork(summary_dim=32)          # a return series
    ```

    Where sufficient statistics exist, a learned summary rediscovers them.

    ## 3 · The configuration slot 4 trains

    The whole network of slots 4 and 5 is this function (`src/sbi4f_tutorial/inference.py`):
    """)
    return


@app.cell(hide_code=True)
def _(inference, inspect, mo):
    mo.ui.code_editor(inspect.getsource(inference.build_workflow), language="python", disabled=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each line answers a practical problem:

    1. **Bounded parameters.** `constrain` maps each parameter from its prior box to the real line, so
       draws never leave the box: $\rho$ stays in $(-0.95, -0.1)$.
    2. **Scales.** Parameters and prices are standardized with training statistics, because SPY's 63 prices
       range from about $1.5 \times 10^{-5}$ to $0.07$ of the forward.
    3. **Noise.** Quote noise is drawn fresh in every batch during training, so the network never sees the
       same noisy smile twice.
    4. **Capacity.** Four dense layers of 256 in the velocity network.
    5. **Training times** are drawn with a power law (`time_power_law_alpha`; −0.5 for the trained Heston
       network).
    6. **Sampler.** `inference.sample` integrates with 100 Euler steps on the grid $\tau_i = 1 - (1 - i/100)^2$,
       dense where the flow ends. The configuration stored with a network has 50 uniform steps.

    ## 4 · How many steps does the sampler need?

    Fewer steps are faster and less accurate. The exact Heston posterior on SPY (slot 4) is the reference.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    steps_choice = mo.ui.multiselect(options=["10", "25", "50", "100"], value=["10", "25", "100"],
                                     label="Euler steps on the non-uniform grid")
    steps_choice
    return (steps_choice,)


@app.cell
def _(
    ASSETS,
    heston_task,
    heston_workflow,
    inference,
    mo,
    np,
    spy_conditions,
    steps_choice,
    teaching,
    time,
):
    exact_spy = np.load(ASSETS / "data" / "heston-spy-exact.npy")
    _rows = []
    _network = heston_workflow.approximator.inference_network
    for _steps in sorted(int(s) for s in steps_choice.value):
        _network.integrate_kwargs = {**_network.integrate_kwargs, "method": "euler", "steps": inference.sampler_times(_steps)}
        _start = time.perf_counter()
        _draws = heston_workflow.sample(num_samples=2000, conditions=spy_conditions, seed=5)
        _seconds = time.perf_counter() - _start
        _row = {"steps": _steps, "seconds for 2,000 draws": round(_seconds, 2)}
        for _j, _name in enumerate(heston_task["parameter_names"]):
            _x = np.asarray(_draws[_name])[0, :, 0]
            _row[f"{teaching.LABELS[_name]}: width / exact"] = round(float(_x.std() / exact_spy[:, _j].std()), 2)
        _rows.append(_row)
    inference.use_sampler(heston_workflow)
    mo.ui.table(_rows, selection=None, pagination=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A width ratio of 1 means the network's posterior is as wide as the exact one. Too few steps shrink
    sharp posteriors.

    ## Takeaways

    1. **Flow matching carries noise to the posterior**; one trained network does it for any smile.
    2. **The summary network follows the data's shape.** The design grid needs none; real chains need a
       set network; returns need a time-series network.
    3. **Bounds, scales, fresh noise and the sampler's grid** are what make the same network work on
       option prices.

    **Next, slot 4:** the whole workflow in BayesFlow, on Heston and the SPY smile, checked against the
    exact posterior.

    *Evidence and code:* the [curriculum](../curriculum/03-slot3-architectures.md),
    `src/sbi4f_tutorial/inference.py` and the Heston checkpoint in `assets/checkpoints/heston/`.
    """)
    return


if __name__ == "__main__":
    app.run()
