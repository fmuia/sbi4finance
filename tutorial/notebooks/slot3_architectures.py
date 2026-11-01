import marimo

__generated_with = "0.25.1"
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

    from sbi4f_tutorial import full_surface, inference, plots, teaching

    ROOT = mo.notebook_dir().parent
    ASSETS = ROOT / "assets"
    RESULTS = ASSETS / "results" / "calibration"
    CHECKPOINTS = ASSETS / "checkpoints"
    plots.style()
    return full_surface, inspect, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Slot 3 · Network architectures for amortized inference

    **Which networks represent the posterior, and how they read the data**

    Slot 2 ended on the object to build: a conditional distribution $q_\phi(\theta \mid y)$, trained on
    simulated pairs. This notebook shows the networks that build it, using the current cached Heston
    development example and its PyMC reference. Historical SPY results remain labelled separately in slot 4.

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

    This formula uses a conceptual noise-to-parameter time. BayesFlow retains its released
    time convention and sampler. In this conceptual convention, sampling integrates
    $dz/d\tau = u_\phi(z, \tau, y)$ from $\tau = 0$ to $1$. A normalizing flow does
    the same transport in one invertible map with an exact density; flow matching trades that for a
    regression loss that scales to many parameters and odd shapes, and the tutorial's networks use it.

    The cached current Heston example below shows the endpoint and its reference.
    Flow trajectories depend on the trained network and its sampler; we do not invent intermediate states from cached endpoint draws.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The archived BayesFlow 2.0 Euler-step experiment is retained in git history. Current estimators use BayesFlow 2.1's native adaptive solver; this notebook does not override it. Inspect all current Heston parameter pairs in the cached example below, and use the backbone review for explicit native sampling.
    """)
    return


@app.cell
def _(mo):
    from sbi4f_tutorial import backbones
    _task, _report, _arrays = backbones.load_example("heston-final500")
    mo.vstack([mo.md("**Current cached Heston development case**, with the PyMC reference. "
                    "No network is loaded on opening this lesson."), backbones.corner(_task, _arrays)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The current cached endpoint is a synthetic development observation with its own PyMC reference.
    It is not SPY. Flow paths need not be straight in physical parameter units because the adapter
    works in standardized, unbounded coordinates.

    ## 2 · Reading the data

    | Data | Shape | Network | Where |
    |:---|:---|:---|:---|
    | A smile on the design grid | a fixed vector of 63 prices (and, in slot 5, the forward-variance curve as context) | no summary network: the standardized prices enter as conditions | slots 4 and 5 |
    | A real option chain | a set of (log-moneyness, maturity, price, noise) tuples, of varying size and order | a set network (DeepSet, set transformer), invariant to the order of the quotes | the extension that removes the design grid |
    | Daily returns | any prefix of 1 to 1,260 daily observations | a causal GRU summary of the ordered returns | `trajectory_inference.py` (`pixi run trajectories`) |
    | Any data, for misspecification tests | any | a summary network trained with a distance term in its output space (Schmitt et al. 2024) | slot 6 |

    In BayesFlow each is one line, and the rest of the workflow does not change:

    ```python
    bf.networks.DeepSet(summary_dim=16)                    # a set of quotes, any order, any number
    bf.networks.SetTransformer(summary_dim=16)             # the same, with attention between quotes
    bf.networks.RecurrentNetwork(hidden_dim=64, summary_dim=32, recurrent_type="gru")  # ordered returns
    ```

    The current set-encoder experiments still use complete 63-quote surfaces. Supporting
    variable or missing quotes is a separate follow-up; a set network alone does not establish it.
    Where sufficient statistics exist, a learned summary can learn to approximate them.

    ## 3 · The configuration slot 4 trains

    The current network factory for slots 4 and 5 is this function (`src/sbi4f_tutorial/full_surface.py`):
    """)
    return


@app.cell(hide_code=True)
def _(full_surface, inspect, mo):
    mo.ui.code_editor(inspect.getsource(full_surface.build_workflow), language="python", disabled=True)
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
    5. **Training times** follow the current BayesFlow family defaults. The earlier 2.0 study
       used a custom time distribution; its results retain that historical identity.
    6. **Sampler.** `full_surface.sample` preserves BayesFlow 2.1's native adaptive solver,
       with the integration settings bound in the checkpoint metadata.

    ## 4 · How many steps does the sampler need?

    Fewer steps are faster and less accurate. The exact Heston posterior on SPY (slot 4) is the reference.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _message = ("The default sampler is BayesFlow 2.1's native adaptive solver. "
                "Numerical posterior-sampling error is distinct from simulator pricing error. "
                "Check both while preserving the recorded sampler identity.")
    mo.md(_message)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The archived BayesFlow 2.0 Euler-step experiment is retained in git history. Current estimators use BayesFlow 2.1's native adaptive solver; this notebook does not override it. Inspect all current Heston parameter pairs in the cached example below, and use the backbone review for explicit native sampling.
    """)
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
    `src/sbi4f_tutorial/full_surface.py`, the bound cases in `assets/backbones/`,
    and [the cached/live backbone review](backbone_review.py).
    """)
    return


if __name__ == "__main__":
    app.run()
