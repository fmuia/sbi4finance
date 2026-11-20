import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Slot 3 · Network architectures")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from sbi4f_tutorial import plots

    plots.style()
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Neural network architectures for amortized inference

    **Which networks represent the posterior, and how they process the data**

    Slot 2 ended on the object to build: a neural estimator $q_\phi(\theta \mid y)$ trained on simulated pairs. This notebook shows the networks that build it, using the current cached Heston development example and its PyMC reference. Historical SPY results remain labelled separately in slot 4.

    The slot in brief:

    1. **An inference network represents the posterior.** Flow matching learns a velocity field that transports Gaussian noise to the posterior along an ordinary differential equation.
    2. **A summary network, when needed, embeds the data.** Irregular or non-vector data types (e.g., time series, sets, graphs) benefit from an embedding step $h(y)$ trained jointly with the inference network.
    3. **A few details decide whether a network works on financial data:** bounded parameters, scales, noise in every batch, and the sampler's steps.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid("""
    flowchart LR
      Y["data y: quotes, returns, etc"] --> H["summary net h (optional)"]
      H --> Q["inference net q(θ | h(y))"]
      Y -. "a fixed grid needs no summary" .-> Q
      Z["Gaussian noise z"] --> Q
      Q --> D["posterior draws of θ"]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Flow matching: transporting noise to the posterior

    For each simulated pair $(\theta,y)$, we draw Gaussian noise $z_1 \sim \mathcal{N}(0, I)$ and a time $t \sim \mathcal{U}(0, 1)$. We construct a straight path between the parameter and the noise:

    $$
    z_t = (1 - t) \theta + t z_1.
    $$

    At $t = 0$, the path starts at the parameter $\theta$; at $t = 1$, it ends at Gaussian noise $z_1$. Its velocity is simply $z_1 - \theta$. We train a neural network $u_\phi(z_t, t, y)$ to predict this velocity by minimizing a regression loss:

    $$
    \mathcal{L}(\phi) = \mathbb{E}_{p(\theta, y)p(z)p(t)}\left[\left\|u_\phi((1 - t)\theta + t z_1, t, y)-(z_1 -\theta)\right\|^2\right].
    $$

    **How does this generate posterior samples?** Given observed data $y$, we draw $x_1\sim\mathcal{N}(0,I)$ and solve the learned ODE backward in time:

    $$
    \frac{dz_t}{dt}=u_\phi(x_t,t,y),\qquad t: 1 \rightarrow 0.
    $$

    The endpoint $z_0$ is approximately distributed according to the posterior $q_\phi(\theta \mid y)$. Training is very fast, but sampling requires numerical ODE integration.

    The cached Heston example below shows posterior samples and their reference distribution. Intermediate trajectories depend on the trained velocity field and numerical solver and cannot be reconstructed from endpoint samples alone.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The [BayesFlow 2.1 inference-network guide](https://bayesflow.org/v2.1.0/user_guide/inference_networks.html) compares the available posterior estimators:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _rows = [
        {"Network": "CouplingFlow", "Architecture": "Normalizing flow", "Inference speed": "⚡⚡⚡", "Exact density": "✓", "Best for": "Simple to moderately complex posteriors"},
        {"Network": "FlowMatching", "Architecture": "Flow matching (OT)", "Inference speed": "⚡", "Exact density": "✓", "Best for": "Complex, multimodal posteriors"},
        {"Network": "DiffusionModel", "Architecture": "Score-based SDE/ODE", "Inference speed": "⚡", "Exact density": "✓", "Best for": "Complex, multimodal; compositional inference"},
        {"Network": "ConsistencyModel", "Architecture": "Consistency training", "Inference speed": "⚡⚡", "Exact density": "—", "Best for": "Fast single-step distillation training"},
        {"Network": "StableConsistencyModel", "Architecture": "Stable CT", "Inference speed": "⚡⚡", "Exact density": "—", "Best for": "Continuous consistency training without discretization"},
        {"Network": "ScoringRuleNetwork", "Architecture": "Feed-forward + heads", "Inference speed": "⚡⚡⚡⚡", "Exact density": "Parametric", "Best for": "Various Bayes estimators and parametric distributions"},
        {"Network": "PointNetwork", "Architecture": "Feed-forward + heads", "Inference speed": "⚡⚡⚡⚡", "Exact density": "—", "Best for": "Posterior mean and quantile estimation"},
    ]
    mo.ui.table(_rows, selection=None, pagination=False)
    return


@app.cell(hide_code=True)
def _(mo):
    from sbi4f_tutorial import backbones
    cache_task, _report, cache_arrays = backbones.load_example("heston-final500")
    mo.vstack([mo.md("**Current cached Heston development case**, with the PyMC reference."), backbones.corner(cache_task, cache_arrays)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Processing the observed financial data

    It is useful to remember that amortized Bayes solves two tasks at the same time: **representation learning** and **probabilistic inference**. In BayesFlow, these are handled together by separate neural networks:

    - **Summary network:** Transforms the observed data $y$ into an informative representation $s_\psi(y)$ for estimating the parameters $\theta$.
    - **Inference network:** Learns the conditional posterior $q_\phi(\theta\mid s_\psi(y))$ and uses this representation to generate posterior samples.

    The summary network does not need to reconstruct the observations or retain every detail. Its goal is to preserve information relevant to the parameters. Ideally, $p(\theta\mid y)=p(\theta\mid s_\psi(y))$. Both networks are trained jointly, so the summary network learns which features of the data are useful for posterior inference.

    **When do we need a summary network?** For small, fixed-dimensional inputs, we can pass the observations directly to the inference network. For variable-sized data, long time series, or unordered sets, we need an architecture that converts observations into a fixed-dimensional representation while respecting their symmetry.

    | Data | Shape | Network | Where |
    |:---|:---|:---|:---|
    | A smile on the design grid | A fixed vector of 63 prices (and, in slot 5, the forward-variance curve as context) | No summary network: standardized prices enter directly as conditions | Slots 4 and 5 |
    | A real option chain | A set of (log-moneyness, maturity, price, noise) tuples, with varying size and order | DeepSet or SetTransformer: invariant to the order of quotes | Extension beyond the fixed design grid |
    | Daily returns | Any prefix of 1 to 1,260 daily observations | Recurrent network (GRU): processes the ordered sequence | `trajectory_inference.py` (`pixi run trajectories`) |

    In BayesFlow, choosing an appropriate summary network takes just one line:

    ```python
    bf.networks.DeepSet(summary_dim=16)  # Unordered sets of quotes
    bf.networks.SetTransformer(summary_dim=16)  # Sets with attention between quotes
    bf.networks.RecurrentNetwork(hidden_dim=64, summary_dim=32, recurrent_type="gru")  # Ordered returns
    ```

    The [BayesFlow 2.1 summary-network guide](https://bayesflow.org/v2.1.0/user_guide/summary_networks.html) compares the available summary networks. You can even use pretrained foundation models as backbone, which is described in this more [advanced tutorial](https://github.com/bayesflow-org/bayesflow/blob/dev/examples/Foundation_Models.ipynb).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. A Heston workflow is one plain BayesFlow specification:

    ```python
    import bayesflow as bf

    adapter = (
        bf.Adapter()
        .constrain("initial_variance", lower=0.01, upper=0.16, inclusive="none")
        .constrain("theta", lower=0.01, upper=0.16, inclusive="none")
        .constrain("vol_of_vol", lower=0.1, upper=1.5, inclusive="none")
        .constrain("rho", lower=-0.95, upper=-0.1, inclusive="none")
        .concatenate(
            ["initial_variance", "theta", "vol_of_vol", "rho"],
            into="inference_variables",
        )
        .rename("observations", "inference_conditions")
    )

    inference_network = bf.networks.FlowMatching(
        subnet_kwargs={"widths": (256,)*4}
    )

    workflow = bf.BasicWorkflow(
        adapter=adapter,
        inference_network=inference_network,
        standardize="all"
    )

    history = workflow.fit_offline(
        data=training_data,
        epochs=500,
        batch_size=256
    )
    ```

    - The adapter prepares simulated data for neural network training. It maps the four Heston parameters to unconstrained values, respecting their specified bounds, and combines them into a single parameter vector (`inference_variables`). It also tells BayesFlow which simulated observations should be used as inputs (`inference_conditions`).

    - The workflow is a high-level wrapper that brings everything together: the adapter, the flow-matching network, and automatic standardization. During training, BayesFlow processes the simulated parameter–observation pairs, standardizes the inputs, and trains the network to approximate the posterior. Here, we train for 500 epochs using batches of 256 simulations.

    - Once trained, the same workflow can generate posterior samples for new option prices without retraining the network.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    1. **Flow matching learns to turn noise into posterior samples.** Once trained, the same network can perform inference for different volatility smiles.
    2. **How we process the data depends on its structure.** A fixed design grid can go straight into the inference network. Option chains with varying numbers of quotes need a set network, while daily returns call for a time-series network.
    3. **The practical details matter.** Parameter bounds, standardization, and network hyperparameters all play a role in getting reliable posterior estimates.

    **Next, slot 4:** We put everything together in BayesFlow, using the Heston model and real SPY option prices, and compare the results against the exact posterior.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
