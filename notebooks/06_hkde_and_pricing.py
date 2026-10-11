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

    from sbi4finance import hkde, option_bank

    ROOT = mo.notebook_dir().parent
    return ROOT, hkde, json, mo, np, option_bank, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    # 6. HKDE: from a jump law to an option-panel observation

    The reusable building block here is a **risk-neutral simulator/pricer and
    synthetic option bank**. A small numerical example is not an accepted
    posterior. Heston returns inference remains provisional; this independent
    options branch can advance without silently passing its failed checks.

    This notebook uses hypothetical contracts, rates and parameters. It runs
    offline, with no market files, neural training or cluster submission.
    Recorded reports describe **S: synthetic numerical evidence** only. The
    eventual live tutorial may still choose returns, options, or a carefully
    separated pair of examples.

    ## Why pricing code appears in an SBI project

    A market option quote estimates an expectation, not one randomly realized
    future payoff. To simulate a quote panel, draw a Q parameter vector, compute
    model prices, then apply a declared observation-error law. Later SBI can
    reverse that experiment: noisy panel → joint parameter posterior.

    | Workflow | Inputs | Training target or output |
    |---|---|---|
    | Reference pricing, implemented building block | Parameters, contracts, context | Discounted expected payoff |
    | Parameter inference, deferred | Noisy panel and known context | Joint Q parameters |
    | Pricing surrogate, deferred | Parameters, contracts, context | Accurate reference-price labels |
    | Direct price posterior, deferred | Noisy panel and context | Target prices generated from the same parameters |

    A posterior can be pushed through the reference pricer before a surrogate
    exists. A surrogate approximates the pricing map; it does not eliminate the
    need for trustworthy training prices. No workflow here forecasts tomorrow's
    market option chain or supplies physical-measure risk probabilities.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Heston variance plus double-exponential price jumps

    Let $X=\log(S/S_0)$, and measure time in years. Under Q,

    $$dV_t=\kappa(\theta-V_t)dt+\xi\sqrt{V_t}\,dW_t^V,$$
    $$dX_t=(r-q-\lambda c_J-V_t/2)dt+\sqrt{V_t}\,dW_t^S+J\,dN_t,
    \qquad d\langle W^S,W^V\rangle_t=\rho\,dt.$$

    The jump count has intensity $\lambda$ per year. Each log jump is positive
    with probability $p_+$; its magnitude then has exponential rate $a_+$.
    Otherwise it is minus an exponential with rate $a_-$. Thus

    $$f_J(j)=p_+a_+e^{-a_+j}\mathbf1_{j\ge0}
       +(1-p_+)a_-e^{a_-j}\mathbf1_{j<0},$$
    $$c_J=\mathbb E[e^J-1]
       =\frac{p_+a_+}{a_+-1}+\frac{(1-p_+)a_-}{a_-+1}-1.$$

    Rates are inverse sizes: mean upward log-jump magnitude is $1/a_+$.
    With positive upward mass, $a_+>1$ is needed for the compensator.
    Zero-probability terms are omitted. Subtracting $\lambda\mathbb E[J]$
    would not compensate proportional price growth.

    HKDE and Gaussian-jump Heston are sibling choices of jump law; neither
    adds jumps to variance. The Q parameters are pricing parameters. Changing
    a fitted P drift to $r-q$ does not identify the other Q parameters or a
    measure-change model.

    For $F_T=S_0e^{(r-q)T}$ and $D_T=e^{-rT}$, European prices are
    $C=D_T\mathbb E_Q[(S_T-K)^+]$ and
    $P=D_T\mathbb E_Q[(K-S_T)^+]$, with $C-P=D_T(F_T-K)$.
    Fourier inversion evaluates these expectations from the model's
    characteristic function; it has its own integration truncation and
    quadrature errors. It is not the neural inference step. This implementation
    uses direct Fourier quadrature, not the paper's PROJ or barrier pricer.
    """)
    return


@app.cell
def _(mo):
    jump_control = mo.ui.slider(0.0, 1.5, step=0.1, value=0.5,
                                label="Jump intensity λ / year", show_value=True)
    noise_control = mo.ui.slider(0.0, 0.3, step=0.025, value=0.1,
                                 label="Quote-noise SD, price units", show_value=True)
    mo.vstack([
        mo.md("## A small synthetic panel\n\nSix contracts at two maturities; spot 100, annual rate 2%, dividend yield zero. Controls are illustrations, not a new validated prior."),
        mo.hstack([jump_control, noise_control]),
    ])
    return jump_control, noise_control


@app.cell
def _(hkde, jump_control, noise_control, np):
    parameters = dict(kappa=2.0, theta=0.04, vol_of_vol=0.3, rho=-0.6,
                      jump_intensity=jump_control.value, jump_up_probability=0.3,
                      jump_up_rate=12.0, jump_down_rate=8.0)
    maturity = np.repeat([0.25, 1.0], 3)
    strike = np.tile([90.0, 100.0, 110.0], 2)
    forward = 100 * np.exp(0.02 * maturity)
    discount = np.exp(-0.02 * maturity)
    clean_put = hkde.european_price(forward, strike, discount, maturity,
                                    parameters, 0.04, kind="put")
    clean_call = hkde.european_price(forward, strike, discount, maturity,
                                     parameters, 0.04, kind="call")
    quote_noise = np.random.default_rng(6401).normal(0, noise_control.value, size=6)
    noisy_put = clean_put + quote_noise
    return clean_call, clean_put, discount, forward, maturity, noisy_put, parameters, strike


@app.cell
def _(clean_put, maturity, noisy_put, np, plt, strike):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True, layout="constrained")
    for _ax, _time in zip(_axes, np.unique(maturity), strict=True):
        _mask = maturity == _time
        _ax.plot(strike[_mask], clean_put[_mask], "o-", label="Model expectation estimate")
        _ax.scatter(strike[_mask], noisy_put[_mask], marker="x", s=70,
                    label="Observed synthetic quote", color="#bb653f")
        _ax.set(xlabel="Strike, price units", title=f"Maturity {_time:g} years")
    _axes[0].set_ylabel("European put price, price units")
    _axes[1].legend(fontsize=8)
    _fig
    return


@app.cell
def _(clean_call, clean_put, discount, forward, hkde, maturity, mo, np, parameters, pd, strike):
    _refined = hkde.european_price(forward, strike, discount, maturity, parameters,
                                  0.04, kind="put", upper_limit=300, nodes=768)
    _parity = clean_call - clean_put - discount * (forward - strike)
    _table = pd.DataFrame({
        "Maturity (years)": maturity, "Strike": strike,
        "Clean put": clean_put, "Refined put": _refined,
        "Absolute refinement difference": np.abs(_refined - clean_put),
        "Put-call parity residual": _parity,
    })
    mo.vstack([
        mo.md("""
        ## Numerical error and observation noise answer different questions

        This local comparison changes the Fourier integration cutoff and node
        count. Small differences support this particular panel; two numerical
        settings agreeing is not an independent proof of accuracy. Put–call
        parity checks consistency, and may be enforced by the pricing route
        itself. The recorded validation below includes the separate checks.

        Gaussian quote errors in the illustration are intentional measurement
        noise, in price units. They are not estimates of pricing error. A
        Gaussian model can produce negative or arbitrage-inconsistent observed
        quotes; clipping them would change that observation law. The clean
        model prices must obey their own financial/numerical checks.
        """),
        mo.ui.table(_table, selection=None, show_column_summaries=False),
    ])
    return


@app.cell
def _(ROOT, json, mo, pd):
    hkde_report = json.loads((ROOT / "docs/results/hkde-validation.json").read_text())
    _rows = []
    for _case in hkde_report["pricing_fixtures"]:
        _rows.append({
            "Fixture": _case["name"],
            "Largest Fourier refinement difference / DF": max(c["cutoff_node_refinement_max_error"] for c in _case["contracts"]),
            "Independent inversion difference / DF": max(c["independent_payoff_inversion_max_error"] for c in _case["contracts"]),
            "Riccati reference CF error": _case["riccati_reference_max_error"],
            "Recorded result": "PASS" if _case["passed"] else "FAIL",
        })
    mo.vstack([
        mo.md("""
        ## Recorded evidence: read the scope as well as the numbers

        This is the saved numerical report, not a computation launched when
        opening the notebook. Its cases, refinement settings, independent
        references and tolerances delimit the evidence. A passed software
        check or a small pilot bank does not validate every point of a larger
        prior. No posterior, learned surrogate or empirical market assessment
        is accepted by this report.
        """),
        mo.ui.table(pd.DataFrame(_rows), selection=None, show_column_summaries=False),
        mo.md(f"""
        The Fourier refinement budget is
        **{hkde_report['fixed_budgets']['fourier_refinement']:g} of DF**,
        with forward and discount both one in the fixtures. These normalized
        units differ from the live spot-100 panel's currency units. The
        Riccati characteristic-function comparison budget is
        **{hkde_report['fixed_budgets']['riccati_reference']:g}**.
        Overall recorded numerical result:
        **{'PASS' if hkde_report['passed'] else 'FAIL'}** within the saved scope.
        Path Monte Carlo in the full report retains correlated price errors
        and a separate diffusion-refinement allowance; those allowances are
        not certified bounds on unknown continuous-model bias.
        """),
        mo.accordion({"Full report, path references and limitations": mo.json(hkde_report)}),
    ])
    return (hkde_report,)


@app.cell
def _(ROOT, json, mo, option_bank, pd):
    hkde_bank_config = json.loads((ROOT / "configs/tasks/hkde_option_bank.json").read_text())
    _example = option_bank.generate_examples(hkde_bank_config, "train", [0])
    _table = pd.DataFrame({
        "Contract index": range(_example["observations"].shape[1]),
        "Observed noisy quote": _example["observations"][0, :, 0],
        "Reference price (not observed)": _example["reference_prices"][0],
        "Refinement difference (not observed)": _example["refinement_difference"][0],
    })
    mo.vstack([
        mo.md("""
        ## One example from the shared bank generator

        This executes exactly one seeded training example from the committed
        configuration; it neither writes a bank nor trains a network. The
        recorded task fixes nine put contracts: strikes 90/100/110 at
        0.25/0.5/1 years, spot 100, rate 2%, dividend zero and Gaussian
        quote-noise SD 0.25 price units. It varies theta and jump intensity;
        the remaining HKDE parameters are fixed. The configuration below is
        the authority for prior bounds, target order and numerical settings.
        Its restricted prior is not the full supported model domain.
        """),
        mo.ui.table(_table, selection=None, show_column_summaries=False),
        mo.json(hkde_bank_config),
    ])
    return (hkde_bank_config,)


@app.cell
def _(mo):
    mo.md(r"""
    ## What is one observation, and what belongs in a bank?

    One calibration example is **one complete option panel** at one synthetic
    valuation date. Its contracts have known strikes, maturities, exercise
    type, forwards and discount factors. The target is a selected joint Q
    parameter vector. A clean price, simulated jump realization, exact
    parameter or numerical error estimate is not an extra observed quote.
    In this task, batch arrays are `observations[B, 9, 1]` and
    `parameters[B, 2]` ordered as theta and jump intensity. Initial variance
    0.04 is known context. `reference_prices[B, 9]` and numerical diagnostics
    remain separate from the observed channel.

    | Role | Stored content | May condition a later estimator? |
    |---|---|---|
    | Observations | Noisy quotes in the declared contract order | Yes |
    | Known context | Contracts, synthetic rates/spot or forward/discount inputs | Yes, within the task's declared support |
    | Truth labels | Drawn parameters and clean model-price labels | Targets or checks; not hidden input channels |
    | Numerical diagnostics | Refinement residuals, Monte Carlo covariance if used | No, unless a separate observed-error task declares them |
    | Identity and provenance | Base-surface ID, split, task/source hashes | Integrity and grouped splitting |

    Draw each base surface into exactly one training, validation or test split
    before creating quote-noise augmentations. A shared bank architecture can
    retain these roles for different models while keeping their parameter and
    numerical contracts distinct. An option panel is not a return sequence.

    **S** needs no real market observation. Optional **M** conditions a trained
    network on a compatible actual panel, normally with frozen weights and
    without parameter labels. Optional **E** compares to contemporaneous held-out
    quotes under a leakage-free selection/preprocessing design. Target market
    quotes are not required merely to generate a posterior distribution of
    model prices. A single market case does not establish general performance.

    ## Review exercises

    1. If $a_+$ decreases toward one, why does a small change in a tail
       parameter threaten pricing? Inspect the exponential moment, not just
       the mean log jump.
    2. Could one simulated terminal payoff replace each clean price label?
       Explain which expectation and numerical uncertainty would be lost.
    3. If two noisy versions of the same clean surface enter different splits,
       what does validation reuse? Keep the underlying parameter/context draw
       and all its quote augmentations together.

    Sources and reproduction: [task configuration](../configs/tasks/hkde_option_bank.json),
    [HKDE source](../src/sbi4finance/hkde.py),
    [shared bank source](../src/sbi4finance/option_bank.py),
    [numerical report](../docs/results/hkde-validation.json),
    [reference design and reproduction](../docs/hkde-reference.md),
    [M4 subplan](../docs/simulator-plan/milestones/04-hkde-and-options.md),
    [mathematical notes](../docs/simulator-plan/notes/02-models-and-numerics.md),
    [inference/pricing workflows](../docs/simulator-plan/notes/03-inference-and-pricing.md),
    [next analysis](../docs/simulator-analysis-next-steps.md), and the
    [HKDE paper](https://arxiv.org/html/2502.13824v1).
    """)
    return


if __name__ == "__main__":
    app.run()
