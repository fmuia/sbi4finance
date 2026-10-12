import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from sbi4finance.datasets.hkde_teaching import load_teaching_example

    ROOT = mo.notebook_dir().parent
    return ROOT, io, load_teaching_example, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # A local option panel → posterior → another option's model price

    This offline walkthrough follows one **synthetic** European-option panel.
    Upstream, the simulator workstream generated the training bank, trained a
    neural posterior and audited it against a numerical posterior. Here we
    load the local observation artifact and replay that audit's cached draws
    and prices. **No inference, training or new statistical evaluation runs.**

    The fitted quantities are just two risk-neutral HKDE parameters: long-run
    variance `theta` and jump intensity. All other parameters and the quote
    design remain fixed. This is not a fit of all nine HKDE parameters.
    """)
    return


@app.cell
def _(ROOT, load_teaching_example, mo):
    panel, report, arrays = load_teaching_example(ROOT / "data/synthetic/hkde-walkthrough")
    mo.callout(mo.md(f"""
    **Provisional example — published scientific validation has not passed.**

    Case: `{report['case_id']}`. The original audit's maximum price-quantile
    error was **{report['original_max_price_quantile_error_currency']:.5f}**
    currency, above its **{report['price_error_gate_currency']:.2f}** allowance.
    The checkpoint is not accepted for participant pricing. Successful data
    loading or notebook execution does not change that status.
    """), kind="warn")
    return arrays, panel, report


@app.cell
def _(mo, panel):
    _task = panel.metadata["scientific_task"]
    _sigma = _task["context"]["quote_noise_std"]
    mo.vstack([
        mo.md(f"""
        ## 1. What was observed, and what does a fit mean?

        Each row is one noisy option premium, in currency per underlying unit.
        Strikes, maturities and option types identify the fixed contracts.
        There are no bid/ask spreads or implied-volatility inputs here.

        For a parameter pair, HKDE computes nine model prices. The likelihood
        compares these with the observed prices using independent Gaussian
        errors with known standard deviation **{_sigma:g} currency**. The
        posterior combines that likelihood with the stated prior.

        Simulator truth is available only because this panel is synthetic.
        It belongs to evaluation, never to the network's conditioning inputs.
        """),
        mo.ui.table(list(panel.rows), selection=None, show_column_summaries=False),
    ])
    return


@app.cell
def _(arrays, mo, panel, report):
    mo.vstack([
        mo.md("""
        ## 2. The inference boundary

        The reader verifies file hashes, the full saved task and the ordered
        panel. Matching array dimensions alone would be insufficient. Only
        the nine prices below enter conditioning; contract order and known
        context must match the checkpoint. Truth and auxiliary fields do not
        become observations. The two targets are inferred jointly.
        """),
        mo.ui.table([
            {"Input position": _index, "Price": float(_price)}
            for _index, _price in enumerate(arrays["conditioning_prices"])
        ], selection=None, show_column_summaries=False),
        mo.accordion({"Fixed scientific task and prior": mo.json(panel.metadata["scientific_task"]),
                      "Source identities and cached-example record": mo.json(report)}),
    ])
    return


@app.cell
def _(arrays, io, mo, plt, report):
    _draws = arrays["parameter_draws"]
    _fig, _ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
    _ax.scatter(_draws[::4, 0], _draws[::4, 1], s=7, alpha=.25, color="#276678")
    _ax.set(xlabel="Long-run variance θ", ylabel="Jump intensity λ, per year")
    _image = io.BytesIO()
    _fig.savefig(_image, format="png", dpi=120)
    plt.close(_fig)
    mo.vstack([
        mo.md(f"""
        ## 3. Keep parameter draws paired

        The cache contains **{len(_draws):,}** joint draws for `{report['case_id']}`.
        Each row describes one possible parameter pair. The plot displays
        every fourth row; all rows contribute to the price summaries below.
        Independently shuffling the columns would destroy posterior dependence.
        These draws replay the provisional neural candidate, not a fresh call
        to the checkpoint and not an exact posterior sample.
        """),
        mo.image(_image.getvalue()),
    ])
    return


@app.cell
def _(arrays, io, mo, np, plt, report):
    _prices = arrays["model_price_draws"]
    _probabilities = np.asarray(report["quantiles"])
    _candidate = np.quantile(_prices, _probabilities)
    _reference = np.asarray(report["reference_price_quantiles"])
    _contract = report["target_contract"]
    _fig, _ax = plt.subplots(figsize=(6, 3.3), layout="constrained")
    _ax.hist(_prices, bins=35, density=True, color="#276678", alpha=.65)
    _ax.set(xlabel="Held-out option's model price, currency", ylabel="Posterior density")
    _image = io.BytesIO()
    _fig.savefig(_image, format="png", dpi=120)
    plt.close(_fig)
    mo.vstack([
        mo.md(f"""
        ## 4. Price a contract outside the conditioning panel

        Target: European {_contract['option_type']}, strike **{_contract['strike']:g}**,
        maturity **{_contract['maturity']:g} years**. For each matched parameter
        pair, the upstream reference pricer computed this contract's expected
        discounted payoff. We summarize those cached prices here.

        This distribution describes **uncertainty about a model expectation**.
        It is neither the distribution of a realized option payoff nor a
        forecast of tomorrow's market quote. No target quote entered the fit.
        """),
        mo.image(_image.getvalue()),
        mo.ui.table([
            {"Quantile": float(_q), "Numerical posterior": float(_r),
             "Neural candidate": float(_c), "Difference, currency": float(_c - _r)}
            for _q, _r, _c in zip(_probabilities, _reference, _candidate, strict=True)
        ], selection=None, show_column_summaries=False),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## What would real data add?

    Synthetic data suffice to build this workflow, check recovery against
    known truth, and compare numerical and neural posteriors under the chosen
    model. Real quotes are needed to assess fit to an actual market, residual
    structure or pricing performance on withheld market contracts. There is
    no known market parameter truth, and risk-neutral calibration does not
    establish real-world return or loss forecasts.

    A market example needs synchronized premiums, contract conventions and
    consistent pricing context. This fixed checkpoint cannot simply absorb
    arbitrary chains, American contracts or new rates: those require a
    compatible observation model and inference setup. A market snapshot does
    not replace the synthetic training bank.

    See the [offline example notes](../docs/hkde-offline-teaching.md) for
    provenance, reproduction and optional checkpoint/pricer execution.

    Continue with the [corrected HKDE lesson](10_hkde_corrected_workflow.py).
    It loads this same local panel and compares the preserved raw result with
    cached importance-corrected prices. Run `pixi run demo-hkde-corrected`
    for its opt-in provisional live calculation; formal scientific acceptance
    remains separate.
    """)
    return


if __name__ == "__main__":
    app.run()
