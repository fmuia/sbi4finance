import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Desk case · front-month smiles, 1-year products",
)


@app.cell(hide_code=True)
def _():
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from sbi4finance.calibration import plots

    ROOT = mo.notebook_dir().parent
    CHECKS = ROOT / "checks"
    sys.path.insert(0, str(CHECKS / "desk_case"))
    import desk_figures as F  # noqa: E402  (tutorial_final/checks/desk_case/desk_figures.py)

    plots.style()
    data = F.load(CHECKS / "desk_case")
    key = F.key_numbers(data)
    return F, data, key, mo, plt


@app.cell(hide_code=True)
def _(key, mo):
    mo.md(rf"""
    # A desk case · Marking 1-year products from front-month smiles

    **When the trusted quotes stop at three months, the 1-year marks rest on a parameter the quotes barely see.
    The posterior measures that uncertainty. Gauss–Newton and fixing $H$ do not.**

    This case uses everything built in slots 4 and 5: rough Bergomi, its Monte Carlo pricer, the importance-sampling
    correction and the desk products. It changes one thing: the quotes. A thinner market than SPY's offers reliable
    smiles only for the front three monthly expiries.

    Three claims, each backed by evidence below:

    1. **The front months leave $H$ open, and the posterior bends.** Rough Bergomi's 90% posterior interval for $H$
       runs from {key['h_post'][0]:.2f} to {key['h_post'][1]:.2f}. Along it, $\eta$ and $\rho$ move on a curved ridge
       that Gauss–Newton's ellipse cannot follow. For example, the posterior's 95% point for $\eta$ is
       {key['eta_post'][1]:.2f}, against {key['eta_gn'][1]:.2f} for Gauss–Newton.
    2. **The 1-year products move along that ridge.** The forward start's one-sided 90% reserve is
       {key['fs']['posterior'][0]:.0f}% of the mark (long) and {key['fs']['posterior'][1]:.0f}% (short) under the
       posterior. The delta method gives {key['fs']['delta'][0]:.0f}% on both sides. Fixing $H$ at 0.10, as desks
       do, gives {key['fs']['h_fixed_delta'][0]:.0f}%.
    3. **{key['validation_claim']}**

    One caveat comes with the real data. The full SPY surface prices the forward start
    {key['full_gap']:.0f}% above the front-month mark, outside every range above. That gap is model error, not
    parameter uncertainty, and no calibration method closes it. It is the subject of slot 6.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The desk's problem

    A structured-products desk marks four 1-year products on an index. Its listed market is thin. Only the front
    three monthly expiries, at 21, 42 and 63 trading days, have smiles it trusts. Beyond three months it has the
    forward-variance curve, from its variance-swap marks, and nothing else. This is the usual position for single
    stocks and for smaller indices.

    Here the market is SPY's real smile of 21 October 2026, cut down to those 27 quotes. The quote noise is the
    design noise of the tutorial: 0.3 vol points at the money and 0.4 to 0.7 in the put wing. A switch below also
    shows 1.5 times that, for a thinner market.

    | Product | Payoff, in units of the initial level |
    |:---|:---|
    | 1-year autocallable | called at 3, 6 or 9 months if the index is at or above 100%, paying 1 plus 1.25% per quarter elapsed; at 1 year, 1.05 if at or above 100%, 1 if at or above 70%, the index level otherwise |
    | 6m→1y forward start | $(S_{1y}/S_{6m} - 1)^+$, the strike set at six months at the money |
    | 1-year 80% put | $(0.8 - S_{1y})^+$ |
    | 1-year capped cliquet | $\max\big(\sum_{i=1}^{12} \min(R_i, 2\%),\, 0\big)$, with $R_i$ the monthly return |

    All four are priced by Monte Carlo on rough Bergomi paths, with 16,384 paths and common random numbers.
    """)
    return


@app.cell(hide_code=True)
def _(F, data, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(11.5, 3.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.2, 1]})
    F.quotes_panel(_axes[0], data)
    F.timeline_panel(_axes[1])
    _fig
    return


@app.cell(hide_code=True)
def _(key, mo):
    mo.md(rf"""
    ## 2 · What the front months say about $H$

    Fix $H$, refit $\eta$ and $\rho$ to the 27 quotes, and record how well they fit. That is the profile likelihood
    in $H$. It is also what a desk gets when it fixes $H$, for every value it could choose.

    The fit is almost flat. From $H$ = {key['profile_flat'][0]:.2f} to {key['profile_flat'][1]:.2f} the
    chi-square stays within 1.64 of its minimum, the one-sided 90% threshold. Over that range the forward start moves
    from {key['profile_fs'][0]:+.0f}% to {key['profile_fs'][1]:+.0f}% of the mark.

    The reason is what $H$ controls. The front-month smiles fix the skew between one and three months. $H$ sets how
    fast the skew and the vol-of-vol decay beyond that. A forward start from six months to a year depends on exactly
    that decay.
    """)
    return


@app.cell(hide_code=True)
def _(F, data, plt):
    _fig, _ax = plt.subplots(figsize=(8.2, 3.9), constrained_layout=True)
    F.profile_panel(_ax, data)
    _fig
    return


@app.cell(hide_code=True)
def _(key, mo):
    noise = mo.ui.radio({"design noise": 1.0, "1.5 × design noise": 1.5}, value="design noise",
                        label="quote noise", inline=True)
    mo.md(f"""
    ## 3 · The posterior and Gauss–Newton

    The posterior is computed exactly, by importance sampling with the Monte Carlo likelihood (slot 5, section 5), in two
    rounds of proposals: {key['proposals']:,} Monte Carlo surfaces in all. Gauss–Newton is the least-squares fit with the
    covariance $(J^\\top \\Sigma^{{-1}} J)^{{-1}}$, on the same pricer.

    {noise}
    """)
    return (noise,)


@app.cell(hide_code=True)
def _(F, data, noise, plt):
    _fig = plt.figure(figsize=(12.0, 7.0), constrained_layout=True)
    _grid = _fig.add_gridspec(2, 3)
    for _i in range(3):
        F.marginal_panel(_fig.add_subplot(_grid[0, _i]), data, _i, noise.value)
    F.joint_panel(_fig.add_subplot(_grid[1, :2]), data, 0, 2, noise.value)
    _ax = _fig.add_subplot(_grid[1, 2])
    F.joint_panel(_ax, data, 0, 1, noise.value)
    _fig
    return


@app.cell(hide_code=True)
def _(F, data, mo, noise):
    _k = F.key_numbers(data, noise.value)
    mo.md(rf"""
    The posterior follows the ridge of section 2, bounded by the prior box and by $\rho > -1$. Gauss–Newton replaces it
    by an ellipse around the least-squares point $H$ = {_k['ls'][0]:.3f}, $\eta$ = {_k['ls'][1]:.2f},
    $\rho$ = {_k['ls'][2]:.2f}.

    - $H$: posterior {_k['h_post'][0]:.3f} to {_k['h_post'][1]:.3f}; Gauss–Newton {_k['h_gn'][0]:.3f} to
      {_k['h_gn'][1]:.3f}.
    - $\eta$: posterior {_k['eta_post'][0]:.2f} to {_k['eta_post'][1]:.2f}; Gauss–Newton {_k['eta_gn'][0]:.2f} to
      {_k['eta_gn'][1]:.2f}. The posterior's upper tail is longer, because at large $H$ the ridge needs a larger
      $\eta$ (the profile path of section 2).
    - $\rho$: posterior {_k['rho_post'][0]:.2f} to {_k['rho_post'][1]:.2f}; Gauss–Newton {_k['rho_gn'][0]:.2f} to
      {_k['rho_gn'][1]:.2f}.
    - The importance weights are reliable: Pareto $\hat k$ = {_k['khat']:.2f}, effective sample size
      {_k['ess']:.0f}.

    The interval widths differ by {_k['width_diff'][0]:.0f}% to {_k['width_diff'][2]:.0f}%. The shape differs more. In the $(H, \eta)$ panel the ridge bends upward at both
    ends, and the ellipse is straight. The products of section 4 sit on that ridge, so the shape is what decides their
    reserves.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    level = mo.ui.slider(0.80, 0.99, step=0.01, value=0.90, label="one-sided confidence level", show_value=True)
    mo.md(f"""
    ## 4 · The reserves

    Each method gives a mark and a prudent value on each side. The reserve for a long position is the mark minus the
    low quantile. For a short position it is the high quantile minus the mark. Prudent valuation sets the level at 90%.
    The four methods use the same 27 quotes:

    - **Posterior:** quantiles of the product over the posterior draws, around the least-squares mark.
    - **Gauss–Newton, delta method:** $z\\,\\sqrt{{g^\\top C g}}$, with $g$ the product's gradient at the fit.
    - **Profile likelihood:** the extremes of the product over the parameters whose chi-square is within $z^2$ of the
      minimum. This is the frequentist "range of plausible valuations".
    - **$H$ fixed at 0.10:** the delta method on $\\eta$ and $\\rho$ alone. This is what a desk does when it borrows $H$
      from the rough-volatility literature.

    {level}
    """)
    return (level,)


@app.cell(hide_code=True)
def _(F, data, level, mo, noise, plt):
    _fig, _ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    F.reserves_panel(_ax, data, noise.value, level.value)
    mo.vstack([_fig, mo.ui.table(F.reserve_rows(data, noise.value, level.value), selection=None, pagination=False)])
    return


@app.cell(hide_code=True)
def _(F, data, mo, noise):
    _k = F.key_numbers(data, noise.value)
    mo.md(rf"""
    Three things stand out:

    - **Fixing $H$ removes most of the forward start's reserve.** It drops from {_k['fs']['posterior'][0]:.0f}% to
      {_k['fs']['h_fixed_delta'][0]:.0f}% of the mark. The desk's choice of $H$ is not in any reserve.
    - **The delta method is symmetric; the posterior is not.** The 80% put's posterior reserve is
      {_k['put']['posterior'][0]:.0f}% long and {_k['put']['posterior'][1]:.0f}% short. The delta method gives
      {_k['put']['delta'][0]:.0f}% on both sides.
    - **The profile likelihood is the closest frequentist answer.** It also reads the ridge. But it is a range, not a
      probability. Its width depends on the threshold, and it has no coverage statement.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Which reserve is right?

    Coverage settles this. Draw parameter sets the desk cannot rule out from what it sees: resample them from the
    front-month posterior. For each, simulate a thin chain with an independent Monte Carlo run and fresh quote noise. Then
    let each method mark the products and set its 90% reserves from that chain alone. A correct one-sided 90% reserve
    is breached 10% of the time on each side.
    """)
    return


@app.cell(hide_code=True)
def _(F, data, mo):
    F.validation_view(data, mo)
    return


@app.cell(hide_code=True)
def _(key, mo):
    mo.md(rf"""
    ## 6 · And the real market?

    SPY does quote the longer maturities, so the case can be checked against the full surface. The check is sobering.

    - The full surface's fit, $H$ = {key['full_theta'][0]:.3f}, $\eta$ = {key['full_theta'][1]:.2f},
      $\rho$ = {key['full_theta'][2]:.2f}, is rejected by the front months alone: its chi-square on the 27 quotes is
      {key['full_chi2_front']:.1f}, against {key['ls_chi2']:.1f} at their own optimum.
    - The front-month fit is rejected by the other 36 quotes: chi-square {key['front_chi2_other']:.0f}, against
      {key['full_chi2_other']:.0f} for the full fit.
    - So the full surface prices the forward start {key['full_gap']:.0f}% above the front-month mark.

    One-factor rough Bergomi cannot fit the front months and the longer maturities with one parameter set. This is the
    known limit of the model's term structure (Rømer 2022; Abi Jaber and Li 2024). Within the model, the posterior
    gives the right range. The gap to the full surface is model error, and slot 6 deals with it.
    """)
    return


@app.cell(hide_code=True)
def _(F, data, plt):
    _fig, _ax = plt.subplots(figsize=(8.0, 3.8), constrained_layout=True)
    F.market_panel(_ax, data, 1, 1.0)
    _fig
    return


@app.cell(hide_code=True)
def _(key, mo):
    mo.md(rf"""
    ## 7 · What it costs, and where amortization comes in

    | Step | Cost here |
    |:---|:---|
    | Gauss–Newton: fit, Jacobian, product gradients | about 2 CPU-minutes |
    | Exact posterior by importance sampling | {key['proposals']:,} Monte Carlo surfaces in two rounds, about {key['cpu_hours']:.1f} CPU-hours |
    | The validation of section 5 | each chain's posterior by reweighting the same priced proposals |

    The first round of importance sampling, centred on Gauss–Newton, was unreliable: $\hat k$ =
    {key['khat_round1']:.2f}. The posterior is longer than the ellipse, so the ellipse is a poor proposal exactly where
    the posterior matters. An amortized network gives the proposal instead. The full bank prices all 63 quotes, so the
    same bank, restricted to the 27 front-month columns, trains a front-month network. That network gives each chain's
    posterior in about a second. The audit checks it with one round of importance sampling, and simulation-based
    calibration on thousands of simulated chains replaces the {key['replicas']} chains of section 5.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    1. **Where the quotes stop, the posterior's shape matters.** With the front months only, rough Bergomi's $H$ lies on
       a long, curved ridge. Gauss–Newton's ellipse does not follow it.
    2. **The reserve that matters is the one nobody computes.** Fixing $H$, the usual desk practice, removes most of the
       forward start's reserve. The posterior puts it back, and its coverage can be tested.
    3. **The posterior measures parameter uncertainty within the model, not model error.** The full SPY surface lies
       outside every front-month range. Slot 6 takes that up.
    """)
    return


if __name__ == "__main__":
    app.run()
