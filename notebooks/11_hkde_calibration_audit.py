import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import hashlib
    import io
    import json
    import math

    import marimo as mo
    import numpy as np

    ROOT = mo.notebook_dir().parent
    return ROOT, hashlib, io, json, math, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # 11. A calibration alarm: is the approximation at fault?

    **The question:** the corrected HKDE sampler and its numerical reference
    have almost the same departure from uniform ranks. Does that mean we
    should fix the sampler, change the gate, or investigate the experiment?

    The existing synthetic task infers long-run variance $\theta$ and jump
    intensity $\lambda$ from nine noisy option prices under the risk-neutral
    measure Q. Each saved panel has generating parameters known only for
    evaluation. Those truths never enter posterior conditioning.

    This **post-hoc audit of one saved cohort** separates an absolute
    simulation-based calibration (SBC) question from a paired approximation
    question. Allow about 15 minutes. Three cached graphs and their numerical
    results are verified before display. A tiny live calculation below shows
    rank moments; opening this notebook runs no inference, simulation bank,
    training, new panels, or random diagnostic controls.
    """)
    return


@app.cell
def _(ROOT, hashlib, io, json, np):
    audit_directory = ROOT / "docs/results/hkde-gate-audit"
    report = json.loads((audit_directory / "report.json").read_text())
    artifact_bytes = {}
    for _name in ("checks.npz", "oracle-rank-control.png", "paired-rank-errors.png", "conditional-cdf-errors.png"):
        _content = (audit_directory / _name).read_bytes()
        if hashlib.sha256(_content).hexdigest() != report["artifact_sha256"][_name]:
            raise ValueError(f"Audit artifact differs from report: {_name}")
        artifact_bytes[_name] = _content
    with np.load(io.BytesIO(artifact_bytes["checks.npz"]), allow_pickle=False) as _saved:
        checks = dict(_saved)
    gate = report["gate"]
    if checks["reference_pit_129"].shape != (gate["cases"], gate["coordinates"]):
        raise ValueError("Reference percentile array does not match the reported cohort")
    if report["scientific_acceptance_changed"] or report["new_model_panels"]:
        raise ValueError("This notebook requires the saved-cohort, no-acceptance-override audit")
    return artifact_bytes, audit_directory, checks, gate, report


@app.cell
def _(gate, math, mo, report):
    _bound = math.sqrt(math.log(2 * gate["coordinates"] / gate["family_alpha"]) / (2 * gate["cases"]))
    if not math.isclose(_bound, gate["dkw_bound"], rel_tol=0, abs_tol=1e-15):
        raise ValueError("Displayed DKW formula does not reproduce the saved gate")
    mo.vstack([
        mo.md(r"""
        ## Two questions, two comparisons

        **Absolute calibration:** across independently generated parameter–panel
        pairs, do posterior ranks look uniform? For one coordinate and case
        $i$, let $u_i=F_i(a_i^{\mathrm{true}})$ be the numerical reference
        posterior CDF evaluated at its generating truth. If we could draw
        $M=512$ independent samples from that posterior, the count below the
        truth would satisfy $R_i\mid u_i\sim\mathrm{Binomial}(M,u_i)$.
        Randomization gives $Z_i=(R_i+J_i)/(M+1)$, with independent
        $J_i\sim\mathrm{Uniform}(0,1)$.

        The empirical CDF **across cases** is
        $\widehat G_n(z)=n^{-1}\sum_i\mathbf1\{Z_i\le z\}$.
        The reported KS distance is
        $D=\sup_z|\widehat G_n(z)-z|$.
        This differs from each case's posterior CDF $F_i$.

        **Paired accuracy:** on the *same* panels, how different are the
        corrected estimates from the numerical reference? Nearly equal KS
        distances alone cannot establish this: two different distributions can
        be equally far from uniform. We also inspect paired ranks and CDF errors.

        With $n=256$ cases and $d=2$ coordinates, the Dvoretzky–Kiefer–Wolfowitz
        bound and a union bound give the frozen threshold
        $\epsilon=\sqrt{\log(2d/\alpha)/(2n)}$ at family $\alpha=0.05$.
        Under independent cases with genuinely uniform marginal ranks, this is
        a valid conservative rule; independence *between coordinates* is not
        required. Finite-pool SIR is an approximate sampler, so its uniformity
        is a diagnostic question rather than a guaranteed null assumption.
        """),
        mo.ui.table([
            {"Coordinate": ("θ", "λ")[_row["coordinate"]],
             "Saved SIR KS": round(_row["sir_ks"], 6),
             "Reference PIT KS": round(_row["reference_ks"], 6),
             "Frozen bound": round(_bound, 6)}
            for _row in report["conditional_oracle"]
        ], selection=None),
        mo.md(f"The reconstructed threshold is **{_bound:.8f}**. Its ideal-uniform per-coordinate "
              f"rejection probability is **{gate['ideal_uniform_coordinate_exact_rejection_probability']:.3%}**; "
              f"the two-coordinate union bound is **{gate['ideal_two_coordinate_rejection_upper_bound']:.3%}**. "
              "The arithmetic is not too strict for its stated null. What was too strong was treating a shared alarm as proof of a correction-specific defect."),
    ])
    return


@app.cell
def _(mo):
    reference_percentile = mo.ui.slider(start=0.01, stop=0.99, step=0.01, value=0.5,
                                        label="Hypothetical reference CDF at the truth, u")
    mo.vstack([
        mo.md(r"""
        ## Hold the cohort fixed: what would ideal rank sampling do?

        The cached control holds all 256 reference percentiles fixed. For each
        coordinate separately it generates 20,000 sets of binomial counts and
        independent jitter, using $M=512$. **These are not new panels or new
        cohorts.** The control asks whether ideal independent sampling from this
        numerical reference would reproduce the observed rank alarm. It cannot
        establish calibration across future cohorts.

        The live slider evaluates only the exact conditional moments:

        $$\mathrm E[Z-u\mid u]=\frac{1/2-u}{M+1},\qquad
          \mathrm{Var}(Z\mid u)=\frac{Mu(1-u)+1/12}{(M+1)^2}.$$

        The small conditional mean offset comes from randomized finite ranks.
        Under the ideal prior-predictive experiment the randomized ranks are
        uniform marginally. These formulas generate no random numbers and do
        not alter the cached study or its gate.
        """),
        reference_percentile,
    ])
    return (reference_percentile,)


@app.cell
def _(artifact_bytes, gate, math, mo, reference_percentile, report):
    _u = reference_percentile.value
    _m = gate["posterior_draws"]
    _mean = (0.5 - _u) / (_m + 1)
    _sd = math.sqrt((_m * _u * (1 - _u) + 1 / 12) / (_m + 1) ** 2)
    _theta = report["conditional_oracle"][0]
    mo.vstack([
        mo.md(f"**Live analytic calculation:** at $u={_u:.2f}$ and $M={_m}$, "
              f"the mean rank-minus-reference difference is **{_mean:.6f}** and its conditional standard deviation is **{_sd:.5f}**."),
        mo.image(artifact_bytes["oracle-rank-control.png"]),
        mo.md(f"For θ, **{_theta['conditional_iid_reference_gate_failure_fraction']:.2%}** of the ideal-reference "
              "rank controls fail the frozen gate on this same cohort. The saved corrected KS lies at about the "
              f"**{100 * _theta['sir_ks_percentile_in_control']:.1f}th percentile** of those controls. "
              "The near-equality of 0.10325 and 0.10349 therefore has a concrete interpretation: ideal reference sampling "
              "would almost always retain this particular cohort's θ alarm. This does not mean a correct method fails "
              "98.85% of the time across fresh cohorts."),
        mo.callout("The reference is numerical, not analytic truth. The control works one marginal at a time, "
                   "does not reproduce cross-parameter dependence, and cannot rule out errors shared by generation, pricing or the reference.", kind="info"),
    ])
    return


@app.cell
def _(artifact_bytes, mo, report):
    mo.vstack([
        mo.md(r"""
        ## Agreement is close, but SIR is not an exact posterior sampler

        Sampling-importance-resampling (SIR) draws 512 rows from a finite,
        likelihood-weighted proposal cloud. Conditional on that cloud, the
        resampling distribution is its empirical distribution. Pool error is
        shared by all resampled rows. The ideal-reference control omits this
        extra error, so it is **not a null model for finite-pool SIR**.

        The points below compare each saved randomized rank to the reference
        percentile of the *same* truth. The gray envelope is pointwise sampling
        variation for ideal-reference ranks; it is not a simultaneous acceptance
        band for the corrected algorithm.
        """),
        mo.image(artifact_bytes["paired-rank-errors.png"]),
        mo.ui.table([
            {"Coordinate": ("θ", "λ")[_row["coordinate"]],
             "Paired correlation": round(_row["sir_reference_correlation"], 4),
             "Mean SIR − reference": round(_row["sir_minus_reference_mean"], 5),
             "Saved SIR RMSE": round(_row["sir_minus_reference_rmse"], 5),
             "Ideal-reference root expected MSE": round(_row["iid_reference_root_expected_mse"], 5),
             "Ideal RMSE 97.5th percentile": round(_row["conditional_iid_reference_rmse"]["quantiles_025_50_975"][2], 5)}
            for _row in report["conditional_oracle"]
        ], selection=None),
        mo.md("Across all 256 cases, the paired correlations are about **0.997**, with small mean differences. "
              "But observed RMSE is about **0.0217–0.0221**, above the ideal-reference range "
              "(upper 97.5th percentiles about **0.0201**). The remaining scatter exceeds pure IID rank noise. "
              "Finite importance-pool error is an expected additional source; these aggregate ranks alone cannot "
              "fully separate that error from resampling or establish that it is the only source. "
              "The individual 256 calibration clouds and SIR draws were not saved."),
    ])
    return


@app.cell
def _(artifact_bytes, mo, report):
    mo.vstack([
        mo.md(r"""
        ## Remove resampling: inspect 16 saved weighted CDFs directly

        Separate conditional-check clouds were saved for the first 16 panels.
        Evaluate the weighted indicator
        $\widehat F_i(a_i^{\mathrm{true}})=\sum_kw_{ik}
        \mathbf1\{a_{ik}<a_i^{\mathrm{true}}\}$ and compare it to $u_i$.
        This uses the pool itself, with no SIR draw noise. Truth is used only to
        define this evaluation point; it was not available to the estimator.

        These clouds use different seeds from the calibration clouds above.
        Within each case, the 2,048-row pool is a subset of the 8,192-row pool;
        the two budgets are not independent replications. Error bars are twice
        an asymptotic plug-in self-normalized importance-sampling standard error,
        not guaranteed simultaneous confidence intervals.
        """),
        mo.image(artifact_bytes["conditional-cdf-errors.png"]),
        mo.ui.table([
            {"Proposals": _row["proposals"], "Cases": _row["cases"],
             "θ CDF RMSE": round(_row["rmse_cdf_error"][0], 5),
             "λ CDF RMSE": round(_row["rmse_cdf_error"][1], 5),
             "Largest absolute CDF error": round(max(_row["max_abs_cdf_error"]), 5)}
            for _row in report["saved_conditional_pool_cdf"]
        ], selection=None),
        mo.md("Errors shrink from roughly **0.009** RMSE at 2,048 proposals to **0.0035–0.0051** "
              "at 8,192. Their scale is broadly consistent with the displayed plug-in Monte Carlo errors. "
              "This supports useful conditional approximation on these 16 cases; it is not an all-cohort "
              "joint-distribution guarantee or a new pass criterion."),
    ])
    return


@app.cell
def _(mo, report):
    mo.vstack([
        mo.md(r"""
        ## What this audit changes—and what it does not

        The gate's formula, randomized-rank reconstruction, and saved hashes
        check out. Independently integrating the same saved likelihood grid
        reproduces the reference percentiles; comparing 65 and 129 points per
        parameter axis supplies a numerical sensitivity check. These checks
        strengthen the reference comparison without providing an independent
        model or pricer.

        **Conclusion:** the absolute θ alarm is shared by the numerical
        reference and is typical of ideal-reference rank sampling *conditional
        on this saved cohort*. Treating it as proof of neural-correction failure
        would overstate the evidence. Paired checks nevertheless reveal finite
        approximation error, so close KS values do not make SIR exact.

        We retain the historical failed flag. No gate is relaxed, no estimator
        is tuned, and notebook 10's participant execution is not enabled here.
        A decision to revise acceptance should be explicit and prospective,
        distinguish conditional approximation from population calibration, and
        state what additional evidence it requires. These results alone do not
        prove that a fresh cohort will pass or that the model fits real markets.
        """),
        mo.ui.table([
            {"Recorded status": "Historical test passed", "Value": report["historical_test_passed"]},
            {"Recorded status": "Scientific acceptance changed", "Value": report["scientific_acceptance_changed"]},
            {"Recorded status": "New model panels", "Value": report["new_model_panels"]},
            {"Recorded status": "New estimator proposals", "Value": report["new_estimator_proposals"]},
            {"Recorded status": "Training runs", "Value": report["training_runs"]},
        ], selection=None),
    ])
    return


@app.cell
def _(ROOT, audit_directory, mo):
    mo.md(f"""
    ## Review trail, reproduction and exercises

    - [Machine-readable audit report]({audit_directory / 'report.json'}) and [saved control arrays]({audit_directory / 'checks.npz'}).
    - [Audit script]({ROOT / 'scripts/audit_hkde_calibration_gate.py'}) and [unchanged historical test]({ROOT / 'docs/results/hkde-correction/test.json'}).
    - [Frozen boundary-correction protocol]({ROOT / 'docs/hkde-correction-boundary-protocol.md'}).
    - [Previous corrected-workflow notebook]({ROOT / 'notebooks/10_hkde_corrected_workflow.py'}).

    Open the cached lesson from the repository root:

    ```sh
    PYTHONPATH=src pixi run marimo edit notebooks/11_hkde_calibration_audit.py --no-token
    ```

    To reproduce the explicit diagnostic controls into a **separate directory**:

    ```sh
    PYTHONPATH=src pixi run python scripts/audit_hkde_calibration_gate.py --out /tmp/hkde-gate-audit-review
    ```

    That command runs the declared rank controls, not new model panels or
    neural inference. Compare its numerical report with the committed report;
    rendered image bytes can depend on plotting-library versions. The notebook
    verifies each displayed PNG and `checks.npz` against the hashes recorded in
    its loaded report before displaying them.

    1. Why can two posteriors have nearly identical KS distances from uniform
       yet differ materially on individual panels? Which graph addresses this?
    2. Move the percentile slider toward 0.01 or 0.99. Why does conditional rank
       variance fall? Why does a 98.85% failure rate for this fixed cohort not
       imply a 98.85% false-alarm rate over new cohorts?
    3. Which uncertainty disappears when we inspect a weighted CDF instead of
       SIR ranks? Which uncertainties remain, even with 8,192 proposals?
    """)
    return


if __name__ == "__main__":
    app.run()
