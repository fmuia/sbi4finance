# Bergomi price-grid refinement with paired paths

The reusable addition is a comparison function, not another simulator or bank
format. It calls the existing one-factor and rough Bergomi kernels on nested
grids and measures the uncertainty of their **difference**. Model parameters,
the quote law and existing bank files are unchanged.

Read the [frozen experiment](bergomi-refinement-protocol.md) for the fixed cases,
grids, seeds and stopping rule. [Notebook 08](../notebooks/08_bergomi_models.py)
adds a small live example, cached results, plots and review exercises. It runs
offline without market fixtures or expensive automatic jobs.

## Measured outcome

**The one-factor comparisons resolve within the budget. Rough Bergomi's
128-step comparison remains unresolved at its low-H/high-eta corner; its
512-to-1,024-step comparison resolves at every tested contract and case.**

The fixed study took **309.27 seconds** locally for both models' ten cases,
262,144 paths per case and four grids. The following values are the largest
\(|\widehat\Delta|+4\widehat{\mathrm{SE}}\) over 45 case/contract combinations
per model. The unchanged threshold is **0.025 currency units**.

| Grid comparison | One factor: largest envelope | Rough: largest envelope | Rough unresolved contracts/cases |
|---|---:|---:|---:|
| 128 → 256 | 0.00517 | 0.02524 | 1 / 45 |
| 256 → 512 | 0.00280 | 0.03066 | 2 / 45 |
| 512 → 1,024 | 0.00143 | 0.02011 | 0 / 45 |
| 128 → 1,024 | 0.00827 | 0.03839 | 3 / 45 |

All one-factor comparisons pass the finite-grid resolution screen. All rough
misses occur at \(H=0.1,\eta=1.2,\rho=-0.6\). In the worst 128-to-1,024
comparison, the strike-110 half-year put changes by **−0.01870 currency**, with
paired SE **0.004922**. Its envelope is **0.03839**. This is **unresolved against
the tolerance**, not proof that its true grid difference or continuum bias
exceeds 0.025. The adjacent rough differences are not uniformly monotone, so
this report does not fit or extrapolate a convergence rate.

The largest marginal price SE is **0.02195** for one factor and **0.01869** for
rough, both within the separate 0.025 sampling budget. For the 128-to-1,024
comparison, median independent/paired variance ratios are about **6,682** and
**135**, respectively. These ratios describe the precision of a grid comparison;
they are neither standalone price precision nor measured runtime speedups.

Inspect the [signed differences and error bars](results/bergomi-refinement-figures/coupled-grid-differences.png),
[pairing precision graph](results/bergomi-refinement-figures/pairing-precision.png),
and [complete report](results/bergomi-coupled-refinement.json). Parameter cases are
the Cartesian corner order from the task files followed by the center. Within
each group, the nine puts run through maturities 0.25, 0.5 and 1 year, each with
strikes 90, 100 and 110.

The result supports **512 steps as a candidate numerical setting for a new
restricted rough inference pilot**, with an explicitly declared finite-grid
observation law. It does not certify uniform continuum accuracy or retroactively
change the 128-step banks. The protocol's stop rule was applied: no extra seeds,
paths, grids, bank generation or neural training followed. Heston and HKDE
inference results and their reserved tests remain separate.

Eight new focused tests check marginal covariance, OU endpoints, rough
normalization/leverage, direct difference covariance, deterministic limits,
chunking and invalid inputs. **92 tests pass on PR #14's standalone stack**;
**112 pass when combined with PR #15's HKDE implementation**. Numerical source
files are identical in both worktrees. The measured run used source commit
`3b539bc`; source/config/protocol hashes in the report identify the executed
code. Wall time is a local study measurement, with software tests running
concurrently during part of the run, not a scaling benchmark.

The [notebook verification](results/bergomi-refinement-notebook.json) records a
strict marimo check and isolated offline export: **33 output cells, zero errors,
11.81 seconds**, without market data, checkpoints or HKDE inference files. Its
missing-report check also succeeds with an explicit pending message. The
rendered diagnostic figure was visually inspected. Review execution is a
separate check from the price-resolution results above.

## Why independent runs left a gap

The old corner/center checks used independent random paths at 128 and 512 steps.
The estimated changes reached 0.0309 currency for one-factor Bergomi and 0.0427
for rough Bergomi, but their sampling SEs were about 0.0190 and 0.0248. Those
comparisons could not resolve a 0.025 price-error budget.

Pairing gives each coarse/fine calculation the same independent path index. If
the two prices react similarly to its randomness, their difference has much
less noise:

\[
\operatorname{Var}(\bar Y_f-\bar Y_c)
=\frac{\operatorname{Var}(Y_f)+\operatorname{Var}(Y_c)
-2\operatorname{Cov}(Y_f,Y_c)}N.
\]

The sample must preserve both marginal laws. For example, simply subsampling a
rough hybrid path does not produce the hybrid approximation defined on the
coarse grid. A precise comparison of the wrong models would not answer the
refinement question.

## One factor: exact OU endpoints, approximate asset integrals

For a step \(h\), write the exact OU innovation as

\[
U_h=a_hZ+b_hB,\qquad
a_h=\frac{1-e^{-\kappa h}}{\kappa\sqrt h},\qquad
b_h^2=\frac{1-e^{-2\kappa h}}{2\kappa}-a_h^2,
\]

where \(Z,B\) are independent standard normals and \(\sqrt hZ\) is the
Brownian increment driving leverage. A coarse cell with \(m\) fine cells has

\[
U_{mh}=\sum_{j=0}^{m-1}e^{-\kappa h(m-1-j)}U_{h,j},\qquad
Z_c=\frac1{\sqrt m}\sum_{j=0}^{m-1}Z_j,
\]
\[
B_c=\frac{U_{mh}-a_{mh}Z_c}{b_{mh}}.
\]

This Gaussian projection makes \(Z_c,B_c\) independent standard normals and
reproduces fine factor values at every coarse endpoint. The coarse asset
integration still uses its own left-point variance values. Its option price
can therefore change even when the factor endpoints agree exactly. Inputs
where floating-point cancellation cannot resolve the OU residual are rejected.

## Rough model: preserve each discrete law

For each independent normal channel, use
\(Z_{c,i}=m^{-1/2}\sum_{j=0}^{m-1}Z_{f,mi+j}\). Coarse cells remain independent,
and channels remain independent. Then recompute the coarse hybrid weights,
convolution and discrete driver-variance normalization. The fine model does
the same at its own step size.

This couples two correctly distributed approximations. It does not identify
their grid-specific residual channels with exact shared continuum Brownian
bridges. Both kernel approximation and asset integration change across grids.
The underlying [hybrid-method paper](https://arxiv.org/abs/1507.03004) explains
the near-singularity treatment; the particular channel aggregation here is a
variance-reduction construction for this finite comparison.

## API and reproduction

`sbi4finance.bergomi_refinement.coupled_option_panels(...)` returns price means
and SEs with axes `[grid, contract]`, full mean covariance with axes
`[grid, contract, grid, contract]`, and adjacent/finest-grid differences with
their paired covariance. Difference moments are accumulated directly to avoid
subtracting nearly equal covariance entries. Chunking changes memory use,
not the prescribed random stream.

The price estimator is the existing bounded conditional-put estimator. Each
contract column shares paths with the others. The independent-run SE in the
report is a counterfactual calculated from the same sample's marginal
variances, not a separately executed benchmark. A variance-reduction ratio is
not an end-to-end runtime speedup, nor a reduction in an individual price's SE.

From the repository root:

```sh
PYTHONPATH=src pixi run python -m unittest discover -s tests -p test_bergomi_refinement.py -v
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pixi run bergomi-refinement --output /tmp/bergomi-coupled-refinement.json
PYTHONPATH=src pixi run python scripts/plot_bergomi_refinement.py --report /tmp/bergomi-coupled-refinement.json --out /tmp/bergomi-refinement-figures
PYTHONPATH=src pixi run review-bergomi
```

The full measured study is opt-in and refuses to overwrite an existing report.
Source hashes, task hashes, protocol, seeds and environment are saved with the
result. The numerical protocol was committed before execution. No training
bank, network candidate or held-out inference test is changed by this study.
