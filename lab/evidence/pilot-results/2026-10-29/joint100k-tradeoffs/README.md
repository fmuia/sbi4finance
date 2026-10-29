# Joint100k: parameter tradeoffs in saved validation posteriors

The learned six-parameter posterior has substantial parameter tradeoffs. Two
examples meet an explicit interior criterion in **all six dimensions**; one
shows a curved long-level/decay ridge and a skewed long-level marginal. A
separate near-flat-curve example illustrates weak decay identification. These
are exploratory diagnostics of the amortized estimator, pending an independent
conditional posterior reference. They do not establish the geometry of the true
posterior or demonstrate multiple modes.

This analysis uses the existing 2,000 validation cases, with 1,000 saved draws
per case. It generates no posterior/prior samples, simulations or MCMC runs.
The reserved test split is untouched. The full six-dimensional parameter vectors
are retained for the figures.

## Model and source identities

The independent physical uniform priors are H in [0.02, 0.35], eta in [0.5, 3.5],
rho in [-0.99, -0.20], short variance S and long variance L each in [0.01, 0.16],
and decay kappa in [0.25, 8] per year. The generator imposes no rising/falling
curve constraint. Parameter ordering in every artifact is H, eta, rho, S, L,
kappa.

The joint model uses `xi(t) = L + (S - L) exp(-kappa t)` on the simulator's fine
time grid, with t in years. Its integrated variance is
`W(T) = L T + (S - L) (1 - exp(-kappa T)) / kappa`.
The observations are option quotes at maturities 5, 10, 21, 42, 63, 126 and 252
trading days (252 days/year), rather than observations of this curve. The seven
auxiliary curve values stored by the task are derived interval averages,
not extra observed conditioning in joint inference.

- Model checkpoint SHA256:
  `19c24d64290651f592d3c13d7d0a1616d0d8ae080d487d0298aee8f97779498b`.
- Bank manifest SHA256:
  `eb72588b9b7665e0e416cb5d3a1bcf8cd78a1c97f2b43f18ea8246c749a871d8`.
- Saved panel SHA256:
  `d7b6e67f74228aa15de413fd0a2802cbe0c7a586888853ce3ef4403856ee422d`.

The input task and training/evaluation record hashes, analysis source hash and
runtime versions are recorded in [tradeoffs.json](tradeoffs.json). The local calculation took
about 1.2 seconds with Python 3.12.16, NumPy 2.5.3 and SciPy 1.18.1 on arm64.

## Tradeoffs across the validation panel

Each correlation is calculated **within one observed case**; the table
summarizes those 2,000 correlations. Pooling posterior draws across cases would
answer a different question and can hide the conditional geometry.

| Pair | Median Pearson | Median Spearman | Median absolute Spearman | Cases with absolute Spearman >= 0.8 |
| --- | ---: | ---: | ---: | ---: |
| L / kappa | 0.012 | 0.008 | 0.846 | 62.75% |
| eta / rho | 0.508 | 0.497 | 0.601 | 24.20% |
| H / eta | 0.237 | 0.226 | 0.445 | 11.10% |

The near-zero signed median for L/kappa hides strong associations of opposite
signs. Splitting descriptively by the **true** L-S, with an absolute contrast of
0.005 defining the near-flat group:

| True curve group | Cases | Median within-case L/kappa Spearman |
| --- | ---: | ---: |
| Rising, L-S > 0.005 | 951 | -0.872 |
| Falling, L-S < -0.005 | 934 | +0.840 |
| Near-flat, absolute L-S <= 0.005 | 115 | +0.124 |

This is structurally plausible. The initial curve slope is
`b = kappa (L - S)`. Keeping S and b fixed gives `L = S + b / kappa`: a negative
L/kappa association for rising curves and a positive one for falling curves.
This calculation explains an available compensation mechanism; the option
likelihood also depends on the full curve and the other model parameters.

For eta/rho, compensating movements can preserve their product while each
coefficient changes. Case 1941 below makes that distinction visible. Its
product uncertainty is much smaller relative to its own prior than either
coefficient's uncertainty. The product's prior is not uniform, and the linked
contraction audit uses its correct analytic moments.

## Exploratory case selection and interior checks

The [selection declaration](selection-declaration.json) was fixed before
inspecting candidate rankings. All six true parameters must be at least 10%
of their prior width from either endpoint. Every marginal central 98% interval
must stay at least 5% from either endpoint, and every central 99% interval must
stay at least 2.5% away. Exactly **122/2,000 cases (6.1%)** qualify.

Among those cases, 1983 maximizes absolute Spearman dependence among the three
curve-parameter pairs; 1941 maximizes absolute eta/rho Spearman among the
remaining cases. These are selected teaching examples, not representative
random cases. No criterion was relaxed and truth coverage was not used for
selection. The interval criterion does not claim every individual tail draw is
equally far from the bounds.

| Case | Role | Smallest truth margin | Smallest central-98% margin | Smallest central-99% margin | Main within-case dependence |
| --- | --- | ---: | ---: | ---: | --- |
| validation:1983 | Interior, falling curve | 14.30% | 6.89% | 4.89% | L/kappa Pearson 0.935, Spearman 0.975 |
| validation:1941 | Interior, rising curve | 11.04% | 7.68% | 6.30% | eta/rho Pearson 0.957, Spearman 0.957 |
| validation:1962 | Separate near-flat, weak-decay case | 24.16% | 0.48% | 0.22% | **Fails the posterior interior criterion** |

Margins are the smallest distance from either prior endpoint over all six
dimensions, expressed as a fraction of the corresponding full prior width.

Case 1983 is a useful candidate for geometry beyond an ellipse: its physical
L marginal has skewness -1.305, and a residualized quadratic predictor adds
0.073 R-squared for L/kappa beyond a linear fit. The same descriptive statistic
is 0.154 after transforming the physical prior fractions to logits, so the
observed curvature does not disappear under that transform. Case 1941's
eta/rho relation is mostly linear in physical coordinates: strong correlation
alone does not establish a non-Gaussian posterior. No multimodality claim is
made for either case. These skewness/curvature measures are descriptive
in-sample diagnostics, not hypothesis tests or acceptance criteria.

For transparency, these are the empirical percentiles of the truth within each
saved posterior marginal. All six truths are inside their marginal central
90% intervals in all three examples; this post-selection observation does not
establish conditional coverage.

| Case | H | eta | rho | S | L | kappa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1983 | 31.9 | 68.2 | 64.4 | 79.5 | 92.3 | 86.8 |
| 1941 | 51.9 | 15.1 | 13.4 | 72.2 | 30.0 | 58.0 |
| 1962 | 80.3 | 27.4 | 21.1 | 64.6 | 93.8 | 25.0 |

## What the flat-curve example teaches

Case 1962 has true S=0.09163 and L=0.08968. The derivative
`d xi(t) / d kappa = -t (S-L) exp(-kappa t)` is small when these levels are
close. If S=L exactly and those levels are held fixed, the likelihood is
independent of kappa. After integrating uncertain levels, the marginal kappa
posterior need not equal its prior because the available parameter volume can
depend on kappa.

Here kappa is broad: its SD is 1.022 times its prior SD and its central 90%
width is 1.047 times its prior central 90% width. Nevertheless the inferred
curve function can be precise. Its W(1) SD is only 0.0341 times its corresponding
prior SD, and xi(1)'s ratio is 0.0464. In case 1983, L and kappa SD ratios are
0.222 and 0.185 while W(1)'s is 0.0296. In case 1941, eta and rho ratios are
0.0605 and 0.1817 while rho*eta's is 0.0181. See the independent
[case contraction calculation](../joint100k-contraction-audit/illustrative-case-contraction.json)
and [full audit](../joint100k-contraction-audit/README.md), which use analytic
prior variances without new prior draws.

Thus a broad coefficient need not imply a poorly determined economically
relevant curve. Case 1962 remains explicitly separate from the interior
geometry showcase because its posterior reaches close to the prior boundaries.

## Reuse

- [tradeoffs.json](tradeoffs.json): all 15 pair distributions, orientation
  strata, candidate metrics, truth percentiles and input/source hashes.
- [within-case-metrics.npz](within-case-metrics.npz): all case-specific Pearson,
  Spearman and directional quadratic statistics, plus the strict-interior mask.
- [candidate-1983.npz](candidate-1983.npz) and
  [candidate-1941.npz](candidate-1941.npz): all 1,000 original six-dimensional
  parameter draws and truth, physical bounds and names, derived curve draws,
  observed maturities, W(T)/T, interval-average variance, and rho*eta. The
  derived curves are functions of the parameter draws, not extra model draws.
- [weak-decay-1962.npz](weak-decay-1962.npz): all 1,000 original parameter draws,
  truth, bounds/names, and model/bank identities for the separate weak example.
- [analyze_tradeoffs.py](analyze_tradeoffs.py): deterministic analysis; invoke
  with `--panel-dir` pointing to the verified saved panel and `--out` to an
  output directory. It checks exact panel identity and preserves existing
  candidate NPZ bytes when arrays agree.
- [sha256sums.txt](sha256sums.txt): artifact checksums.

Figures should show every parameter, all six true values, and the physical
prior marginals. Compare curve-function uncertainty as well as coefficient
uncertainty. An independent conditional reference remains the next scientific
check before treating a selected learned geometry as a tutorial conclusion.
