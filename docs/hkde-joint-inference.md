# Restricted HKDE inference: first baseline and measured outcome

The option bank now supports a complete local experiment: synthetic quotes →
joint Q parameter inference → posterior model-price distributions. The numerical
reference passes its refinement checks. The first neural checkpoint learns
substantially from the quotes and passes the declared parameter and marginal
SBC checks, but **fails its downstream price-quantile gate**. It remains
provisional; the reserved final test has not been evaluated.

This completes the first bounded M5.1 implementation/review experiment, not
M5.1 scientific acceptance, a learned pricing surrogate or M8 delivery. The
[frozen protocol](hkde-inference-protocol.md) was committed before inspecting
results. Its one-run stop rule was applied. The final tutorial application
remains open, and the Heston and Bergomi findings are preserved.

## Inspect the result

Open [notebook 06](../notebooks/06_hkde_and_pricing.py) with
`PYTHONPATH=src pixi run review-hkde`. Its new sections start at validation case
257, which exposes the price discrepancy, and let you change a quote to see
the joint grid posterior update. Cached prices make that calculation small;
opening the notebook never launches training, bank generation or cluster work.

The main graphs are also available without running marimo:

- [Prior-to-posterior updating](results/hkde-inference/figures/prior-to-posterior.png).
- [SBC rank diagnostics](results/hkde-inference/figures/sbc-ranks.png).
- [Conditional target-price errors](results/hkde-inference/figures/target-price-errors.png).
- [Joint posterior for case 257](results/hkde-inference/figures/joint-posterior-case-257.png).

Exact records: [reference](results/hkde-inference/reference.json),
[training](results/hkde-inference/training.json),
[neural validation](results/hkde-inference/neural-validation.json), and
[checkpoint card](../checkpoints/hkde-theta-intensity-provisional/README.md).

## What was inferred

The two targets are long-run variance `theta` and jump intensity `lambda`.
The observation contains nine noisy European put prices; all other HKDE
parameters, initial variance, contracts, spot and carry are fixed known Q
context. The independent uniform prior is `[0.025,0.065] × [0.25,1.0]`, and
quote errors are independent Gaussian with SD 0.25 currency units.

The numerical Fourier price map uses 1,024 nodes and cutoff 400. A normalized
two-dimensional posterior grid evaluates the corresponding Gaussian likelihood.
Refining 65 to 129 points per axis passes every declared reference check:

| Largest change over 20 cases | Measured | Limit |
|---|---:|---:|
| Parameter mean, fine-reference SD units | 0.000774 | 0.01 |
| Relative parameter SD | 0.00263 | 0.01 |
| Correlation | 0.00199 | 0.01 |
| Downstream price quantile, currency | 0.00127 | 0.0025 |
| Log evidence | 0.000516 | 0.001 |

The separate 2,048-node/cutoff-800 price check at the four prior corners and
center changes prices by at most 3.32e-9 of quote sigma, including the two
target contracts. These are numerical comparisons on a declared domain,
not a rigorous uniform continuum error bound.

## What the neural checks show

The spline flow uses all nine ordered noisy prices directly, with logit
parameter transforms and training-only global normalization. It trained for
the fixed 80 epochs on 8,192 panels; 256 separate validation panels selected
the checkpoint. Another 512 validation panels supply the diagnostic audit.
Sixteen of these plus four predeclared near-corner cases supply conditional
references. Corner cases are excluded from SBC.

| Conditional validation quantity | Measured | Limit | Result |
|---|---:|---:|---|
| Mean RMSE, reference SD units | 0.123 | 0.15 | Pass |
| Largest absolute mean error, reference SD units | 0.343 | 0.40 | Pass |
| Largest relative SD error | 0.125 | 0.25 | Pass |
| Largest correlation error | 0.0488 | 0.10 | Pass |
| Largest target-price quantile error, currency | **0.0439** | **0.03** | **Fail** |
| Largest error in mean 90% interval reference mass | 0.00675 | 0.03 | Pass |

The exact prior's conditional mean RMSE is **2.09 reference SD**, compared
with **0.123** for the network. Across these cases, median reference posterior
SD is 56% of prior SD for `theta` and 72% for intensity. The posterior parameter
correlations range from approximately −0.94 to −0.51. There is substantial
updating and a strong tradeoff between variance and jump intensity; this is
not a reproduction of the independent prior.

**SBC is included.** On 512 prior-predictive audit cases, randomized marginal
rank ECDF deviations are 0.0319 and 0.0305, below the two-target DKW threshold
0.0654. Nominal 90% interval coverage is 92.8% and 92.4%. The individual Wilson
intervals are roughly `[90.2%,94.7%]` and `[89.8%,94.4%]`; these suggest mildly
conservative intervals and are not simultaneous guarantees. The rank check
passes, but it neither certifies every conditional posterior nor overrides the
price-distribution failure.

## The price discrepancy in financial units

The two targets are puts at `(strike 95, maturity 0.75 years)` and
`(strike 80, maturity 1 year)`, excluded from conditioning. Matched parameter
draws are priced using the reference engine; no surrogate is involved.

For validation case 257, the first contract's reference 5th-percentile price
is **4.69582**. The network gives **4.65195**, an error of **−0.04387 currency**
(about 0.93% of that reference price). Its median also misses the 0.03 gate,
by an error of about −0.03309. Other cases' largest quantile errors remain
below the gate. The original thresholds are unchanged.

The individual 95% order-statistic interval for the worst neural quantile
corresponds to an error interval of **[−0.04884,−0.03657]**, entirely beyond
−0.03. This is sampling uncertainty for the fixed neural distribution, not
a confidence interval for model adequacy. The reference refinement error is
much smaller; 32 independent reference-draw repetitions give a 97.5th-percentile
maximum price-quantile sampling error of about 0.0103. The observed miss is
not explained by ordinary 4,096-draw variation alone.

This illustrates why a price-sensitive check accompanies parameter moments,
dependence and SBC. A modest shift along a joint parameter distribution can
matter at an output tolerance that its separate parameter checks permit.

## Cost and scope

| Calculation | Measured wall time |
|---|---:|
| Generate/write the full 9,216-panel bank | 31.33 s |
| Selected-ID bank verification | 0.063 s |
| Build/check both reference grids and fixed cases | 22.20 s |
| Train the single 80-epoch baseline, including compilation | 22.73 s |
| Draw 4,096 joint samples for each of 20 cases, including compilation | 2.23 s |
| Directly price those samples and compute conditional comparisons | 23.78 s |
| Draw/check 512 samples for each of 512 SBC cases | 1.73 s |

These are CPU measurements from this local execution, not isolated throughput
benchmarks. The reference-grid price matrix is reusable across observations;
this deliberately tractable example makes no claim that neural inference is
faster than its grid reference. The checkpoint is about 0.94 MB. Same-seed
draws match exactly after reload.

## Reproduce

Use the pinned environment from the repository root. Each output must be a
new directory; the following commands are opt-in, not notebook startup work.

```bash
PYTHONPATH=src pixi run python scripts/generate_option_bank.py \
  --config configs/tasks/hkde_inference_bank.json --out /tmp/hkde-bank-new
PYTHONPATH=src pixi run python scripts/check_hkde_reference.py --out /tmp/hkde-reference-new
PYTHONPATH=src pixi run python scripts/train_hkde.py \
  --bank /tmp/hkde-bank-new --reference /tmp/hkde-reference-new --out /tmp/hkde-checkpoint-new
PYTHONPATH=src pixi run python scripts/evaluate_hkde.py \
  --bank /tmp/hkde-bank-new --reference /tmp/hkde-reference-new \
  --checkpoint /tmp/hkde-checkpoint-new --out /tmp/hkde-validation-new
```

The evaluation deliberately returns a nonzero exit on failed scientific gates
after saving its report. `--split test` additionally requires a matching passed
validation report; the saved failed candidate cannot unlock it. The raw bank
is regenerated from its configuration instead of being committed. Small
reference/check arrays, checkpoint, reports and figures are retained for review.

Source hashes identify the actual recorded execution. Its code is preserved
in commit `45d94fb`; subsequent script-only provenance guards reject stale
reference sources and mismatched training/validation identities. Those guards
do not alter the computation or rewrite its original reports. See
[run provenance](results/hkde-inference/run-provenance.json).

## Remaining work

Keep the first candidate provisional and the 256 final-test panels unevaluated.
Record any subsequent bounded experiment separately, retaining this result
and the original gate. The existing grid is already a usable reference
calibration/pricing building block. M5.2 surrogate work has separate accuracy
checks; it must not hide the current inference discrepancy. M8 still requires
an accepted, paced participant workflow. Neither optional real data nor another
round of Heston/Bergomi development is needed to inspect this synthetic result.
