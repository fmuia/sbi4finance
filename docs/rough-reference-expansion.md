# Rough-Bergomi reference precision: one 4,096 → 8,192 follow-up

**Later held-out result:** [issue #23's separate study](rough-heldout-test.md)
has now evaluated the reserved cohort. Its reference passes, but one conditional
eta-spread gate fails; six of seven neural gates pass, including SBC and target
means. The candidate remains provisional and all 256 test cases are now used.
The report below preserves the development study and its then-reserved test status.

This bounded follow-up preserves the [first pilot](rough-joint-inference.md),
which stopped before neural training: 19/20 reference cases passed and
`validation:260` missed ESS and maximum-weight thresholds. The new
[protocol](../configs/rough_reference8192_protocol.json) is specified before
additional rows are generated. Its sole scientific change is the number of
independent prior-reference rows. The training bank, observation law, parameter
box, numerical grid/path counts, 20 cases, quote/target seeds, neural candidate
and every accuracy threshold remain unchanged. Already inspected audit cases
are development evidence, not fresh confirmation. The final test stays reserved.

## Ordered subplan

1. **Extend, without regenerating the prefix.** Keep the original bank intact.
   Reuse its 4,096 training/reference rows and two unused compatibility rows.
   Generate only `train:4096` through `train:8191`, using the same task and root
   seed with new stable IDs. A new bank manifest records the verified source
   bank and every suffix shard. No failed panel is redrawn.
2. **Check the expanded reference once.** Reweight all 8,192 rows for the same
   16 validation and four stress panels. Retain all eight thresholds, including
   ESS ≥800 and maximum weight ≤0.005, plus the existing stress-pricing screen.
   Compare the original and expanded diagnostics and report actual cost. Stop
   on failure; do not automatically double again or introduce a new proposal.
3. **Train only after reference acceptance.** If all reference checks pass,
   run the existing single 80-epoch spline-flow candidate on the unchanged
   1,024 training bases, with eight quote augmentations per base. Evaluate its
   conditional/joint accuracy, prior control, marginal SBC and target means
   using the fixed original settings. A checkpoint reload is not posterior
   acceptance. Leave any neural failure visible and the final test reserved.
4. **Teach and hand over the outcome.** Update notebook09 with the original
   stop, extension provenance, before/after precision and measured outcomes.
   Its default review remains offline and uses cached expensive calculations.

## Why more reference rows can help

A row is an independent parameter/pricing-randomness pair `(theta_i, U_i)`.
For a fixed nine-quote observation `y`, weights satisfy

\[
 w_i(y)\propto\exp\{-\|y-\widehat C(\theta_i,U_i)\|^2/(2\cdot0.25^2)\}.
\]

This integrates pricing randomness in the declared finite Monte Carlo model;
adding its saved covariance to the likelihood would count that noise twice.
Doubling the bank gives more opportunities to represent the small region favored
by the observation. ESS often grows approximately with bank size and Monte Carlo
error often decreases roughly as the square root of size, but neither improvement
is guaranteed for a realized bank. ESS is a concentration diagnostic, not proof
of posterior correctness. Conditional moments, block agreement and target-price
mean precision remain separate checks.

The two diagnostic blocks now contain 4,096 rows each: the original prefix and
the new suffix. Their underlying random streams are disjoint. Because this
extension was chosen after inspecting the prefix, its diagnostics are an adaptive
development check, not a new independent confirmatory experiment. Target prices
still use separate random numbers and estimate posterior mean expectation prices;
no distribution of noisy payoff averages is called a price posterior.

## Reproduction

Use the original pilot bank and the new protocol explicitly; the original
protocol and reports are never overwritten. Generate suffix chunks with
`configs/tasks/rough_reference8192_bank.json`, then assemble with the existing
CLI's `--base-bank` option:

```bash
PYTHONPATH=src python scripts/assemble_option_bank.py \
  --config configs/tasks/rough_reference8192_bank.json \
  --base-bank docs/results/rough-inference/banks/reference \
  --shards /path/to/new-suffix-shards --out /path/to/new-reference-bank
```

The reused bank and suffix must have identical numerical generation versions;
the assembler may run on another machine and records its own versions separately.
With the delivered banks, run the following from the repository root with the
pinned environment. Choose a fresh output directory; these commands refuse to
overwrite a completed study. The shell condition and trainer both enforce
reference acceptance before training.

```bash
rough_run=/path/to/new-rough-reference-review
if PYTHONPATH=src python scripts/check_rough_reference.py \
    --bank docs/results/rough-inference/banks/pilot \
    --reference-bank docs/results/rough-reference8192/banks/reference \
    --protocol configs/rough_reference8192_protocol.json \
    --out "$rough_run/reference"; then
  PYTHONPATH=src KERAS_BACKEND=jax python scripts/train_rough.py \
    --bank docs/results/rough-inference/banks/pilot \
    --reference "$rough_run/reference" \
    --protocol configs/rough_reference8192_protocol.json \
    --out "$rough_run/checkpoint" &&
  PYTHONPATH=src KERAS_BACKEND=jax python scripts/evaluate_rough.py \
    --bank docs/results/rough-inference/banks/pilot \
    --reference "$rough_run/reference" --checkpoint "$rough_run/checkpoint" \
    --protocol configs/rough_reference8192_protocol.json \
    --out "$rough_run/evaluation"
fi
```

Evaluation saves its diagnostics even when scientific gates fail; its nonzero
exit in that case is an intentional stop, not an instruction to tune or retry
the network. For the inexpensive didactic review, use
`pixi run review-rough-inference` instead.

This is synthetic Q evidence only. It does not resolve continuum discretization
error, real-market adequacy, rough-memory returns forecasting or curriculum
selection. No real market data or new storage framework is required.

## Measured bank delivery — 12 October 2026

The extension bank is delivered: **4,096 new reference rows plus the unchanged
4,096-row prefix**. The original training bank and two unused reference
compatibility rows are retained. This completes bank generation and transport;
it does not establish reference-posterior or neural acceptance.

The [cluster record](results/rough-reference8192/cluster.json) includes both
successful and unsuccessful task attempts:

| Quantity | Measured result |
|---|---:|
| First task start to last task end | 56.30 minutes |
| Original array | 109 completed tasks; 19 timed out |
| Retry of the same 19 task IDs | 19 completed tasks |
| Total task attempts / unique completed task IDs | 147 / 128 |
| Allocated CPU hours, all attempts | 37.8186 |
| Included allocated CPU hours from timeouts | 14.3925 |
| Maximum marginal price MCSE / screen | 0.01873845 / 0.025 currency |

The 19 timeouts were rerun with the same IDs, seeds and numerical settings;
incomplete outputs were archived and successful original chunks retained.
Both attempts count toward cost. Allocated CPU hours sum elapsed task time
times allocated CPUs, rather than measured CPU utilization. The wall span
excludes queueing before the first task; these figures exclude assembly,
reference checking and neural training. The MCSE screen controls estimated
sampling uncertainty, not time-discretization bias.

All reused prefix arrays are **bitwise identical**. Transfer archive hashes
match, and an [independent bank audit](results/rough-reference8192/independent-bank-audit.json)
checks source/protocol identities, every new parameter/quote stream, 8,192
distinct pricing seeds and unchanged scientific gates. It hashes the reserved
pilot test for immutability only; no test outcomes are evaluated.

A [full seeded replay of `train:4096` on the local Mac](results/rough-reference8192/local-replay.json)
agrees numerically with Linux generation: the H value differs by one ULP
(2.78e-17), and the largest price/observation difference is **1.24e-14 currency**.
The all-4,096-row parameter-stream audit finds a maximum difference of
2.22e-16. Cross-platform replay therefore uses explicit absolute tolerances
(1e-15 for parameters; 1e-10 for the full price replay), while stored-prefix
equality and artifact hashes remain exact. The initial bitwise parameter
comparison failed and is recorded. These checks establish numerical
portability at the stated tolerances; they neither relax the scientific gates
nor supply a reference or neural acceptance result.

## Measured reference result: all 20 cases pass

The [8,192-row reference report](results/rough-reference8192/reference.json)
passes every unchanged reference check on the same 16 validation and four
stress panels. The [original report](results/rough-inference/reference.json)
remains a 19/20 result. On its failing case, `validation:260`, ESS rises from
**600.55 to 1,207.19**, and the largest weight falls from **0.005392 to
0.002716**. Both now meet their original thresholds; no threshold or case was
replaced.

The table records the worst value across all 20 cases for each check:

| Reference check | Original 4,096 rows | Extended 8,192 rows | Unchanged gate |
|---|---:|---:|---:|
| Minimum importance ESS | 600.55 | 1,207.19 | ≥800 |
| Maximum normalized weight | 0.005392 | 0.002716 | ≤0.005 |
| Maximum parameter-mean MCSE / reference SD | 0.03664 | 0.02603 | ≤0.05 |
| Maximum absolute block mean difference / reference SD | 0.06107 | 0.12957 | ≤0.20 |
| Maximum absolute block relative SD difference | 0.05090 | 0.02319 | ≤0.20 |
| Maximum absolute block correlation difference | 0.02171 | 0.09549 | ≤0.20 |
| Maximum target-mean MCSE, currency | 0.003576 | 0.002512 | ≤0.02 |
| Maximum absolute block target-mean difference, currency | 0.004332 | 0.003351 | ≤0.06 |

The stress-panel pricing screen also passes: maximum MCSE is 0.01842469
currency against 0.025. Not every diagnostic improves monotonically: the
block mean and correlation differences increase, while remaining within their
gates. The original comparison used two 2,048-row blocks; the new comparison
uses the old and new 4,096-row blocks. This is a finite-bank precision check on
previously inspected development cases, not a fresh population-calibration
study or a proof of continuum/model correctness.

The local reference calculation took **821.26 seconds**, including **796.72
seconds** for independent target-price simulation and 24.43 seconds for input
handling and stress panels. These costs are separate from the cluster bank
generation above. Most of this run prices targets; it is not a timing of
importance reweighting alone. Reference passage permits the unchanged single
neural candidate to run. Neural accuracy, SBC and downstream agreement still
require their own measured results; the final test remains reserved.

### What the reference already says about learning H and η

Before assessing a network, inspect how much the declared observation can
teach us. For a uniform prior on `[a,b]`, prior SD is `(b-a)/sqrt(12)`.
Dividing each saved reference marginal SD by its corresponding prior SD gives:

| Parameter | Prior | Prior SD | Median reference posterior SD / prior SD |
|---|---|---:|---:|
| Roughness H | Uniform[0.1, 0.3] | 0.057735 | 0.99767 |
| Amplitude η | Uniform[0.4, 1.2] | 0.230940 | 0.57266 |

These medians are calculated from the 20 saved reference summaries, including
the four stress panels. H retains almost its entire prior marginal spread,
whereas η's spread narrows substantially. This behavior comes from the
numerical likelihood/reference calculation, **before any neural comparison**.
A network returning a broad H marginal therefore need not be copying the
prior: the panel may weakly identify H under this fixed context and noise
model. Similar SDs also do not imply identical distributions; means, shape
and H–η dependence can change. Judge the candidate against the conditional
reference and prior control together, rather than requiring contraction in
every parameter. These reference summaries alone do not establish neural
learning or general identifiability across other observation designs. The
separate neural evidence follows below.

## Measured neural result: all seven validation gates pass

After the reference passed, the unchanged single spline-flow candidate was
trained and evaluated. The [neural validation report](results/rough-reference8192/neural-validation.json)
passes all six conditional/price gates and the marginal-SBC gate. There was no
architecture sweep, seed search, threshold change or further bank doubling.
This is a **validation pass for the declared finite-Monte-Carlo observation
task**, on the recorded development/validation cases. It is not a reserved-test
result, a continuum-pricing certificate or an assessment against real markets.

| Neural validation check | Measured value | Unchanged gate |
|---|---:|---:|
| Pooled parameter-mean RMSE / reference SD | 0.11142 | ≤0.15 |
| Maximum absolute parameter-mean error / reference SD | 0.27892 | ≤0.40 |
| Maximum absolute relative SD error | 0.16624 | ≤0.25 |
| Maximum absolute H–η correlation error | 0.08399 | ≤0.10 |
| Largest absolute deviation of mean reference mass from 0.90, across parameters | 0.01095 | ≤0.03 |
| Maximum target-mean error plus 2 combined MCSE, currency | 0.02526 | ≤0.06 |
| Marginal SBC rank KS: H / η | 0.05741 / 0.04387 | Each ≤0.06542 |

The conditional comparisons use 4,096 paired neural draws for each of the same
20 cases. SBC uses 512 audit panels and 512 draws per panel; its DKW threshold
uses family α=0.05 across the two marginal rank checks. Passing these checks
does not guarantee every conditional regime or every aspect of the joint
posterior. The price gate concerns posterior **mean** numerical target prices
with sampling uncertainty, not posterior price quantiles or discretization bias.

### Is the network merely reproducing the prior?

The prior-only control also passes marginal SBC: its H/η KS statistics are
0.05007/0.04939 against the same 0.06542 limit. But its conditional mean RMSE
is **1.47263 reference SD**, compared with **0.11142** for the network. Thus
SBC alone cannot distinguish these two methods, while conditional comparisons
provide evidence that the network responds to the observed quotes. H's broad
marginal is consistent with the reference's weak H identification; η updating
and agreement with the conditional joint reference matter as well.

Inspect the [prior/reference/neural intervals](results/rough-reference8192/plots/posterior-intervals.png),
[conditional errors](results/rough-reference8192/plots/conditional-errors.png),
[SBC with prior control](results/rough-reference8192/plots/marginal-sbc.png),
and [target-mean errors](results/rough-reference8192/plots/target-mean-errors.png).
The [independent diagnostic audit](results/rough-reference8192/independent-neural-audit.json)
recomputes the saved case metrics, rank checks and prior control, agreeing on
all gates; the largest case-metric discrepancy is 7.11e-15. That audit does not
rerun the network or pricer, and the neural SBC coverage check uses saved counts
rather than unavailable individual SBC draws.

### Delivered checkpoint and measured local cost

The [training report](results/rough-reference8192/training.json) records one
80-epoch run on 1,024 independent training bases, each with eight quote-noise
augmentations. Training, including compilation, took **22.39 seconds** on the
local CPU. Checkpoint selection used the separate 256-panel selection set;
same-seed joint samples were identical after reloading the delivered
[model.keras](../checkpoints/rough-hurst-eta-reference8192-provisional/model.keras).
Its adjacent task, protocol and training metadata define compatibility.
Reload equality checks delivery, while the validation results above supply the
separate inference evidence.

Full validation took **222.38 seconds**: 2.576 seconds for conditional sampling
including compilation (20 cases × 4,096 draws), **215.63 seconds** for target
pricing and conditional checks, 1.82 seconds for SBC/prior controls, and 2.21
seconds for loading. This distinguishes fast posterior sampling from the
remaining target-pricing cost; it is not an end-to-end millisecond pricing
claim or a demonstrated speed advantage over the reusable reference bank.

## Verified offline review

Run `pixi run review-rough-inference` to open notebook09. Its default review
uses cached results and a small live scalar importance-sampling example. It
does not launch bank generation, training or final-test evaluation.

The [verification record](results/rough-reference8192/notebook-verification.json)
records strict marimo checking and an isolated headless export: **22 cells,
zero errors, 9.46 seconds**. External connections were blocked; market fixtures,
raw banks and checkpoints were absent. All eight rendered plots were visually
inspected. The [before/after reference graph](results/rough-reference8192/plots/reference-expansion.png)
shows all 20 cases under the unchanged gates. The standalone software suite
passes **125 tests**, including prefix reuse, exact coverage and rejection of
incompatible or overlapping chunks. Test success, notebook execution and
scientific validation remain separately recorded achievements.

## Narrow next scientific step

Review notebook09 and this candidate before deciding the tutorial curriculum.
The next confirmatory step is a **separately specified and frozen evaluation
on the reserved 256-case test split**, using the selected checkpoint and fixed
numerics, checks and reporting rules. That evaluation has **not been run**.
Retain the original failed reference study and this adaptive follow-up; do not
relabel their inspected cases as a fresh holdout or tune the candidate against
future test outcomes. No additional reference expansion or architecture tuning
is part of the completed follow-up.
