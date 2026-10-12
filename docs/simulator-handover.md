# Collaborator handover: finance simulator ladder

**Snapshot: 12 October 2026.** Repository: [fmuia/sbi4finance](https://github.com/fmuia/sbi4finance).
The simulator/bank building blocks are substantially implemented. The final
participant tutorial is **not yet scientifically accepted**. This note hands
over code, evidence and decisions; it does not claim that every model has a
validated neural posterior. No PR was merged in this workstream.

## Start here

For a module map, data-flow diagrams and extension points, read the companion
[code architecture guide](code-architecture.md). It describes the implemented
layout and distinguishes additions that still live on separate branches.

1. Read the [plan and mathematical reading guide](https://github.com/fmuia/sbi4finance/blob/docs/simulator-plan/docs/simulator-plan/README.md)
   in [plan PR #3](https://github.com/fmuia/sbi4finance/pull/3).
2. Open [notebook10](../notebooks/10_hkde_corrected_workflow.py) and the
   [HKDE correction report](hkde-corrected-inference.md). These show the most
   recent result, including the retained acceptance failure.
3. Inspect the [price-error graph](results/hkde-correction/plots/price-errors.png)
   and [reference-versus-SIR rank graph](results/hkde-correction/plots/reference-pit.png).
4. For Bergomi, use the latest [PR #14 study](https://github.com/fmuia/sbi4finance/blob/feat/rough-bergomi-option-bank/docs/rough-joint-inference.md)
   and its notebook09. The latest rough-inference additions are not all in
   the HKDE/tutorial branch ancestry; see the branch boundary below.

The project should stay easy to change. Prefer the existing functions, explicit
task dictionaries, NPZ/JSON banks, Pixi environment and thin cluster scripts.
No additional registry, service, storage framework or general-purpose trainer
is needed to continue this tutorial work.

## What we are building, and what is available

The original objective was reusable simulation building blocks, pre-simulated
training banks and cluster generation for a finance SBI tutorial. We preserved
two distinct applications: **P returns inference and conditional forecasting**,
and **Q option calibration and downstream model pricing**. Choosing the final
running example remains a curriculum decision. The existing MMAR example is
preserved. This work is not a reproduction/import of Francesco's full SVJ study.

| Component | Delivered | Acceptance boundary |
|---|---|---|
| M0–M2: conventions, GBM, banks, cluster | Exact GBM simulation, analytic inference/prediction references, bank generation/loading, training/reload, cluster smoke and notebook01 | The small analytic task is the foundation; its success does not validate other models. |
| M3: Heston and Gaussian price jumps | Reference/Numba trajectory kernels, continuation, numerical checks, banks, joint parameter/state inference and notebooks04–05 | Restricted Heston neural candidates learn from the data but retain conditional failures. Full SVJ/SVCJ inference is not accepted. |
| M4: HKDE and option panels | Double-exponential price jumps, direct Fourier European pricing, fixed noisy-panel tasks, common option banks and notebook06 | This uses numerical Fourier quadrature, not an imported PROJ implementation. Wider domains and American/exotic products need separate work. |
| M5: HKDE inference and learned pricing | Frozen neural checkpoint, refined numerical posterior grid, two-contract price surrogate, importance correction and notebook07 | The surrogate passes. The raw flow and corrected pipeline remain provisional for different recorded reasons. |
| M6: one-factor Bergomi | Forward-variance context, exact OU/asset covariance, complete continuation, European Monte Carlo prices, banks and notebook08 | No two-factor implementation or accepted neural posterior is claimed. |
| M7: rough Bergomi | Gaussian reference and FFT hybrid driver, option banks, coupled-grid study, restricted H/eta reference pilot and notebook09 | Reference passes 19/20 panels; one efficiency failure stops training/SBC. No accepted rough checkpoint. |
| M8: paced options candidate | Notebook10, equations, diagnostics, exercises and verified offline review | Draft: the failed held-out calibration gate disables live participant inference. |

Numba accelerates implemented trajectory loops; independent/reference paths
remain available for checks. Vectorized numerical work and the rough FFT
driver use their appropriate existing implementations. There is no requirement
to rewrite every simulator in Numba or Cython.

## Code and statistical boundaries

| Responsibility | Entry points |
|---|---|
| Models | `src/sbi4finance/{gbm,heston,hkde,bergomi,rough_bergomi}.py` |
| Returns banks/inference | `gbm_bank.py`, `gbm_inference.py`, `heston_bank.py`, `heston_inference.py`, `heston_reference.py`, `heston_forecast.py` |
| Shared option observations/banks | `option_panel.py`, `option_bank.py`, `configs/tasks/*.json`, `scripts/generate_option_bank.py` |
| HKDE inverse/forward calculations | `hkde_inference.py`, `hkde_reference.py`, `hkde_surrogate.py`, `hkde_correction.py` |
| Cluster | `cluster/simulate.sbatch`, `cluster/option_bank.sbatch`; latest rough array workflow is in PR #14 |
| Evidence | `docs/results/`, checkpoint metadata, frozen `configs/*protocol*.json`, milestone review notebooks |

A returns observation is an ordered decimal log-return array
`[batch,time,1]` with positive year-fraction time increments. Continuation must
retain the model's required state and its dependence with parameters. For the
full-truncation Heston task this includes its declared signed auxiliary state;
rough memory cannot be replaced by current scalar variance.

An options observation is an ordered noisy premium panel
`[batch,contract,1]`, with contracts, carry, spot and other known model inputs
as context. The fixed HKDE task has nine European puts, quote-noise SD 0.25,
and two unknowns: long-run variance and jump intensity. Bergomi additionally
requires an explicit forward-variance curve. Changing the observation design,
prior or context is a new compatibility question, not an array relabelling.

The pricing workflow is:

> parameters + contracts → numerical expectation prices → synthetic noisy panel
> → joint parameter posterior approximation → target-price distribution.

SBI learns the inverse distribution. Numerical pricers provide the generative
map and references. A separately validated forward surrogate can replace
supported target-price evaluations. In the current correction, all nine
likelihood prices still use the numerical pricer; the surrogate replaces only
the two excluded target puts. A single simulated payoff is not a price label.
Q calibration supplies neither tomorrow's market chain nor P risk probabilities.

Bank manifests bind the task, model/measure, parameter order, numerical law,
source/version hashes and seed/ID streams. Observations, synthetic truth,
reference prices and numerical diagnostics have separate roles. Preserve
independent base draws/splits and group all augmentations of a surface together.
Do not replace failed numerical draws silently or feed simulator-only labels
into conditioning. [Bank details and commands](option-banks.md) are the contract.

## Where the work lives

The implementation PRs use **N/10-simulator-plan-…**. Numbering describes planned
scope; it is not necessarily Git dependency order. The actual PR bases determine
review/integration order. The plan itself is outside that count.

Remote state was checked for this note: `main` is still `b892fae`, containing
the merged scaffold PR #2. The following implementation PRs are **open**;
only #19 is draft. A collaborator starting from `main` will not find this
implementation stack there yet.

| Increment / subject | PR | Actual base |
|---|---|---|
| 1/10 — GBM reference | [#7](https://github.com/fmuia/sbi4finance/pull/7) | Plan #3 branch |
| 2/10 — GBM bank/training | [#8](https://github.com/fmuia/sbi4finance/pull/8) | #7 branch |
| 3/10 — CPU cluster smoke | [#9](https://github.com/fmuia/sbi4finance/pull/9) | #8 branch |
| 4/10 — Heston/SVJ kernels | [#10](https://github.com/fmuia/sbi4finance/pull/10) | #9 branch |
| 5/10 — Heston joint inference | [#11](https://github.com/fmuia/sbi4finance/pull/11) | #10 branch |
| 6/10 — HKDE option banks | [#12](https://github.com/fmuia/sbi4finance/pull/12) | #11 branch |
| 8/10 — one-factor Bergomi | [#13](https://github.com/fmuia/sbi4finance/pull/13) | #12 branch |
| 9/10 — rough Bergomi/inference | [#14](https://github.com/fmuia/sbi4finance/pull/14) | #13 branch |
| 7/10 — HKDE inference/pricing | [#15](https://github.com/fmuia/sbi4finance/pull/15) | #14 branch |
| 10/10 — offline options tutorial, **draft** | [#19](https://github.com/fmuia/sbi4finance/pull/19) | #15 branch |

The latest important heads at this snapshot are:

| PR | Branch | Reviewed head | Role |
|---|---|---|---|
| [#3](https://github.com/fmuia/sbi4finance/pull/3) | `docs/simulator-plan` | `30982c4` | Authoritative roadmap/subplans |
| [#14](https://github.com/fmuia/sbi4finance/pull/14) | `feat/rough-bergomi-option-bank` | `073ca8c` | Latest rough banks, refinement and stopped reference pilot |
| [#15](https://github.com/fmuia/sbi4finance/pull/15) | `feat/hkde-inference` | `60ebe4d` | HKDE posterior, surrogate and correction evidence; PR base is #14's branch |
| [#19](https://github.com/fmuia/sbi4finance/pull/19), **draft** | `feat/hkde-offline-tutorial` | `a21d019` before this documentation addition | M8 review; PR base is #15's branch |

**Branch caveat:** a PR targeting another branch does not automatically contain
that branch's later commits. PR #19 is a good checkout for the HKDE review;
PR #14 contains the latest rough-inference work. The non-mutating combined
snapshot passed **171 tests** and combined without conflicts. It was not a
merge, and no single published head should be advertised as that tested combined
release. See the exact [heads/tree verification](results/hkde-correction/combined-stack-verification.json).
Recheck composition and tests when integrating the PRs.
The plan-only branch also has later status updates absent from #19's embedded
plan snapshots: use plan PR #3 for the authoritative current roadmap.

The separate data workstream, described by [plan PR #4](https://github.com/fmuia/sbi4finance/pull/4), supplies
local CSV/JSON roles, return-window preparation and the fixed HKDE panel handoff.
Its later [bank integration, PR #17](https://github.com/fmuia/sbi4finance/pull/17)
and [cached walkthrough](https://github.com/fmuia/sbi4finance/blob/feat/hkde-offline-walkthrough/docs/hkde-offline-teaching.md)
exercise the actual bank/adapter/checkpoint boundary. That walkthrough preserves
the **original raw-flow** discrepancy; it is not the corrected pipeline in
notebook10. IO compatibility and successful replay do not establish scientific
acceptance. Coordinate integration; do not invent a competing data schema.
Its separate open Git stack is [#5 fixtures](https://github.com/fmuia/sbi4finance/pull/5)
→ [#6 return windows](https://github.com/fmuia/sbi4finance/pull/6)
→ [#16 HKDE panels](https://github.com/fmuia/sbi4finance/pull/16)
→ [#17 bank integration](https://github.com/fmuia/sbi4finance/pull/17)
→ [#18 cached teaching](https://github.com/fmuia/sbi4finance/pull/18), starting
from `main`. Those commits are not part of the #19 checkout or the recorded
171-test simulator snapshot.

## First review session

In a dedicated checkout of `feat/hkde-offline-tutorial`, use the locked
environment. Installation needs package access; execution of the review needs
no market files or provider credentials.

```bash
pixi install --locked
PYTHONPATH=src pixi run review-hkde-corrected
PYTHONPATH=src pixi run review-hkde-surrogate
PYTHONPATH=src pixi run test-simulators
```

Use `PYTHONPATH=src`, especially if sharing an editable environment between
worktrees. Opening these reviews starts no training, bank generation or
cluster job. Notebook10 currently shows cached scientific evidence and
explicitly skips live inference. Its isolated export passed **17 cells, zero
errors, 8.29 seconds**, with market fixtures/raw banks absent and external
connections blocked. The existing pinned environment was reused; this is not
a fresh-install test or a live-inference timing. The [verification record](results/hkde-correction/notebook-verification.json)
binds the sources/assets and records `live_executed=false`.

PR #15's implementation suite passes **133 tests**. The **171** count belongs
to the combined snapshot, not a plain PR #19 checkout. On the latest PR #14
checkout, `PYTHONPATH=src pixi run review-rough-inference` opens notebook09;
`review-bergomi` covers both Bergomi models and numerical checks.

For bank generation, use the existing JSON presets and a new output directory.
The 24-panel presets check the route; they are not adequate neural training
sets. Actual Oscar smoke jobs generated HKDE/one-factor/rough banks in 4/158/164
seconds respectively, including verification. The later rough pilot generated
6,146 panels in 98 tasks, with a 34.28-minute wall span. These are measured runs,
not scaling promises. Read the [cluster run record](https://github.com/fmuia/sbi4finance/blob/feat/rough-bergomi-option-bank/docs/option-cluster-smoke.md)
and [rough study](https://github.com/fmuia/sbi4finance/blob/feat/rough-bergomi-option-bank/docs/rough-joint-inference.md)
before another submission; discover the collaborator's current allocation and
use their authenticated environment. Preserve manifests, job IDs and transfer
hashes. Do not reuse historical account settings blindly.

## The scientific result that needs a decision

The correction retains the trained flow as a proposal. Its final fixed mixture
is 80% flow, 10% uniform prior and 10% product-arcsine boundary density. The
target prior remains uniform. The full mixture density divides likelihood
times prior; weights are normalized without clipping. These are finite
self-normalized importance estimates, not exact posterior draws.

| Evidence | Recorded outcome |
|---|---|
| Raw neural posterior | Maximum target-price quantile error **0.04387 > 0.03**, despite passing marginal SBC. Remains provisional. |
| Two-contract forward surrogate | Maximum observed price error **3.84e-7 < 0.003**. Accepted only for its declared parameter box/context/two puts. |
| First correction | Accuracy improved, but one larger-pool boundary weight caused a reliability stop. Failure retained. |
| Boundary correction: validation | All gates pass: 20 conditional cases at 2,048/8,192 proposals and 512 calibration panels. |
| Boundary correction: new test | All conditional/weight checks pass; largest price-quantile error **0.01125 < 0.03** at 2,048 proposals. Theta SBC fails: **KS 0.10325 > 0.09251** on 256 panels. |
| Same-panel numerical-reference PIT | Theta KS **0.10349**; 65-point-per-axis posterior-grid check **0.10340**. The reference shares the rank shift. This suggests finite-cohort variability, but does not prove the failure harmless or override the frozen gate. |

The corrected experiment's root-95026 test is **used and closed**. The original
raw-HKDE root-62026 test remains untouched. Heston's later root-43022 test and
the rough pilot's reserved test remain unevaluated. Preserve each experiment's
own history; do not call an inspected test a fresh holdout.

Independent audits reconstruct saved weights and conditional metrics, check
source/input bindings, and replay test parameter/quote-noise seed identities.
SBC rank/coverage arithmetic is checked from saved summaries. Individual SIR
draws were not saved, so those coverage events cannot be independently rebuilt
from the cache alone. Numerical reference PIT is quadrature/interpolation, not
an analytic oracle. [Full results and reproduction](hkde-corrected-inference.md)
state these limitations.

## Recommended next steps

1. **Review and consolidate the existing stack.** Use the actual base branches,
   retain each milestone's review/evidence, and recheck the combined code.
   No further infrastructure expansion is needed for this decision.
2. **Decide the teaching application.** The options candidate is well developed;
   choosing it does not follow automatically from implementation progress.
   A returns example requires its own accepted parameter/state-to-forecast path.
3. **Resolve the calibration acceptance question as a bounded scientific task.**
   Review the shared reference/SIR shift before proposing more neural tuning.
   Any new acceptance study should state its question, justified criteria and
   independent evaluation design in advance, while retaining this failed result.
   Do not keep drawing new seeds until a green result appears.
4. **Then finish M8's live gate.** An accepted chosen pipeline must actually run
   inference and downstream output offline, with component timings and no
   market fixtures. A skipped live cell cannot meet that requirement.
5. **Resume other inference studies only if useful to the curriculum.** Heston
   retains conditional discrepancies; rough Bergomi first needs a bounded
   reference-efficiency follow-up (one case has ESS 600.55 < 800 and maximum
   weight 0.005392 > 0.005). The current rough run stopped before neural training
   and SBC. Two-factor Bergomi, full collaborator SVJ parity, direct-price
   posteriors, American/exotic pricing and P/Q linkage remain extensions.

The latest [analysis note](https://github.com/fmuia/sbi4finance/blob/feat/rough-bergomi-option-bank/docs/simulator-analysis-next-steps.md)
preserves older decisions; its chronological “next” sections can describe work
now completed. Use the current plan and measured reports for present status.

## Data evidence and local orientation

**S: synthetic teaching is the required core.** Simulation, training, truth-based
checks and a complete offline lesson need no real market observations. **M:
market application** conditions normally frozen weights on a compatible real
window/panel; it does not supply true parameter labels or constitute supervised
training. **E: empirical evaluation** needs separately withheld future outcomes
or excluded quotes, usually many origins/dates for broader claims. Acquisition
cannot block S, and synthetic validity does not certify market adequacy.

On the current workstation, the useful worktrees under
`/Users/afengler/Projects/proj_sbi4finance` are:

- `sbi4finance-offline-tutorial`: this handover and draft PR #19.
- `sbi4finance-hkde-inference`: PR #15's scientific implementation/evidence.
- `sbi4finance-option-banks`: latest PR #14 rough work.
- `sbi4finance-plan-pr`: plan-only PR #3.
- `sbi4finance-data-impl`: separate data workstream; coordinate edits there.

The sibling directory named simply `sbi4finance` is an older research-notes
worktree, not the integrated implementation. Other machines should use the
remote PR branches and committed artifacts above; no `/private/tmp` files,
unshared chat history or workstation-specific credentials are needed to read
the delivered evidence.
