# Collaborator handover: finance simulator ladder

**Snapshot: 12 October 2026.** Repository: [fmuia/sbi4finance](https://github.com/fmuia/sbi4finance).
The scoped simulator/bank implementation workstream is complete. The final
participant tutorial is **not yet scientifically accepted**. This note hands
over code, evidence and decisions; it does not claim that every model has a
validated neural posterior. Remaining review and scientific decisions are
tracked in the [follow-up issues below](#tracked-follow-ups). No PR was merged
in this workstream; issue creation launches no further experiments.

## Start here

For a module map, data-flow diagrams and extension points, read the companion
[code architecture guide](code-architecture.md). It describes the implemented
layout and distinguishes additions that still live on separate branches.

1. Read the [plan and mathematical reading guide](https://github.com/fmuia/sbi4finance/blob/docs/simulator-plan/docs/simulator-plan/README.md)
   in [plan PR #3](https://github.com/fmuia/sbi4finance/pull/3).
2. Open [notebook10](../notebooks/10_hkde_corrected_workflow.py) for the provisional
   live synthetic demonstration, and read the [HKDE correction report](hkde-corrected-inference.md)
   for the retained scientific acceptance failure.
3. Inspect the live [joint posterior](results/hkde-live-demo/live-joint.png)
   and [target-price distributions](results/hkde-live-demo/live-target-prices.png),
   then the [price-error graph](results/hkde-correction/plots/price-errors.png)
   and [reference-versus-SIR rank graph](results/hkde-correction/plots/reference-pit.png).
4. For Bergomi's current status, use the [held-out study](https://github.com/fmuia/sbi4finance/blob/feat/rough-heldout-test/docs/rough-heldout-test.md)
   and notebook09 in [PR #28](https://github.com/fmuia/sbi4finance/pull/28).
   The [passed development validation](https://github.com/fmuia/sbi4finance/blob/feat/rough-bergomi-option-bank/docs/rough-reference-expansion.md)
   and [original stopped study](https://github.com/fmuia/sbi4finance/blob/feat/rough-bergomi-option-bank/docs/rough-joint-inference.md)
   remain unchanged in PR #14. The latest rough-inference additions are not all in
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
| M7: rough Bergomi | Gaussian reference and FFT hybrid driver, option banks, coupled-grid study, restricted H/eta inference and notebook09 | Development validation passes after the bounded 8,192-row reference expansion. Held-out study: 16/16 references and six of seven neural gates pass, including 256-case SBC and target means; one eta SD is 31.0% too small against a 25% limit. All test cases are used; candidate remains provisional. |
| M8: paced options candidate | Notebook10, equations, diagnostics, exercises and provisional live offline demonstration | Fixed development example runs; historical theta SBC failure and pending scientific acceptance remain visible. |

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
| [#14](https://github.com/fmuia/sbi4finance/pull/14) | `feat/rough-bergomi-option-bank` | `dfaaee4` | Expanded reference bank, passed bounded neural validation and preserved original stop |
| [#28](https://github.com/fmuia/sbi4finance/pull/28) | `feat/rough-heldout-test` | `f1bfa72` | One held-out study; reference passes, neural eta-spread failure retained, updated review and raw-draw audit |
| [#15](https://github.com/fmuia/sbi4finance/pull/15) | `feat/hkde-inference` | `60ebe4d` | HKDE posterior, surrogate and correction evidence; PR base is #14's branch |
| [#19](https://github.com/fmuia/sbi4finance/pull/19), **draft** | `feat/hkde-offline-tutorial` | `b8f2700` before this held-out-result handover update | M8 review; PR base is #15's branch |

The separate scientific follow-up [PR #28](https://github.com/fmuia/sbi4finance/pull/28)
(`feat/rough-heldout-test`) adds the frozen one-run rough test evaluation,
its results and updated notebook09. It is not another simulator-ladder
implementation increment or an import into PR #19.

**Branch caveat:** a PR targeting another branch does not automatically contain
that branch's later commits. PR #19 is a good checkout for the HKDE review;
PR #14 contains the rough development study and PR #28 its held-out follow-up.
The non-mutating combined
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

The data team has also completed the bounded teaching integration. These newer
PRs use combined review bases containing specific earlier parent revisions:

| Addition | PR / head at this snapshot | Review base |
|---|---|---|
| Local-data teaching integration plan | [#20](https://github.com/fmuia/sbi4finance/pull/20) | #18 branch |
| Shared panel and cached teaching route (I1–I3) | [#21](https://github.com/fmuia/sbi4finance/pull/21), `f385225` | `integration/hkde-data-teaching-base` |
| Provisional live route (I4a) | [#22](https://github.com/fmuia/sbi4finance/pull/22), `663ac4e` | `integration/hkde-live-demo-base` |

Read their recorded composition before integrating; neither combined base
automatically contains later #14 or #19 commits. The data workstream is closed
for I1–I3 and I4a. Formal I4 remains conditional on scientific acceptance for
the actual HKDE participant configuration. The 168-test #21 report and #22
live execution record cover their own snapshots, not a final merged tree.
[Integration issue #27](https://github.com/fmuia/sbi4finance/issues/27) owns this
composition check; no data-side worktree was changed for this handover.

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
cluster job. Notebook10 now runs fixed synthetic **validation:257** with
**2,048 proposals**, the same proposal seed and unchanged numerical tolerances.
It displays the joint correction and two target-price distributions as a
**provisional demonstration**. No real observations or generating truth enter
conditioning. A passing single-case check does not change the used test or
establish population calibration.

The [live execution record](results/hkde-live-demo/verification.json) separates
technical execution and single-case checks from `scientific_acceptance=false`.
It records component timings in an isolated offline export with market
fixtures/raw banks absent. The earlier [cached-only verification](results/hkde-correction/notebook-verification.json)
passed **17 cells, zero errors, 8.29 seconds**, with `live_executed=false`;
that historical record remains unchanged and is not a live-inference timing.
Both checks reuse the pinned environment rather than testing a fresh installation.

PR #15's implementation suite passes **133 tests**. The **171** count belongs
to the combined snapshot, not a plain PR #19 checkout. On PR #28's
`feat/rough-heldout-test` checkout, `PYTHONPATH=src pixi run review-rough-inference`
opens the updated notebook09; it reads cached results without rerunning the test.
That snapshot passes **129 tests** and its cached offline review exports
**26 cells, zero errors, 10.72 seconds**, without market fixtures, raw banks or
checkpoints. An independent audit reconstructs all reference/conditional metrics
and every neural/prior SBC rank and coverage event from raw saved draws.
These execution/audit passes preserve the scientific failure. PR #14's older
125-test/22-cell development verification remains historical; the counts are
separate snapshots, not numbers to add together.

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

## The HKDE scientific result and scoped decision

**Issue #24 decision:** [retain provisional teaching status at 2,048 proposals](hkde-acceptance-decision.md).
Published for review in [PR #29](https://github.com/fmuia/sbi4finance/pull/29),
based on PR #19; no merge is part of this decision.
The decision and source/artifact bindings are recorded in
[the acceptance record](results/hkde-acceptance/decision.json); notebook11
teaches the retained tolerances and shared-alarm policy. No fresh cohort is
needed for this selected-case scope. Broader calibration and full joint accuracy
remain unestablished, historical acceptance stays false, and formal data I4
remains deferred. I4a's runnable provisional example is unaffected.

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
raw-HKDE root-62026 test and Heston's later root-43022 test remain untouched.
The rough pilot's full 256-case test is now **used and closed** after PR #28's
evaluation. Preserve each experiment's own history; do not call an inspected
test a fresh holdout.

A subsequent [saved-cohort gate audit](hkde-calibration-gate-audit.md) verifies
the threshold and shows that ideal reference sampling triggers this same theta
alarm in 98.85% of conditional controls. It identifies measurable extra rank
dispersion and direct-CDF errors consistent with importance-sampling uncertainty
on the 16 saved conditional cases. Notebook11 explains the distinction; no
historical flag or estimator was changed. The follow-up permits a provisional
live demonstration while keeping the scientific acceptance flag false.

Independent audits reconstruct saved weights and conditional metrics, check
source/input bindings, and replay test parameter/quote-noise seed identities.
SBC rank/coverage arithmetic is checked from saved summaries. Individual SIR
draws were not saved, so those coverage events cannot be independently rebuilt
from the cache alone. Numerical reference PIT is quadrature/interpolation, not
an analytic oracle. [Full results and reproduction](hkde-corrected-inference.md)
state these limitations.

## Tracked follow-ups

For assignment to multiple agents, use the [parallel handover index](parallel-agent-handover.md).
It links the individual briefs, covers every open PR, and separates work that
can proceed now from conditional scientific/participant steps.

Review and integration can proceed without accepting every model scientifically.
Each issue has an ordered subplan, source/result links and explicit completion
criteria. Start with the integration map and existing results, choose the
teaching route, and schedule only the scientific work that decision needs.

| Issue | Concrete outcome | Boundary |
|---|---|---|
| [#27 — review and consolidate the PR stacks](https://github.com/fmuia/sbi4finance/issues/27) | Review dependencies, integrate the intended revisions, verify the actual combined code/notebooks and record the reviewed/merged disposition. | Existing snapshot test counts do not certify the resulting tree. Merging does not change scientific acceptance. |
| [#26 — choose the tutorial and finish participant review](https://github.com/fmuia/sbi4finance/issues/26) | Select P returns versus Q options, primary model/companions, claims and live/cached lesson; check clean setup and the complete offline S route. | Does not require every catalog model to pass or any market acquisition. |
| [#23 — rough reserved-test evaluation](https://github.com/fmuia/sbi4finance/issues/23) | **Study completed in [PR #28](https://github.com/fmuia/sbi4finance/pull/28):** frozen checkpoint evaluated once; reference, conditional/joint, SBC and target-mean results audited and taught in notebook09. | All 256 cases used and closed. Reference passes; one conditional eta-spread gate fails, so candidate remains provisional. No retry, tuning or reference expansion followed. |
| [#24 — HKDE scientific acceptance decision](https://github.com/fmuia/sbi4finance/issues/24) | **Decision recorded:** [retain provisional selected-case teaching at 2,048 proposals](hkde-acceptance-decision.md); unchanged tolerances, explicit shared-alarm policy, no new replication. Notebook11 teaches the decision. | Historical test remains failed and closed; broader acceptance is deferred. Formal data-teaching I4 stays deferred; I4a already runs. |
| [#25 — Heston conditional state/forecast decision](https://github.com/fmuia/sbi4finance/issues/25) | Decide defer, retain as a diagnostic lesson, or investigate one mechanism with a bounded protocol and notebook05 update. | Optional unless the returns route needs it. Both v2 and v3 remain provisional; root-43022 stays reserved. |

The dedicated [#24 agent handover](hkde-acceptance-handover.md) preserves the
original briefing and links the completed decision. Reopen scientific work only
when a stronger claim is chosen and its prospective study is justified and
frozen; another cohort is not an automatic closeout requirement.

The [rough held-out study](https://github.com/fmuia/sbi4finance/blob/feat/rough-heldout-test/docs/rough-heldout-test.md)
passes all 16 reference cases and six of seven neural gates, including marginal
SBC and target-price means. On `test:3`, eta's posterior SD is **31.0% too small**
against the unchanged 25% limit; its nominal 90% interval contains only **75.64%**
of the reference posterior's mass. That is a single-case conditional diagnostic,
not cohort coverage. Inspect the [conditional failure](https://github.com/fmuia/sbi4finance/blob/feat/rough-heldout-test/docs/results/rough-heldout-test/plots/conditional-failure.png)
alongside the [SBC/prior control](https://github.com/fmuia/sbi4finance/blob/feat/rough-heldout-test/docs/results/rough-heldout-test/plots/marginal-sbc.png).
The network improves mean RMSE from the prior's 1.4817 to 0.1121 reference SD,
but remains provisional. The earlier passed development result is preserved;
no further run is prescribed by #23, and this is not an HKDE I4 prerequisite.

An explicit negative or deferred scientific decision is a valid issue outcome;
the aim is not to keep experimenting until every status is green. Two-factor
Bergomi, full collaborator SVJ parity, direct-price posteriors, American/exotic
pricing, rough P-memory inference and P/Q linkage remain optional extensions.
They are not hidden closeout requirements.

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
- `sbi4finance-option-banks`: PR #14 rough development work.
- `sbi4finance-rough-heldout`: separate PR #28 test evaluation and current rough review.
- `sbi4finance-plan-pr`: plan-only PR #3.
- `sbi4finance-data-impl`: separate data workstream; coordinate edits there.

The sibling directory named simply `sbi4finance` is an older research-notes
worktree, not the integrated implementation. Other machines should use the
remote PR branches and committed artifacts above; no `/private/tmp` files,
unshared chat history or workstation-specific credentials are needed to read
the delivered evidence.
