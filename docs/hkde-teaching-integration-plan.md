# Plan: connect local HKDE data to the corrected teaching workflow

**Outcome:** one prepared synthetic option panel can enter the existing corrected
HKDE lesson through the local-data interface. Reviewers can follow the observation,
raw neural result, weighted correction and held-out price uncertainty. The
participant live calculation stays disabled until scientific acceptance.

This is a small follow-up to [data plan #4](https://github.com/fmuia/sbi4finance/pull/4)
and [simulator plan #3](https://github.com/fmuia/sbi4finance/pull/3). Reuse
[data PR #18](https://github.com/fmuia/sbi4finance/pull/18) and
[simulator tutorial PR #19](https://github.com/fmuia/sbi4finance/pull/19), including
its correction and pricing dependencies from PR #15.

## Scope and current boundary

Use the existing synthetic `validation:257` panel: nine ordered European put
premiums in currency, known task context, and joint inference of `theta` and
`jump_intensity`. Target puts are `(K=95,T=.75)` and `(K=80,T=1)`. Preserve the
complete task, fixed prior/noise model, parameter pairing and artifact provenance.

The raw neural checkpoint remains provisional. The corrected algorithm passes
conditional pricing checks but still fails its new-test theta calibration gate.
The numerical-reference comparison does not override that decision. Immediate
work verifies integration and cached teaching; it does not re-evaluate scientific
acceptance. The old reserved neural test remains unevaluated; the correction's new
test is already used and must not become another tuning or teaching input pool.

Reuse existing loaders, correction, weighted summaries, surrogate and plots.
Keep the task bounded: no provider acquisition, new banks, retraining, model
expansion, generic adapter framework or changes to scientific thresholds.

## Milestones

### I1 — Local panel as the shared entry point

**Work:** load the committed CSV/JSON panel in the existing corrected lesson,
check its complete task against the saved checkpoint task, and extract the
ordered `[9,1]` price array. Show the local panel even while live inference is
disabled. Feed that array to the existing correction path when its gate allows.

**Done when:**

- Prices and task match the bound upstream `validation:257` observation exactly.
- Removing or changing truth and auxiliary fields leaves conditioning unchanged.
- Incompatible context, quote order or units fail before inference.
- The existing source-bound scientific gate still controls live execution.

### I2 — One connected teaching sequence

**Work:** retain the small data notebook as an introduction and link to the
existing corrected-inference lesson. Add a compact cached raw/corrected/reference
comparison for the same panel and contracts. Use the common `(95,.75)` target
for the introductory comparison; the advanced lesson can show both targets.
Preserve the original raw cache and its recorded failure.

**Done when:**

- Corrected parameter rows, normalized weights and target-price rows stay aligned;
  upstream weighted-summary functions produce the comparison.
- Each result names its actual sample budget and cached/live status. An 8,192-
  proposal cache is not presented as a 2,048-proposal participant execution.
- The raw failure, corrected pricing improvement and remaining calibration failure
  are distinct. No unweighted proposal quantile is called a corrected result.
- Existing cached outputs are reused; any small extracted slice retains source
  hashes and observation identity rather than introducing a new cache framework.

### I3 — Combined offline delivery check

**Work:** exercise the data and simulator stacks together in an isolated combined
checkout, recording their exact commits. Run focused boundary tests, marimo checks
and the cached lesson with network access blocked and market files/raw banks absent.
Inspect plots, links and status messages; document one reproducible entry command.

**Done when:** the combined example executes without errors, preserves the input
boundary, and visibly records `scientific_acceptance=false` and
`live_executed=false`. Save a small verification report with artifact/source hashes.
The simulator-only 171-test report does not substitute for this combined check.

### I4 — Live participant acceptance, deferred

**Prerequisite:** the simulator/inference workstream supplies a scientifically
accepted corrected algorithm with evidence bound to the actual artifacts and
participant configuration. No local override or automatic rerun is part of this plan.

**Work:** run the intended 2,048-proposal live calculation through the same local
panel entry point, offline. Check weighted parameter/price outputs against the
approved reference using its established criteria. Record full elapsed time,
proposal count, diagnostics and source identities.

**Done when:** the actual live execution passes those checks and its recorded
status agrees with the approved scientific evidence. An accepted larger proposal
budget or a successful cached export alone does not complete I4.

## Delivery and ownership

The data workstream owns I1–I3 and the eventual I4 execution check. The
simulator/inference workstream owns I4's scientific prerequisite. Implement I1–I3
in one PR titled **`1/1-data-integration-hkde-teaching-handoff`**, referencing this
plan and both dependency stacks. Review its diff against a combined base so the
PR shows integration changes rather than copied upstream implementations. Record
how that base was assembled; leave the other workstream's worktrees untouched.
No PR or main-branch merge is part of this plan.

I1–I3 complete the immediate integration deliverable. I4 remains explicitly
deferred, not silently completed by the engineering checks. Real-market examples
and empirical evaluation remain optional separate work.

| Milestone | Initial status | Dependency |
|---|---|---|
| I1: local entry point | Planned | Published data and simulator artifacts |
| I2: connected cached lesson | Planned | I1 |
| I3: combined offline check | Planned | I1–I2 |
| I4: live acceptance | Deferred | Scientific acceptance plus I1–I3 |
