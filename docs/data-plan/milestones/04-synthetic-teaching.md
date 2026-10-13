# D4 Complete offline synthetic teaching path

**Status:** proposed. **Scope:** required S delivery for the selected application. **Depends on:** [D2](02-compatibility.md) and PR #3's accepted selected scientific/inference/pricing track. **Does not depend on:** D3, D5 or any market provider. Return to the [roadmap](../README.md).

## Purpose

Connect the data layer to a complete teaching workflow, with no market files or credentials. Select either returns → joint parameter/state posterior → conditional future losses, or synthetic option panel → Q posterior → target model-price uncertainty. Other applications remain optional; this milestone does not mandate both curricula.

## Dependencies and inputs

Use scientifically generated fixtures, a compatible validated checkpoint, independent synthetic validation/reference results and accepted forward calculations. PR #3 owns those scientific assets and M8's statistical acceptance. Data milestones cannot replace them with a parser fixture or an unchecked checkpoint. Cached synthetic reference outputs are allowed when labelled and reproducible.

## Implementation sequence

1. Assemble a small local fixture bundle and checkpoint manifest for the selected path. Preserve generation version/seeds, observation law, known context and any truth available only to diagnostics. Keep large training banks on their intended storage rather than bundling them for participants.
2. Wire notebook loading through D1/D2 without vendor imports or automatic downloads. Show synthetic origin and the specific inference question. Display truth after inference for explanation/recovery, never as encoder context.
3. Execute the selected posterior and downstream calculation using the scientific workstream's implementation. Returns must continue matched parameters and sufficient state; option outputs must represent model expectations, with pricing approximation error separated from posterior spread.
4. Show the relevant independent checks and state the model-conditional scope. A short toy training run demonstrates mechanics only; it cannot replace the validated checkpoint. Keep optional real-data/evaluation cells separate and visibly unavailable if their assets are absent.
5. Rehearse in an installed, clean supported CPU environment with network disabled, provider packages/credentials unavailable and all market fixtures absent. Measure loading, posterior sampling and downstream work together; record cached versus participant-executed steps.

## Deliverables

A local synthetic fixture bundle, checkpoint/data compatibility card, notebook integration, reproduction instructions and an acceptance report linked to PR #3's numerical/posterior checks. Declare dependencies directly in the existing environment if needed. Preserve the existing MMAR example and its checkpoint contract unless a separate curriculum change authorizes replacement.

## Acceptance and verification

The whole selected workflow runs from local synthetic inputs to the intended financial output under the absence conditions above. No missing market input triggers a download or silently relabelled synthetic replacement. A deliberately incompatible fixture/checkpoint fails clearly. Validate reloads, field boundaries and output meaning using accepted reference cases; data-path tests supplement rather than replace the scientific validation report. Record measured runtime without promising a universal speedup.

## Exit and deferred scope

Release S when both data integration and the selected scientific track pass. If the model/checkpoint is not ready, report that concrete dependency while keeping D0/D1 and local D2 work reusable. Missing market access never blocks release. American/exotic teaching can be fully synthetic, but waits for its validated solver; learned pricing, additional models and empirical claims retain their own acceptance work.
