# SVJ integration and shared workflow review

Francesco's daily-return application and the HKDE options lesson now execute one
shared inference workflow. The common core owns preparation, joint inference,
diagnostic scope, correction/fallback routing, downstream status and evidence
reporting. Both training paths use shared fitting, validation-based checkpoint
selection and reload verification. Architectures, summary networks, likelihood
references and downstream questions remain application-specific.

The [workflow guide](shared-inference-workflow.md) is the current implementation
contract. It supersedes a boundary-only interpretation of homogenization.
SVJ is an optional P/returns companion to the selected Q/options curriculum;
this integration does not replace the two-hour participant route.

## Source and integration choices

The repository was fetched before implementation and again before closeout on
13 October 2026. Both checks found these same heads:

| Source | Revision |
|---|---|
| Main | `b2a575149bf5035a95ac1da6d60749ba10b61c5f` |
| Francesco's PR #33 | `2e3165f7b8a25bfaaf2639fa6b759e20d1a7516b` |
| Compatibility overview, PR #34 | `49b44099635d442fd0a23c3e4fa5d75a412fbcde` |

An isolated integration branch merged both contributions with their ancestry.
README and environment conflicts were resolved by retaining the combined
applications, existing pinned environment and Numba dependency. The lockfile did
not change. The original daily SVJ law was retained as its own task rather than
silently equated with the substepped Heston simulator.

The completed work follows this order:

1. Establish executable inference and training cores, with real SVJ and HKDE
   consumers and explicit failure/status semantics.
2. Bind task, prior, feature channels, targets, sampler settings and reference
   identity to checkpoint metadata; reject incompatible prepared inputs.
3. Add deterministic saved SVJ banks and the same local/cluster generator.
   Training consumes observed features separately from synthetic targets.
4. Preserve historical scientific evidence and require explicit new destinations
   for training, validation and study outputs. Classical resume verifies a matching
   run; merges read explicit shard inputs and write a new result.
5. Teach the shared lifecycle in both lessons, with a cached synthetic SVJ default,
   a bounded live opt-in, and joint uncertainty propagated to the stated quantity.
6. Verify regressions, real training/reload, cluster reproduction and tutorial
   delivery before merging the integration.

## Engineering verification

The new worktree installed successfully with `pixi install --locked`. The complete
suite then passed **295 tests in 69.448 seconds**, using its own pinned JAX
environment. Coverage includes the original simulator/inference tests, actual
small SVJ offline training, selected-checkpoint reload, a real HKDE nonfinal-best
epoch selection, joint weights/state, persistent observation flags, failed audit
normalization, reference mismatch rejection, bank reproduction, prepared input
roles, cached teaching integrity and study output guards. A separate real SVJ
online training smoke also passed.

An earlier concurrent test run correctly rejected a teaching cache after a source
change. The cache was regenerated from the final sources; the complete suite above
passed afterwards. This was an artifact-identity check, not a scientific rerun
chosen to improve a failed diagnostic.

The live HKDE lesson uses the common runner with the original 2,048-proposal
mixture, numerical likelihood and eight conditional/proposal checks. Cached and
live exports completed with zero cell errors and four/six figures respectively.
All six live figures were visually inspected, including paired weighted draws
and the two target-price distributions. The first measured live run took
6.116 seconds including setup. Historical validation/test decisions remain
passed/failed respectively.

The SVJ default exported in an isolated directory without market files or raw
banks, with neural imports and external Python socket access blocked. The measured
export took 2.23 seconds before final presentation-only rounding/link edits.
Its two figures were visually inspected. Chrome interaction with the running
lesson verified that the checkbox recomputes a live synthetic case and retains
the diagnostic and scientific-status explanation. The bounded case uses 128
neural draws, 256 importance proposals, 128 particles and four forward paths per
joint draw. It reports ESS 17.01/256, efficiency 6.64% and log-likelihood SD 0.747:
`locally_checked`, with scientific acceptance `not_assessed_by_this_run`.

The final revision-bound participant-route export record is added alongside
these checks in `docs/results/svj-integration/`. These checks establish integration
behavior; they do not expand the selected route's scientific acceptance or claim
a browser-wide offline certification.

## Oscar bank check

Oscar CPU job **7345295** completed with exit `0:0`: one CPU, five seconds elapsed,
2.475 CPU seconds and 82,692 KiB peak resident memory. The smoke bank contains
16/8/4 train/validation/test examples, each with 1,000 daily observations. Generation
and writing took 0.125 seconds; verified loading took 0.016 seconds; loaded arrays
occupy 251,008 bytes.

The cluster checker regenerated nine selected examples in reverse order and
matched exactly. Downloaded files matched independently retrieved remote SHA-256
hashes. Local arm64 regeneration passed the predeclared portable tolerances
`rtol=1e-10`, `atol=1e-12`; it is not bitwise identical across platforms. The largest
absolute difference was `1.11e-15` in terminal variance. The committed source
hashes match those submitted to Oscar. See the machine-readable
[cluster record](results/svj-integration/oscar-bank.json).

## Preserved evidence and limits

The [preservation inventory](results/svj-integration/preservation.json) verifies
the three original SVJ checkpoint files, 26 historical SVJ result files, 29 design
images and the original lockfile byte-for-byte. The existing participant verifier
also checked 59 frozen HKDE dependencies. The added checkpoint sidecar describes
known historical facts and leaves unavailable source/seed provenance unknown;
it does not claim that the new training lifecycle produced the old checkpoint.

Initial SVJ SBC covered 8/9 targets inside the simultaneous bands; the retained
replicate covered 9/9. Conditional PMMH discrepancies and flagged importance
cases remain visible. Successful local concentration/noise diagnostics, a passing
replicate or successful software execution do not establish posterior calibration.
HKDE's historical failed gate and deferred I4 are unchanged.

The retained SVJ estimand is **percent log-loss**, not percent-of-position loss.
Risk reports and the lesson now say so explicitly without changing historical
numbers. A position-loss v2 must transform each simulated future before computing
tail statistics and generate separately versioned risk/backtest evidence.

Further scientific work can choose new prior/task/network candidates and evaluate
them under a declared conditional, calibration and downstream protocol. Full
population validation, larger tail-risk budgets and empirical market evaluation
are separate experiments. GBM and Heston retain their existing runners; this
integration's first executable convergence is SVJ/HKDE.
