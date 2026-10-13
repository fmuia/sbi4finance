# One inference workflow, specialized financial applications

The tutorial's common core is an inference workflow. SVJ and HKDE execute the
same conditional-inference runner and training lifecycle; their architecture,
likelihood reference and downstream question remain task-specific components.
This guide supersedes the earlier integration proposal's boundary-only approach.

## The lifecycle

```mermaid
flowchart LR
    T[Declare task and evidence scope] --> S[Inspect prior predictive simulations]
    S --> D[Independent training and validation data]
    D --> F[Fit and select by validation loss]
    F --> C[Freeze checkpoint and task; verify reload]
    C --> O[Prepare compatible observation]
    O --> I[Infer joint unknowns]
    I --> A[Diagnose estimator and observation]
    A --> R[Correct, fall back, report flags or stop]
    R --> P[Propagate joint uncertainty]
    P --> E[Evaluate and communicate limitations]
```

Task definition includes the generative model, prior, observation design, known
context, target coordinates, relevant hidden state, numerical settings and the
quantity ultimately requested. Changing these may create a new task or candidate;
matching tensor shape alone never establishes checkpoint compatibility.

Inspect simulation behavior before training. Training and model selection use
separate data; reserved test cases are not tuning material. Afterwards evaluate
conditional/reference accuracy, simulation-based calibration, joint quantities
and downstream consequences using a declared task-specific protocol. These are
complementary checks, not a single success score. Posterior predictive checks
ask whether fitted models reproduce pertinent observations; they do not replace
parameter-calibration or reference checks. The current historical studies remain
the evidence for the shipped candidates; this integration does not run a new
population acceptance campaign.

## Executable common core

[`inference_workflow.py`](../src/sbi4finance/inference_workflow.py) owns the sequence
`prepare → infer → diagnose → correct/fallback → downstream`. Applications supply
small functions in `WorkflowHooks`; the runner owns routing, stage reports and
status propagation. There is no simulator registry or network superclass.

- `PreparedObservation` contains validated observations and declared provenance.
- `JointDraws` contains finite named joint rows and optional normalized weights.
  Weights apply to entire rows; resampling is an explicit operation. Components
  must not independently shuffle parameters and terminal state.
- `Estimate` identifies the method and its diagnostics. Correction creates a
  new estimate; the report retains the earlier stage and its limitations.
- `Diagnostic` records scope, measured values, predeclared thresholds and outcome.
  An **observation** flag persists across estimator changes. An **estimator** flag
  may trigger a configured fallback, whose own diagnostics must then be assessed.
- `WorkflowPolicy` selects `report`, `fallback` or `raise`. Report-only analysis
  keeps flags attached to downstream results. Fail-closed mode stops before
  propagation. Missing fallback/checks are visible rather than implied passes.
- `WorkflowResult` returns the selected joint representation plus a report.
  Local reliability is `unassessed`, `locally_checked` or `flagged`.
  Scientific acceptance is separately `not_assessed_by_this_run`.

The word `locally_checked` refers only to the named diagnostics and thresholds
in that report. For instance, acceptable weight concentration does not establish
posterior accuracy. Historical acceptance flags remain separate from this local
execution status. A high summary-space p-value is likewise not market adequacy.

The applications are real consumers of this runner:

| Common stage | SVJ P/returns | HKDE Q/options |
|---|---|---|
| Input | Complete 1,000-step daily log-return artifact | Complete ordered nine-quote panel |
| Representation | Two percent-return features, no per-window standardization | Declared premiums and fixed contract context |
| Amortized estimator | Convolutional/recurrent summary, flow matching, nine targets | Direct quote input, spline flow, two targets |
| Correction | Student-t parameter proposal and particle likelihood; filter supplies state | Neural/prior/boundary mixture and Fourier/Gaussian quote likelihood |
| Local diagnostics | Summary-space screen; importance concentration and likelihood noise; PMMH mixing if invoked | Weight concentration plus optional numerical-reference diagnostics |
| Failure response | Explicit flagged reporting, PMMH fallback or stop | Explicit flagged reporting or stop; no invented PMMH substitute |
| Propagation | Matched parameter/state rows → futures → log-loss risk | Weighted parameter rows → target model prices |

[`svj/pipeline.py`](../src/sbi4finance/svj/pipeline.py) retains the historical
`run_window` compatibility entry point; `run_prepared_window` accepts the strict
local artifact. Both now use the shared runner. The particle model and frozen
50-Euler-step neural sampler are unchanged. The larger SVJ application improves
the project through explicit observation screening and fallback/status handling.

[`hkde_workflow.py`](../src/sbi4finance/hkde_workflow.py) supplies `run_panel` with
the full checkpoint task and optional downstream pricing/reference callbacks.
The accepted target-price surrogate can supply downstream prices; it never
replaces the numerical likelihood. The live HKDE lesson attaches its existing
conditional reference gates to the common diagnostic stage. The original
correction/reference implementations, gates and scientific results remain intact.

GBM remains the exact introductory reference. Heston remains the restricted
parameter/state diagnostic rung; its historical commands and reports retain
their original contracts. Their conceptual stages map to this same lifecycle.
This first executable convergence is SVJ/HKDE, where the different state, weight,
likelihood and downstream requirements exercise the core most usefully.

## Shared training, application-specific architecture

[`training_workflow.py`](../src/sbi4finance/training_workflow.py) supplies actual
online/offline fitting, validation-loss checkpoint selection, selected-checkpoint
reload, output-directory reservation and seeded reload verification. Both
`scripts/train_hkde.py` and `svj/amortized.py` call it. Existing HKDE numerical
reference prerequisites remain enforced by its task-specific runner.

Applications provide the BayesFlow workflow/adapter, architecture, data, budget
and comparison interface. The core does not prescribe a summary network,
density estimator or common normalization for unlike observations. Training
targets and observation features stay separate. An SVJ bank preserves the eight
native coordinates and floored terminal variance; its adapter constructs the
ninth transformed target and two feature channels explicitly.

For offline fitting, records distinguish unique histories from repeated training
exposures. Online batches are freshly simulated; its budget does not silently
become a finite-bank epoch count. New candidates are selected/reloaded explicitly,
then separately evaluated. A tiny train/reload run demonstrates plumbing only.

The shipped SVJ checkpoint remains byte-identical to the contribution, including
its original training record and misspecification reference. Its new sidecar
binds known task/feature/solver/version facts and artifact hashes, while marking
unknown historical provenance and the original final-epoch selection honestly.
It does not retroactively claim the new training lifecycle generated that model.

## Runnable paths

```bash
# Cached defaults; no market data or network access after installation.
pixi run review-svj
pixi run review-hkde-corrected

# Explicit, bounded synthetic inference; no retraining or cluster submission.
pixi run demo-svj
pixi run demo-hkde-corrected

# New directories only. The saved reference is never the training destination.
pixi run svj-bank --config configs/tasks/svj_daily_smoke.json --out /tmp/svj-bank
pixi run svj-bank-check --bank /tmp/svj-bank
pixi run svj-train --bank /tmp/svj-bank --epochs 1 --batch-size 8 --out /tmp/svj-offline
pixi run svj-train --preset smoke --out /tmp/svj-online
pixi run svj-daily --artifact data/synthetic/svj-workflow/window --out /tmp/svj-case.json
```

The generation CLI and thin `cluster/svj_bank.sbatch` run the same bank code.
Stable task/root/split/example identities survive reordering and batch changes.
The new bank RNG protocol is versioned separately from historical online replay.
Bank generation stores decimal returns, native coordinates and terminal variance;
full paths/features are not duplicated in training storage. The independent bank
checker verifies provenance and replays selected examples, beyond file integrity.

Synthetic **S** workflows have known truth; optional market **M** conditioning
does not. Empirical **E** evaluation needs withheld outcomes or quotes across
appropriate dates. Shared orchestration does not connect P and Q probability
measures or transfer acceptance between applications.

## Evidence and follow-ups

Current SVJ VaR/ES is explicitly **percent log-loss**, preserving the historical
functional. Position-loss v2 would apply `-100 * expm1(R / 100)` to every future
before calculating tail statistics and would require versioned risk/backtest
results. It requires no inference retraining solely for that transformation.

The original SVJ 8/9 SBC result, 9/9 replicate, conditional PMMH discrepancies and
flagged importance cases are retained. HKDE's failed historical gate also remains.
Software integration and selected-case checks do not grant scientific acceptance.

See the [integration record](svj-integration-review.md) for measured checks,
source revisions, artifact preservation, cluster evidence and remaining limits.
