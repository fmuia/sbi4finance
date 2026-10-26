# MCMC framework policy

Recorded on 26 October 2026 from the user's requirement: **use PyMC for any MCMC runs in this project**. The corresponding agent instructions are in [AGENTS.md](../AGENTS.md); [CLAUDE.md](../CLAUDE.md) points Claude-based collaborators to the same rule.

## Scope

Every new execution of MCMC must use PyMC, whether it is a production study, a pilot, a tutorial example, a course-figure generator or a test. This covers Heston numerical references, price-emulator MCMC, rough-Bergomi anchors and particle/pseudo-marginal work. It also covers rerunning an old experiment: preserved historical source is not an exception to the framework requirement.

Reading cached chains, plotting existing results, independent posterior draws from a trained neural estimator, importance sampling and ordinary prior/simulator draws are not MCMC runs. Algorithms that include MCMC transitions must identify and satisfy the requirement for those transitions.

## PyMC does not mean NUTS for every target

Use PyMC to declare the model and manage sampling, with built-in PyMC step methods where possible. State the actual sampler/backend explicitly; wrapping a standalone legacy acceptance loop and calling the result PyMC is not a migration.

| Target | Implementation requirement |
| --- | --- |
| Deterministic differentiable posterior, including a differentiable emulator | Use an appropriate PyMC step; NUTS requires validated values and gradients. PyMC Metropolis remains an option for teaching or a justified benchmark. |
| Deterministic black-box numerical likelihood without validated gradients | Use a supported PyMC black-box interface and an appropriate gradient-free step method; do not invent gradients or silently choose NUTS. |
| Stochastic likelihood, particle or pseudo-marginal target | Represent auxiliary randomness and retained likelihood state correctly within a validated PyMC implementation. Ordinary Metropolis with fresh random likelihood evaluations is not automatically pseudo-marginal MCMC. |

A specialized algorithm may require a custom PyMC step. Treat that as an explicit implementation task with correctness tests on a tractable target, while retaining PyMC orchestration and structured outputs. Do not hide fresh random evaluations inside a supposedly deterministic PyTensor operation. A pseudo-marginal rejection must retain the current likelihood estimate and its associated randomness; the SVJ parameter/state path must also preserve its retained terminal-state semantics.

PyMC documents how to integrate [external black-box likelihoods](https://www.pymc.io/projects/examples/en/latest/howto/blackbox_external_likelihood_numpy.html); its [sampling API](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html) provides the framework entry point. Numerical-target validity and pricing accuracy still need their own checks.

## Environment and output requirements

Before the first new run:

1. Resolve and lock a compatible reference environment containing PyMC, PyTensor and the diagnostic stack. Keep archived BayesFlow/checkpoint environments reproducible; PyMC is not yet part of the existing project lock.
2. Verify the log density, constraints and transformation Jacobians against the intended target. Existing logit-coordinate callbacks already include Jacobians; do not count them again through PyMC's automatic transforms. Compare any custom likelihood/gradient with an independent calculation.
3. Test sampling and diagnostics on a tractable case, using more than one chain. Verify seeds, save/reload, chain ordering and failure reporting.
4. Save PyMC's structured trace with named variables and chain/draw coordinates, sampler statistics, warm-up and retained-draw settings, seeds, timings and exact package/backend versions. Use the compatible ArviZ diagnostic interface rather than presenting a custom convergence calculation as the standard diagnostic.
5. Report convergence/precision requirements from the experiment protocol, including rank-normalized split R-hat and bulk/tail ESS. Record divergences, acceptance and other sampler-specific statistics when applicable. Failed chains remain visible.

PyMC and BayesFlow can use separate locked execution environments and exchange versioned task/data artifacts when their dependency stacks require that separation.

## Current migration status

The rule is prospective; it does not assert that the current code already uses PyMC. The inspected course branch has custom samplers, including:

- [calibration/correction.py](../src/sbi4finance/calibration/correction.py), `random_walk_chain`, used for numerical references.
- [calibration/pricer_net.py](../src/sbi4finance/calibration/pricer_net.py), `metropolis`, used by [the S4 runner](../tutorial_final/studies/pricer_route.py).
- [svj/pmmh.py](../src/sbi4finance/svj/pmmh.py), `run_pmmh`, and its application callers.
- Course/check scripts with their own chain loops, including `scripts/course/lecture10.py` and `scripts/course/lecture11.py`.
- Tests that execute those implementations, including the pricer/chain test in `tests/test_calibration.py`.

This list is a starting inventory, not a blanket certification of other entry points. Search callers and figure/test generators as each workflow is migrated. Tests or reproduction commands that start legacy chains need migration before execution; cached artifacts can still be inspected.

Preserve the original implementation metadata and historical results. New PyMC results get new run identities and measured timing/accuracy reports; the old sampler's cost is not automatically a PyMC benchmark. Update runbooks and course labels when the new path is verified. The next implementation work must add the environment and migrated runners before new MCMC jobs are queued.
