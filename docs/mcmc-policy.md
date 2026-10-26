# MCMC framework policy

Updated on 26 October 2026 following the user's clarification: **use PyMC where it is a natural fit, with flexibility for specialized cases or disproportionate integration cost**. The goal is a reusable inference workflow and a useful PyMC component in the tutorial's technology stack. Adoption can be deferred if it adds more trouble than value. The corresponding agent instructions are in [AGENTS.md](../AGENTS.md); [CLAUDE.md](../CLAUDE.md) points Claude-based collaborators to the same guidance.

## Scope

PyMC is the preferred framework for standard posterior references and price-emulator MCMC when its model and sampling interfaces fit the task. Deterministic Heston references and differentiable Heston/rough-Bergomi emulators are the first candidates to assess. We do not require every existing chain loop, figure generator or test to be migrated before work continues.

Custom or other-framework samplers are allowed when a specialized algorithm is central to the example, when preserving a validated implementation is useful, or when migration would add disproportionate implementation, maintenance or compute cost. Examples include a small teaching demonstration of an acceptance rule and a compiled particle/pseudo-marginal sampler. An exception can cover a reference run when justified; it is not limited to toy examples. Record the actual method, why it fits, and its validation and limitations in the relevant documentation or run record.

Reading cached chains, plotting existing results, independent posterior draws from a trained neural estimator, importance sampling and ordinary prior/simulator draws are not MCMC runs.

## PyMC does not mean NUTS for every target

For a PyMC route, use PyMC to declare the model and manage sampling, with built-in step methods where appropriate. State the actual sampler/backend explicitly; wrapping a standalone legacy acceptance loop and calling the result PyMC is not a migration.

| Target | Preferred approach |
| --- | --- |
| Deterministic differentiable posterior, including a differentiable emulator | Assess a PyMC implementation with validated values and gradients, then evaluate NUTS. PyMC Metropolis remains an option for teaching or a justified benchmark. |
| Deterministic black-box numerical likelihood without validated gradients | Assess a supported PyMC black-box interface and an appropriate gradient-free step method. Keep a documented alternative if integration costs outweigh the benefits; do not invent gradients to enable NUTS. |
| Stochastic likelihood, particle or pseudo-marginal target | Prioritize a correct implementation of auxiliary randomness and retained likelihood state. A validated specialized sampler can remain outside PyMC. Ordinary Metropolis with fresh random likelihood evaluations is not automatically pseudo-marginal MCMC. |

A custom PyMC step is an option if it brings a concrete benefit; it is not required just for framework consistency. Any new specialized implementation needs correctness tests on a tractable target. Do not hide fresh random evaluations inside a supposedly deterministic PyTensor operation. A pseudo-marginal rejection must retain the current likelihood estimate and its associated randomness; the SVJ parameter/state path must also preserve its retained terminal-state semantics.

PyMC documents how to integrate [external black-box likelihoods](https://www.pymc.io/projects/examples/en/latest/howto/blackbox_external_likelihood_numpy.html); its [sampling API](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html) provides the framework entry point. Numerical-target validity and pricing accuracy still need their own checks.

## Environment and output requirements

For a new or migrated workflow:

1. Record and lock the actual sampler environment. For a PyMC route, resolve compatible PyMC, PyTensor and diagnostic packages. Keep archived BayesFlow/checkpoint environments reproducible; PyMC is not yet part of the existing project lock.
2. Verify the log density, constraints and transformation Jacobians against the intended target. Existing logit-coordinate callbacks already include Jacobians; do not count them again through PyMC's automatic transforms. Compare any custom likelihood/gradient with an independent calculation.
3. Test sampling and diagnostics on a tractable case, using multiple chains for posterior-reference claims. Verify seeds, save/reload, chain ordering and failure reporting. Bind partial-result caches to the target, sampler, settings and code identity so historical chains cannot silently enter a migrated run.
4. Save structured output with named variables and chain/draw coordinates, sampler statistics, warm-up and retained-draw settings, seeds, timings and exact package/backend versions. Use PyMC's structured trace for PyMC routes and an explicitly labelled compatible representation for other samplers. Prefer the compatible ArviZ diagnostic interface for reference studies.
5. Apply the experiment protocol's convergence/precision requirements to reference claims, including rank-normalized split R-hat and bulk/tail ESS. Record divergences, acceptance and other sampler-specific statistics when applicable. Failed chains remain visible. Teaching demonstrations can use a smaller setup with their limited purpose stated; they are not certified reference posteriors.

PyMC and BayesFlow can use separate locked execution environments and exchange versioned task/data artifacts when their dependency stacks require that separation.

## Current implementation status and priorities

PyMC adoption is planned, not implemented by this documentation. The inspected course branch has custom samplers, including:

- [calibration/correction.py](../src/sbi4finance/calibration/correction.py), `random_walk_chain`, used for numerical references.
- [calibration/pricer_net.py](../src/sbi4finance/calibration/pricer_net.py), `metropolis`, used by [the S4 runner](../tutorial_final/studies/pricer_route.py).
- [svj/pmmh.py](../src/sbi4finance/svj/pmmh.py), `run_pmmh`, and its application callers.
- Course/check scripts with their own chain loops, including `scripts/course/lecture10.py` and `scripts/course/lecture11.py`.
- Tests that execute those implementations, including the pricer/chain test in `tests/test_calibration.py`.

This list is a starting inventory, not a blanket certification of other entry points. Assess callers and figure/test generators as each workflow is reviewed; they can retain a documented custom implementation where appropriate. Start with one bounded core PyMC route, measure the benefit, and expand adoption where useful. Specialized PMMH migration is not a prerequisite for the core Heston/rough-Bergomi tutorial or the data/training campaign.

Preserve the original implementation metadata and historical results. New or migrated results get distinct run identities and measured timing/accuracy reports; the old sampler's cost is not automatically a PyMC benchmark. Update runbooks and course labels when a new path is verified. Describe PyMC as an implemented part of the technology stack once a working, validated route exists; until then label it as planned.
