# Repository instructions

These instructions apply throughout the repository. More specific instructions may add requirements within their directories.

## MCMC framework

**Use PyMC for every new MCMC run.** This includes reference posteriors, emulator-based inference, numerical pilots, course figures, toy examples, tests that execute MCMC, and particle/pseudo-marginal experiments.

- Define and execute new MCMC through PyMC. Use its built-in step methods where appropriate; do not introduce or run standalone hand-written acceptance loops or silently substitute another MCMC framework.
- Choose the step method for the target. NUTS requires a deterministic differentiable log density; black-box or noisy likelihoods need an appropriate, validated PyMC implementation.
- Preserve historical chains, reports and source provenance. Their existence does not authorize starting their legacy sampler for a new run, and they must not be relabelled as PyMC results.
- Migrate the relevant legacy entry point and any tests that execute it before running it again. If the required method is not correctly supported yet, report that implementation gap instead of falling back to a legacy loop.
- Persist chain/draw structure, sampler diagnostics, seeds, target identity and exact framework/environment versions.

The human-readable requirements and migration checklist are in [the MCMC policy](docs/mcmc-policy.md). This rule records the user's framework requirement; implementation migration is separate work.
