# Repository instructions

These instructions apply throughout the repository. More specific instructions may add requirements within their directories.

## MCMC framework

**Prefer PyMC where it is a natural fit.** Aim for a reusable PyMC workflow for standard reference posteriors and emulator-based inference. Adoption is a practical goal, not a prerequisite for every experiment.

- Use PyMC's built-in step methods where they suit the target. NUTS requires a deterministic differentiable log density with validated gradients.
- Custom or other-framework samplers are allowed for specialized teaching examples, particle/pseudo-marginal methods, or workflows where migration would add disproportionate complexity or cost. Record the method, reason and limitations in the relevant documentation or run record; a custom PyMC step is not required just for framework consistency.
- Preserve the intended target, transformation Jacobians and stochastic sampler state in any implementation. A framework exception does not relax correctness or evidence requirements.
- Scope migration to workflows that benefit from it; existing samplers and their tests need not all be rewritten before work continues. PyMC adoption may be deferred if its cost outweighs its value.
- Preserve historical chains, reports and actual sampler provenance. Describe PyMC as an implemented part of the stack only once a working, validated route exists.
- Persist chain/draw structure, sampler diagnostics, seeds, target identity and exact framework/environment versions.

The rationale, exceptions and validation requirements are in [the MCMC policy](docs/mcmc-policy.md).
