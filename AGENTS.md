# Repository instructions

These instructions apply throughout the repository. More specific instructions may add requirements within their directories.

## Repository structure

The repository has three areas; the [README](README.md) explains why.

- `tutorial/` is self-contained. Code under `tutorial/` must not import `sbi4finance`, edit `sys.path`, or read files outside `tutorial/`. Notebooks read inputs only from `tutorial/assets/`. `tutorial/tests/test_independence.py` enforces this.
- Files enter `tutorial/assets/` only through lab tools: `lab/studies/accept.py`, `lab/studies/manifest.py` and `lab/checks/heston_spy_exact.py`. The one exception is slot 2's toy network, which `tutorial/notebooks/slot2_amortized_bayes.py` trains and saves when the file is missing (see `tutorial/MANIFEST.md`). After any change, run `pixi run manifest` and `pixi run claims` in `lab/`. Never hand-copy or hand-edit an asset, `tutorial/MANIFEST.md` or the tables of `tutorial/CLAIMS.md`.
- When a notebook needs lab code at run time, copy the lab functions unchanged, under the same names, into the module of the same name in `tutorial/src/sbi4f_tutorial/`, with `pixi run copy-to-tutorial --add MODULE.NAME` in `lab/` (`COPIED_FROM` in `lab/studies/copy_to_tutorial.py` maps each module to its lab file). `lab/tests/test_tutorial_parity.py` fails when a copy differs from the lab; after changing a copied lab function, run `pixi run copy-to-tutorial`. Code only the tutorial needs is listed in `WRITTEN_FOR_THE_TUTORIAL` in that script, with the reason. Do not make the tutorial import the lab, and do not make the lab import the tutorial outside that test.
- `lab/` keeps only code on the critical path: the full-surface Heston and rough-Bergomi calibration, its simulators, studies, quick checks, cluster jobs and tests.
- The sparse-chain code in `lab/` (`calibration/thin.py`, `thin_products.py`, the thin studies, `checks/thin_chains/`, `configs/thin/`) is deferred. Keep it working, but do not extend it before the sparse follow-up is specified (`lab/plans/SPARSE-QUOTE-CASE-PLAN.md`).
- Run lab commands from `lab/` and tutorial commands from `tutorial/`. Each folder has its own pixi environment.
- Documents: update a plan or status page in place; do not add a new handover or review document for each step. Superseded documents are deleted; git keeps them. Commit compact run records (README, summary JSON, key figures). Banks, model payloads and full posterior draws stay on the cluster, with their SHA-256 recorded.
- Before a pull request, run `pixi run test` in every area you changed. For the tutorial, also run `pixi run test-notebooks` and `pixi run check`; for the lab, `pixi run release-check`.

## MCMC framework

**Prefer PyMC where it is a natural fit.** Aim for a reusable PyMC workflow for standard reference posteriors and emulator-based inference. Adoption is a practical goal, not a prerequisite for every experiment.

- Use PyMC's built-in step methods where they suit the target. NUTS requires a deterministic differentiable log density with validated gradients.
- Custom or other-framework samplers are allowed for specialized teaching examples, particle/pseudo-marginal methods, or workflows where migration would add disproportionate complexity or cost. Record the method, reason and limitations in the relevant documentation or run record; a custom PyMC step is not required just for framework consistency.
- Preserve the intended target, transformation Jacobians and stochastic sampler state in any implementation. A framework exception does not relax correctness or evidence requirements.
- Scope migration to workflows that benefit from it; existing samplers and their tests need not all be rewritten before work continues. PyMC adoption may be deferred if its cost outweighs its value.
- Preserve historical chains, reports and actual sampler provenance. Describe PyMC as an implemented part of the stack only once a working, validated route exists.
- Persist chain/draw structure, sampler diagnostics, seeds, target identity and exact framework/environment versions.

The rationale, exceptions and validation requirements are in [the MCMC policy](lab/plans/mcmc-policy.md).
