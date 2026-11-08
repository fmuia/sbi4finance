# Repository instructions

These instructions apply throughout the repository. More specific instructions may add requirements within their directories.

## Repository structure

The repository has three areas; the [README](README.md) explains why.

- `tutorial/` is self-contained. Code under `tutorial/` must not import `sbi4finance`, edit `sys.path`, or read files outside `tutorial/`. Notebooks read inputs only from `tutorial/assets/`. `tutorial/tests/test_independence.py` enforces this.
- Files enter `tutorial/assets/` only through lab tools: `lab/studies/accept.py`, `lab/studies/manifest.py` and `lab/checks/heston_spy_exact.py`. Slot 2's toy training is an explicit session-only action; it does not publish or write assets. After any change, run `pixi run manifest` and `pixi run claims` in `lab/`. Never hand-copy or hand-edit an asset, `tutorial/MANIFEST.md` or the tables of `tutorial/CLAIMS.md`.
- The authoritative notebook runtime lives in `tutorial/src/sbi4f_tutorial/`: application helpers under `heston_options/`, `rough_options/` and `trajectories/`, reusable numerical components under `shared/`. Edit the executing implementation there. Lab modules reuse that package; compatibility imports preserve older APIs and contain no second implementation. Do not restore the lab-to-tutorial function-copy workflow. Keep the runtime dependency-light; the participant environment adds its BayesFlow 2.1/marimo stack separately from the lab's historical/current environments. The tutorial must never import the lab. Run the lab runtime-boundary tests after changing shared definitions.
- `lab/` keeps bank generation, production training, reference computations, studies, quick checks, compute workers, publication tools and evidence for the three core applications. Shared simulators and inference helpers used by the lessons are owned in the tutorial runtime and imported by lab workflows. Earlier returns-exclusion decisions do not exclude the current trajectory application. Install/deploy the complete checkout so the sibling tutorial package is available.
- The sparse-chain code in `lab/` (`calibration/thin.py`, `thin_products.py`, the thin studies, `checks/thin_chains/`, `configs/thin/`) is deferred. Keep it working, but do not extend it before the sparse follow-up is specified (`lab/plans/SPARSE-QUOTE-CASE-PLAN.md`).
- Run lab commands from `lab/` and tutorial commands from `tutorial/`. Each folder has its own pixi environment; the lab installs the sibling tutorial runtime as a local dependency. Existing data/model identities remain immutable. Moving source changes generator provenance: read/train on preserved banks normally, but extend or resume their generation only from their matching producer checkout/environment.
- Besides the three areas, `site/` is the public tutorial webpage: static HTML and CSS, published to GitHub Pages by `.github/workflows/pages.yml` when a push to `main` changes it. The published site is public although the repository is private, so it holds only what may be public: no unreleased results, private links or credentials. See `site/README.md`.
- Documents: update a plan or status page in place; do not add a new handover or review document for each step. Superseded documents are deleted; git keeps them. Commit compact run records (README, summary JSON, key figures). Large banks, model payloads and full posterior draws stay outside Git, with SHA-256 identities and versioned release/Drive downloads. Explicit ignored download caches under `tutorial/assets/local-checkpoints/` are allowed when the native loader validates their task and model hashes.
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
