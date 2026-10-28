# Final500 Heston teaching export and replay

The new final500 checkpoint export was created on Oscar without modifying the original selected402 run. It contains the exact audited candidate bytes, task, original training metadata, a clearly derived final500 training record and the existing scoped gate/product reports. It is loadable by the ordinary `full_surface.load_checkpoint` API; no compatibility bypass or old-checkpoint overwrite is required.

Model location: `/oscar/scratch/afengler/sbi4finance-heston-teaching-20261028/checkpoint-final500/`.

The initial cache at `data/synthetic/full-surface-heston-final500/validation-0/` replays the existing development case0. It contains the complete 63-price observation, 10,000 paired neural parameter rows, four PyMC chains of 8,000 rows and corresponding analytic variance products. The cache is about 716 KiB and initializes no neural backend. It remains an integration fixture and boundary example.

The separate, now-completed primary cache `data/synthetic/full-surface-heston-final500/interior-midpoint/` is about 724 KiB and binds the new ex-ante midpoint reference and neural output. Its own convergence/refinement/interior checks passed. Product summaries are descriptive for that case, not copied gate verdicts from the old case0. No new inference was needed to build either cache.

## Actual roundtrip check

A temporary copy of the model outside Git was loaded locally in the existing core environment. We compared the ordinary exported-directory loader with direct workflow construction plus the same candidate load. **Ten joint draws at seed 280500 were bit-identical**, shape `[1,10,4]`. The normal loader succeeded, the cache's complete task/checkpoint identity checks passed, and model/metadata hashes were unchanged before and after. Total read-only local runtime was 11.75 seconds, with BayesFlow2.1.0 / Keras3.15.1 / JAX0.11.1 / NumPy2.5.3. Default JAX float32 behavior was retained.

The [verification](verification.json) and [check source](check_export_reload.py) retain the result and exact module/test hashes. This verifies a software/reload route on one development observation; it is not new training, pricing, MCMC, notebook execution or scientific acceptance. Historical seeded-reload fields in the original record remain unchanged. The derived export conservatively keeps its seeded-publication verification flag false; this separate report supplies the actual later verification rather than retroactively rewriting it.

The initial seven synthetic software tests passed for original metadata preservation, correct final500 identity, cached chain axes, expected-task/context and model binding, corruption, rehashed product mispairing, scope-flag preservation, and backend-free replay. Two later tests cover completed interior-case identity/no inherited product verdict and rejection of an unchecked interior case. All nine pass. The offline test actively rejects imports of BayesFlow, Keras, JAX and SciPy.

The [export manifest](export-manifest.json) lists model and metadata hashes. [Exact initial producer source](producer_full_surface_teaching.py) is retained; the library later adds a separate completed-development-cache command without changing the original loader/export behavior. Analytic product checks allow only floating-point roundoff across platforms. Model and full bank payloads are not committed. No Slurm allocation was submitted by this packaging unit.

See the [participant/instructor guide](../../../full-surface-artifact-handover.md) for loading, optional model transfer/live sampling and instructor regeneration commands.
