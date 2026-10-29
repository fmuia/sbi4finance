# Inference performance - 28 October 2026

[Read the rendered eight-page report](inference-performance-2026-10-28.pdf).

The main figures retain three completed baseline models: **Heston 100k final 500**, **known-curve rough 50k**, and **joint-curve rough 20k**. Page 8 adds the newly completed **known-curve rough 100k** fit. Each was evaluated on 2,000 cases with 1,000 posterior draws per case. Heston uses a frozen held-out test panel; the rough models use development validation, with reserved inference-test outcomes untouched. All recorded simultaneous rank screens pass. This does not imply perfect, conditional or full joint calibration.

The baseline known-curve 50k and later 100k fit have distinct model and price-emulator hashes. The new results do not overwrite the 50k cache or its current reference. rho x eta is a downstream product of aligned joint draws, not an additional inferred target.

## Recorded rank screens and 95% coverage

| Model | Diagnostic target | Rank KS | Declared limit | 95% coverage |
| --- | --- | ---: | ---: | ---: |
| Heston | Initial variance | 0.02639 | 0.03562 | 95.20% |
| Heston | Long-run variance | 0.01781 | 0.03562 | 94.40% |
| Heston | Vol-of-vol | 0.01745 | 0.03562 | 94.55% |
| Heston | rho | 0.01841 | 0.03562 | 94.80% |
| Known curve | H | 0.02276 | 0.03562 | 94.55% |
| Known curve | eta | 0.02376 | 0.03562 | 94.45% |
| Known curve | rho | 0.02194 | 0.03562 | 94.85% |
| Known curve | rho x eta* | 0.01562 | 0.03562 | 93.85% |
| Joint curve | H | 0.01403 | 0.03753 | 95.15% |
| Joint curve | eta | 0.01868 | 0.03753 | 94.00% |
| Joint curve | rho | 0.01668 | 0.03753 | 95.10% |
| Joint curve | Short variance | 0.02229 | 0.03753 | 96.35% |
| Joint curve | Long variance | 0.01485 | 0.03753 | 95.25% |
| Joint curve | Curve decay | 0.02805 | 0.03753 | 95.05% |
| Joint curve | rho x eta* | 0.01341 | 0.03753 | 95.90% |

Pointwise coverage qualifications remain: Heston initial/long-run variance 50% coverage is 52.25/52.30%; known-curve rho x eta 95% coverage is 93.85%; joint decay 50% coverage is 46.75%, eta 95% is 94.00%, and short variance 80/90/95% coverage is 82.3/92.0/96.35%. Retained Wilson intervals are exploratory pointwise checks, without multiplicity adjustment.

## Reference and pricing evidence

- Heston's complete interior 4D PyMC/Fourier comparison is descriptive development evidence: maximum R-hat 1.00297, minimum bulk ESS 1,908, maximum neural/reference mean difference .1013 reference SD and 90% width ratios 1.016-1.040. It is mostly regular, not evidence of dramatic non-Gaussian geometry.
- All 12 Heston analytic product quantile comparisons pass their declared uncertainty-aware tolerances. These are variance-swap and forward-variance rates, not path-product acceptance.
- Current pricing-emulator validation-label RMSE is .08287 quote SD (known 50k) and .10118 (joint 20k). Worst quote errors are 1.311 and 3.888 respectively; worst joint surface RMS is 1.522. Mean error alone does not validate an emulator-based likelihood.
- Earlier known 20k conditional comparisons passed 1/3 cases; independent price and grid checks detected up to .742 and .276 quote SD discrepancies. Current known 50k PyMC job 8059665 is complete: all three four-chain x 8,000-draw diagnostic screens pass, and parameter-agreement checks pass cases 0/1 and fail case 2 (2/3). The same observations, known curves, 0.3-reference-SD mean and 25% relative-width rules were reused. Case 2 mean errors for H/eta/rho are .822/.789/.290 reference SD and 90% width ratios 1.814/1.318/1.423; downstream rho x eta has mean error .103 SD and width ratio 1.451. Eta's mean discrepancy improves from 2.052 SD, but current neural comparisons reuse 1,000 draws versus 10,000 previously, reference-quantile MCSE is uncomputed, and no data-count-only causal claim is warranted. The reference remains a deterministic mean-emulator likelihood with independent Gaussian quote noise; MC label covariance, emulator uncertainty and grid bias are omitted. Joint conditional reference and full model/product/grid acceptance remain unfinished. [S13,S14]
- A new independent fixed-point rescore uses the current 50k pricer against the same 48 saved independent batches. Coarse RMS errors improve at all six fixed points: case 0 old/current pairs are .08165/.03811, .07635/.03588 and .07725/.03626; case 2 pairs are .25359/.10523, .16579/.05189 and .19455/.07426. Current refined RMS remains .0526-.1641, and the largest inspected refined-quote error is .5683 quote SD. Point roles are old neural/reference vectors, not current posterior means. This rescore ran no new Monte Carlo or MCMC; the separately completed current PyMC run is described above. No causal attribution solely to data count is established. Full pointwise t7 intervals and replicate covariance remain in S10.
- Ten local deterministic gradient checks pass: six known 50k and four joint 20k points agree across analytic derivatives, finite differences and compiled PyTensor log-density gradients. This verifies the checked implementation, not likelihood accuracy or conditional posterior agreement. Actual local platform/runtime metadata is retained in S10 and S12.

## Later known-curve 100k fit: matched learning comparison

The 100k fit completed. All 5,000 validation surfaces and the original 50k training prefix retain identical hashes; the same 2,000 validation cases, 1,000 draws and sampling seed are compared. Both fits use seed 0 and final epoch 500, but perform 98,000 versus 195,500 posterior updates, so this is neither a replicated nor fixed-compute learning curve. Both four-target rank families pass. Primary rank KS values become slightly smaller, while 90% coverage changes little; this does not establish a material posterior improvement. [S15,S16,S19]

The new price emulator regresses: label RMSE rises from 0.08287 to 0.09843 quote SD (18.8%), and independent coarse/refined RMS errors are higher at all six reused fixed points. It ran 116 epochs and selected 86, versus 300/298 for the 50k emulator. Different optimization histories are a confound, not a proven explanation or evidence that more data is harmful. [S15,S17]

Both saved neural estimators were also compared against exactly the same completed **50k-pricer** PyMC chains, observations and curves, with 1,000 neural draws each and no new MCMC. Both pass 2/3 fixed-case parameter screens; case 2 still fails H and eta. Case-2 H/eta/rho mean discrepancies change from .822/.789/.290 to .565/.662/.103 reference SD, with width ratios 1.814/1.318/1.423 to 1.587/1.293/1.169. Downstream rho x eta mean discrepancy changes .103 to .259 SD while its width ratio improves 1.451 to 1.188. This holds the approximate reference fixed; the new 100k pricer is not that reference. Neural and reference-quantile Monte Carlo errors remain unassessed. No scientific acceptance or cache promotion follows from these development results. [S18]

## Reproduce

Use a plotting/PDF environment with numpy, matplotlib, reportlab and pypdf (PyMuPDF optional for page rendering). Run `python scripts/render_inference_performance_report.py --source-ref feat/parallel-tutorial-delivery` from the repository, optionally replacing the ref with an immutable source commit. This reads retained evidence and generates figures, a PDF in `output/pdf/`, the shareable PDF here, this companion and source-hash inventory. It runs no notebooks, models, simulations or inference. Temporary dependencies used for the delivered rendering were isolated from the project's locked environments; exact renderer versions are recorded in the source inventory.

## Exact sources

- [S1] [Heston frozen-candidate held-out evaluation](../../docs/pilot-results/2026-10-28/heston-candidate-acceptance/results/acceptance/evaluation.json).
- [S2] [Known-curve 50k validation](../../docs/pilot-results/2026-10-28/known50k-completed/evaluation.json).
- [S3] [Joint-curve 20k validation](../../docs/pilot-results/2026-10-28/joint20k-completed/evaluation.json).
- [S4] [Heston analytic product comparisons](../../docs/pilot-results/2026-10-28/heston-candidate-acceptance/results/products/products.json).
- [S5] [Heston interior PyMC / neural verification](../../docs/pilot-results/2026-10-28/interior-corners/heston-interior-verification.json).
- [S6] [Known-curve 50k pricing-emulator report](../../docs/pilot-results/2026-10-28/known50k-completed/pricer-run.json).
- [S7] [Joint-curve 20k pricing-emulator report](../../docs/pilot-results/2026-10-28/joint20k-completed/pricer-run.json).
- [S8] [Earlier known-curve independent pricing checks (20k model)](../../docs/backbone-validation-2026-10-28.md).
- [S9] [Complete Heston interior corner provenance](../../docs/pilot-results/2026-10-28/interior-corners/heston-interior-complete-corner.json).
- [S10] [Known 50k fixed-point rescore and compiled gradient checks](../../docs/pilot-results/2026-10-28/rough-emulator-readiness/known50k/readiness.json).
- [S11] [Pinned offline backbone artifact catalogue](../../data/synthetic/full-surface-backbones/artifacts.json).
- [S12] [Joint 20k compiled gradient checks](../../docs/pilot-results/2026-10-28/rough-emulator-readiness/joint20k/readiness.json).
- [S13] [Current known 50k PyMC completion and fixed-case comparisons](../../docs/pilot-results/2026-10-28/known50k-current-pymc/terminal-summary.json).
- [S14] [Current known 50k reference target and Monte Carlo qualifications](../../docs/pilot-results/2026-10-28/known50k-current-pymc/README.md).
- [S15] [Matched known-curve 50k versus 100k learning comparison](../../docs/pilot-results/2026-10-28/known100k-completed-fit/comparison-summary.json).
- [S16] [Unchanged validation data and new model identity proof](../../docs/pilot-results/2026-10-28/known100k-completed-fit/bank-and-model-identity-proof.json).
- [S17] [100k pricing-emulator independent fixed-point rescore](../../docs/pilot-results/2026-10-28/rough-emulator-readiness/known100k/readiness.json).
- [S18] [50k and 100k neural estimators against fixed 50k PyMC chains](../../docs/pilot-results/2026-10-28/known100k-fixed50k-pymc/comparison.json).
- [S19] [Completed known-curve 100k validation evaluation](../../docs/pilot-results/2026-10-28/known100k-completed-fit/known100k/known/validation/evaluation.json).
