# Reviewing the current tutorial backbones

The participant runtime is self-contained under `tutorial/` and uses BayesFlow
2.1. The primary editing files are the Heston and rough-Bergomi notebooks;
`tutorial/notebooks/backbone_review.py` supplies the detailed cached/live review.
No notebook was executed during this integration.

Run the review from `tutorial/`:

```bash
pixi install --locked
pixi run backbones
```

For optional authoring through marimo Pair/MCP and Studio/Lens, also from
`tutorial/`:

```bash
pixi run --locked -e collaboration collaborate notebooks/backbone_review.py
```

The published [data/model downloads](training-data-guide.md) are independent of
Oscar access. The training guide's commands run from `lab/`; the participant
commands above run from `tutorial/`.

## Cached examples and their scope

The catalogue `tutorial/assets/backbones/artifacts.json` binds the exact cache,
task and model hashes. Cached viewing imports neither a neural backend nor a
model. It does not train, reprice observations or run MCMC.

| Example | Inferred quantities | Observed conditions | Evidence scope |
| --- | --- | --- | --- |
| Heston final500 | Initial variance, long variance, vol-of-vol, rho | Complete 63 quotes and fixed design/noise | Interior development example with its saved PyMC/analytic comparisons |
| Known-curve rough 100k | H, eta, rho | Complete quotes plus seven known curve levels | Saved validation:0 replay; no new conditional reference |
| Joint rough 100k | H, eta, rho, short variance, long variance, decay | Complete quotes/design/noise; no generating curve | Saved validation:0 and selected cases 1983, 1941 and 1962; conditional PyMC remains pending |

The joint interior candidates show the full six-dimensional posterior and its
curved trade-offs. The weak-decay example illustrates limited identification
for a nearly flat curve. Selection and scientific limits remain in each
bundle; copying a development replay does not promote its acceptance status.
See the [posterior sanity report](joint-rough-posterior-sanity.md) and
[completed joint 100k comparison](../evidence/pilot-results/2026-10-29/joint100k-completed/README.md).

Historical known50k/joint20k caches remain byte-identical under `lab/evidence/`.
Historical SPY outputs keep their own task, checkpoint and numerical provenance;
they are not relabelled as predictions from the new 100k estimators.

## Optional fresh inference

Download the pretrained-and-cases archive named in the training-data guide.
Use the lab's `studies/accept.py --checkpoint SOURCE --to local-checkpoints/NAME`
route to import a matching native checkpoint under `tutorial/assets/`, or follow
the participant README's explicit download-cache instructions. The native model,
`task.json` and `training.json` stay together and retain their original hashes.
No optimizer stripping or metadata substitution is applied to BF2.1 exports.

The notebook's live section requires a matching checkpoint path below
`tutorial/assets/local-checkpoints/` and an explicit button press. Its loader
checks task and model identity before drawing 1,000 fresh samples for the exact
cached observation. Cached draws retain their original batched sampling seeds;
a fresh call need not reproduce that batch bit-for-bit. BayesFlow's native
adaptive sampler is retained.

## Known curves and summary networks

For the known-curve task, seven log curve levels are separate global observed
context. Vector, DeepSet and SetTransformer encoders summarize the quote inputs;
they do not infer that known curve. Fresh quote noise changes only prices.

For the joint task, the stored generating curve is an auxiliary label and is
excluded from inference. Set inputs still represent complete surfaces; missing
quotes remain a separately specified follow-up. The copied runtime/parity,
asset-hash and conditioning tests protect these boundaries.
