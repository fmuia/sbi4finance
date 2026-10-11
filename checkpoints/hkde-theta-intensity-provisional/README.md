# HKDE theta/intensity checkpoint — provisional

This is the single 80-epoch spline baseline from the
[frozen protocol](../../docs/hkde-inference-protocol.md). It is retained for
reproducible review, **not accepted as the participant pricing checkpoint**.

- Input: nine ordered noisy European put prices `[batch,9,1]`, in currency.
- Output: matched `theta` and `jump_intensity` arrays `[batch,draw,1]` under Q.
- Task: independent uniform bounds `[0.025,0.065]` and `[0.25,1.0]`; all other
  model parameters, initial variance, contracts and market context fixed in
  `task.json`. Full task equality is required; matching dimensions is insufficient.
- Adapter: `hkde_ordered_currency_panel_v1`; bounded float64 logit transforms
  before float32 and training-only global normalization. No clean prices,
  truths or numerical diagnostics enter conditioning.
- Architecture: four spline coupling layers, widths 64/64, eight bins.
  Seed 6204, 80 epochs, batch 128; minimum loss on selection-validation IDs
  0–255 chooses the saved model. Diagnostic IDs 256–767 were not used for that
  selection. The bank's final-test split remains unevaluated.
- Runtime: BayesFlow 2.0.14, Keras 3.15.1, JAX 0.11.1 on CPU. Detailed hashes,
  task, training history and versions are in `training.json`/`protocol.json`.

Validation passes the declared marginal SBC and parameter checks but fails
the downstream price-quantile gate: **0.04387 > 0.03 currency**. Keep parameter
draws paired when pricing. Results are synthetic model-conditional evidence;
there is no market, P-risk, American/exotic or rough-memory claim.

See [measured results and graphs](../../docs/hkde-joint-inference.md) and the
[raw validation](../../docs/results/hkde-inference/neural-validation.json).
The notebook uses cached results and does not automatically load or retrain
this checkpoint. A failed gate cannot be removed by relabelling this card.
