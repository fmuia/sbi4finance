# Provisional Heston checkpoint with a signed asinh state transform

This candidate **failed first-stage validation** on root 43022: maximum mean
error, relative posterior-SD error and predictive-interval endpoint error exceed
the unchanged tolerances. The higher-draw stage and reserved final test were
not run. It is retained for reproduction, not promoted over the v2 candidate
or accepted for forecasting.

`task.json` identifies the restricted two-target P Heston task and adapter
`heston_returns_time_asinh_state_v3`. Observed inputs match v2: decimal log
returns with known end times and eight observed trailing moments. The state
target is transformed as `asinh(a / 0.01)` before global training normalization;
the saved inverse returns the original **signed** auxiliary variance. Keep
theta and state draws paired. Banks retain raw targets and returns.

Use `build_workflow` before loading: it registers the small serializable
`SignedStateAsinh` transform. Frozen CLI evaluation reads the saved adapter ID.
The complete architecture, scale and normalization are in `model.keras`.
Versions remain BayesFlow 2.0.14, Keras 3.15.1 and JAX 0.11.1.

See the [predeclared design and outcome](../../docs/heston-state-transform.md),
[validation report](../../docs/results/heston-state-transform/neural-validation.json)
and [review notebook](../../notebooks/05_heston_joint_inference.py).
The report pins checkpoint SHA256
`c21fe517c6fab643c63a34377bf13dcd6f3a56e10db9daf1764192dd55c1a7ac`.
