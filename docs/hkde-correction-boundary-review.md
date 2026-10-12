# HKDE correction: a boundary-coverage diagnosis

This note records the analytical reason for one new, fixed proposal after the
first correction run stopped. It does not change that run's acceptance result,
the posterior target, or any gate. The original
[correction protocol](hkde-correction-protocol.md) remains the record of the
first attempt. No new simulation or counterfactual acceptance calculation was
used for this diagnosis.

## What the stopped run shows

The `hkde_defensive_importance_v1`
[report](results/hkde-correction-v1/validation.json) and
[saved proposal clouds](results/hkde-correction-v1/validation-checks.npz)
show that all six conditional accuracy gates passed at
both 2,048 and 8,192 proposals. The worst target-price quantile errors were
0.012607 and 0.005773 currency units, respectively, against the unchanged 0.03
gate. The larger pool stopped on proposal reliability: minimum ESS/N was
0.333597 rather than at least 0.5, and its largest weight was 0.014938 rather
than at most 0.01. SBC and the new test were not evaluated.

The issue is concentrated in `stress:2`, zero-based proposal row 5,675:

| Saved quantity | Value |
|---|---:|
| Long-run variance, theta | 0.06144575248 |
| Jump intensity, lambda | 0.2501578861 |
| Position above lambda's lower bound, as a fraction of its prior width | 0.000210515 |
| Proposal component | Uniform prior |
| Neural density / uniform-prior density | 0.052007 |
| Numerical-reference posterior density / prior density | Approximately 18.1477 |
| Normalized importance weight | 0.01493848 |
| Share of the sum of squared weights | 60.985% |

Its log likelihood is only 0.00146 below the largest likelihood in the saved
pool. Thus this is an important region near a parameter boundary that the
neural density undersupplies. It is not merely a remote, irrelevant tail point.
The reference-density ratio uses the accepted grid's numerical evidence.

The full pool's absolute ESS is 2,732.82, versus 1,739.52 for this case's
2,048-proposal prefix. Its accuracy also improves. The stop therefore means
that the declared proposal-reliability criteria were not met; it does not show
that more draws made the posterior estimate worse.

## One fixed boundary-covering component

Let `a = (theta, lambda)`, with bounds `l = (0.025, 0.25)` and widths
`d = (0.04, 0.75)`. The target prior remains the uniform density
`p(a) = 1 / (d_1 d_2)`. Write `u_j = (a_j - l_j) / d_j` and define a proposal
component by independent `Beta(1/2, 1/2)` coordinates:

\[
h(a)=\frac{p(a)}{\pi^2\sqrt{u_1(1-u_1)u_2(1-u_2)}}.
\]

This is a normalized **joint density in the original parameter units**. The
factor `p(a)` is the coordinate-change Jacobian; omitting it would invalidate
the mixture. Both `h`, the uniform prior `p`, and the neural density `q` have
units inverse to the product of the two parameter units. BayesFlow's `q`
already includes its adapter and standardization Jacobians.

The proposed second estimator uses independent categorical component draws:

\[
g_2(a\mid y)=0.8q(a\mid y)+0.1p(a)+0.1h(a),\qquad
w_i\propto \frac{L(y\mid a_i)p(a_i)}{g_2(a_i\mid y)}.
\]

Every draw uses the entire mixture denominator. The arcsine component changes
the proposal only: it does not introduce a new prior or change the likelihood.
It covers all four boundaries symmetrically, with no fitted parameters or
case-specific region. Boundary endpoints have probability zero under the
continuous proposal; numerical invalid draws must not be silently clipped.

Two pointwise facts explain this choice. For the original
`g_1 = 0.9q + 0.1p`,

\[
g_2\geq\frac89g_1,\qquad
g_2\geq\left(0.1+\frac{0.4}{\pi^2}\right)p
\approx0.140528p.
\]

The first bound limits the density reduction anywhere to 11.1%; the second
improves the minimum defensive density. At the saved troublesome point,
`h/p = 24.5452`: `g_2/p = 2.59613`, compared with `g_1/p = 0.146806`.
The denominator there is 17.68 times larger. This is a pointwise analytical
comparison, **not** an estimate of the new run's ESS, normalized weights,
accuracy or acceptance probability.

## What must still be established

The [second protocol](hkde-correction-boundary-protocol.md) and its
[settings](../configs/hkde_correction_boundary_protocol.json) freeze this
single new mixture before execution, retaining the checkpoint, likelihood,
uniform prior, budgets, seeds and all gates. The first attempt remains stopped.
The second attempt must pass both budget
checks, finite-pool SIR calibration, and the still-unseen new test before any
release claim. The same test seed remains available because no test outcomes
were generated. No proposal selection, seed search or gate relaxation follows
from this note. Neither a successful correction nor a high ESS would certify
the raw neural posterior, real-market adequacy, or the final curriculum choice.
