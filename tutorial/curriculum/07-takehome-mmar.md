# Take home tutorial on multifractal returns

The take-home follows the application lessons. Its intended reading and exercise
route is about one hour, using the complete saved results. Optional model loading
and live training add hands-on work without blocking the main questions.

The central question is how a model of volatility clustering connects observed
returns, uncertain parameters and replicated drawdown scenarios. Students keep
one observed market window fixed throughout.

Open `notebooks/mmar_takehome.py` with `pixi run --locked mmar-takehome` from
`tutorial/`. The notebook is self-contained; it does not import the upstream
repository or require a live market-data service. Its saved results work without
a checkpoint download. Training and fresh inference are explicit actions.

**Observe and explain the model**

Start with 256 VOO simple returns, 3 January 2020 to 7 January 2021. Compare
the time series and its empirical tail frequencies with a constant-volatility
Gaussian benchmark fitted to the same returns. This selected window motivates
the model; it is not evidence of general forecasting superiority.

The model is

\[
r_t=\mu+\bar\sigma\sqrt{\theta_t(q,d)}\,\varepsilon_t,
\qquad \varepsilon_t=z_t\sqrt{(\nu-2)/U_t}.
\]

Here μ is daily location, σ̄ baseline scale, q cascade contrast and ν innovation
tail thickness. These four quantities are unknown. The known number of terminal
volatility blocks, d, controls their duration: each lasts 256/d days. Cascade
intensities average one; random split orientations and a circular shift generate
different realizations. Gaussian z and chi-square U generate standardized
Student-t innovations before the model's path-acceptance condition.

The component simulator connects intensity, innovations, returns and wealth.
Students should explain why q and ν represent different mechanisms even though
both can affect observed extreme returns. Detailed math and the whole-path
redraw rule for returns at or below −100% remain available beside the controls.

**Construct and check the estimator**

Use one conditional estimator throughout at d=8, the original eight
32-day blocks. The input diagram distinguishes the observed 256×1 return
sequence from the supplied scalar log d. A FusionTransformer learns 16 summary
values; these and log d condition a four-layer spline coupling flow over four
parameter coordinates. This is a fixed-window estimator, not the trajectory
lesson's masked arbitrary-prefix GRU.

The visible code explains simulation pairs, adapter constraints, conditions,
summary and density networks, offline fitting with independent validation,
restoration and sampling. The actual prior on ν−2 is log-uniform; bounds alone
do not describe it. Latent cascade weights and generating labels never enter
the conditions used for inference.

Inspect recovery, finite-rank calibration and interval coverage on independent
synthetic histories before interpreting the market posterior. A prior-only
baseline distinguishes learning from merely returning an uninformative prior.
Then compare the market posterior with the prior and check replicated return
magnitudes and temporal dependence. These checks answer different questions.

**Use the answer**

Propagate complete posterior rows through independently generated 256-day
histories. Compare replicated wealth and positive maximum drawdown with process
simulations at one posterior-mean parameter vector. Both calculations retain
process randomness. Threshold changes reuse the same scenario outcomes.

These are fresh-window scenarios, not continuations conditioned on the observed
window's final latent volatility phase. A drawdown event probability reflects
the fitted model, approximate posterior and finite simulation budget.

The notebook ends after section 7's drawdown scenarios with links to the expanded
story. Fresh results from a small session fit need their own checks; they do not
inherit the supplied estimator's evidence.

**Expanded story**

The model and lesson are inspired by
[amortized-fractals](https://github.com/stefanradev93/amortized-fractals), pinned
for this implementation at `069d5dc`. Its
[univariate introduction](https://github.com/stefanradev93/amortized-fractals/blob/069d5dc5943dbe85505849f2fa8bfb563617ff97/mmar_univariate.ipynb)
and [resolution experiment](https://github.com/stefanradev93/amortized-fractals/blob/069d5dc5943dbe85505849f2fa8bfb563617ff97/mmar_sensitivity.ipynb)
are the direct continuation. The full repository extends the story to rolling
risk, market-wide reuse, joint assets and signed-memory models. Those extensions
remain outside this concise take-home.
