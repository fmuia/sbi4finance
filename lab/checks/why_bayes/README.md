# Where the posterior and the classical answer part on complete surfaces

Quick check of 28 October 2026. It looks for a case in the core tutorial, Heston and rough Bergomi on the complete
63-quote design, where the full posterior changes a product's 90% bound, and proposes how to test it. It is a quick
check, not a study: nothing here is registered in `tutorial/CLAIMS.md`, and no number goes on screen from it.
Sections 2 and 3 use the joint-curve rough-Bergomi task of PR #48 (`rough_joint_calibration_v1`,
`tasks.joint_rough_task`), which this branch does not have yet; their scripts run once that task is ported.

## Why products, not corner plots

On the complete design with the forward-variance curve given, both models are in the Bernstein–von Mises regime:
the posterior is close to the Gauss–Newton normal. Lecture 11, Section 11.4 and Lecture 15, Section 15.4 show it
on SPY, and PR #48's reference posteriors are largely elliptical, apart from skew where a chosen prior limit binds
(vol-of-vol at 1.5, ρ at −0.1, η at 3.5; PR #48, `docs/interior-corner-plots.md` and
`docs/pilot-results/2026-10-28/{heston,rough}-posterior-geometry/`). A banana in those tasks is not to be expected.

The object the rules ask for is a product's 90% point (Lecture 6, Section 6.6). A non-Gaussian parameter posterior
changes it only along directions the product loads on. A product linear in the parameters cannot separate a
posterior from a Laplace answer at all: PR #48's twelve product comparisons
(`docs/pilot-results/2026-10-28/heston-candidate-acceptance/results/products/products.json`) use the variance swap
and a forward variance, both linear in $v_0$ and $\theta$ at fixed $\kappa$. The check therefore scores products,
and looks where the quotes do not determine what a product needs.

## 1. Heston: the low vol-of-vol corner (study S6)

`heston_s6_strata.py` splits study S6's 200 test smiles by the fitted vol-of-vol, an observable, so a correct
posterior keeps its coverage inside each stratum. The posterior here is the BayesFlow 2.0 network's, not an exact
one.

| Fitted $\sigma_v$ | Smiles | Optimum on a bound | No Gauss–Newton interval | Two or more optima | Gauss–Newton over posterior 90% width, $\sigma_v$ | Same, $\rho$ |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| below 0.3 | 28 | 7 | 4 | 6 | 1.48 (10–90%: 0.84–3.10) | 1.36 (0.60–5.54) |
| 0.3 and above | 172 | 8 | 7 | 0 | 1.03 (0.95–1.15) | 1.01 (0.93–1.10) |

Below 0.3 the skew pins only $\rho\sigma_v$, and the valley of Lecture 7, Section 7.7 appears in Heston, cut by
$\rho = -0.95$ and $\sigma_v = 0.1$. Row 2115 shows it: truth $(\sigma_v, \rho) = (0.144, -0.833)$, least squares
$(0.209, -0.579)$, both with $\rho\sigma_v \approx -0.12$. Study S3's exact chains on row 2000 (true
$\sigma_v = 0.104$) show it with a reference posterior: $\sigma_v$'s 5%, 50% and 95% quantiles are 0.102, 0.125
and 0.182, and the sd of $\rho$ falls from 0.053 to 0.022 from the lowest to the highest decile of $\sigma_v$. The
corner is real model geometry, not a chosen limit. Whether it moves the tutorial's reserves was not checked: its
products depend on the vol-of-vol mostly through its square, which is small here, and SPY ($\sigma_v \approx 1.1$)
is far from it. A product priced by the vol-of-vol itself, such as an option on realized variance, would show it.

## 2. Joint-curve rough Bergomi: a local Gauss–Newton screen

`joint_gn_screen.py` computes, at eight parameter vectors, the expected information of the 63 design quotes with
the design noise. It uses the task's Monte Carlo pricer on one seed (32,768 paths per block, common random numbers)
and central differences. "Given" holds the curve at its true values; "inferred" estimates its three coefficients
with $(H, \eta, \rho)$. The 2-year 90% put is the tutorial's product beyond the last quote.

| Point: $(H, \eta, \rho)$; short, long, decay | $\eta$ sd, inferred over given | Long-variance sd over its prior width | 2-year put, delta-method sd: inferred, given |
| --- | ---: | ---: | --- |
| A, rising, slow decay: (0.09, 2.6, −0.73); 0.02, 0.06, 0.5 | 1.30 | 0.64 | 10.6%, 2.8% |
| B, rising, fast decay: A with decay 4.0 | 1.26 | 0.025 | 2.5%, 1.9% |
| C, flat: A's dynamics; 0.04, 0.04, 2.0 | 1.35 | 0.016 | 2.1%, 1.9% |
| D, falling: A's dynamics; 0.10, 0.04, 1.5 | 1.50 | 0.11 | 4.5%, 1.1% |
| E, $\rho$ near its bound: (0.07, 1.9, −0.95); A's curve | 1.16 | 0.54 | 9.8%, 1.9% |
| F, low vol-of-vol: (0.15, 0.8, −0.5); 0.03, 0.05, 1.0 | 1.08 | 0.14 | 6.4%, 1.2% |
| G, rising, very slow decay: A's dynamics; 0.02, 0.08, 0.3 | 1.30 | 1.73 | 11.9%, 2.8% |
| H, A's curve, low $\eta$: (0.12, 1.2, −0.6) | 1.16 | 0.51 | 9.7%, 1.8% |

The forward start's delta-method sd is 1.0% to 2.3% of its price inferred and 0.8% to 1.7% given, at every point.

- **Inferring the curve does not reopen the $\rho\eta$ valley.** The sd of $\eta$ grows by a factor of 1.1 to 1.5 and
  stays at 2% to 5% of its prior width; the wings still separate $\eta$ from $\rho$. The forward start and the
  volatility swap, which $\eta$ drives, do not carry the case.
- **The long end does.** When the curve still moves at one year (decay 0.3 to 0.5 per year), the quotes barely
  inform the long variance locally: its Gauss–Newton 90% interval crosses both prior bounds, decay's crosses the
  lower one, and a $\chi^2$ slice along the weakest direction leaves the prior box within one Gauss–Newton sd. The
  posterior there cannot be the Gauss–Newton normal. The 2-year put's delta-method sd rises four- to fivefold
  against the curve given. Section 3 measures what the posterior does instead.
- **A flat curve leaves decay unidentified and harms nothing:** the products do not depend on it.

## 3. One slow-decay smile, by importance sampling

`joint_is_posterior.py` computes the posterior at point A for one noisy smile (the design noise, seed 1), with the
task's pricer on one seed, by importance sampling; the target is the posterior of that fixed-seed pricer, as in
`rough_spy_is.py`. The proposal is a t distribution in the coordinates the quotes pin down, the curve's values at 6
and 12 months in place of (long variance, decay), mapped back in closed form with its Jacobian. 357 of 1,024
proposals are valid curves inside the prior box; the effective sample size is 98 and the largest weight 0.033, so
the 5% and 95% quantiles below are good to a few percentile points. A first attempt, with the Gauss–Newton normal in
the model's own parameters as the proposal, kept 370 of 2,048 proposals inside the box and reached an effective
sample size of 5: that normal is a poor description of this posterior. The Gauss–Newton column is the Laplace
approximation at the truth, which separates its shape from the location error of a least-squares point.

| Quantity | Truth | Posterior 5%, 50%, 95% | Gauss–Newton 90% |
| --- | ---: | --- | --- |
| $H$ | 0.090 | 0.080, 0.105, 0.130 | 0.064 to 0.116 |
| $\eta$ | 2.60 | 2.40, 2.54, 2.70 | 2.45 to 2.75 |
| $\rho$ | −0.73 | −0.81, −0.72, −0.64 | −0.82 to −0.64 |
| short variance | 0.0200 | 0.0191, 0.0198, 0.0205 | 0.0190 to 0.0210 |
| long variance | 0.060 | 0.030, 0.050, 0.088 | −0.097 to 0.217 |
| decay (per year) | 0.50 | 0.28, 0.66, 2.95 | −1.79 to 2.79 |
| 2-year variance-swap strike | 0.0347 | 0.0281, 0.0332, 0.0366 | 0.0244 to 0.0450 |
| forward variance, 1 to 2 years | 0.0409 | 0.0298, 0.0384, 0.0440 | 0.0217 to 0.0601 |
| forward start, 63 to 126 days | 0.02216 | 0.02140, 0.02213, 0.02280 | 0.02136 to 0.02296 |
| 2-year 90% put | 0.04151 | 0.03645, 0.03980, 0.04252 | 0.03426 to 0.04875 |

The two option rows come from `joint_is_products.py`: a systematic resample of 300 posterior draws (148 distinct),
priced with 8,192 paths on common random numbers. Their "truth" is the price under the true parameters. The
script's reserve lines measure from that price, while the posterior comes from a noisy smile, so they mix location
with shape; the comparison below uses widths and percentiles instead.

![Point A: the posterior of the curve's long end against the Gauss–Newton region, and two products beyond the last quote under each method](joint_is_A.png)

How to read the figure: on the left, the shaded area lies outside the prior box; the blue dots are 600 draws
resampled from the weighted posterior; the orange band is the Gauss–Newton 90% region at the truth (the star). The
posterior is a bent ridge, either a fast decay with a low long variance or a decay near its floor with a higher one,
and the straight Gauss–Newton band cannot follow the bend; most of it lies outside the prior. In the centre and on
the right, the blue histogram is the posterior's distribution of the product, the orange curve the delta method's
normal, the green line or curve the answer with the curve held flat beyond one year, and the dashed line the truth.
The histograms are spiky because the effective sample size is about 100.

- **$(H, \eta, \rho)$** have the Gauss–Newton widths, to within 4% to 15%: the regular part of the posterior, as on
  the known-curve task. The short variance's posterior range is 0.7 times Gauss–Newton's, through its coupling to
  the long end.
- **The long end** does not. The long variance's 90% range is a fifth of Gauss–Newton's, and decay sits near its
  floor (34% of the posterior below 0.5 per year) with a long upper tail. The quotes pin the curve's level and slope
  within the year and only weakly its curvature. Given the curve at 0, 6 and 12 months, the long variance is
  $(\xi_1 S - \xi_{0.5}^2)/(\xi_1 + S - 2\xi_{0.5})$, a ratio whose denominator is that curvature, and the prior's
  floor on decay limits how far the curve can keep rising.
- **The forward variance from one to two years** has a posterior 90% range 0.37 times Gauss–Newton's, and skewed:
  0.0086 below the median, 0.0056 above. No posterior mass lies outside Gauss–Newton's interval, which here
  over-reserves on both sides rather than missing risk. Holding the curve flat beyond one year gives 0.0324, a point
  21% below the truth and at the posterior's 13th percentile; the truth is at its 65th.
- **The 2-year 90% put** follows. The posterior's 90% range is 15% of the price, mildly skewed downward;
  Gauss–Newton's is 35%, 2.4 times wider. Holding the curve flat beyond one year moves the mark to 0.03844, 7.4%
  below the price on the true curve, with a range of ±4.3% from $(H, \eta, \rho)$ alone, which misses the truth. The
  truth lies at the posterior's 89th percentile.
- **The forward start**, inside the quoted year, is the control: Gauss–Newton's 90% range is 1.14 times the
  posterior's, and holding the curve flat beyond one year does not change its price.

## 4. SPY and the joint task's curve family

`spy_curve_fit.py` fits the joint task's curve, $\xi_0(t) = L + (S - L)e^{-kt}$, to SPY's seven replicated
forward-variance levels of 21 October 2026, with 5% error per level (the long-end check's assumption). The fit
prefers a fast rise to a plateau: decay 5.9 ± 1.1 per year, long variance 0.052 ± 0.003, $\chi^2$ = 9.8 on 7
levels. It misses the continued rise from six months to one year (0.0517 against 0.0546) and puts the forward
variance from one to 2.23 years at 0.052, against 0.0575 implied by SPY's 2.23-year LEAP, which the design does not
use. Slow decays fit worse ($\chi^2$ of 43 to 60 for decay 0.25 to 1).

Under this family SPY sits in the identified, fast-decay regime of Section 2. Its long end looks precise because
one exponential cannot express two time scales: a near-zero-width assumption about the curve beyond the last quote,
which the held-out LEAP contradicts. That is a misspecification lesson for slot 6, and it bears on the plan's check
of the hump-capable family, not a case for the posterior on SPY.

## What follows

**The case.** Joint-curve rough Bergomi, a product beyond the last quote, on smiles whose curve still moves at one
year. The two classical answers bracket the posterior. Holding the curve flat beyond one year, as the known-curve
task does, gives the long end no range and a mark biased by the extrapolation: at point A, 21% low for the forward
variance and 7.4% low for the 2-year put, whose range then misses the truth. Gauss–Newton with the curve inferred
gives a symmetric range 2.4 to 2.7 times the posterior's, for the curve products and the put alike, from a parameter
region that runs through negative variances and decays. The posterior's range lies
between them and is skewed. Its upper side is set by the prior's floor on decay (a half-life of at most 2.8 years):
beyond the last quote the range is the prior's as much as the quotes', and the slide must say so. Heston is the
control: with $\kappa$ fixed at 2 its curve reaches $\theta$ within the quoted year, so its 2-year put looks precise
because an input is fixed, the zero-width choice of Lecture 6, Section 6.7.

The argument is therefore not "the classical interval is too narrow". One classical answer invents precision, the
other spends capital on curves the model does not allow, and the size of each error shows in a reserve. The
posterior is right if its prior is, which is testable. The non-Gaussian shape sits in the curve's coefficients, not
in $(H, \eta, \rho)$, which is why the known-curve and Heston runs did not show it.

**The test, on the joint validation and test banks** (a proposed study; nothing here runs it):

1. Stratify the smiles by an observable, the posterior median of decay below 1 per year (or the long variance's
   contraction below 0.5), against the rest. Bayesian calibration holds within any stratum defined by the data, when
   the truths follow the prior. A rough count from the curve's second difference over the year,
   $(S - L)(1 - e^{-k/2})^2$, puts 6% to 10% of the prior in this regime: some 120 to 200 of 2,000 validation smiles.
2. For the 2-year put and the 1-to-2-year forward variance (analytic in the curve coefficients), report one-sided
   breaches of the 90% bounds, each of which should be 10%, and the reserves, for four methods: the posterior; least
   squares with Gauss–Newton and the delta method, from the price emulator; the curve held flat beyond one year;
   and the profile likelihood of each product. The last is the strongest classical competitor, since it follows the
   curvature and respects the box, at one constrained fit per product. If it matches the posterior, the report says
   so: the posterior's advantage is then calibration tested on average, every product from one set of draws, and an
   explicit prior.
3. Choose the exemplar by a fixed rule, the median smile of the stratum, and audit its posterior with the Monte
   Carlo likelihood (S5's machinery). The network's least-calibrated marginal is decay (46.75% coverage at 50% in
   the joint 20k run), the direction this case depends on.
4. Repeat the breaches with truths from another law for the long end, as S10 does for EWG, so that the posterior's
   advantage is not only its own prior.

**What it does not show.** That the posterior differs from Gauss–Newton on SPY's grid (Lecture 15, Section 15.4);
that point A is typical of its stratum (one smile, one noise draw); or that the joint task's prior on the long end
is right. Section 4 suggests the one-exponential family is too rigid for SPY.

## Files

| Script | Output | Runs on this branch |
| --- | --- | --- |
| `heston_s6_strata.py` | `heston_s6_strata.json` | yes |
| `joint_gn_screen.py` | `joint_gn_screen.json` | after PR #48's joint task is ported |
| `joint_is_posterior.py`, `joint_is_products.py` | `joint_is_A.npz`, `joint_is_A-products.npz`, `joint_is_A.txt` | after PR #48's joint task is ported |
| `joint_is_figure.py` | `joint_is_A.png` | yes, from the committed npz files |
| `spy_curve_fit.py` | `spy_curve_fit.json` | yes |

Run every script from `lab/`; the commands are in each docstring. The recorded outputs were made on 28 October 2026
with PR #48's source at `4281d4e`, outside the lab's locked environment: Python 3.13.17, NumPy 2.5.4, SciPy 1.18.1,
Numba 0.68.1, Matplotlib 3.11.2, on four cores. The screen took 63 to 98 seconds a point on one core; the importance
sampling about ten minutes and the product pricing about four, on four.
