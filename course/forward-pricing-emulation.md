# Forward pricing emulation and fair inference comparisons

[Course index](README.md) · Companion to [Lecture 11, deep calibration](11-calibrating-heston.md#117-deep-calibration-and-the-neural-pricer)

A forward price emulator is a fast approximation to a model's numerical pricing function. We generate prices with the original financial model, train a regression network to reproduce them, and use that network inside calibration or MCMC. The emulator learns prices; the inference algorithm still determines which parameter values explain the observations.

For Heston, the original pricing calculation is a Fourier integral. For rough Bergomi it is Monte Carlo. The emulator replaces repeated evaluations of that calculation after an initial data-generation and training cost. This note explains how to obtain it, what the literature establishes, and what is needed for comparisons with amortized posterior inference when quotes are missing.

## Three different learned maps

Let `theta` denote model parameters, `xi` a forward-variance curve, and `y` the observed option prices. A model price is an expected discounted payoff under the pricing measure. An individual simulated payoff is not the price label.

| Method | Learned map | What happens for a new market observation |
| --- | --- | --- |
| Forward price emulator | Parameters, curve and contract information → model prices | An optimizer or MCMC algorithm evaluates the fast pricing map repeatedly. |
| Inverse point estimator | Observed prices and context → one parameter estimate | The network returns a calibration, without a joint posterior. |
| Amortized posterior estimator | Observed prices and context → a parameter distribution | The network directly generates joint posterior draws. |

The two posterior routes can use the same simulation bank but approximate different objects. The forward emulator amortizes pricing. The posterior estimator amortizes the inverse problem as well.

There are two common forward designs. A **fixed-grid emulator** returns a vector of prices on a declared contract grid. A **contract-conditioned emulator** takes strike and maturity as additional inputs and returns one price. The latter needs training and validation over those contract coordinates; a fixed-grid network does not acquire that ability through interpolation alone.

## How we train the emulator

The existing bank contains the main ingredients:

1. Draw parameters and, for rough Bergomi, a curve from the declared training distribution.
2. Evaluate the original numerical pricer on the contract grid, retaining the numerical-error records.
3. Use parameters and curve as regression inputs and the clean model-price estimates as labels.
4. Fit on the training split, choose the checkpoint on validation data, and assess price and inference accuracy on separate cases.

“Clean” means before adding the observation noise used to simulate market quotes. Monte Carlo labels still have numerical uncertainty. Reusing these labels avoids another large pricing campaign for the initial grid emulator; independent higher-precision validation adds computation.

Our current [implementation](../lab/src/sbi4finance/calibration/pricer_net.py) is a fully connected network with four hidden layers of 256 GELU units by default. Heston has four parameter inputs. The liquid rough task has three model parameters plus seven curve levels. Both predict the 63 design prices. Parameters are transformed into unconstrained coordinates, curve levels are logged, and training statistics standardize the inputs and outputs.

Prices are first divided by the design quote-noise standard deviations. This makes held-out errors interpretable relative to the likelihood's precision. The current shared trainer also standardizes outputs internally, then restores their scales in the loss so training and epoch selection use **quote-noise MSE**. Its `errors()` function reports discrepancies in those same units. Historical S4 records labelled `best_val_mse_standardized` retain their original meaning and are not directly comparable to the new loss. The [instructor guide](../lab/plans/full-surface-calibration.md#forward-price-emulation-and-rough-pymc-inference) gives the reusable training and PyMC commands.

The source and recorded [Heston pricer comparison](../tutorial/assets/results/calibration/heston/s4-exact/pricer-route.json) already exist. The new full rough-bank emulator and its validation are work to perform; their accuracy does not follow from the Heston result.

## Using the emulator for a posterior

Suppose the observation model is

```math
y = P(\theta,\xi) + \epsilon,\qquad \epsilon\sim\mathcal N(0,\Sigma).
```

The original likelihood evaluates the numerical prices `P`. The emulator likelihood substitutes its prediction:

```math
\widetilde p(y\mid\theta,\xi)
=\mathcal N\big(y;\widehat P_\psi(\theta,\xi),\Sigma\big).
```

MCMC combines this likelihood with the chosen prior. It samples an approximation to the original model's posterior. A converged chain removes neither emulator error nor the underlying model's numerical error.

There is another approximation when the bank's labels are Monte Carlo prices. Amortized training adds fresh quote noise to those estimates, retaining pricing randomness; the mean emulator's Gaussian likelihood includes only the declared quote noise. A perfect mean emulator therefore need not reproduce the finite-Monte-Carlo simulator posterior. Independent posterior-region checks must show that pricing randomness and covariance are negligible at the required accuracy, or the comparison must justify how they are treated. Validation residuals against the same numerical labels do not measure shared grid bias.

This is a learned mean function with an explicit observation-noise model. It need not learn a general likelihood density. It also does not generate the price distributions of arbitrary downstream products: a network trained on 63 vanilla quotes needs additional labels and validation to emulate an exotic product.

## Missing quotes

The forward map does not consume observed quotes. Given proposed parameters and curve values, it can predict all 63 prices even when only a few market quotes are observed. The likelihood should use only the observed subset, denoted `O`:

```math
\widetilde p(y_O\mid\theta,\xi,O)
=\mathcal N\big(y_O;\widehat P_{\psi,O}(\theta,\xi),\Sigma_{OO}\big).
```

For independent noise, the log likelihood is a sum over observed contracts. Missing values supply no residual and need no invented price.

For example, suppose three predicted prices in forward units are `(0.061, 0.043, 0.027)`. Only the first and third are observed, at `(0.062, 0.026)`, each with noise standard deviation `0.002`. Their standardized residuals are `0.5` and `-0.5`. The squared-residual sum is `0.5`; the unobserved middle contract contributes nothing, although the emulator predicted it.

This makes a forward emulator reusable across observed subsets of its grid. Missing quotes can nevertheless produce a broader or differently shaped posterior that reaches less accurate parts of the emulator's domain. Validate pricing and posterior accuracy under each sparse-data regime.

### What changes require more work

| Change | What the forward route needs |
| --- | --- |
| Some design-grid quotes disappear | Select the corresponding predicted prices and observation covariance. |
| Quote spreads or noise levels change | Use the actual observation noise; do not retain the training design's noise by accident. |
| New strikes or maturities appear | An original numerical pricer, a suitably trained contract-conditioned emulator, or a new validated output grid. |
| The curve is no longer supplied | Joint inference of curve and model parameters under their declared prior. |
| Quote availability is informative | A model or explicit assumption for the selection mechanism, applied equally to all methods. |

Treating an externally fixed/random subset as known design is a useful controlled experiment. If availability depends on latent quantities or unobserved prices, ignoring that mechanism is a modelling assumption, not general robustness to missing data.

## The thin curve task

The liquid rough task treats seven curve levels as known context and infers three parameters. The thin task instead infers all ten quantities jointly.

An emulator can accept curve values as forward inputs without those values being known observations: at every MCMC step, the chain proposes a curve along with the other parameters. The posterior must include the same structured curve prior as the amortized estimator. Giving one method a curve recovered from richer, unavailable quotes would change the information supplied to it.

The current EWG neural proposal reads a 27-point layout derived on the design grid. The real-data audit evaluates 61 cleaned contracts at their actual coordinates. These are different objects. Selecting outputs from the 63-grid emulator does not automatically provide an emulator for those 61 raw contracts.

## Current code and the adaptation still needed

The historical [S4 runner](../lab/studies/pricer_route.py) uses a complete grid and, for liquid rough Bergomi, a supplied curve. Its `log_posterior()` wrapper has no general observed-indices argument and does not implement ten-dimensional curve inference. The new shared [PyMC reference command](../lab/scripts/calibration_reference.py) supports both three-target known-curve and six-target joint rough calibration through a fixed emulator. It also requires complete quotes. Missing-data-aware emulator MCMC remains a downstream extension to build and test.

The implementation returns `price / design_sigma`, not unscaled forward-unit prices. For another layout, reconstruct prices using the design scale, select observed contracts, then use that layout's actual noise. Tests should also enforce:

- Slice observed entries before arithmetic; multiplying a missing `NaN` by zero does not remove it.
- Sum likelihood contributions. Averaging them by quote count changes the posterior.
- For correlated noise, select the covariance submatrix first. Selecting entries of the full precision matrix is not equivalent to marginalizing missing observations.
- Retain Gaussian normalization terms if the noise scale is inferred.
- Use the correct joint prior and transformation Jacobians when the curve is inferred.

The [thin-task code](../lab/src/sbi4finance/calibration/thin.py) supplies relevant priors, target definitions and noise conventions. The generic emulator wrapper still needs to connect them.

## What makes the comparison fair

For numerical-pricer MCMC, emulator MCMC and amortized posterior inference, use identical observed contracts, prices, joint priors, unknowns, known context and observation-noise assumptions. Specify the same numerical target, or measure and label the difference between targets.

A classical fit should receive matched observations, noise and parameter support, with its regularization, fixed parameters and curve-continuation assumptions stated explicitly. It need not use a Bayesian prior. A comparison of flat extrapolation with a Bayesian curve prior can teach the consequences of those assumptions, but it is not a test of whether two algorithms approximate the same posterior.

For the amortized estimator, missingness must be represented during training: either a fixed subset for a task-specific network or varying masks/design tokens for a flexible set encoder. Replacing absent prices with zeros in a network trained only on complete surfaces is not sufficient. The forward price map can often be reused when only the observed subset changes; that flexibility belongs in the cost comparison.

Use dense surfaces, random subsets, structured omissions of maturities/wings and the fixed EWG layout as separate controlled comparisons. Test reordered quote records, padding, variable noise and weakly identified posterior directions. Report retraining/adaptation cost as well as warm inference time.

## Pricing accuracy is only the first check

A small average pricing error does not give a general bound on posterior error. With independent Gaussian noise, define `r = (y - P) / sigma` and emulator error `e = (P_hat - P) / sigma`. Then

```math
\log\widetilde L-\log L
=\sum_c\left(r_c e_c-\frac12 e_c^2\right).
```

Errors can accumulate coherently across quotes and alter weakly identified directions. Validate held-out errors in noise units, errors in the posterior-relevant region, relevant price bounds/no-arbitrage relations, and posterior/product summaries against independent numerical calculations. A generic regression network does not enforce those price relations automatically.

For Heston, compare emulator MCMC against refined Fourier-likelihood MCMC. For rough Bergomi, combine independent higher-precision pricing checks, simulation-based audits and selected direct numerical reference cases. The emulator and posterior network share a bank; their agreement alone cannot rule out shared bias or bound either method's error.

## Established use and the tutorial's comparison

Forward neural pricing is established in the quantitative-finance research literature:

- Horvath, Muguruza and Tomas, *Deep Learning Volatility* (published 2021), train offline approximations to expensive pricing functions for fast calibration, including rough-volatility models. [Paper](https://arxiv.org/abs/1901.09647), [journal DOI](https://doi.org/10.1080/14697688.2020.1817974).
- Bayer, Horvath, Muguruza, Stemper and Tomas (2019), *On deep calibration of (rough) stochastic volatility models*, compare forward-emulation designs. Section 4.2.1 uses the neural map in an MCMC likelihood; Section 5.3 demonstrates Bayesian rough-Bergomi calibration on synthetic and SPX observations. [Paper and full text](https://arxiv.org/html/1908.08806v1).

These are precedents for the method, not evidence that every institution uses it in production or that our expanded curve task is already validated.

For the tutorial, emulator MCMC is a serious competitor to direct posterior estimation. Both pay an upfront simulation/training cost. One then runs a sampler for each observation; the other returns posterior draws directly. Compare their accuracy, adaptability and complete cost for the intended recurring workload. The useful result is learning when each route is appropriate.
