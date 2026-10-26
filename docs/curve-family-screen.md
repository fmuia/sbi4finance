# Bounded initial-curve price screen

This development screen challenges the proposed Heston-shaped exponential initial curve against the five-coefficient family in [Baschetti, Bormetti and Rossi, Section 4.3](https://arxiv.org/html/2306.11061v2#S4.SS3):

```text
xi(t) = b0 + b1 * exp(-t/tau1) + b2 * (t/tau2) * exp(-t/tau2)
```

It uses actual rough Bergomi option prices on the complete 63-contract design. It is a first price-level diagnostic, **not completion of the curve-family adequacy gate** and not a new training-bank family. The candidate six-parameter baseline prior and numerical settings remain provisional.

Three cases are declared before running: a flat control, an exponential monotone control, and a positive hump challenger. They share H=0.12, eta=1.5 and rho=-0.7. The generating coefficients are recorded in every request/report. Positivity is checked on the whole one-year horizon using endpoints and every possible interior stationary point, rather than checking endpoints or a discretized grid alone.

For each case:

1. Generate a numerical price label with a declared Monte Carlo stream, then one Gaussian quote-error realization using the task's existing noise model.
2. Fit all six baseline unknowns within the candidate prior bounds. A curve projection supplies only an optimizer starting point. The price objective reuses a fixed 2,048-path stream and is capped at 80 actual forward-price calls, including finite-difference calls. This makes optimization reproducible but permits overfitting low-path numerical noise; finite optimizer success is not evidence of a global optimum.
3. Audit the fitted and generating curves with two independent 32,768-path price estimates each, using streams absent from optimization and mutually independent between truth and fitted prices. Preserve the full sum covariance of their mean-price difference. Report price discrepancies relative to quote noise together with estimated Monte Carlo uncertainty and an approximate normal/Bonferroni band over the 63 contracts.
4. Record fitted/generating variance-swap and forward-variance **point values**. These are not posterior intervals or product-uncertainty validation.

All curve tables include every simulator node. The existing pricer's control-variate expectation uses the matching discrete left-cell sum, not the continuous analytic integral. A higher grid factor rebuilds the tables at the finer nodes. The continuous integrals are used for curve-derived point values only.

Run one case per bounded CPU job:

```sh
pixi run --locked -e core python scripts/check_curve_adequacy.py \
  --case flat --out /scratch/NEW-flat-screen
```

Repeat with `--case monotone` and `--case hump`. Options expose the fixed fitting/audit path counts, audit replicates, grid factor and evaluation limit. The output directory must be new. A pricing failure is recorded and not silently replaced by another seed. `request.json` survives an interrupted job; `report.json` records completion or an inconclusive failure. Keep Slurm's wall-clock limit as an external bound.

If the monotone or flat control fits poorly, diagnose optimizer and numerical precision first. A poor fit to the hump challenger can motivate a more flexible curve, but a capped single-start fit cannot prove that the entire six-parameter family fails. A successful fit is likewise only local evidence on this small panel.

Before choosing the production curve family, a limited sensitivity study must still compare relevant posterior curve/product uncertainty under explicit family-specific priors, the same observations/noise assumptions, and checked numerical targets. Narrower intervals alone do not establish superiority. The present screen always reports `scientific_acceptance: false`; it neither supplies a Bayes factor nor claims the two families yield equivalent joint posteriors.

## Controlled follow-ups

When a challenger fit hits a parameter boundary, first distinguish the candidate bound, numerical fitting noise and optimizer starting point. `--replay-report ORIGINAL/report.json` reuses the original observation vector exactly and checks that the truth case, quote design/noise and actual pricing grid are unchanged. It records both canonical-report and source-file hashes. It permits explicitly changed parameter bounds, without treating them as an approved production prior.

`--initial-parameters H ETA RHO SHORT LONG DECAY` supplies a physical starting point. `--audit-namespace NAME` creates new audit streams without changing observations or the deterministic fitting stream. Use the same fresh namespace across declared follow-ups for a common comparison; their audit results are then correlated across methods and should not be described as independent method replicates. Truth and fitted-price audit streams remain separate within each run.

The first hump follow-ups use 8,192 fitting paths and at most 160 price calls: current bounds from the previous best start; a candidate decay upper bound of 32 from the same physical start; and the widened bound from a contrasting declared start. `--decay-upper 32` changes that bound explicitly. Widening a bound also changes the normalized optimizer geometry, so a better found fit does not globally isolate the prior as the cause. Keep all outcomes, including failures, and do not select a winning development fit as final acceptance evidence. The audit remains two independent 32,768-path price estimates per side; replay separates that choice from the original numerical price label used to generate observations.
