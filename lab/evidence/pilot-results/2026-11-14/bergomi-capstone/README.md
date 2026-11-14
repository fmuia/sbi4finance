# Bergomi closing example: compact evidence

`results.json` is the compact plotting input to the closing section of
`lab/notebooks/heston_spx_calibration.py`: market aggregates plus synthetic
recovery coordinates and marginal quantiles. It contains no raw joint
posterior draws, simulated quotes, bank payloads, checkpoints, private paths,
or compute-cluster metadata. It is a
development result, not a scientific-acceptance or numerical-pricing certificate.

## Working posterior and matched comparison

Every displayed comparison uses the same 63 SPXW observations, contract order,
forwards, discounts and observed price half-spreads `h`. The likelihood is

\[
Y_i=P_i(\phi)+c h_i Z_i,\qquad Z_i\overset{\mathrm{iid}}\sim N(0,1),
\qquad c\sim\mathrm{Uniform}(0.5,50).
\]

Thus `c*h_i` is the price-error standard deviation. This is the spread-anchored
Gaussian model; it is not a common raw-price sigma model. The wider support was
informed by viewed market data. The earlier Heston lesson cells use `[0.5,2]`;
the final comparison explicitly uses separate Heston and Bergomi references
with common `[0.5,50]` support.

The Heston-versus-Bergomi fit curves average clean prices over 1,024 reference
draws each: 256 evenly spaced retained draws from each of four chains. RMS of
the posterior-average residual, divided by each quote's half-spread, is
**13.34705 for Heston and 9.62270 for Bergomi**, a 27.9% reduction. This is a
posterior-calibrated comparison on these quotes, not a prior predictive or
general model ranking. The Bergomi reference prices were independently
regenerated through the original canonical emulator and matched the preserved
comparison arrays exactly.

## The inference control

The Bergomi amortizer shown here is the **fixed-emulator training control**.
Its training mean and the MCMC likelihood use the same deterministic float64
mean-price function, with six physical/curve coefficients and `c` inferred
jointly. It is distinct from an amortizer trained on stochastic MC price labels.

The displayed checkpoint is the primary final **epoch 300**, with 200,000
training rows, batch size 256, 234,600 optimizer updates, and seed 9011. It uses
fixed offline noise realizations and training-fitted whitening of `Y/h`.
The inherited physical training subset was availability-selected; this evidence
does not establish prior-wide acceptance.

| Method | Median `c` | Central 90% interval |
| --- | ---: | ---: |
| Same-emulator MCMC | 10.27075 | [8.87263, 12.07703] |
| Fixed-emulator-trained amortizer | 4.75850 | [3.76548, 5.74047] |

All seven marginal summaries and the `c`/Hurst histograms use every raw draw:
8,000 MCMC and 4,096 amortizer draws. No draw is clipped or replaced; the raw
draws in this example have zero prior-support violations. The reader validates
the stored priors, target order, source identities and contract geometry.

The reference used PyMC NUTS, four chains and 2,000 retained draws per chain.
Its deterministic value/gradient checks passed. There were zero divergences;
maximum parameter R-hat was 1.002185, minimum bulk ESS 2,841.6 and minimum tail
ESS 2,530.4. The recorded requirements are R-hat at most 1.01 and both ESS values
at least 400. These diagnose the fixed-emulator posterior only.

## Predictives and interpretation

Both methods use 1,024 reproducibly selected paired `(physical, curve, c)`
draws. MCMC selection is balanced across chains. Clean-price bands evaluate
the shared frozen mean function; noisy-quote bands additionally generate
`clean + c*h*Z` using seed 20261114. The same Gaussian innovations are used for
both methods to make the comparison reproducible. The original inferred `c`
stays paired with the physical draw; there is no conditional `c` redraw.

The stored emulator output is in forward-normalized units,
`P_forward = P_raw/(D*F)`. Both clean prices and half-spreads convert by `D*F`
exactly once. The predictive panels use raw price units and show central 90%
pointwise bands. Newly generated noisy bands contain **57/63 viewed quotes for
MCMC and 40/63 for the fixed-emulator control**. These are descriptive checks
on calibration observations, not held-out or repeated-sampling coverage.

The working posterior still disagrees between methods despite their matched
mean function. Model misspecification does not by itself explain that
disagreement: an approximate financial model still defines a posterior the
amortizer should approximate. These results do not establish that additional
data, capacity or optimization cannot help.

## Synthetic parameter recovery

The closing section also shows all seven parameters on the exact-control
amortizer's saved **800 simulated development datasets**, using 256 posterior
draws per dataset. Each point is the generating value against its posterior
median; central 90% marginal widths are displayed for 40 cases per parameter,
spread across that parameter's generating range. The public record contains
only generating values, 5/50/95% marginal quantiles, finite-panel recovery
metrics and support counts. It contains no raw joint draws or quote payloads.

The preparer verifies the parent bank's manifest and payload hashes, checks
that the generating values match its preserved development partition, and
checks that those full parameter vectors do not occur in the 200,000 training
or 200 training-monitor rows. The truths and quantiles retain their common
dataset ordering. Recovery summaries are recomputed from the original,
unfiltered posterior draws and agree with the saved evaluation.

The physical bank was availability-selected, and this development panel has
already been inspected. These figures do not establish prior-wide acceptance
or real-market calibration. Good pooled synthetic recovery does not settle
the conditional neural/reference disagreement on the market observation.
The raw neural draws have some prior-support violations, which are reported
rather than clipped away. Marginal interval frequencies are descriptive
frequencies on these 800 cases, not a calibration certificate.

The displayed driving equations use the same normalized Volterra kernel,
variance compensator and correlated Brownian drivers as the trajectory lesson
in #75. Here the measure is risk neutral, deterministic carry is removed from
the price, and the mean variance curve is jointly inferred rather than supplied.
The mathematical compensator uses `q(t)=t^(2H)`; the finite-grid pricing labels
use the hybrid driver's actual grid variance. No equation change is used to
relabel the existing simulation bank or trained checkpoint.

The frozen mean emulator approximates **finite-grid MC price estimates**.
MC uncertainty, discretization bias and emulator error are excluded from the
displayed bands and remain numerically unqualified, especially near low `c`.
This is an approximate emulator posterior, not exact continuous-time Bergomi
inference. No fresh MC pricing, training or MCMC was run to prepare this section.

## Reproduction and source identities

From `lab/`, with private inputs supplied by their authorized operator:

```sh
python scripts/prepare_bergomi_capstone_results.py \
  --artifacts "$PRIVATE_EXPERIMENTS" \
  --frozen-prices "$PRIVATE_FROZEN_PRICES"
```

The private frozen-price NPZ and its sibling manifest must first be generated
with `Actual63Emulator.prices` in the preserved research checkout. The script
documents their schema, verifies source/payload hashes and exact posterior
selection, checks byte-identical public host observations/geometry, rederives
fit and posterior summaries, and publishes only aggregates. It duplicates no
financial kernel. Raw inputs remain private; reproduction requires those inputs.

All immutable source bindings are recorded by SHA-256 in `results.json`,
including the contracts, observations, spreads, fixed emulator, training task,
checkpoint, raw posterior payloads, source fit arrays and aggregation script.
The recovery block additionally binds its original development posterior
payload, checkpoint, preserved parent bank and ordered generating/quantile
summaries. Updating these public aggregates does not modify their raw sources.
The common emulator pricer hash is
`fbc7c5ad60205112a42b739a3ebb3a3fbb5f36a3f8b46723674895c15966053a`.
The exact-control checkpoint hash is
`b6ef39b38150a47af7e084511ad795726ad334ba6fe7ba6337db57ff5f615089`.
