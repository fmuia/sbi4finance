# Synthetic option-panel banks

The option branch now has a small reusable boundary: draw parameters, estimate
European prices, add declared quote noise, save the whole panel, and reload it
with its context and targets. The first configuration uses HKDE. Later Bergomi
increments reuse these functions; there is no model registry, storage service
or neural adapter in this increment.

## What constitutes an example

One example is nine European put prices: strikes 90, 100 and 110 at maturities
0.25, 0.5 and 1 year, in maturity-major order. Spot is 100, the continuously
compounded rate is 0.02/year, and dividend yield is zero. All are **known
synthetic Q context**. The observation is

\[
y_j=\widehat P(\theta;c_j)+\epsilon_j,\qquad
\epsilon_j\overset{\mathrm{iid}}\sim N(0,0.25^2).
\]

Prices and quote noise use currency units. Negative or arbitrage-inconsistent
noisy values are possible under this deliberately simple Gaussian law; no
clipping or rejection changes that law. This is a controlled experiment, not
a market cleaning policy. The prior is an independent uniform box in the
configuration's explicit `parameter_names` order. Each model has its own task
identity, prior, known model context and numerical controls.

| Saved field | Role and shape |
|---|---|
| `observations` | Conditioning: noisy panel `[batch,contract,1]` |
| `parameters` | Synthetic supervised target `[batch,parameter]`; never an observed input |
| `reference_prices` | Auxiliary expectation estimates `[batch,contract]` |
| `pricing_standard_errors` | Auxiliary MC sampling errors, same panel shape; Fourier zeros mean no sampling |
| `pricing_covariance` | Auxiliary covariance of the **price means**, `[batch,contract,contract]` |
| `refinement_difference` | Auxiliary refined-minus-base Fourier prices; MC zeros mean **not calculated** |
| `example_id` | Split-qualified base-panel/group identity; never conditioning |

Contracts, spot, rates, quote sigma and known model context are stored once in
the manifest's full configuration. There are no hidden latent inputs or future
evaluation outcomes in the conditioning set. If later augmenting a panel with
additional noise, keep its base surface in one split.

HKDE draws `theta` and `jump_intensity`; other model parameters and initial
variance are fixed. Refined Fourier estimates are the labels. The maximum
base-to-refined difference must be at most 0.001 times quote sigma, or 0.00025
currency. Doubling nodes **and** cutoff guards against two different sources
of quadrature error; this difference is still an indicator, not a rigorous
error bound. See the independent checks in [HKDE reference](hkde-reference.md).

An MC bank retains sampling error separately from quote error and grid bias.
These stored estimates do not become mathematically exact labels by being
called `reference_prices`. A bounded-put estimator supplies justified finite
sampling variance; call values can use parity in the supported martingale
domain. One simulated payoff is not a price label. Numerical errors shared
across contracts are correlated and must not be interpreted as independent
quote noise.

## Generate and inspect

```bash
pixi run option-bank --config configs/tasks/hkde_option_bank.json --out /tmp/new-hkde-bank
pixi run review-hkde
```

The default pilot has 16 training, four validation and four test examples.
This checks the interface and numerical labels; it is far too small to train
a useful neural posterior. `generate_examples(config, split, ids)` reproduces
arbitrary IDs independent of request ordering, chunking and split counts.
Task settings, root seed, split and ID determine separate prior/pricing/noise
streams. Changing a scientific or numerical setting changes the task identity.

The same `scripts/generate_option_bank.py` command writes NPZ files and a JSON
manifest, checks file hashes, shapes, parameter support and MC covariance,
then exactly regenerates first/middle/last examples in reverse order. It saves
`verification.json` beside the bank. Existing directories are never overwritten.
A failed numerical draw saves its original identity in `failure.json` and
stops: there is no silent redraw or conditional-prior selection.

The manifest records S origin, Q task, parameter order, conditioning cutoff,
contract selection, truth availability, source hashes and Python/NumPy/SciPy
versions. Exact regeneration is a same-environment check. Transferring a bank
to another platform preserves file integrity but needs a separately reported
numerical reproduction comparison; binary equality across platforms is not
promised. Numerical screens, posterior checks and empirical model assessment
remain separate.

## CPU cluster path

`cluster/option_bank.sbatch` calls the same CLI, with one CPU, 4 GB RAM and a
10-minute limit. Submit from a committed source checkout, using the actual
allocation discovered by the existing cluster instructions:

```bash
sbatch --account="$BANK_ACCOUNT" --partition="$BANK_PARTITION" --qos="$BANK_QOS" \
  cluster/option_bank.sbatch /absolute/path/to/python /new/scratch/bank configs/tasks/hkde_option_bank.json
```

Use a Python 3.12 environment containing the pinned NumPy 2.5.3, SciPy 1.18.1
and Numba 0.68.0, or the locked default Pixi environment. No package installation
occurs in the job. Keep its source revision, manifest, verification file and
scheduler accounting. This new script has local CLI and shell-syntax checks;
an actual option-pricing job on the cluster is still pending. The earlier
[GBM cluster smoke](gbm-cluster-smoke.md) establishes the connection, not this
workload's Linux performance. No scaling/orchestration framework is required
before measuring that job.

## What comes after the bank

For posterior estimation, training pairs are `(noisy panel, known context)` →
synthetic parameters. A fitted posterior can then feed a validated target
pricer. A pricing surrogate instead learns parameters/contracts → accurate
expectation estimates; a direct-price posterior learns noisy panel → target
prices generated from the same parameter draw. Those neural workflows and
their conditional/reference checks remain later milestones.

The full generation/review path is **S**, runs offline and needs no real data.
Optional **M** conditions a trained model on a compatible market panel; actual
parameters are unknown. Optional **E** compares against excluded target quotes
with leakage controls. Synthetic numerical success does not certify real
model adequacy, P risk forecasts, future option-chain dynamics or American/
exotic pricing. See [analysis to resume](simulator-analysis-next-steps.md).
