# First implementation: exact GBM reference

This implements the small M0/M1 starting point from the [simulator plan](simulator-plan/README.md): simulate a synthetic return history, compute its exact drift posterior, and simulate future losses. A separate Q experiment checks European option prices. All inputs are generated or hypothetical; no market fixture, neural checkpoint or cluster is needed.

## Run it

```bash
pixi install --locked
pixi run test-gbm
pixi run gbm-demo
# Optional: save this one example's arrays and report to a NEW directory.
pixi run gbm-demo --out /tmp/gbm-reference
```

The default [configuration](../configs/tasks/gbm_drift.json) generates 252 observations at 1/252 year, infers drift with known 20% volatility, and generates 20,000 ten-step futures. This uses a synthetic trading-time convention. The kernels also accept explicitly supplied irregular year fractions; they do not infer a calendar from timestamps.

`--out` writes `returns_example.npz` and `report.json`. The NPZ contains P observations, known context, separate truth and predictive outputs; the JSON explains each array's role and records the configuration, RNG and NumPy version. This is one reproducible demonstration, not the M2 training-bank format or the data workstream's CSV/JSON observation fixture format. No existing output directory is overwritten.

The existing MMAR example and its checkpoint are unchanged. The GBM module imports only NumPy and the standard library. Neural training, custom loaders and model registries are not needed here.

## M0 conventions and worked records

| Quantity | Convention in this implementation |
|---|---|
| Drift `mu` | Annual proportional price drift: `dS/S = mu dt + sigma dW`. Expected log-return drift is `mu - sigma²/2`. |
| Volatility `sigma` | Annual decimal standard deviation; 0.20 means 20%, variance is 0.04 per year. Known in the first inference task. |
| Observation | Decimal log returns, float64 `[batch, time, 1]`. One complete sequence is one inference example. |
| Time | Positive finite 1D `dt` in years, one interval per return, shared within the batch. No missing-value fill, masks or per-window standardization. |
| Initial/continuation state | A known positive price. Constant-volatility GBM has no hidden variance state. Relative future returns use the same sampled drift throughout each path. |
| P inference target | Drift only, `[batch, 1]` in the saved example. Gaussian prior with explicitly given mean and positive standard deviation. |
| Q pricing context | Forward `F`, discount factor `D`, strike `K`, integrated variance `w`. The P drift posterior is not passed to the pricer. |
| Randomness | Explicit NumPy Generator in the simulator. The example uses independent PCG64 streams for history, posterior draws, future shocks and Q pricing. |
| Origin and claims | Synthetic, S only. Known synthetic truths and future outcomes never enter observed conditioning. |

The first worked returns record is a batch of one history with `dt=[0.25,0.75]` and returns `[[[0.05],[-0.02]]]`. With known sigma 0.2 and prior `mu ~ Normal(0,0.1²)`, the posterior mean is 0.01 and variance is 0.008. The information cutoff is one year; the record's synthetic generating drift, if recorded, is a separate label. The calculation uses only the two returns and declared context.

A worked **option-panel contract sketch** has Q context `F=100 exp(0.03)`, `D=exp(-0.05)`, one-year expiry and strikes `[90,100,110]`. With known integrated variance 0.04, the clean call-price row is approximately `[15.123708,9.227006,5.188582]`, shaped `[1,3,1]` for a panel observation. This milestone computes those clean expectations but does not implement noisy panel generation or an option-calibration posterior. In that later task, the noise law and inferred Q quantities must be explicit; the clean prices/variance truth stay separate from the noisy conditioning panel.

Both worked tasks distinguish observation, context, optional synthetic truth and output. A real return fixture can later supply complete compatible arrays and its declared `dt`; it supplies no known drift/state labels. A successful parse alone does not establish model or checkpoint compatibility.

## Small function interface

The implementation stays together in [gbm.py](../src/sbi4finance/gbm.py) while this reference is small. Scientific responsibilities remain separate functions; no empty packages or general framework are introduced.

| Function | Input → output |
|---|---|
| `simulate_log_returns` | Explicit drift, volatility, time intervals, RNG → `[B,T,1]` exact increments |
| `prices_from_log_returns` | Complete log-return history and initial price → `[B,T+1,1]` positive-price path, including its start |
| `drift_log_likelihood` | History, candidate drift, positive known volatility, intervals → Gaussian log likelihood per history/candidate |
| `drift_posterior` | History, positive known volatility, intervals and Gaussian prior → posterior mean/std `[B]` |
| `predictive_log_return` | Drift posterior mean/std, volatility, horizon → analytic total log-return mean/std `[B]` |
| `european_price` | Broadcastable `F,K,D,w`, call/put → Q reference prices; supports zero integrated variance |

Scalar or one-dimensional drift/volatility inputs broadcast across simulation paths. The inference history layout stays `[B,T,1]`, including singleton axes; squeezed or incomplete inputs fail explicitly. Likelihood and posterior formulas exclude zero volatility, while simulation and pricing include the deterministic limit. A zero analytic forecast horizon produces zero return.

The simulator consumes normal draws in time-major order, so successive calls with the same Generator and batch order reproduce the corresponding parts of an uninterrupted path. This does **not** promise identity under reordering batch members or distributing examples between workers. Stable per-example streams and bank splits are M2 work; no scheduler-specific randomness is hidden in this kernel.

## What the reference checks

The [mathematical guide](simulator-plan/notes/01-foundations.md#an-analytic-first-inference-problem) derives the posterior. For total time H and total observed log return R:

```text
posterior_variance = 1 / (1 / prior_variance + H / sigma²)
posterior_mean = posterior_variance * (prior_mean / prior_variance + (R + sigma² H / 2) / sigma²)
future_log_return_mean = (posterior_mean - sigma²/2) h
future_log_return_variance = sigma² h + posterior_variance h²
```

The future sampler draws **one drift per future path** and then new process shocks. Drawing a new drift every day would give a different predictive law. Loss is `1 - future_price/current_price`; the reported 99% VaR is a quantile of this predictive loss distribution, expressed as a fraction of current value. It is not an average of conditional VaRs or a market backtest.

For deterministic variance, Q pricing depends on `w = integral sigma(t)² dt`, with financing summarized by F and D. Zero w gives the discounted deterministic payoff. The example's Q simulation uses independently specified `rate - dividend_yield` as drift; it is a numerical pricing check, not a P/Q calibration link.

The tests check independent numerical likelihood/pricing fixtures, numerical integration of prior times likelihood, Gaussian moments/covariances, exact seeded continuation, posterior-predictive moments, shape semantics and meaningful invalid inputs. The demo additionally compares simulated forecasts and option payoffs with analytic references, recording sampling error and one-run timing rather than a portable performance claim.

## Initial result and remaining scope

With the committed configuration and NumPy 2.5.3, the generated history gives drift posterior mean about 0.06470 and standard deviation 0.08944. The ten-step predictive loss VaR is about 0.08724 analytically and 0.08735 by simulation. The three simulated option prices differ from their references by about 0.85–1.21 Monte Carlo standard errors. These are checks of this declared synthetic experiment, not claims of real-market adequacy.

The 20,000 ten-step forecast paths took about 3 ms to generate in the initial local run, excluding process startup; rerunning records a new timing. NumPy is sufficient for this first reference. More extensive benchmarking and acceleration can wait for a workload that needs them.

M0's GBM conventions and the two observation-role sketches are resolved. Later-model priors, numerical tolerances and state targets remain decisions for their respective milestones. M1 now supplies the exact baseline and seeded checks. Next, M2 can reuse these functions for a small saved synthetic bank and training round trip, followed by a real cluster smoke job. No inference network, Heston/SVJ/Bergomi kernel, market adapter or cluster execution is claimed by this increment.
