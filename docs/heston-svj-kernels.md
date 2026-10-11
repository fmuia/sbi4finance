# Heston and Gaussian-price-jump kernels

This implements **M3.1–M3.2**, the fourth of ten envisioned simulator-plan PRs:
one batched NumPy reference and one serial Numba kernel, with the same explicit
random inputs. It does not yet add a Heston/SVJ bank or posterior network.
The live tutorial's returns-versus-options choice remains open; these are
shared model building blocks.

## Run the small synthetic examples

```bash
pixi install --locked
pixi run test-heston
pixi run heston-validate --out /tmp/heston-validation.json
pixi run heston-benchmark --out /tmp/heston-benchmark.json
```

Output files must be new. Both scripts work without market observations,
network access or a checkpoint. The benchmark uses the declared
[P pilot configuration](../configs/tasks/heston_svj_pilot.json); its history and
ten-observation continuation use known parameters. This is a numerical **S**
example, not inferred volatility or an empirical forecast. The refinement
script separately declares its regular, near-zero and jump cases.

For a genuinely cold compilation measurement, set `NUMBA_CACHE_DIR` to a new
empty directory **before** starting the benchmark. Otherwise its first call
may load a compiled cache. Imports are outside the measured interval. Warm
timings include argument checks, output allocation and stepping; complete
timings additionally include random-input generation, but exclude bank IO.

## Model and units

Write \(X_t=\log(S_t/S_0)\). The continuous reference model is

\[
dV_t=\kappa(\theta-V_t)dt+\xi\sqrt{V_t}\,dW^V_t,
\qquad d\langle W^S,W^V\rangle_t=\rho\,dt,
\]
\[
dX_t=(b-\lambda c-\tfrac12 V_t)dt+\sqrt{V_t}\,dW^S_t+J\,dN_t,
\qquad J\sim\mathcal N(m_J,s_J^2),\quad
c=e^{m_J+s_J^2/2}-1.
\]

Here \(N\) is a Poisson process independent of the Brownian drivers, and its
jump sizes are independent normal **log-price** jumps. There are no variance
jumps. Setting \(\lambda=0\) gives Heston. This is not SVCJ and is not yet a
reproduction of Francesco's particular study.

Time is in years; observed returns are decimal log returns. `drift` is \(b\),
the total expected proportional price-growth rate including jumps. Under P
the task supplies \(\mu\); under Q it supplies \(r-q\) and separately chosen
Q parameters. The compensator preserves that drift convention. Passing P
parameters does not make them risk-neutral or produce option-pricing labels.

The other parameter names are `kappa`, `theta`, `vol_of_vol` (\(\xi\)), `rho`,
`jump_intensity`, `jump_mean` and `jump_std`. The last three default to zero.
Parameters may be scalars or vectors with one value per example. The kernel
requires \(\kappa>0\), \(\theta,\xi,\lambda,s_J\ge0\), and \(|\rho|\le1\).
Finite values and outputs are checked. These algebraic supports are broader
than the numerically checked pilot domain; they do not certify arbitrary
parameters or timesteps.

## Discrete law and complete continuation state

An observed interval \(\Delta_t\) has `substeps` internal updates of length
\(h=\Delta_t/L\). Let \(a\) denote the signed auxiliary variance and
\(v^+=\max(a,0)\). With independent standard normals \(Z_V,Z_\perp\), each update
uses the same **pre-update** \(v^+\):

\[
\Delta X=(b-\tfrac12v^+)h+
\sqrt{v^+h}\bigl(\rho Z_V+\sqrt{1-\rho^2}Z_\perp\bigr),
\qquad
a'=a+\kappa(\theta-v^+)h+\xi\sqrt{v^+h}\,Z_V.
\]

This is full-truncation Euler, including retention of the signed auxiliary
state. Continuous variance is nonnegative, but this numerical auxiliary can
be negative. Reported/effective variance is `maximum(a, 0)`; **restart from
`a`, not its clipped value**. Clipping the restart state changes the discrete
transition law. The same convention will be required in any same-model filter
or posterior target. A future choice of a different scheme will require a
new task specification and regenerated training examples. See
[Lord, Koekkoek and van Dijk](https://repub.eur.nl/pub/18571).

At the end of each observed interval, draw the full count
\(n\sim\mathrm{Poisson}(\lambda\Delta_t)\) and add

\[
n m_J+s_J\sqrt{n}\,Z_J-\lambda c\Delta_t
\]

to its accumulated log return. This sums multiple independent Gaussian jumps
exactly conditional on the count. Aggregation over the observed interval is
valid because these jumps do not affect variance. Individual jump times and
within-interval extrema are **not represented**: this output is insufficient
for a continuously monitored barrier pricer. No individual payoff is asserted
to be an accurate price label.

## Small function interface

[`heston.py`](../src/sbi4finance/heston.py) exposes two functions, without a
model registry or new adapter framework:

```python
import numpy as np
from sbi4finance import heston

parameters = dict(drift=0.05, kappa=2.0, theta=0.04,
                  vol_of_vol=0.3, rho=-0.6,
                  jump_intensity=0.7, jump_mean=-0.07, jump_std=0.1)
dt = np.full(252, 1 / 252)
ids = np.arange(32)
noise = heston.draw_innovations(
    dt, 8, seed=7412, example_ids=ids,
    jump_intensity=parameters["jump_intensity"], stream_id=0,
)
history = heston.simulate(parameters, dt, 0.04, noise, backend="numba")

future_dt = np.full(10, 1 / 252)
future_noise = heston.draw_innovations(
    future_dt, 8, seed=7412, example_ids=ids,
    jump_intensity=parameters["jump_intensity"], stream_id=1,
)
future = heston.simulate(parameters, future_dt,
                          history["variance_state"][:, -1, 0], future_noise)
```

| Array | Shape | Role |
|---|---|---|
| `log_returns` | `[batch, observations, 1]` | Observed conditioning, with a declared cutoff and positive `dt` context |
| `variance_state` | `[batch, observations + 1, 1]` | Signed auxiliary truth, including the initial state; a terminal inference target/continuation input when the task declares it |
| `brownian` | `[batch, observations, substeps, 2]` | Prescribed independent standard normals, simulation-only |
| `jump_counts`, `jump_normals` | `[batch, observations]` | Simulation-only jump inputs, not observed labels for a real window |

Use **the same intensity** when drawing inputs and simulating. Prescribed
inputs make backend comparison and coupled numerical refinement possible;
the kernel checks their shape and values, not whether their distribution is
correct. `dt` supports unequal positive year fractions; all examples in a
batch currently share that grid and the internal step count.

PCG64/SeedSequence streams separate each unsigned 64-bit example ID, root
seed, stream ID and innovation purpose. Batch order and chunk size therefore
do not change an example with the same task, ID and stream. Use distinct
streams for histories and independently simulated futures; exact restart
tests instead slice one already-drawn trajectory's inputs. Internal refinement
also needs explicitly coupled increments, as in the validation script.

Inputs occupy \(O(BTL)\) memory, so generate modest batches rather than a
whole bank at once. The default timing batch uses about 8.9 MiB of innovation
arrays and 1.0 MiB of output arrays, excluding temporaries and compiler memory.
Numba is serial, float64, with `fastmath=False`; there is no parallel RNG or
thread-pool layer.

## Validation and measurements

The seven focused test groups cover hand-computed truncation, negative-state
restart, matched NumPy/Numba outputs, exact constant-variance GBM limits,
one-step variance/return moments and leverage covariance, compound-Poisson
moments and price-growth compensation, ID/stream behavior, and invalid inputs.
The full 26-test simulator suite passes, including the existing offline GBM
bank/train/reload check. Locked installation and `marimo check` also pass.

The independent refinement pilot uses 8,192 paths per case, 64 observations
over one year, and 1/4/16 internal substeps. It compares CIR moments and expected
log returns, plus four joint characteristic-function frequencies of terminal
log return and effective variance. The latter reference integrates the
continuous model's affine-generator Riccati equations by complex RK4; it does
not reuse the full-truncation simulator. Refining that reference from 2,048 to
4,096 steps changes these values by at most about \(3\times10^{-14}\).
For the affine transform background see
[the Federal Reserve's Heston simulation study, Appendix E](https://www.federalreserve.gov/econres/feds/files/2023053pap.pdf).

The [recorded numerical report](results/heston-svj-validation.json) fixes cases,
seeds, tolerances, source hashes and all errors. Finest-grid reference checks
and the last paired refinement check pass for regular Heston, a Feller-violating
near-zero case, and modest Gaussian jumps. Acceptance compares each discrepancy
with its stated absolute budget **plus five Monte Carlo standard errors**.
This is a bounded diagnostic pilot, not a proven remaining-bias bound or
simultaneous confidence statement. In particular:

- The near-zero case retains **13.9% negative terminal auxiliary states** at
  the finest grid. Those are not negative continuous variances; the rate is
  exposed rather than hidden by rejecting paths or imposing the Feller condition.
- Its largest last-step joint-transform component change is about 0.00266
  with standard error 0.00316. This does not establish a 0.003 bias bound.
- Broad priors, tighter pricing tolerances, filtering and tail-risk claims
  require a task-specific numerical study before large banks are generated.

The [recorded timing report](results/heston-svj-benchmark.json) measures a
single macOS ARM64 run with Python 3.12.15, NumPy 2.5.3 and Numba 0.68.0.
It uses 256 paths per batch, 252 observations, eight substeps, three timing
repetitions and four batches for complete simulation:

| Measurement | NumPy | Numba |
|---|---:|---:|
| Warm simulation, one batch | 13.0 ms | 4.3 ms |
| Complete simulation, 1,024 paths | 93.7 ms | 60.7 ms |

Random-input generation alone took 11.9 ms per batch. The Numba first call
with an empty cache took 0.62 s. The measured gain is about **3.0× for warm
simulation and 1.5× including inputs**, compared with NumPy vectorized across
paths. These are illustrative timings, not a throughput guarantee; no inference
or pricing acceleration follows from them. NumPy/Numba outputs were identical
in this run; the tested contract uses an absolute tolerance of `1e-13`, not a
cross-platform bitwise guarantee.

Numba and llvmlite were added to the lock without changing NumPy, BayesFlow,
Keras or JAX versions. This increment's measurements are local; the preceding
GBM Oscar smoke does not establish remote Heston/Numba performance.

## Next boundary

M3.3 will define a deliberately small joint parameter/state inference task,
its prior and initialization, and a same-model inference reference. Then it
will connect these kernels to the existing bank/training path. Latent states,
jump counts and future outcomes must stay outside observed conditioning;
forecasts will continue paired posterior parameter/state draws.

This increment needs no market acquisition. M3.3's synthetic training and
truth-based checks likewise need none. Optional market application supplies
observations to compatible frozen weights; empirical evaluation needs a
separate held-out design. Neither is certified here. Collaborator parity
remains pending receipt and reproduction of Francesco's actual artifacts.
