# Bergomi simulator speed investigation

**11 November 2026 — local research prototype; no Oscar deployment.** Worktree
`sbi4finance-bergomi-speed-20261111`, branch
`perf/bergomi-simulator-20261111`, based on `41478c5`. The running production
campaign, repair checkout, bank journals and original evidence were not changed.

The measured first improvement is a small compiled path-integration kernel:
**1.94–2.10 times faster for a complete 63-quote surface on this Mac.** A full
Cython/C++ rewrite is not the first step supported by these measurements.

## Matched full-surface experiment

The benchmark uses the committed synthetic challenge panel in
`preparation-portable/work-order.json`, never unpublished production draws.
Each surface has the seven actual expiries and nine contracts per expiry;
base steps are **622 / 1241 / 664 / 1327 / 1791 / 950 / 1796**. Both backends
use 32,768 paths per block (the production ladder's first rung), the same seed,
three-channel normal stream, 2,048-pair chunks, curve, left-point scheme,
antithetics, conditional puts, control fit, covariance and full diagnostics.
Nothing is made faster by reducing paths, steps or the numerical tolerance.

| Predeclared case | Reference median | Prototype median | Speedup |
| --- | ---: | ---: | ---: |
| Typical | 10.684 s | 5.493 s | 1.94× |
| Low H / high eta | 10.872 s | 5.165 s | 2.10× |
| High variance | 10.657 s | 5.435 s | 1.96× |

These are three warmed serial repeats per case on macOS ARM64, with backend
order alternated. **All prices, MC standard errors, covariance matrices and
captured raw/control diagnostics were bit-for-bit equal** on every repeat.
[Full measurements and exact source hashes](full-surface.json) retain timings
for individual blocks, CPU time, compiler warmup and comparisons of every
numeric output. Other local work was not stopped; this is not a dedicated
machine benchmark.

The experiment measures a fixed-path surface. It excludes adaptive escalation,
SQLite, Slurm scheduling and network training. It does not establish a twofold
reduction in current campaign completion time. Oscar/x86 measurements, later
ladder rungs, adaptive threshold decisions and a broader seeded panel remain
necessary before deployment. Local bitwise equality does not promise equality
on another CPU/compiler or certify continuous-model pricing accuracy.

## What the prototype changes

The reference path is already substantially native NumPy/FFT work. It computes
the same linear Gaussian convolution independently for `z` and `-z`, then
materializes full variance, integrated-variance and driver-integral trajectories
for conditional pricing. The reference profile attributes about 28% of the
longest block to FFT calls and 13% to cumulative sums; allocation, RNG and
elementwise work consume much of the remainder. Payoffs and the control fit
are comparatively small in this fixture. See [profile](reference-profile.json).

`rough_block(..., path_backend="numba")` is an explicitly selected experimental
route in the authoritative tutorial runtime. It:

- Computes one hybrid driver and negates it for the antithetic sign.
- Compiles variance evaluation and left-to-right IV/IW accumulation with
  `fastmath=False` and no internal threads; stores only requested maturity totals.
- Reuses the existing payoff, control regression and diagnostic implementation.
- Preserves three RNG channels and their order, including integrated-out W-perp.
- Checks the unused terminal variance and retains the reference's conservative
  overflow certificate; uncertain cases use the full reference checks.

The default remains `path_backend="reference"`; no bank runner selects Numba.
Numba is already pinned in the lab and available through the tutorial's optional
`acceleration` extra. Import and compilation are lazy and explicitly fail if
that dependency is unavailable. The compiler is applied only to the useful
array loop. Cython typed loops could serve the same role, but adding another
build/deployment route has no demonstrated advantage here. Relevant primary
references: [Numba performance tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html),
[Cython memoryviews](https://docs.cython.org/en/latest/src/userguide/memoryviews.html).

Separate exploratory results provide context:

- The compiled integration portion alone was **2.9–3.8×** faster; those numbers
  exclude RNG/FFT/payoffs and must not be called full-simulator speedups.
  [Microbenchmark](fused-reduction-microbenchmark.json).
- Reusing the FFT alone helped roughly **1.1–1.4×** in the block prototype.
  Pure NumPy terminal sums reached about **1.5–1.7×**, with small rounding changes.
- SciPy FFT padding plus Numba showed a further possible gain in two exploratory
  block fixtures, with rounding changes. It was not included in the selected
  backend. [Exploratory variants](exploratory-variants.json);
  [SciPy FFT documentation](https://docs.scipy.org/doc/scipy/reference/fft.html).
- Changing chunk sizes from 128 through 4,096 pairs showed no convincing
  improvement over the current 2,048. [Chunk sweep](chunk-sweep.json).

## Other opportunities and implementation order

[The pipeline audit](pipeline-audit.json) finds that startup and summary storage
are small locally: about half a second of imports per 50-row worker and about
4 ms per synthetic row for packing plus eight durable local SQLite commits.
Oscar filesystem latency is unmeasured. There is no evidence to weaken
checkpoint durability to obtain speed.

The eight workers in an allocation each receive one fixed 50-row shard. Once a
worker finishes it waits for its slowest sibling. The saved initial-tranche
bootstrap modeled **1.2435× elapsed overhead** from this barrier. Refilling an
eight-worker pool from additional shards could recover part of that overhead
without changing row RNG or numerics. Perfect removal would shorten that model
by about 20%; it is a modeled opportunity, not a measured campaign saving.

The adaptive doubling ladder also discards earlier-rung samples. Reusing samples
could nearly halve path work for highly escalated blocks, but changes the RNG,
control fit, stopping rule and realized labels. Sharing one path across all
seven expiries similarly changes grids and cross-expiry MC covariance. Those
are separate numerical experiments after the conservative kernel optimization.
Adding inferred `c` itself costs only observation augmentation, not repricing.

Recommended next sequence: run the matched benchmark on one allocated Oscar
CPU, extend parity checks to adaptive and difficult longer-rung cases, then
declare a **new** producer/backend identity binding the acceleration helper
and compiler/environment. Integrate worker refilling separately. Existing
source-bound campaign journals must continue with their original producer;
do not replace source files in an active checkout or mix their identities.

## Reproduction and validation

From `lab/`, use an environment containing the pinned NumPy, SciPy and Numba:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
PYTHONPATH=.:src:../tutorial/src python scripts/benchmark_rough_pricing_speed.py \
  --cases 0 1 2 --paths 32768 --repeats 3 --out /private/tmp/rough-speed-new.json
```

The output path must be new. The script opens only committed synthetic metadata,
does not validate that historical pilot as a current production admission, and
never opens production journals or submits jobs. It binds the new research
source and the historical task separately.

**Passed:** 65 tutorial tests (including eight acceleration checks and four
independent mathematical checks), 45 lab
tests covering rough kernels, control diagnostics, tutorial parity and runtime
reuse; manifest, claims and the 183-symbol runtime boundary checks. Tests cover
multiple maturities, chunk tails, eta/rho boundaries, missing compiler, forced
fallback and terminal variance overflow. [Validation record](validation.json).

The added independent checks in
`tutorial/tests/test_rough_pricing_oracles.py` compare FFT drivers with explicit
causal sums whose coefficients are computed directly from the analytic cell
integrals, without calling the runtime weight helper. They verify antithetic
sign symmetry at all seven production grid lengths and four H values, compare
accelerated IV/IW with direct scalar model calculations, and check both pricing
backends against an independent scalar Black formula in the deterministic
volatility limit, across strikes and maturities. Expected Black prices use the
discrete left-point variance integral, preserving the chosen grid law.

Additional existing lab checks cover Gaussian covariance and Brownian
cross-covariance, variance normalization, the near-Brownian limit and bounded
small-grid comparisons with a separate Gaussian-covariance driver. These are
independent checks of selected mathematical properties. The optimized backend
also agrees with the full reference outputs. Monte Carlo covariance uses
antithetic pair averages as its independent samples, never the two signs as
independent paths. These checks do not certify prior-wide discretization bias
or pointwise coverage of MC uncertainty from same-batch fitted controls.

Five attempted frozen production-fixture tests reject the changed pricing
source, as the producer guard requires. The historical work order was not
rewritten to make them pass. A fresh research fixture and the full lab/notebook
release checks remain part of a future integration PR; no PR or deployment
is included in this investigation.
