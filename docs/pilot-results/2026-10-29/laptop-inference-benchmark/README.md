# Local CPU inference timing

All three saved estimators can provide a live posterior on this Mac CPU:
1,000 joint draws for one existing option surface took **1.40–3.62 seconds
in warm calls**. The two rough-model processes peaked at approximately
**1.00 and 1.12 GiB RSS** during these short measurements. This supports using
the pretrained estimators locally, with the hardware and scope limits below.
The measurement includes the physical-parameter adapter. No notebook,
training, numerical simulation or MCMC was run.

| Saved estimator | Load/runtime initialization | First 1,000 draws | Warm call 1 | Warm call 2 |
| --- | ---: | ---: | ---: | ---: |
| Heston, 100k bank, final epoch 500 | 10.773 s | 2.083 s | 1.397 s | 1.538 s |
| Rough Bergomi, known curve, 100k bank | 11.114 s | 2.505 s | 1.773 s | 1.925 s |
| Rough Bergomi, joint curve, 100k bank | 9.717 s | 2.421 s | 3.590 s | 3.620 s |

The rough benchmarks ran entirely from the subsequently downloaded and
hash-verified local model bundles. No network or cluster allocation was needed
for inference. The [transfer manifest](../local-backbone-transfer/README.md)
records the bundle identities.

The first Heston process also spent 0.594 seconds importing the repository and
reading/checking its inputs. Import plus load plus first sampling summed to
13.450 seconds. Load includes importing BayesFlow and its dependencies, device
initialization and restoring the model. This particular first process also
built a matplotlib font cache during BayesFlow import; that overhead is retained
in the measurement. First sampling can include JAX compilation; warm calls use
the same process and observation with different fixed seeds. These labels
describe process state: a warm call is not guaranteed to be faster than the
first call. This is one fixed case per model and two warm repeats, rather than
a latency distribution. OS file caches are not flushed and no persistent XLA
compilation cache was configured. For rough, the measured process start to
completion of its first draw call was 14.586 s (known) and 12.542 s (joint).

The host is an Apple M5 Max, 18 CPU cores and 128 GiB RAM, running macOS 26.4.1.
This is a high-end Mac measurement. **128 GiB is the host configuration, not a
RAM requirement**. The rough runs recorded the whole-process high-water RSS
using Darwin `resource.getrusage`: 1,075,544,064 bytes (known) and 1,201,061,888
bytes (joint), including imports, loading and three draw calls. The
post-load high-water marks were 487,309,312 and 499,302,400 bytes. RSS increased
across these three calls; this is not a steady-session memory bound, minimum
RAM specification or measurement of a full notebook. Heston memory was not
recorded, and it was not rerun solely to add that measurement. These results do
not establish timing on slower laptops, CPU training feasibility or GPU speedups.

The locked local runtime is Python 3.12.16, BayesFlow 2.1.0, Keras 3.15.1,
JAX/JAXlib 0.11.1, NumPy 2.5.3 and SciPy 1.18.1. `KERAS_BACKEND=jax` and
`JAX_PLATFORMS=cpu` force the CPU route; the default float32 inference dtype is
retained. Each checkpoint uses BayesFlow's unchanged native FlowMatching
`tsit5` adaptive solver. There is no Euler override or reduced integration step
count. BLAS/OpenMP and nominal JAX/TensorFlow thread environment limits are one;
XLA Eigen CPU multithreading is disabled and one host device is exposed. These
settings constrain library threading, rather than imposing OS CPU affinity or
a hard core allocation on macOS. In particular, the measured rough sampling
CPU-seconds/wall-seconds ratios range from 2.4 to 4.7: internal parallel work
remains despite the nominal thread environment limits. These are CPU-host
measurements, not one-core or four-core benchmark claims.

The benchmark restores the actual candidate with the repository's
`full_surface.load_checkpoint`, then calls its normal `full_surface.sample` for
one fixed saved `validation:0` observation. Each call returns a finite,
support-checked `[1,1000,parameter_count]` NumPy array, forcing backend execution
to finish before timing stops. Seeds are 271400, 271401 and 271402. Parameter
count is four for Heston, three for known-curve rough and six for joint rough.
The known curve is supplied only for its declared conditioning task.

[heston.json](heston.json), [known100k.json](known100k.json) and
[joint100k.json](joint100k.json) record timing, output-array hashes, runtime,
source hashes and successful before/after input stability checks.
[summary.json](summary.json) contains the compact comparison. Exact model hashes
are:

- Heston: `8d4a243ba099524cf4cda40e4f0616e895879f053829a7f60dadbe65b98301db`.
- Known curve: `39f1a81986a38994dc6eacc520ad95254b34346abc02e758c08779c7c5916fc4`.
- Joint curve: `19c24d64290651f592d3c13d7d0a1616d0d8ae080d487d0298aee8f97779498b`.

Heston is the cached, loadable final-500 export, rather than the earlier
validation-selected epoch-402 checkpoint; its original timing/source record is
preserved. The three `*-case.json` files retain already-saved observations and
their task identities; no new labels were generated.
[hardware.json](hardware.json) records hardware. [benchmark_cpu.py](benchmark_cpu.py)
is the original Heston script; the versioned
[benchmark_cpu_with_memory.py](benchmark_cpu_with_memory.py) adds RSS/CPU-time
recording and a 120-second process alarm for rough. Each rough process was also
bounded by a 125-second external timeout and completed in under 21 seconds.
[local-bundle-verification.json](local-bundle-verification.json) binds those
runs to the verified local download manifest; logs and
[sha256sums.txt](sha256sums.txt) retain the execution evidence.

All three weight bundles are now durably preserved outside the Git worktree at
`/Users/afengler/Projects/proj_sbi4finance/local-artifacts/tutorial-backbones-20261029/oscar-scratch/`.
Heston is under `sbi4finance-heston-teaching-20261028/checkpoint-final500`, known
under `sbi4finance-independent-work-20261028/known100k/known/posterior`, and joint
under `sbi4finance-independent-work-20261029/joint100k/joint/posterior`. Model
binaries are not added to this evidence folder or Git. The benchmark used the
temporary Heston copy before that durable preservation; its model bytes match.

This is an operational latency check. It does not validate posterior accuracy
or promote a model. The independent scientific reports and pending conditional
reference checks keep their existing scope.
