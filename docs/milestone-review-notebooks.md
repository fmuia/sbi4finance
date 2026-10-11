# Review the implemented milestones in marimo

From the repository root, install the pinned environment with
`pixi install --locked`, then choose a review:

| Command | Notebook | What to inspect |
|---|---|---|
| `pixi run review-foundations` | [01_gbm_foundations.py](../notebooks/01_gbm_foundations.py) | M0–M2 / increments 1–3: observation and truth roles, exact GBM posterior/prediction, separate Q prices, tiny saved bank, historical training and actual cluster evidence |
| `pixi run review-kernels` | [04_heston_svj_kernels.py](../notebooks/04_heston_svj_kernels.py) | M3.1–M3.2 / increment 4: diffusion and price jumps, signed-state restart, matched NumPy/Numba calculations and recorded numerical/timing checks |
| `pixi run review-heston` | [05_heston_joint_inference.py](../notebooks/05_heston_joint_inference.py) | M3.3 / increment 5: joint parameter/state inference, four original cases, neural/reference posteriors, future losses, failed frozen test, spline follow-up and bounded-time diagnosis/validation |

These are development review tutorials. M8 will separately select and pace the
participant course. Each notebook explains the mathematics and conventions,
executes a bounded example, distinguishes recorded results from fresh
illustrations, links the evidence, and includes exercises. No default cell
downloads market observations, trains a full network or submits a cluster job.
The joint-inference notebook loads the original provisional checkpoint and
recomputes a small reference for the selected old regression case. Its later
sections compare the spline and explicit-time experiments, with gate diagnostics
and the two remaining conditional failures. Neither candidate is accepted.
The [diagnostic guide](heston-diagnostics.md) and notebook now distinguish
marginal SBC, conditional reference accuracy and engineering diagnostics.
Standalone prior/posterior graphs demonstrate data updating. A frozen-weight
precision follow-up retains every inspected case and explains which errors
persist and which are sensitive to finite sampling.

## Verification and reproduction

All three pass strict marimo checks:

```bash
pixi run marimo check --strict notebooks/01_gbm_foundations.py \
  notebooks/04_heston_svj_kernels.py notebooks/05_heston_joint_inference.py
pixi run marimo export html notebooks/01_gbm_foundations.py -o /tmp/foundations.html
pixi run marimo export html notebooks/04_heston_svj_kernels.py -o /tmp/kernels.html
pixi run marimo export html notebooks/05_heston_joint_inference.py -o /tmp/joint-inference.html
```

Exports were additionally executed in an isolated copy containing only source,
configurations, documentation and the needed checkpoints, with **no market-data
directory**. A temporary Python socket guard blocked external connections while
allowing local kernel communication. All commands exited zero; exported outputs
contained no notebook exceptions or tracebacks, and plotted results were inspected.
The run record pins notebook and HTML hashes in
[`results/milestone-notebooks.json`](results/milestone-notebooks.json).
The verification helper is not a new project framework; the commands above are
the ordinary reproduction path.

On the local CPU, the isolated foundations and kernels exports took about
1.6 seconds each. The latest joint-notebook export, including the prior/posterior and refined-reference
diagnostics, passed the isolation check in 13.98 seconds.
Startup/cache effects make these notebook runtimes, not simulator or amortization
benchmarks. HTML exports are
review previews; use marimo for live Python-backed controls.

The 40 software tests also pass, including both flow families' offline
train/normalization/save/reload checks. Notebook execution is distinct from
scientific acceptance: the original checkpoint failed conditional final-test
checks, both spline candidates failed the expanded validation checks, and the new
final test remains unevaluated. The notebook exposes those results deliberately.
