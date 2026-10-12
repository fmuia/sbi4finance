# Review the implemented milestones in marimo

From the repository root, install the pinned environment with
`pixi install --locked`, then choose a review:

| Command | Notebook | What to inspect |
|---|---|---|
| `pixi run review-foundations` | [01_gbm_foundations.py](../notebooks/01_gbm_foundations.py) | M0–M2 / increments 1–3: observation and truth roles, exact GBM posterior/prediction, separate Q prices, tiny saved bank, historical training and actual cluster evidence |
| `pixi run review-kernels` | [04_heston_svj_kernels.py](../notebooks/04_heston_svj_kernels.py) | M3.1–M3.2 / increment 4: diffusion and price jumps, signed-state restart, matched NumPy/Numba calculations and recorded numerical/timing checks |
| `pixi run review-heston` | [05_heston_joint_inference.py](../notebooks/05_heston_joint_inference.py) | M3.3 / increment 5: joint parameter/state inference, four original cases, neural/reference posteriors, future losses, failed frozen test, spline follow-up and bounded-time diagnosis/validation |
| `PYTHONPATH=src pixi run review-hkde-corrected` | [10_hkde_corrected_workflow.py](../notebooks/10_hkde_corrected_workflow.py) | M8 / increment 10: cached options-correction review; new-test theta SBC failed, live inference disabled, participant acceptance pending |

The earlier notebooks are development review tutorials. Notebook 10 packages
one possible M8 participant segment; the final returns-versus-options curriculum
remains open. Each notebook explains the mathematics and conventions,
distinguishes recorded results from bounded live examples where enabled,
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
Its final experiment tests a reversible signed-state asinh coordinate, explains
the Jacobian and shows the failed raw-unit validation comparison. The fixed
stop rule was applied; no new final-test evaluation followed.

## M8 options candidate: what the live lesson includes

The [candidate subplan](m8-options-candidate.md) connects existing components:
nine synthetic noisy option prices → a frozen neural proposal → weights from
the numerical Gaussian likelihood → a joint parameter posterior approximation
→ distributions of two excluded European put prices from the accepted surrogate.
The proposal combines 80% flow, 10% uniform prior and 10% fixed boundary density;
the inference prior stays uniform. The raw flow's failed price gate and the first
correction's reliability stop remain part of the explanation.

The planned live cell is deliberately bounded to 2,048 proposals for the saved
synthetic `validation:257` panel. It is currently **disabled** by the failed test.
When enabled after acceptance, it would compute nine numerical prices for each
likelihood and report loading, proposal, likelihood, target pricing and comparison
costs. Opening the current review trains no network, generates no bank and submits
no cluster job.
The two-output surrogate does not replace the likelihood pricer, and the lesson
makes no purely amortized speed claim.

**Frozen outcome: participant acceptance remains pending.** The 512-case
validation passed. The new 256-case test passed conditional and reliability
checks but failed the theta SIR/SBC rank gate: KS 0.103253 exceeded the declared
0.092513 limit. The frozen study stopped without tuning or a rerun. The original
neural final test remains unused. The notebook displays these stages and skips
live inference; better conditional price accuracy does not override this failure.
The independent-grid posterior PIT on the same panels also fails theta
(0.103485 against 0.092513), close to the SIR value. This comparison does not
isolate correction-specific bias or justify ignoring the frozen gate.

The next export verifies the **cached diagnostic review only**, with all market
fixtures and raw training banks absent. It must show the live calculation skipped
and no cell errors. It cannot establish end-to-end participant acceptance or a
live-inference runtime:

```bash
PYTHONPATH=src pixi run marimo check --strict notebooks/10_hkde_corrected_workflow.py
PYTHONPATH=src pixi run marimo export html notebooks/10_hkde_corrected_workflow.py \
  -o /tmp/hkde-corrected.html
```

The export command alone does not enforce network isolation; the review run
must separately disable external access while allowing local kernel communication.
Record its outcome separately from scientific acceptance; review-export timing
must not be reported as a live correction benchmark. Scientific
correction evidence lives in [the correction report](hkde-corrected-inference.md).
Neither this candidate nor its execution changes Heston or rough Bergomi's
unresolved inference status, selects the final curriculum, or supplies market
evidence.

## Verification and reproduction

The original three reviews passed strict marimo checks:

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
1.6 seconds each. The latest joint-notebook export, including the signed-state transform and its
failed validation comparison, passed the isolation check in 13.44 seconds.
Startup/cache effects make these notebook runtimes, not simulator or amortization
benchmarks. HTML exports are
review previews; use marimo for live Python-backed controls.

The 43 software tests also pass, including both flow families and the signed
state transform's offline train/normalization/save/reload checks. Notebook execution is distinct from
scientific acceptance: the original checkpoint failed conditional final-test
checks, all three spline candidates failed the expanded validation checks, and the new
final test remains unevaluated. The notebook exposes those results deliberately.

Open [notebook07](../notebooks/07_hkde_price_surrogate.py) with
`pixi run review-hkde-surrogate`. The [M5.2 surrogate review](hkde-price-surrogate.md) adds an independently
passing forward-price component while preserving HKDE's inference failure.
Notebook07 passes strict checking and an isolated offline HTML export: 14 cells,
zero errors, 9.17 seconds, without market data, raw regression banks or the
neural checkpoint. Its small coefficient file supports bounded live prediction.
The current HKDE implementation suite passes 117 tests; the combined snapshot
with the latest rough stack passes 155. This verification and
the regression's new test do not unlock the posterior's reserved final test.
