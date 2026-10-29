# Next independent product-pricer check

This is a **prepared work specification, not a submitted run**. It does not depend on finishing joint 100k training. Its purpose is to establish whether the product pricer is precise enough before evaluating posterior product quantiles.

Use the two existing development vectors `typical_flat` and `high_variance` from `calibration.numerical_pilot.DEFAULT_CASES`. Keep their six parameters fixed. These are deterministic stress checks, not accepted posterior examples.

Check only the declared ATM forward-start contract, `E[max(S_126d - S_63d, 0)]`, in initial-spot units. Call the existing `products.forward_start_call_spot` implementation. The legacy return-ratio payoff, two-year put and flat curve extension are outside this check. The current `products_study.py` calls the supplied-curve API and therefore must not be used unchanged for the joint task.

For the joint task, evaluate the analytic exponential curve from the six parameters on every simulated node. Couple 4032 and 8064 steps per year over the half-year horizon, preserving the existing hybrid Gaussian-channel marginal laws. Use eight independent final batches of 65,536 antithetic paths at each fixed vector; keep paired price differences and both reported and external replicate uncertainty. Retain failures without changing seeds or vectors. A timing batch counts among the sixteen final batches.

Use the one-year variance-swap mean as an analytic control: `long + (short-long) * (1-exp(-decay))/decay`. In the known-curve lesson, integrate the supplied step curve instead; its variance-swap mean is fixed by that observed curve. The product path grid's left-sum mean and this continuous analytic integral must be labelled separately.

Keep the protocol's proposed absolute materiality floor, **0.0001 initial-spot units**, and refinement limit **T/4 = 0.000025**. Compare the paired grid movement together with its finite-batch uncertainty. Unresolved precision or grid movement remains inconclusive. This does not yet evaluate a posterior price distribution, its q10/q90, or the uncertainty of those quantiles.

A possible finite envelope is sixteen one-CPU, 16-GiB workers with 15-minute limits, plus five minutes each for metadata and reduction; add 240 seconds cleanup per allocation. The full new exposure is **19,320 CPU-seconds**. The previous completed follow-up used **4,472 allocated CPU-seconds** under a seven-hour total allowance: pooled arithmetic would be **23,792 seconds = 6.6089 hours**, below seven hours. However, that historical allowance had separate purpose-specific subcaps. **A reviewed reassignment of its unused allowance and an exact frozen source/stream manifest are required before submitting this new work.** It is not silently charged to the already-running price panel or bank generation.

Reuse the existing source and payoff tests, then benchmark one retained batch before releasing the rest. If that timing or precision is unsuitable, preserve the result and revise the proposed work rather than automatically raising paths, wall time or budget.
