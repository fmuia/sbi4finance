# One boundary-covering proposal follow-up

This protocol follows the saved [v1 validation](results/hkde-correction-v1/validation.json).
V1 passed every conditional accuracy gate at both budgets: maximum target-price
errors were 0.01261 (2,048 proposals) and 0.005773 (8,192), below 0.03.
Its precision run stopped on proposal reliability: ESS/N 0.3336 < 0.5 and
maximum weight 0.01494 > 0.01, both in `stress:2`. Calibration and the new test
were not run. The absolute ESS increased with the larger budget; the failed
reliability rules do not mean its posterior estimates became less accurate.

The largest-weight row lies near the lower jump-intensity boundary and has
likelihood almost equal to the case maximum. The logit-based neural density
underrepresents this region. This is a cause for a proposal change, not a reason
to change the posterior prior, discard that row, or relax its gate.

## Fixed change, before new outcomes

Let the normalized coordinates be
`u_d=(a_d-l_d)/(r_d-l_d)` in `(0,1)`. Add the product of two arcsine densities:

\[
h(a)=\frac{p(a)}{\pi^2\sqrt{u_1(1-u_1)u_2(1-u_2)}},\qquad
g_2(a\mid y)=0.8q(a\mid y)+0.1p(a)+0.1h(a).
\]

Sample `h` by independent Beta(1/2,1/2) coordinates transformed back to raw
units. Each row independently chooses one of the three components. Evaluate
**the entire mixture** in the denominator, regardless of its source. Retain
uniform-prior density in the numerator. No weights or coordinates are clipped.
The proposal covers all four parameter boundaries symmetrically; it has no
fitted model, chosen boundary strip, or observation-specific shape parameter.

Since `g_2 >= (8/9) g_1` everywhere, density in any previously covered region
falls by at most 11.1%. This is an analytical proposal comparison, not proof of
improved ESS or a passing future run. The [independent boundary review](hkde-correction-boundary-review.md)
records the diagnosed point and density calculation.

All original and v1 numerical/accuracy/reliability thresholds remain fixed.
Keep both conditional budgets, 512 calibration cases, roots 95021/95026 and
all split rules. Reusing development-validation cases is explicit. The new
256-panel test remains unseen and is generated only after this v2 validation
passes. The raw baseline's old test remains untouched. Reports identify the
new protocol and execution source; v1 artifacts remain immutable.

This is one bounded v2 proposal experiment, not an adaptive proposal search.
If it fails, retain its diagnostics and leave scientific participant acceptance
pending. The notebook can still review development evidence without silently
claiming that its live candidate has passed.
