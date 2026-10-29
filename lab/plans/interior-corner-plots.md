# Complete posterior corner plots

These figures show every inferred parameter and every parameter pair. They use development examples; none is selected from the reserved Heston acceptance panel.

## Rough Bergomi: known curve, three inferred parameters

[Open the complete interior corner plot](../evidence/pilot-results/2026-10-28/interior-corners/rough-interior-complete-corner.png).

The plot includes H, eta and correlation, with the seven known curve levels supplied as conditions. Black shows the saved four-chain PyMC reference, blue the current amortized posterior, and green the generating parameters. The reference is based on the approximate forward-pricing emulator, with its numerical limitations retained.

This is genuinely away from the prior boundaries: every central 99% marginal interval lies inside the inner 90% of its prior range. The smallest distance from those intervals to a prior boundary is 20.8% of the corresponding prior width. The posterior is largely elliptical; we do not claim a banana shape or multiple modes.

[Inputs, checks and plotting source](../evidence/pilot-results/2026-10-28/interior-corners/README.md).

## Heston: four inferred parameters

[Open the complete four-parameter interior Heston corner plot](../evidence/pilot-results/2026-10-28/interior-corners/heston-interior-complete-corner.png).

The observation's truth was fixed in advance at the midpoint of all four prior intervals: initial variance .085, long-run variance .085, vol-of-vol .8 and correlation −.525. Mean reversion remains fixed at 2. The new reference passed chain-convergence, Fourier-precision and interior checks. Every central 99% interval is comfortably inside its prior: the nearest endpoint remains 44.45% of a prior width from a boundary. Neural/reference mean errors are at most .102 reference SD and central-90% width ratios range from 1.016 to 1.040. This posterior is mostly regular, not an artificially selected exotic shape.

Job **8032768** completed successfully in 476 seconds with four CPUs, using **1,904 CPU-seconds**, no GPU, under the separate 3-CPU-hour ceiling. No case or seed was substituted. The older boundary examples remain historical diagnostics, not the main interior showcase.

## Handover and current work

The [collaborator handover](collaborator-numerical-status.md) explains the tutorial objective, completed checks, remaining work and what is on GitHub. The [delivery work order](backbone-delivery-work-order.md) records the active bank generation: known-curve array **8032877** and joint array **8032878** started with 64 and 120 running workers, respectively, and the durable controller refills subsequent ranges within budget.
