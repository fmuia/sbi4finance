# Complete posterior corner plots

These figures show every inferred parameter and every parameter pair. They use development examples; none is selected from the reserved Heston acceptance panel.

## Rough Bergomi: known curve, three inferred parameters

[Open the complete interior corner plot](pilot-results/2026-10-28/interior-corners/rough-interior-complete-corner.png).

The plot includes H, eta and correlation, with the seven known curve levels supplied as conditions. Black shows the saved four-chain PyMC reference, blue the current amortized posterior, and green the generating parameters. The reference is based on the approximate forward-pricing emulator, with its numerical limitations retained.

This is genuinely away from the prior boundaries: every central 99% marginal interval lies inside the inner 90% of its prior range. The smallest distance from those intervals to a prior boundary is 20.8% of the corresponding prior width. The posterior is largely elliptical; we do not claim a banana shape or multiple modes.

[Inputs, checks and plotting source](pilot-results/2026-10-28/interior-corners/README.md).

## Heston: four inferred parameters

A new interior development observation is being evaluated with a truth fixed in advance at the midpoint of all four prior intervals: initial variance .085, long-run variance .085, vol-of-vol .8 and correlation −.525. Mean reversion remains fixed at 2. The complete four-parameter corner plot will be linked here after its PyMC reference and numerical checks finish.

Job **8032768** was submitted with four CPUs, 16 GiB and a 30-minute limit under a separate 3-CPU-hour ceiling. A failed or boundary-reaching result will be retained rather than replaced by a better-looking case. The older boundary examples remain historical diagnostics, not the main interior showcase.

## Handover and current work

The [collaborator handover](collaborator-numerical-status.md) explains the tutorial objective, completed checks, remaining work and what is on GitHub. The [current work order](tutorial-backbone-work-order.md) gives the numerical agenda. The known-curve 50k bank extension and completion of the original joint-curve bank are being prepared under bounded generation plans; this figure page does not imply that unsubmitted bank jobs are already running.
