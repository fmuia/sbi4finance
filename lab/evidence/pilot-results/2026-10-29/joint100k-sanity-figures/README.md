# Figures for the joint posterior sanity audit

These figures use retained validation draws and compact committed audit summaries. They run no model, inference, simulation or MCMC. The [main explanation](../../../joint-rough-posterior-sanity.md) distinguishes aggregate evidence of contraction from unfinished conditional PyMC validation.

- [Contraction and case-pairing control](contraction-and-case-pairing.png): physical interval widths, recovery error and coverage, with all off-diagonal case pairings as a negative control.
- [Complete six-dimensional corner, case1983](complete-corner-validation-1983.png): an interior long-variance/decay ridge.
- [Complete six-dimensional corner, case1941](complete-corner-validation-1941.png): an interior eta/rho dependence example.
- [Initial variance curves and weak decay](curve-function-and-weak-decay.png): posterior and prior pointwise bands; the nearly-flat example is separate from the interior showcases.

Each PNG has a corresponding SVG for zooming or export. All four PNGs were visually checked for labels, layout and clipping. Gaussian ellipses illustrate a moment-matched comparison; they are not posterior credible contours. Curve prior quantiles use deterministic256-node quadrature; the128-node comparison differs by at most1.03e-8 variance units.

From the repository root:

```bash
pixi run --locked -e core python scripts/plot_joint_posterior_sanity.py
```

No cluster access or full2,000-case draw panel is needed for these plots. Exact input/output hashes, plot-source hash and plotting versions are recorded in [provenance.json](provenance.json). The full numerical audits retain their own input identities and reproduction scripts.
