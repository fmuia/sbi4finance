# Heston numerical admission pilot

The isolated [check](../../../../checks/path_projected_euler.py) compares signed
full-truncation Euler with projected Euler using identical Brownian increments
at 8, 16 and 32 internal steps per daily observation. The fixed design contains
48 marginally stratified prior settings and 16 corner settings, four independent
innovation histories per setting, and 1,260 daily observations. It is not an iid
prior sample or a posterior evaluation.

Oscar job `8206381`, source `5eeefc4`, completed successfully on one CPU with no
GPU. The allocation lasted 13 seconds; measured scientific runtime was 3.99
seconds and peak RSS was 203.3 MiB. Every recorded metric was finite. At every
setting/grid, the signed control was bitwise equal to the production simulator.

| Internal steps/day | Signed control: zero integrated-variance days | Projected candidate: zero integrated-variance days |
| --- | ---: | ---: |
| 8 | 11,412 | 0 |
| 16 | 6,691 | 0 |
| 32 | 3,342 | 0 |

These counts describe this deliberate diagnostic panel, not a population atom
rate. Floating equality of a stored return to drift times one day is recorded
separately and is not an atom-mass estimator. Projected daily independent-noise
variance remained positive in the panel, consistent with the analytical support
argument in the check.

Projection removes the specific signed-state mechanism but changes numerical
bias. Its latent variance still has a zero endpoint mass. Four histories per
setting cannot certify weak accuracy, a price martingale or posterior accuracy;
32 internal steps are not continuous-model truth. The next numerical step is a
separately bounded fixed-parameter precision comparison before selecting a new
task or generating a replacement bank. Characteristic-function comparisons were
not performed; any later comparison must remove the deterministic physical
drift from accumulated log returns to match the repository's carry-free API.

The original tasks, banks, models and reference outputs remain unchanged.
[Lord, Koekkoek and van Dijk](https://papers.tinbergen.nl/06046.pdf) describe the
projected/absorption scheme and its pricing-bias limitations; the source-backed
[numerical follow-up](../../../../plans/PATH-INFERENCE-PLAN.md#e2-first-candidate-pilot-projected-euler)
records the selection constraints.

[`summary.json`](summary.json) binds the source archive, full report, result
archive and provider verification. Source, complete paired metrics, logs,
accounting and verification receipts are preserved under
`oscar-history-20261101/numerical-pilot-20261101/` in the existing
[workshop Drive folder](https://drive.google.com/drive/folders/1LJsR0Qb0RkD4h8XiF3t6OQQlyYDKcI2k).
Local checksums and provider-reported upload completion are verified; an
independent cloud redownload was not performed.
