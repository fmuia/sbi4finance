# Notation-aligned summary recovery figures

These display-only derivatives use the SPX notebook's `MATH_LABELS`, including
initial variance `v_0`, long-run variance `v_inf` and volatility of variance
`sigma`. They preserve all 512 truths, posterior quantiles, metrics, raw support
violations and shared axes from the frozen summary study. The original PNGs and
`results.json` remain unchanged in the sibling `heston_summary_dimensions`
directory.

`manifest.json` binds each derivative to its original image, raw draw payload,
truth/observation identities and reproduced recovery metrics. No raw draws or
simulated truths are distributed here.

To reproduce, extract the private experiment archive linked in the original
study README, then run from `lab/`:

```sh
python scripts/export_heston_summary_notation.py --archive /path/to/heston-summary-dimensions-20261114
```

The exporter checks the preserved source exporter, validates the raw inputs and
reproduces the recorded metrics before rendering. No training or inference runs.

## Label-only updates without the archive

When only `MATH_LABELS` changes, run from `lab/`:

```sh
python scripts/relabel_heston_summary_notation.py
```

It rebuilds the exporter's layout without data, at matplotlib's stock defaults,
for the old and the new labels. It requires identical axes for both label sets,
requires the old rebuild to reproduce the committed pixels exactly wherever the
two rebuilds differ, and replaces only those pixels. The manifest's `relabelled`
entries record the previous labels and image checksums, the changed pixel boxes
and the matplotlib/FreeType versions. The long-run variance label changed from
`L` to `v_inf` this way; only its panel title changed in each figure.
