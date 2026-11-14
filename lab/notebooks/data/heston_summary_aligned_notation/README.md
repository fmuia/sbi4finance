# Notation-aligned summary recovery figures

These display-only derivatives use the SPX notebook's `MATH_LABELS`, including
long variance `L` and volatility of variance `sigma`. They preserve all 512 truths,
posterior quantiles, metrics, raw support violations and shared axes from the
frozen summary study. The original PNGs and `results.json` remain unchanged in
the sibling `heston_summary_dimensions` directory.

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
