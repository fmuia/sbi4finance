# Parser fixtures only

These tiny, hand-constructed fixtures exercise file IO and column separation.
They are **not real market data, validated simulator output, or training banks**.

- `returns`: log-return-shaped records, including zero and a missing value. The
  arbitrary `example_parameter` label and `future_return` auxiliary column only
  test that labels/outcomes stay outside observations and context.
- `options_iv`: decimal implied-volatility-shaped bid/ask values, hypothetical
  contract context and unknown quote timestamps. No synthetic truth labels.
- `options_raw`: premium-shaped bid/ask values using the same IO convention.
  The quotes are illustrative numbers, not prices validated by a pricing model.

The manifest declares every column's type and role, alongside the row count and
CSV hash. Empty cells mean unknown/null, not zero. There is no download step.

`hkde_task.json` is the exact task specification copied from
`checkpoints/hkde-theta-intensity-provisional/task.json` at simulator PR #15,
commit `729ac26067a1a3e34112d6b153b5783dae4219f5`. It freezes the interface for
small local unit tests; it contains no prices, model weights or truth labels.
The separate checkpoint regression reads actual observations and weights from
the simulator checkout, without committing another copy here.
