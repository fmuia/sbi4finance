# Small development replay artifacts

`artifacts.json` connects the checked Heston interior cache and two rough
Bergomi neural replays to `tutorial_final/notebooks/backbone_review.py`.

Both rough cases are the fixed existing `validation:0` row, with 1,000 saved
joint draws and full task/training/validation metadata. They have no conditional
reference chains or validated product distributions. The joint case approaches
prior boundaries; it is a software fixture, not an interior showcase.

Known-curve replay includes seven observed curve levels. Joint-curve replay
omits its generating curve entirely from the inference arrays. Neither cache
has scientific acceptance. Models, recovery checkpoints and raw banks remain
outside Git on Oscar.

See [the review guide](../../../docs/tutorial-backbone-review.md) for cached and
live commands, numerical scope and compatibility with the proposed PR49 layout.
