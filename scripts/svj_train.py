"""Train the SVJ amortized posterior online and save a checkpoint.

    pixi run svj-train --preset smoke --out /tmp/svj-smoke     # seconds: wiring check
    pixi run svj-train --preset laptop --out /tmp/svj-laptop   # ~10 minutes on a laptop CPU, learns little about jumps
    pixi run svj-train                                          # reference run, 2.56 million series -> checkpoints/svj

The networks and the simulator are defined in src/sbi4finance/svj; a checkpoint only loads into the same
architecture, and is stale as soon as the model, the prior, the window length or the input definition changes.
"""

import argparse

from sbi4finance.svj import amortized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--preset", choices=sorted(amortized.PRESETS), default="reference")
    parser.add_argument("--epochs", type=int, help="override the preset")
    parser.add_argument("--batches-per-epoch", type=int, help="override the preset")
    parser.add_argument("--batch-size", type=int, help="override the preset")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="checkpoints/svj")
    args = parser.parse_args()

    budget = dict(amortized.PRESETS[args.preset])
    for key in ("epochs", "batches_per_epoch", "batch_size"):
        if getattr(args, key) is not None:
            budget[key] = getattr(args, key)
    _, record = amortized.train(args.out, seed=args.seed, **budget)
    print(
        f"done: {record['simulated_series']:,} series in {record['seconds'] / 60:.1f} min; "
        f"loss {record['loss'][-1]:.3f}, val_loss {record['val_loss'][-1]:.3f}; saved under {args.out}/"
    )


if __name__ == "__main__":
    main()
