"""Train the SVJ posterior online or from a saved bank into a new output directory.

    pixi run svj-train --preset smoke --out /tmp/svj-smoke     # seconds: wiring check
    pixi run svj-train --preset laptop --out /tmp/svj-laptop   # ~10 minutes on a laptop CPU, learns little about jumps
    pixi run svj-train --bank /tmp/svj-bank --epochs 1 --batch-size 8 --out /tmp/svj-offline

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
    parser.add_argument("--n-validation", type=int, default=512, help="online validation examples; offline uses bank validation")
    parser.add_argument("--bank", help="optional saved SVJ bank; its test split remains reserved")
    parser.add_argument("--out", required=True, help="new run directory; existing directories are never overwritten")
    args = parser.parse_args()

    budget = dict(amortized.PRESETS[args.preset])
    for key in ("epochs", "batches_per_epoch", "batch_size"):
        if getattr(args, key) is not None:
            budget[key] = getattr(args, key)
    _, record = amortized.train(args.out, seed=args.seed, n_validation=args.n_validation, bank=args.bank, **budget)
    print(
        f"done: {record['lifecycle']['mode']}, {record['lifecycle']['training_exposures']:,} training exposures "
        f"in {record['seconds'] / 60:.1f} min; "
        f"loss {record['loss'][-1]:.3f}, val_loss {record['val_loss'][-1]:.3f}; saved under {args.out}/"
    )


if __name__ == "__main__":
    main()
