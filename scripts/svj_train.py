"""Train the SVJ posterior online or from a saved bank into a new output directory.

    pixi run svj-train --preset smoke --out /tmp/svj-smoke     # seconds: wiring check
    pixi run svj-train --preset laptop --out /tmp/svj-laptop   # ~10 minutes on a laptop CPU, learns little about jumps
    pixi run svj-train --bank /tmp/svj-bank --epochs 1 --batch-size 8 --out /tmp/svj-offline

The networks and the simulator are defined in src/sbi4finance/svj; a checkpoint only loads into the same
architecture, and is stale as soon as the model, the prior, the window length or the input definition changes.
Offline mode requires explicit --epochs and --batch-size. An offline epoch visits
all bank training examples; the bank validation split supplies selection data.
Online presets and simulation/validation budgets cannot be combined with --bank.
"""

import argparse

from sbi4finance.svj import amortized


def training_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--preset", choices=sorted(amortized.PRESETS), help="online preset (default: reference); unavailable with --bank")
    parser.add_argument("--epochs", type=int, help="override the preset")
    parser.add_argument("--batches-per-epoch", type=int, help="override the preset")
    parser.add_argument("--batch-size", type=int, help="override the preset")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-validation", type=int, help="online validation examples (default: 512); unavailable with --bank")
    parser.add_argument("--bank", help="optional saved SVJ bank; its test split remains reserved")
    parser.add_argument("--out", required=True, help="new run directory; existing directories are never overwritten")
    args = parser.parse_args(argv)

    if args.bank is not None:
        if any(getattr(args, name) is not None for name in ("preset", "batches_per_epoch", "n_validation")):
            parser.error("--bank cannot be combined with --preset, --batches-per-epoch or --n-validation")
        if args.epochs is None or args.batch_size is None:
            parser.error("--bank requires explicit --epochs and --batch-size; each epoch uses the complete training split")
        budget = {"epochs": args.epochs, "batch_size": args.batch_size, "batches_per_epoch": None}
    else:
        budget = dict(amortized.PRESETS[args.preset or "reference"])
        for key in ("epochs", "batches_per_epoch", "batch_size"):
            if getattr(args, key) is not None:
                budget[key] = getattr(args, key)
    for key, value in budget.items():
        if value is not None and value <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.seed < 0 or (args.n_validation is not None and args.n_validation <= 0):
        parser.error("--seed must be nonnegative and --n-validation positive")
    return args, budget


def main() -> None:
    args, budget = training_arguments()
    _, record = amortized.train(args.out, seed=args.seed, n_validation=args.n_validation, bank=args.bank, **budget)
    print(
        f"done: {record['lifecycle']['mode']}, {record['lifecycle']['training_exposures']:,} training exposures "
        f"in {record['seconds'] / 60:.1f} min; "
        f"loss {record['loss'][-1]:.3f}, val_loss {record['val_loss'][-1]:.3f}; saved under {args.out}/"
    )


if __name__ == "__main__":
    main()
