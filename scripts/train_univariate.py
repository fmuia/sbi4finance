"""Train the univariate MMAR approximator from scratch and save a checkpoint.

    pixi run python scripts/train_univariate.py --n-train 2000 --epochs 2    # laptop smoke test
    sbatch cluster/train.sbatch                                              # Oscar GPU, full run

Adapter and networks mirror notebooks/01_mmar_univariate.py. Keep them in sync:
a checkpoint only loads into an identical architecture.
"""

import argparse
import os
import time

os.environ.setdefault("KERAS_BACKEND", "jax")  # must precede the bayesflow import

import bayesflow as bf

from sbi4finance.mmar import univariate as model


def build_workflow(checkpoint_dir: str | None = None) -> bf.BasicWorkflow:
    adapter = (
        bf.Adapter()
        .convert_dtype("float64", "float32")
        .constrain("sigma_bar", lower=model.PRIOR.sigma_low, upper=model.PRIOR.sigma_high)
        .constrain("q", lower=model.PRIOR.q_low, upper=model.PRIOR.q_high)
        .constrain("nu", lower=model.PRIOR.nu_low, upper=model.PRIOR.nu_high)
        .concatenate(model.PARAMETER_NAMES, into="inference_variables")
        .as_time_series("returns")
        .rename("returns", "summary_variables")
    )
    return bf.BasicWorkflow(
        adapter=adapter,
        summary_network=bf.networks.FusionTransformer(dropout=0.1),
        inference_network=bf.networks.CouplingFlow(transform="spline", depth=4),
        standardize="all",
        checkpoint_filepath=checkpoint_dir,
        save_best_only=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="checkpoints/univariate-retrained",
                        help="checkpoint dir; rename to checkpoints/univariate once you trust it")
    args = parser.parse_args()

    train = model.simulate(args.n_train, seed=args.seed)
    val = model.simulate(args.n_val, seed=args.seed + 1)
    workflow = build_workflow(args.out)
    start = time.time()
    history = workflow.fit_offline(data=train, validation_data=val, epochs=args.epochs, batch_size=args.batch_size)
    losses = history.history
    print(f"done in {time.time() - start:.0f} s: loss {losses['loss'][-1]:.3f}, "
          f"val_loss {losses['val_loss'][-1]:.3f}; best model saved under {args.out}/")


if __name__ == "__main__":
    main()
