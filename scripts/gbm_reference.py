"""Synthetic history -> exact posterior -> future losses, plus separate Q prices.

Run ``pixi run gbm-demo``. Optionally save arrays and a report with
``pixi run gbm-demo --out /tmp/gbm-reference`` (a new directory).
No market files, trained checkpoint or network access are needed.
"""

import argparse
import json
from pathlib import Path
from statistics import NormalDist
from time import perf_counter

import numpy as np

from sbi4finance import gbm


def run(config):
    """Run the declared synthetic experiment; return a report and P-only arrays."""
    if config["data_origin"] != "synthetic" or config["evidence_scope"] != "S":
        raise ValueError("this demonstration accepts only a synthetic S configuration")
    physical, pricing = config["returns"], config["pricing"]
    if physical["measure"] != "P" or pricing["measure"] != "Q":
        raise ValueError("returns and pricing experiments must be declared P and Q")
    # Stream purposes are stable and independent; changing the forecast budget
    # cannot change the conditioning history. Per-example bank IDs come in M2.
    streams = [np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [config["seed"], purpose]))) for purpose in range(4)]
    dt = np.full(physical["n_observations"], physical["dt_years"])
    history = gbm.simulate_log_returns(
        physical["true_drift"], physical["known_volatility"], dt, rng=streams[0]
    )
    prices = gbm.prices_from_log_returns(history, physical["initial_price"])
    mean, std = gbm.drift_posterior(
        history, physical["known_volatility"], dt,
        prior_mean=physical["prior_mean"], prior_std=physical["prior_std"],
    )
    future_dt = np.full(physical["forecast_steps"], physical["dt_years"])
    forecast_mean, forecast_std = gbm.predictive_log_return(
        mean, std, physical["known_volatility"], future_dt.sum()
    )
    started = perf_counter()
    # Each path retains one sampled drift across all its future increments.
    draws = streams[1].normal(mean[0], std[0], size=physical["forecast_draws"])
    futures = gbm.simulate_log_returns(
        draws, physical["known_volatility"], future_dt, rng=streams[2]
    )
    total_returns = futures.sum(axis=1)[:, 0]
    forecast_seconds = perf_counter() - started
    losses = -np.expm1(total_returns)  # loss as a fraction of current asset value
    confidence = physical["loss_confidence"]
    tail_z = NormalDist().inv_cdf(1 - confidence)
    reference_var = -np.expm1(forecast_mean[0] + forecast_std[0] * tail_z)

    # This Q experiment has its own explicitly given inputs. The P drift
    # posterior is not used to calibrate the pricing model.
    maturity = pricing["maturity_years"]
    forward = pricing["spot"] * np.exp((pricing["rate"] - pricing["dividend_yield"]) * maturity)
    discount = np.exp(-pricing["rate"] * maturity)
    strikes = np.asarray(pricing["strikes"])
    variance = pricing["volatility"] ** 2 * maturity
    reference_calls = gbm.european_price(forward, strikes, discount, variance)
    q_returns = gbm.simulate_log_returns(
        np.full(pricing["mc_draws"], pricing["rate"] - pricing["dividend_yield"]),
        pricing["volatility"], [maturity], rng=streams[3],
    )
    terminal = pricing["spot"] * np.exp(q_returns[:, 0, 0])
    payoffs = discount * np.maximum(terminal[:, None] - strikes, 0)
    if variance == 0:
        # All paths have the same payoff. Repeated summation would manufacture
        # tiny roundoff differences and misleading standard-error scores.
        mc_prices = payoffs[0]
        mc_se = np.zeros_like(mc_prices)
    else:
        mc_prices = payoffs.mean(axis=0)
        mc_se = payoffs.std(axis=0, ddof=1) / np.sqrt(len(payoffs))
    error_scores = [float((estimate - reference) / se) if se > 0 else None
                    for estimate, reference, se in zip(mc_prices, reference_calls, mc_se)]
    report = {
        "task_id": config["task_id"], "data_origin": "synthetic", "evidence_scope": "S",
        "configuration": config,
        "numpy_version": np.__version__, "rng": "PCG64 / SeedSequence([seed, purpose])",
        "stream_purposes": {"history": 0, "posterior_drift": 1, "future_shocks": 2, "q_pricing": 3},
        "returns_result": {
            "measure": "P", "conditioning_cutoff_years": float(dt.sum()),
            "truth_drift": physical["true_drift"], "observations_shape": list(history.shape),
            "posterior_mean": float(mean[0]), "posterior_std": float(std[0]),
            "terminal_observed_price": float(prices[0, -1, 0]),
            "forecast_horizon_years": float(future_dt.sum()),
            "predictive_log_return_mean": float(forecast_mean[0]),
            "predictive_log_return_std": float(forecast_std[0]),
            "mc_log_return_mean": float(total_returns.mean()),
            "mc_mean_standard_error": float(forecast_std[0] / np.sqrt(len(draws))),
            "mc_log_return_std": float(total_returns.std(ddof=1)),
            "loss_definition": "1 - future_price / current_price (decimal fraction)",
            "predictive_var_reference": float(reference_var),
            "predictive_var_mc": float(np.quantile(losses, confidence)),
            "forecast_generation_seconds": forecast_seconds,
        },
        "pricing_result": {
            "measure": "Q", "strikes": strikes.tolist(), "forward": float(forward),
            "discount": float(discount), "integrated_variance": variance,
            "reference_calls": reference_calls.tolist(),
            "mc_calls": mc_prices.tolist(), "mc_standard_errors": mc_se.tolist(),
            "error_in_standard_errors": error_scores,
            "standard_error_note": "A null score means zero sample variance; it does not certify accuracy for an unsampled stochastic tail.",
        },
        "array_roles": {
            "observations": "P conditioning, decimal log returns [batch,time,1]",
            "dt": "known conditioning context in year fractions",
            "parameters": "synthetic P drift truth [batch,1]; excluded from conditioning",
            "observed_prices": "derived from history and known initial price",
            "posterior_drift_draws": "inference output; one per future path",
            "future_log_returns": "predictive output; excluded from conditioning",
            "future_dt": "declared forecast context in year fractions",
        },
    }
    arrays = {
        "observations": history, "dt": dt,
        "parameters": np.array([[physical["true_drift"]]]), "observed_prices": prices,
        "posterior_drift_draws": draws[:, None], "future_log_returns": futures,
        "future_dt": future_dt,
    }
    return report, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1]
                        / "configs/tasks/gbm_drift.json")
    parser.add_argument("--out", type=Path, help="optional NEW directory for report and arrays")
    args = parser.parse_args()
    if args.out is not None and args.out.exists():
        parser.error("--out must be a new directory; existing results are not overwritten")
    report, arrays = run(json.loads(args.config.read_text()))
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.mkdir(parents=True)
        np.savez(args.out / "returns_example.npz", **arrays)
        (args.out / "report.json").write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
