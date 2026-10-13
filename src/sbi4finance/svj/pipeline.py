"""One window, end to end: the daily process of the design document.

1. Build the network input from the last 1,000 daily log returns (return and log squared return).
2. Summary network and 50 Euler steps of the flow: 1,000 joint draws of the 8 parameters and today's spot vol.
3. Misspecification test: does the window look like the training data, in the space of the learned summaries?
4. Audit: importance sampling with the particle filter checks and corrects the draws; a low efficiency
   flags the window, which then goes to particle MCMC if requested.
5. Forecast: simulate 20 paths of 10 days from every draw and read the 99% VaR and 97.5% ES off the pooled
   returns, with the plug-in forecast and the posterior of the conditional VaR beside them.
"""

import time
from dataclasses import asdict, dataclass

import numpy as np

from . import model, risk
from .audit import DEFAULT_THRESHOLD, audit
from .pmmh import run_pmmh


@dataclass
class PipelineConfig:
    n_draws: int = 1_000
    run_audit: bool = True
    n_proposals: int = 3_000
    audit_threshold: float = DEFAULT_THRESHOLD
    pmmh_on_flag: bool = False
    pmmh_iterations: int = 12_500
    pmmh_burn_in: int = 2_000
    horizons: tuple = risk.HORIZONS
    n_paths: int = 20
    variance_split: bool = True
    seed: int = 0


def describe(targets):
    """Posterior medians and 90% intervals in the units of the prior table."""
    q = model.quantities(targets)
    return {
        name: {"median": float(np.median(q[:, j])), "q05": float(np.quantile(q[:, j], 0.05)), "q95": float(np.quantile(q[:, j], 0.95))}
        for j, name in enumerate(model.QUANTITY_LABELS[: q.shape[1]])
    }


def run_window(returns, posterior, misspecification=None, config=PipelineConfig()):
    """Run the daily process on one window of log returns in percent (T,). Returns a JSON-ready report."""
    returns = np.asarray(returns, dtype=float)
    if returns.shape != (model.WINDOW,):
        raise ValueError(f"the network is trained for windows of exactly {model.WINDOW} days, got {returns.shape}")
    seed = config.seed
    timings = {}
    report = {"config": asdict(config)}

    start = time.process_time()
    draws = posterior.sample(returns, num_samples=config.n_draws, seed=seed)[0]
    timings["network_cpu_s"] = time.process_time() - start
    report["network_posterior"] = describe(draws)
    report["network_risk"] = risk.risk_numbers(draws, config.horizons, config.n_paths, seed)

    if misspecification is not None:
        start = time.process_time()
        report["misspecification_pvalue"] = float(misspecification.pvalue(posterior.summaries(returns))[0])
        timings["misspecification_cpu_s"] = time.process_time() - start

    final, route = draws, "network"
    if config.run_audit:
        result = audit(draws, returns, seed, n_proposals=config.n_proposals, n_out=config.n_draws)
        timings["audit_cpu_s"] = result.seconds
        flagged = result.flagged(config.audit_threshold)
        report["audit"] = {
            "efficiency": result.efficiency,
            "ess": result.ess,
            "n_particles": result.n_particles,
            "loglik_sd": result.loglik_sd,
            "flagged": bool(flagged),
            "threshold": config.audit_threshold,
        }
        final, route = result.posterior, "network + importance sampling"
        if flagged and config.pmmh_on_flag:
            chains = run_pmmh(
                returns, draws, seed, n_iter=config.pmmh_iterations, burn_in=config.pmmh_burn_in
            )
            timings["pmmh_cpu_s"] = chains.seconds
            report["pmmh"] = {
                "accept_rate": chains.accept_rate,
                "max_rhat": float(chains.rhat.max()),
                "min_ess_bulk": float(chains.ess_bulk.min()),
                "n_particles": chains.n_particles,
            }
            pooled = chains.pooled()
            final = pooled[np.random.default_rng(seed).choice(len(pooled), config.n_draws, replace=False)]
            route = "particle MCMC"

    report["route"] = route
    report["posterior"] = describe(final)
    start = time.process_time()
    report["risk"] = risk.forecast(final, returns, seed, config.horizons, config.n_paths, config.variance_split)
    timings["forecast_cpu_s"] = time.process_time() - start
    report["timings"] = {k: round(v, 2) for k, v in timings.items()}
    return report, final
