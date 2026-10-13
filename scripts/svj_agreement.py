"""Agreement of the amortized posterior with particle MCMC on four held-out series (feasibility result 3).

    pixi run svj-agreement                         # 12,500 iterations x 4 chains per series; long
    pixi run svj-agreement --n-iter 3000 --burn-in 500 --out /tmp/svj-agreement   # quick look

The four series span the prior: rare or frequent jumps (15% and 85% prior quantiles of the jump frequency),
crossed with low or high vol-of-variance (same quantiles of c); the other coordinates are prior draws.
For each, the network posterior and its importance-sampling correction are compared with particle MCMC,
target by target: the criterion is a mean difference below 0.3 posterior sd and 90% widths within 25%.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

from sbi4finance.svj import amortized, audit, model, plots, pmmh

ROOT = Path(__file__).resolve().parents[1]


def compare(draws, reference):
    """Per target: mean difference in reference sd, and the ratio of 90% interval widths."""
    ref_mean, ref_sd = reference.mean(0), reference.std(0)
    width = lambda d: np.quantile(d, 0.95, axis=0) - np.quantile(d, 0.05, axis=0)
    return (draws.mean(0) - ref_mean) / ref_sd, width(draws) / width(reference)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "svj"))
    parser.add_argument("--n-iter", type=int, default=12_500)
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--out", default=str(ROOT / "docs" / "results" / "svj-agreement"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    posterior = amortized.AmortizedPosterior.load(args.checkpoint)
    rng = np.random.default_rng(args.seed)
    z = norm.ppf(0.85)
    prior = model.PRIOR
    cases = []
    for jumps in ("rare", "frequent"):
        for vol_of_var in ("low", "high"):
            coords = model.draw_coordinates(1, rng)[0]
            coords[5] = np.log(prior.jumps_per_year) + (z if jumps == "frequent" else -z) * prior.jumps_per_year_sd
            coords[3] = np.log(prior.cv) + (z if vol_of_var == "high" else -z) * prior.cv_sd
            cases.append((f"{jumps} jumps, {vol_of_var} vol-of-variance", coords))

    results, panels = [], []
    for k, (name, coords) in enumerate(cases):
        sims = model.simulate_from_coordinates(coords[None], rng)
        y, truth = sims["returns"][0], sims["targets"][0]
        start = time.process_time()
        draws = posterior.sample(y, num_samples=1_000, seed=k)[0]
        network_cpu = time.process_time() - start
        checked = audit.audit(draws, y, seed=k)
        chains = pmmh.run_pmmh(y, draws, seed=k, n_iter=args.n_iter, burn_in=args.burn_in)
        reference = chains.pooled()
        diff_net, width_net = compare(draws, reference)
        diff_is, width_is = compare(checked.posterior, reference)
        passes = lambda d, w: bool(np.all(np.abs(d) < 0.3) and np.all(np.abs(w - 1) <= 0.25))
        record = {
            "series": name,
            "truth": dict(zip(model.QUANTITY_LABELS, model.quantities(truth).round(4).tolist())),
            "pmmh": {
                "n_particles": chains.n_particles,
                "loglik_sd": chains.loglik_sd,
                "accept_rate": chains.accept_rate,
                "max_rhat": float(chains.rhat.max()),
                "min_ess_bulk": float(chains.ess_bulk.min()),
                "cpu_s": chains.seconds,
                "cpu_s_per_400_ess": chains.seconds * 400 / chains.ess_bulk.min(),
            },
            "network": {
                "cpu_s": network_cpu,
                "largest_mean_difference_sd": float(np.abs(diff_net).max()),
                "worst_target": model.QUANTITY_LABELS[int(np.abs(diff_net).argmax())],
                "width_ratio_range": [float(width_net.min()), float(width_net.max())],
                "passes": passes(diff_net, width_net),
            },
            "importance_sampling": {
                "efficiency": checked.efficiency,
                "ess": checked.ess,
                "n_particles": checked.n_particles,
                "cpu_s": checked.seconds,
                "cpu_s_per_400_effective_draws": checked.seconds * 400 / max(checked.ess, 1e-9),
                "largest_mean_difference_sd": float(np.abs(diff_is).max()),
                "worst_target": model.QUANTITY_LABELS[int(np.abs(diff_is).argmax())],
                "width_ratio_range": [float(width_is.min()), float(width_is.max())],
                "passes": passes(diff_is, width_is),
            },
            "per_target": {
                label: {
                    "network_diff_sd": float(diff_net[j]),
                    "network_width_ratio": float(width_net[j]),
                    "is_diff_sd": float(diff_is[j]),
                    "is_width_ratio": float(width_is[j]),
                }
                for j, label in enumerate(model.QUANTITY_LABELS)
            },
        }
        results.append(record)
        panels.append((name, reference, draws, checked.posterior, truth))
        print(f"{name}: PMMH {chains.n_particles} particles, R-hat {chains.rhat.max():.3f}, ESS {chains.ess_bulk.min():.0f}, "
              f"{chains.seconds:.0f} CPU-s | network max diff {np.abs(diff_net).max():.2f} sd ({record['network']['worst_target']}) | "
              f"IS eff {checked.efficiency:.1%}, max diff {np.abs(diff_is).max():.2f} sd, {checked.seconds:.0f} CPU-s")
        (out / "agreement.json").write_text(json.dumps({"n_iter": args.n_iter, "burn_in": args.burn_in, "series": results}, indent=1))

    plots.agreement_panels(panels, model.QUANTITY_LABELS).savefig(out / "agreement.png", dpi=150)


if __name__ == "__main__":
    main()
