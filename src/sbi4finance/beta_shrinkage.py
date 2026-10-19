"""Betas across many stocks: one regression per stock, or Bayes' rule with a prior from the cross-section.

Each stock follows the market model r_i = alpha_i + beta_i r_m + e_i, with Gaussian errors. Least
squares gives each beta b_i and its standard error se_i. The Bayesian answer puts a normal prior on
the betas, beta_i ~ N(m, tau^2), standing for the population of stocks. With the residual variance
at its estimate (exact enough with 60 or more observations) the posterior is normal, and its mean
pulls each estimate toward m in proportion to the estimate's own noise:

    E[beta_i | data] = m + w_i (b_i - m),    w_i = tau^2 / (tau^2 + se_i^2).

Empirical Bayes (Vasicek 1973) estimates m and tau^2 from the cross-section, net of the sampling
noise (Lally 1998): tau^2 = var(b) - mean(se^2). Bloomberg's adjusted beta is the fixed rule
0.67 b + 0.33, from Blume's (1971) finding that betas regress toward 1. James and Stein (1961)
proved that, for three or more stocks with Gaussian errors of known variance, shrinking toward the
common mean has lower total squared error than the separate estimates, whatever the true betas.

``study`` measures all of this on simulated universes where the true betas are known.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats

BLOOMBERG_WEIGHT = 0.67
LEVEL = 0.90


@dataclass(frozen=True)
class Setting:
    """A data frequency and window: observations, market sd and median idiosyncratic sd per period."""

    label: str
    n_obs: int
    market_sd: float
    idio_sd: float


# Market volatility 16% a year and a median idiosyncratic volatility of 25% a year, per period.
SETTINGS = {
    "daily": Setting("daily returns, 1 year", 250, 0.16 / np.sqrt(252), 0.25 / np.sqrt(252)),
    "weekly": Setting("weekly returns, 2 years", 104, 0.16 / np.sqrt(52), 0.25 / np.sqrt(52)),
    "monthly": Setting("monthly returns, 5 years", 60, 0.16 / np.sqrt(12), 0.25 / np.sqrt(12)),
}
POPULATION = {"beta_mean": 1.0, "beta_sd": 0.3, "idio_dispersion": 0.35}


def simulate(rng, setting, n_stocks=500, beta_mean=1.0, beta_sd=0.3, idio_dispersion=0.35):
    """A universe of stocks: true betas from the population, then market-model returns for one window."""
    beta = rng.normal(beta_mean, beta_sd, n_stocks)
    idio_sd = setting.idio_sd * np.exp(rng.normal(0.0, idio_dispersion, n_stocks))
    market = rng.normal(0.0, setting.market_sd, setting.n_obs)
    returns = beta[:, None] * market[None, :] + idio_sd[:, None] * rng.standard_normal((n_stocks, setting.n_obs))
    return {"beta": beta, "idio_sd": idio_sd, "market": market, "returns": returns}


def least_squares(returns, market):
    """One market-model regression per stock (rows of `returns`): the beta and its standard error."""
    returns = np.atleast_2d(returns)
    x = market - market.mean()
    centred = returns - returns.mean(axis=1, keepdims=True)
    b = centred @ x / (x @ x)
    resid = centred - b[:, None] * x[None, :]
    s2 = np.sum(resid**2, axis=1) / (len(market) - 2)
    return b, np.sqrt(s2 / (x @ x))


def posterior(b, se, prior_mean, prior_sd):
    """The normal posterior of each beta under a normal prior: mean, sd and the weight on the estimate."""
    w = prior_sd**2 / (prior_sd**2 + se**2)
    return prior_mean + w * (b - prior_mean), np.sqrt(w) * se, w


def empirical_bayes(b, se):
    """Vasicek's adjustment: the prior's mean and sd estimated from the cross-section, net of sampling noise."""
    prior_mean = float(np.mean(b))
    prior_sd = float(np.sqrt(max(np.var(b) - np.mean(se**2), 1e-6)))
    mean, sd, w = posterior(b, se, prior_mean, prior_sd)
    return {"mean": mean, "sd": sd, "weight": w, "prior_mean": prior_mean, "prior_sd": prior_sd}


def bloomberg(b):
    """Bloomberg's adjusted beta: two thirds of the estimate plus one third of 1."""
    return BLOOMBERG_WEIGHT * b + (1 - BLOOMBERG_WEIGHT)


def estimates(universe, population=POPULATION):
    """Every estimator on one universe, with the intervals the two interval methods give.

    The oracle is Bayes' rule with the true population as the prior, which empirical Bayes estimates.
    """
    b, se = least_squares(universe["returns"], universe["market"])
    eb = empirical_bayes(b, se)
    oracle, _, _ = posterior(b, se, population["beta_mean"], population["beta_sd"])
    t = stats.t.ppf(0.5 + LEVEL / 2, len(universe["market"]) - 2)
    z = stats.norm.ppf(0.5 + LEVEL / 2)
    return {
        "point": {"least_squares": b, "bloomberg": bloomberg(b), "bayes": eb["mean"], "oracle": oracle},
        "interval": {"least_squares": (b - t * se, b + t * se), "bayes": (eb["mean"] - z * eb["sd"], eb["mean"] + z * eb["sd"])},
        "se": se, "eb": eb,
    }


POINT_METHODS = ("least_squares", "bloomberg", "bayes", "oracle")
INTERVAL_METHODS = ("least_squares", "bayes")


def study(setting, n_universes=200, n_stocks=500, top=50, seed=0, population=POPULATION, keep_draws=False):
    """Average accuracy over simulated universes, and the hedge of the `top` stocks by estimated beta.

    The hedge holds one unit of each of the `top` stocks with the highest least-squares beta and
    sells each estimator's beta in the market; its residual market exposure is the sum of the true
    betas minus the estimated ones, in units of one stock's position. `keep_draws` adds every universe's exposure to the result.
    """
    rng = np.random.default_rng(seed)
    sq = {m: [] for m in POINT_METHODS}
    top_bias = {m: [] for m in POINT_METHODS}
    exposure = {m: [] for m in POINT_METHODS}
    cover = {m: [] for m in INTERVAL_METHODS}
    width = {m: [] for m in INTERVAL_METHODS}
    weights, prior_sd = [], []
    for _ in range(n_universes):
        universe = simulate(rng, setting, n_stocks, **population)
        est = estimates(universe, population)
        beta = universe["beta"]
        chosen = np.argsort(est["point"]["least_squares"])[-top:]
        for m, value in est["point"].items():
            sq[m].append(np.mean((value - beta) ** 2))
            top_bias[m].append(np.mean(value[chosen] - beta[chosen]))
            exposure[m].append(np.sum(beta[chosen] - value[chosen]))
        for m, (lo, hi) in est["interval"].items():
            cover[m].append(np.mean((lo <= beta) & (beta <= hi)))
            width[m].append(np.median(hi - lo))
        weights.append(np.median(est["eb"]["weight"]))
        prior_sd.append(est["eb"]["prior_sd"])
    sq = {m: np.array(v) for m, v in sq.items()}
    out = {
        "setting": setting.label, "n_universes": n_universes, "n_stocks": n_stocks, "top": top,
        "median_se": float(np.median(est["se"])), "median_weight_on_estimate": float(np.median(weights)),
        "estimated_prior_sd": float(np.mean(prior_sd)),
        "rmse": {m: float(np.sqrt(v.mean())) for m, v in sq.items()},
        "mse_vs_least_squares": {m: float(v.mean() / sq["least_squares"].mean()) for m, v in sq.items()},
        "universes_beating_least_squares": {m: int(np.sum(v < sq["least_squares"])) for m, v in sq.items()},
        "top_bias": {m: float(np.mean(v)) for m, v in top_bias.items()},
        "hedge_exposure": {m: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for m, v in exposure.items()},
        "coverage": {m: float(np.mean(v)) for m, v in cover.items()},
        "median_width": {m: float(np.mean(v)) for m, v in width.items()},
    }
    if keep_draws:
        out["hedge_exposure_draws"] = {m: [float(x) for x in v] for m, v in exposure.items()}
    return out


def window_betas(prices, market, weeks=104):
    """Least-squares betas of weekly log returns in consecutive, non-overlapping windows.

    `prices` is a DataFrame of weekly adjusted closes (columns are stocks); `market` is the market's
    column of the same frame. A stock enters a window only with every week of it present.
    """
    returns = np.log(prices).diff().iloc[1:]
    rows = []
    for start in range(0, len(returns) - weeks + 1, weeks):
        window = returns.iloc[start:start + weeks]
        full = window.columns[window.notna().all()]
        stocks = [c for c in full if c != market]
        if len(stocks) < 3 or market not in full:
            continue
        b, se = least_squares(window[stocks].to_numpy().T, window[market].to_numpy())
        for name, beta, err in zip(stocks, b, se):
            rows.append({"window_start": window.index[0], "window_end": window.index[-1], "stock": name,
                         "beta": float(beta), "se": float(err)})
    return rows


def blume_test(rows, n_boot=0, seed=0, level=LEVEL):
    """Blume's test on real data: how well each estimator in one window predicts the next window's beta.

    The next window's least-squares beta is a noisy target, but the noise is the same for every
    predictor, so differences in mean squared error carry over to the true betas. With `n_boot`,
    a bootstrap over pairs of windows (the stocks of one pair share the same market shocks) gives an
    interval for each predictor's change in error relative to least squares.

    `constant_beta_slope` is the slope of next-window betas on current ones that estimation noise
    alone would produce if betas did not change; a lower observed slope means the betas also drift.
    """
    by_window = {}
    for r in rows:
        by_window.setdefault(r["window_start"], {})[r["stock"]] = r
    starts = sorted(by_window)
    blocks, pairs, noise_slopes = [], [], []
    for now, nxt in zip(starts[:-1], starts[1:]):
        common = sorted(set(by_window[now]) & set(by_window[nxt]))
        if len(common) < 3:
            continue
        b = np.array([by_window[now][s]["beta"] for s in common])
        se = np.array([by_window[now][s]["se"] for s in common])
        target = np.array([by_window[nxt][s]["beta"] for s in common])
        eb = empirical_bayes(b, se)
        preds = {"least_squares": b, "bloomberg": bloomberg(b), "bayes": eb["mean"],
                 "cross_section_mean": np.full_like(b, b.mean())}
        blocks.append({m: (p - target) ** 2 for m, p in preds.items()})
        pairs.extend(zip(b, target))
        noise_slopes.append(eb["prior_sd"] ** 2 / np.var(b))
    methods = list(blocks[0])
    pairs = np.array(pairs)
    out = {
        "window_pairs": len(blocks), "pairs": int(len(pairs)),
        "mse": {m: float(np.concatenate([blk[m] for blk in blocks]).mean()) for m in methods},
        "window_pairs_beating_least_squares": {m: int(sum(blk[m].mean() < blk["least_squares"].mean() for blk in blocks))
                                               for m in methods},
        "slope_next_on_current": float(np.polyfit(pairs[:, 0], pairs[:, 1], 1)[0]),
        "constant_beta_slope": float(np.mean(noise_slopes)),
    }
    if n_boot:
        rng = np.random.default_rng(seed)
        draws = {m: [] for m in methods}
        for _ in range(n_boot):
            pick = rng.integers(0, len(blocks), len(blocks))
            base = np.concatenate([blocks[i]["least_squares"] for i in pick]).mean()
            for m in methods:
                draws[m].append(np.concatenate([blocks[i][m] for i in pick]).mean() / base - 1)
        out["relative_change_interval"] = {m: [float(np.quantile(v, (1 - level) / 2)), float(np.quantile(v, (1 + level) / 2))]
                                           for m, v in draws.items()}
    return out
