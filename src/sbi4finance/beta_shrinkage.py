"""Betas across many stocks: one regression per stock, or Bayes' rule with a prior from the cross-section.

Each stock follows the market model r_i = alpha_i + beta_i r_m + e_i, with Gaussian errors. Least
squares gives each beta b_i and its standard error se_i. The Bayesian answer puts a normal prior on
the betas, beta_i ~ N(m, tau^2), standing for the population of stocks. With the residual variance
at its estimate (exact enough with 60 or more observations) the posterior is normal, and its mean
pulls each estimate toward m in proportion to the estimate's own noise:

    E[beta_i | data] = m + w_i (b_i - m),    w_i = tau^2 / (tau^2 + se_i^2).

Empirical Bayes (Vasicek 1973) estimates m and tau^2 from the cross-section, net of the sampling
noise (Lally 1998): tau^2 = var(b) - mean(se^2). A frequentist reaches the same estimator as the
empirical best linear unbiased predictor of a random-coefficients model (Robinson 1991).
Bloomberg's adjusted beta is the fixed rule 0.67 b + 0.33, from Blume's (1971) finding that betas
regress toward 1.

James and Stein (1961) proved that, for three or more means estimated with Gaussian errors of one
common, known variance, shrinking toward a fixed point has lower expected total squared error than
the separate estimates, whatever the true means; shrinking toward their own average needs four or
more (Lindley 1962). With unequal variances, as here, the theorem does not cover this estimator:
`MISSPECIFIED` holds populations where a fixed rule beats it, and one where least squares does.

``study`` measures all of this on simulated universes where the true betas are known, and
``blume_test`` on real stocks, where the next window's beta is the target.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

BLOOMBERG_WEIGHT = 0.67
LEVEL = 0.90
TAU2_FLOOR = 1e-6  # the smallest prior variance empirical Bayes allows; below it every estimate is pooled


@dataclass(frozen=True)
class Setting:
    """A data frequency and window: observations, periods per year and the market's annual volatility."""

    label: str
    n_obs: int
    per_year: int
    market_vol: float = 0.16

    @property
    def market_sd(self):
        """The market's sd per period."""
        return self.market_vol / np.sqrt(self.per_year)


SETTINGS = {
    "daily": Setting("daily returns, 1 year", 250, 252),
    "weekly": Setting("weekly returns, 2 years", 104, 52),
    "monthly": Setting("monthly returns, 5 years", 60, 12),
}


@dataclass(frozen=True)
class Population:
    """Where a universe's true betas and annual idiosyncratic volatilities come from.

    Betas are normal around `beta_mean` with sd `beta_sd`, or right-skewed with the same mean and sd
    when `skew` (the shape of a log-normal) is positive. Idiosyncratic volatility is log-normal
    around `idio_vol` with dispersion `idio_dispersion`, and its log is correlated `vol_on_beta`
    with the beta and `vol_on_distance` with the beta's distance from `beta_mean`. A share `fringe`
    of the stocks instead have betas at `beta_mean` ± `fringe_distance` and idiosyncratic
    volatility `fringe_vol`.
    """

    label: str
    beta_mean: float = 1.0
    beta_sd: float = 0.3
    idio_vol: float = 0.25
    idio_dispersion: float = 0.35
    skew: float = 0.0
    vol_on_beta: float = 0.0
    vol_on_distance: float = 0.0
    fringe: float = 0.0
    fringe_distance: float = 0.0
    fringe_vol: float = 0.0

    @property
    def mean(self):
        """The mean of the population's betas."""
        return self.beta_mean

    @property
    def sd(self):
        """The sd of the population's betas, fringe included."""
        return float(np.sqrt((1 - self.fringe) * self.beta_sd**2 + self.fringe * self.fringe_distance**2))

    def draw(self, rng, n):
        """True betas and annual idiosyncratic volatilities of `n` stocks."""
        if self.skew > 0:
            z = rng.lognormal(0.0, self.skew, n)
            z = (z - np.exp(self.skew**2 / 2)) / np.sqrt((np.exp(self.skew**2) - 1) * np.exp(self.skew**2))
        else:
            z = rng.standard_normal(n)
        beta = self.beta_mean + self.beta_sd * z
        distance = (np.abs(z) - np.sqrt(2 / np.pi)) / np.sqrt(1 - 2 / np.pi)  # |z| standardized, for a normal z
        rest = np.sqrt(1 - self.vol_on_beta**2 - self.vol_on_distance**2)
        log_vol = self.vol_on_beta * z + self.vol_on_distance * distance + rest * rng.standard_normal(n)
        idio_vol = self.idio_vol * np.exp(self.idio_dispersion * log_vol)
        k = int(round(self.fringe * n))
        if k:
            beta[:k] = self.beta_mean + self.fringe_distance * rng.choice([-1.0, 1.0], k)
            idio_vol[:k] = self.fringe_vol
        return beta, idio_vol


# Market volatility 16% a year and a median idiosyncratic volatility of 25% a year.
POPULATION = Population("normal betas, noise unrelated to beta")
MISSPECIFIED = {
    "skewed": Population("right-skewed betas", skew=0.25),
    "vol_on_beta": Population("right-skewed betas, noise rises with beta", skew=0.25, vol_on_beta=0.5),
    "vol_on_distance": Population("noise rises with distance from 1", vol_on_distance=0.5),
    "fringe": Population("a quiet core and a noisy fringe", beta_sd=0.1, idio_vol=0.15, idio_dispersion=0.0,
                         fringe=0.1, fringe_distance=0.6, fringe_vol=0.5),
}
UTILITIES = Population("utilities", beta_mean=0.6, beta_sd=0.2)


def simulate(rng, setting, n_stocks=500, population=POPULATION):
    """A universe of stocks: true betas from the population, then market-model returns for one window."""
    beta, idio_vol = population.draw(rng, n_stocks)
    idio_sd = idio_vol / np.sqrt(setting.per_year)
    market = rng.normal(0.0, setting.market_sd, setting.n_obs)
    returns = beta[:, None] * market[None, :] + idio_sd[:, None] * rng.standard_normal((n_stocks, setting.n_obs))
    return {"beta": beta, "idio_vol": idio_vol, "market": market, "returns": returns}


def least_squares(returns, market):
    """One market-model regression per stock: the beta and its standard error.

    `returns` has one row per stock and one column per period; `market` has one entry per period.
    """
    returns, market = np.atleast_2d(np.asarray(returns, dtype=float)), np.asarray(market, dtype=float)
    if returns.shape[1] != len(market):
        raise ValueError(f"returns have {returns.shape[1]} periods and the market {len(market)}")
    x = market - market.mean()
    if not np.any(x):
        raise ValueError("the market return is constant, so beta is not identified")
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
    """Vasicek's adjustment: the prior's mean and sd estimated from the cross-section, net of sampling noise.

    If the estimates spread no more than their noise explains, the prior variance is floored at
    TAU2_FLOOR and every estimate is pooled to the mean (`at_floor`).
    """
    prior_mean = float(np.mean(b))
    tau2 = float(np.var(b) - np.mean(se**2))
    prior_sd = float(np.sqrt(max(tau2, TAU2_FLOOR)))
    mean, sd, w = posterior(b, se, prior_mean, prior_sd)
    return {"mean": mean, "sd": sd, "weight": w, "prior_mean": prior_mean, "prior_sd": prior_sd, "at_floor": tau2 < TAU2_FLOOR}


def empirical_bayes_by_group(b, se, n_groups=5, min_group=10):
    """Empirical Bayes with one prior per group of stocks with similar noise.

    The groups are quantiles of the standard error; within one window every stock shares the
    market, so this ranks the stocks by idiosyncratic volatility. Each group's prior is learned from
    its own stocks. With fewer than `min_group` stocks per group, the groups are merged.
    """
    n_groups = max(1, min(n_groups, len(b) // min_group))
    group = np.argsort(np.argsort(se)) * n_groups // len(b)
    mean, sd = np.empty_like(b, dtype=float), np.empty_like(b, dtype=float)
    for g in range(n_groups):
        members = group == g
        eb = empirical_bayes(b[members], se[members])
        mean[members], sd[members] = eb["mean"], eb["sd"]
    return {"mean": mean, "sd": sd, "group": group}


def bloomberg(b):
    """Bloomberg's adjusted beta: two thirds of the estimate plus one third of 1."""
    return BLOOMBERG_WEIGHT * b + (1 - BLOOMBERG_WEIGHT)


POINT_METHODS = ("least_squares", "bloomberg", "bayes", "bayes_by_volatility", "oracle")
INTERVAL_METHODS = ("least_squares", "bayes")


def estimates(universe, population=POPULATION):
    """Every estimator on one universe, with the intervals the two interval methods give.

    The oracle is Bayes' rule with a normal prior at the population's true mean and sd, which
    empirical Bayes estimates.
    """
    b, se = least_squares(universe["returns"], universe["market"])
    eb = empirical_bayes(b, se)
    oracle, _, _ = posterior(b, se, population.mean, population.sd)
    t = stats.t.ppf(0.5 + LEVEL / 2, len(universe["market"]) - 2)
    z = stats.norm.ppf(0.5 + LEVEL / 2)
    return {
        "point": {"least_squares": b, "bloomberg": bloomberg(b), "bayes": eb["mean"],
                  "bayes_by_volatility": empirical_bayes_by_group(b, se)["mean"], "oracle": oracle},
        "interval": {"least_squares": (b - t * se, b + t * se), "bayes": (eb["mean"] - z * eb["sd"], eb["mean"] + z * eb["sd"])},
        "se": se, "eb": eb,
    }


def study(setting, population=POPULATION, n_universes=200, n_stocks=500, top=50, seed=0, keep_draws=False):
    """Average accuracy over simulated universes, and the hedge of the `top` stocks by estimated beta.

    The hedge holds one unit of each of the `top` stocks with the highest least-squares beta and
    sells each estimator's beta in the market; its residual market exposure is the sum of the true
    betas minus the estimated ones, in units of one stock's position. The same hedge on `top` stocks
    drawn at random (`random_hedge_exposure`) shows that the bias comes from the selection.
    `book_idio_risk` is the annual sd of the selected book's idiosyncratic P&L, in positions.

    Coverage is measured on all stocks, on the selected ones and on stocks whose true beta lies more
    than 1.5 population sds from the centre. `best_fixed_weight` is the weight on the estimate that
    minimizes expected squared error for one fixed rule, tau^2 / (tau^2 + mean se^2), with tau the
    population's sd. `keep_draws` adds every universe's exposure to the result.
    """
    if not 1 <= top <= n_stocks:
        raise ValueError(f"top must be between 1 and n_stocks ({n_stocks}), not {top}")
    rng = np.random.default_rng(seed)
    pick = np.random.default_rng([seed, 1])  # the random books, kept apart so the universes match across studies
    sq, top_bias, exposure, random_exposure = ({m: [] for m in POINT_METHODS} for _ in range(4))
    inside = {m: {"all": [0, 0], "top": [0, 0], "extreme": [0, 0]} for m in INTERVAL_METHODS}
    width = {m: [] for m in INTERVAL_METHODS}
    book_risk, se_median, best_weight, mean_weight, prior_sd, floors = [], [], [], [], [], 0
    for _ in range(n_universes):
        universe = simulate(rng, setting, n_stocks, population)
        est = estimates(universe, population)
        beta = universe["beta"]
        chosen = np.argsort(est["point"]["least_squares"])[n_stocks - top:]
        drawn = pick.choice(n_stocks, top, replace=False)
        subsets = {"all": np.ones(n_stocks, bool), "top": np.isin(np.arange(n_stocks), chosen),
                   "extreme": np.abs(beta - population.mean) > 1.5 * population.sd}
        for m, value in est["point"].items():
            sq[m].append(np.mean((value - beta) ** 2))
            top_bias[m].append(np.mean(value[chosen] - beta[chosen]))
            exposure[m].append(np.sum(beta[chosen] - value[chosen]))
            random_exposure[m].append(np.sum(beta[drawn] - value[drawn]))
        for m, (lo, hi) in est["interval"].items():
            covers = (lo <= beta) & (beta <= hi)
            for name, members in subsets.items():
                inside[m][name][0] += int(covers[members].sum())
                inside[m][name][1] += int(members.sum())
            width[m].append(np.median(hi - lo))
        book_risk.append(np.sqrt(np.sum(universe["idio_vol"][chosen] ** 2)))
        se_median.append(np.median(est["se"]))
        best_weight.append(population.sd**2 / (population.sd**2 + np.mean(est["se"] ** 2)))
        mean_weight.append(np.mean(est["eb"]["weight"]))
        prior_sd.append(est["eb"]["prior_sd"])
        floors += est["eb"]["at_floor"]
    sq = {m: np.array(v) for m, v in sq.items()}
    exposure = {m: np.array(v) for m, v in exposure.items()}
    out = {
        "setting": setting.label, "population": population.label, "n_universes": n_universes, "n_stocks": n_stocks, "top": top,
        "median_se": float(np.median(se_median)), "best_fixed_weight": float(np.mean(best_weight)),
        "mean_weight_on_estimate": float(np.mean(mean_weight)), "estimated_prior_sd": float(np.mean(prior_sd)),
        "prior_sd_at_floor": int(floors),
        "rmse": {m: float(np.sqrt(v.mean())) for m, v in sq.items()},
        "mse_vs_least_squares": {m: float(v.mean() / sq["least_squares"].mean()) for m, v in sq.items()},
        "universes_beating_least_squares": {m: int(np.sum(v < sq["least_squares"])) for m, v in sq.items()},
        "top_bias": {m: float(np.mean(v)) for m, v in top_bias.items()},
        "hedge_exposure": {m: {"mean": float(v.mean()), "sd": float(v.std()), "universes_short": int(np.sum(v < 0))}
                           for m, v in exposure.items()},
        "random_hedge_exposure": {m: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for m, v in random_exposure.items()},
        "book_idio_risk": float(np.mean(book_risk)),
        "coverage": {m: {name: hits / max(n, 1) for name, (hits, n) in counts.items()} for m, counts in inside.items()},
        "median_width": {m: float(np.median(v)) for m, v in width.items()},
    }
    if keep_draws:
        out["hedge_exposure_draws"] = {m: [float(x) for x in v] for m, v in exposure.items()}
    return out


def window_betas(prices, market, weeks=104):
    """Least-squares betas of weekly log returns in consecutive, non-overlapping windows.

    `prices` is a DataFrame of weekly adjusted closes indexed by date, one column per stock; `market`
    names the market's column. The windows end at the last week and run back in time, so the most
    recent data are always used; `window` numbers them from the oldest, and consecutive numbers are
    adjacent in time. A non-positive price counts as missing. A window is skipped if one of its
    returns spans more than twelve days (a missing week; the market's closure after 11 September
    2001 made one return span eleven) or the market has a gap in it; a stock enters a window only
    with every week of it present.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must be indexed by date")
    if market not in prices.columns:
        raise ValueError(f"no column {market!r} in the prices")
    prices = prices.sort_index()
    returns = np.log(prices.where(prices > 0)).diff().iloc[1:]
    returns = returns.where(np.isfinite(returns))
    span = np.diff(prices.index.values).astype("timedelta64[D]").astype(int)  # days covered by each return
    n_windows = len(returns) // weeks
    rows = []
    for k in range(n_windows):
        start = len(returns) - (n_windows - k) * weeks
        window = returns.iloc[start:start + weeks]
        if span[start:start + weeks].max() > 12:
            continue
        full = window.columns[window.notna().all()]
        stocks = [c for c in full if c != market]
        if len(stocks) < 3 or market not in full:
            continue
        b, se = least_squares(window[stocks].to_numpy().T, window[market].to_numpy())
        for name, beta, err in zip(stocks, b, se):
            rows.append({"window": k, "window_start": window.index[0], "window_end": window.index[-1], "stock": name,
                         "beta": float(beta), "se": float(err)})
    return rows


def blume_test(rows, n_boot=0, block=3, seed=0, level=LEVEL):
    """Blume's test on real data: how well each estimator in one window predicts the next window's beta.

    Windows k and k + 1 (the `window` numbers from `window_betas`) form a pair, with the stocks
    present in both. The next window's least-squares beta is a noisy target, but its noise is
    independent of every predictor, so differences in mean squared error carry over to the true
    betas. With `n_boot`, a moving-block bootstrap over consecutive pairs, in blocks of `block` pairs
    because consecutive pairs share a window, gives an interval for each predictor's change in
    error relative to least squares.

    The drift check fits next-window betas on current ones within each pair, net of the pair's
    averages, and compares the slope with `constant_beta_slope`, the slope estimation noise alone
    would give if betas did not change. A lower observed slope means the betas drift, or the
    standard errors understate the noise, or both. `change_over_noise` is the mean squared change in
    a beta between windows over the mean of the two squared standard errors' sum: 1 if betas are
    constant and the standard errors right.
    """
    by_window = {}
    for r in rows:
        by_window.setdefault(int(r["window"]), {})[r["stock"]] = r
    blocks, now, nxt, signal, spread, change, noise, away, below = [], [], [], [], [], [], [], [], []
    for k in sorted(by_window):
        common = sorted(set(by_window[k]) & set(by_window.get(k + 1, {})))
        if len(common) < 3:
            continue
        b = np.array([by_window[k][s]["beta"] for s in common])
        se = np.array([by_window[k][s]["se"] for s in common])
        target = np.array([by_window[k + 1][s]["beta"] for s in common])
        target_se = np.array([by_window[k + 1][s]["se"] for s in common])
        eb = empirical_bayes(b, se)
        preds = {"least_squares": b, "bloomberg": bloomberg(b), "bayes": eb["mean"], "cross_section_mean": np.full_like(b, b.mean())}
        blocks.append({m: (p - target) ** 2 for m, p in preds.items()})
        now.append(b - b.mean())
        nxt.append(target - target.mean())
        signal.append(len(b) * eb["prior_sd"] ** 2)
        spread.append(len(b) * np.var(b))
        change.append((target - b) ** 2)
        noise.append(se**2 + target_se**2)
        away.append(np.abs(bloomberg(b) - target) > np.abs(b - target))
        below.append(b < 1)
    if not blocks:
        raise ValueError("no two adjacent windows share three stocks")
    methods = list(blocks[0])
    x, y = np.concatenate(now), np.concatenate(nxt)
    out = {
        "window_pairs": len(blocks), "pairs": int(len(x)),
        "mse": {m: float(np.concatenate([blk[m] for blk in blocks]).mean()) for m in methods},
        "window_pairs_beating_least_squares": {m: int(sum(blk[m].mean() < blk["least_squares"].mean() for blk in blocks))
                                               for m in methods},
        "slope_next_on_current": float(x @ y / (x @ x)),
        "constant_beta_slope": float(np.sum(signal) / np.sum(spread)),
        "change_over_noise": float(np.concatenate(change).mean() / np.concatenate(noise).mean()),
        "bloomberg_moved_away": float(np.concatenate(away).mean()),
        "share_below_one": float(np.concatenate(below).mean()),
    }
    if n_boot:
        if len(blocks) < 2 * block:
            raise ValueError(f"{len(blocks)} window pairs are too few for blocks of {block}")
        rng = np.random.default_rng(seed)
        n_blocks = -(-len(blocks) // block)
        draws = {m: [] for m in methods}
        for _ in range(n_boot):
            starts = rng.integers(0, len(blocks) - block + 1, n_blocks)
            pick = np.concatenate([np.arange(s, s + block) for s in starts])[:len(blocks)]
            base = np.concatenate([blocks[i]["least_squares"] for i in pick]).mean()
            for m in methods:
                draws[m].append(np.concatenate([blocks[i][m] for i in pick]).mean() / base - 1)
        out["relative_change_interval"] = {m: [float(np.quantile(v, (1 - level) / 2)), float(np.quantile(v, (1 + level) / 2))]
                                           for m, v in draws.items()}
    return out
