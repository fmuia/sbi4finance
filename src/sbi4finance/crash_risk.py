"""How likely is a crash day? Exact Bayesian and classical answers from 1,000 daily returns.

Daily log returns y_t are in percent. Two tail models, each with an exact likelihood:

* ``JumpModel``: y = mu + sd * eps + B * xi, with B ~ Bernoulli(jumps_per_year / 252) and
  xi ~ N(mean_jump, jump_sd^2). This is Merton's (1976) jump model in daily form, the jump part of
  slot 5's model, and its prior is slot 5's.
* ``StudentModel``: y = mu + scale * t_nu, the usual fat-tailed alternative.

The question is ``crash_probability``: the probability of at least one day with a loss of ``x``% or
more over the next year. ``analyse`` answers it, from one history, with every method compared in
the crash-day notebook (``notebooks/tutorial_6_crash_risk.py``):

* Bayes: the exact posterior on a grid, its predictive probability (the posterior mean) and its
  equal-tailed interval, under the model's prior and under a flat prior on the grid's box;
* maximum likelihood: the plug-in probability, a Wald interval by the delta method on the logit
  scale, and the profile likelihood-ratio interval;
* the posterior mode (penalized likelihood) with a Laplace interval;
* the bootstrap, nonparametric (resampled days) and parametric (histories simulated from the fit):
  the average plug-in probability over the refits and its percentile interval;
* the other tail family, fitted by maximum likelihood: what a practitioner who picked the other
  model would report;
* model-free baselines: the empirical frequency and peaks over threshold (a generalized Pareto tail
  on the largest 5% of losses, and on 2.5% and 10%).

For the jump model it also returns intervals for the jump frequency itself, including the
classical route that first flags jump days with a threshold and then counts them.

The drift is fixed at the sample mean for every method: it barely moves a tail probability.
"""

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats

YEAR = 252
LEVEL_INTERVAL = 0.90          # intervals for the crash probability
LEVEL_PARAMETER = 0.95         # intervals for the jump frequency
CHI2 = {0.90: 2.705543454095404, 0.95: 3.841458820694124}
LOG_2PI = np.log(2 * np.pi)


def year_probability(daily, days=YEAR):
    """P(at least one event in `days` independent days), from the daily probability; stable for small values."""
    daily = np.clip(np.asarray(daily, dtype=float), 0.0, 1.0)
    with np.errstate(divide="ignore"):
        return -np.expm1(days * np.log1p(-daily))


def _logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def _weighted_quantiles(values, weights, probs):
    """Quantiles of a discrete distribution; each atom carries its mass at its centre."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cdf = np.cumsum(w) - 0.5 * w
    return np.interp(probs, cdf / w.sum(), v)


def _nearest_pd_inverse(H):
    """Inverse of a symmetric Hessian, or None when it is not positive definite."""
    H = 0.5 * (H + H.T)
    if not np.all(np.isfinite(H)):
        return None
    try:
        if np.linalg.eigvalsh(H).min() <= 0:
            return None
        return np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None


def _hessian(grad, x, step=1e-4):
    k = len(x)
    H = np.empty((k, k))
    for i in range(k):
        e = np.zeros(k)
        e[i] = step
        H[i] = (grad(x + e) - grad(x - e)) / (2 * step)
    return 0.5 * (H + H.T)


def _numeric_grad(f, x, step=1e-5):
    g = np.empty(len(x))
    for i in range(len(x)):
        e = np.zeros(len(x))
        e[i] = step
        g[i] = (f(x + e) - f(x - e)) / (2 * step)
    return g


# --------------------------------------------------------------------------------------------------
# The jump model
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class JumpModel:
    """Free coordinates x = (log jumps per year, log jump sd %, mean jump %, log diffusion sd % per day)."""

    name: str = "jump"
    names: tuple = ("log_jumps_per_year", "log_jump_sd", "mean_jump", "log_sd")
    prior_mean: tuple = (np.log(2.5), np.log(3.0), -1.0, np.log(100 * 0.25 / np.sqrt(YEAR)))
    prior_sd: tuple = (0.75, 0.45, 2.5, 0.40)
    drift_prior: tuple = (8.0 / YEAR, 10.0 / YEAR)
    max_daily_jump_probability: float = 0.5

    # ---- the model ----
    def bounds(self):
        b = [(m - 6 * s, m + 6 * s) for m, s in zip(self.prior_mean, self.prior_sd)]
        b[0] = (b[0][0], min(b[0][1], np.log(self.max_daily_jump_probability * YEAR) - 1e-6))
        return b

    def draw_truth(self, rng):
        while True:
            x = rng.normal(self.prior_mean, self.prior_sd)
            if np.exp(x[0]) / YEAR < self.max_daily_jump_probability:
                return x, rng.normal(*self.drift_prior)

    def simulate(self, x, mu, n, rng, with_jumps=False):
        lam, jump_sd, mean_jump, sd = np.exp(x[0]), np.exp(x[1]), x[2], np.exp(x[3])
        jumps = rng.random(n) < lam / YEAR
        y = mu + sd * rng.standard_normal(n) + jumps * rng.normal(mean_jump, jump_sd, n)
        return (y, jumps) if with_jumps else y

    def loglik(self, x, y, mu):
        value, _ = self.loglik_and_grad(x, y, mu)
        return value

    def loglik_and_grad(self, x, y, mu):
        log_lam, log_jsd, mean_jump, log_sd = x
        p = np.exp(log_lam) / YEAR
        s2, j2 = np.exp(2 * log_sd), np.exp(2 * log_jsd)
        v2 = s2 + j2
        d = y - mu
        e = d - mean_jump
        a = np.log1p(-p) - 0.5 * (LOG_2PI + np.log(s2)) - 0.5 * d * d / s2
        b = np.log(p) - 0.5 * (LOG_2PI + np.log(v2)) - 0.5 * e * e / v2
        ll = np.logaddexp(a, b)
        wa, wb = np.exp(a - ll), np.exp(b - ll)
        db_dv2 = -0.5 / v2 + 0.5 * e * e / v2**2
        grad = np.array([
            np.sum(wa * (-p / (1 - p)) + wb),
            np.sum(wb * db_dv2 * 2 * j2),
            np.sum(wb * e / v2),
            np.sum(wa * (-1 + d * d / s2) + wb * db_dv2 * 2 * s2),
        ])
        return float(ll.sum()), grad

    def log_prior_and_grad(self, x):
        m, s = np.array(self.prior_mean), np.array(self.prior_sd)
        z = (np.asarray(x) - m) / s
        return float(-0.5 * np.sum(z * z)), -z / s

    def daily_crash_probability(self, threshold, mu, lam, jump_sd, mean_jump, sd):
        p = np.asarray(lam) / YEAR
        no_jump = stats.norm.cdf((-threshold - mu) / sd)
        jump = stats.norm.cdf((-threshold - mu - mean_jump) / np.sqrt(sd**2 + jump_sd**2))
        return (1 - p) * no_jump + p * jump

    def crash_probability(self, x, threshold, mu):
        x = np.asarray(x, dtype=float)
        q = self.daily_crash_probability(threshold, mu, np.exp(x[..., 0]), np.exp(x[..., 1]), x[..., 2],
                                         np.exp(x[..., 3]))
        return year_probability(q)

    def starts(self, y, mu):
        sd = 1.4826 * np.median(np.abs(y - np.median(y)))
        return [np.array([np.log(2.5), np.log(3.0), -1.0, np.log(sd)]),
                np.array([np.log(0.5), np.log(6.0), -3.0, np.log(sd)]),
                np.array([np.log(20.0), np.log(1.2), -0.5, np.log(0.8 * sd)])]

    # ---- the exact posterior on a grid ----
    def grid(self, y, mu, n_lam=45, n_jsd=33, n_mj=43, n_var=11):
        """Log-likelihood and log-prior on a grid in (log jumps per year, log jump sd, mean jump, log variance).

        The variance coordinate is the total daily variance, which the data pin down well and which is
        nearly orthogonal to the jump split; the diffusion sd follows from it. The prior is slot 5's,
        carried over with the Jacobian of that change of coordinates.
        """
        n = len(y)
        d = y - mu
        var = np.var(y)
        kurt = np.mean(d**4) / var**2
        half = max(0.15, 4 * np.sqrt(max(kurt - 1, 2) / n))
        (lam_lo, lam_hi), (jsd_lo, jsd_hi) = self.bounds()[:2]
        axes = {  # the box of the maximum-likelihood fit, so that the profile sees the same parameters
            "log_lam": np.linspace(lam_lo, lam_hi, n_lam),
            "log_jsd": np.linspace(jsd_lo, jsd_hi, n_jsd),
            "mean_jump": np.linspace(-12.0, 9.0, n_mj),  # steps of 0.5%: finer than any posterior here
            "log_var": np.linspace(np.log(var) - half, np.log(var) + half, n_var),
        }
        L, J, M, V = np.meshgrid(*axes.values(), indexing="ij")
        p = np.exp(L) / YEAR
        j2 = np.exp(2 * J)
        s2 = np.exp(V) - p * j2 - p * (1 - p) * M**2
        valid = s2 > 1e-4 * np.exp(V)
        s2 = np.where(valid, s2, np.nan)
        ll = np.full(L.shape, -np.inf)
        d32 = d.astype(np.float32)  # per-day terms in float32, sums in float64: about 1e-3 on a log-likelihood
        dd = d32 * d32
        for i in range(n_lam):
            ok = valid[i]
            s2i, v2i = s2[i][ok][:, None], (s2[i] + j2[i])[ok][:, None]
            const_a = (np.log1p(-p[i][ok]) - 0.5 * (LOG_2PI + np.log(s2[i][ok])))[:, None].astype(np.float32)
            const_b = (np.log(p[i][ok]) - 0.5 * (LOG_2PI + np.log((s2[i] + j2[i])[ok])))[:, None].astype(np.float32)
            a = const_a - (0.5 / s2i).astype(np.float32) * dd
            e = d32 - M[i][ok][:, None].astype(np.float32)
            b = const_b - (0.5 / v2i).astype(np.float32) * e * e
            row = ll[i]
            row[ok] = np.logaddexp(a, b).sum(axis=1, dtype=np.float64)
            ll[i] = row
        m, s = self.prior_mean, self.prior_sd
        log_sd = 0.5 * np.log(s2)
        lp = (-0.5 * ((L - m[0]) / s[0]) ** 2 - 0.5 * ((J - m[1]) / s[1]) ** 2 - 0.5 * ((M - m[2]) / s[2]) ** 2
              - 0.5 * ((log_sd - m[3]) / s[3]) ** 2 + np.log(0.5 * np.exp(V) / s2))
        lp = np.where(valid, lp, -np.inf)
        params = np.stack([L, J, M, log_sd], axis=-1)  # free coordinates at every grid point
        return {"axes": axes, "ll": ll, "lp": lp, "x": params, "valid": valid}


# --------------------------------------------------------------------------------------------------
# The Student-t model
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StudentModel:
    """Free coordinates x = (log nu, log scale % per day); the prior is on nu and on the annualized sd."""

    name: str = "student"
    names: tuple = ("log_nu", "log_scale")
    nu_prior: tuple = (np.log(5.0), 0.6)
    sd_prior: tuple = (np.log(100 * 0.25 / np.sqrt(YEAR)), 0.40)
    drift_prior: tuple = (8.0 / YEAR, 10.0 / YEAR)
    nu_min: float = 2.1
    nu_max: float = 300.0

    def bounds(self):
        return [(np.log(self.nu_min), np.log(self.nu_max)), (np.log(0.05), np.log(10.0))]

    def _log_sd(self, x):
        nu = np.exp(x[..., 0])
        return x[..., 1] + 0.5 * np.log(nu / (nu - 2))

    def draw_truth(self, rng):
        while True:
            log_nu = rng.normal(*self.nu_prior)
            if self.nu_min < np.exp(log_nu) < self.nu_max:
                break
        nu = np.exp(log_nu)
        log_sd = rng.normal(*self.sd_prior)
        return np.array([log_nu, log_sd - 0.5 * np.log(nu / (nu - 2))]), rng.normal(*self.drift_prior)

    def simulate(self, x, mu, n, rng):
        return mu + np.exp(x[1]) * rng.standard_t(np.exp(x[0]), n)

    def loglik(self, x, y, mu):
        return self.loglik_and_grad(x, y, mu)[0]

    def loglik_and_grad(self, x, y, mu):
        nu, scale = np.exp(x[0]), np.exp(x[1])
        z2 = ((y - mu) / scale) ** 2
        n = len(y)
        log1p = np.log1p(z2 / nu)
        ll = n * (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - x[1]) \
            - 0.5 * (nu + 1) * log1p.sum()
        d_nu = n * 0.5 * (special.digamma((nu + 1) / 2) - special.digamma(nu / 2) - 1 / nu) \
            - 0.5 * log1p.sum() + 0.5 * (nu + 1) * np.sum(z2 / (nu * (nu + z2)))
        d_log_scale = np.sum(-1 + (nu + 1) * z2 / (nu + z2))
        return float(ll), np.array([nu * d_nu, d_log_scale])

    def log_prior_and_grad(self, x):
        def f(z):
            return float(-0.5 * ((z[0] - self.nu_prior[0]) / self.nu_prior[1]) ** 2
                         - 0.5 * ((self._log_sd(z) - self.sd_prior[0]) / self.sd_prior[1]) ** 2)
        x = np.asarray(x, dtype=float)
        return f(x), _numeric_grad(f, x)

    def crash_probability(self, x, threshold, mu):
        x = np.asarray(x, dtype=float)
        q = stats.t.cdf((-threshold - mu) / np.exp(x[..., 1]), np.exp(x[..., 0]))
        return year_probability(q)

    def starts(self, y, mu):
        scale = 1.4826 * np.median(np.abs(y - np.median(y)))
        return [np.array([np.log(5.0), np.log(scale)]), np.array([np.log(30.0), np.log(scale)]),
                np.array([np.log(3.0), np.log(0.8 * scale)])]

    def grid(self, y, mu, n_nu=81, n_scale=61):
        scale = 1.4826 * np.median(np.abs(y - np.median(y)))
        axes = {"log_nu": np.linspace(np.log(self.nu_min), np.log(self.nu_max), n_nu),
                "log_scale": np.linspace(np.log(scale) - 0.6, np.log(scale) + 0.6, n_scale)}
        N, S = np.meshgrid(*axes.values(), indexing="ij")
        nu, sc = np.exp(N)[..., None], np.exp(S)[..., None]
        z = (y - mu) / sc
        ll = np.sum(special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(sc)
                    - (nu + 1) / 2 * np.log1p(z * z / nu), axis=-1)
        x = np.stack([N, S], axis=-1)
        lp = (-0.5 * ((N - self.nu_prior[0]) / self.nu_prior[1]) ** 2
              - 0.5 * ((self._log_sd(x) - self.sd_prior[0]) / self.sd_prior[1]) ** 2)
        return {"axes": axes, "ll": ll, "lp": lp, "x": x, "valid": np.ones(N.shape, bool)}


MODELS = {"jump": JumpModel(), "student": StudentModel()}


# --------------------------------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------------------------------


def fit(model, y, mu, *, prior=False, starts=None):
    """Maximum likelihood (prior=False) or the posterior mode (prior=True), with the Hessian of the objective."""
    def objective(x):
        ll, g = model.loglik_and_grad(x, y, mu)
        if prior:
            lp, gp = model.log_prior_and_grad(x)
            ll, g = ll + lp, g + gp
        return -ll, -g

    bounds = model.bounds()
    lo, hi = np.array(bounds).T
    best = None
    for s in starts if starts is not None else model.starts(y, mu):
        r = optimize.minimize(objective, np.clip(s, lo + 1e-6, hi - 1e-6), jac=True, method="L-BFGS-B",
                              bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    on_bound = bool(np.any(np.minimum(np.abs(x - lo), np.abs(hi - x)) < 1e-3))
    cov = _nearest_pd_inverse(_hessian(lambda z: objective(z)[1], x))
    return {"x": x, "value": -best.fun, "on_bound": on_bound, "cov": cov}


def delta_interval(model, fitted, threshold, mu, level=LEVEL_INTERVAL):
    """Gaussian (Wald or Laplace) interval for the crash probability, by the delta method on the logit scale."""
    if fitted["cov"] is None:
        return (np.nan, np.nan)
    g = lambda z: _logit(model.crash_probability(z, threshold, mu))
    grad = _numeric_grad(g, fitted["x"])
    se = float(np.sqrt(max(grad @ fitted["cov"] @ grad, 0.0)))
    z = stats.norm.ppf(0.5 + level / 2)
    centre = g(fitted["x"])
    return tuple(special.expit([centre - z * se, centre + z * se]))


def wald_parameter_interval(fitted, index=0, level=LEVEL_PARAMETER):
    if fitted["cov"] is None:
        return (np.nan, np.nan)
    z = stats.norm.ppf(0.5 + level / 2)
    se = np.sqrt(fitted["cov"][index, index])
    return (fitted["x"][index] - z * se, fitted["x"][index] + z * se)


def _posterior_summaries(log_post, ok, crash, first):
    w = np.exp(log_post[ok] - log_post[ok].max())
    w /= w.sum()
    return {"predictive": float(np.sum(w * crash[ok])),
            "interval": tuple(_weighted_quantiles(crash[ok], w, [0.5 - LEVEL_INTERVAL / 2, 0.5 + LEVEL_INTERVAL / 2])),
            "parameter_interval": tuple(_weighted_quantiles(first[ok], w, [0.5 - LEVEL_PARAMETER / 2,
                                                                            0.5 + LEVEL_PARAMETER / 2]))}


def grid_summaries(model, grid, threshold, mu, ll_max):
    """Exact posterior summaries and profile likelihood-ratio intervals, from one grid evaluation.

    The posterior is computed twice: under the model's prior ("bayes"), and under a flat prior on the
    grid's box in the grid's own coordinates ("bayes_flat"), which separates what the prior adds from
    what averaging over the likelihood adds.
    """
    ll, lp, x = grid["ll"], grid["lp"], grid["x"]
    ok = np.isfinite(ll) & np.isfinite(lp)
    crash = np.where(ok, model.crash_probability(np.where(ok[..., None], x, 0.0), threshold, mu), np.nan)
    first = x[..., 0]
    out = {"bayes": _posterior_summaries(ll + lp, ok, crash, first),
           "bayes_flat": _posterior_summaries(ll, ok, crash, first)}
    # The likelihood-ratio regions are measured from the best value found, by the optimizer or the grid.
    # When the optimum lies off the grid (a mean jump beyond the axis), fall back to the grid's own best.
    ll_max = max(ll_max, ll[ok].max())
    out["profile_off_grid"] = bool(2 * (ll_max - ll[ok].max()) > CHI2[LEVEL_INTERVAL])
    if out["profile_off_grid"]:
        ll_max = ll[ok].max()
    region = ok & (2 * (ll_max - ll) <= CHI2[LEVEL_INTERVAL])
    out["profile_interval"] = (float(np.nanmin(crash[region])), float(np.nanmax(crash[region])))
    region_p = ok & (2 * (ll_max - ll) <= CHI2[LEVEL_PARAMETER])
    lo_axis, hi_axis = first[ok].min(), first[ok].max()
    lo, hi = float(first[region_p].min()), float(first[region_p].max())
    out["profile_parameter_interval"] = (lo, hi)
    out["profile_parameter_open"] = bool(lo <= lo_axis + 1e-9 or hi >= hi_axis - 1e-9)
    out["grid_mode"] = x[np.unravel_index(np.argmax(np.where(ok, ll, -np.inf)), ll.shape)]
    return out


def jump_plane(model, y, mu, grid, threshold):
    """The (jump frequency, jump size) plane of a jump-model grid.

    At each point: the profile log-likelihood (maximized over the mean jump and the volatility, from the
    grid's best cell and the neighbouring points), the crash probability at that best fit, and the
    posterior marginal (the grid's mean jump and variance summed out).
    """
    log_lam, log_jsd = grid["axes"]["log_lam"], grid["axes"]["log_jsd"]
    shape = (len(log_lam), len(log_jsd))
    ll = np.where(np.isfinite(grid["ll"]), grid["ll"], -np.inf).reshape(*shape, -1)
    grid_x = grid["x"].reshape(*shape, -1, 4)
    profile, x = np.full(shape, -np.inf), np.full((*shape, 4), np.nan)
    for i, j in np.ndindex(shape):
        fixed = (log_lam[i], log_jsd[j])

        def objective(z):
            value, grad = model.loglik_and_grad(np.array([*fixed, *z]), y, mu)
            return -value, -grad[2:]

        k = int(np.argmax(ll[i, j]))
        starts = [grid_x[i, j, k, 2:]] if np.isfinite(ll[i, j, k]) else []
        starts += [x[a, b, 2:] for a, b in ((i, j - 1), (i - 1, j)) if min(a, b) >= 0]
        for s in starts or [np.array([model.prior_mean[2], np.log(np.std(y))])]:
            r = optimize.minimize(objective, s, jac=True, method="L-BFGS-B", bounds=model.bounds()[2:])
            if -r.fun > profile[i, j]:
                profile[i, j], x[i, j] = -r.fun, [*fixed, *r.x]
    ok = np.isfinite(grid["ll"]) & np.isfinite(grid["lp"])
    log_post = np.where(ok, grid["ll"] + grid["lp"], -np.inf)
    w = np.exp(log_post - log_post[ok].max())
    return {"jumps_per_year": np.exp(log_lam), "jump_sd": np.exp(log_jsd), "profile": profile,
            "crash": model.crash_probability(x, threshold, mu), "x": x, "posterior": (w / w.sum()).sum(axis=(2, 3))}


def ridge_ends(plane, level=0.95):
    """The points of the joint likelihood region with the lowest and the highest crash probability."""
    inside = 2 * (np.nanmax(plane["profile"]) - plane["profile"]) <= stats.chi2.ppf(level, 2)
    crash = np.where(inside, plane["crash"], np.nan)
    ends = {}
    for name, k in (("low", np.nanargmin(crash)), ("high", np.nanargmax(crash))):
        i, j = np.unravel_index(k, crash.shape)
        ends[name] = {"crash": float(crash[i, j]), "jumps_per_year": float(plane["jumps_per_year"][i]),
                      "jump_sd": float(plane["jump_sd"][j]), "mean_jump": float(plane["x"][i, j, 2]),
                      "annual_vol": float(np.exp(plane["x"][i, j, 3]) * np.sqrt(YEAR) / 100)}
    return ends


def bootstrap(model, y, mu, x_hat, n_boot, rng):
    """Nonparametric bootstrap of the days: the maximum-likelihood refit of every resampled history."""
    starts = [x_hat, model.starts(y, mu)[0]]
    return np.array([fit(model, rng.choice(y, size=len(y), replace=True), mu, starts=starts)["x"]
                     for _ in range(n_boot)])


def parametric_bootstrap(model, y, mu, x_hat, n_boot, rng):
    """Parametric bootstrap: histories simulated from the maximum-likelihood fit, each refitted. Averaging the
    plug-in probability over them is the bootstrap predictive distribution of Harris (1989)."""
    starts = [x_hat, model.starts(y, mu)[0]]
    return np.array([fit(model, model.simulate(x_hat, mu, len(y), rng), mu, starts=starts)["x"]
                     for _ in range(n_boot)])


def pot_fit(y, tail_fraction=0.05):
    """Peaks over threshold: a generalized Pareto tail fitted to the largest 5% of daily losses."""
    losses = -np.asarray(y)
    u = np.quantile(losses, 1 - tail_fraction)
    excess = losses[losses > u] - u
    shape, _, scale = stats.genpareto.fit(excess, floc=0)
    return {"u": float(u), "rate": len(excess) / len(losses), "shape": float(shape), "scale": float(scale),
            "losses": losses}


def pot_crash_probability(pot, threshold):
    if threshold <= pot["u"]:
        daily = np.mean(pot["losses"] >= threshold)
    else:
        daily = pot["rate"] * stats.genpareto.sf(threshold - pot["u"], pot["shape"], loc=0, scale=pot["scale"])
    return float(year_probability(daily))


def threshold_jump_count(y, k=3.0, level=LEVEL_PARAMETER):
    """Flag days beyond k robust sds as jumps, then a Poisson interval for the yearly count (Garwood)."""
    centre = np.median(y)
    sd = 1.4826 * np.median(np.abs(y - centre))
    count = int(np.sum(np.abs(y - centre) > k * sd))
    years = len(y) / YEAR
    a = 1 - level
    lo = stats.chi2.ppf(a / 2, 2 * count) / 2 if count > 0 else 0.0
    hi = stats.chi2.ppf(1 - a / 2, 2 * count + 2) / 2
    return count / years, (lo / years, hi / years)


def analyse(model, y, thresholds=(5.0, 7.0), *, n_boot=100, rng=None, grid=None):
    """Every method's crash probability, a point forecast and an interval, for each loss threshold in %."""
    rng = np.random.default_rng() if rng is None else rng
    mu = float(np.mean(y))
    g = model.grid(y, mu) if grid is None else grid
    ok = np.isfinite(g["ll"])
    mode = g["x"][np.unravel_index(np.argmax(np.where(ok, g["ll"], -np.inf)), g["ll"].shape)]
    mle = fit(model, y, mu, starts=model.starts(y, mu) + [mode])
    lp_ok = ok & np.isfinite(g["lp"])
    post_mode = g["x"][np.unravel_index(np.argmax(np.where(lp_ok, g["ll"] + g["lp"], -np.inf)), g["ll"].shape)]
    mapfit = fit(model, y, mu, prior=True, starts=model.starts(y, mu) + [post_mode])
    boot_x = bootstrap(model, y, mu, mle["x"], n_boot, rng)
    param_x = parametric_bootstrap(model, y, mu, mle["x"], n_boot, rng)
    pots = {"evt": pot_fit(y), "evt_2.5": pot_fit(y, 0.025), "evt_10": pot_fit(y, 0.10)}
    other = MODELS["student" if model.name == "jump" else "jump"]  # the other tail family, fitted by maximum likelihood
    other_fit = fit(other, y, mu)
    lo_q, hi_q = 0.5 - LEVEL_INTERVAL / 2, 0.5 + LEVEL_INTERVAL / 2
    out = {
        "mu": mu, "grid_points": int(ok.sum()),
        "mle_x": mle["x"].tolist(), "mle_on_bound": mle["on_bound"], "mle_hessian_ok": mle["cov"] is not None,
        "map_x": mapfit["x"].tolist(), "evt_shape": pots["evt"]["shape"], "other_family_x": other_fit["x"].tolist(),
        "by_threshold": {},
    }
    summ = None
    for threshold in thresholds:
        summ = grid_summaries(model, g, threshold, mu, mle["value"])
        boot_crash = model.crash_probability(boot_x, threshold, mu)
        param_crash = model.crash_probability(param_x, threshold, mu)
        out["by_threshold"][f"{threshold:g}"] = {
            "point": {
                "bayes": summ["bayes"]["predictive"],
                "bayes_flat": summ["bayes_flat"]["predictive"],
                "mle": float(model.crash_probability(mle["x"], threshold, mu)),
                "map": float(model.crash_probability(mapfit["x"], threshold, mu)),
                "bootstrap": float(np.mean(boot_crash)),
                "param_bootstrap": float(np.mean(param_crash)),
                **{name: pot_crash_probability(pot, threshold) for name, pot in pots.items()},
                "other_family": float(other.crash_probability(other_fit["x"], threshold, mu)),
                "empirical": float(year_probability(np.mean(-y >= threshold))),
            },
            "interval": {
                "bayes": [float(v) for v in summ["bayes"]["interval"]],
                "bayes_flat": [float(v) for v in summ["bayes_flat"]["interval"]],
                "wald": [float(v) for v in delta_interval(model, mle, threshold, mu)],
                "laplace": [float(v) for v in delta_interval(model, mapfit, threshold, mu)],
                "profile": [float(v) for v in summ["profile_interval"]],
                "bootstrap": [float(v) for v in np.quantile(boot_crash, [lo_q, hi_q])],
                "param_bootstrap": [float(v) for v in np.quantile(param_crash, [lo_q, hi_q])],
            },
        }
        out["profile_off_grid"] = summ["profile_off_grid"]
    if model.name == "jump":
        lo_p, hi_p = 0.5 - LEVEL_PARAMETER / 2, 0.5 + LEVEL_PARAMETER / 2
        rate, (c_lo, c_hi) = threshold_jump_count(y)
        out["jump_frequency_interval"] = {  # log jumps per year
            "bayes": [float(v) for v in summ["bayes"]["parameter_interval"]],
            "bayes_flat": [float(v) for v in summ["bayes_flat"]["parameter_interval"]],
            "wald": [float(v) for v in wald_parameter_interval(mle)],
            "laplace": [float(v) for v in wald_parameter_interval(mapfit)],
            "profile": [float(v) for v in summ["profile_parameter_interval"]],
            "bootstrap": [float(v) for v in np.quantile(boot_x[:, 0], [lo_p, hi_p])],
            "param_bootstrap": [float(v) for v in np.quantile(param_x[:, 0], [lo_p, hi_p])],
            "threshold_count": [float(np.log(c_lo)) if c_lo > 0 else float("-inf"), float(np.log(c_hi))],
        }
        out["profile_parameter_open"] = summ["profile_parameter_open"]
        out["threshold_rate"] = rate
    return out


POINT_METHODS = ("bayes", "bayes_flat", "mle", "map", "bootstrap", "param_bootstrap", "evt", "evt_2.5", "evt_10",
                 "other_family", "empirical")
INTERVAL_METHODS = ("bayes", "bayes_flat", "wald", "laplace", "profile", "bootstrap", "param_bootstrap")
FREQUENCY_METHODS = ("bayes", "bayes_flat", "wald", "laplace", "profile", "bootstrap", "param_bootstrap", "threshold_count")
CALIBRATION_EDGES = (0.0, 0.005, 0.02, 0.05, 0.1, 0.2, 1.0)


def score(records, threshold="7"):
    """Accuracy and calibration of every method's crash probability, against the true probability."""
    p = np.array([r["p_true"][threshold] for r in records])
    out = {"n_series": len(records), "mean_true_probability": float(p.mean()), "point": {}, "interval": {},
           "calibration": {}}
    for m in POINT_METHODS:
        q = np.array([r["result"]["by_threshold"][threshold]["point"][m] for r in records])
        qc = np.clip(q, 1e-5, 1 - 1e-5)
        kl = p * np.log(np.clip(p, 1e-12, 1) / qc) + (1 - p) * np.log(np.clip(1 - p, 1e-12, 1) / (1 - qc))
        log_ratio = np.abs(np.log(np.clip(q, 1e-12, 1) / np.clip(p, 1e-12, 1)))
        out["point"][m] = {
            "excess_log_loss": float(kl.mean()),
            "brier_excess": float(np.mean((q - p) ** 2)),
            "within_factor_2": float(np.mean(log_ratio <= np.log(2))),
            "median_factor": float(np.exp(np.median(log_ratio))),
            "zero_forecasts": float(np.mean(q < 1e-6)),
            "understated_10x": float(np.mean(q < p / 10)),  # the misses that hurt a risk manager
            "overstated_10x": float(np.mean(q > 10 * p)),
        }
        bins = np.digitize(q, CALIBRATION_EDGES[1:-1])
        out["calibration"][m] = [
            {"from": CALIBRATION_EDGES[b], "to": CALIBRATION_EDGES[b + 1], "count": int(np.sum(bins == b)),
             "mean_forecast": float(q[bins == b].mean()) if np.any(bins == b) else None,
             "mean_truth": float(p[bins == b].mean()) if np.any(bins == b) else None}
            for b in range(len(CALIBRATION_EDGES) - 1)]
    for m in INTERVAL_METHODS:
        iv = np.array([r["result"]["by_threshold"][threshold]["interval"][m] for r in records], dtype=float)
        have = np.all(np.isfinite(iv), axis=1)
        cover = have & (iv[:, 0] <= p) & (p <= iv[:, 1])
        width = np.log(np.clip(iv[have, 1], 1e-6, 1) / np.clip(iv[have, 0], 1e-6, 1))
        out["interval"][m] = {"coverage": float(cover.mean()), "missing": float(1 - have.mean()),
                              "median_log_width": float(np.median(width)) if width.size else None,
                              "share_wider_than_100x": float(np.mean(width > np.log(100))) if width.size else None}
    if "jump_frequency_interval" in records[0]["result"]:
        lam = np.array([r["truth_x"][0] for r in records])
        out["jump_frequency"] = {}
        for m in FREQUENCY_METHODS:
            iv = np.array([r["result"]["jump_frequency_interval"][m] for r in records], dtype=float)
            have = ~np.isnan(iv).any(axis=1)
            cover = have & (iv[:, 0] <= lam) & (lam <= iv[:, 1])
            out["jump_frequency"][m] = {"coverage": float(cover.mean()), "missing": float(1 - have.mean()),
                                        "median_width_factor": float(np.exp(np.median(np.diff(iv[have], axis=1))))
                                        if have.any() else None}
        out["mle_on_bound"] = float(np.mean([r["result"]["mle_on_bound"] for r in records]))
        out["mle_hessian_not_pd"] = float(np.mean([not r["result"]["mle_hessian_ok"] for r in records]))
        out["profile_interval_open"] = float(np.mean([r["result"]["profile_parameter_open"] for r in records]))
    return out
