"""The deck's running example: Heston on synthetic option quotes, with the model true by construction.

One parameter set generates 35 out-of-the-money prices: 7 strikes on the tutorial's standardized-moneyness grid at
each of 1, 3 and 6 months and 1 and 2 years. Each price is observed once, with the tutorial's error band (0.3
volatility points at the money, 1 point at |delta| = 2). A *design* keeps the quotes up to a last maturity, so the
same surface can be read with quotes that reach 2 years or stop at 1 month. For a design this module computes

- the classical route: bounded least squares from several starts, its Gauss-Newton covariance, and the delta-method
  range of a product. Where the optimum sits on a bound of the parameter box, no interval is reported;
- the posterior under a uniform prior on the application's parameter box, by importance sampling. The proposal
  mixes the prior with a few multivariate t components in logit coordinates, placed along the posterior's longest
  direction and fitted to a flattened posterior, so that a curved valley and its tails are covered. The draws are
  resampled to equal weights, and the effective sample size and the share of failed pricings are reported;
- two products under any parameter set: a 2-year put struck at 90% of the forward, and a call that starts at the
  money in 1 year and pays at 2 years (closed form up to one Fourier integral).

Prices are the tutorial runtime's Fourier prices (`sbi4f_tutorial.shared.pricing.heston_surface`). Everything is
deterministic given SEED. These are illustrations for slides, not results of the tutorial: the model is right by
construction, the quote errors are independent, and one simulated surface is shown.

The sampler is importance sampling, not Markov chain Monte Carlo, so that the figures need only the tutorial's own
dependencies. Each of the five posteriors was compared once with four long PyMC DEMetropolisZ chains (R-hat at most
1.004); the slides' README gives the agreement.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, gammaln, logit, logsumexp, roots_legendre

from sbi4f_tutorial.heston_options import model as heston
from sbi4f_tutorial.shared import pricing, surface

NAMES = ("v0", "theta", "kappa", "vol_of_vol", "rho")
TRUTH = np.array([0.02, 0.05, 2.0, 0.9, -0.65])
# The uniform prior: the box of the option application (variances 10% to 40% in volatility).
LOWER = np.array([0.01, 0.01, 0.1, 0.1, -0.95])
UPPER = np.array([0.16, 0.16, 10.0, 3.0, -0.10])
DAYS = (21, 63, 126, 252, 504)
REACH = ("1 month", "3 months", "6 months", "1 year", "2 years")
DELTAS = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0)
SEED = 20261114  # the day of the tutorial; chosen before looking at any figure
NODES, CUTOFF = 1024, 600.0  # Gauss-Legendre nodes and frequency cutoff: the rung the Heston lesson uses
PUT_STRIKE, PUT_MATURITY = 0.9, 2.0
START, END = 1.0, 2.0  # the forward-start call


def heston_parameters(theta):
    """The runtime's parameter dictionary and initial variance for (v0, theta, kappa, vol of vol, rho)."""
    v0, long_run, kappa, vol_of_vol, rho = (float(x) for x in theta)
    return {"kappa": kappa, "theta": long_run, "vol_of_vol": vol_of_vol, "rho": rho, "jump_intensity": 0.0,
            "jump_up_probability": 0.5, "jump_up_rate": 10.0, "jump_down_rate": 10.0}, v0


def surface_prices(theta, contracts, *, nodes=NODES, cutoff=CUTOFF):
    parameters, v0 = heston_parameters(theta)
    return pricing.heston_surface(parameters, v0, contracts, nodes=nodes, upper_limit=cutoff)


def put_price(theta, strike=PUT_STRIKE, maturity=PUT_MATURITY):
    """A European put in units of the forward, by the runtime's Fourier integral."""
    parameters, v0 = heston_parameters(theta)
    return float(heston.european_price(1.0, strike, 1.0, maturity, parameters, v0, kind="put",
                                       nodes=NODES, upper_limit=CUTOFF))


_ROOTS, _WEIGHTS = roots_legendre(NODES)
_U = 0.5 * CUTOFF * (_ROOTS + 1.0)
_DU = 0.5 * CUTOFF * _WEIGHTS


def _exponents(u, maturity, kappa, long_run, vol_of_vol, rho):
    """A and B with E[exp(iu log(X_T / X_0)) | v_0] = exp(A + B v_0): the runtime's characteristic function, split."""
    iu = 1j * u
    b = kappa - rho * vol_of_vol * iu
    d = np.sqrt(b * b + vol_of_vol**2 * (u * u + iu))
    a = -(u * u + iu) / (b + d)
    g = vol_of_vol**2 * a / (b + d)
    decay = np.exp(-d * maturity)
    log_ratio = np.log1p(-g * decay) - np.log1p(-g)
    return (kappa * long_run * (a * maturity - 2.0 * log_ratio / vol_of_vol**2),
            a * (-np.expm1(-d * maturity)) / (1.0 - g * decay))


def forward_start_call(theta, start=START, end=END):
    """E[(X_end - X_start)^+] in units of the forward: a call struck at the money on the start date.

    With the price X as numeraire up to the start date, the variance is again a square-root process, with mean
    reversion kappa - rho * vol_of_vol. The return from start to end depends on the past only through the variance at
    the start, and its characteristic function is exp(A + B v_start). So the price is one Lewis integral of exp(A)
    times the moment generating function of v_start at B (Lucic 2003; Kruse and Noegel 2005). Checked against the
    vanilla call as start -> 0, against Black-Scholes as vol_of_vol -> 0, and against a Monte Carlo simulation.
    """
    v0, long_run, kappa, vol_of_vol, rho = (float(x) for x in theta)
    a, b = _exponents(_U - 0.5j, end - start, kappa, long_run, vol_of_vol, rho)
    kappa_share = kappa - rho * vol_of_vol
    scale = vol_of_vol**2 * (-np.expm1(-kappa_share * start)) / (4.0 * kappa_share)
    denominator = 1.0 - 2.0 * b * scale
    moment = np.exp(-(2.0 * kappa * long_run / vol_of_vol**2) * np.log(denominator)
                    + b * np.exp(-kappa_share * start) * v0 / denominator)
    integrand = np.real(np.exp(a) * moment) / (_U * _U + 0.25)
    return float(1.0 - np.sum(integrand * _DU) / np.pi)


def smile(theta, maturity, strikes):
    """Implied volatilities at one maturity, from out-of-the-money prices (puts up to the forward, calls above)."""
    strikes = np.asarray(strikes, dtype=float)
    contracts = [{"strike": float(strike), "maturity": float(maturity), "option_type": "put" if strike <= 1 else "call"}
                 for strike in strikes]
    kind = np.array([c["option_type"] for c in contracts])
    return surface.implied_vol(surface_prices(theta, contracts), np.log(strikes), maturity, kind)


@dataclass(frozen=True)
class Design:
    """The quotes up to a last maturity: contracts, their error band in price, and one observed surface."""

    contracts: tuple
    k: np.ndarray
    maturity: np.ndarray
    kind: np.ndarray
    noise: np.ndarray
    observed: np.ndarray
    reach: str

    def residuals(self, theta):
        """(observed - model) / error band; a failed pricing counts as a very large misfit."""
        try:
            return (self.observed - surface_prices(theta, self.contracts)) / self.noise
        except ValueError:
            return np.full(len(self.contracts), 1e3)

    def log_likelihood(self, theta):
        r = self.residuals(theta)
        return -0.5 * float(r @ r)

    def implied_quotes(self):
        """The observed prices as implied volatilities, with the band from price -/+ one error."""
        mid = surface.implied_vol(self.observed, self.k, self.maturity, self.kind)
        low = surface.implied_vol(np.maximum(self.observed - self.noise, 1e-12), self.k, self.maturity, self.kind)
        high = surface.implied_vol(self.observed + self.noise, self.k, self.maturity, self.kind)
        return mid, low, high


def error_band(k, maturity, kind, prices):
    """Each price's error, one standard deviation: the tutorial's band in volatility times the quote's own vega."""
    implied = surface.implied_vol(prices, k, maturity, kind)
    delta = surface.standardized_moneyness(k, maturity)
    return surface.bs_vega(k, maturity, implied) * surface.implied_vol_noise(delta)


def designs(seed=SEED):
    """The five nested designs of one simulated surface, keyed by their reach ('1 month' ... '2 years')."""
    contracts = surface.design_grid(days=DAYS, deltas=DELTAS)
    k, maturity, kind = surface.contract_arrays(contracts)
    clean = surface_prices(TRUTH, contracts)
    noise = error_band(k, maturity, kind, clean)
    refined = surface_prices(TRUTH, contracts, nodes=4 * NODES, cutoff=4 * CUTOFF)
    if np.max(np.abs(clean - refined) / noise) > 1e-3:
        raise ValueError("the Fourier integral is not converged to a thousandth of the error band")
    observed = clean + noise * np.random.default_rng(seed).standard_normal(len(contracts))
    out = {}
    for j, reach in enumerate(REACH):
        n = (j + 1) * len(DELTAS)
        out[reach] = Design(tuple(contracts[:n]), k[:n], maturity[:n], kind[:n], noise[:n], observed[:n], reach)
    return out


# ---------------------------------------------------------------------------------------------------------------
# The classical route: least squares, Gauss-Newton, the delta method


@dataclass(frozen=True)
class Fit:
    theta: np.ndarray
    chi2: float
    free: tuple  # indices of the fitted parameters
    covariance: np.ndarray  # of the free parameters; None when the optimum is on a bound
    on_bound: tuple  # names of the parameters that sit on a bound of the box


def _jacobian(design, theta, free, step=1e-4):
    columns = []
    for i in free:
        h = step * (UPPER[i] - LOWER[i])
        up, down = theta.copy(), theta.copy()
        up[i], down[i] = min(theta[i] + h, UPPER[i]), max(theta[i] - h, LOWER[i])
        columns.append((design.residuals(down) - design.residuals(up)) / (up[i] - down[i]))
    return np.column_stack(columns)  # d(model / noise) / d theta


def fit(design, fixed=None, starts=8, seed=0):
    """Bounded least squares from `starts` points in the box; `fixed` maps a parameter index to its value."""
    fixed = fixed or {}
    free = tuple(i for i in range(len(NAMES)) if i not in fixed)

    def full(x):
        theta = np.empty(len(NAMES))
        theta[list(free)] = x
        for i, value in fixed.items():
            theta[i] = value
        return theta

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(starts):
        start = (LOWER + (UPPER - LOWER) * rng.uniform(0.1, 0.9, len(NAMES)))[list(free)]
        result = least_squares(lambda x: design.residuals(full(x)), start,
                               bounds=(LOWER[list(free)], UPPER[list(free)]), x_scale=(UPPER - LOWER)[list(free)],
                               xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or result.cost < best.cost:
            best = result
    theta = full(best.x)
    on_bound = tuple(NAMES[i] for i in free
                     if min(theta[i] - LOWER[i], UPPER[i] - theta[i]) < 1e-6 * (UPPER[i] - LOWER[i]))
    covariance = None
    if not on_bound:
        j = _jacobian(design, theta, free)
        covariance = np.linalg.inv(j.T @ j)
    return Fit(theta, 2.0 * best.cost, free, covariance, on_bound)


def delta_range(result, product, level=0.90, step=1e-4):
    """The product at the fit and its symmetric range by the delta method; None where no covariance exists."""
    point = product(result.theta)
    if result.covariance is None:
        return point, None
    gradient = []
    for i in result.free:
        h = step * (UPPER[i] - LOWER[i])
        up, down = result.theta.copy(), result.theta.copy()
        up[i] += h
        down[i] -= h
        gradient.append((product(up) - product(down)) / (2 * h))
    gradient = np.array(gradient)
    half = {0.90: 1.6448536269514722}[level] * float(np.sqrt(gradient @ result.covariance @ gradient))
    return point, (point - half, point + half)


def gauss_newton_ellipse(result, i, j, level=0.90, points=200):
    """The Gauss-Newton `level` ellipse of two free parameters, as an array of (theta_i, theta_j)."""
    rows = [result.free.index(i), result.free.index(j)]
    cov = result.covariance[np.ix_(rows, rows)]
    values, vectors = np.linalg.eigh(cov)
    radius = np.sqrt(-2.0 * np.log(1.0 - level))  # chi-square with two degrees of freedom
    angle = np.linspace(0, 2 * np.pi, points)
    circle = np.vstack([np.cos(angle), np.sin(angle)])
    return (result.theta[[i, j]][:, None] + radius * vectors @ (np.sqrt(values)[:, None] * circle)).T


# ---------------------------------------------------------------------------------------------------------------
# The posterior: importance sampling in logit coordinates


@dataclass(frozen=True)
class Posterior:
    draws: np.ndarray  # equal-weight parameter draws, shape (keep, 5)
    effective_sample_size: float  # of the weighted proposals the draws were resampled from
    proposals: int  # number of those proposals
    failed: float  # share of them whose pricing failed (counted as zero likelihood)


def _to_theta(u):
    return LOWER + (UPPER - LOWER) * expit(u)


def _log_prior(u):
    """The uniform prior on the box, as a density of the logit coordinates (a product of logistic densities)."""
    e = expit(u)
    with np.errstate(divide="ignore"):
        return np.sum(np.log(e * (1.0 - e)), axis=-1)


def _log_t(u, mean, chol, dof):
    d = u.shape[-1]
    z = np.linalg.solve(chol, (u - mean).T).T
    return (gammaln(0.5 * (dof + d)) - gammaln(0.5 * dof) - 0.5 * d * np.log(dof * np.pi)
            - np.sum(np.log(np.diag(chol))) - 0.5 * (dof + d) * np.log1p(np.sum(z * z, axis=-1) / dof))


def _components(u, w, pieces, inflate):
    """Slices of the weighted proposals along their principal axis: an (equal share, mean, Cholesky factor) each."""
    centred = u - w @ u
    axis = np.linalg.eigh((centred * w[:, None]).T @ centred)[1][:, -1]
    order = np.argsort(centred @ axis)
    cumulative = np.cumsum(w[order])
    found = []
    for k in range(pieces):
        chosen = order[(cumulative > k / pieces) & (cumulative <= (k + 1) / pieces + 1e-12)]
        if len(chosen) < 2 * u.shape[1]:
            continue
        share = w[chosen] / w[chosen].sum()
        mean = share @ u[chosen]
        spread = u[chosen] - mean
        cov = inflate * (spread * share[:, None]).T @ spread + 1e-6 * np.eye(u.shape[1])
        found.append((mean, np.linalg.cholesky(cov)))
    return [(1.0 / len(found), mean, chol) for mean, chol in found]


def posterior(design, *, adapt=(6000, 6000, 6000), temper=0.5, pieces=4, batch=8000, target=6000.0, most=120000,
              keep=4000, prior_share=0.2, dof=5.0, inflate=2.0, seed=1):
    """Equal-weight draws from p(theta | quotes) under the uniform prior on the box, by importance sampling.

    The proposal is a mixture in logit coordinates: `prior_share` from the prior, the rest from multivariate t
    components. It starts as one t at the least-squares fit with its Gauss-Newton shape. Each `adapt` round weights
    its proposals by prior times likelihood to the power `temper` (a flattened posterior, which reaches further along
    a valley than the posterior itself), cuts them into `pieces` slices along their principal axis, and puts one t
    on each slice with an inflated covariance. With the proposal then frozen, batches are drawn until the effective
    sample size reaches `target` or `most` proposals are used. The final weights are prior times likelihood over the
    mixture density; `keep` draws are resampled systematically from them.
    """
    rng = np.random.default_rng(seed)
    dim = len(NAMES)
    start = fit(design)
    inside = np.clip(start.theta, LOWER + 1e-3 * (UPPER - LOWER), UPPER - 1e-3 * (UPPER - LOWER))
    mean = logit((inside - LOWER) / (UPPER - LOWER))
    e = expit(mean)
    j = _jacobian(design, inside, tuple(range(dim))) * ((UPPER - LOWER) * e * (1.0 - e))
    components = [(1.0, mean, np.linalg.cholesky(inflate * np.linalg.inv(j.T @ j + 0.05 * np.eye(dim))))]

    def propose(n):
        shares = [prior_share] + [(1.0 - prior_share) * share for share, _, _ in components]
        source = rng.choice(len(shares), size=n, p=shares)
        u = logit(rng.uniform(1e-12, 1 - 1e-12, (n, dim)))  # the rows drawn from the prior keep these
        z = rng.standard_normal((n, dim)) / np.sqrt(rng.chisquare(dof, n) / dof)[:, None]
        for k, (_, centre, chol) in enumerate(components):
            rows = source == k + 1
            u[rows] = centre + z[rows] @ chol.T
        log_prior = _log_prior(u)
        parts = [np.log(shares[0]) + log_prior]
        parts += [np.log(share) + _log_t(u, centre, chol, dof)
                  for share, (_, centre, chol) in zip(shares[1:], components, strict=True)]
        theta = _to_theta(u)
        log_like = np.array([design.log_likelihood(t) for t in theta])
        return u, theta, log_like, log_prior - logsumexp(np.stack(parts), axis=0)

    def normalized(log_w):
        log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
        w = np.exp(log_w - log_w.max())
        return w / w.sum()

    for n in adapt:
        u, _, log_like, log_ratio = propose(n)
        components = _components(u, normalized(temper * log_like + log_ratio), pieces, inflate)
    thetas, log_ws, log_likes = [], [], []
    while True:
        _, theta, log_like, log_ratio = propose(batch)
        thetas.append(theta)
        log_ws.append(log_like + log_ratio)
        log_likes.append(log_like)
        w = normalized(np.concatenate(log_ws))
        ess = 1.0 / np.sum(w * w)
        if ess >= target or len(w) >= most:
            break
    theta = np.concatenate(thetas)
    positions = (rng.uniform() + np.arange(keep)) / keep
    chosen = np.minimum(np.searchsorted(np.cumsum(w), positions), len(w) - 1)
    failed = np.mean(np.concatenate(log_likes) <= -0.49e6 * len(design.contracts))
    return Posterior(theta[rng.permutation(chosen)], float(ess), int(len(w)), float(failed))


def within(draws, lower=None, upper=None):
    """The draws inside a smaller box: the posterior under a uniform prior on that box."""
    lower = LOWER if lower is None else np.asarray(lower, dtype=float)
    upper = UPPER if upper is None else np.asarray(upper, dtype=float)
    return draws[np.all((draws >= lower) & (draws <= upper), axis=1)]
