"""The slot 1 deck's running example: a thin option chain simulated from Heston, the model right by construction.

The example keeps the Heston SPX notebook's model, prior, error law and "market-like" parameters
(`lab/notebooks/heston_spx_calibration.py`), and changes one thing: the chain. The notebook's 63 SPX quotes are so
many and so tight that they pin every Heston parameter, and nothing is left to see. The chain here is thinner and
quoted less tightly, as for a single stock, so that what the quotes leave open is visible:

- 35 out-of-the-money options: 7 strikes at standardized moneyness delta = ln(K/F) / (0.16 sqrt T) from -3.5 to 1
  (the notebook's nine targets without its two extremes) at each of 1, 3 and 6 months and 1 and 2 years;
- half-spreads of 0.3 volatility points at the money, rising by 0.35 points per unit of |delta| (1.5 points at
  delta = -3.5), converted to prices with each quote's own vega: about ten times the notebook's SPX half-spreads;
- each quote observed once with the notebook's error law

      y_i = P_i(theta) + c h_i Z_i,    Z_i independent standard normals,    c = 1,

  h_i the quote's half-spread.

A *reading* keeps the quotes up to a last expiry, so the same surface can be read with quotes that reach 2 years or
stop at 1 month. For a reading this module computes

- the classical route: least squares in half-spreads from several starts; the error scale by its unbiased estimate,
  c_hat^2 = R / (n - p), with R the sum of squared misses in half-spreads, n quotes and p fitted parameters; the
  Gauss-Newton covariance c_hat^2 (J'J)^-1; and Student-t intervals, for a parameter or, by the delta method, for a
  product. Where the optimum sits on a bound of the parameter box, no interval is reported;
- the posterior under the notebook's priors: uniform on the Heston box, and c uniform on [0.5, 2], inferred. c is
  integrated out of the likelihood in closed form (an incomplete gamma function), the five Heston parameters are
  sampled by importance sampling followed by Metropolis moves, and each draw then receives a c from its exact
  conditional;
- two products under any parameter set: a 2-year put struck at 90% of the forward, and a call that is struck at the
  money in 1 year and pays at 2 years (closed form up to one Fourier integral).

`notebook_case` returns the notebook's own simulated case on its 63 quotes, for one check that the deck quotes: on
those quotes the classical intervals reproduce the notebook's exact posterior.

Prices are Lewis Fourier integrals with an analytic Black control variate, for many parameter sets at once. The
formulas are the tutorial runtime's (`sbi4f_tutorial.heston_options.model`); the tests compare the two. Everything is
deterministic given SEED. These are illustrations for slides, not results of the tutorial: the model is right by
construction, the quote errors are independent, and one simulated surface is shown.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import least_squares
from scipy.special import expit, gammainc, gammaincc, gammainccinv, gammaincinv, gammaln, logit, logsumexp, ndtr
from scipy.special import roots_legendre

from sbi4f_tutorial.shared import surface

HERE = Path(__file__).resolve().parent
DESIGN_FILE = HERE / "spx-design.json"
CASE_FILE = HERE / "market-like-case.json"

NAMES = ("v0", "kappa", "long_run", "vol_of_vol", "rho")  # the order of the Heston notebook
# The notebook's case "market-like parameters" (generating c = 1): market-like-case.json holds the same numbers.
TRUTH = np.array([0.027111476938590673, 4.33564875507931, 0.056477632243131036, 1.7993356588948335,
                  -0.6674551198176153])
C_TRUE = 1.0
# The notebook's uniform priors.
LOWER = np.array([0.01, 0.1, 0.01, 0.1, -0.95])
UPPER = np.array([0.16, 10.0, 0.16, 3.0, -0.10])
C_LOWER, C_UPPER = 0.5, 2.0

REFERENCE_VOLATILITY = 0.16  # delta = ln(K/F) / (0.16 sqrt T), as in the notebook
DELTAS = (-3.5, -2.25, -1.25, -0.5, 0.0, 0.5, 1.0)
MONTHS = (1, 3, 6, 12, 24)
REACH = ("1 month", "3 months", "6 months", "1 year", "2 years")
HALF_SPREAD_AT_THE_MONEY, HALF_SPREAD_SLOPE = 0.003, 0.0035  # in volatility: 0.3 points, plus 0.35 per unit of |delta|
SEED = 20261114  # the day of the tutorial; chosen before looking at any figure
RUNG, CHECK_RUNG = (1024, 600.0), (2048, 1200.0)  # Gauss-Legendre nodes and cutoff; the gate compares the two
NOTEBOOK_RUNG, NOTEBOOK_CHECK_RUNG = (2048, 1200.0), (4096, 2400.0)  # for the notebook's much tighter half-spreads
GATE = 0.01  # refinement moves no price by more than 0.01 of its half-spread
PUT_STRIKE, PUT_MATURITY = 0.9, 2.0
START, END = 1.0, 2.0  # the forward-start call


# ---------------------------------------------------------------------------------------------------------------
# Prices


_QUADRATURE = {}


def _quadrature(rung):
    if rung not in _QUADRATURE:
        roots, weights = roots_legendre(int(rung[0]))
        _QUADRATURE[rung] = (0.5 * rung[1] * (roots + 1.0), 0.5 * rung[1] * weights)
    return _QUADRATURE[rung]


def _exponents(z, times, thetas):
    """A and B with E[exp(i z ln(F_T / F_0))] = exp(A + B v_0), for each parameter set, time and frequency.

    The decaying-exponential form of the tutorial runtime's Heston characteristic function, split into the part that
    multiplies v_0 and the rest. `thetas` has shape (m, 5); the result has shape (m, len(times), len(z)).
    """
    kappa, long_run, xi, rho = (thetas[:, j][:, None, None] for j in (1, 2, 3, 4))
    z = np.asarray(z)[None, None, :]
    t = np.asarray(times, dtype=float)[None, :, None]
    iz = 1j * z
    b = kappa - rho * xi * iz
    d = np.sqrt(b * b + xi * xi * (z * z + iz))
    a = -(z * z + iz) / (b + d)
    g = xi * xi * a / (b + d)
    decay = np.exp(-d * t)
    log_ratio = np.log1p(-g * decay) - np.log1p(-g)
    return kappa * long_run * (a * t - 2.0 * log_ratio / (xi * xi)), a * (-np.expm1(-d * t)) / (1.0 - g * decay)


def expected_variance(thetas, times):
    """E[integral of v from 0 to T] under Heston, shape (m, len(times)): the total variance of the Black control."""
    v0, kappa, long_run = thetas[:, 0:1], thetas[:, 1:2], thetas[:, 2:3]
    times = np.asarray(times, dtype=float)[None, :]
    return long_run * times + (v0 - long_run) * (-np.expm1(-kappa * times)) / kappa


def _black(strike, variance, put):
    """Black prices in forward units for total variance `variance`."""
    root = np.sqrt(variance)
    d1 = (-np.log(strike) + 0.5 * variance) / root
    d2 = d1 - root
    return np.where(put, strike * ndtr(-d2) - ndtr(-d1), ndtr(d1) - strike * ndtr(d2))


def prices(theta, strike, maturity, put, rung=RUNG, chunk=512):
    """European prices in forward units (forward 1, discount 1), for one parameter set or for many.

    price = Black(w) + sqrt(K)/pi * integral of Re[exp(-iu ln K) (phi_Black - phi_Heston)(u - i/2)] / (u^2 + 1/4),
    with w the expected total variance (Lewis's integral with a Black control variate). `theta` has shape (5,) or
    (m, 5), and the result (n,) or (m, n). A parameter set whose prices are not finite or leave the no-arbitrage
    bounds gets a row of NaN.
    """
    single = np.ndim(theta) == 1
    thetas = np.atleast_2d(np.asarray(theta, dtype=float))
    strike = np.atleast_1d(np.asarray(strike, dtype=float))
    maturity = np.broadcast_to(np.asarray(maturity, dtype=float), strike.shape)
    put = np.broadcast_to(np.asarray(put, dtype=bool), strike.shape)
    u, weights = _quadrature(rung)
    times, inverse = np.unique(maturity, return_inverse=True)
    kernel = weights / (u * u + 0.25)
    basis = [(np.exp(-1j * u[:, None] * np.log(strike[inverse == j])[None, :]) * kernel[:, None])
             for j in range(len(times))]  # each (nodes, quotes of that expiry)
    lower = np.maximum(np.where(put, strike - 1.0, 1.0 - strike), 0.0)
    upper = np.where(put, strike, 1.0)
    out = np.empty((len(thetas), len(strike)))
    with np.errstate(all="ignore"):
        for first in range(0, len(thetas), chunk):
            block = thetas[first:first + chunk]
            a, b = _exponents(u - 0.5j, times, block)
            variance = expected_variance(block, times)
            difference = (np.exp(-0.5 * variance[:, :, None] * (u * u + 0.25))
                          - np.exp(a + b * block[:, 0][:, None, None]))
            values = _black(strike, variance[:, inverse], put)
            for j in range(len(times)):
                columns = np.flatnonzero(inverse == j)
                values[:, columns] += np.sqrt(strike[columns]) / np.pi * np.real(difference[:, j, :] @ basis[j])
            bad = ~np.isfinite(values).all(axis=1) | np.any(values < lower - 1e-12, axis=1) \
                | np.any(values > upper + 1e-12, axis=1)
            values[bad] = np.nan
            out[first:first + chunk] = values
    return out[0] if single else out


def smile(theta, maturity, strikes):
    """Implied volatilities at one maturity from out-of-the-money prices (puts up to the forward, calls above)."""
    strikes = np.asarray(strikes, dtype=float)
    put = strikes <= 1.0
    values = prices(theta, strikes, maturity, put, CHECK_RUNG)
    return surface.implied_vol(values, np.log(strikes), maturity, np.where(put, "put", "call"))


# ---------------------------------------------------------------------------------------------------------------
# Products


def put_price(theta, strike=PUT_STRIKE, maturity=PUT_MATURITY):
    """A European put in units of the forward; one value per parameter set."""
    return prices(theta, [strike], [maturity], [True], CHECK_RUNG)[..., 0]


def forward_start_call(theta, start=START, end=END):
    """E[(X_end - X_start)^+] in units of the forward: a call struck at the money on the start date.

    With the price X as numeraire up to the start date, the variance is again a square-root process, with mean
    reversion kappa - rho * vol_of_vol. The return from start to end depends on the past only through the variance at
    the start, and its characteristic function is exp(A + B v_start). So the price is one Lewis integral of exp(A)
    times the moment generating function of v_start at B (Lucic 2003; Kruse and Noegel 2005). The tests check it
    against the vanilla call as start -> 0 and against Black-Scholes as vol_of_vol -> 0. One value per parameter set.
    """
    single = np.ndim(theta) == 1
    thetas = np.atleast_2d(np.asarray(theta, dtype=float))
    u, weights = _quadrature(CHECK_RUNG)
    out = np.empty(len(thetas))
    for first in range(0, len(thetas), 512):
        block = thetas[first:first + 512]
        v0, kappa, long_run, xi, rho = (block[:, j][:, None] for j in range(5))
        a, b = (x[:, 0, :] for x in _exponents(u - 0.5j, [end - start], block))
        kappa_share = kappa - rho * xi
        scale = xi * xi * (-np.expm1(-kappa_share * start)) / (4.0 * kappa_share)
        denominator = 1.0 - 2.0 * b * scale
        moment = np.exp(-(2.0 * kappa * long_run / (xi * xi)) * np.log(denominator)
                        + b * np.exp(-kappa_share * start) * v0 / denominator)
        out[first:first + 512] = 1.0 - np.real(np.exp(a) * moment) @ (weights / (u * u + 0.25)) / np.pi
    return float(out[0]) if single else out


PRODUCTS = {"put": put_price, "forward_start": forward_start_call}


# ---------------------------------------------------------------------------------------------------------------
# Quotes: the thin chain, its readings, and the notebook's case


@dataclass(frozen=True)
class Quotes:
    """A set of quotes in forward units: contracts, half-spreads and one observed price each."""

    strike: np.ndarray
    maturity: np.ndarray
    put: np.ndarray
    half_spread: np.ndarray
    observed: np.ndarray
    label: str
    rung: tuple = RUNG

    @property
    def n(self):
        return len(self.strike)

    def keep(self, n, label):
        """The first n quotes, as a reading of the same surface."""
        return Quotes(self.strike[:n], self.maturity[:n], self.put[:n], self.half_spread[:n], self.observed[:n],
                      label, self.rung)

    def model(self, theta, rung=None):
        return prices(theta, self.strike, self.maturity, self.put, rung or self.rung)

    def misses(self, theta):
        """(observed - model) / half-spread; a failed pricing counts as a very large miss."""
        r = (self.observed - self.model(theta)) / self.half_spread
        return np.where(np.isfinite(r), r, 1e4)

    def sum_of_squares(self, theta):
        """R: the sum of squared misses in half-spreads; one value per parameter set."""
        r = self.misses(theta)
        return np.sum(r * r, axis=-1)

    def gate(self, theta, check_rung):
        """The largest change of a quote's price from this rung to `check_rung`, in half-spreads."""
        return float(np.max(np.abs(self.model(theta, check_rung) - self.model(theta)) / self.half_spread))

    def implied(self):
        """The observed prices as implied volatilities, with the band from price -/+ one half-spread."""
        k, kind = np.log(self.strike), np.where(self.put, "put", "call")
        mid = surface.implied_vol(self.observed, k, self.maturity, kind)
        low = surface.implied_vol(np.maximum(self.observed - self.half_spread, 1e-14), k, self.maturity, kind)
        high = surface.implied_vol(self.observed + self.half_spread, k, self.maturity, kind)
        return mid, low, high


def half_spread_volatility(delta):
    """The chain's half-spread in volatility: 0.3 points at the money, plus 0.35 points per unit of |delta|."""
    return HALF_SPREAD_AT_THE_MONEY + HALF_SPREAD_SLOPE * np.abs(np.asarray(delta, dtype=float))


def thin_chain(seed=SEED):
    """The 35 quotes of the thin chain, observed once at the generating parameters with c = 1."""
    maturity = np.repeat(np.array(MONTHS) / 12.0, len(DELTAS))
    delta = np.tile(np.array(DELTAS), len(MONTHS))
    strike = np.exp(delta * REFERENCE_VOLATILITY * np.sqrt(maturity))
    put = strike <= 1.0
    clean = prices(TRUTH, strike, maturity, put, CHECK_RUNG)
    implied = surface.implied_vol(clean, np.log(strike), maturity, np.where(put, "put", "call"))
    half_spread = surface.bs_vega(np.log(strike), maturity, implied) * half_spread_volatility(delta)
    noise = np.random.default_rng(seed).standard_normal(len(strike))
    chain = Quotes(strike, maturity, put, half_spread, clean + C_TRUE * half_spread * noise, REACH[-1])
    if chain.gate(TRUTH, CHECK_RUNG) > GATE:
        raise ValueError("the Fourier integral is not converged to the gate at the generating parameters")
    return chain


def readings(seed=SEED):
    """The five nested readings of the thin chain, keyed by their reach ('1 month' ... '2 years')."""
    chain = thin_chain(seed)
    return {reach: chain.keep((j + 1) * len(DELTAS), reach) for j, reach in enumerate(REACH)}


def notebook_case():
    """The Heston notebook's simulated case "market-like parameters" on its 63 SPX quotes, in forward units.

    The strikes, maturities, forwards, discounts and half-spreads are those of the 63 SPX quotes of 9 November 2026
    (`spx-design.json`); the observed prices are the notebook's (`market-like-case.json`), generated at TRUTH with
    c = 1. Raises if this module's prices do not reproduce the case's clean prices to the gate.
    """
    design = json.loads(DESIGN_FILE.read_text())["quotes"]
    case = json.loads(CASE_FILE.read_text())
    if case["parameters"] != list(NAMES) or not np.array_equal(case["truth"], TRUTH) or case["generating_c"] != C_TRUE:
        raise ValueError("market-like-case.json does not hold the generating parameters of this module")
    forward = np.array([q["forward"] for q in design])
    points = np.array([q["discount"] for q in design]) * forward  # a price in forward units times this: index points
    quotes = Quotes(np.array([q["strike"] for q in design]) / forward, np.array([q["maturity"] for q in design]),
                    np.array([q["option_type"] == "put" for q in design]),
                    np.array([q["half_spread"] for q in design]) / points, np.asarray(case["observations"]) / points,
                    "the notebook's 63 quotes", NOTEBOOK_RUNG)
    generated = quotes.model(TRUTH) + C_TRUE * quotes.half_spread * np.asarray(case["noise_draws"])
    if max(np.max(np.abs(quotes.observed - generated) / quotes.half_spread),
           quotes.gate(TRUTH, NOTEBOOK_CHECK_RUNG)) > GATE:
        raise ValueError("this module's prices differ from the notebook's clean prices by more than the gate")
    return quotes


# ---------------------------------------------------------------------------------------------------------------
# The classical route: least squares, the error scale, Gauss-Newton, Student-t intervals, the delta method


@dataclass(frozen=True)
class Fit:
    theta: np.ndarray
    sum_of_squares: float  # R: the sum of squared misses in half-spreads
    scale: float  # c_hat = sqrt(R / (n - p)): the unbiased estimate of the error scale; nan when n <= p
    dof: int  # n - p
    free: tuple  # indices of the fitted parameters
    covariance: np.ndarray  # of the free parameters, c_hat^2 (J'J)^-1; None when the optimum is on a bound or n <= p
    on_bound: tuple  # names of the parameters that sit on a bound of the box

    def quantile(self, level=0.90):
        """The two-sided Student-t multiplier of a `level` interval."""
        return float(stats.t.ppf(0.5 + 0.5 * level, self.dof))


def _jacobian(quotes, theta, free, step=1e-4):
    columns = []
    for i in free:
        h = step * (UPPER[i] - LOWER[i])
        up, down = theta.copy(), theta.copy()
        up[i], down[i] = min(theta[i] + h, UPPER[i]), max(theta[i] - h, LOWER[i])
        columns.append((quotes.misses(down) - quotes.misses(up)) / (up[i] - down[i]))
    return np.column_stack(columns)  # d(model / half-spread) / d theta


def fit(quotes, fixed=None, starts=8, seed=0):
    """Least squares in half-spreads from `starts` points in the box; `fixed` maps a parameter index to its value."""
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
        result = least_squares(lambda x: quotes.misses(full(x)), start,
                               bounds=(LOWER[list(free)], UPPER[list(free)]), x_scale=(UPPER - LOWER)[list(free)],
                               xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or result.cost < best.cost:
            best = result
    theta = full(best.x)
    r2, dof = 2.0 * best.cost, quotes.n - len(free)
    scale = float(np.sqrt(r2 / dof)) if dof > 0 else float("nan")
    on_bound = tuple(NAMES[i] for i in free
                     if min(theta[i] - LOWER[i], UPPER[i] - theta[i]) < 1e-6 * (UPPER[i] - LOWER[i]))
    covariance = None
    if not on_bound and dof > 0:
        j = _jacobian(quotes, theta, free)
        covariance = scale**2 * np.linalg.inv(j.T @ j)
    return Fit(theta, float(r2), scale, dof, free, covariance, on_bound)


def parameter_intervals(result, level=0.90):
    """The `level` Student-t interval of each free parameter, shape (len(free), 2); None where no covariance exists."""
    if result.covariance is None:
        return None
    half = result.quantile(level) * np.sqrt(np.diag(result.covariance))
    centre = result.theta[list(result.free)]
    return np.column_stack([centre - half, centre + half])


def delta_range(result, product, level=0.90, step=1e-4):
    """The product at the fit and its symmetric range by the delta method; None where no covariance exists."""
    point = float(product(result.theta))
    if result.covariance is None:
        return point, None
    gradient = []
    for i in result.free:
        h = step * (UPPER[i] - LOWER[i])
        up, down = result.theta.copy(), result.theta.copy()
        up[i] += h
        down[i] -= h
        gradient.append((float(product(up)) - float(product(down))) / (2 * h))
    gradient = np.array(gradient)
    half = result.quantile(level) * float(np.sqrt(gradient @ result.covariance @ gradient))
    return point, (point - half, point + half)


def gauss_newton_ellipse(result, i, j, level=0.90, points=200):
    """The joint `level` region of two free parameters under the Gauss-Newton covariance, as (theta_i, theta_j)."""
    rows = [result.free.index(i), result.free.index(j)]
    cov = result.covariance[np.ix_(rows, rows)]
    values, vectors = np.linalg.eigh(cov)
    radius = np.sqrt(2.0 * stats.f.ppf(level, 2, result.dof))  # two parameters, the scale estimated on `dof`
    angle = np.linspace(0, 2 * np.pi, points)
    circle = np.vstack([np.cos(angle), np.sin(angle)])
    return (result.theta[[i, j]][:, None] + radius * vectors @ (np.sqrt(values)[:, None] * circle)).T


# ---------------------------------------------------------------------------------------------------------------
# The posterior: c integrated out, importance sampling in logit coordinates, then Metropolis moves


def log_c_integral(r2, n):
    """log of the integral of c^-n exp(-R / (2 c^2)) over the prior range of c, up to a constant: c integrated out.

    With t = R / (2 c^2) it is (R/2)^((1-n)/2) Gamma(a) [P(a, 2R) - P(a, R/8)] / 2, a = (n - 1)/2, for c in [0.5, 2].
    """
    a = 0.5 * (n - 1)
    r2 = np.atleast_1d(np.asarray(r2, dtype=float))
    lo, hi = r2 / (2.0 * C_UPPER**2), r2 / (2.0 * C_LOWER**2)
    mass = np.where(lo > a, gammaincc(a, lo) - gammaincc(a, hi), gammainc(a, hi) - gammainc(a, lo))
    with np.errstate(divide="ignore"):
        log_mass = np.log(mass)
    # Far in the tail Q(a, lo) underflows: its leading asymptotic term (the weight is negligible there anyway).
    far = ~np.isfinite(log_mass) & (lo > a)
    log_mass[far] = (a - 1.0) * np.log(lo[far]) - lo[far] - gammaln(a)
    return 0.5 * (1 - n) * np.log(r2 / 2.0) + gammaln(a) + log_mass


def draw_c(r2, n, rng):
    """c from its exact conditional given the misses R: t = R/(2c^2) is a gamma(a) variable truncated to the prior."""
    a = 0.5 * (n - 1)
    r2 = np.asarray(r2, dtype=float)
    lo, hi = r2 / (2.0 * C_UPPER**2), r2 / (2.0 * C_LOWER**2)
    u = rng.uniform(size=r2.shape)
    t = np.where(lo > a,
                 gammainccinv(a, gammaincc(a, lo) - u * (gammaincc(a, lo) - gammaincc(a, hi))),
                 gammaincinv(a, gammainc(a, lo) + u * (gammainc(a, hi) - gammainc(a, lo))))
    return np.clip(np.sqrt(r2 / (2.0 * t)), C_LOWER, C_UPPER)


def to_theta(u):
    return LOWER + (UPPER - LOWER) * expit(u)


def log_prior(u):
    """The uniform prior on the box, as a density of the logit coordinates (a product of logistic densities)."""
    e = expit(u)
    with np.errstate(divide="ignore"):
        return np.sum(np.log(e * (1.0 - e)), axis=-1)


def log_posterior(quotes, u):
    """log p(theta | quotes) in logit coordinates, c integrated out, up to a constant; and R for each row of u."""
    r2 = quotes.sum_of_squares(to_theta(u))
    value = log_c_integral(r2, quotes.n) + log_prior(u)
    value[r2 >= 0.99e8] = -np.inf  # failed pricings
    return value, r2


@dataclass(frozen=True)
class Posterior:
    draws: np.ndarray  # equal-weight draws of (v0, kappa, long-run variance, vol of vol, rho, c), shape (keep, 6)
    effective_sample_size: float  # of the weighted proposals the draws were resampled from
    proposals: int  # number of those proposals
    failed: float  # share of them whose pricing failed (counted as zero likelihood)
    acceptance: float  # of the Metropolis moves made after resampling


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


def metropolis_moves(quotes, u, sweeps, rng):
    """Differential-evolution Metropolis sweeps over a population of draws, each leaving the posterior invariant.

    For each draw, the proposal adds gamma times the difference of two other draws of the population as it stood at
    the start of the sweep, plus a small jitter; it is symmetric, so the Metropolis ratio is the ratio of the posterior
    densities in logit coordinates. gamma starts at 2.38/sqrt(2d) and is scaled between sweeps towards an acceptance
    of about a quarter (a thin curved ridge needs shorter steps); in a tenth of the moves it is 1, to cross between
    modes. Returns the moved population, its R and the acceptance rate.
    """
    count, dim = u.shape
    value, r2 = log_posterior(quotes, u)
    accepted, scale = 0, 2.38 / np.sqrt(2 * dim)
    for _ in range(sweeps):
        a = rng.integers(0, count - 1, count)
        a += a >= np.arange(count)
        b = rng.integers(0, count - 2, count)
        b += b >= np.minimum(a, np.arange(count))
        b += b >= np.maximum(a, np.arange(count))
        gamma = np.where(rng.uniform(size=count) < 0.1, 1.0, scale)
        proposal = u + gamma[:, None] * (u[a] - u[b]) + 1e-4 * rng.standard_normal((count, dim))
        new_value, new_r2 = log_posterior(quotes, proposal)
        accept = np.log(rng.uniform(size=count)) < new_value - value
        u, r2, value = np.where(accept[:, None], proposal, u), np.where(accept, new_r2, r2), np.where(accept, new_value,
                                                                                                    value)
        accepted += int(accept.sum())
        rate = accept.mean()
        scale *= 0.8 if rate < 0.15 else 1.2 if rate > 0.35 else 1.0
    return u, r2, accepted / (sweeps * count)


def posterior(quotes, *, adapt=(8000, 8000, 8000, 8000), temper=0.5, pieces=4, batch=20000, target=6000.0,
              most=200000, keep=4000, prior_share=0.2, dof=5.0, inflate=2.0, sweeps=50, seed=1):
    """Equal-weight draws from p(theta, c | quotes) under the notebook's uniform priors.

    c is integrated out of the likelihood in closed form, so the sampler works on the five Heston parameters, in logit
    coordinates of the prior box. Importance sampling: the proposal mixes the prior (`prior_share`) with multivariate
    t components. It starts as one t at the least-squares fit with its Gauss-Newton shape. Each `adapt` round weights
    its proposals by prior times likelihood to the power `temper` (a flattened posterior, which reaches further along
    a valley than the posterior itself), cuts them into `pieces` slices along their principal axis, and puts one t on
    each slice with an inflated covariance. With the proposal then frozen, batches are drawn until the effective
    sample size reaches `target` or `most` proposals are used; `keep` draws are resampled systematically from the
    weights. The resampled draws are then moved by `sweeps` differential-evolution Metropolis sweeps at the posterior
    (resample-move), which repairs what a finite set of weighted proposals misses along a thin ridge. Each draw
    finally gets a c drawn from its exact conditional given that draw's misses.
    """
    rng = np.random.default_rng(seed)
    dim = len(NAMES)
    start = fit(quotes)
    inside = np.clip(start.theta, LOWER + 1e-3 * (UPPER - LOWER), UPPER - 1e-3 * (UPPER - LOWER))
    mean = logit((inside - LOWER) / (UPPER - LOWER))
    e = expit(mean)
    scale = start.scale if np.isfinite(start.scale) else 1.0
    j = _jacobian(quotes, inside, tuple(range(dim))) * ((UPPER - LOWER) * e * (1.0 - e)) / scale
    components = [(1.0, mean, np.linalg.cholesky(inflate * np.linalg.inv(j.T @ j + 0.05 * np.eye(dim))))]

    def propose(n):
        shares = [prior_share] + [(1.0 - prior_share) * share for share, _, _ in components]
        source = rng.choice(len(shares), size=n, p=shares)
        u = logit(rng.uniform(1e-12, 1 - 1e-12, (n, dim)))  # the rows drawn from the prior keep these
        z = rng.standard_normal((n, dim)) / np.sqrt(rng.chisquare(dof, n) / dof)[:, None]
        for k, (_, centre, chol) in enumerate(components):
            rows = source == k + 1
            u[rows] = centre + z[rows] @ chol.T
        prior = log_prior(u)
        parts = [np.log(shares[0]) + prior]
        parts += [np.log(share) + _log_t(u, centre, chol, dof)
                  for share, (_, centre, chol) in zip(shares[1:], components, strict=True)]
        r2 = quotes.sum_of_squares(to_theta(u))
        log_like = log_c_integral(r2, quotes.n)
        log_like[r2 >= 0.99e8] = -np.inf  # failed pricings
        return u, r2, log_like, prior - logsumexp(np.stack(parts), axis=0)

    def normalized(log_w):
        log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
        w = np.exp(log_w - log_w.max())
        return w / w.sum()

    for n in adapt:
        u, _, log_like, log_ratio = propose(n)
        components = _components(u, normalized(temper * log_like + log_ratio), pieces, inflate)
    us, r2s, log_ws = [], [], []
    while True:
        u, r2, log_like, log_ratio = propose(batch)
        us.append(u)
        r2s.append(r2)
        log_ws.append(log_like + log_ratio)
        w = normalized(np.concatenate(log_ws))
        ess = 1.0 / np.sum(w * w)
        if ess >= target or len(w) >= most:
            break
    u, r2 = np.concatenate(us), np.concatenate(r2s)
    failed = np.mean(r2 >= 0.99e8)
    positions = (rng.uniform() + np.arange(keep)) / keep
    chosen = rng.permutation(np.minimum(np.searchsorted(np.cumsum(w), positions), len(w) - 1))
    u, r2, acceptance = metropolis_moves(quotes, u[chosen], sweeps, rng)
    c = draw_c(r2, quotes.n, rng)
    return Posterior(np.column_stack([to_theta(u), c]), float(ess), int(len(w)), float(failed), float(acceptance))
