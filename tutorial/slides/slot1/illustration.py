"""The deck's running example: Heston on simulated SPX quotes, with the model true by construction.

The design is the tutorial's (`spx-design.json`): the strikes, maturities, forwards, discounts and half-spreads of
the 63 actual SPX quotes of 9 November 2026, 9 strikes at standardized moneyness z = ln(K/F)/(0.16 sqrt T) on each
of 7 expiries from 7 to 325 days. One parameter set, the option application's reference case "market-like,
simulated", generates the 63 prices in forward units, and each is observed once with the application's error law

    y_i = P_i(theta) + c h_i Z_i,    Z_i independent standard normals,    c = 1,

h_i the quote's own half-spread. A *reading* keeps the quotes up to a last expiry, so the same surface can be read
with quotes that reach 325 days or stop at 7 days. For a reading this module computes

- the classical route: least squares in half-spreads from several starts, the error scale by maximum likelihood,
  c_hat = sqrt(R/n) with R the sum of squared misses in half-spreads, the Gauss-Newton covariance c_hat^2 (J'J)^-1,
  and the delta-method range of a product. Where the optimum sits on a bound of the parameter box, no interval is
  reported;
- the posterior under the application's priors: uniform on the Heston box, and c uniform on [0.5, 2], inferred. The
  likelihood is the application's, with its -n ln c term. c is integrated out in closed form (an incomplete gamma
  function), the five Heston parameters are sampled by importance sampling, and each draw then receives a c from its
  exact conditional. The proposal mixes the prior with a few multivariate t components in logit coordinates, placed
  along the posterior's longest direction and fitted to a flattened posterior, so that a curved valley and its tails
  are covered. The draws are resampled to equal weights; the effective sample size and the share of failed pricings
  are reported;
- three products under any parameter set: the application's forward-start call, struck at the money in 6 months
  and paid at 1 year (closed form up to one Fourier integral); the forward-starting variance over the application's
  later window, 81 to 325 days (closed form); and the listed 325-day at-the-money call.

Prices are Lewis Fourier integrals with the analytic Black control variate of the option application (its section
4.1), built on the tutorial runtime's Heston characteristic function: the 7-day put at z = -5 is worth about 1e-4 of
the forward and a half-spread about 1e-5, which the plain formula cannot resolve. Everything is deterministic given
SEED. These are illustrations for slides, not results of the tutorial: the model is right by construction, the quote
errors are independent, and one simulated surface is shown.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, gammainc, gammaincc, gammainccinv, gammaincinv, gammaln, logit, logsumexp
from scipy.special import roots_legendre

from sbi4f_tutorial.heston_options import model as heston
from sbi4f_tutorial.shared import gbm, surface

HERE = Path(__file__).resolve().parent
DESIGN_FILE = HERE / "spx-design.json"

NAMES = ("v0", "kappa", "long_run", "vol_of_vol", "rho")  # the order of the option application
# The option application's reference case "market-like, simulated" (generating c = 1): its section 8.1.
TRUTH = np.array([0.027111476938590673, 4.33564875507931, 0.056477632243131036, 1.7993356588948335,
                  -0.6674551198176153])
C_TRUE = 1.0
# The application's uniform priors: its section 4.1.
LOWER = np.array([0.01, 0.1, 0.01, 0.1, -0.95])
UPPER = np.array([0.16, 10.0, 0.16, 3.0, -0.10])
C_LOWER, C_UPPER = 0.5, 2.0
SEED = 20261114  # the day of the tutorial; chosen before looking at any figure
RUNG, CHECK_RUNG = (2048, 1200.0), (4096, 2400.0)  # Gauss-Legendre nodes and cutoff; the gate compares the two
GATE = 0.01  # the application's accuracy gate: refinement moves no price by more than 0.01 of its half-spread
START, END = 0.5, 1.0  # the forward-start call: struck at the money in 6 months, paid at 1 year


def _load_design():
    stored = json.loads(DESIGN_FILE.read_text())
    quotes = stored["quotes"]
    forward = np.array([q["forward"] for q in quotes])
    discount = np.array([q["discount"] for q in quotes])
    expiry = np.array([q["expiration"] for q in quotes])
    return {
        "strike": np.array([q["strike"] for q in quotes]) / forward,  # forward units
        "maturity": np.array([q["maturity"] for q in quotes]),
        "put": np.array([q["option_type"] == "put" for q in quotes]),
        "half_spread": np.array([q["half_spread"] for q in quotes]) / (discount * forward),  # forward units
        "expiry": expiry,
        "target_z": np.array([q["target_z"] for q in quotes]),
        "windows": stored["windows"],
        "reference_volatility": float(stored["reference_volatility"]),
    }


SPX = _load_design()
SPX_REFERENCE = SPX["reference_volatility"]  # z = ln(K/F) / (0.16 sqrt T)
EXPIRIES = tuple(dict.fromkeys(SPX["expiry"]))  # in order, 7 to 325 days
DAYS = tuple(int(round(365 * SPX["maturity"][SPX["expiry"] == e][0])) for e in EXPIRIES)
# A reading keeps the expiries up to its last one. Keyed by the last expiry's days.
READINGS = {f"{DAYS[j]} days": j + 1 for j in (6, 4, 2, 1, 0)}
REACH = tuple(READINGS)  # widest first: '325 days', '81 days', '30 days', '14 days', '7 days'
LATER_WINDOW = tuple(SPX["windows"]["later_variance_window"])  # 81 to 325 days, in years
LAST = SPX["maturity"].max()  # the 325-day expiry


def heston_parameters(theta):
    """The runtime's parameter dictionary and initial variance for (v0, kappa, long-run variance, vol of vol, rho)."""
    v0, kappa, long_run, vol_of_vol, rho = (float(x) for x in theta)
    return {"kappa": kappa, "theta": long_run, "vol_of_vol": vol_of_vol, "rho": rho, "jump_intensity": 0.0,
            "jump_up_probability": 0.5, "jump_up_rate": 10.0, "jump_down_rate": 10.0}, v0


def expected_variance(theta, maturity):
    """E[integral of v from 0 to T] under Heston: the total variance of the Black control."""
    v0, kappa, long_run = theta[0], theta[1], theta[2]
    maturity = np.asarray(maturity, dtype=float)
    return long_run * maturity + (v0 - long_run) * (-np.expm1(-kappa * maturity)) / kappa


def prices(theta, strike, maturity, put, rung=RUNG):
    """European prices in forward units (forward 1, discount 1) by Lewis's integral with the Black control variate.

    price = Black(w) + sqrt(K)/pi * integral of Re[exp(-iu ln K) (phi_Black - phi_Heston)(u - i/2)] / (u^2 + 1/4),
    with w the expected total variance. A price outside the no-arbitrage bounds raises ValueError.
    """
    parameters, v0 = heston_parameters(theta)
    strike, maturity = np.broadcast_arrays(np.asarray(strike, dtype=float), np.asarray(maturity, dtype=float))
    put = np.broadcast_to(np.asarray(put, dtype=bool), strike.shape)
    variance = expected_variance(theta, maturity)
    black = np.where(put, gbm.european_price(1.0, strike, 1.0, variance, option="put"),
                     gbm.european_price(1.0, strike, 1.0, variance, option="call"))
    u, weights = heston._quadrature(int(rung[0]), float(rung[1]))
    times, inverse = np.unique(maturity.reshape(-1), return_inverse=True)
    phi_heston = heston.characteristic_function(u - 0.5j, times[:, None], parameters, v0)
    phi_black = np.exp(-0.5 * expected_variance(theta, times)[:, None] * (u * u + 0.25))
    correction = (phi_black - phi_heston)[inverse].reshape((*strike.shape, len(u)))
    phase = np.exp(-1j * u * np.log(strike)[..., None])
    price = black + np.sqrt(strike) * np.sum(np.real(phase * correction) / (u * u + 0.25) * weights, axis=-1) / np.pi
    lower = np.maximum(np.where(put, strike - 1.0, 1.0 - strike), 0.0)
    upper = np.where(put, strike, 1.0)
    if not np.isfinite(price).all() or np.any(price < lower - 1e-12) or np.any(price > upper + 1e-12):
        raise ValueError("Fourier price outside the no-arbitrage bounds")
    return price


def quote_prices(theta, n=len(SPX["strike"]), rung=RUNG):
    """The model prices of the first n quotes of the design, in forward units."""
    return prices(theta, SPX["strike"][:n], SPX["maturity"][:n], SPX["put"][:n], rung)


def gate(theta, n=len(SPX["strike"])):
    """The largest change of a quote's price from RUNG to CHECK_RUNG, in half-spreads (the gate asks for <= 0.01)."""
    return float(np.max(np.abs(quote_prices(theta, n, CHECK_RUNG) - quote_prices(theta, n)) / SPX["half_spread"][:n]))


def call(theta, strike, maturity, rung=CHECK_RUNG):
    return float(prices(theta, strike, maturity, False, rung))


def smile(theta, maturity, strikes):
    """Implied volatilities at one maturity from out-of-the-money prices (puts up to the forward, calls above)."""
    strikes = np.asarray(strikes, dtype=float)
    put = strikes <= 1.0
    values = prices(theta, strikes, np.full_like(strikes, maturity), put, CHECK_RUNG)
    return surface.implied_vol(values, np.log(strikes), maturity, np.where(put, "put", "call"))


# ---------------------------------------------------------------------------------------------------------------
# Products


_ROOTS, _WEIGHTS = roots_legendre(2048)
_U = 0.5 * 600.0 * (_ROOTS + 1.0)
_DU = 0.5 * 600.0 * _WEIGHTS


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


def forward_start_calls(theta, strikes=(1.0,), start=START, end=END):
    """E[X_start (X_end / X_start - m)^+] in units of the forward at `end`: calls struck at m times the index at start.

    With the price X as numeraire up to the start date, the variance is again a square-root process, with mean
    reversion kappa - rho * vol_of_vol. The return from start to end depends on the past only through the variance at
    the start, and its characteristic function is exp(A + B v_start). So each price is one Lewis integral of exp(A)
    times the moment generating function of v_start at B (Lucic 2003; Kruse and Noegel 2005). The at-the-money price
    was checked against a Monte Carlo simulation when the deck was first built; the tests check every strike against
    the vanilla call as start -> 0 and the at-the-money price against Black-Scholes as vol_of_vol -> 0.
    """
    v0, kappa, long_run, vol_of_vol, rho = (float(x) for x in theta)
    a, b = _exponents(_U - 0.5j, end - start, kappa, long_run, vol_of_vol, rho)
    kappa_share = kappa - rho * vol_of_vol
    scale = vol_of_vol**2 * (-np.expm1(-kappa_share * start)) / (4.0 * kappa_share)
    denominator = 1.0 - 2.0 * b * scale
    moment = np.exp(-(2.0 * kappa * long_run / vol_of_vol**2) * np.log(denominator)
                    + b * np.exp(-kappa_share * start) * v0 / denominator)
    transform = np.exp(a) * moment
    out = []
    for m in np.atleast_1d(strikes):
        integrand = np.real(np.exp(-1j * _U * np.log(m)) * transform) / (_U * _U + 0.25)
        out.append(1.0 - np.sqrt(m) * np.sum(integrand * _DU) / np.pi)
    return np.array(out)


def forward_start_call(theta, start=START, end=END):
    """The application's forward-start call: struck at the money at `start`, paid at `end`, in units of the forward."""
    return float(forward_start_calls(theta, (1.0,), start, end)[0])


def forward_smile(theta, strikes, start=START, end=END):
    """Implied volatilities of forward-start calls: the smile that the model expects at `start`, for `end`."""
    values = forward_start_calls(theta, strikes, start, end)
    strikes = np.asarray(strikes, dtype=float)
    return surface.implied_vol(values, np.log(strikes), end - start, np.full(len(strikes), "call"))


def forward_variance(theta, window=LATER_WINDOW):
    """The fair strike of a forward-starting variance swap, as a volatility: its section 4.1 closed form."""
    v0, kappa, long_run = theta[0], theta[1], theta[2]
    t1, t2 = window
    level = long_run + (v0 - long_run) * np.exp(-kappa * t1) * (-np.expm1(-kappa * (t2 - t1))) / (kappa * (t2 - t1))
    return float(np.sqrt(level))


def last_call(theta):
    """The listed 325-day at-the-money call, in units of the forward: the design's last expiry."""
    return call(theta, 1.0, LAST)


PRODUCTS = {"forward_start": forward_start_call, "forward_variance": forward_variance, "last_call": last_call}


# ---------------------------------------------------------------------------------------------------------------
# A reading of the simulated surface


def _observed(seed=SEED):
    clean = quote_prices(TRUTH)
    if gate(TRUTH) > GATE:
        raise ValueError("the Fourier integral is not converged to the gate at the generating parameters")
    return clean, clean + C_TRUE * SPX["half_spread"] * np.random.default_rng(seed).standard_normal(len(clean))


def _log_c_integral(r2, n):
    """log of the integral of c^-n exp(-R / (2 c^2)) over the prior range of c, up to a constant: c integrated out.

    With t = R / (2 c^2) it is (R/2)^((1-n)/2) Gamma(a) [P(a, 2R) - P(a, R/8)] / 2, a = (n - 1)/2, for c in [0.5, 2].
    """
    a = 0.5 * (n - 1)
    r2 = np.atleast_1d(np.asarray(r2, dtype=float))
    lo, hi = r2 / (2.0 * C_UPPER**2), r2 / (2.0 * C_LOWER**2)
    lower_tail = gammainc(a, hi) - gammainc(a, lo)
    upper_tail = gammaincc(a, lo) - gammaincc(a, hi)
    mass = np.where(lo > a, upper_tail, lower_tail)
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


@dataclass(frozen=True)
class Reading:
    """The quotes up to a last expiry: their count, half-spreads and one observed surface."""

    n: int
    half_spread: np.ndarray
    observed: np.ndarray
    reach: str

    def misses(self, theta):
        """(observed - model) / half-spread; a failed pricing counts as a very large miss."""
        try:
            return (self.observed - quote_prices(theta, self.n)) / self.half_spread
        except ValueError:
            return np.full(self.n, 1e4)

    def sum_of_squares(self, theta):
        r = self.misses(theta)
        return float(r @ r)

    def log_likelihood(self, theta):
        """log p(y | theta) with c integrated out under its uniform prior, up to a constant."""
        return float(_log_c_integral(self.sum_of_squares(theta), self.n)[0])

    def implied_quotes(self):
        """The observed prices as implied volatilities, with the band from price -/+ one half-spread."""
        k, maturity = np.log(SPX["strike"][:self.n]), SPX["maturity"][:self.n]
        kind = np.where(SPX["put"][:self.n], "put", "call")
        mid = surface.implied_vol(self.observed, k, maturity, kind)
        low = surface.implied_vol(np.maximum(self.observed - self.half_spread, 1e-14), k, maturity, kind)
        high = surface.implied_vol(self.observed + self.half_spread, k, maturity, kind)
        return mid, low, high


def readings(seed=SEED):
    """The five nested readings of one simulated surface, keyed by their last expiry ('325 days' ... '7 days')."""
    _, observed = _observed(seed)
    out = {}
    for reach, expiries in READINGS.items():
        n = 9 * expiries
        out[reach] = Reading(n, SPX["half_spread"][:n], observed[:n], reach)
    return out


# ---------------------------------------------------------------------------------------------------------------
# The classical route: least squares, the error scale, Gauss-Newton, the delta method


@dataclass(frozen=True)
class Fit:
    theta: np.ndarray
    sum_of_squares: float  # R: the sum of squared misses in half-spreads
    c: float  # the maximum-likelihood error scale sqrt(R/n), within the prior's range
    free: tuple  # indices of the fitted parameters
    covariance: np.ndarray  # of the free parameters; None when the optimum is on a bound
    on_bound: tuple  # names of the parameters that sit on a bound of the box


def _jacobian(reading, theta, free, step=1e-4):
    columns = []
    for i in free:
        h = step * (UPPER[i] - LOWER[i])
        up, down = theta.copy(), theta.copy()
        up[i], down[i] = min(theta[i] + h, UPPER[i]), max(theta[i] - h, LOWER[i])
        columns.append((reading.misses(down) - reading.misses(up)) / (up[i] - down[i]))
    return np.column_stack(columns)  # d(model / half-spread) / d theta


def fit(reading, fixed=None, starts=8, seed=0):
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
        result = least_squares(lambda x: reading.misses(full(x)), start,
                               bounds=(LOWER[list(free)], UPPER[list(free)]), x_scale=(UPPER - LOWER)[list(free)],
                               xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or result.cost < best.cost:
            best = result
    theta = full(best.x)
    r2 = 2.0 * best.cost
    c = float(np.clip(np.sqrt(r2 / reading.n), C_LOWER, C_UPPER))
    on_bound = tuple(NAMES[i] for i in free
                     if min(theta[i] - LOWER[i], UPPER[i] - theta[i]) < 1e-6 * (UPPER[i] - LOWER[i]))
    covariance = None
    if not on_bound:
        j = _jacobian(reading, theta, free)
        covariance = c**2 * np.linalg.inv(j.T @ j)
    return Fit(theta, r2, c, free, covariance, on_bound)


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
# The posterior: importance sampling in logit coordinates, c integrated out and then drawn


@dataclass(frozen=True)
class Posterior:
    draws: np.ndarray  # equal-weight draws of (v0, kappa, long-run variance, vol of vol, rho, c), shape (keep, 6)
    effective_sample_size: float  # of the weighted proposals the draws were resampled from
    proposals: int  # number of those proposals
    failed: float  # share of them whose pricing failed (counted as zero likelihood)
    acceptance: float = None  # of the Metropolis moves, where the draws were moved after resampling; else None


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


def _moves(reading, u, r2, sweeps, rng):
    """Differential-evolution Metropolis sweeps over a population of draws, each leaving the posterior invariant.

    For each draw, the proposal adds gamma times the difference of two other draws of the population as it stood at
    the start of the sweep, plus a tiny jitter; it is symmetric, so the Metropolis ratio is the ratio of the posterior
    densities in logit coordinates. gamma starts at 2.38/sqrt(2d) and is scaled between sweeps towards an acceptance
    of about a quarter (a thin curved ridge needs shorter steps); in a tenth of the moves it is 1, to cross between
    modes.
    """
    count, dim = u.shape
    log_post = _log_c_integral(r2, reading.n) + _log_prior(u)
    accepted, scale = 0, 2.38 / np.sqrt(2 * dim)
    for _ in range(sweeps):
        a = rng.integers(0, count - 1, count)
        a += a >= np.arange(count)
        b = rng.integers(0, count - 2, count)
        b += b >= np.minimum(a, np.arange(count))
        b += b >= np.maximum(a, np.arange(count))
        gamma = np.where(rng.uniform(size=count) < 0.1, 1.0, scale)
        proposal = u + gamma[:, None] * (u[a] - u[b]) + 1e-6 * rng.standard_normal((count, dim))
        new_r2 = np.array([reading.sum_of_squares(t) for t in _to_theta(proposal)])
        new_log_post = _log_c_integral(new_r2, reading.n) + _log_prior(proposal)
        new_log_post[new_r2 >= 1e8 * reading.n * 0.99] = -np.inf
        accept = np.log(rng.uniform(size=count)) < new_log_post - log_post
        u[accept], r2[accept], log_post[accept] = proposal[accept], new_r2[accept], new_log_post[accept]
        accepted += int(accept.sum())
        rate = accept.mean()
        scale *= 0.8 if rate < 0.15 else 1.2 if rate > 0.35 else 1.0
    return u, r2, accepted / (sweeps * count)


def posterior(reading, *, adapt=(4000, 4000, 4000), temper=0.5, pieces=4, batch=6000, target=5000.0, most=72000,
              keep=4000, prior_share=0.2, dof=5.0, inflate=2.0, rejuvenate_below=2000.0, sweeps=100, seed=1):
    """Equal-weight draws from p(theta, c | quotes) under the application's uniform priors, by importance sampling.

    c is integrated out of the likelihood in closed form, so the importance sampler works on the five Heston
    parameters. The proposal is a mixture in logit coordinates: `prior_share` from the prior, the rest from
    multivariate t components. It starts as one t at the least-squares fit with its Gauss-Newton shape. Each `adapt`
    round weights its proposals by prior times likelihood to the power `temper` (a flattened posterior, which reaches
    further along a valley than the posterior itself), cuts them into `pieces` slices along their principal axis, and
    puts one t on each slice with an inflated covariance. With the proposal then frozen, batches are drawn until the
    effective sample size reaches `target` or `most` proposals are used. The final weights are prior times likelihood
    over the mixture density; `keep` draws are resampled systematically from them. Where the effective sample size
    stays below `rejuvenate_below`, as on a reading whose posterior is a thin curved ridge, the resampled draws are
    then moved by `sweeps` differential-evolution Metropolis sweeps at the posterior (resample-move). Each draw finally
    gets a c drawn from its exact conditional given that draw's misses.
    """
    rng = np.random.default_rng(seed)
    dim = len(NAMES)
    start = fit(reading)
    inside = np.clip(start.theta, LOWER + 1e-3 * (UPPER - LOWER), UPPER - 1e-3 * (UPPER - LOWER))
    mean = logit((inside - LOWER) / (UPPER - LOWER))
    e = expit(mean)
    j = _jacobian(reading, inside, tuple(range(dim))) * ((UPPER - LOWER) * e * (1.0 - e)) / start.c
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
        r2 = np.array([reading.sum_of_squares(t) for t in theta])
        log_like = _log_c_integral(r2, reading.n)
        log_like[r2 >= 1e8 * reading.n * 0.99] = -np.inf  # failed pricings
        return u, theta, r2, log_like, log_prior - logsumexp(np.stack(parts), axis=0)

    def normalized(log_w):
        log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
        w = np.exp(log_w - log_w.max())
        return w / w.sum()

    for n in adapt:
        u, _, _, log_like, log_ratio = propose(n)
        components = _components(u, normalized(temper * log_like + log_ratio), pieces, inflate)
    thetas, r2s, log_ws = [], [], []
    while True:
        _, theta, r2, log_like, log_ratio = propose(batch)
        thetas.append(theta)
        r2s.append(r2)
        log_ws.append(log_like + log_ratio)
        w = normalized(np.concatenate(log_ws))
        ess = 1.0 / np.sum(w * w)
        if ess >= target or len(w) >= most:
            break
    theta, r2 = np.concatenate(thetas), np.concatenate(r2s)
    failed = np.mean(r2 >= 1e8 * reading.n * 0.99)
    positions = (rng.uniform() + np.arange(keep)) / keep
    chosen = rng.permutation(np.minimum(np.searchsorted(np.cumsum(w), positions), len(w) - 1))
    theta, r2, acceptance = theta[chosen], r2[chosen], None
    if ess < rejuvenate_below:
        u = logit(np.clip((theta - LOWER) / (UPPER - LOWER), 1e-15, 1 - 1e-15))
        u, r2, acceptance = _moves(reading, u, r2.copy(), sweeps, rng)
        theta = _to_theta(u)
    c = draw_c(r2, reading.n, rng)
    return Posterior(np.column_stack([theta, c]), float(ess), int(len(w)), float(failed), acceptance)


def within(draws, lower=None, upper=None):
    """The draws inside a smaller box: the posterior under a uniform prior on that box."""
    lower = LOWER if lower is None else np.asarray(lower, dtype=float)
    upper = UPPER if upper is None else np.asarray(upper, dtype=float)
    theta = draws[:, :len(NAMES)]
    return draws[np.all((theta >= lower) & (theta <= upper), axis=1)]
