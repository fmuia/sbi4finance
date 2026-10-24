"""The stochastic-volatility toy of Lecture 9.7, for slot 1: a posterior that is not Gaussian.

Nine "quotes" at standardized strikes x_j in {-1, -0.75, ..., 1}:

    y_j = rho * eta * x_j + 0.05 * eta^2 * x_j^2 + s_j * eps_j,    s_j = 0.01 + 0.10 * w * x_j^2,

with eta in (0.5, 3.5) and rho in (-0.99, -0.2). The odd part, the skew, carries the product rho * eta and is
large against the noise; the even part, the curvature, carries eta^2, is small, and is largest in the wings,
where the noise is largest. It is the structure of a stochastic-volatility smile (Heston's sigma_v and rho,
rough Bergomi's eta and rho) whose wings are few or noisy; w scales the wing noise (w = 1 is the course's).
The constants are those of scripts/course/lecture09.py.
"""

import numpy as np
from scipy.optimize import least_squares

X = np.linspace(-1.0, 1.0, 9)
CURVATURE = 0.05
ETA_BOX = (0.5, 3.5)
RHO_BOX = (-0.99, -0.2)
TRUTHS = {"near the bound: η = 1.0, ρ = −0.87": (1.0, -0.87), "inside the box: η = 2.0, ρ = −0.435": (2.0, -0.435)}
ETA_GRID = np.linspace(*ETA_BOX, 321)
RHO_GRID = np.linspace(*RHO_BOX, 241)


def noise(wing=1.0):
    return 0.01 + 0.10 * wing * X**2


def model(eta, rho, x=X):
    eta, rho = np.asarray(eta, dtype=float), np.asarray(rho, dtype=float)
    return rho[..., None] * eta[..., None] * x + CURVATURE * eta[..., None] ** 2 * x**2


def data(truth, rng, wing=1.0):
    return model(np.array(truth[0]), np.array(truth[1])) + noise(wing) * rng.standard_normal(len(X))


def posterior_grid(y, wing=1.0):
    """The exact posterior under the uniform prior on the box, on a 321 x 241 grid (rows: rho, columns: eta)."""
    eta, rho = np.meshgrid(ETA_GRID, RHO_GRID)
    chi2 = np.sum(((y - model(eta, rho)) / noise(wing)) ** 2, axis=-1)
    p = np.exp(-0.5 * (chi2 - chi2.min()))
    return eta, rho, p / p.sum()


def fit(y, wing=1.0, starts=((2.0, -0.6), (1.0, -0.9), (3.0, -0.3))):
    """Bounded least squares from three starts; the best optimum and its chi-square."""
    s = noise(wing)
    best = None
    for start in starts:
        r = least_squares(lambda t: (y - model(np.array(t[0]), np.array(t[1]))) / s, start,
                          bounds=([ETA_BOX[0], RHO_BOX[0]], [ETA_BOX[1], RHO_BOX[1]]), xtol=1e-12, ftol=1e-12)
        if best is None or r.cost < best.cost:
            best = r
    return best.x, 2 * best.cost


def gauss_newton_cov(eta, rho, wing=1.0):
    j = np.column_stack([rho * X + 2 * CURVATURE * eta * X**2, eta * X]) / noise(wing)[:, None]
    return np.linalg.inv(j.T @ j)


def weighted_quantiles(values, weights, qs):
    order = np.argsort(values)
    c = np.cumsum(weights[order])
    return np.interp(qs, c / c[-1], values[order])


def summary(y, wing=1.0, x_product=2.0):
    """Posterior and Gauss–Newton 90% intervals for eta, rho, rho*eta and the quote at an unquoted strike."""
    eta, rho, p = posterior_grid(y, wing)
    best, chi2 = fit(y, wing)
    cov = gauss_newton_cov(*best, wing)
    flat = p.ravel()
    quantities = {"η": eta.ravel(), "ρ": rho.ravel(), "ρη": (eta * rho).ravel(),
                  f"quote at x = {x_product:g}": model(eta, rho, np.array([x_product]))[..., 0].ravel()}
    grads = {"η": np.array([1.0, 0.0]), "ρ": np.array([0.0, 1.0]), "ρη": np.array([best[1], best[0]]),
             f"quote at x = {x_product:g}": np.array([best[1] * x_product + 2 * CURVATURE * best[0] * x_product**2,
                                                       best[0] * x_product])}
    points = {"η": best[0], "ρ": best[1], "ρη": best[0] * best[1],
              f"quote at x = {x_product:g}": float(model(best[0], best[1], np.array([x_product]))[0])}
    rows = []
    for name, values in quantities.items():
        lo, med, hi = weighted_quantiles(values, flat, [0.05, 0.5, 0.95])
        sd = float(np.sqrt(grads[name] @ cov @ grads[name]))
        rows.append({"quantity": name, "posterior_90": (float(lo), float(hi)), "posterior_median": float(med),
                     "gauss_newton_90": (points[name] - 1.645 * sd, points[name] + 1.645 * sd),
                     "least_squares": points[name]})
    return {"eta": eta, "rho": rho, "p": p, "best": best, "chi2": chi2, "cov": cov, "rows": rows}
