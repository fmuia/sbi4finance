"""The opening argument's example: two decisions that rest on estimated inputs, on 20 years of US returns.

The first five slides argue that finance needs the uncertainty of its estimates, and that Bayes states it. They use
one example, kept as small as it can be:

- a portfolio of the ten US industry portfolios with the highest Sharpe ratio, short positions allowed. Its inputs
  are the industries' average excess returns and their covariances. Raising one average by 10%, a third of its
  standard error, moves the weights by tens of points (slide 1);
- the same portfolio under a prior belief that the ten industries have the same expected return (slides 3 and 4).
  The prior is mu ~ N(eta 1, Sigma / tau) with eta diffuse (Jorion 1986): it is worth tau months of data, so its
  weight against T months of data is tau / (T + tau). Weight 0 is the plug-in route, weight 1 takes the prior as
  certain and gives the minimum-variance portfolio, whatever the averages are;
- the share in stocks of a buy-and-hold investor with power utility, against the horizon, with the estimated mean and
  volatility of the market taken as exact and with their uncertainty included (slide 2; Barberis 2000, Section II).

Inputs: `intro-inputs.json`, the averages and covariances of monthly returns in excess of the one-month bill rate,
October 2006 to September 2026, computed from Kenneth French's data library (CRSP). `build_inputs` recomputes the
file from the library's two CSV files and records their SHA-256.

These are illustrations of a mechanism on real returns, not results of the tutorial. `intro_figures.py` draws them.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "intro-inputs.json"

YEARS = 20  # the estimation window, ending with the last month of the library's files
INDUSTRIES = {  # Kenneth French's ten industry portfolios, by the names the slides use
    "NoDur": "Nondurables", "Durbl": "Durables", "Manuf": "Manufacturing", "Enrgy": "Energy", "HiTec": "Technology",
    "Telcm": "Telecom", "Shops": "Retail", "Hlth": "Health care", "Utils": "Utilities", "Other": "Other",
}
CHANGED = "HiTec"  # the industry whose average return is raised: the one with the highest average
CHANGE = 0.10
RISK_AVERSION = 5.0  # of the buy-and-hold investor
SHARES = np.linspace(0.0, 0.99, 991)  # as in Barberis (2000): no short sales, no buying on margin
Z = np.linspace(-10.0, 10.0, 4001)  # the grid of the standardized cumulative return
LIBRARY = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {"industries": "10_Industry_Portfolios", "factors": "F-F_Research_Data_Factors"}


@dataclass(frozen=True)
class Inputs:
    """Monthly excess returns, as decimals: what the two decisions are computed from."""

    months: int
    first_month: int
    last_month: int
    codes: tuple
    mean: np.ndarray  # the industries' average excess returns
    covariance: np.ndarray  # their sample covariance (unbiased)
    market_mean: float  # the market's average excess log return
    market_volatility: float  # its sample standard deviation

    @property
    def names(self):
        return tuple(INDUSTRIES[code] for code in self.codes)

    @property
    def changed(self):
        return self.codes.index(CHANGED)

    @property
    def standard_error(self):
        return np.sqrt(np.diag(self.covariance) / self.months)


def load(path=INPUTS):
    stored = json.loads(Path(path).read_text())
    window, industries, market = stored["window"], stored["industries"], stored["market"]
    return Inputs(window["months"], window["first_month"], window["last_month"], tuple(industries["codes"]),
                  np.array(industries["mean"]), np.array(industries["covariance"]),
                  market["mean_excess_log_return"], market["volatility_excess_log_return"])


# ---------------------------------------------------------------------------------------------------------------
# The portfolio


def max_sharpe(mean, covariance):
    """The fully invested portfolio with the highest Sharpe ratio, short positions allowed."""
    raw = np.linalg.solve(covariance, mean)
    if raw.sum() <= 0:
        raise ValueError("the maximum Sharpe ratio portfolio is not fully invested for these inputs")
    return raw / raw.sum()


def max_sharpe_long_only(mean, covariance):
    """The same portfolio with no short positions, by enumeration of the industries it holds (exact for ten)."""
    n, best, held = len(mean), -np.inf, None
    for subset in range(1, 2**n):
        index = [i for i in range(n) if subset >> i & 1]
        raw = np.linalg.solve(covariance[np.ix_(index, index)], mean[index])
        if np.all(raw > 0) and float(mean[index] @ raw) > best:  # the squared Sharpe ratio of this subset
            best, held = float(mean[index] @ raw), np.zeros(n)
            held[index] = raw / raw.sum()
    return held


def minimum_variance(covariance):
    raw = np.linalg.solve(covariance, np.ones(len(covariance)))
    return raw / raw.sum()


def prior_months(weight, months):
    """The months of data a prior of this weight is worth: weight = tau / (months + tau)."""
    return months * weight / (1.0 - weight)


def predictive(mean, covariance, months, weight):
    """The mean and covariance of next month's returns under the prior that the industries share one expected return.

    The prior is mu ~ N(eta 1, Sigma / tau) with eta diffuse and Sigma known, set to the sample covariance scaled
    by (T - 1) / (T - N - 2) as in Jorion (1986). `weight` = tau / (T + tau) runs from 0, no prior, to 1, the prior
    taken as certain. The posterior of mu is normal with mean (1 - weight) average + weight common and covariance
    Sigma / (T + tau) + tau / (T (T + tau)) 1 1' / (1' Sigma^-1 1); the predictive covariance adds Sigma.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError("the weight on the prior lies between 0 and 1")
    n = len(mean)
    sigma = covariance * (months - 1) / (months - n - 2)
    inverse_one = np.linalg.solve(sigma, np.ones(n))
    common = float(inverse_one @ mean / inverse_one.sum())  # the average return of the minimum-variance portfolio
    shared = np.ones((n, n)) / inverse_one.sum()
    if weight == 1.0:
        return np.full(n, common), sigma + shared / months
    tau = prior_months(weight, months)
    return ((1.0 - weight) * mean + weight * common,
            sigma * (1.0 + 1.0 / (months + tau)) + tau / (months * (months + tau)) * shared)


def suggested_weight(mean, covariance, months):
    """The weight on the prior that the spread of the averages suggests (Jorion 1986, an empirical Bayes estimate)."""
    n = len(mean)
    sigma = covariance * (months - 1) / (months - n - 2)
    gap = mean - float(minimum_variance(sigma) @ mean)
    return (n + 2) / ((n + 2) + months * float(gap @ np.linalg.solve(sigma, gap)))


def weights(inputs, weight, raised=None):
    """The portfolio under a prior of this weight; `raised` is the industry whose average goes up by CHANGE."""
    mean = inputs.mean.copy()
    if raised is not None:
        mean[raised] *= 1.0 + CHANGE
    return max_sharpe(*predictive(mean, inputs.covariance, inputs.months, weight))


def one_parameter(inputs, weight):
    """The prior, the data and the posterior of the changed industry's expected return: (mean, standard deviation).

    One parameter, with the common value taken as given and the industry's sample variance as known: the prior is
    N(common, variance / tau), the data say N(average, variance / T), and the posterior is their product.
    """
    k, tau = inputs.changed, prior_months(weight, inputs.months)
    variance = inputs.covariance[k, k]
    common = float(minimum_variance(inputs.covariance) @ inputs.mean)
    return {"prior": (common, float(np.sqrt(variance / tau))),
            "data": (float(inputs.mean[k]), float(np.sqrt(variance / inputs.months))),
            "posterior": ((1.0 - weight) * float(inputs.mean[k]) + weight * common,
                          float(np.sqrt(variance / (inputs.months + tau))))}


# ---------------------------------------------------------------------------------------------------------------
# The share in stocks


def best_share(location, scale, density):
    """The share that maximizes expected power utility when the cumulative excess log return is location + scale z."""
    growth = np.exp(location + scale * Z)  # a unit of stocks relative to a unit of bills, at the horizon
    mass = density(Z)
    mass = mass / np.trapezoid(mass, Z)
    wealth = (1.0 - SHARES[:, None]) + SHARES[:, None] * growth[None, :]
    expected = np.trapezoid(wealth ** (1.0 - RISK_AVERSION) / (1.0 - RISK_AVERSION) * mass[None, :], Z, axis=1)
    return float(SHARES[np.argmax(expected)])


def shares(inputs, years):
    """The share in stocks at each horizon: the estimates taken as exact, and their uncertainty included.

    Excess log returns are i.i.d. normal. Taken as exact, the cumulative return over h months is N(h m, h s^2). With
    a flat prior on the mean and the log variance, it is m h + s sqrt(h + h^2 / T) t, with t Student on T - 1 degrees
    of freedom: the uncertainty about the mean adds a variance that grows with the square of the horizon.
    """
    m, s, months = inputs.market_mean, inputs.market_volatility, inputs.months
    exact, uncertain = [], []
    for horizon in 12.0 * np.asarray(years, dtype=float):
        exact.append(best_share(horizon * m, s * np.sqrt(horizon), stats.norm.pdf))
        uncertain.append(best_share(horizon * m, s * np.sqrt(horizon + horizon**2 / months),
                                    lambda z: stats.t.pdf(z, df=months - 1)))
    return np.array(exact), np.array(uncertain)


# ---------------------------------------------------------------------------------------------------------------
# The inputs, from Kenneth French's data library


def monthly_table(path, columns):
    """The first monthly table of one of the library's CSV files: rows 'YYYYMM, v1, v2, ...'."""
    rows, started = [], False
    for line in Path(path).read_text().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == columns + 1 and parts[0].isdigit() and len(parts[0]) == 6:
            started = True
            rows.append([int(parts[0])] + [float(part) for part in parts[1:]])
        elif started:
            break
    return np.array(rows)


def build_inputs(folder, retrieved):
    """`intro-inputs.json` from the library's two files, unzipped in `folder` beside the downloaded zip files."""
    folder = Path(folder)
    industries = monthly_table(folder / f"{FILES['industries']}.csv", 10)
    factors = monthly_table(folder / f"{FILES['factors']}.csv", 4)
    if not np.array_equal(industries[:, 0], factors[:, 0]) or np.any(industries[:, 1:] <= -99):
        raise ValueError("the two files cover different months, or returns are missing")
    lines = (folder / f"{FILES['industries']}.csv").read_text().splitlines()
    header = next(line for line in lines if line.startswith(","))  # the column names of the first table
    codes = [code.strip() for code in header.split(",")[1:]]
    if codes != list(INDUSTRIES):
        raise ValueError(f"unexpected industry columns: {codes}")
    months = 12 * YEARS
    riskfree = factors[-months:, 4] / 100.0
    excess = industries[-months:, 1:] / 100.0 - riskfree[:, None]
    market = np.log1p((factors[-months:, 1] + factors[-months:, 4]) / 100.0) - np.log1p(riskfree)
    vintage = lines[0].strip()
    return {
        "what": "Inputs of the opening figures of slot 1: averages and covariances of monthly US excess returns.",
        "units": "monthly returns in excess of the one-month bill rate, as decimals; the market's are log returns",
        "source": {
            "provider": "Kenneth R. French, Data Library (CRSP)",
            "page": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
            "vintage": vintage,
            "retrieved": retrieved,
            "files": {f"{name}_CSV.zip": {
                "url": f"{LIBRARY}{name}_CSV.zip",
                "sha256": hashlib.sha256((folder / f"{name}_CSV.zip").read_bytes()).hexdigest(),
            } for name in FILES.values()},
            "tables": {"industries": "Average Value Weighted Returns -- Monthly", "market": "Mkt-RF plus RF, and RF"},
        },
        "window": {"first_month": int(industries[-months, 0]), "last_month": int(industries[-1, 0]),
                   "months": months},
        "industries": {"codes": codes, "mean": excess.mean(axis=0).tolist(),
                       "covariance": np.cov(excess, rowvar=False).tolist()},
        "market": {"mean_excess_log_return": float(market.mean()),
                   "volatility_excess_log_return": float(market.std(ddof=1))},
    }
