"""SVJ in its daily discrete form: prior, simulator, forward paths and network input.

The model is the Bates (1996) stochastic-volatility model with Gaussian return jumps,
discretized as in Eraker, Johannes and Polson (2003). Returns are daily log returns in
percent, the variance v_t is in %² per day, and one step is one trading day:

    y_t = mu + sqrt(v_{t-1}) eps1_t + B_t xi_t,      B_t ~ Bernoulli(lam), xi_t ~ N(mu_J, sigma_J²)
    v_t = max(v_{t-1} + kappa (theta - v_{t-1}) + sigma_v sqrt(v_{t-1}) (rho eps1_t + sqrt(1 - rho²) eps2_t), V_MIN)
    v_0 ~ Gamma(1 / c², theta c²),                   c = sigma_v / sqrt(2 kappa theta)

The discretization is the model: the simulator, the particle filter, the network and the
forecast all use these equations. Parameters are drawn and inferred in coordinates a risk
manager can judge, each mapped to an unbounded scale (`COORDINATE_NAMES`). The ninth
inference target is log spot volatility at the end of the window, log sigma_T with
sigma_T = sqrt(252 v_T) / 100, so one network pass returns what a forecast needs.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import betaln, log_expit
from scipy.stats import beta

WINDOW = 1_000
DAYS_PER_YEAR = 252
V_MIN = 1e-3  # %² per day, 0.5% annualized
LAMBDA_MAX = 0.5  # cap on the daily jump probability; reached only far in the prior tail

COORDINATE_NAMES = (
    "drift",  # 252 mu, % per year
    "log_vol_lr",  # log of long-run volatility sqrt(252 theta) / 100
    "log_half_life",  # log of ln 2 / kappa, trading days
    "log_cv",  # log of c = sigma_v / sqrt(2 kappa theta)
    "atanh_rho",  # leverage
    "log_jumps_per_year",  # log of 252 lambda
    "mean_jump",  # mu_J, %
    "log_jump_sd",  # log of sigma_J, %
)
TARGET_NAMES = COORDINATE_NAMES + ("log_spot_vol",)  # log sigma_T at the last day
N_PARAMS = len(COORDINATE_NAMES)
N_TARGETS = len(TARGET_NAMES)

QUANTITY_LABELS = (
    "drift (%/yr)",
    "long-run vol",
    "half-life (days)",
    "CV of variance",
    "leverage rho",
    "jumps per year",
    "mean jump (%)",
    "jump sd (%)",
    "spot vol",
)
MODEL_PARAM_NAMES = ("mu", "theta", "kappa", "sigma_v", "rho", "lam", "mu_j", "sigma_j")


@dataclass(frozen=True)
class Prior:
    """Independent priors on the unbounded coordinates; rho via (rho + 1) / 2 ~ Beta(a, b)."""

    drift_loc: float = 8.0
    drift_sd: float = 10.0
    vol_lr: float = 0.25
    vol_lr_sd: float = 0.40
    half_life: float = 30.0
    half_life_sd: float = 0.70
    cv: float = 0.6
    cv_sd: float = 0.35
    rho_a: float = 2.0
    rho_b: float = 4.0
    jumps_per_year: float = 2.5
    jumps_per_year_sd: float = 0.75
    mean_jump_loc: float = -1.0
    mean_jump_sd: float = 2.5
    jump_sd: float = 3.0
    jump_sd_sd: float = 0.45

    def normal_coordinates(self):
        """(index, mean, sd) of the seven coordinates with a normal prior."""
        return (
            (0, self.drift_loc, self.drift_sd),
            (1, np.log(self.vol_lr), self.vol_lr_sd),
            (2, np.log(self.half_life), self.half_life_sd),
            (3, np.log(self.cv), self.cv_sd),
            (5, np.log(self.jumps_per_year), self.jumps_per_year_sd),
            (6, self.mean_jump_loc, self.mean_jump_sd),
            (7, np.log(self.jump_sd), self.jump_sd_sd),
        )


PRIOR = Prior()


def prior_table(prior=PRIOR):
    """The prior on the eight parameter coordinates, each with its definition and 95% range."""
    import pandas as pd

    z = 1.959964
    rows = [
        ("Drift", "252 mu, % per year", f"Normal({prior.drift_loc:g}, {prior.drift_sd:g}²)",
         prior.drift_loc - z * prior.drift_sd, prior.drift_loc + z * prior.drift_sd),
        ("Long-run volatility", "sqrt(252 theta) / 100", f"log ~ Normal(log {prior.vol_lr:g}, {prior.vol_lr_sd:g}²)",
         prior.vol_lr * np.exp(-z * prior.vol_lr_sd), prior.vol_lr * np.exp(z * prior.vol_lr_sd)),
        ("Half-life of variance shocks", "ln 2 / kappa, days",
         f"log ~ Normal(log {prior.half_life:g}, {prior.half_life_sd:g}²)",
         prior.half_life * np.exp(-z * prior.half_life_sd), prior.half_life * np.exp(z * prior.half_life_sd)),
        ("CV of variance", "sigma_v / sqrt(2 kappa theta)", f"log ~ Normal(log {prior.cv:g}, {prior.cv_sd:g}²)",
         prior.cv * np.exp(-z * prior.cv_sd), prior.cv * np.exp(z * prior.cv_sd)),
        ("Leverage", "rho", f"(rho + 1) / 2 ~ Beta({prior.rho_a:g}, {prior.rho_b:g})",
         2 * beta.ppf(0.025, prior.rho_a, prior.rho_b) - 1, 2 * beta.ppf(0.975, prior.rho_a, prior.rho_b) - 1),
        ("Jump frequency", "252 lambda, jumps per year",
         f"log ~ Normal(log {prior.jumps_per_year:g}, {prior.jumps_per_year_sd:g}²)",
         prior.jumps_per_year * np.exp(-z * prior.jumps_per_year_sd),
         prior.jumps_per_year * np.exp(z * prior.jumps_per_year_sd)),
        ("Mean jump size", "mu_J, %", f"Normal({prior.mean_jump_loc:g}, {prior.mean_jump_sd:g}²)",
         prior.mean_jump_loc - z * prior.mean_jump_sd, prior.mean_jump_loc + z * prior.mean_jump_sd),
        ("Jump size sd", "sigma_J, %", f"log ~ Normal(log {prior.jump_sd:g}, {prior.jump_sd_sd:g}²)",
         prior.jump_sd * np.exp(-z * prior.jump_sd_sd), prior.jump_sd * np.exp(z * prior.jump_sd_sd)),
    ]
    return pd.DataFrame(rows, columns=["quantity", "definition", "prior", "2.5%", "97.5%"]).set_index("quantity")


def draw_coordinates(n, rng, prior=PRIOR):
    """n draws of the eight parameter coordinates from the prior, shape (n, 8)."""
    coords = np.empty((n, N_PARAMS))
    for i, mean, sd in prior.normal_coordinates():
        coords[:, i] = rng.normal(mean, sd, n)
    u = np.clip(rng.beta(prior.rho_a, prior.rho_b, n), 1e-9, 1 - 1e-9)
    coords[:, 4] = np.arctanh(2 * u - 1)
    return coords


def log_prior(coords, prior=PRIOR):
    """Log prior density of the unbounded coordinates, shape (...,)."""
    coords = np.asarray(coords, dtype=float)
    out = np.zeros(coords.shape[:-1])
    for i, mean, sd in prior.normal_coordinates():
        out += -0.5 * ((coords[..., i] - mean) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi)
    # u = (tanh z + 1) / 2 = expit(2z); dz -> du has Jacobian 2 u (1 - u)
    z = coords[..., 4]
    log_u, log_1mu = log_expit(2 * z), log_expit(-2 * z)
    out += (prior.rho_a - 1) * log_u + (prior.rho_b - 1) * log_1mu - betaln(prior.rho_a, prior.rho_b)
    out += np.log(2) + log_u + log_1mu
    return out


def model_params(coords):
    """Map coordinates (..., >=8) to the daily model's (mu, theta, kappa, sigma_v, rho, lam, mu_J, sigma_J)."""
    coords = np.asarray(coords, dtype=float)
    drift, log_vol, log_h, log_c, z_rho, log_jpy, mean_jump, log_jsd = np.moveaxis(coords[..., :N_PARAMS], -1, 0)
    theta = (100 * np.exp(log_vol)) ** 2 / DAYS_PER_YEAR
    kappa = np.log(2) / np.exp(log_h)
    sigma_v = np.exp(log_c) * np.sqrt(2 * kappa * theta)
    lam = np.minimum(np.exp(log_jpy) / DAYS_PER_YEAR, LAMBDA_MAX)
    return np.stack(
        [drift / DAYS_PER_YEAR, theta, kappa, sigma_v, np.tanh(z_rho), lam, mean_jump, np.exp(log_jsd)], axis=-1
    )


def quantities(targets):
    """Coordinates or targets (..., 8 or 9) in the units of the prior table."""
    targets = np.asarray(targets, dtype=float)
    out = np.empty_like(targets)
    out[..., 0] = targets[..., 0]
    out[..., 1] = np.exp(targets[..., 1])
    out[..., 2] = np.exp(targets[..., 2])
    out[..., 3] = np.exp(targets[..., 3])
    out[..., 4] = np.tanh(targets[..., 4])
    out[..., 5] = np.exp(targets[..., 5])
    out[..., 6] = targets[..., 6]
    out[..., 7] = np.exp(targets[..., 7])
    if targets.shape[-1] > N_PARAMS:
        out[..., 8] = np.exp(targets[..., 8])
    return out


def spot_vol_from_variance(v):
    """Annualized spot volatility as a fraction: sqrt(252 v) / 100."""
    return np.sqrt(DAYS_PER_YEAR * np.asarray(v)) / 100


def log_spot_vol(v):
    return 0.5 * np.log(DAYS_PER_YEAR * np.asarray(v, dtype=float)) - np.log(100)


def variance_from_log_spot_vol(log_sigma):
    return (100 * np.exp(np.asarray(log_sigma, dtype=float))) ** 2 / DAYS_PER_YEAR


def initial_variance(coords, rng):
    """v_0 from the declared Gamma starting law, floored at V_MIN.

    This initialization is not an established stationary distribution of the
    clipped daily recursion.
    """
    theta = model_params(coords)[..., 1]
    c2 = np.exp(2 * np.asarray(coords)[..., 3])
    return np.maximum(rng.gamma(shape=1 / c2, scale=theta * c2), V_MIN)


def run_daily_model(params, v_start, n_days, rng):
    """Run the daily model for n_days from v_start; params (n, 8), v_start (n,).

    Returns the returns (n, n_days), the variance path v_1..v_n_days (n, n_days) and the jump indicators.
    """
    n = len(v_start)
    eps1 = rng.standard_normal((n_days, n))
    eps2 = rng.standard_normal((n_days, n))
    uniforms = rng.random((n_days, n))
    jump_normals = rng.standard_normal((n_days, n))
    result = daily_model_from_innovations(
        params, v_start, eps1, eps2, uniforms, jump_normals, retain_auxiliary=True
    )
    return result["returns"], result["variance"], result["jumps"]


def daily_model_from_innovations(
    params, v_start, eps1, eps2, jump_uniforms, jump_normals, *, retain_auxiliary=False
):
    """Advance the same daily law with explicit random inputs [time,batch].

    The saved-bank path assigns independent random streams to each example,
    then advances a whole batch together. By default only returns and the
    terminal variance are retained, avoiding full latent-path storage. The
    legacy generator above keeps its original draw order and auxiliary output.
    No parameter or state here is an observed neural conditioning input.
    """
    params = np.asarray(params, dtype=np.float64)
    v_start = np.asarray(v_start, dtype=np.float64)
    inputs = [np.asarray(a, dtype=np.float64) for a in
              (eps1, eps2, jump_uniforms, jump_normals)]
    if (params.ndim != 2 or params.shape[1] != N_PARAMS
            or v_start.shape != (len(params),) or not len(params)
            or inputs[0].ndim != 2 or not inputs[0].shape[0]
            or any(a.shape != inputs[0].shape for a in inputs)
            or inputs[0].shape[1] != len(params)):
        raise ValueError("daily inputs require params [batch,8], state [batch], and shocks [time,batch]")
    if (not np.isfinite(params).all() or not np.isfinite(v_start).all()
            or any(not np.isfinite(a).all() for a in inputs)
            or np.any(v_start < 0)):
        raise ValueError("daily inputs must be finite, with nonnegative starting variance")
    mu, theta, kappa, sigma_v, rho, lam, mu_j, sigma_j = (params[:, i] for i in range(N_PARAMS))
    if (np.any(theta <= 0) or np.any(kappa <= 0) or np.any(sigma_v <= 0)
            or np.any(np.abs(rho) >= 1) or np.any((lam < 0) | (lam > LAMBDA_MAX))
            or np.any(sigma_j <= 0)):
        raise ValueError("daily model parameters outside supported domains")
    eps1, eps2, jump_uniforms, jump_normals = inputs
    if np.any((jump_uniforms < 0) | (jump_uniforms >= 1)):
        raise ValueError("jump uniforms must lie in [0,1)")
    jumps = jump_uniforms < lam
    sizes = mu_j + sigma_j * jump_normals
    n_days, n = eps1.shape
    rho_bar = np.sqrt(1 - rho**2)
    returns = np.empty((n_days, n))
    variance = np.empty((n_days, n)) if retain_auxiliary else None
    v = v_start
    for t in range(n_days):
        sd = np.sqrt(v)
        returns[t] = mu + sd * eps1[t] + jumps[t] * sizes[t]
        v = np.maximum(v + kappa * (theta - v) + sigma_v * sd * (rho * eps1[t] + rho_bar * eps2[t]), V_MIN)
        if retain_auxiliary:
            variance[t] = v
    if not np.isfinite(returns).all() or not np.isfinite(v).all():
        raise ValueError("nonfinite daily model trajectory; failed inputs are not redrawn")
    result = {"returns": returns.T, "terminal_state": v[:, None]}
    if retain_auxiliary:
        result.update(variance=variance.T, jumps=jumps.T)
    return result


def simulate_from_coordinates(coords, rng, n_days=WINDOW):
    """One series per coordinate vector, using the declared Gamma starting law.

    Returns a dict with the returns (n, n_days), the nine targets (n, 9), the variance path and the jump days.
    """
    coords = np.atleast_2d(coords)
    returns, variance, jumps = run_daily_model(model_params(coords), initial_variance(coords, rng), n_days, rng)
    targets = np.concatenate([coords, log_spot_vol(variance[:, -1])[:, None]], axis=1)
    return {"returns": returns, "targets": targets, "variance": variance, "jumps": jumps}


def simulate(n, seed=None, prior=PRIOR, n_days=WINDOW):
    rng = np.random.default_rng(seed)
    return simulate_from_coordinates(draw_coordinates(n, rng, prior), rng, n_days)


def network_input(returns):
    """The two input channels per day: the return in % and log(y² + 0.01). Shape (..., T, 2), float32."""
    returns = np.asarray(returns, dtype=float)
    return np.stack([returns, np.log(returns**2 + 0.01)], axis=-1).astype("float32")


def forward_returns(targets, horizon, n_paths, rng):
    """Simulate n_paths future paths of `horizon` days from each posterior draw (theta, v_T).

    targets: (m, 9) draws in target coordinates. Returns daily returns of shape (m, n_paths, horizon).
    """
    targets = np.atleast_2d(targets)
    params = np.repeat(model_params(targets), n_paths, axis=0)
    v_t = np.repeat(variance_from_log_spot_vol(targets[:, N_PARAMS]), n_paths)
    returns, _, _ = run_daily_model(params, v_t, horizon, rng)
    return returns.reshape(len(targets), n_paths, horizon)


class SVJSimulator:
    """BayesFlow-compatible simulator: fresh prior draws and 1,000-day series on every call."""

    def __init__(self, seed=None, prior=PRIOR, n_days=WINDOW):
        self.rng = np.random.default_rng(seed)
        self.prior = prior
        self.n_days = n_days

    def sample(self, batch_shape, **kwargs):
        n = int(np.prod(batch_shape))
        sims = simulate_from_coordinates(draw_coordinates(n, self.rng, self.prior), self.rng, self.n_days)
        return {"targets": sims["targets"].astype("float32"), "features": network_input(sims["returns"])}


def training_batch(n, seed=None, prior=PRIOR, n_days=WINDOW):
    """A fixed simulated set in the simulator's output format (validation and held-out data)."""
    return SVJSimulator(seed, prior, n_days).sample((n,))


if __name__ == "__main__":  # self-check: pixi run svj-selfcheck
    sims = simulate(64, seed=0)
    assert sims["returns"].shape == (64, WINDOW) and sims["targets"].shape == (64, N_TARGETS)
    assert np.isfinite(sims["returns"]).all() and (sims["variance"] >= V_MIN).all()
    coords = draw_coordinates(10_000, np.random.default_rng(1))
    assert np.allclose(quantities(coords)[:, 4].mean(), -1 / 3, atol=0.02)
    print(prior_table().round(2).to_string())
    print("ok: SVJ simulator")
