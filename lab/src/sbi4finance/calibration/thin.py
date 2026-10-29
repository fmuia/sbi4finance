"""The thin-chain posterior (study S2-T): rough-Bergomi parameters and the forward-variance curve from a thin chain.

A thin layout is a fixed subset of the 63 design contracts, with its own quote noise: the contracts a real thin
chain quotes, and the half-spreads it quotes them with (configs/thin/*.json, built by
lab/checks/thin_chains/build_layout.py). Unlike the liquid network of inference.py, which is given
the forward-variance curve read from the chain, this network infers the curve: a thin chain's strip does not
support a replicated curve, and giving the estimator a curve read from another, fuller chain is the leak this
module exists to avoid.

Conditions: each layout quote over its noise sd times a noise scale s, y_c / (s sigma_c), and log s. Training
draws s log-uniformly on scale_bounds for every surface of every batch, so the same network serves the chain's
own noise (s = 1) and a sensitivity to it. Targets: (H, eta, rho) and the log forward-variance levels, each
constrained by a logit to its prior support (the box, and the curve prior's vol_bounds).

The prior density of the curve (needed by the importance-sampling audit) is curve_log_prior: the power law in log
time times a Gaussian random walk of tasks.draw_context, with sigma0 fixed by the first level and the exponent b
integrated out on a grid. The truncation to vol_bounds only cuts the support, so its normalizing constant cancels
in self-normalized weights; the defensive mixture of the proposal, where it does not cancel, divides by it
(curve_prior_mass).
"""

import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit, logsumexp
from scipy.stats import multivariate_t

from ..gbm_bank import _integer
from . import inference, pricing, tasks

LAYOUT_FORMAT = "thin_layout_v1"
LAYOUT_KEYS = {"format", "ticker", "asof", "source", "contract_indices", "noise_sd", "half_width_vol",
               "maturity_days", "delta", "expiry_map", "forward", "task_fingerprint"}
_TAG = 0x5448494E  # "THIN"
SCALE_BOUNDS = (0.5, 2.0)
EXPONENT_NODES = 401


def load_layout(path, task=None):
    layout = json.loads(Path(path).read_text())
    validate_layout(layout, task)
    return layout


def validate_layout(layout, task=None):
    if not isinstance(layout, dict) or set(layout) != LAYOUT_KEYS or layout["format"] != LAYOUT_FORMAT:
        raise ValueError(f"a thin layout needs exactly {sorted(LAYOUT_KEYS)}")
    idx = np.asarray(layout["contract_indices"])
    noise = np.asarray(layout["noise_sd"], dtype=float)
    if idx.ndim != 1 or not len(idx) or len(set(idx.tolist())) != len(idx) or noise.shape != idx.shape \
            or not np.all(np.isfinite(noise)) or np.any(noise <= 0):
        raise ValueError("unique contract indices and one positive noise sd each required")
    if task is not None:
        if np.any(idx < 0) or np.any(idx >= len(task["context"]["contracts"])):
            raise ValueError("layout indices outside the task's design grid")
        if np.any(noise < tasks.sigma(task)[idx] * (1 - 1e-9)):
            raise ValueError("a thin layout's noise cannot be below the design noise")


def curve_names(task):
    return [f"log_xi_{j + 1}" for j in range(tasks.context_size(task))]


def target_names(task):
    return list(task["parameter_names"]) + curve_names(task)


def curve_prior(task):
    prior = task["context_prior"]["forward_variance"]
    if tasks.curve_kind(prior) != "power_law_walk":
        raise ValueError("the thin network needs the power_law_walk curve prior")
    return prior


def target_bounds(task):
    """Support of every target: the parameter box, then 2 log vol_bounds for each log level."""
    lo, hi = (2 * np.log(v) for v in curve_prior(task)["vol_bounds"])
    curve = np.tile([lo, hi], (tasks.context_size(task), 1))
    return np.vstack([np.asarray(task["prior_bounds"], dtype=float), curve])


def curve_log_prior(log_levels, task):
    """log p(log xi) up to a constant, rows of [N, K]; -inf outside vol_bounds or the level bounds.

    log xi_1 = 2 log sigma0 fixes sigma0 (log-uniform: a constant density in log xi_1 on its range). Given b,
    the increments of log xi_j - log xi_1 - b x_j are independent normals with variance walk_sd^2 dx; b is
    uniform on exponent_bounds and integrated out on EXPONENT_NODES points (trapezoid rule).
    """
    prior = curve_prior(task)
    ell = np.atleast_2d(np.asarray(log_levels, dtype=float))
    mid = tasks.interval_midpoints(task)
    x = np.log(mid) - np.log(mid[0])
    dx = np.diff(x)
    rel = ell[:, 1:] - ell[:, :1]  # [N, K-1]
    b = np.linspace(*prior["exponent_bounds"], EXPONENT_NODES)
    increments = np.diff(np.concatenate([np.zeros((len(ell), 1)), rel], axis=1), axis=1)  # [N, K-1]
    resid = increments[:, :, None] - b[None, None, :] * dx[None, :, None]  # [N, K-1, nodes]
    sd2 = prior["walk_sd"] ** 2 * dx
    log_n = -0.5 * np.sum(resid**2 / sd2[None, :, None] + np.log(2 * np.pi * sd2)[None, :, None], axis=1)
    weights = np.full(EXPONENT_NODES, 1.0)
    weights[[0, -1]] = 0.5
    log_mix = logsumexp(log_n + np.log(weights)[None, :], axis=1) + np.log(b[1] - b[0]) \
        - np.log(prior["exponent_bounds"][1] - prior["exponent_bounds"][0])
    lo_l, hi_l = (2 * np.log(v) for v in prior["level_bounds"])
    lo_v, hi_v = (2 * np.log(v) for v in prior["vol_bounds"])
    inside = (ell[:, 0] >= lo_l) & (ell[:, 0] <= hi_l) & np.all((ell >= lo_v) & (ell <= hi_v), axis=1)
    return np.where(inside, log_mix - np.log(hi_l - lo_l), -np.inf)


_CURVE_MASS = {}


def curve_prior_mass(task, draws=400_000):
    """Z, the probability that an untruncated power_law_walk curve has every level inside vol_bounds: the
    acceptance rate of tasks.draw_context. The truncated prior density is exp(curve_log_prior) / Z. Z cancels in
    self-normalized weights, but not where the prior is mixed with another density (Proposal.log_density).
    Estimated once per prior on a fixed seed; its relative standard error is about 0.1%."""
    prior = curve_prior(task)
    mid = tasks.interval_midpoints(task)
    key = json.dumps(prior, sort_keys=True) + json.dumps(np.asarray(mid, dtype=float).tolist())
    if key not in _CURVE_MASS:
        rng = np.random.default_rng(np.random.SeedSequence([_TAG, 10]))
        x = np.log(mid) - np.log(mid[0])
        level = np.exp(rng.uniform(*np.log(prior["level_bounds"]), draws))
        exponent = rng.uniform(*prior["exponent_bounds"], draws)
        steps = rng.standard_normal((draws, len(x) - 1)) * prior["walk_sd"] * np.sqrt(np.diff(x))
        log_vol = 0.5 * (2 * np.log(level)[:, None] + exponent[:, None] * x[None, :]
                         + np.concatenate([np.zeros((draws, 1)), np.cumsum(steps, axis=1)], axis=1))
        low, high = np.log(prior["vol_bounds"])
        _CURVE_MASS[key] = float(np.mean(np.all((log_vol >= low) & (log_vol <= high), axis=1)))
    return _CURVE_MASS[key]


def log_prior(targets, task):
    """log prior density of rows [N, P + K] in natural units: the uniform box times the curve prior, up to the
    truncation constant (curve_prior_mass)."""
    targets = np.atleast_2d(np.asarray(targets, dtype=float))
    p = len(task["parameter_names"])
    box = np.asarray(task["prior_bounds"], dtype=float)
    inside = np.all((targets[:, :p] > box[:, 0]) & (targets[:, :p] < box[:, 1]), axis=1)
    return np.where(inside, -np.sum(np.log(box[:, 1] - box[:, 0])) + curve_log_prior(targets[:, p:], task), -np.inf)


def draw_prior(task, rng, n):
    """n joint prior draws [n, P + K] of the parameters and the log levels."""
    box = np.asarray(task["prior_bounds"], dtype=float)
    theta = rng.uniform(box[:, 0], box[:, 1], (n, len(box)))
    curves = np.log(np.array([tasks.draw_context(task, rng)[0] for _ in range(n)]))
    return np.hstack([theta, curves])


# ----------------------------------------------------------------------------- training and sampling


def layout_arrays(layout):
    return np.asarray(layout["contract_indices"], dtype=int), np.asarray(layout["noise_sd"], dtype=float)


def thin_clean_data(split, task, layout):
    """Clean layout prices over their noise sd, with the targets one [B, 1] array per name; train with thin_noise."""
    idx, noise = layout_arrays(layout)
    clean = np.asarray(split["reference_prices"], dtype=np.float64)[:, idx] / noise
    targets = np.hstack([np.asarray(split["parameters"], dtype=np.float64),
                         np.log(np.asarray(split["context_values"], dtype=np.float64))])
    _check_targets(targets, task)
    return {"clean_scaled": clean, **{name: targets[:, i:i + 1] for i, name in enumerate(target_names(task))}}


def _check_targets(targets, task):
    bounds = target_bounds(task)
    if targets.shape[1] != len(bounds) or not np.isfinite(targets).all():
        raise ValueError("finite targets [batch, parameters + levels] required")
    if np.any(targets <= bounds[:, 0]) or np.any(targets >= bounds[:, 1]):
        raise ValueError("targets must lie strictly inside their support")


def _noisy(clean_scaled, rng, scale_bounds):
    n = len(clean_scaled)
    log_s = rng.uniform(np.log(scale_bounds[0]), np.log(scale_bounds[1]), (n, 1))
    return {"observations": clean_scaled / np.exp(log_s) + rng.standard_normal(clean_scaled.shape),
            "log_scale": log_s}


def thin_noise(seed, scale_bounds=SCALE_BOUNDS):
    """BayesFlow batch augmentation: a noise scale per surface and a fresh unit-normal error on every quote."""
    rng = np.random.default_rng(np.random.SeedSequence([_TAG, _integer(seed, "noise seed")]))

    def augment(batch):
        batch = dict(batch)
        batch.update(_noisy(batch.pop("clean_scaled"), rng, scale_bounds))
        return batch

    augment.rng = rng
    return augment


def thin_validation_data(split, task, layout, seed, scale_bounds=SCALE_BOUNDS):
    """The validation split with one fixed draw of the noise scale and the quote errors."""
    data = thin_clean_data(split, task, layout)
    rng = np.random.default_rng(np.random.SeedSequence([_TAG, 1, _integer(seed, "noise seed")]))
    data.update(_noisy(data.pop("clean_scaled"), rng, scale_bounds))
    return data


def thin_conditions(quotes, layout, scale=1.0):
    """Conditions for quotes [B, L] on the layout, priced in forward units, with noise scale s (scalar or [B])."""
    _, noise = layout_arrays(layout)
    quotes = np.atleast_2d(np.asarray(quotes, dtype=np.float64))
    if quotes.shape[1] != len(noise) or not np.isfinite(quotes).all():
        raise ValueError(f"finite quotes [batch, {len(noise)}] required")
    s = np.broadcast_to(np.asarray(scale, dtype=np.float64).reshape(-1, 1), (len(quotes), 1))
    if np.any(s <= 0):
        raise ValueError("the noise scale must be positive")
    return {"observations": quotes / (s * noise), "log_scale": np.log(s)}


def build_thin_workflow(task, checkpoint_dir=None, *, subnet_widths=(256, 256, 256, 256), optimizer=None,
                        time_power_law_alpha=0.5):
    """FlowMatching on logit-constrained targets (parameters and log levels); quotes and log s as conditions."""
    import bayesflow as bf

    names = target_names(task)
    adapter = bf.Adapter()
    for name, (lower, upper) in zip(names, target_bounds(task), strict=True):
        adapter = adapter.constrain(name, lower=float(lower), upper=float(upper), inclusive="none")
    adapter = (adapter.convert_dtype("float64", "float32")
               .concatenate(names, into="inference_variables")
               .concatenate(["observations", "log_scale"], into="inference_conditions"))
    return bf.BasicWorkflow(
        adapter=adapter,
        inference_network=bf.networks.FlowMatching(subnet_kwargs={"widths": list(subnet_widths)},
                                                   integrate_kwargs={"method": "euler",
                                                                     "steps": inference.EULER_STEPS},
                                                   time_power_law_alpha=time_power_law_alpha),
        standardize=["inference_variables", "inference_conditions"],
        optimizer=optimizer,
        checkpoint_filepath=None if checkpoint_dir is None else str(checkpoint_dir),
        save_weights_only=False, save_best_only=True,
    )


def sample_thin(workflow, task, layout, quotes, num_samples, scale=1.0, seed=0):
    """Posterior draws [B, N, P + K]: (H, eta, rho) then the log levels, in natural units."""
    conditions = thin_conditions(quotes, layout, scale)
    inference.use_sampler(workflow)
    drawn = workflow.sample(num_samples=num_samples, conditions=conditions, seed=int(seed))
    return np.concatenate([np.asarray(drawn[name], dtype=np.float64) for name in target_names(task)], axis=-1)


# ----------------------------------------------------------------------------- the audit in P + K dimensions


def to_unbounded(targets, bounds):
    return logit((np.asarray(targets, dtype=float) - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]))


def to_bounded(z, bounds):
    return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * expit(np.asarray(z, dtype=float))


def log_jacobian(z, bounds):
    z = np.asarray(z, dtype=float)
    return np.sum(np.log(bounds[:, 1] - bounds[:, 0]) - np.logaddexp(0.0, z) - np.logaddexp(0.0, -z), axis=-1)


class Proposal:
    """(1 - f) Student t fitted to draws in logit coordinates + f joint prior; draws and log densities in natural
    units. As correction.propose, with the joint prior of the parameters and the curve as the defensive part."""

    def __init__(self, draws, task, *, prior_fraction=0.1, df=5, inflation=1.2, weights=None, shrink_to=None):
        self.task, self.prior_fraction = task, float(prior_fraction)
        self.bounds = target_bounds(task)
        self.log_prior_mass = np.log(curve_prior_mass(task))
        width = self.bounds[:, 1] - self.bounds[:, 0]
        draws = np.clip(np.asarray(draws, dtype=float), self.bounds[:, 0] + 1e-9 * width,
                        self.bounds[:, 1] - 1e-9 * width)
        z = to_unbounded(draws, self.bounds)
        if z.ndim != 2 or len(z) < 2 or not np.isfinite(z).all():
            raise ValueError("at least two finite draws strictly inside the support required")
        w = np.full(len(z), 1.0 / len(z)) if weights is None else np.asarray(weights, dtype=float) / np.sum(weights)
        mean = w @ z
        unbias = 1.0 - np.sum(w**2)
        cov = (z - mean).T @ ((z - mean) * w[:, None]) / unbias if unbias > 1e-6 else np.zeros((z.shape[1],) * 2)
        if weights is not None:
            # Few effective points make the weighted covariance unreliable (one point makes it zero): shrink it, in
            # proportion to the effective sample size, towards shrink_to (the previous round's covariance) or, if
            # none is given, the unweighted spread of the points. The unweighted spread of pooled points includes
            # the defensive prior draws and is far too wide to adapt with.
            ess = 1.0 / np.sum(w**2)
            alpha = ess / (ess + z.shape[1])
            target = np.cov(z.T) if shrink_to is None else np.asarray(shrink_to, dtype=float)
            cov = alpha * cov + (1.0 - alpha) * target
        values, vectors = np.linalg.eigh(0.5 * (cov + cov.T))
        values = np.maximum(values, 1e-8 * max(values.max(), 1e-12) + 1e-10)
        cov = (vectors * values) @ vectors.T
        self.covariance = cov  # before the inflation; the next round shrinks towards it
        self.t = multivariate_t(loc=mean, shape=cov * inflation**2, df=df)

    def sample(self, num_samples, rng):
        from_prior = rng.random(num_samples) < self.prior_fraction
        z = np.clip(np.atleast_2d(self.t.rvs(size=num_samples, random_state=rng)), -30, 30)
        targets = to_bounded(z, self.bounds)
        if from_prior.any():
            targets[from_prior] = draw_prior(self.task, rng, int(from_prior.sum()))
        return targets, from_prior

    def log_density(self, targets):
        z = to_unbounded(targets, self.bounds)
        log_t = self.t.logpdf(z) - log_jacobian(z, self.bounds)
        return np.logaddexp(np.log1p(-self.prior_fraction) + log_t,
                            np.log(self.prior_fraction) + log_prior(targets, self.task) - self.log_prior_mass)


def propose(draws, task, num_samples, seed, *, prior_fraction=0.1, df=5, inflation=1.2):
    """IID proposals from Proposal(draws), with their log densities in natural units."""
    proposal = Proposal(draws, task, prior_fraction=prior_fraction, df=df, inflation=inflation)
    rng = np.random.default_rng(np.random.SeedSequence([_TAG, 2, int(seed)]))
    targets, from_prior = proposal.sample(num_samples, rng)
    return targets, proposal.log_density(targets), from_prior


def correct(log_likelihood, log_q, targets, task):
    """Self-normalized weights for prior x likelihood / proposal, with ESS and k-hat (correction.correct)."""
    from . import correction

    log_w = np.asarray(log_likelihood, dtype=float) + log_prior(targets, task) - np.asarray(log_q, dtype=float)
    if np.isnan(log_w).any() or not np.isfinite(log_w).any():
        raise ValueError("importance weights failed: no finite weight")
    log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
    weights = np.exp(log_w - logsumexp(log_w))
    ess = float(1.0 / np.sum(weights**2))
    return {"weights": weights, "ess": ess, "efficiency": ess / len(weights), "max_weight": float(weights.max()),
            "khat": correction.pareto_khat(log_w[np.isfinite(log_w)])}


def gaussian_loglik(observation, prices, noise, mc_variance=None, max_ratio=0.9):
    """correction.gaussian_loglik with an explicit noise sd per quote (the layout's, times the scale)."""
    sigma2 = np.asarray(noise, dtype=float) ** 2
    prices = np.atleast_2d(np.asarray(prices, dtype=float))
    variance = np.broadcast_to(sigma2, prices.shape)
    if mc_variance is not None:
        mc_variance = np.broadcast_to(np.asarray(mc_variance, dtype=float), prices.shape)
        if np.any(mc_variance < 0) or np.any(mc_variance > max_ratio**2 * sigma2):
            raise ValueError("Monte Carlo SE above max_ratio of the noise: price with more paths")
        variance = sigma2 - mc_variance
    y = np.asarray(observation, dtype=float).reshape(-1)
    return -0.5 * np.sum((prices - y) ** 2 / variance + np.log(2 * np.pi * variance), axis=-1)


# ----------------------------------------------------------------------------- pricing and the classical fit


def blocks_for(days):
    """Hybrid-scheme blocks for contracts maturing on the given trading days, grouped as in tasks.rough_task:
    up to 10 days and up to 63 days at 8 steps a day, the rest at 2 steps a day."""
    days = sorted(set(int(d) for d in days))
    blocks = []
    for limit, per_day in ((10, 8), (63, 8), (max(days), 2)):
        inside = [d for d in days if d <= limit and (not blocks or d > blocks[-1]["days"])]
        if inside:
            blocks.append({"days": max(inside), "n_steps": per_day * max(inside)})
    return [{"horizon": b["days"] / 252, "n_steps": b["n_steps"]} for b in blocks]


def layout_contracts(task, layout):
    idx, _ = layout_arrays(layout)
    return [task["context"]["contracts"][i] for i in idx]


def contract_days(contracts):
    return [int(round(c["maturity"] * 252)) for c in contracts]


def split_targets(task, row):
    p = len(task["parameter_names"])
    row = np.asarray(row, dtype=float)
    return tasks.model_parameters(task, row[:p]), tasks.curve(task, np.exp(row[p:]))


def price_contracts(task, row, contracts, noise, paths_ladder, seed, budget=0.25, cap=0.9, covariance=False):
    """Prices and standard errors of the contracts for one row of targets (parameters, log levels); with
    covariance=True also the covariance of the price means (contracts on the same paths have correlated errors)."""
    parameters, curve = split_targets(task, row)
    prices, errors, cov, _ = pricing.rough_surface(
        parameters, curve, contracts, blocks_for(contract_days(contracts)), list(paths_ladder), noise, budget,
        int(seed), cap=cap)
    return (prices, errors, cov) if covariance else (prices, errors)


def priced_for_audit(row, seed, *, task, contracts, noise, paths, max_doublings=4, budget=0.5):
    """One proposal priced for the audit on its own seed: (prices, covariance, paths) or (None, None, reason).

    correction.price_for_likelihood: a pilot pricing chooses the path count (paths times a power of two, up to
    2**max_doublings) at which every quote's Monte Carlo SE is at most budget times its noise sd, the covariance's
    largest eigenvalue relative to the noise at most correction.MAX_RELATIVE_EIGENVALUE (the estimate stays valid)
    and its trace at most correction.MAX_RELATIVE_TRACE (the estimate's noise does not swamp the weights); the
    prices come from an independent pricing at that count, with the pilot's covariance scaled to it, so neither
    the count nor the covariance depends on the prices that enter the likelihood.
    """
    from . import correction

    codes = {"pilot": 10, "final": 11}

    def price(n, purpose, attempt):
        stream = int(np.random.SeedSequence([_TAG, 9, int(seed), codes[purpose], int(attempt), int(n)])
                     .generate_state(1, dtype=np.uint32)[0])
        return price_contracts(task, row, contracts, noise, [n], stream, budget=np.inf, cap=None, covariance=True)

    return correction.price_for_likelihood(price, noise, paths, budget=budget, max_doublings=max_doublings)


def importance_rounds(draws, task, price_one, observation, noise, num_proposals, seed, *, max_rounds=1,
                      khat_target=0.7, map_fn=map, data_covariance=None):
    """Importance sampling from the network's draws, adapted over rounds and pooled with the balance heuristic.

    price_one(row, seed) returns (prices, covariance, info) or (None, None, reason); every proposal gets its own
    seed, so the Monte Carlo errors are independent across proposals and N(y; P_hat, diag(noise^2) - S) is an
    unbiased estimate of the likelihood (correction.gaussian_loglik_mc; data_covariance as there, the bank's Monte
    Carlo error for a simulated chain and None for a real one). Round r + 1 refits the Student t to the
    weighted points of rounds 1 to r; every point is weighted against the mixture of all rounds' proposals. Stops
    when k-hat is at most khat_target or after max_rounds. A proposal that cannot be priced gets zero weight and is
    counted: its weight should have been positive, so a large count makes the audit unreliable.
    """
    from . import correction

    proposals, samples, log_lik, rounds = [], [], [], []
    fit_draws, fit_weights, shrink_to = draws, None, None
    for r in range(max_rounds):
        proposal = Proposal(fit_draws, task, weights=fit_weights, shrink_to=shrink_to)
        rng = np.random.default_rng(np.random.SeedSequence([_TAG, 7, int(seed), r]))
        targets, _ = proposal.sample(num_proposals, rng)
        seeds = [int(np.random.SeedSequence([_TAG, 8, int(seed), r, j]).generate_state(1, dtype=np.uint32)[0])
                 for j in range(len(targets))]
        ll = np.full(len(targets), -np.inf)
        unpriced = []
        for i, (prices, cov, info) in enumerate(map_fn(price_one, targets, seeds)):
            if prices is None:
                unpriced.append(str(info))
            else:
                ll[i] = correction.gaussian_loglik_mc(observation, prices, noise, cov,
                                                      data_covariance=data_covariance)
        proposals.append(proposal)
        samples.append(targets)
        log_lik.append(ll)
        pooled = np.concatenate(samples)
        log_q = logsumexp(np.stack([np.log(len(x)) + q.log_density(pooled) for q, x in zip(proposals, samples)]),
                          axis=0) - np.log(len(pooled))
        result = correct(np.concatenate(log_lik), log_q, pooled, task)
        rounds.append({"round": r + 1, "proposals": int(len(pooled)), "ess": result["ess"],
                       "efficiency": result["efficiency"], "khat": result["khat"], "unpriced": len(unpriced),
                       "unpriced_reasons": unpriced[:3]})
        if result["khat"] <= khat_target:
            break
        fit_draws, fit_weights, shrink_to = pooled, result["weights"], proposal.covariance
    return {"targets": pooled, "weights": result["weights"], "ess": result["ess"], "efficiency": result["efficiency"],
            "khat": result["khat"], "rounds": rounds, "unpriced": int(sum(x["unpriced"] for x in rounds))}


def classical_curve_map(task, quoted_days):
    """For each of the task's curve intervals, the index of the quoted interval [T_(m-1), T_m] that holds its
    midpoint; intervals beyond the last quoted maturity take the last one (flat extrapolation)."""
    edges = np.array(sorted(set(int(d) for d in quoted_days)), dtype=float) / 252
    mid = tasks.interval_midpoints(task)
    return np.minimum(np.searchsorted(edges, mid, side="left"), len(edges) - 1)


def classical_rows(task, z, curve_map, fixed_hurst=None):
    """Targets (parameters, log levels) from classical coordinates z: logit of the free box parameters, then
    one log level per quoted interval."""
    box = np.asarray(task["prior_bounds"], dtype=float)
    free = [i for i, n in enumerate(task["parameter_names"]) if not (fixed_hurst is not None and n == "hurst")]
    theta = np.empty(len(box))
    theta[free] = box[free, 0] + (box[free, 1] - box[free, 0]) * expit(np.asarray(z[:len(free)], dtype=float))
    if fixed_hurst is not None:
        theta[task["parameter_names"].index("hurst")] = fixed_hurst
    levels = np.asarray(z[len(free):], dtype=float)[curve_map]
    return np.concatenate([theta, levels])


def classical_start(task, start_row, curve_map, fixed_hurst=None):
    box = np.asarray(task["prior_bounds"], dtype=float)
    p = len(box)
    free = [i for i, n in enumerate(task["parameter_names"]) if not (fixed_hurst is not None and n == "hurst")]
    row = np.asarray(start_row, dtype=float)
    unit = np.clip((row[free] - box[free, 0]) / (box[free, 1] - box[free, 0]), 1e-4, 1 - 1e-4)
    levels = [float(np.mean(row[p:][curve_map == m])) for m in range(curve_map.max() + 1)]
    return np.concatenate([logit(unit), levels])


def classical_fit(task, observation, contracts, noise, start_row, quoted_days, paths, seed, fixed_hurst=None,
                  max_nfev=200):
    """Least squares on the quotes with the classical curve (one level per quoted interval, flat beyond), the
    Monte Carlo pricer at a fixed seed (common random numbers), and the Gauss-Newton covariance of z. The levels
    stay inside the curve prior's vol_bounds and the logits inside (-12, 12). start_row must not come from a
    posterior: classical_screen gives a start from the quotes and the prior box alone."""
    from scipy.optimize import least_squares

    curve_map = classical_curve_map(task, quoted_days)
    y = np.asarray(observation, dtype=float)

    def residuals(z):
        try:
            prices, _ = price_contracts(task, classical_rows(task, z, curve_map, fixed_hurst), contracts, noise,
                                        [paths], seed, budget=np.inf, cap=None)
        except ValueError:  # a point the pricer cannot price: a large residual keeps the fit away from it
            return np.full(len(y), 1e3)
        return (prices - y) / noise

    z0 = classical_start(task, start_row, curve_map, fixed_hurst)
    free = len(z0) - (curve_map.max() + 1)
    low, high = 2 * np.log(np.asarray(curve_prior(task)["vol_bounds"], dtype=float))
    bounds = (np.r_[np.full(free, -12.0), np.full(len(z0) - free, low)],
              np.r_[np.full(free, 12.0), np.full(len(z0) - free, high)])
    z0 = np.clip(z0, bounds[0] + 1e-6, bounds[1] - 1e-6)
    result = least_squares(residuals, z0, method="trf", diff_step=1e-3, x_scale="jac", max_nfev=max_nfev,
                           bounds=bounds)
    jacobian = result.jac
    covariance = np.linalg.pinv(jacobian.T @ jacobian)
    return {"z": result.x, "row": classical_rows(task, result.x, curve_map, fixed_hurst), "chi2": float(2 * result.cost),
            "covariance": covariance, "nfev": int(result.nfev), "njev": int(result.njev or 0),
            "converged": bool(result.status > 0), "status": int(result.status), "curve_map": curve_map,
            "fixed_hurst": fixed_hurst}


def classical_screen(task, observation, contracts, noise, quoted_days, paths, seed, candidates=16):
    """A classical starting point: the lowest chi-square among the prior box's centre and candidates joint prior
    draws, each with its curve set flat per quoted interval. Uses the quotes and the prior support only."""
    curve_map = classical_curve_map(task, quoted_days)
    box = np.asarray(task["prior_bounds"], dtype=float)
    rng = np.random.default_rng(np.random.SeedSequence([_TAG, 10, int(seed)]))
    rows = draw_prior(task, rng, candidates)
    centre = np.concatenate([box.mean(axis=1), np.median(rows[:, len(box):], axis=0)])
    rows = np.vstack([centre, rows])
    y = np.asarray(observation, dtype=float)
    chi2 = []
    for row in rows:
        row = classical_rows(task, classical_start(task, row, curve_map), curve_map)
        prices, _ = price_contracts(task, row, contracts, noise, [paths], seed, budget=np.inf, cap=None)
        chi2.append(float(np.sum(((prices - y) / noise) ** 2)))
    order = np.argsort(chi2)
    return rows[order], np.asarray(chi2)[order]


def power_law_extension(task, row, quoted_days):
    """The row with every level beyond the last quoted maturity replaced by a power law in time fitted to the
    quoted intervals' levels (log level linear in the log midpoint): a stated extrapolation rule, against the flat
    one of classical_rows."""
    p = len(task["parameter_names"])
    row = np.asarray(row, dtype=float).copy()
    edges = np.array([0.0] + sorted(set(int(d) for d in quoted_days)), dtype=float) / 252
    curve_map = classical_curve_map(task, quoted_days)
    mid = tasks.interval_midpoints(task)
    beyond = mid > edges[-1]
    if not beyond.any():
        return row
    quoted_mid = 0.5 * (edges[:-1] + edges[1:])
    levels = np.array([row[p:][(curve_map == m) & ~beyond].mean() for m in range(len(quoted_mid))])
    slope, intercept = np.polyfit(np.log(quoted_mid), levels, 1)
    low, high = 2 * np.log(np.asarray(task["context_prior"]["forward_variance"]["vol_bounds"], dtype=float))
    row[p:][beyond] = np.clip(intercept + slope * np.log(mid[beyond]), low, high)
    return row


def delta_method(task, fit, product_fn, step=1e-2):
    """Product prices at the fit and their delta-method sds, by central differences in the classical coordinates."""
    z = fit["z"]
    base = product_fn(fit["row"])
    gradient = np.empty((len(base), len(z)))
    for j in range(len(z)):
        up, down = z.copy(), z.copy()
        up[j] += step
        down[j] -= step
        gradient[:, j] = (product_fn(classical_rows(task, up, fit["curve_map"], fit["fixed_hurst"]))
                          - product_fn(classical_rows(task, down, fit["curve_map"], fit["fixed_hurst"]))) / (2 * step)
    sd = np.sqrt(np.maximum(np.einsum("pi,ij,pj->p", gradient, fit["covariance"], gradient), 0.0))
    return base, sd
