"""Figures of Lecture 9: calibration as estimation.

Usage: python course/scripts/lecture09.py   (writes course/images/l09-*.png and prints the numbers quoted in the text)

The toy "smile" model of Sections 9.6-9.9 is defined here and reused by lecture10.py:
nine quotes y_j = rho*eta*x_j + 0.05*eta^2*x_j^2 + s_j*eps_j on x_j in [-1, 1], with wing noise
larger than the centre noise. The odd part identifies the product rho*eta, the even part eta^2.
"""

import json
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from common import COLORS, DAYS, DELTAS, GREY, ROOT, load_spy, plt, save, sf, tidy

REPORT = {}
ROOT_RESULTS = sf.__file__  # placeholder to keep the import used

# ---------------------------------------------------------------------------------------------
# The toy smile
# ---------------------------------------------------------------------------------------------
X = DELTAS / 2.0                       # nine "strikes" on [-1, 1]
S = 0.01 + 0.10 * X**2                 # quote noise: small at the centre, large in the wings
CURV = 0.05                            # weight of the curvature term
ETA_BOX = (0.5, 3.5)
RHO_BOX = (-0.99, -0.2)
TRUTH_EDGE = (1.0, -0.87)              # near the bound rho = -0.99 along rho*eta = -0.87
TRUTH_INSIDE = (2.0, -0.435)           # the same product, inside the box


def toy_model(eta, rho):
    eta, rho = np.asarray(eta, dtype=float), np.asarray(rho, dtype=float)
    return rho[..., None] * eta[..., None] * X + CURV * eta[..., None] ** 2 * X**2


def toy_data(truth, rng):
    return toy_model(np.array(truth[0]), np.array(truth[1])) + S * rng.standard_normal(len(X))


def toy_chi2(y, eta, rho):
    return np.sum(((y - toy_model(eta, rho)) / S) ** 2, axis=-1)


def toy_jacobian(eta, rho):
    """d f_j / d(eta, rho), shape [9, 2]."""
    return np.column_stack([rho * X + 2 * CURV * eta * X**2, eta * X])


def toy_fit(y, start=(2.0, -0.6)):
    r = least_squares(lambda t: (y - toy_model(np.array(t[0]), np.array(t[1]))) / S, start,
                      bounds=([ETA_BOX[0], RHO_BOX[0]], [ETA_BOX[1], RHO_BOX[1]]), xtol=1e-12, ftol=1e-12)
    return r.x, 2 * r.cost


def toy_gn_cov(eta, rho):
    J = toy_jacobian(eta, rho) / S[:, None]
    info = J.T @ J
    return np.linalg.inv(info), info


def levenberg_marquardt(y, start, lam0=1e-2, max_iter=200, rtol=1e-4):
    """Plain Levenberg-Marquardt that records its iterates; it stops when the objective improves by
    less than rtol (relative) in one step, the usual practical stopping rule."""
    theta = np.array(start, dtype=float)
    lam = lam0
    path = [theta.copy()]
    f = toy_chi2(y, theta[0], theta[1])
    for _ in range(max_iter):
        r = (y - toy_model(np.array(theta[0]), np.array(theta[1]))) / S
        J = toy_jacobian(*theta) / S[:, None]
        A = J.T @ J
        g = J.T @ r
        step = np.linalg.solve(A + lam * np.diag(np.diag(A)), g)
        new = np.clip(theta + step, [ETA_BOX[0], RHO_BOX[0]], [ETA_BOX[1], RHO_BOX[1]])
        f_new = toy_chi2(y, new[0], new[1])
        if f_new < f:
            improvement = (f - f_new) / f
            theta, f, lam = new, f_new, lam / 3
            path.append(theta.copy())
            if improvement < rtol:
                break
        else:
            lam *= 4
    return np.array(path), f


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------
def fig_observation_model(spy):
    prices = np.array(spy["prices_forward_units"]).reshape(len(DAYS), len(DELTAS))
    sigma = np.array(sf.quote_noise(sf.design_grid())).reshape(len(DAYS), len(DELTAS))
    snr = prices / sigma
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    for color, j in zip(COLORS, range(len(DAYS))):
        ax.semilogy(DELTAS, snr[j], "o-", color=color, ms=3.5, label=f"{DAYS[j]} d")
    ax.axhline(1, color="0.5", ls=":", lw=1)
    ax.set_xlabel(r"standardized moneyness $\delta$")
    ax.set_ylabel(r"price over noise, $P_c / \sigma_c$")
    ax.set_title("SPY price in units of its quote noise")
    ax.legend(frameon=False, fontsize=8, ncol=2, title="maturity", title_fontsize=8)
    ax = axes[1]
    for color, j in zip(COLORS, range(len(DAYS))):
        ax.plot(DELTAS, norm.cdf(-snr[j]) * 100, "o-", color=color, ms=3.5, label=f"{DAYS[j]} d")
    ax.set_xlabel(r"standardized moneyness $\delta$")
    ax.set_ylabel("probability of a negative noisy quote, %")
    ax.set_title(r"$\mathbb{P}(y_c < 0) = N(-P_c/\sigma_c)$ under the noise model")
    tidy(*axes)
    save(fig, "l09-observation-model.png")
    REPORT["snr"] = dict(min=float(snr.min()), argmin=[int(DAYS[np.unravel_index(snr.argmin(), snr.shape)[0]]),
                                                       float(DELTAS[np.unravel_index(snr.argmin(), snr.shape)[1]])],
                         max=float(snr.max()), p_negative_max=float(norm.cdf(-snr.min())),
                         count_snr_below_3=int((snr < 3).sum()),
                         count_snr_below_3_calls=int((snr[:, DELTAS > 0] < 3).sum()),
                         snr_1w_call2=float(snr[0, -1]), snr_1w_put2=float(snr[0, 0]), snr_1y_atm=float(snr[-1, 4]))


def bs_grid_prices(sig):
    out = []
    for j, day in enumerate(DAYS):
        T = day / 252
        k = DELTAS * sf.SIGMA_REF * np.sqrt(T)
        out.append(sf.bs_price(k, T, sig, np.where(DELTAS <= 0, "put", "call")))
    return np.concatenate(out)


def bs_grid_vega(sig):
    out = []
    for j, day in enumerate(DAYS):
        T = day / 252
        out.append(sf.bs_vega(DELTAS * sf.SIGMA_REF * np.sqrt(T), T, sig))
    return np.concatenate(out)


def fit_sigma(y, w, lo=0.03, hi=0.8):
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda s: np.sum(w * (y - bs_grid_prices(s)) ** 2), bounds=(lo, hi), method="bounded",
                          options=dict(xatol=1e-10))
    return float(res.x)


def fig_bs_ml():
    sd = np.array(sf.quote_noise(sf.design_grid()))
    truth = 0.20
    rng = np.random.default_rng(21)
    y = bs_grid_prices(truth) + sd * rng.standard_normal(63)
    w = 1 / sd**2
    s_hat = fit_sigma(y, w)
    vega = bs_grid_vega(s_hat)
    info = np.sum(vega**2 / sd**2)
    se = 1 / np.sqrt(info)
    grid = np.linspace(s_hat - 5 * se, s_hat + 5 * se, 301)
    chi2 = np.array([np.sum(((y - bs_grid_prices(s)) / sd) ** 2) for s in grid])
    chi2_min = np.sum(((y - bs_grid_prices(s_hat)) / sd) ** 2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot((grid - truth) * 100, chi2 - chi2_min, color=COLORS[0], lw=2, label=r"exact $\chi^2(\sigma) - \chi^2_{\min}$")
    ax.plot((grid - truth) * 100, info * (grid - s_hat) ** 2, color=COLORS[1], ls="--",
            label=r"quadratic: $(\sigma - \hat\sigma)^2\,J^\top W J$")
    ax.axhline(norm.ppf(0.95) ** 2, color="0.5", ls=":", lw=1)
    ax.text(-0.37, norm.ppf(0.95) ** 2 + 0.3, r"$1.645^2$: 90% interval", fontsize=8.5, color="0.4")
    ax.set_xlabel(r"$\sigma$ minus the true 20%, volatility points")
    ax.set_ylabel(r"increase of the objective")
    ax.set_title("One synthetic smile: objective and its quadratic approximation")
    ax.set_ylim(0, 25)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center")
    REPORT["bs_ml"] = dict(s_hat=s_hat, se=float(se), chi2_min=float(chi2_min), dof=62)

    ax = axes[1]
    cover, hats, ses, chis = 0, [], [], []
    n_rep = 2000
    for i in range(n_rep):
        yy = bs_grid_prices(truth) + sd * rng.standard_normal(63)
        sh = fit_sigma(yy, w, lo=0.15, hi=0.25)
        e = 1 / np.sqrt(np.sum(bs_grid_vega(sh) ** 2 / sd**2))
        hats.append(sh)
        ses.append(e)
        chis.append(np.sum(((yy - bs_grid_prices(sh)) / sd) ** 2))
        cover += abs(sh - truth) <= norm.ppf(0.95) * e
    hats, ses = np.array(hats), np.array(ses)
    ax.hist((hats - truth) / ses, bins=60, density=True, color=COLORS[2], alpha=0.7,
            label=r"$(\hat\sigma - \sigma^*) / \widehat{\mathrm{se}}$, 2,000 smiles")
    xx = np.linspace(-4, 4, 200)
    ax.plot(xx, norm.pdf(xx), color="0.2", ls="--", label="standard normal")
    ax.set_xlabel("standardized estimation error")
    ax.set_ylabel("density")
    ax.set_title(f"Gauss–Newton 90% intervals cover the truth {cover / n_rep:.1%} of the time")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l09-black-scholes-least-squares.png")
    REPORT["bs_coverage"] = dict(coverage=cover / n_rep, se_median=float(np.median(ses)),
                                 sd_of_hat=float(hats.std()), chi2_mean=float(np.mean(chis)))


def fig_objectives(spy):
    y = np.array(spy["prices_forward_units"])
    sd = np.array(spy["noise_sd"])
    iv = []
    for j, day in enumerate(DAYS):
        T = day / 252
        iv.append(sf.implied_vol(y[9 * j:9 * j + 9], DELTAS * 0.2 * np.sqrt(T), T, np.where(DELTAS <= 0, "put", "call")))
    iv = np.concatenate(iv)
    from scipy.optimize import minimize_scalar
    objectives = {}
    objectives["price, weights $1/\\sigma_c^2$\n(maximum likelihood)"] = fit_sigma(y, 1 / sd**2)
    objectives["price, equal weights"] = fit_sigma(y, np.ones(63))
    objectives["price, weights $1/\\mathrm{vega}_c(20\\%)^2$\n(≈ implied vol, equal)"] = fit_sigma(y, 1 / bs_grid_vega(0.2) ** 2)
    objectives["implied vol, equal weights"] = float(np.mean(iv))
    s_iv = sf.implied_vol_noise(np.tile(DELTAS, len(DAYS)))
    objectives["implied vol, weights $1/s_{iv}^2$"] = float(np.sum(iv / s_iv**2) / np.sum(1 / s_iv**2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axes[0]
    names = list(objectives)
    vals = [objectives[n] * 100 for n in names]
    ax.barh(range(len(names)), vals, color=COLORS[:len(names)])
    for i, v in enumerate(vals):
        ax.text(v + 0.1, i, f"{v:.2f}%", va="center", fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlim(10, 22)
    ax.invert_yaxis()
    ax.set_xlabel(r"fitted Black–Scholes $\hat\sigma$, %")
    ax.set_title("SPY, one $\\sigma$ for all 63 quotes: the weights choose the answer")
    ax = axes[1]
    weights = {"ML $1/\\sigma_c^2$": 1 / sd**2, "price, equal": np.ones(63), "$1/\\mathrm{vega}(20\\%)^2$": 1 / bs_grid_vega(0.2) ** 2}
    bottom = np.zeros(len(weights))
    shares = {}
    for color, j in zip(COLORS, range(len(DAYS))):
        part = []
        for name, wgt in weights.items():
            # weight of maturity j in the objective's curvature at the fit, d2/dsigma2 sum w (y - P)^2 ~ sum w vega^2
            v = bs_grid_vega(0.145)
            tot = np.sum(wgt * v**2)
            part.append(np.sum((wgt * v**2)[9 * j:9 * j + 9]) / tot)
        part = np.array(part)
        ax.bar(range(len(weights)), part * 100, bottom=bottom * 100, color=color, label=f"{DAYS[j]} d")
        bottom += part
        shares[int(DAYS[j])] = [float(p) for p in part]
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels(list(weights), fontsize=8.5)
    ax.set_ylabel(r"share of the information about $\sigma$, %")
    ax.set_title("Which maturities each objective listens to")
    ax.legend(frameon=False, fontsize=8, title="maturity", title_fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    tidy(*axes)
    save(fig, "l09-objectives.png")
    REPORT["objectives"] = {n.replace("\n", " "): float(v) for n, v in objectives.items()}
    # exact identity: (y - P(s)) / sigma_c = [secant vega / vega(20%)] (iv - s) / s_iv, secant = (y - P(s)) / (iv - s)
    s_ml = objectives["price, weights $1/\\sigma_c^2$\n(maximum likelihood)"]
    secant = (y - bs_grid_prices(s_ml)) / (iv - s_ml)
    v20, v_ml = bs_grid_vega(0.2), bs_grid_vega(s_ml)
    REPORT["objective_conversion"] = dict(
        sigma=float(s_ml), factor_min=float(np.min(secant / v20)), factor_max=float(np.max(secant / v20)),
        vega_ratio_at_delta2=float(np.max(v20 / v_ml)), secant_over_vega_min=float(np.min(secant / v_ml)),
        secant_over_vega_max=float(np.max(secant / v_ml)), iv_min=float(np.min(iv)), iv_max=float(np.max(iv)))

    # the two parts of the factor, one at a time, in the implied-volatility objective
    def iv_fit(weight):
        res = minimize_scalar(lambda x: np.sum(weight(x) * (iv - x) ** 2), bounds=(0.10, 0.20), method="bounded",
                              options=dict(xatol=1e-9))
        return float(res.x)
    sec = lambda x: (y - bs_grid_prices(x)) / (iv - x)
    REPORT["objective_conversion"].update(
        exact_identity_fit=iv_fit(lambda x: (sec(x) / v20) ** 2 / s_iv**2),
        reference_vega_only_fit=iv_fit(lambda x: (bs_grid_vega(x) / v20) ** 2 / s_iv**2),
        secant_only_fit=iv_fit(lambda x: (sec(x) / bs_grid_vega(x)) ** 2 / s_iv**2))
    REPORT["objective_shares"] = shares


def fig_toy_landscape():
    rng = np.random.default_rng(7)
    y = toy_data(TRUTH_EDGE, rng)
    eta = np.linspace(*ETA_BOX, 400)
    rho = np.linspace(-1.05, -0.15, 400)
    E, R = np.meshgrid(eta, rho)
    C = toy_chi2(y, E, R)
    best, chi_best = toy_fit(y)
    starts = [(3.2, -0.25), (3.0, -0.5), (2.5, -0.9), (1.5, -0.3), (0.6, -0.4), (2.0, -0.6), (1.2, -0.95),
              (3.4, -0.9), (0.8, -0.8), (2.8, -0.3)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    E2, R2 = np.meshgrid(np.linspace(0.0, 3.5, 400), np.linspace(-1.8, -0.15, 400))
    C2 = toy_chi2(y, E2, R2)
    for ax in axes:
        lv = chi_best + np.array([1, 2.3, 4.6, 9.2, 20, 50, 120])
        ax.contour(E2, R2, C2, levels=lv, colors="0.6", linewidths=0.8)
        ax.axhline(RHO_BOX[0], color=COLORS[3], ls="--", lw=1)
        ax.fill_between(eta, -1.05, RHO_BOX[0], color=COLORS[3], alpha=0.08)
        ax.plot(eta, TRUTH_EDGE[0] * TRUTH_EDGE[1] / eta, color=COLORS[4], ls=":", lw=1.2,
                label=r"$\rho\eta = -0.87$")
        ax.plot(*TRUTH_EDGE, "*", color="k", ms=11, label="truth")
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$\rho$")
        ax.set_ylim(-1.05, -0.15)
    ax = axes[0]
    ax.set_xlim(*ETA_BOX)
    ends, ends_chi = [], []
    for st in starts:
        path, fend = levenberg_marquardt(y, st)
        ax.plot(path[:, 0], path[:, 1], "-", color=COLORS[0], lw=0.8, alpha=0.8)
        ax.plot(*path[-1], "o", color=COLORS[1], ms=5)
        ends.append(path[-1])
        ends_chi.append(fend)
    ends = np.array(ends)
    ax.plot([], [], "o", color=COLORS[1], label="end points of the 10 starts")
    ax.set_title("Levenberg–Marquardt from ten starts: one optimum, a long valley")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax = axes[1]
    cov, _ = toy_gn_cov(*best)
    vals, vecs = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * np.pi, 200)
    r90 = np.sqrt(4.605)  # chi-square, 2 dof, 90%
    ell = best[:, None] + vecs @ (np.sqrt(vals)[:, None] * np.vstack([np.cos(t), np.sin(t)])) * r90
    ax.plot(ell[0], ell[1], color=COLORS[0], lw=2, label="Gauss–Newton 90% ellipse")
    inside = np.where((E2 >= ETA_BOX[0]) & (R2 >= RHO_BOX[0]), C2 - chi_best, np.inf)
    ax.contourf(E2, R2, inside, levels=[-1, 4.605], colors=[COLORS[1]], alpha=0.3)
    ax.fill_between(np.linspace(0.0, 3.5, 10), -1.8, RHO_BOX[0], color=COLORS[3], alpha=0.08)
    ax.axvline(ETA_BOX[0], color=COLORS[3], ls="--", lw=1)
    ax.plot([], [], color=COLORS[1], alpha=0.4, lw=8, label=r"exact region $\Delta\chi^2 \leq 4.6$ inside the box")
    ax.plot(*best, "o", color=COLORS[0], ms=6, label="least-squares optimum")
    ax.set_xlim(0.0, 3.5)
    ax.set_ylim(-1.8, -0.15)
    ax.text(2.0, -1.15, r"outside the box: $\rho < -0.99$", color=COLORS[3], fontsize=8.5)
    ax.set_title("The ellipse is symmetric, the valley is curved and cut")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    tidy(*axes)
    save(fig, "l09-toy-landscape.png")
    sd_gn = np.sqrt(np.diag(cov))
    REPORT["toy_landscape"] = dict(best=[float(v) for v in best], chi2_best=float(chi_best),
                                   gn_sd=[float(v) for v in sd_gn],
                                   gn_rho_interval=[float(best[1] - 1.645 * sd_gn[1]), float(best[1] + 1.645 * sd_gn[1])],
                                   gn_eta_interval=[float(best[0] - 1.645 * sd_gn[0]), float(best[0] + 1.645 * sd_gn[0])],
                                   ends_eta_range=[float(ends[:, 0].min()), float(ends[:, 0].max())],
                                   ends_rho_range=[float(ends[:, 1].min()), float(ends[:, 1].max())],
                                   ends_chi_excess_max=float(np.max(ends_chi) - chi_best),
                                   ell_min_rho=float(ell[1].min()), ell_eta_range=[float(ell[0].min()), float(ell[0].max())],
                                   exact_eta_range=[float(E2[inside <= 4.605].min()), float(E2[inside <= 4.605].max())],
                                   exact_rho_range=[float(R2[inside <= 4.605].min()), float(R2[inside <= 4.605].max())],
                                   condition=float(np.linalg.cond(np.linalg.inv(cov))),
                                   info_eigenvalues=[float(v) for v in np.linalg.eigvalsh(np.linalg.inv(cov))],
                                   at_truth=_sd_at_truth())


def _sd_at_truth():
    """Gauss-Newton standard deviations of eta, rho and rho*eta at the edge truth."""
    eta, rho = TRUTH_EDGE
    J = toy_jacobian(np.array(eta), np.array(rho))
    C = np.linalg.inv(J.T @ np.diag(1 / S**2) @ J)
    g = np.array([rho, eta])
    return dict(sd_eta=float(np.sqrt(C[0, 0])), sd_rho=float(np.sqrt(C[1, 1])), sd_product=float(np.sqrt(g @ C @ g)))


def gn_interval(y):
    best, chi = toy_fit(y)
    cov, info = toy_gn_cov(*best)
    sd = np.sqrt(np.diag(cov))
    return best, sd, chi


def fig_toy_coverage():
    rng = np.random.default_rng(7)
    n = 2000
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, truth in (("near the bound", TRUTH_EDGE), ("inside the box", TRUTH_INSIDE)):
        cov_e = cov_r = beyond = 0
        widths = []
        for i in range(n):
            y = toy_data(truth, rng)
            best, sd, _ = gn_interval(y)
            lo, hi = best - 1.645 * sd, best + 1.645 * sd
            cov_e += lo[0] <= truth[0] <= hi[0]
            cov_r += lo[1] <= truth[1] <= hi[1]
            beyond += lo[1] < -1.0
            widths.append(hi - lo)
        res[label] = dict(eta=cov_e / n, rho=cov_r / n, below_minus_one=beyond / n,
                          width_median=[float(v) for v in np.median(widths, axis=0)])
    # random truths over the box
    cov_e = cov_r = 0
    truths, hits = [], []
    for i in range(n):
        tr = (rng.uniform(*ETA_BOX), rng.uniform(*RHO_BOX))
        y = toy_data(tr, rng)
        best, sd, _ = gn_interval(y)
        he = abs(best[0] - tr[0]) <= 1.645 * sd[0]
        hr = abs(best[1] - tr[1]) <= 1.645 * sd[1]
        cov_e += he
        cov_r += hr
        truths.append(tr)
        hits.append((he, hr))
    truths, hits = np.array(truths), np.array(hits)
    res["truths from the box"] = dict(eta=cov_e / n, rho=cov_r / n)
    ax = axes[0]
    labels = list(res)
    xs = np.arange(len(labels))
    ax.bar(xs - 0.18, [res[l]["eta"] * 100 for l in labels], 0.36, color=COLORS[0], label=r"$\eta$")
    ax.bar(xs + 0.18, [res[l]["rho"] * 100 for l in labels], 0.36, color=COLORS[1], label=r"$\rho$")
    ax.axhline(90, color="0.3", ls="--", lw=1)
    ax.text(2.25, 92.5, "nominal 90%", fontsize=8.5, color="0.3")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("coverage of the Gauss–Newton 90% interval, %")
    ax.set_title("Toy smile, 2,000 data sets per case")
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax = axes[1]
    # coverage of eta as a function of the true rho*eta "valley position": true rho
    bins = np.linspace(*RHO_BOX, 9)
    idx = np.clip(np.digitize(truths[:, 1], bins) - 1, 0, len(bins) - 2)
    cen = 0.5 * (bins[1:] + bins[:-1])
    ax.plot(cen, [hits[idx == i, 0].mean() * 100 for i in range(len(cen))], "o-", color=COLORS[0], label=r"$\eta$")
    ax.plot(cen, [hits[idx == i, 1].mean() * 100 for i in range(len(cen))], "o-", color=COLORS[1], label=r"$\rho$")
    ax.axhline(90, color="0.3", ls="--", lw=1)
    ax.set_xlabel(r"true $\rho$ (truths drawn uniformly from the box)")
    ax.set_ylabel("coverage, %")
    ax.set_ylim(0, 105)
    ax.set_title("Coverage by the position of the truth in the box")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    tidy(*axes)
    save(fig, "l09-toy-coverage.png")
    REPORT["toy_coverage"] = res
    REPORT["toy_coverage_by_rho"] = dict(centres=[float(c) for c in cen],
                                         eta=[float(hits[idx == i, 0].mean()) for i in range(len(cen))],
                                         rho=[float(hits[idx == i, 1].mean()) for i in range(len(cen))])


def fig_cached_classical():
    base = str(ROOT / "tutorial" / "assets" / "results" / "calibration") + "/"
    heston = json.load(open(base + "heston/s6/classical.json"))
    rough = json.load(open(base + "rough-pilot/s6/classical.json"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    labels = [r"$v_0$", r"$\theta_H$", r"$\sigma_v$", r"$\rho$"]
    ratios = [[] for _ in labels]
    for s in heston["surfaces"]:
        if s["gauss_newton_90"] is None:
            continue
        for i in range(4):
            g = s["gauss_newton_90"][i]
            p = s["posterior_90"][i]
            ratios[i].append((g[1] - g[0]) / (p[1] - p[0]))
    bp = ax.boxplot(ratios, positions=range(4), widths=0.5, showfliers=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS[0])
        patch.set_alpha(0.35)
    ax.axhline(1, color="0.3", ls="--", lw=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("Gauss–Newton 90% width over posterior 90% width")
    n_int = len(ratios[0])
    ax.set_title(f"Heston, {n_int} test smiles with an interval")
    ax = axes[1]
    summ = heston["summary"]
    rs = rough["summary"]
    names = ["Heston\nGauss–Newton", "Heston\nposterior", "rough pilot\nGauss–Newton", "rough pilot\nposterior"]
    h_gn = list(summ["gauss_newton_coverage_90"].values())
    h_po = list(summ["posterior_coverage_90"].values())
    r_gn = list(rs["gauss_newton_coverage_90"].values())
    r_po = list(rs["posterior_coverage_90"].values())
    for i, (vals, name) in enumerate(zip((h_gn, h_po, r_gn, r_po), names)):
        ax.plot(np.full(len(vals), i) + np.linspace(-0.15, 0.15, len(vals)), np.array(vals) * 100, "o",
                color=COLORS[i % 2], ms=6)
    ax.axhline(90, color="0.3", ls="--", lw=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("90% coverage per parameter, %")
    ax.set_ylim(60, 105)
    ax.set_title("Coverage: 200 Heston smiles, 8 rough-pilot smiles")
    tidy(*axes)
    save(fig, "l09-cached-classical.png")
    REPORT["cached_classical"] = dict(
        heston_ratio_median=[float(np.median(r)) for r in ratios],
        heston_ratio_q10=[float(np.quantile(r, 0.1)) for r in ratios],
        heston_ratio_q90=[float(np.quantile(r, 0.9)) for r in ratios],
        heston_n_interval=n_int, heston_singular=summ["gauss_newton_singular"], heston_gn=h_gn, heston_post=h_po,
        heston_modes=summ["modes"], heston_on_bound=summ["best_on_bound_share"],
        rough_gn=r_gn, rough_post=r_po, rough_singular=rs["gauss_newton_singular"],
        rough_cpu=rs["cpu_seconds_per_surface_median"], heston_cpu=summ["cpu_seconds_per_surface_median"])


def fig_tikhonov():
    rng = np.random.default_rng(7)
    y = toy_data(TRUTH_EDGE, rng)
    ref = np.array([np.mean(ETA_BOX), np.mean(RHO_BOX)])
    scale = np.array([ETA_BOX[1] - ETA_BOX[0], RHO_BOX[1] - RHO_BOX[0]])
    lams = np.logspace(-3, 2, 40)
    path = []
    for lam in lams:
        r = least_squares(lambda t: np.r_[(y - toy_model(np.array(t[0]), np.array(t[1]))) / S,
                                          np.sqrt(lam) * (t - ref) / scale], [1.2, -0.7],
                          bounds=([ETA_BOX[0], RHO_BOX[0]], [ETA_BOX[1], RHO_BOX[1]]))
        path.append(r.x)
    path = np.array(path)
    base = str(ROOT / "tutorial" / "assets" / "results" / "calibration") + "/"
    heston = json.load(open(base + "heston/s6/classical.json"))
    s2102 = [s for s in heston["surfaces"] if s["test_row"] == 2102][0]
    pen = [t["penalty"] for t in s2102["tikhonov"]]
    xi = [t["theta"][2] for t in s2102["tikhonov"]]
    rh = [t["theta"][3] for t in s2102["tikhonov"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    eta = np.linspace(*ETA_BOX, 300)
    rho = np.linspace(-1.0, -0.15, 300)
    E, R = np.meshgrid(eta, rho)
    C = toy_chi2(y, E, R)
    ax.contour(E, R, C, levels=C.min() + np.array([1, 4.6, 9.2, 20, 50]), colors="0.6", linewidths=0.8)
    sc = ax.scatter(path[:, 0], path[:, 1], c=np.log10(lams), cmap="viridis", s=14, zorder=3)
    ax.plot(*ref, "s", color=COLORS[3], ms=7, label=r"reference point $\theta_{ref}$ (box centre)")
    ax.plot(*TRUTH_EDGE, "*", color="k", ms=11, label="truth")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$\log_{10}\lambda$")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("Toy smile: the regularized point slides along the valley")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax = axes[1]
    ax.semilogx(pen, rh, "o-", color=COLORS[1], label=r"$\rho$")
    ax.semilogx(pen, xi, "o-", color=COLORS[0], label=r"$\sigma_v$")
    ax.semilogx(pen, np.array(xi) * np.array(rh), "s--", color=COLORS[3], label=r"$\rho\,\sigma_v$")
    ax.axhline(s2102["truth"][3], color=COLORS[1], ls=":", lw=1)
    ax.axhline(s2102["truth"][2], color=COLORS[0], ls=":", lw=1)
    ax.set_xlabel(r"penalty $\lambda$")
    ax.set_title("Heston test smile 2,102 (cached S6): the product stays")
    ax.legend(frameon=False, fontsize=9)
    tidy(*axes)
    save(fig, "l09-tikhonov.png")
    REPORT["tikhonov"] = dict(toy_at=[[float(l), float(p[0]), float(p[1])] for l, p in zip(lams[::13], path[::13])],
                              toy_last=[float(lams[-1]), float(path[-1, 0]), float(path[-1, 1])],
                              heston_2102=dict(penalty=pen, vol_of_vol=xi, rho=rh,
                                               product=[float(a * b) for a, b in zip(xi, rh)],
                                               truth=s2102["truth"], best=s2102["best_theta"],
                                               on_bound=s2102["on_bound"]))


def fig_deep_calibration():
    """A small neural pricer for Heston (kappa = 2) on the 63 design contracts, then least squares with it."""
    from sklearn.neural_network import MLPRegressor
    from sbi4finance.calibration import pricing, tasks
    task = tasks.heston_task()
    C = task["context"]["contracts"]
    sd = tasks.sigma(task)
    F = task["fixed_parameters"]
    lo = np.array([0.01, 0.01, 0.1, -0.95])
    hi = np.array([0.16, 0.16, 1.5, -0.1])
    # keep away from the corner of the box where the first quadrature rung is not accurate
    lo_d = lo + 0.05 * (hi - lo)
    hi_d = hi - 0.05 * (hi - lo)

    def surface(t):
        params = dict(F, theta=t[1], vol_of_vol=t[2], rho=t[3])
        try:
            return pricing.heston_surface(params, t[0], C, nodes=1024, upper_limit=600)
        except ValueError:
            return tasks.heston_prices(task, params, t[0])[0]  # the repository's quadrature ladder

    rng = np.random.default_rng(11)
    n_train, n_test = 3000, 300
    t0 = time.time()
    thetas = rng.uniform(lo_d, hi_d, (n_train + n_test, 4))
    Y = np.array([surface(t) for t in thetas]) / sd
    t_bank = time.time() - t0
    Z = (thetas - lo) / (hi - lo)
    mu, scale = Y[:n_train].mean(axis=0), Y[:n_train].std(axis=0)
    t0 = time.time()
    net = MLPRegressor(hidden_layer_sizes=(128, 128, 128), activation="tanh", max_iter=800, tol=1e-7,
                       learning_rate_init=2e-3, batch_size=128, random_state=0, n_iter_no_change=40)
    net.fit(Z[:n_train], (Y[:n_train] - mu) / scale)
    t_train = time.time() - t0
    pred = net.predict(Z[n_train:]) * scale + mu
    err = pred - Y[n_train:]  # in units of sigma_c
    rms_by_contract = np.sqrt(np.mean(err**2, axis=0))

    # least squares on noisy test smiles: exact pricer against neural pricer
    def fit(y, pricer, start):
        r = least_squares(lambda t: pricer(t) - y, start, bounds=(lo, hi), x_scale=hi - lo)
        return r.x

    exact = lambda t: surface(t) / sd
    neural = lambda t: net.predict(((np.asarray(t) - lo) / (hi - lo))[None, :])[0] * scale + mu
    diffs, gn_sds = [], []
    for i in range(25):
        tr = thetas[n_train + i]
        y = Y[n_train + i] + rng.standard_normal(63)
        a = fit(y, exact, tr)
        b = fit(y, neural, tr)
        J = np.empty((63, 4))
        for j in range(4):
            h = 1e-5 * (hi[j] - lo[j])
            e = np.zeros(4)
            e[j] = h
            J[:, j] = (exact(a + e) - exact(a - e)) / (2 * h)
        cov = np.linalg.inv(J.T @ J)
        gn_sds.append(np.sqrt(np.diag(cov)))
        diffs.append((b - a) / np.sqrt(np.diag(cov)))
    diffs = np.array(diffs)

    # the one-step inverse map: quotes -> parameters, a point estimate
    inv = MLPRegressor(hidden_layer_sizes=(128, 128), activation="tanh", max_iter=600, random_state=1,
                       learning_rate_init=2e-3, batch_size=128, n_iter_no_change=30, tol=1e-7)
    noisy = Y[:n_train] + rng.standard_normal(Y[:n_train].shape)
    inv.fit((noisy - mu) / scale, Z[:n_train])
    noisy_test = Y[n_train:] + rng.standard_normal(Y[n_train:].shape)
    z_hat = inv.predict((noisy_test - mu) / scale)
    th_hat = lo + np.clip(z_hat, 0, 1) * (hi - lo)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    for color, j in zip(COLORS, range(len(DAYS))):
        ax.plot(DELTAS, rms_by_contract[9 * j:9 * j + 9], "o-", color=color, ms=3.5, label=f"{DAYS[j]} d")
    ax.set_xlabel(r"standardized moneyness $\delta$")
    ax.set_ylabel(r"held-out RMS error, units of $\sigma_c$")
    ax.set_title(f"A neural pricer trained on {n_train:,} Heston surfaces")
    ax.set_ylim(0, 0.32)
    ax.legend(frameon=False, fontsize=8, ncol=4, title="maturity", title_fontsize=8, loc="upper center")
    ax = axes[1]
    labels = [r"$v_0$", r"$\theta_H$", r"$\sigma_v$", r"$\rho$"]
    ax.boxplot([diffs[:, i] for i in range(4)], positions=range(4), widths=0.5)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("neural minus exact least-squares fit,\nin Gauss–Newton standard deviations")
    ax.set_title("Deep calibration on 25 noisy test smiles")
    ax = axes[2]
    ax.plot(thetas[n_train:, 2], th_hat[:, 2], "o", color=COLORS[0], ms=3, alpha=0.6)
    ax.plot([0.1, 1.5], [0.1, 1.5], color="0.4", ls="--", lw=1)
    ax.set_xlabel(r"true $\sigma_v$")
    ax.set_ylabel(r"one-step estimate of $\sigma_v$")
    ax.set_title("One-step inverse map: a point, no interval")
    tidy(*axes)
    save(fig, "l09-deep-calibration.png")
    REPORT["deep"] = dict(n_train=n_train, n_test=n_test, box_shrink_per_edge=float((lo_d[0] - lo[0]) / (hi[0] - lo[0])), bank_seconds=t_bank, train_seconds=t_train, epochs=int(net.n_iter_),
                          rms_median=float(np.median(rms_by_contract)), rms_max=float(rms_by_contract.max()),
                          abs_q99=float(np.quantile(np.abs(err), 0.99)),
                          diff_median_abs=[float(v) for v in np.median(np.abs(diffs), axis=0)],
                          diff_max_abs=[float(v) for v in np.max(np.abs(diffs), axis=0)],
                          inverse_rmse=[float(v) for v in np.sqrt(np.mean((th_hat - thetas[n_train:]) ** 2, axis=0))],
                          gn_sd_median=[float(v) for v in np.median(gn_sds, axis=0)])


def main():
    spy = load_spy()
    fig_observation_model(spy)
    fig_bs_ml()
    fig_objectives(spy)
    fig_toy_landscape()
    fig_toy_coverage()
    fig_cached_classical()
    fig_tikhonov()
    fig_deep_calibration()
    print(json.dumps(REPORT, indent=1, default=float))


if __name__ == "__main__":
    main()
