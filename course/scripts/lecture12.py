"""Figures of Lecture 12: amortized inference.

Usage: python course/scripts/lecture12.py   (writes course/images/l12-*.png and prints the numbers quoted in the text)
"""

import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm

from common import COLORS, DAYS, DELTAS, GREY, ROOT, load_spy, plt, save, tidy

REPORT = {}
RESULTS = ROOT / "tutorial" / "assets" / "results" / "calibration"


def load(path):
    return json.loads((RESULTS / path).read_text())


# Figure 12.1: simulated pairs carry the posterior --------------------------------
ETA = (0.5, 3.5)
RHO = (-0.99, -0.2)
NOISE = 0.05


def exact_eta_marginal(y_obs, eta):
    """p(eta | y) for y = rho*eta + N(0, NOISE^2), uniform priors: integrate rho analytically."""
    upper = norm.cdf((RHO[1] * eta - y_obs) / NOISE)
    lower = norm.cdf((RHO[0] * eta - y_obs) / NOISE)
    dens = (upper - lower) / eta
    return dens / np.trapezoid(dens, eta)


def fig_joint():
    rng = np.random.default_rng(0)
    n = 400_000
    eta = rng.uniform(*ETA, n)
    rho = rng.uniform(*RHO, n)
    y = rho * eta + NOISE * rng.standard_normal(n)
    y_obs, band = -1.0, 0.01
    keep = np.abs(y - y_obs) < band
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    sub = slice(0, 6000)
    ax.scatter(y[sub], eta[sub], s=2, color=COLORS[2], alpha=0.4, label="simulated pairs $(y, \\eta)$")
    ax.axvspan(y_obs - band, y_obs + band, color=COLORS[3], alpha=0.25, label=rf"$|y - y_{{obs}}| < {band}$")
    ax.set_xlabel(r"simulated observation $y = \rho\eta + $ noise")
    ax.set_ylabel(r"$\eta$")
    ax.set_title("Draw from the prior, then simulate")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", markerscale=4)
    ax = axes[1]
    ax.scatter(eta[keep][:3000], rho[keep][:3000], s=3, color=COLORS[3], alpha=0.5, label="pairs inside the band")
    g_eta = np.linspace(*ETA, 300)
    g_rho = np.linspace(*RHO, 300)
    E, R = np.meshgrid(g_eta, g_rho)
    dens = norm.pdf(y_obs, loc=R * E, scale=NOISE)
    ax.contour(E, R, dens / dens.max(), levels=[0.05, 0.3, 0.7], colors="0.2", linewidths=1)
    ax.plot(g_eta, y_obs / g_eta, color="0.2", ls=":", lw=1, label=r"the curve $\rho\eta = y_{obs}$")
    ax.set_xlim(*ETA)
    ax.set_ylim(*RHO)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\rho$")
    ax.set_title("Their parameters follow the exact posterior (contours)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", markerscale=3)
    ax = axes[2]
    ax.hist(eta[keep], bins=40, density=True, color=COLORS[3], alpha=0.5, label=f"{keep.sum():,} accepted pairs")
    ax.plot(g_eta, exact_eta_marginal(y_obs, g_eta), color="0.2", lw=1.5, label=r"exact $p(\eta \mid y_{obs})$")
    ax.axvline(abs(y_obs) / 0.99, color=GREY, ls=":", lw=1)
    ax.text(abs(y_obs) / 0.99 + 0.05, 0.04, r"$|y_{obs}|/0.99$", color="0.35", fontsize=8.5)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("density")
    ax.set_title(r"The smile's skew alone leaves $\eta$ wide")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l12-simulated-pairs-carry-the-posterior.png")
    exact = exact_eta_marginal(y_obs, g_eta)
    cdf = np.cumsum(exact) / np.sum(exact)
    REPORT["joint"] = dict(simulated=n, accepted=int(keep.sum()), accept_rate=float(keep.mean()),
                           eta_q05_exact=float(np.interp(0.05, cdf, g_eta)), eta_q95_exact=float(np.interp(0.95, cdf, g_eta)),
                           eta_q05_abc=float(np.quantile(eta[keep], 0.05)), eta_q95_abc=float(np.quantile(eta[keep], 0.95)),
                           eta_lower_bound=abs(y_obs) / 0.99)


# Figure 12.2: amortized fit of a conditional density ------------------------------
def fig_npe():
    rng = np.random.default_rng(1)
    n = 40_000
    theta = rng.uniform(-2, 2, n)
    y = theta**3 / 4 + 0.3 * rng.standard_normal(n)

    centres = np.linspace(-2.5, 2.5, 11)

    def features(v):
        v = np.atleast_1d(v)
        return np.column_stack([np.ones_like(v), v] + [np.exp(-0.5 * ((v - c) / 0.5) ** 2) for c in centres])

    F = features(y)
    nf = F.shape[1]

    def nll(p):
        m = F @ p[:nf]
        logs = F @ p[nf:]
        return np.mean(0.5 * ((theta - m) / np.exp(logs)) ** 2 + logs)

    res = minimize(nll, np.zeros(2 * nf), method="BFGS", options=dict(maxiter=5000, gtol=1e-9))
    a, b = res.x[:nf], res.x[nf:]
    mean_of = lambda v: features(v) @ a
    sd_of = lambda v: np.exp(features(v) @ b)
    grid_t = np.linspace(-2, 2, 2001)

    def exact(yv):
        dens = norm.pdf(yv, loc=grid_t**3 / 4, scale=0.3)
        dens = dens / np.trapezoid(dens, grid_t)
        m = np.trapezoid(grid_t * dens, grid_t)
        s = np.sqrt(np.trapezoid((grid_t - m) ** 2 * dens, grid_t))
        return dens, m, s

    ys = np.linspace(-2.2, 2.2, 89)
    ex = np.array([exact(v)[1:] for v in ys])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.scatter(y[:4000], theta[:4000], s=2, color=COLORS[2], alpha=0.35, label="simulated pairs")
    ax.plot(ys, ex[:, 0], color="0.2", lw=1.8, label="exact posterior mean")
    ax.fill_between(ys, ex[:, 0] - 2 * ex[:, 1], ex[:, 0] + 2 * ex[:, 1], color="0.2", alpha=0.12,
                    label=r"exact mean $\pm 2$ sd")
    m = mean_of(ys)
    s = sd_of(ys)
    ax.plot(ys, m, color=COLORS[0], lw=1.5, ls="--", label=r"fitted $q_\phi$: mean")
    ax.plot(ys, m - 2 * s, color=COLORS[0], lw=1, ls=":")
    ax.plot(ys, m + 2 * s, color=COLORS[0], lw=1, ls=":", label=r"fitted mean $\pm 2$ sd")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.4, 2.4)
    ax.set_xlabel("observation $y$")
    ax.set_ylabel(r"parameter $\theta$")
    ax.set_title(r"One fit, every $y$: model $y = \theta^3/4 + 0.3\,\varepsilon$")
    ax.legend(frameon=False, fontsize=8, loc="upper left", markerscale=4)
    ax = axes[1]
    out = {}
    for color, yv in zip((COLORS[0], COLORS[1], COLORS[3]), (-1.5, 0.0, 1.0)):
        dens, mm, ss = exact(yv)
        ax.plot(grid_t, dens, color=color, lw=1.8, label=f"exact, $y$ = {yv}")
        mq, sq = float(mean_of(yv)[0]), float(sd_of(yv)[0])
        ax.plot(grid_t, norm.pdf(grid_t, mq, sq), color=color, lw=1.2, ls="--")
        out[yv] = dict(exact_mean=float(mm), exact_sd=float(ss), fitted_mean=float(mq), fitted_sd=float(sq))
    ax.plot([], [], color="0.3", ls="--", label="fitted Gaussian $q_\\phi$")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("posterior density")
    ax.set_title("A Gaussian family cannot follow a skewed posterior")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l12-amortized-conditional-density.png")
    REPORT["npe"] = dict(pairs=n, fit=out, features=int(nf))


# Figure 12.3: a normalizing flow is a change of variables --------------------------
def fig_flow_map():
    g_eta = np.linspace(*ETA, 2001)
    dens = exact_eta_marginal(-1.0, g_eta)
    cdf = np.cumsum(dens)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    z = np.linspace(-3.5, 3.5, 701)
    gz = np.interp(norm.cdf(z), cdf, g_eta)  # g = F^{-1} o Phi
    rng = np.random.default_rng(2)
    zs = rng.standard_normal(100_000)
    pushed = np.interp(norm.cdf(zs), cdf, g_eta)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(z, gz, color=COLORS[0], lw=2)
    ax.set_xlabel(r"base variable $z \sim \mathcal{N}(0, 1)$")
    ax.set_ylabel(r"$\eta = g(z)$")
    ax.set_title(r"An increasing map $g = F^{-1} \circ \Phi$")
    twin = ax.twinx()
    twin.fill_between(z, 0, norm.pdf(z), color=COLORS[2], alpha=0.2)
    twin.set_yticks([])
    twin.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    ax.hist(pushed, bins=80, density=True, color=COLORS[0], alpha=0.4, label=r"$g(z)$ for 100,000 draws of $z$")
    ax.plot(g_eta, dens, color="0.2", lw=1.5, label=r"target $p(\eta \mid y_{obs})$")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("density")
    ax.set_title(r"$p(\eta) = \varphi(g^{-1}(\eta))\,|dg^{-1}/d\eta|$")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(axes[0], axes[1])
    save(fig, "l12-normalizing-flow-map.png")


# Figure 12.4: flow matching on a two-component target ---------------------------
MIX = [(0.4, -1.0, 0.30), (0.6, 1.2, 0.20)]  # weight, mean, sd


def velocity(x, tau):
    """Exact marginal velocity E[theta - z | x_tau = x] for the straight-line coupling and a Gaussian mixture."""
    x = np.asarray(x, dtype=float)
    num = np.zeros_like(x)
    den = np.zeros_like(x)
    for w, m, s in MIX:
        var = (1 - tau) ** 2 + tau**2 * s**2
        cov = tau * s**2 - (1 - tau)
        lik = w * norm.pdf(x, loc=tau * m, scale=np.sqrt(var))
        num += lik * (m + cov / var * (x - tau * m))
        den += lik
    return num / np.maximum(den, 1e-300)


def target_pdf(x):
    return sum(w * norm.pdf(x, m, s) for w, m, s in MIX)


def p_tau(x, tau):
    return sum(w * norm.pdf(x, tau * m, np.sqrt((1 - tau) ** 2 + tau**2 * s**2)) for w, m, s in MIX)


def fig_flow_matching():
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    z = rng.standard_normal(40)
    comp = rng.choice(2, size=40, p=[MIX[0][0], MIX[1][0]])
    th = np.array([rng.normal(MIX[c][1], MIX[c][2]) for c in comp])
    for zi, ti in zip(z, th):
        ax.plot([0, 1], [zi, ti], color=COLORS[2], lw=0.8, alpha=0.7)
    ax.set_xlabel(r"time $\tau$")
    ax.set_ylabel(r"$x_\tau = (1-\tau)z + \tau\theta$")
    ax.set_title("Training pairs: straight lines from noise to target")
    ax = axes[1]
    T, X = np.meshgrid(np.linspace(0, 0.995, 200), np.linspace(-3, 3, 240))
    V = velocity(X, T)
    im = ax.pcolormesh(T, X, V, cmap="RdBu_r", vmin=-4, vmax=4, shading="auto")
    fig.colorbar(im, ax=ax, label=r"velocity $v(x, \tau)$")
    zs = np.linspace(-2.4, 2.4, 25)
    n = 400
    taus = np.linspace(0, 1, n + 1)
    paths = np.empty((n + 1, len(zs)))
    x = zs.copy()
    paths[0] = x
    for i in range(n):
        x = x + (taus[i + 1] - taus[i]) * velocity(x, taus[i])
        paths[i + 1] = x
    for j in range(len(zs)):
        ax.plot(taus, paths[:, j], color="0.1", lw=0.7)
    ax.set_xlabel(r"time $\tau$")
    ax.set_ylabel("$x$")
    ax.set_title("The averaged field, and the paths that follow it")
    ax = axes[2]
    zz = rng.standard_normal(50_000)
    x = zz.copy()
    snaps = {}
    for i in range(n):
        x = x + (taus[i + 1] - taus[i]) * velocity(x, taus[i])
        for t in (0.5, 0.8):
            if abs(taus[i + 1] - t) < 1e-9:
                snaps[t] = x.copy()
    grid = np.linspace(-3, 3, 400)
    for color, t in zip((COLORS[2], COLORS[1]), (0.5, 0.8)):
        ax.hist(snaps[t], bins=100, range=(-3, 3), density=True, histtype="step", color=color, lw=1.2,
                label=rf"flow at $\tau$ = {t}")
        ax.plot(grid, p_tau(grid, t), color=color, lw=0.8, ls=":")
    ax.hist(x, bins=100, range=(-3, 3), density=True, color=COLORS[0], alpha=0.4, label=r"flow at $\tau$ = 1")
    ax.plot(grid, target_pdf(grid), color="0.1", lw=1.5, label="target density")
    ax.set_xlabel("$x$")
    ax.set_ylabel("density")
    ax.set_title("Following the field carries N(0, 1) to the target")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    tidy(axes[0], axes[2])
    save(fig, "l12-flow-matching.png")
    # agreement of the transported law with the target
    hist, edges = np.histogram(x, bins=60, range=(-3, 3), density=True)
    mid = 0.5 * (edges[1:] + edges[:-1])
    REPORT["flow_matching"] = dict(mean_flow=float(x.mean()), mean_target=float(sum(w * m for w, m, s in MIX)),
                                   share_right=float((x > 0).mean()), weight_right=MIX[1][0],
                                   max_abs_density_gap=float(np.max(np.abs(hist - target_pdf(mid)))))


# Figure 12.5: Euler steps undershoot sharp posteriors ---------------------------
def spread_ratio(s, grid):
    """Final spread of Euler with the exact linear field, divided by the exact final spread s."""
    factor = 1.0
    for t0, t1 in zip(grid[:-1], grid[1:]):
        k = (t0 * s * s - (1 - t0)) / ((1 - t0) ** 2 + t0 * t0 * s * s)
        factor *= 1 + (t1 - t0) * k
    return abs(factor) / s


def fig_euler():
    sgrid = np.geomspace(0.005, 1.0, 80)
    schemes = {
        "50 uniform steps": np.linspace(0, 1, 51),
        "100 uniform steps": np.linspace(0, 1, 101),
        "200 uniform steps": np.linspace(0, 1, 201),
        "50 end-dense steps": 1 - (1 - np.arange(51) / 50) ** 2,
        "100 end-dense steps": 1 - (1 - np.arange(101) / 100) ** 2,
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    styles = [(COLORS[3], "-"), (COLORS[1], "-"), (COLORS[2], "-"), (COLORS[3], "--"), (COLORS[0], "--")]
    table = {}
    for (name, grid), (color, ls) in zip(schemes.items(), styles):
        r = np.array([spread_ratio(s, grid) for s in sgrid])
        ax.semilogx(sgrid, r, color=color, ls=ls, label=name)
        table[name] = {str(s): float(spread_ratio(s, grid)) for s in (0.02, 0.05, 0.1)}
    for s in (0.02, 0.05, 0.1):
        ax.axvline(s, color="0.88", lw=0.8, zorder=0)
    ax.axhline(1, color="0.5", lw=0.8)
    ax.set_ylim(0.4, 1.05)
    ax.set_xlabel(r"posterior standard deviation $s$ (standardized units)")
    ax.set_ylabel("sampled spread / exact spread")
    ax.set_title("Euler steps with the exact field undershoot sharp posteriors")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax = axes[1]
    taus = np.linspace(0, 0.9999, 4000)
    for color, s in zip((COLORS[3], COLORS[1], COLORS[0]), (0.02, 0.1, 0.5)):
        k = (taus * s * s - (1 - taus)) / ((1 - taus) ** 2 + taus**2 * s * s)
        ax.loglog(1 - taus, np.abs(k), color=color, label=f"$s$ = {s}")
    ax.loglog(1 - taus, 1 / (1 - taus), color=GREY, ls=":", label=r"$1/(1-\tau)$")
    ax.set_xlabel(r"time left, $1 - \tau$")
    ax.set_ylabel(r"$|k_\tau|$, slope of the field in $x$")
    ax.set_title(r"The field steepens near $\tau = 1$ until $1-\tau \approx s$")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l12-euler-undershoot.png")
    REPORT["euler"] = table


# Figure 12.6: where in time the loss is trained ---------------------------------
def fig_time_sampling():
    t = np.linspace(0.001, 0.999, 999)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(t, np.ones_like(t), color=GREY, label=r"$\tau = u$ (uniform)")
    ax.plot(t, (2 / 3) * t ** (-1 / 3), color=COLORS[3], label=r"$\tau = u^{1.5}$: density $\frac{2}{3}\tau^{-1/3}$")
    ax.plot(t, 2 * t, color=COLORS[0], label=r"$\tau = u^{0.5}$: density $2\tau$")
    ax.set_ylim(0, 2.6)
    ax.set_xlabel(r"time $\tau$")
    ax.set_ylabel("density of training times")
    ax.set_title("Three ways to draw the training time")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center")
    ax = axes[1]
    for color, (name, dens) in zip((GREY, COLORS[3], COLORS[0]),
                                   (("uniform", lambda x: np.ones_like(x)), (r"$u^{1.5}$", lambda x: (2 / 3) * x ** (-1 / 3)),
                                    (r"$u^{0.5}$", lambda x: 2 * x))):
        share = []
        cuts = np.array([0.5, 0.8, 0.9, 0.95, 0.98, 0.99])
        for c in cuts:
            xx = np.linspace(c, 1, 2000)
            share.append(np.trapezoid(dens(xx), xx))
        ax.semilogx(1 - cuts, share, "o-", color=color, ms=4, label=name)
        REPORT.setdefault("time_share", {})[name] = {str(c): float(v) for c, v in zip(cuts, share)}
    ax.invert_xaxis()
    ax.set_xlabel(r"last stretch of time, $1 - \tau_0$")
    ax.set_ylabel(r"share of training times with $\tau > \tau_0$")
    ax.set_title("How much training happens near the data end")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l12-time-sampling.png")


# Figure 12.7: what the network sees ------------------------------------------------
def fig_inputs():
    spy = load_spy()
    ex = load("smile/example-rough.json")
    y_spy = np.array(spy["prices_forward_units"]) / np.array(spy["noise_sd"])
    y_ex = np.array(ex["noisy_quotes"]) / np.array(ex["quote_noise_sd"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    for color, j in zip(COLORS, range(len(DAYS))):
        ax.semilogy(DELTAS, y_spy[9 * j:9 * j + 9], "o-", color=color, ms=3.5, label=f"{DAYS[j]} d")
        ax.semilogy(DELTAS, np.maximum(y_ex[9 * j:9 * j + 9], 1e-2), "x:", color=color, ms=4, lw=0.8)
    ax.set_xlabel(r"standardized moneyness $\delta$")
    ax.set_ylabel(r"$y_c / \sigma_c$")
    ax.set_title("63 conditions: quotes in units of their noise")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, title="SPY (dots); example surface (crosses)", title_fontsize=7.5)
    ax = axes[1]
    edges = np.r_[0, DAYS]
    ax.stairs(np.log(spy["forward_variance_levels"]), edges, color=COLORS[3], lw=2, baseline=None, label="SPY, 21 Oct 2026")
    ax.stairs(np.log(ex["forward_variance_levels"]), edges, color=COLORS[0], lw=2, baseline=None,
              label="example surface (flat 25%)")
    ax.set_xlabel("time, business days")
    ax.set_ylabel(r"$\log \xi_0$ on each interval")
    ax.set_title("7 more conditions: the forward-variance curve")
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[2]
    zz = np.linspace(-8, 8, 400)
    ax.plot(zz, np.exp(-zz) / (1 + np.exp(-zz)) ** 2, color=COLORS[0], lw=2,
            label=r"logit of a uniform: logistic, variance $\pi^2/3$")
    ax.plot(zz, norm.pdf(zz, scale=np.pi / np.sqrt(3)), color=GREY, ls="--", label=r"normal with the same variance")
    ax.set_xlabel(r"$z = \mathrm{logit}((\theta - a)/(b - a))$")
    ax.set_ylabel("prior density in $z$")
    ax.set_title("3 targets on the real line: $H$, $\\eta$, $\\rho$")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    tidy(*axes)
    save(fig, "l12-network-inputs.png")
    REPORT["inputs"] = dict(spy_min=float(y_spy.min()), spy_max=float(y_spy.max()),
                            spy_atm_1y=float(y_spy[9 * 6 + 4]), spy_call_1w_d2=float(y_spy[8]),
                            ex_min=float(y_ex.min()), ex_max=float(y_ex.max()), n_neg_example=int((y_ex < 0).sum()),
                            spy_log_xi=[float(v) for v in np.log(spy["forward_variance_levels"])],
                            ex_log_xi=float(np.log(ex["forward_variance_levels"][0])))


# Figure 12.8: the training bank -------------------------------------------------
def fig_bank():
    rough = load("rough-pilot/s1/bank-quality.json")
    heston = load("heston/s1/bank-quality.json")
    rungs = rough["paths_ladder"]
    blocks = rough["share_of_blocks_by_rung"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    ax = axes[0]
    width = 0.27
    names = {"block 0": "10-day grid (1 and 2 weeks)", "block 1": "63-day grid (1, 2, 3 months)", "block 2": "1-year grid (6 and 12 months)"}
    for i, (key, color) in enumerate(zip(("block 0", "block 1", "block 2"), (COLORS[0], COLORS[1], COLORS[3]))):
        shares = [blocks[key][str(r)] for r in rungs]
        ax.bar(np.arange(len(rungs)) + (i - 1) * width, np.array(shares) * 100, width=width, color=color, label=names[key])
    ax.set_yscale("log")
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels([f"$2^{{{int(np.log2(r))}}}$" for r in rungs])
    ax.set_xlabel("paths per maturity block at which the error budget was met")
    ax.set_ylabel("share of maturity blocks, %")
    ax.set_title("Rough pilot bank, 2,200 surfaces: the path ladder")
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1]
    labels = ["median", "90%", "99%", "largest"]
    keys = ["q50", "q90", "q99", "max"]
    r = [rough["error_quantiles_all_quotes"][k] for k in keys]
    h = [heston["error_quantiles_all_quotes"][k] for k in keys]
    x = np.arange(4)
    ax.bar(x - 0.2, r, width=0.4, color=COLORS[3], label="rough Bergomi: Monte Carlo error / $\\sigma_c$")
    ax.bar(x + 0.2, np.maximum(h, 1e-9), width=0.4, color=COLORS[0], label="Heston: quadrature error / $\\sigma_c$")
    ax.axhline(rough["budget"], color=COLORS[3], ls="--", lw=1, label="rough budget 0.25")
    ax.axhline(heston["budget"], color=COLORS[0], ls="--", lw=1, label="Heston budget 0.1")
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("quantile over all quotes of the bank")
    ax.set_ylabel("numerical error in units of the quote noise")
    ax.set_title("Numerical error of the stored prices, against each budget")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    tidy(*axes)
    save(fig, "l12-training-bank.png")
    REPORT["bank"] = dict(rough_cpu_per_surface=rough["generation"]["cpu_seconds_per_surface"],
                          rough_cpu_h_per_1000=rough["generation"]["cpu_hours_per_1000_surfaces"],
                          heston_cpu_per_surface=heston["generation"]["cpu_seconds_per_surface"],
                          heston_cpu_h_per_1000=heston["generation"]["cpu_hours_per_1000_surfaces"],
                          rough_q=r, heston_q=h,
                          first_rung={k: blocks[k]["32768"] for k in blocks})


# Figure 12.9: what the three Heston networks taught ----------------------------
def fig_networks():
    # design document, "The simulator and the training bank" and "Simulation-based calibration"
    band = 0.03562012708860619  # heston/s2/validation.json, dkw_bound
    v3 = load("heston/s2/validation.json")["sbc"]["rank_ks"]
    ks = {"network 1\n4 fixed noise draws,\n40 epochs": (0.086, 0.051),
          "network 2\nfresh noise per batch,\n500 epochs": (0.110, 0.043),
          "network 3\n+ end-dense sampler,\n$\\tau = u^{0.5}$": (v3[0], v3[1])}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    x = np.arange(3)
    ax.bar(x - 0.18, [v[0] for v in ks.values()], width=0.36, color=COLORS[3], label=r"$v_0$")
    ax.bar(x + 0.18, [v[1] for v in ks.values()], width=0.36, color=COLORS[1], label=r"$\theta_H$")
    ax.axhline(band, color="0.3", ls="--", lw=1)
    ax.text(2.35, band + 0.003, "DKW band 0.036", fontsize=8.5, color="0.3", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(list(ks.keys()), fontsize=8)
    ax.set_ylabel("largest SBC rank-ECDF distance")
    ax.set_title("SBC on 2,000 held-out Heston smiles")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax = axes[1]
    names = ["network 2\n(uniform sampler fixed)", "network 3"]
    ax.bar([0, 1], [0.35, 0.16], width=0.4, color=COLORS[3], label=r"$v_0$ mean error, posterior sd")
    twin = ax.twinx()
    twin.plot([0, 1], [28, 80], "o-", color=COLORS[0], ms=6, label="smiles meeting the S3 criterion, %")
    twin.set_ylim(0, 100)
    twin.set_ylabel("smiles meeting the criterion on all 4 targets, %", color=COLORS[0])
    twin.spines[["top"]].set_visible(False)
    ax.set_ylim(0, 0.45)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("median $v_0$ mean error, in exact-posterior sd")
    ax.set_title(r"Training at $\tau = u^{0.5}$ moved the sharp posterior's location")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="upper center")
    tidy(axes[0])
    ax.spines[["top"]].set_visible(False)
    save(fig, "l12-three-heston-networks.png")
    REPORT["networks"] = dict(band=band, v3=v3)


# Figure 12.10: variable quote sets -----------------------------------------------
def fig_masks():
    spy = load_spy()
    exps = spy["diagnostics"]["expirations"]
    rng = np.random.default_rng(4)
    masks = []
    m = np.ones((7, 9))
    m[1] = 0
    masks.append(("a missing maturity", m.copy()))
    m = np.ones((7, 9))
    m[:3, :2] = 0
    m[:3, 7:] = 0
    masks.append(("missing short-dated wings", m.copy()))
    m = (rng.random((7, 9)) > 0.3).astype(float)
    masks.append(("random 30% missing", m.copy()))
    fig = plt.figure(figsize=(15, 3.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.3, 1, 1, 1])
    ax = fig.add_subplot(gs[0])
    for e in exps:
        lo, hi = e["strip_deltas"]
        ax.plot([lo, hi], [e["days"], e["days"]], color=COLORS[0] if e["replication_covered"] else COLORS[2], lw=2)
    ax.axvspan(-2, 2, color=COLORS[1], alpha=0.12)
    for d in DAYS:
        ax.axhline(d, color=COLORS[1], lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_xlim(-12, 6)
    ax.set_xlabel(r"range of $\delta$ covered by the cleaned quotes")
    ax.set_ylabel("expiration, business days")
    ax.plot([], [], color=COLORS[0], lw=2, label="strip reaches $\\delta \\leq -8$")
    ax.plot([], [], color=COLORS[2], lw=2, label="strip cut short")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_title("SPY's 32 expirations against the design grid (shaded)", fontsize=9)
    axes = [fig.add_subplot(gs[i]) for i in (1, 2, 3)]
    for ax_, (title, mm) in zip(axes, masks):
        ax_.imshow(mm, cmap="Greys", vmin=-0.2, vmax=1.6, aspect="auto")
        ax_.set_title(f"mask: {title} (light = missing)", fontsize=8.5)
        ax_.set_xticks(range(9))
        ax_.set_xticklabels([f"{d:g}" for d in DELTAS], fontsize=7)
        ax_.set_yticks(range(7))
        ax_.set_yticklabels([f"{d} d" for d in DAYS], fontsize=7)
        ax_.set_xlabel(r"$\delta$", fontsize=8)
    tidy(ax)
    save(fig, "l12-quote-masks.png")
    lo = np.array([e["strip_deltas"][0] for e in exps])
    hi = np.array([e["strip_deltas"][1] for e in exps])
    REPORT["masks"] = dict(expirations=len(exps), covered=int(sum(e["replication_covered"] for e in exps)),
                           lowest_delta=float(lo.min()), highest_delta=float(hi.max()),
                           short_hi=[float(h) for h, e in zip(hi, exps) if e["days"] <= 10],
                           two=spy["diagnostics"]["grid_point_readings"]["two expirations"],
                           one=spy["diagnostics"]["grid_point_readings"]["one expiration"])


# Figure 12.11: the neural-pricer route ------------------------------------------
def fig_pricer():
    s4 = load("heston/s4/pricer-route.json")["summary"]["pricer"]
    rms = s4["held_out_rms_error_in_noise_sd"]
    q99 = s4["held_out_abs_error_q99_in_noise_sd"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    vals = [rms["median_over_contracts"], rms["max"], q99]
    labels = ["RMS error,\nmedian contract", "RMS error,\nworst contract", "99% of all\nabsolute errors"]
    ax.bar(range(3), vals, color=COLORS[0], width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.15, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, color=COLORS[3], ls="--", lw=1)
    ax.text(2.3, 0.465, "allowance 0.5", color=COLORS[3], fontsize=8.5, ha="right")
    ax.axhline(0.25, color=COLORS[1], ls=":", lw=1)
    ax.text(2.3, 0.265, "Monte Carlo budget 0.25", color=COLORS[1], fontsize=8.5, ha="right")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel(r"pricer error in units of $\sigma_c$")
    held_out = load("heston/s4/pricer-route.json")["settings"]["first"]  # errors are measured on test[:first]
    ax.set_title(f"Heston neural pricer on {held_out:,} held-out test surfaces")
    ax = axes[1]
    e = np.linspace(0, 0.6, 61)
    ax.plot(e, 0.5 * 63 * e**2, color=COLORS[3], label=r"systematic shift $\frac{1}{2}\sum_c e_c^2$")
    ax.plot(e, np.sqrt(63) * e, color=COLORS[2], ls="--", label=r"spread $\sqrt{\sum_c e_c^2}$ over noisy quotes")
    ax.axvline(rms["median_over_contracts"], color=COLORS[0], ls=":", lw=1)
    ax.text(rms["median_over_contracts"] + 0.01, 5, "Heston pricer,\nmedian contract", color=COLORS[0], fontsize=8.5)
    ax.set_xlabel(r"pricer error $e_c$ in units of $\sigma_c$, the same on all 63 quotes")
    ax.set_ylabel("log-likelihood units")
    ax.set_title("How a pricer error distorts the likelihood")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.1, 1.0))
    tidy(*axes)
    save(fig, "l12-neural-pricer.png")
    REPORT["pricer"] = dict(rms=rms, q99=q99, shift_at_0_5=0.5 * 63 * 0.25, per_quote_0_5=0.125)


def main():
    fig_joint()
    fig_npe()
    fig_flow_map()
    fig_flow_matching()
    fig_euler()
    fig_time_sampling()
    fig_inputs()
    fig_bank()
    fig_networks()
    fig_masks()
    fig_pricer()
    print(json.dumps(REPORT, indent=1, default=float))


if __name__ == "__main__":
    main()
