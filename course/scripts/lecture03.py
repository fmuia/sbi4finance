"""Figures of Lecture 3: Brownian motion, stochastic differential equations and Itô's lemma.

Usage: python scripts/course/lecture03.py   (writes docs/course/images/l03-*.png and prints the numbers quoted in the text)
"""

import json

import numpy as np
from scipy.stats import norm

from common import COLORS, GREY, load_voo, plt, save, tidy

REPORT = {}


def fig_random_walk():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    steps = rng.choice([-1.0, 1.0], size=10_000)
    for color, n in zip(COLORS, (10, 100, 10_000)):
        # the same coin flips, coarse-grained: n steps on [0, 1], each the sum of 10000/n flips, scaled by 1/sqrt(10000)
        block = steps.reshape(n, -1).sum(axis=1) / np.sqrt(10_000)
        ax.step(np.linspace(0, 1, n + 1), np.r_[0, np.cumsum(block)], where="post", color=color, lw=1,
                label=f"{n} steps")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$W^{(n)}_t$")
    ax.set_title(r"A random walk with steps $\pm 1/\sqrt{n}$")
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1]
    t = np.linspace(0, 1, 501)
    W = np.c_[np.zeros(200), np.cumsum(rng.standard_normal((200, 500)) * np.sqrt(1 / 500), axis=1)]
    for w in W[:60]:
        ax.plot(t, w, color=COLORS[2], lw=0.4, alpha=0.6)
    for m, ls in ((1, "--"), (2, ":")):
        ax.plot(t, m * np.sqrt(t), color=COLORS[3], ls=ls, lw=1.2, label=rf"$\pm{m}\sqrt{{t}}$")
        ax.plot(t, -m * np.sqrt(t), color=COLORS[3], ls=ls, lw=1.2)
    ax.set_xlabel("time $t$")
    ax.set_ylabel("$W_t$")
    ax.set_title("Brownian paths spread like the square root of time")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax = axes[2]
    many = np.cumsum(rng.standard_normal((20_000, 500)) * np.sqrt(1 / 500), axis=1)
    ax.plot(t[1:], many.var(axis=0), color=COLORS[0], label=r"sample variance of $W_t$, 20,000 paths")
    ax.plot(t, t, color="0.3", ls="--", label="$t$")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("variance")
    ax.set_title(r"$\mathrm{Var}(W_t) = t$")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l03-random-walk-to-brownian.png")


def fig_quadratic_variation():
    rng = np.random.default_rng(1)
    ns = np.array([10, 30, 100, 300, 1000, 3000, 10_000, 30_000, 100_000])
    reps = 400
    qv, tv = np.empty((len(ns), reps)), np.empty((len(ns), reps))
    for i, n in enumerate(ns):
        dW = rng.standard_normal((reps, n)) * np.sqrt(1 / n)
        qv[i] = (dW**2).sum(axis=1)
        tv[i] = np.abs(dW).sum(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.fill_between(ns, np.quantile(qv, 0.05, axis=1), np.quantile(qv, 0.95, axis=1), color=COLORS[0], alpha=0.2,
                    label="5%–95% over 400 paths")
    ax.plot(ns, qv.mean(axis=1), "o-", color=COLORS[0], ms=3.5, label=r"mean of $\sum (\Delta W)^2$")
    ax.axhline(1, color="0.3", ls="--", lw=1, label="$T = 1$")
    ax.set_xscale("log")
    ax.set_xlabel("number of steps $n$ on $[0, 1]$")
    ax.set_ylabel("quadratic variation")
    ax.set_title(r"$\sum (\Delta W_i)^2 \to T$: the squared increments add up to time")
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1]
    ax.loglog(ns, tv.mean(axis=1), "o-", color=COLORS[3], ms=3.5, label=r"mean of $\sum |\Delta W|$")
    ax.loglog(ns, np.sqrt(2 * ns / np.pi), color="0.3", ls="--", label=r"$\sqrt{2n/\pi}$")
    ax.set_xlabel("number of steps $n$ on $[0, 1]$")
    ax.set_ylabel("total variation")
    ax.set_title("The length of the path diverges")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l03-quadratic-variation.png")
    REPORT["qv_sd"] = {int(n): float(s) for n, s in zip(ns, qv.std(axis=1))}


def fig_fbm():
    n = 400
    t = np.linspace(1 / n, 1, n)
    z = np.random.default_rng(2).standard_normal(n)
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    for color, (H, off) in zip(COLORS, ((0.1, 3.0), (0.3, 1.0), (0.5, -1.0), (0.75, -3.0))):
        cov = 0.5 * (t[:, None] ** (2 * H) + t[None, :] ** (2 * H) - np.abs(t[:, None] - t[None, :]) ** (2 * H))
        L = np.linalg.cholesky(cov + 1e-12 * np.eye(n))
        path = np.r_[0, L @ z]
        ax.plot(np.r_[0, t], path + off, color=color, lw=0.9, label=f"$H$ = {H}" + (" (Brownian)" if H == 0.5 else ""))
    ax.set_yticks([])
    ax.set_xlabel("time $t$")
    ax.set_title("Fractional Brownian motion: the exponent $H$ sets the roughness")
    ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    tidy(ax)
    save(fig, "l03-roughness-exponent.png")


def fig_ito_integral():
    rng = np.random.default_rng(3)
    N = 2 ** 16
    dW = rng.standard_normal(N) * np.sqrt(1 / N)
    W = np.r_[0, np.cumsum(dW)]
    ns = 2 ** np.arange(2, 17)
    left, mid = [], []
    for n in ns:
        Wc = W[:: N // n]
        d = np.diff(Wc)
        left.append(np.sum(Wc[:-1] * d))
        mid.append(np.sum(0.5 * (Wc[:-1] + Wc[1:]) * d))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.semilogx(ns, left, "o-", color=COLORS[0], ms=3.5, label="left-point sums (Itô)")
    ax.semilogx(ns, mid, "o-", color=COLORS[1], ms=3.5, label="trapezoid sums (Stratonovich)")
    ax.axhline(0.5 * W[-1] ** 2 - 0.5, color=COLORS[0], ls="--", lw=1, label=r"$\frac{1}{2} W_1^2 - \frac{1}{2}$")
    ax.axhline(0.5 * W[-1] ** 2, color=COLORS[1], ls="--", lw=1, label=r"$\frac{1}{2} W_1^2$")
    ax.set_xlabel("number of steps $n$")
    ax.set_ylabel(r"$\sum W\,\Delta W$")
    ax.set_title(r"One path: $\int_0^1 W\,dW$ depends on where $W$ is evaluated")
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1]
    paths = np.cumsum(rng.standard_normal((20_000, 500)) * np.sqrt(1 / 500), axis=1)
    Wp = np.c_[np.zeros(20_000), paths]
    d = np.diff(Wp, axis=1)
    ito = np.sum(Wp[:, :-1] * d, axis=1)
    strat = np.sum(0.5 * (Wp[:, :-1] + Wp[:, 1:]) * d, axis=1)
    bins = np.linspace(-1, 4, 120)
    ax.hist(ito, bins=bins, density=True, color=COLORS[0], alpha=0.5, label=f"Itô: mean {ito.mean():.3f}")
    ax.hist(strat, bins=bins, density=True, color=COLORS[1], alpha=0.5, label=f"Stratonovich: mean {strat.mean():.3f}")
    ax.set_xlabel(r"$\int_0^1 W\,dW$")
    ax.set_ylabel("density")
    ax.set_title("20,000 paths: only the Itô integral has mean zero")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l03-ito-integral.png")
    REPORT["ito_mean"] = float(ito.mean())
    REPORT["strat_mean"] = float(strat.mean())


def fig_gbm():
    rng = np.random.default_rng(5)
    mu, sigma, T, n = 0.08, 0.2, 5.0, 1260
    dt = T / n
    t = np.linspace(0, T, n + 1)
    Z = rng.standard_normal((20_000, n))
    logS = np.c_[np.zeros(20_000), np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1)]
    S = 100 * np.exp(logS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for s in S[:40]:
        ax.plot(t, s, color=COLORS[2], lw=0.4, alpha=0.6)
    ax.plot(t, 100 * np.exp(mu * t), color=COLORS[3], lw=2, label=r"mean $S_0 e^{\mu t}$")
    ax.plot(t, 100 * np.exp((mu - 0.5 * sigma**2) * t), color=COLORS[0], lw=2, ls="--",
            label=r"median $S_0 e^{(\mu - \sigma^2/2)t}$")
    ax.plot(t, S.mean(axis=0), color="0.2", lw=1, ls=":", label="sample mean of 20,000 paths")
    ax.set_ylim(30, 350)
    ax.set_xlabel("years")
    ax.set_ylabel("$S_t$")
    ax.set_title(r"Geometric Brownian motion, $\mu$ = 8%, $\sigma$ = 20%")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax = axes[1]
    ST = S[:, -1]
    ax.hist(ST, bins=np.linspace(0, 450, 120), density=True, color=COLORS[2], alpha=0.6, label="simulated $S_5$")
    x = np.linspace(1, 450, 600)
    m, v = np.log(100) + (mu - 0.5 * sigma**2) * T, sigma**2 * T
    ax.plot(x, np.exp(-(np.log(x) - m) ** 2 / (2 * v)) / (x * np.sqrt(2 * np.pi * v)), color="0.2", label="log-normal density")
    ax.axvline(100 * np.exp(mu * T), color=COLORS[3], lw=1.5, label=f"mean {100 * np.exp(mu * T):.1f}")
    ax.axvline(100 * np.exp((mu - 0.5 * sigma**2) * T), color=COLORS[0], ls="--", lw=1.5,
               label=f"median {100 * np.exp((mu - 0.5 * sigma**2) * T):.1f}")
    ax.set_xlabel("$S_5$")
    ax.set_ylabel("density")
    ax.set_title("After five years the law of $S_T$ is log-normal and skewed")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l03-gbm.png")
    REPORT["gbm"] = dict(mean=float(100 * np.exp(mu * T)), median=float(100 * np.exp((mu - 0.5 * sigma**2) * T)),
                         sample_mean=float(ST.mean()), prob_below_100=float((ST < 100).mean()),
                         prob_below_100_theory=float(norm.cdf(-(mu - 0.5 * sigma**2) * T / (sigma * np.sqrt(T)))))


def fig_euler():
    rng = np.random.default_rng(6)
    mu, sigma, T = 0.08, 0.6, 1.0
    N = 2 ** 12
    paths = 4000
    dW = rng.standard_normal((paths, N)) * np.sqrt(T / N)
    WT = dW.sum(axis=1)
    exact = np.exp((mu - 0.5 * sigma**2) * T + sigma * WT)
    steps = 2 ** np.arange(2, 13)
    strong, weak = [], []
    for n in steps:
        d = dW.reshape(paths, n, -1).sum(axis=2)
        X = np.ones(paths)
        for i in range(n):
            X = X + mu * X * (T / n) + sigma * X * d[:, i]
        strong.append(np.mean(np.abs(X - exact)))
        weak.append(abs(X.mean() - np.exp(mu * T)))
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.loglog(steps, strong, "o-", color=COLORS[0], ms=3.5, label=r"path-by-path error $\mathbb{E}|X^{(n)}_T - S_T|$")
    ax.loglog(steps, strong[0] * (steps / steps[0]) ** -0.5, color=COLORS[0], ls=":", lw=1, label=r"slope $-1/2$")
    ax.set_xlabel("number of Euler steps on one year")
    ax.set_ylabel("error")
    ax.set_title(r"Euler–Maruyama on GBM ($\sigma$ = 60%): strong error $\propto \Delta t^{1/2}$")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(ax)
    save(fig, "l03-euler.png")
    REPORT["euler_strong"] = {int(n): float(e) for n, e in zip(steps, strong)}


def fig_correlation():
    rng = np.random.default_rng(7)
    n = 2000
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    W = rng.standard_normal(n)
    Wp = rng.standard_normal(n)
    for color, rho in zip((COLORS[2], COLORS[3]), (0.0, -0.7)):
        B = rho * W + np.sqrt(1 - rho**2) * Wp
        ax.scatter(W, B, s=3, alpha=0.35, color=color, label=rf"$\rho$ = {rho}: sample correlation {np.corrcoef(W, B)[0, 1]:.2f}")
    ax.set_xlabel(r"$\Delta W / \sqrt{\Delta t}$")
    ax.set_ylabel(r"$\Delta B / \sqrt{\Delta t}$")
    ax.set_title(r"$B = \rho W + \sqrt{1-\rho^2}\,W^\perp$")
    ax.legend(frameon=False, fontsize=8.5, markerscale=4)
    ax = axes[1]
    _, r = load_voo()
    past = r.rolling(21).std().shift(1)                       # realized vol over the 21 days before today
    future = r[::-1].rolling(21).std()[::-1].shift(-1)        # realized vol over the 21 days after today
    z = r / past                                              # today's return in units of recent volatility
    frame = np.c_[z.values, np.log(future / past).values]
    frame = frame[np.isfinite(frame).all(axis=1)]
    bins = np.quantile(frame[:, 0], np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(frame[:, 0], bins) - 1, 0, 9)
    centers = [frame[idx == i, 0].mean() for i in range(10)]
    means = [frame[idx == i, 1].mean() * 100 for i in range(10)]
    ax.bar(range(10), means, color=[COLORS[3] if m > 0 else COLORS[0] for m in means])
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"{c:+.1f}" for c in centers], fontsize=8)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("today's return in units of the past month's volatility (decile averages)")
    ax.set_ylabel("next month's volatility over last month's, log, %")
    ax.set_title("After a fall, volatility rises more: VOO 2020–2026")
    tidy(*axes)
    save(fig, "l03-correlation.png")
    REPORT["leverage"] = dict(corr=float(np.corrcoef(frame[:, 0], frame[:, 1])[0, 1]),
                              deciles=[[float(c), float(m)] for c, m in zip(centers, means)])


def fig_girsanov():
    rng = np.random.default_rng(8)
    T, lam = 1.0, 0.8
    WT = rng.standard_normal(200_000) * np.sqrt(T)
    Z = np.exp(-lam * WT - 0.5 * lam**2 * T)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    x = np.linspace(-4, 4, 400)
    ax.plot(x, np.exp(-lam * x - 0.5 * lam**2 * T), color=COLORS[1])
    ax.axhline(1, color="0.8", lw=0.8)
    ax.set_xlabel("$W_T$")
    ax.set_ylabel(r"weight $Z_T = e^{-\lambda W_T - \lambda^2 T/2}$")
    ax.set_title(rf"Reweighting the paths, $\lambda$ = {lam}")
    ax = axes[1]
    bins = np.linspace(-5, 4, 120)
    ax.hist(WT, bins=bins, density=True, color=COLORS[2], alpha=0.5, label=r"$W_T$ under $\mathbb{P}$: N(0, 1)")
    ax.hist(WT, bins=bins, weights=Z, density=True, color=COLORS[1], alpha=0.5, label=r"$W_T$ under $\mathbb{Q}$ (weighted)")
    ax.plot(x, norm.pdf(x, loc=-lam * T, scale=np.sqrt(T)), color=COLORS[1], ls="--", lw=1.2,
            label=rf"N($-\lambda T$, $T$) = N({-lam * T:.1f}, 1)")
    ax.set_xlabel("$W_T$")
    ax.set_ylabel("density")
    ax.set_title("The weights shift the mean and keep the variance")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l03-girsanov.png")
    REPORT["girsanov"] = dict(weighted_mean=float(np.sum(Z * WT) / np.sum(Z)),
                              weighted_var=float(np.sum(Z * WT**2) / np.sum(Z) - (np.sum(Z * WT) / np.sum(Z)) ** 2),
                              mean_weight=float(Z.mean()))


def fig_estimation():
    rng = np.random.default_rng(9)
    mu, sigma = 0.08, 0.2
    reps = 4000
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    spans = np.array([1, 2, 5, 10, 25, 50])
    sd_mu, sd_sig = [], []
    for T in spans:
        n = 252 * T
        # sufficient statistics of daily log returns, simulated exactly
        r_sum = rng.normal((mu - 0.5 * sigma**2) * T, sigma * np.sqrt(T), reps)
        sq = sigma**2 / 252 * rng.chisquare(n, reps)  # sum of squared deviations, approximately
        sig_hat = np.sqrt(sq / n * 252)
        mu_hat = r_sum / T + 0.5 * sig_hat**2
        sd_mu.append(mu_hat.std())
        sd_sig.append(sig_hat.std())
    ax = axes[0]
    ax.loglog(spans, np.array(sd_mu) * 100, "o-", color=COLORS[3], ms=4, label=r"sd of $\hat\mu$ (simulated)")
    ax.loglog(spans, sigma / np.sqrt(spans) * 100, color=COLORS[3], ls=":", label=r"$\sigma/\sqrt{T}$")
    ax.loglog(spans, np.array(sd_sig) * 100, "o-", color=COLORS[0], ms=4, label=r"sd of $\hat\sigma$ (simulated)")
    ax.loglog(spans, sigma / np.sqrt(2 * 252 * spans) * 100, color=COLORS[0], ls=":", label=r"$\sigma/\sqrt{2n}$")
    ax.axhline(mu * 100, color="0.6", ls="--", lw=1)
    ax.text(1.05, mu * 100 * 1.12, r"true $\mu$ = 8%", color="0.4", fontsize=8.5)
    ax.set_xlabel("years of daily data")
    ax.set_ylabel("standard error, % per year")
    ax.set_title(r"Longer data: $\hat\sigma$ is precise, $\hat\mu$ is not")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax = axes[1]
    per_year = np.array([1, 12, 52, 252, 252 * 13, 252 * 78])
    labels = ["yearly", "monthly", "weekly", "daily", "30 min", "5 min"]
    ax.loglog(per_year, np.full(len(per_year), sigma) * 100, "o-", color=COLORS[3], ms=4, label=r"sd of $\hat\mu$, one year")
    ax.loglog(per_year, sigma / np.sqrt(2 * per_year) * 100, "o-", color=COLORS[0], ms=4, label=r"sd of $\hat\sigma$, one year")
    ax.set_xticks(per_year)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("sampling frequency over one year")
    ax.set_ylabel("standard error, % per year")
    ax.set_title(r"Finer sampling helps $\hat\sigma$ and not $\hat\mu$")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(*axes)
    save(fig, "l03-estimating-mu-and-sigma.png")
    REPORT["estimation"] = dict(spans=spans.tolist(), sd_mu=[float(v) for v in sd_mu], sd_sig=[float(v) for v in sd_sig])
    simple, logr = load_voo()
    years = len(logr) / 252
    REPORT["voo_mu_se"] = float(logr.std() * np.sqrt(252) / np.sqrt(years))


def main():
    fig_random_walk()
    fig_quadratic_variation()
    fig_fbm()
    fig_ito_integral()
    fig_gbm()
    fig_euler()
    fig_correlation()
    fig_girsanov()
    fig_estimation()
    print(json.dumps(REPORT, indent=1))


if __name__ == "__main__":
    main()
