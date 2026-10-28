import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sbi4finance.calibration.plots import BASELINE, INK, SERIES, plt, style
import long_end as L
style()
r = json.load(open("long_end.json")); x = json.load(open("long_end_extra.json"))
spy = L.SPY
fig = plt.figure(figsize=(13.0, 4.4), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.25, 1.0])
# A: variance-swap vol term structure
ax = fig.add_subplot(gs[0])
tt = np.linspace(0.02, 5.0, 200)
def vs_curve(log_xi):
    xi = np.exp(log_xi)
    w = np.array([sum(xi[j] * L.overlap(t, L.KNOTS[j], L.KNOTS[j + 1]) for j in range(len(xi))) for t in tt])
    return 100 * np.sqrt(w / tt)
mean = np.array(r["posterior_mean"]); L.S_PRIOR, L.VS_SD = 0.2, 0.005
_, cov = L.posterior(np.log(np.array(r["xi_flat"])))
draws = np.random.default_rng(1).multivariate_normal(mean[3:], cov[3:, 3:], 400)
curves = np.array([vs_curve(d) for d in draws])
lo, hi = np.quantile(curves, [0.05, 0.95], axis=0)
ax.fill_between(tt, lo, hi, color=SERIES[0], alpha=0.2, label="rough Bergomi posterior, 90%")
ax.plot(tt, vs_curve(np.log(np.array(r["xi_flat"]))), color=INK, linewidth=1.2, linestyle="--", label="flat beyond one year")
T = np.array(x["T"]); hq = np.array(x["heston_kappa_free_vs_vol_q"])
ax.fill_between(T, hq[0], hq[2], color=SERIES[1], alpha=0.35, label="Heston posterior (fitted to ≤ 1 year), 90%")
days = np.array(spy["design"]["days"]) / 252; wd = np.array(spy["diagnostics"]["design_total_variance"])
ax.plot(days, 100 * np.sqrt(wd / days), "o", color=INK, markersize=3.5, label="SPY replicated, design maturities")
lt = np.array([t for t, _ in L.LEAPS]); lv = np.array([100 * np.sqrt(w / t) for t, w in L.LEAPS])
ax.errorbar(lt, lv, yerr=1.645 * 0.5, fmt="s", color=SERIES[2], markersize=4, capsize=3, label="SPY replicated LEAPs (±0.5 pt sd)")
ax.axvspan(2.23, 5.0, color=BASELINE, alpha=0.12)
ax.text(3.6, 26.2, "no quotes beyond 2.2 years", ha="center", fontsize=8, color=INK)
ax.set_xlabel("maturity (years)"); ax.set_ylabel("variance-swap vol (%)"); ax.set_ylim(12, 27)
ax.set_title("SPY's variance term structure and its long end", loc="left", fontsize=9)
ax.legend(fontsize=6.3, loc="lower right", bbox_to_anchor=(1.0, 0.0))
# B: 90% ranges, flat extrapolation vs posterior, with the prior/noise sensitivity as whiskers
ax = fig.add_subplot(gs[1])
names = r["products"]; p = np.array(r["price_posterior_mean_inputs"]); pd = np.array(r["price_desk_point"])
desk = 100 * np.array(r["range90_desk_delta"]) / pd
post = 100 * np.array(r["range90_posterior_mc"]) / np.array(r["posterior_mc_quantiles"])[1]
sens = np.array(list(x["sensitivity_range90_percent"].values()))
labels = ["autocallable 3y\n(bp of notional ÷ 10)", "cliquet 3y", "forward start\n2y→3y", "put 5y 80%", "variance\nswap 3y"]
scale = np.array([10, 1, 1, 1, 1.0])  # autocallable shown in bp of notional / 10
xs = np.arange(len(names))
ax.bar(xs - 0.2, desk * scale, 0.38, color=BASELINE, label="flat extrapolation: H, η, ρ only, ξ0 flat")
ax.bar(xs + 0.2, post * scale, 0.38, color=SERIES[0], label="posterior: all inputs (prior sd 0.2)")
ax.vlines(xs + 0.2, sens.min(0) * scale, sens.max(0) * scale, color=INK, linewidth=1.2, label="prior sd 0.1–0.3, LEAP sd 0.25–1 pt")
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7)
for k in range(len(names)):
    unit = "bp" if k == 0 else "%"
    f = 100 if k == 0 else 1
    ax.text(xs[k] - 0.2, desk[k] * scale[k] + 0.6, f"{desk[k] * f / (100 if k == 0 else 1) * (100 if k == 0 else 1):.0f}{unit}" if k == 0 else f"{desk[k]:.0f}{unit}", ha="center", fontsize=6.5)
    ax.text(xs[k] + 0.2, post[k] * scale[k] + 0.6, f"{post[k] * 100:.0f}{unit}" if k == 0 else f"{post[k]:.0f}{unit}", ha="center", fontsize=6.5)
ax.set_ylabel("90% range, % of price")
ax.set_title("long-dated products: the 90% range", loc="left", fontsize=9)
ax.legend(fontsize=6.5, loc="upper left")
# C: variance shares
ax = fig.add_subplot(gs[2])
grad = np.array(r["gradient"])
groups = {"short end: H, η, ρ and ξ0 to 1 year (option quotes)": slice(0, 10),
          "ξ0 1 to 2.25 years (LEAP variance swaps)": slice(10, 12), "ξ0 2.25 to 5 years (prior only)": slice(12, 14)}
parts = np.array([np.einsum("pi,ij,pj->p", grad[:, g], cov[g, g], grad[:, g]) for g in groups.values()])
shares = parts / parts.sum(0)
bottom = np.zeros(len(names)); cols = [SERIES[0], SERIES[2], INK]
for (g, sh), c in zip(zip(groups, shares), cols):
    ax.bar(xs, 100 * sh, 0.6, bottom=bottom, color=c, label=g, alpha=0.85 if c != INK else 0.6)
    bottom += 100 * sh
print("merged shares %", {n: np.round(100 * shares[:, j], 0).tolist() for j, n in enumerate(names)}, "cross-group terms % of total:",
      np.round(100 * (1 - parts.sum(0) / np.einsum("pi,ij,pj->p", grad, cov, grad)), 1).tolist())
ax.set_xticks(xs); ax.set_xticklabels([l.split("\n")[0] for l in labels], fontsize=7, rotation=20)
ax.set_ylabel("share of the posterior variance (%)")
ax.set_title("where the range comes from", loc="left", fontsize=9)
ax.legend(fontsize=6.3, loc="upper left")
ax.set_ylim(0, 135)
fig.savefig("long_end.png", dpi=160)
