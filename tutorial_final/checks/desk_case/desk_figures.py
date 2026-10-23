"""Figures for the desk case, shared by desk_case_figure.py and the notebook.

Every function draws on axes it is given and reads only the cached results in this folder:
desk_case_combined.npz/json (or desk_case.npz/json), profile_h.json, validate_*.json, and the full-surface posterior
of rough_spy_is.py. load() gathers them into one dictionary.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist
from scipy.stats import gaussian_kde, norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sbi4finance.calibration import surface  # noqa: E402
from sbi4finance.calibration.plots import BASELINE, INK, INK_2, SERIES  # noqa: E402

POST, GN, FIX = SERIES
PRODUCT_LABELS = {"autocallable": "1-year autocallable", "forward_start": "6m→1y forward start",
                  "put_1y_80": "1-year 80% put", "cliquet": "1-year capped cliquet"}
METHOD_LABELS = {"posterior": "posterior", "delta": "Gauss–Newton, delta method", "profile": "profile likelihood",
                 "h_fixed_delta": "H fixed at 0.10, delta method", "truncated_gaussian": "Gauss–Newton normal, priced"}
METHOD_COLORS = {"posterior": POST, "delta": GN, "profile": INK_2, "h_fixed_delta": FIX, "truncated_gaussian": GN}
Z_ONE = 1.2815515655446004
NAMES = ("hurst", "eta", "rho")
SYMBOLS = {"hurst": "H", "eta": "η", "rho": "ρ"}
BOUNDS = {"hurst": (0.02, 0.35), "eta": (0.5, 3.5), "rho": (-0.99, -0.2)}


def load(folder=HERE):
    folder = Path(folder)
    name = "desk_case_combined" if (folder / "desk_case_combined.json").exists() else "desk_case"
    out = {"report": json.loads((folder / f"{name}.json").read_text()), "arrays": dict(np.load(folder / f"{name}.npz")),
           "round1": json.loads((folder / "desk_case.json").read_text()),
           "round1_arrays": dict(np.load(folder / "desk_case.npz")), "source": name}
    if (folder / "profile_h.json").exists():
        out["profile"] = json.loads((folder / "profile_h.json").read_text())
    out["validation"] = {p.stem: json.loads(p.read_text())["report"] for p in sorted(folder.glob("validate_*.json"))}
    if (folder / "real_market.json").exists():
        out["real_market"] = json.loads((folder / "real_market.json").read_text())
    spy = json.loads((ROOT / "tutorial_final/results/calibration/spy/surface.json").read_text())
    out["spy"] = spy
    full = np.load(HERE.parent / "rough_spy_is" / "rough_spy_is_seed0.npz")
    out["full"] = {"theta": full["theta"], "w": full["w_post"], "theta_hat": full["theta_hat"],
                   "cov": np.asarray(json.loads((HERE.parent / "rough_spy_is" / "rough_spy_is_seed0.json").read_text())
                                     ["gauss_newton_covariance"])}
    return out


def weights(data, scale):
    """Posterior weights of the cached proposals at a noise scale (front-month quotes, flat prior on the box)."""
    a = data["arrays"]
    sigma = np.asarray(data["spy"]["noise_sd"])
    y = np.asarray(data["spy"]["prices_forward_units"])
    front = front_rows(data)
    c2 = np.sum(((a["prices"][:, front] - y[front]) / sigma[front]) ** 2, axis=1)
    log_w = -0.5 * c2 / scale**2 - a["log_q"]
    w = np.exp(log_w - log_w.max())
    return w / w.sum(), c2


def front_rows(data):
    days = np.asarray(data["spy"]["design"]["days"])
    return np.repeat(np.isin(days, [21, 42, 63]), len(data["spy"]["design"]["deltas"]))


def wq(values, w, levels):
    order = np.argsort(values)
    cdf = np.cumsum(w[order]) / np.sum(w)
    return np.array([values[order[min(np.searchsorted(cdf, p), len(cdf) - 1)]] for p in levels])


def reserves(data, scale, level=0.9):
    """Long and short reserves at a one-sided confidence level, per product and method, in price units."""
    a, s = data["arrays"], data["report"]["scales"][f"{scale:g}" if scale != 1 else "1.0"]
    w, c2 = weights(data, scale)
    z = norm.ppf(level)
    fit = data["report"]["fit_front"]
    region = c2 / scale**2 - min(c2.min(), fit["chi2"]) / scale**2 <= z**2
    cov_f, cov_x = np.asarray(s["gauss_newton_covariance"]), np.asarray(s["h_fixed_covariance"])
    grad_f, grad_x = data["round1_arrays"]["grad_front"], data["round1_arrays"]["grad_h_fixed"]
    marks = data["round1_arrays"]["mark_front"]
    out = {}
    for j, n in enumerate(PRODUCT_LABELS):
        m = marks[j]
        q = wq(a["products"][:, j], w, (1 - level, level))
        d = z * np.sqrt(grad_f[j] @ cov_f @ grad_f[j])
        x = z * np.sqrt(grad_x[j] @ cov_x @ grad_x[j])
        prof = a["products"][region, j]
        out[n] = {"mark": m, "posterior": (m - q[0], q[1] - m), "delta": (d, d), "h_fixed_delta": (x, x),
                  "profile": (m - prof.min(), prof.max() - m), "mark_h_fixed": data["round1_arrays"]["mark_h_fixed"][j]}
    return out


def quotes_panel(ax, data, scale=1.0):
    """The 27 front-month quotes as implied vols against strike, with the quote noise in vol points."""
    spy = data["spy"]
    days = spy["design"]["days"]
    iv = np.asarray(spy["diagnostics"]["implied_vol_grid"]).reshape(len(days), -1)
    k, t, _ = surface.contract_arrays(_contracts())
    k, t = np.asarray(k).reshape(len(days), -1), np.asarray(t).reshape(len(days), -1)
    sigma = np.asarray(spy["noise_sd"]).reshape(len(days), -1)
    sd = iv * np.sqrt(t)
    vega = norm.pdf((-k + 0.5 * sd**2) / sd) * np.sqrt(t)
    for c, d in enumerate((21, 42, 63)):
        i = days.index(d)
        ax.errorbar(100 * np.exp(k[i]), 100 * iv[i], yerr=100 * scale * sigma[i] / vega[i], marker="o", markersize=3.5,
                    linewidth=1.2, capsize=2, color=SERIES[c], label=f"{d} days")
    ax.set_xlabel("strike, % of the forward")
    ax.set_ylabel("implied vol, %")
    ax.set_ylim(8, 32)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("What the desk trusts: the front three monthly smiles\n(SPY, 21 Oct 2026; bars: quote noise)",
                 loc="left", fontsize=9)


def _contracts():
    task = json.loads((ROOT / "configs/tasks/calibration_rough_pilot.json").read_text())["task"]
    return task["context"]["contracts"]


def timeline_panel(ax):
    """Quoted maturities against the products' dates."""
    ax.axvspan(0, 3, color=BASELINE, alpha=0.5, linewidth=0)
    ax.text(1.5, 4.6, "quoted\nsmiles", ha="center", va="center", fontsize=8, color=INK_2)
    rows = [("autocallable", [3, 6, 9, 12], "o"), ("forward start", [6, 12], "s"), ("80% put", [12], "v"),
            ("cliquet", list(range(1, 13)), "|")]
    for y, (name, months, marker) in enumerate(rows[::-1]):
        ax.plot(months, [y] * len(months), marker=marker, linestyle="-" if name == "forward start" else ":",
                color=INK, markersize=6 if marker != "|" else 10, linewidth=1)
        ax.text(12.4, y, name, va="center", fontsize=8)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(-0.7, 5.3)
    ax.set_yticks([])
    ax.set_xticks([0, 1, 2, 3, 6, 9, 12])
    ax.set_xlabel("months")
    ax.set_title("What the desk must mark: products to one year", loc="left", fontsize=9)


def profile_panel(ax, data):
    """chi-square of the front-month fit along H (left), and the forward start along the same ridge (right)."""
    rows = data["profile"]["rows"]
    h = np.array([r["h"] for r in rows])
    c2 = np.array([r["chi2_front"] for r in rows])
    fs = np.array([r["products"][1] for r in rows])
    mark = data["round1_arrays"]["mark_front"][1]
    ax.plot(h, c2 - c2.min(), color=INK, marker="o", markersize=3.5, linewidth=1.4, label="Δχ² of the front months")
    for level, style in ((Z_ONE**2, ":"), (chi2_dist.ppf(0.9, 1), "--")):
        ax.axhline(level, color=INK_2, linewidth=0.8, linestyle=style)
    ax.text(0.305, Z_ONE**2, " 1.64", va="center", fontsize=7, color=INK_2)
    ax.text(0.305, chi2_dist.ppf(0.9, 1), " 2.71", va="center", fontsize=7, color=INK_2)
    ax.set_ylim(0, 12)
    ax.set_xlim(0.02, 0.33)
    ax.set_xlabel("H, with η and ρ refitted to the front months")
    ax.set_ylabel("Δχ²")
    full_h = data["full"]["theta_hat"][0]
    ax.axvline(full_h, color=INK_2, linewidth=0.8)
    ax.text(full_h, 11.5, " full surface", fontsize=7, color=INK_2, va="top")
    ax.axvline(0.10, color=FIX, linewidth=1.0, linestyle="--")
    ax.text(0.10, 10.6, " desk: H = 0.10", fontsize=7, color=FIX, va="top")
    right = ax.twinx()
    right.plot(h, 100 * (fs / mark - 1), color=POST, marker="s", markersize=3.5, linewidth=1.4)
    right.axhline(0, color=POST, linewidth=0.6, alpha=0.5)
    right.set_ylabel("forward start, % from the mark", color=POST)
    right.tick_params(axis="y", colors=POST)
    right.set_ylim(-45, 45)
    ax.set_title("Along H the fit barely changes; the forward start does", loc="left", fontsize=9)


def marginal_panel(ax, data, i, scale=1.0):
    a = data["arrays"]
    w, _ = weights(data, scale)
    s = data["report"]["scales"]["1.0" if scale == 1 else f"{scale:g}"]
    cov = np.asarray(s["gauss_newton_covariance"])
    mu, sd = data["report"]["fit_front"]["theta"][i], np.sqrt(cov[i, i])
    name = NAMES[i]
    lo, hi = BOUNDS[name]
    span = (max(lo - 0.15 * (hi - lo), mu - 4.5 * sd), min(hi + 0.15 * (hi - lo), mu + 4.5 * sd))
    bins = np.linspace(*span, 36)
    ax.hist(a["theta"][:, i], bins=bins, weights=w / np.diff(bins)[0], color=POST, alpha=0.35, label="posterior")
    g = np.linspace(*span, 300)
    ax.plot(g, norm.pdf(g, mu, sd), color=GN, linewidth=1.6, label="Gauss–Newton")
    for edge in (lo, hi):
        if span[0] < edge < span[1]:
            ax.axvspan(span[0] if edge == lo else edge, edge if edge == lo else span[1], color=BASELINE, alpha=0.5,
                       linewidth=0)
    ax.axvline(data["full"]["theta_hat"][i], color=INK_2, linewidth=0.8)
    top = ax.get_ylim()[1]
    q = wq(a["theta"][:, i], w, (0.05, 0.95))
    ax.hlines(0.95 * top, *q, color=POST, linewidth=3)
    ax.hlines(0.88 * top, mu - 1.645 * sd, mu + 1.645 * sd, color=GN, linewidth=3)
    ax.set_xlim(*span)
    ax.set_yticks([])
    ax.set_title(f"{SYMBOLS[name]}: 90% posterior {q[0]:.2f} to {q[1]:.2f}, Gauss–Newton {mu - 1.645 * sd:.2f} to "
                 f"{mu + 1.645 * sd:.2f}", loc="left", fontsize=8.5)


def joint_panel(ax, data, i=0, j=2, scale=1.0):
    """A pair's posterior with its 90% region, the Gauss–Newton ellipse of the same pair and the profile path."""
    a = data["arrays"]
    w, _ = weights(data, scale)
    xy = a["theta"][:, [i, j]].T
    keep = w > 0.3 / len(w)
    ax.scatter(xy[0, keep], xy[1, keep], s=5, color=POST, alpha=0.3, linewidths=0)
    kde = gaussian_kde(xy, weights=w)
    dens = kde(xy)
    order = np.argsort(dens)[::-1]
    level = dens[order][np.searchsorted(np.cumsum(w[order]), 0.9)]
    s = data["report"]["scales"]["1.0" if scale == 1 else f"{scale:g}"]
    cov = np.asarray(s["gauss_newton_covariance"])[np.ix_([i, j], [i, j])]
    mean = np.asarray(data["report"]["fit_front"]["theta"])[[i, j]]
    t = np.linspace(0, 2 * np.pi, 200)
    vals, vecs = np.linalg.eigh(cov)
    e = mean[:, None] + vecs @ (np.sqrt(chi2_dist.ppf(0.9, 2) * vals)[:, None] * np.stack([np.cos(t), np.sin(t)]))
    limits = []
    for k, c in enumerate((i, j)):
        lo, hi = BOUNDS[NAMES[c]]
        q = wq(a["theta"][:, c], w, (0.002, 0.998))
        span = max(q[1], e[k].max()) - min(q[0], e[k].min())
        limits.append((max(lo - 0.04 * (hi - lo), min(q[0], e[k].min()) - 0.1 * span),
                       min(hi + 0.04 * (hi - lo), max(q[1], e[k].max()) + 0.1 * span)))
    gx, gy = np.linspace(*limits[0], 140), np.linspace(*limits[1], 140)
    X, Y = np.meshgrid(gx, gy)
    Z = kde(np.stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    ax.contour(X, Y, Z, levels=[level], colors=[POST], linewidths=1.6)
    ax.plot(e[0], e[1], color=GN, linewidth=1.4, linestyle="--")
    if "profile" in data and i == 0:
        rows = data["profile"]["rows"]
        ax.plot([r["h"] for r in rows], [r["theta"][j] for r in rows], color=INK, linewidth=0.9, marker=".",
                label="profile path")
    ax.plot(*data["full"]["theta_hat"][[i, j]], marker="*", color=INK_2, markersize=11, linestyle="none",
            label="full-surface fit")
    ax.plot(*mean, marker="+", color=INK, markersize=10, linestyle="none")
    for k, (c, setter) in enumerate(((i, ax.axvspan), (j, ax.axhspan))):
        lo, hi = BOUNDS[NAMES[c]]
        if limits[k][0] < lo:
            setter(limits[k][0], lo, color=BASELINE, alpha=0.6, linewidth=0)
        if limits[k][1] > hi:
            setter(hi, limits[k][1], color=BASELINE, alpha=0.6, linewidth=0)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_xlabel(SYMBOLS[NAMES[i]])
    ax.set_ylabel(SYMBOLS[NAMES[j]])
    ax.legend(fontsize=7, loc="best")
    ax.set_title(f"({SYMBOLS[NAMES[i]]}, {SYMBOLS[NAMES[j]]}): 90% regions, posterior (blue) and Gauss–Newton\n"
                 f"(orange, dashed); black: the profile path", loc="left", fontsize=8.5)


def reserves_panel(ax, data, scale=1.0, level=0.9, products=("forward_start", "put_1y_80", "cliquet")):
    r = reserves(data, scale, level)
    methods = ("posterior", "delta", "profile", "h_fixed_delta")
    labels, y0 = [], 0
    for n in products:
        for s, side in enumerate(("long", "short")):
            for k, m in enumerate(methods):
                ax.barh(y0 - 0.2 * k, 100 * r[n][m][s] / r[n]["mark"], height=0.19, color=METHOD_COLORS[m],
                        label=METHOD_LABELS[m] if (n == products[0] and s == 0) else None)
            labels.append((y0 - 0.3, f"{PRODUCT_LABELS[n]}, {side}"))
            y0 -= 1.0
    ax.set_yticks([p for p, _ in labels])
    ax.set_yticklabels([t for _, t in labels], fontsize=7.5)
    ax.set_xlabel(f"one-sided {100 * level:.0f}% reserve, % of the mark")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"Reserves from the same 27 quotes (noise ×{scale:g})", loc="left", fontsize=9)


def market_panel(ax, data, j=1, scale=1.0):
    """One product: the posterior predictive from the front months, the Gaussian methods, and the full surface."""
    a = data["arrays"]
    w, _ = weights(data, scale)
    r = reserves(data, scale, 0.9)[list(PRODUCT_LABELS)[j]]
    m = r["mark"]
    vals = a["products"][:, j] / m
    full = data["round1_arrays"]["full_products"][:, j] / m
    bins = np.linspace(0.6, 1.45, 60)
    ax.hist(vals, bins=bins, weights=w / np.diff(bins)[0], color=POST, alpha=0.35, label="posterior, front months")
    ax.hist(full, bins=bins, weights=data["full"]["w"] / np.diff(bins)[0], color=INK_2, alpha=0.5,
            label="posterior, full surface")
    g = np.linspace(*bins[[0, -1]], 400)
    sd_d = r["delta"][0] / Z_ONE / m
    ax.plot(g, norm.pdf(g, 1.0, sd_d), color=GN, linewidth=1.5, label="Gauss–Newton, delta")
    mx, sd_x = r["mark_h_fixed"] / m, r["h_fixed_delta"][0] / Z_ONE / m
    ax.plot(g, norm.pdf(g, mx, sd_x), color=FIX, linewidth=1.5, label="H fixed at 0.10")
    ax.axvline(1.0, color=INK, linewidth=0.8)
    ax.set_yticks([])
    ax.set_xlabel(f"{PRODUCT_LABELS[list(PRODUCT_LABELS)[j]]}, price / front-month mark")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("The full surface lies outside every front-month range", loc="left", fontsize=9)


def _scale_key(scale):
    return "1.0" if scale == 1 else f"{scale:g}"


def reserve_rows(data, scale=1.0, level=0.9):
    r = reserves(data, scale, level)
    rows = []
    for n, e in r.items():
        for s, side in enumerate(("long", "short")):
            rows.append({"product": PRODUCT_LABELS[n], "side": side,
                         **{METHOD_LABELS[m]: f"{100 * e[m][s] / e['mark']:.2f}%"
                            for m in ("posterior", "delta", "profile", "h_fixed_delta")}})
    return rows


def validation_summary(data, scale=1.0):
    """Breach rates per method, side and product, pooled over the validation batches at a noise scale."""
    runs = [v for v in data["validation"].values() if abs(v["noise_scale"] - scale) < 1e-9 and v.get("truths") == "thin"]
    if not runs:
        return None
    n = sum(v["replicas"] for v in runs)
    methods = runs[0]["breach_rate"].keys()
    out = {"replicas": n, "rates": {}, "ess_median": float(np.median([v["ess"]["median"] for v in runs]))}
    for m in methods:
        out["rates"][m] = {side: {p: sum(v["breach_rate"][m][side][p] * v["replicas"] for v in runs) / n
                                  for p in runs[0]["breach_rate"][m][side]} for side in ("long", "short")}
    return out


def validation_panel(axes, data, scale=1.0, products=("forward_start", "put_1y_80", "cliquet")):
    v = validation_summary(data, scale)
    methods = ("posterior", "gauss_newton", "profile", "h_fixed")
    colors = {"posterior": POST, "gauss_newton": GN, "profile": INK_2, "h_fixed": FIX}
    labels = {"posterior": "posterior", "gauss_newton": "Gauss–Newton, delta", "profile": "profile likelihood",
              "h_fixed": "H fixed at 0.10"}
    for ax, side in zip(axes, ("long", "short")):
        for k, m in enumerate(methods):
            rates = np.array([v["rates"][m][side][p] for p in products])
            se = np.sqrt(np.maximum(rates * (1 - rates), 0.1 * 0.9) / v["replicas"])
            y = np.arange(len(products))[::-1] + 0.18 * (1.5 - k)
            ax.errorbar(100 * rates, y, xerr=100 * se, fmt="o", color=colors[m], label=labels[m], markersize=5,
                        capsize=2)
        ax.axvline(10, color=INK, linewidth=0.8, linestyle="--")
        ax.text(10.5, len(products) - 0.45, "nominal 10%", fontsize=7, color=INK)
        ax.set_yticks(np.arange(len(products))[::-1])
        ax.set_yticklabels([PRODUCT_LABELS[p] for p in products], fontsize=8)
        ax.set_xlim(-2, 90)
        ax.set_ylim(-0.6, len(products) - 0.3)
        ax.set_xlabel(f"breach rate of the {side} 90% reserve, %")
        move = "falls below" if side == "long" else "rises above"
        ax.set_title(f"{side} positions: how often the true price {move} the prudent value", loc="left",
                     fontsize=8.5)
    axes[0].legend(fontsize=7, loc="lower right")


def validation_view(data, mo, scale=1.0):
    v = validation_summary(data, scale)
    if v is None:
        return mo.callout(mo.md("**Pending:** the validation on simulated thin chains (validate.py)."), kind="warn")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), constrained_layout=True, sharey=True)
    validation_panel(axes, data, scale)
    rows = []
    for m, sides in v["rates"].items():
        for side, per in sides.items():
            rows.append({"method": m.replace("_", " "), "side": side,
                         **{PRODUCT_LABELS[p]: f"{100 * r:.0f}%" for p, r in per.items()}})
    return mo.vstack([fig, mo.md(f"{v['replicas']} simulated chains; median effective sample size of the posterior "
                                 f"weights {v['ess_median']:.0f}. Bars: binomial standard errors."),
                      mo.ui.table(rows, selection=None, pagination=False)])


def key_numbers(data, scale=1.0):
    rep = data["report"]
    s = rep["scales"][_scale_key(scale)]
    p = s["parameters"]
    h_post, h_gn = p["hurst"]["posterior_q05_q50_q95"], p["hurst"]["gauss_newton_90"]
    r_post, r_gn = p["rho"]["posterior_q05_q50_q95"], p["rho"]["gauss_newton_90"]
    res = reserves(data, scale, 0.9)
    pct = {n: {m: (100 * e[m][0] / e["mark"], 100 * e[m][1] / e["mark"])
               for m in ("posterior", "delta", "profile", "h_fixed_delta")} for n, e in res.items()}
    e_post, e_gn = p["eta"]["posterior_q05_q50_q95"], p["eta"]["gauss_newton_90"]
    out = {"h_post": (h_post[0], h_post[2]), "h_gn": tuple(h_gn), "rho_post": (r_post[0], r_post[2]),
           "eta_post": (e_post[0], e_post[2]), "eta_gn": tuple(e_gn),
           "width_diff": sorted(100 * abs((q[2] - q[0]) / (g[1] - g[0]) - 1) for q, g in
                                ((h_post, h_gn), (e_post, e_gn), (r_post, r_gn))),
           "rho_gn": tuple(r_gn), "rho_ratio": (r_post[2] - r_post[0]) / (r_gn[1] - r_gn[0]),
           "ls": rep["fit_front"]["theta"], "ls_chi2": rep["fit_front"]["chi2"], "khat": s["importance"]["khat"],
           "ess": s["importance"]["ess"], "proposals": rep["proposals"], "fs": pct["forward_start"],
           "put": pct["put_1y_80"], "cliquet": pct["cliquet"], "autocall": pct["autocallable"],
           "khat_round1": data["round1"]["scales"]["1.0"]["importance"]["khat"],
           "full_theta": data["full"]["theta_hat"]}
    out["full_gap"] = 100 * (rep["full_surface"]["forward_start"][2] / res["forward_start"]["mark"] - 1)
    if "real_market" in data:
        rm = data["real_market"]
        out.update(full_chi2_front=rm["full_surface"]["chi2_front_27"], full_chi2_other=rm["full_surface"]["chi2_other_36"],
                   front_chi2_other=rm["front_months"]["chi2_other_36"])
    if "profile" in data:
        rows = data["profile"]["rows"]
        h = np.array([r["h"] for r in rows])
        c2 = np.array([r["chi2_front"] for r in rows])
        fs = np.array([r["products"][1] for r in rows]) / res["forward_start"]["mark"] - 1
        fine = np.linspace(h[0], h[-1], 2000)
        inside = np.interp(fine, h, c2 - c2.min()) <= Z_ONE**2
        lo, hi = fine[inside][0], fine[inside][-1]
        out["profile_flat"] = (lo, hi)
        out["profile_fs"] = (100 * np.interp(lo, h, fs), 100 * np.interp(hi, h, fs))
    surfaces = rep["proposals"] + 60
    out["cpu_hours"] = (4.2 * surfaces + 0.56 * rep["proposals"]) / 3600
    v = validation_summary(data, scale)
    if v is None:
        out["replicas"] = 0
        out["validation_claim"] = "**The reserves' coverage is tested on simulated thin chains** (section 5, pending)."
    else:
        out["replicas"] = v["replicas"]
        prods = ("forward_start", "put_1y_80", "cliquet")

        def mean_rate(m):
            return 100 * np.mean([v["rates"][m][side][q] for side in ("long", "short") for q in prods])

        out["validation_rates"] = {m: mean_rate(m) for m in v["rates"]}
        post = [100 * v["rates"]["posterior"][side][q] for side in ("long", "short") for q in prods]
        gn_long = [100 * v["rates"]["gauss_newton"]["long"][q] for q in ("forward_start", "put_1y_80")]
        gn_short = [100 * v["rates"]["gauss_newton"]["short"][q] for q in ("forward_start", "put_1y_80")]
        prof_long = [100 * v["rates"]["profile"]["long"][q] for q in prods]
        widths = {m: np.mean([data["validation"][b]["mean_interval_width_over_truth"][m]["forward_start"]
                              for b in data["validation"] if data["validation"][b].get("truths") == "thin"
                              and abs(data["validation"][b]["noise_scale"] - scale) < 1e-9])
                  for m in ("posterior", "gauss_newton")}
        out["validation"] = {"posterior_range": (min(post), max(post)), "posterior_mean": np.mean(post),
                             "gn_long": (min(gn_long), max(gn_long)), "gn_short": (min(gn_short), max(gn_short)),
                             "profile_long": (min(prof_long), max(prof_long)),
                             "h_fixed_fs_long": 100 * v["rates"]["h_fixed"]["long"]["forward_start"],
                             "h_fixed_put_long": 100 * v["rates"]["h_fixed"]["long"]["put_1y_80"],
                             "narrower": 100 * (1 - widths["posterior"] / widths["gauss_newton"])}
        x = out["validation"]
        out["validation_claim"] = (
            f"**The posterior's reserves hold; the others do not.** On {v['replicas']} simulated thin chains, the "
            f"posterior's 90% reserves are breached {x['posterior_range'][0]:.0f}% to {x['posterior_range'][1]:.0f}% "
            f"of the time. Gauss–Newton's long reserves for the forward start and the put are breached "
            f"{x['gn_long'][0]:.0f}% to {x['gn_long'][1]:.0f}% of the time, and its short ones "
            f"{x['gn_short'][1]:.0f}%. With H fixed, the forward start's long reserve is breached "
            f"{x['h_fixed_fs_long']:.0f}% of the time.")
    return out
