"""Dated market motivation beside an explicit, uncalibrated Bergomi example."""

import numpy as np

from . import simulation


def model_example():
    """A fixed illustrative law and seed, never fitted to the displayed market."""
    task = simulation.teaching_task("rough-known", observations=1260)
    values = {"hurst": .12, "eta": 1.2, "rho": -.7, "drift": .07}
    parameters = np.asarray([[values[name] for name in task["parameter_names"]]] * 16)
    returns = simulation.simulate_parameters(task, parameters, seed=20261114, backend="numpy")
    if returns.shape != (16, 1260, 1) or not np.isfinite(returns).all():
        raise ValueError("the complete illustrative Bergomi panel must remain finite")
    return {"task": task, "parameters": values, "returns": returns[:, :, 0],
            "seed": 20261114, "scope": "fixed illustrative parameters; not fitted to VOO"}


def absolute_survival(values):
    """Actual nonzero empirical survival probabilities, retaining every return."""
    values = np.asarray(values)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("provide a finite return history with at least two observations")
    magnitudes = np.sort(np.abs(values))
    thresholds, first = np.unique(magnitudes, return_index=True)
    return thresholds, (len(values) - first) / len(values)


def comparison_figure(market, model):
    """Identical trace/tail layouts distinguish data from one simulated law."""
    import matplotlib.pyplot as plt
    from scipy.special import ndtr

    observed = np.asarray(market["log_returns"])
    simulated = np.asarray(model["returns"])
    if (observed.ndim != 1 or len(observed) < 2 or simulated.ndim != 2
            or simulated.shape[1] != 1260 or not np.isfinite(observed).all()
            or not np.isfinite(simulated).all()):
        raise ValueError("market and complete simulated histories must be finite")
    figure, axes = plt.subplots(2, 2, figsize=(11, 5.8),
                                gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained")
    floor = .5 / max(len(observed), simulated.shape[1])
    grid = np.linspace(0, .15, 301)
    dates = np.asarray(market["dates"]).astype("datetime64[D]")
    rows = [(observed, dates, "#334155", "VOO observed returns", "Trading date"),
            (simulated[0], np.arange(1, 1261) / 252, "#7c3aed",
             "Rough Bergomi · one of 16 independent histories", "Years since this simulation's origin")]
    for row, (values, time, color, title, xlabel) in enumerate(rows):
        left, right = axes[row]
        left.plot(time, 100 * np.clip(values, -.15, .15), color=color, linewidth=.65)
        outside = np.abs(values) > .15
        if outside.any():
            left.scatter(time[outside], 100 * np.clip(values[outside], -.15, .15),
                         color="#c2410c", s=9, clip_on=False)
        left.axhline(0, color="#94a3b8", linewidth=.6)
        left.set(ylim=(-15, 15), ylabel="Daily log return (%)", xlabel=xlabel, title=title)
        left.text(.02, .96, f"{len(values):,} daily returns · {outside.sum()} outside fixed ±15% view",
                  transform=left.transAxes, va="top", fontsize=7)
        thresholds, survival = absolute_survival(values)
        right.step(100 * thresholds, survival, where="post", color=color,
                   label="Observed VOO" if row == 0 else "This simulated history")
        mean, sd = float(values.mean()), float(values.std())
        if sd <= 0:
            raise ValueError("Gaussian comparison requires nonconstant returns")
        normal = ndtr((-grid - mean) / sd) + ndtr((mean - grid) / sd)
        positive = normal > 0
        right.plot(100 * grid[positive], normal[positive], color="#0f766e", linestyle="--",
                   label="Gaussian with this history's mean / SD")
        right.set(xlim=(0, 15), yscale="log", ylim=(floor, 1.1),
                  xlabel="Absolute daily log return (%)", ylabel="Fraction at least this large",
                  title="Same tail-frequency view")
        if row == 1:
            tails = np.mean(np.abs(simulated[:, :, None]) >= grid[None, None, :], axis=1)
            low, high = np.quantile(tails, [.1, .9], axis=0)
            right.fill_between(100 * grid, low, high, where=(low > 0) & (high > 0),
                               color="#7c3aed", alpha=.13, label="Pointwise 10–90% across 16 histories")
        right.legend(fontsize=6.7, loc="upper right")
        for axis in (left, right):
            axis.tick_params(labelsize=7)
            axis.title.set_fontsize(9)
    figure.suptitle(f"Market versus model · VOO {market['dates'][0]} to {market['dates'][-1]}\n"
                   "Illustrative Bergomi; Gaussian benchmarks match each history's mean and SD", fontsize=10)
    return figure
