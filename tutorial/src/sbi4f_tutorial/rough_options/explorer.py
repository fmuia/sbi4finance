"""Small path and architecture illustrations for the rough-option lesson.

The path explorer calls the same hybrid driver as option pricing. It keeps
innovations fixed as parameters change, uses a supplied flat variance curve,
and adds no observation noise. These paths are never calibration conditions.
Plotting imports are lazy; importing this module does not load a neural stack.
"""

import numpy as np

from . import simulator


PURPLE = "#7755aa"
GREEN = "#2e9675"
INK = "#253340"
MUTED = "#667581"

# Manual viewing windows, shared by every parameter/seed setting. They are not
# asserted to contain all Gaussian/lognormal outcomes. Boundary markers and
# axis_statistics make values outside a selected window explicit.
AXIS_PRESETS = {
    "detail": {"driver": (-4., 4.), "variance": (1e-6, 1.),
               "relative_increments": (-.15, .15), "prices": (.4, 1.8),
               "volatility_percent": (0., 100.)},
    "wide": {"driver": (-6., 6.), "variance": (1e-12, 1e3),
             "relative_increments": (-1., 1.), "prices": (0., 4.),
             "volatility_percent": (0., 1000.)},
}


def _integer(value, name, lower, upper):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lower <= value <= upper):
        raise ValueError(f"{name} must be an integer between {lower} and {upper}")
    return int(value)


def explore_paths(hurst=.1, eta=2., rho=-.7, volatility=.2, *, seed=7, paths=24, steps=252):
    """One year of hybrid rough-Bergomi paths from the original Q origin.

    Volatility is the square root of the supplied mean variance, in annualized
    decimal units. The hybrid driver uses its discrete Gaussian variance in
    the exponential compensator, preserving E[V_i] = xi_0(t_i) on that grid.
    Asset prices retain the canonical left-point log-Euler approximation.

    A fixed seed and grid give the same three innovation channels for every
    parameter choice. Nonfinite paths raise; no clipping or redrawing occurs.
    """
    paths = _integer(paths, "paths", 1, 64)
    steps = _integer(steps, "steps", 1, 2048)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    if (isinstance(volatility, (bool, np.bool_)) or np.ndim(volatility) != 0
            or np.iscomplexobj(volatility) or not np.isfinite(volatility) or volatility <= 0):
        raise ValueError("volatility must be a positive finite annualized decimal")
    volatility = float(volatility)
    parameters = {"hurst": hurst, "eta": eta, "rho": rho}
    innovations = np.random.default_rng(int(seed)).standard_normal((paths, steps, 3))
    curve = {"times": [0., 1.], "values": [volatility**2, volatility**2]}
    dt = 1. / steps
    native = simulator.simulate_rough(parameters, curve, dt, innovations,
                                      method="hybrid", spot=1., rate=0., dividend=0.)
    with np.errstate(over="raise", invalid="raise"):
        try:
            prices = np.exp(native["log_spot"])
        except FloatingPointError as error:
            raise ValueError("asset-price overflow; no paths were redrawn") from error
    log_returns = native["log_returns"][..., 0]
    brownian = np.sqrt(dt) * innovations[..., [0, 2]]
    price_brownian = float(rho) * brownian[..., 0] + np.sqrt(1 - float(rho)**2) * brownian[..., 1]
    compensator = -.5 * float(eta)**2 * native["driver_variance"]
    log_drift = -.5 * native["variance"][:, :-1] * dt
    with np.errstate(over="raise", invalid="raise"):
        try:
            relative_increments = np.expm1(log_returns)
        except FloatingPointError as error:
            raise ValueError("relative-price increment overflow; no paths were redrawn") from error
    return {
        "times_years": native["times"], "prices": prices,
        "variances": native["variance"], "volatility": np.sqrt(native["variance"]),
        "forward_variance": np.full(steps + 1, volatility**2),
        "volterra_driver": native["volterra_driver"], "driver_variance": native["driver_variance"],
        "standard_normals": innovations, "brownian_increments": brownian,
        "price_brownian_increments": price_brownian,
        "log_returns": log_returns, "relative_increments": relative_increments,
        "asset_log_drift": log_drift, "asset_log_diffusion": log_returns - log_drift,
        "variance_compensator": compensator,
        "log_variance_multiplier": float(eta) * native["volterra_driver"] + compensator,
        "variance_multiplier": native["variance"] / volatility**2,
        "settings": {"parameters": {name: float(value) for name, value in parameters.items()},
                     "supplied_volatility": volatility, "seed": int(seed), "paths": paths,
                     "steps": steps, "horizon_years": 1., "rate": 0., "dividend": 0.,
                     "driver_method": "hybrid", "measure": "Q",
                     "dt_years": dt,
                     "normal_channels": ["variance-driving W", "independent last-cell residual", "orthogonal W-perp"],
                     "brownian_channels": ["W", "W-perp"],
                     "relative_increment_definition": "expm1(canonical log return); a finite-step change, not dS/S itself",
                     "compensator": "discrete Gaussian-driver variance",
                     "scope": "illustrative price and hidden-variance paths; no quote noise or inference"},
    }


def _components(result, view):
    times = np.asarray(result["times_years"])
    prices = np.asarray(result["prices"])
    if (times.ndim != 1 or len(times) < 2 or prices.ndim != 2 or not len(prices)
            or prices.shape[1] != len(times) or not np.isfinite(times).all()
            or times[0] != 0 or times[-1] != 1 or np.any(np.diff(times) <= 0)):
        raise ValueError("matching one-year price paths on a finite increasing grid are required")
    if view == "paths":
        items = [("prices", times, prices, "Price / initial price", "Possible price histories"),
                 ("volatility_percent", times, 100 * np.asarray(result["volatility"]),
                  "Annualized volatility (%)", "Their hidden volatility")]
    elif view == "components":
        items = [("driver", times, np.asarray(result["volterra_driver"]), r"Gaussian driver $Y_t$",
                  r"1 · Rough Gaussian driver $Y_t$"),
                 ("variance", times, np.asarray(result["variances"]), "Annualized variance (log scale)",
                  r"2 · Instantaneous variance $V_t$"),
                 ("relative_increments", times[1:], np.asarray(result["relative_increments"]),
                  "One-step relative price change (%)", r"3 · Finite-step change $\exp(\Delta\log S)-1$"),
                 ("prices", times, prices, "Price / initial price", r"4 · Accumulated price $S_t/S_0$")]
    else:
        raise ValueError("view must be 'paths' or 'components'")
    for name, grid, values, _, _ in items:
        if (values.shape != (len(prices), len(grid)) or not np.isfinite(values).all()
                or (name in ("variance", "volatility_percent") and np.any(values < 0))):
            raise ValueError("complete finite components aligned with the price paths are required")
    return items


def axis_statistics(result, *, preset="detail", view="components", highlighted_path=0):
    """Count the actual coordinates outside manual axes without changing data.

    Every row contributes: no extreme trajectory is dropped or resimulated.
    Counts describe plotted coordinates, not a probability estimate. Fixed
    presets do not depend on parameters, the seed or the returned values.
    """
    if not isinstance(preset, str) or preset not in AXIS_PRESETS:
        raise ValueError("preset must be 'detail' or 'wide'")
    items = _components(result, view)
    chosen = _integer(highlighted_path, "highlighted_path", 0, len(items[0][2]) - 1)
    rows = []
    for name, _, values, label, _ in items:
        lower, upper = AXIS_PRESETS[preset][name]
        below, above = values < lower, values > upper
        rows.append({"component": name, "label": label, "lower": lower, "upper": upper,
                     "scale": "log" if name == "variance" else "linear",
                     "points": int(values.size), "below": int(below.sum()), "above": int(above.sum()),
                     "outside": int((below | above).sum()),
                     "highlighted_outside": int((below[chosen] | above[chosen]).sum())})
    return {"preset": preset, "view": view, "components": rows,
            "points_outside": sum(row["outside"] for row in rows),
            "scope": "Counts of coordinates outside a manual viewing window; all original data are retained."}


def _draw_paths(ax, times, values, chosen, bounds, *, logarithmic=False):
    """Plot raw arrays; mark points beyond manual windows at their boundary."""
    lower, upper = bounds
    ax.set_facecolor("#fbfafc")
    if logarithmic:
        ax.set_yscale("log")
    for i, path in enumerate(values):
        if i != chosen:
            ax.plot(times, path, color=PURPLE, alpha=.18, linewidth=.7)
    ax.plot(times, values[chosen], color=GREEN, linewidth=1.8, label="Same highlighted path")
    marked = False
    for selected, color, alpha in [(np.arange(len(values)) != chosen, PURPLE, .22),
                                    (np.arange(len(values)) == chosen, GREEN, .9)]:
        group = values[selected]
        for mask, boundary, marker in [(group < lower, lower, "v"), (group > upper, upper, "^")]:
            _, columns = np.nonzero(mask)
            if len(columns):
                ax.scatter(times[columns], np.full(len(columns), boundary), marker=marker,
                           s=10, color=color, alpha=alpha, clip_on=False, zorder=4,
                           label="Beyond fixed view" if not marked else None)
                marked = True
    ax.set_xlim(0., 1.)
    ax.set_ylim(lower, upper)
    ax.grid(color="#dfe4e8", alpha=.65, linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def _figure(result, preset, view, highlighted_path):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    stats = axis_statistics(result, preset=preset, view=view, highlighted_path=highlighted_path)
    items = _components(result, view)
    chosen = int(highlighted_path)
    if view == "components":
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), layout="constrained")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for ax, (name, times, values, ylabel, title), row in zip(np.ravel(axes), items, stats["components"], strict=True):
        _draw_paths(ax, times, values, chosen, AXIS_PRESETS[preset][name], logarithmic=name == "variance")
        ax.set(xlabel="Time (years)", ylabel=ylabel, title=title)
        ax.set_title(title, fontsize=10)
        if name == "volatility_percent":
            ax.plot(result["times_years"], 100 * np.sqrt(result["forward_variance"]), "--",
                    color=INK, linewidth=1, label=r"$\sqrt{\mathbb{E}[V_t]}$: supplied curve")
        elif name == "variance":
            ax.plot(result["times_years"], result["forward_variance"], "--",
                    color=INK, linewidth=1, label=r"$\mathbb{E}[V_t]=\xi_0(t)$")
        elif name in ("driver", "relative_increments"):
            ax.axhline(0, linestyle=":", color=MUTED, linewidth=.8)
            if name == "relative_increments":
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        elif name == "prices":
            ax.axhline(1, linestyle=":", color=MUTED, linewidth=.8)
        coordinate = "coordinate" if row["outside"] == 1 else "coordinates"
        ax.text(.98, .03, f"{row['outside']:,} {coordinate} outside view",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=MUTED,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 2})
        ax.legend(loc="upper left", fontsize=7, frameon=True)
    parameters = result["settings"]["parameters"]
    fig.suptitle(f"{len(result['prices'])} simulated paths  ·  H = {parameters['hurst']:g}  ·  "
                 f"η = {parameters['eta']:g}  ·  ρ = {parameters['rho']:g}  ·  fixed {preset} axes",
                 x=.04, ha="left", fontsize=11, color=INK)
    return fig


def path_figure(result, *, highlighted_path=0, preset="detail"):
    """Price and latent volatility with fixed axes and explicit outside markers.

    All rows remain in the plot data; triangles mark coordinates beyond a
    viewing window. The curve reference is sqrt(E[V]), not E[sqrt(V)].
    """
    return _figure(result, preset, "paths", highlighted_path)


def component_figure(result, *, highlighted_path=0, preset="detail"):
    """Y -> V -> finite-step price changes -> S, on comparable fixed axes.

    The variance panel uses a logarithmic axis because the exponential driver
    permits a very wide range. Choosing a different preset changes only the
    viewing windows; it never modifies or regenerates a path.
    """
    return _figure(result, preset, "components", highlighted_path)


def _widths(values, name):
    if not isinstance(values, (tuple, list)) or not 1 <= len(values) <= 8:
        raise ValueError(f"{name} must contain one to eight positive block widths")
    return tuple(_integer(value, name, 1, 4096) for value in values)


def architecture_figure(*, demo_widths=(128, 128), pretrained_widths=(256, 256, 256, 256),
                        pricer_widths=(256, 256, 256, 256), parameter_names=("H", "η", "ρ"),
                        quote_count=63, curve_levels=7, pricer_label="Fixed comparison pricer"):
    """Decompose the known-curve vector architectures without loading weights.

    Supply widths from the displayed workflow and stored model configurations.
    FlowMatching's default BF2.1 TimeMLP uses time/condition-modulated residual
    blocks, a Fourier time embedding and a velocity output projection. Widths
    describe those blocks, not a complete list of every Dense layer. The pricer
    is a separate ordinary GELU MLP, followed by a new Bayesian computation.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    demo = _widths(demo_widths, "demo_widths")
    pretrained = _widths(pretrained_widths, "pretrained_widths")
    pricer = _widths(pricer_widths, "pricer_widths")
    if (not isinstance(parameter_names, (tuple, list)) or not parameter_names
            or any(not isinstance(name, str) or not name for name in parameter_names)):
        raise ValueError("nonempty parameter names are required")
    quotes = _integer(quote_count, "quote_count", 1, 10000)
    levels = _integer(curve_levels, "curve_levels", 0, 1000)
    parameters = len(parameter_names)
    names = ", ".join(parameter_names)
    if not isinstance(pricer_label, str) or not pricer_label:
        raise ValueError("a nonempty pricer label is required")

    figure, ax = plt.subplots(figsize=(10.5, 7.0), layout="constrained")
    ax.set(xlim=(0, 1), ylim=(-.15, 4.15))
    ax.axis("off")

    def box(x, y, width, height, color, title, body):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                     boxstyle="round,pad=.012,rounding_size=.025", facecolor=color,
                     edgecolor="#d8dde3", linewidth=.8))
        ax.text(x + width / 2, y + height * .76, title, ha="center", va="center",
                fontsize=13, color=INK, fontweight="bold")
        ax.text(x + width / 2, y + height * .35, body, ha="center", va="center",
                fontsize=12, color=MUTED, linespacing=1.4)

    def arrow(x1, x2, y):
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                     mutation_scale=12, color=MUTED, linewidth=1.1))

    def blocks(widths, y, *, conditional):
        width = min(.065, .28 / len(widths))
        gap = .009
        full = len(widths) * width + (len(widths) - 1) * gap
        left = .505 - full / 2
        for i, units in enumerate(widths):
            x = left + i * (width + gap)
            ax.add_patch(FancyBboxPatch((x, y), width, .22,
                         boxstyle="round,pad=.004,rounding_size=.012",
                         facecolor=PURPLE, edgecolor="none", alpha=.9))
            ax.text(x + width / 2, y + .11, str(units), ha="center", va="center",
                    fontsize=12, color="white")
        kind = ("32D Fourier time + market condition\nmodulate each residual block" if conditional
                else "Dense layers · GELU activation")
        ax.text(.505, y - .13, kind, ha="center", va="center", fontsize=9.5, color=MUTED, linespacing=1.2)

    ax.text(.025, 3.97, "Two posterior fits and a different forward network", fontsize=16,
            ha="left", va="center", color=INK, fontweight="bold")
    for y, label, widths in ((2.78, "Small session posterior", demo),
                              (1.53, "Pretrained posterior", pretrained)):
        ax.text(.025, y + .99, label, color=PURPLE, fontsize=13, fontweight="bold")
        box(.025, y, .255, .80, "#f6f7fa", f"{quotes + levels} condition values",
            f"{quotes} scaled quotes\n+ {levels} log curve levels\nFlow state ({parameters}) + time t")
        box(.335, y, .34, .80, "#f1edf7", "FlowMatching · TimeMLP", "")
        blocks(widths, y + .26, conditional=True)
        box(.73, y, .245, .80, "#edf7f2", f"{parameters}D velocity field",
            f"Gaussian noise\n→ integrate velocity\n→ {names} joint draws")
        arrow(.286, .326, y + .40)
        arrow(.682, .721, y + .40)

    y = .28
    ax.text(.025, y + .99, pricer_label, color=PURPLE, fontsize=13, fontweight="bold")
    box(.025, y, .255, .80, "#f6f7fa", f"{parameters + levels} input values",
        f"{parameters} parameter logits\n+ {levels} log curve levels")
    box(.335, y, .34, .80, "#f1edf7", "Neural pricer · MLP", "")
    blocks(pricer, y + .26, conditional=False)
    box(.73, y, .245, .80, "#edf7f2", f"{quotes} mean-price estimates",
        "Gaussian likelihood\n+ a new MCMC run\nfor each market")
    arrow(.286, .326, y + .40)
    arrow(.682, .721, y + .40)
    ax.text(.025, -.08, "Fixed ordered quote vector: no GRU or recurrent summary. Adapter scaling is separate from these networks.",
            fontsize=11, color=MUTED, va="center")
    return figure
