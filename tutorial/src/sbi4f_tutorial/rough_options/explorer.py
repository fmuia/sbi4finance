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
    native = simulator.simulate_rough(parameters, curve, 1. / steps, innovations,
                                      method="hybrid", spot=1., rate=0., dividend=0.)
    with np.errstate(over="raise", invalid="raise"):
        try:
            prices = np.exp(native["log_spot"])
        except FloatingPointError as error:
            raise ValueError("asset-price overflow; no paths were redrawn") from error
    return {
        "times_years": native["times"], "prices": prices,
        "variances": native["variance"], "volatility": np.sqrt(native["variance"]),
        "forward_variance": np.full(steps + 1, volatility**2),
        "volterra_driver": native["volterra_driver"], "driver_variance": native["driver_variance"],
        "settings": {"parameters": {name: float(value) for name, value in parameters.items()},
                     "supplied_volatility": volatility, "seed": int(seed), "paths": paths,
                     "steps": steps, "horizon_years": 1., "rate": 0., "dividend": 0.,
                     "driver_method": "hybrid", "measure": "Q",
                     "compensator": "discrete Gaussian-driver variance",
                     "scope": "illustrative price and hidden-variance paths; no quote noise or inference"},
    }


def path_figure(result, *, highlighted_path=0):
    """Show price and latent volatility with the same path highlighted in both.

    All simulated paths are shown, without a quantile ribbon or a posterior
    interpretation. The horizontal volatility reference is sqrt(E[V]), not
    E[sqrt(V)] or an empirical average over the few displayed paths.
    """
    import matplotlib.pyplot as plt

    times = np.asarray(result["times_years"])
    prices = np.asarray(result["prices"])
    volatility = np.asarray(result["volatility"])
    if (times.ndim != 1 or prices.ndim != 2 or prices.shape != volatility.shape
            or prices.shape[1] != len(times) or not len(prices)
            or not all(np.isfinite(x).all() for x in (times, prices, volatility))):
        raise ValueError("matching finite price and volatility paths on one time grid are required")
    chosen = _integer(highlighted_path, "highlighted_path", 0, len(prices) - 1)
    reference = 100 * np.sqrt(np.asarray(result["forward_variance"]))
    if reference.shape != times.shape or not np.isfinite(reference).all():
        raise ValueError("one finite forward-variance reference per time point is required")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for ax, values, ylabel, title in zip(
            axes, (prices, 100 * volatility), ("Price / initial price", "Annualized volatility (%)"),
            ("Possible price histories", "Their hidden volatility"), strict=True):
        ax.set_facecolor("#fbfafc")
        for i, path in enumerate(values):
            if i != chosen:
                ax.plot(times, path, color=PURPLE, alpha=.22, linewidth=.8)
        ax.plot(times, values[chosen], color=GREEN, linewidth=2, label="Same highlighted path")
        ax.set(xlabel="Time (years)", ylabel=ylabel, title=title, xlim=(times[0], times[-1]))
        ax.grid(color="#dfe4e8", alpha=.65, linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    axes[0].axhline(1, linestyle=":", color=MUTED, linewidth=.9)
    axes[0].legend(loc="upper left", fontsize=8, frameon=True)
    axes[1].plot(times, reference, "--", color=INK, linewidth=1.1,
                 label=r"$\sqrt{\mathbb{E}[V_t]}$: supplied curve")
    axes[1].legend(loc="upper left", fontsize=8, frameon=True)
    parameters = result["settings"]["parameters"]
    fig.suptitle(f"{len(prices)} simulated paths  ·  H = {parameters['hurst']:g}  ·  "
                 f"η = {parameters['eta']:g}  ·  ρ = {parameters['rho']:g}",
                 x=.04, ha="left", fontsize=11, color=INK)
    return fig


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
