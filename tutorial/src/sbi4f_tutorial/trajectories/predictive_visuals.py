"""Return-path predictive displays with a clearly data-fitted IID control.

These functions consume unchanged model_checks results. They neither infer
parameters nor alter the simulation law. The Gaussian is a descriptive control
fitted to the observed prefix, not another prior-predictive distribution.
"""

import numpy as np
import matplotlib.pyplot as plt

from sbi4f_tutorial.shared.path_statistics import cumulative_log_wealth, sample_acf

MODEL_COLOR = "#3e74b5"
OBSERVED_COLOR = "#182536"
GAUSSIAN_COLOR = "#9a754c"


def _returns(result):
    if np.ma.isMaskedArray(result["observed_returns"]) or np.ma.isMaskedArray(result["replicated_prefix"]):
        raise ValueError("predictive displays require real unmasked prefix rows")
    observed = np.asarray(result["observed_returns"])
    replicated = np.asarray(result["replicated_prefix"])
    if (np.iscomplexobj(observed) or np.iscomplexobj(replicated)
            or observed.shape != (result["length"], 1)
            or replicated.ndim != 3 or replicated.shape[1:] != observed.shape
            or len(replicated) != result["metadata"]["count"]
            or not np.isfinite(observed).all() or not np.isfinite(replicated).all()):
        raise ValueError("predictive displays need all finite, aligned observed and replicated prefix rows")
    return observed[:, 0], replicated[:, :, 0]


def compare(result, *, seed=20261119, gaussian_count=200):
    """Fit a Gaussian to observations only, retaining every model and control row."""
    observed, replicated = _returns(result)
    if isinstance(gaussian_count, bool) or int(gaussian_count) != gaussian_count or gaussian_count < 2:
        raise ValueError("at least two Gaussian control histories required")
    scale = np.max(np.abs(observed))
    scaled = observed / scale if scale else observed
    mean = float(np.mean(scaled) * scale)
    sd = float(np.sqrt(np.mean((scaled - np.mean(scaled)) ** 2)) * scale)
    available = len(observed) >= 2 and sd > 0
    rng = np.random.default_rng(np.random.SeedSequence([0x47504348, seed, len(observed)]))
    with np.errstate(over="raise", invalid="raise"):
        gaussian = (mean + sd * rng.standard_normal((int(gaussian_count), len(observed)))
                    if available else np.empty((0,len(observed))))
    if not np.isfinite(gaussian).all():
        raise FloatingPointError("Gaussian control overflowed; no control histories were discarded")
    threshold = 3 * sd
    groups = {"observed": observed[None], "model": replicated, "gaussian": gaussian}
    features = {}
    for name, values in groups.items():
        absolute_acf = (sample_acf(values, absolute=True) if len(values) else
                        {"lags":np.arange(1,min(20,len(observed)//5)+1) if len(observed)>=21 else np.array([],dtype=int),
                         "values":np.empty((0,min(20,len(observed)//5) if len(observed)>=21 else 0)),
                         "available":np.zeros(0,dtype=bool),"available_count":0,"count":0})
        lag_one = (absolute_acf["values"][:, 0] if len(absolute_acf["lags"])
                   else np.full(len(values), np.nan))
        features[name] = {
            "large_move_fraction": np.mean(np.abs(values - mean) > threshold, axis=1),
            "absolute_lag_one": lag_one,
            "acf": absolute_acf,
        }
    return {"observed": observed.copy(), "replicated": replicated.copy(),
            "gaussian": gaussian, "mean": mean, "sd": sd,
            "large_move_threshold": threshold, "features": features,
            "kind": result["kind"], "length": len(observed),
            "metadata": {"model_count": len(replicated), "gaussian_count": len(gaussian),
                         "gaussian_available":available,
                         "gaussian_seed": seed, "control": "IID Gaussian fitted to observed prefix mean and MLE SD",
                         "scope": "descriptive comparison; no calibrated model test or impossibility claim"}}


def _outside(ax, times, values, limits, color):
    """Mark display exceedances; the underlying arrays remain unchanged."""
    values = np.atleast_2d(values)
    outside = (values < limits[0]) | (values > limits[1])
    rows, columns = np.nonzero(outside)
    if len(rows):
        # Bound marker overplotting without selecting histories: one marker at
        # every day with at least one exceedance, separately for either edge.
        for edge, mask in [(limits[0], values < limits[0]), (limits[1], values > limits[1])]:
            columns = np.flatnonzero(mask.any(axis=0))
            ax.scatter(np.asarray(times)[columns], np.full(len(columns), edge),
                       color=color, marker="x", s=12, clip_on=False, zorder=4)
    return int(outside.sum())


def _band(ax, x, rows, *, color, label, show_paths=False):
    low, median, high = np.quantile(rows, [.05, .5, .95], axis=0)
    if show_paths:
        for row in rows:
            ax.plot(x, row, color=color, alpha=.16, lw=.65)
    else:
        ax.fill_between(x, low, high, color=color, alpha=.17)
    ax.plot(x, median, color=color, lw=1.3, label=label)


def paths_and_distributions(comparison):
    """Aligned path behavior and empirical marginal/tail frequencies, fixed views."""
    c = comparison
    observed, replicated, gaussian = c["observed"], c["replicated"], c["gaussian"]
    n, length = replicated.shape
    times = np.arange(1, length + 1)
    figure, axes = plt.subplots(2, 2, figsize=(7.6, 6.8), layout="constrained")
    ax_return, ax_distribution = axes[0]
    ax_wealth, ax_tail = axes[1]
    small = n <= 32
    _band(ax_return, times, 100 * replicated, color=MODEL_COLOR,
          label=f"Model: all {n} paths" if small else f"Model median; central 90% ({n} paths)", show_paths=small)
    if len(gaussian):
        low, high = np.quantile(100*gaussian, [.05, .95], axis=0)
        ax_return.plot(times, low, color=GAUSSIAN_COLOR, lw=.7, alpha=.65, linestyle="--")
        ax_return.plot(times, high, color=GAUSSIAN_COLOR, lw=.7, alpha=.65, linestyle="--",
                       label="Fitted IID Gaussian: central 90%")
    ax_return.plot(times, 100*observed, color=OBSERVED_COLOR, lw=.95, label="Observed prefix", zorder=3)
    ax_return.set(xlim=(.5, length+.5), ylim=(-15,15), xlabel="Observed trading day", ylabel="Daily log return (%)")
    outside = _outside(ax_return, times, 100*replicated, (-15,15), MODEL_COLOR)
    observed_outside = _outside(ax_return, times, 100*observed, (-15,15), OBSERVED_COLOR)
    control_outside = _outside(ax_return,times,100*gaussian,(-15,15),GAUSSIAN_COLOR) if len(gaussian) else 0
    ax_return.text(.02,.97,f"Outside view: model {outside}, observed {observed_outside}, Gaussian {control_outside}", transform=ax_return.transAxes,va="top",fontsize=7)
    ax_return.legend(fontsize=7.8,loc="lower left")

    model_wealth = 100*cumulative_log_wealth(replicated)
    observed_wealth = 100*cumulative_log_wealth(observed)
    gaussian_wealth = 100*cumulative_log_wealth(gaussian) if len(gaussian) else None
    time_wealth = np.arange(length+1)
    _band(ax_wealth, time_wealth, model_wealth, color=MODEL_COLOR,
          label="Model paths" if small else "Model pointwise central 90%", show_paths=small)
    if len(gaussian):
        lo, hi = np.quantile(gaussian_wealth,[.05,.95],axis=0)
        ax_wealth.fill_between(time_wealth,lo,hi,color=GAUSSIAN_COLOR,alpha=.10,label="Fitted IID Gaussian: central 90%")
    ax_wealth.plot(time_wealth,observed_wealth,color=OBSERVED_COLOR,lw=1.5,label="Observed prefix")
    ax_wealth.set(xlim=(0,max(length,1)),ylim=(-200,200),xlabel="Trading day from model origin",ylabel="Cumulative log wealth (log %)")
    model_outside = _outside(ax_wealth,time_wealth,model_wealth,(-200,200),MODEL_COLOR)
    observed_outside = _outside(ax_wealth,time_wealth,observed_wealth,(-200,200),OBSERVED_COLOR)
    control_outside = _outside(ax_wealth,time_wealth,gaussian_wealth,(-200,200),GAUSSIAN_COLOR) if len(gaussian) else 0
    ax_wealth.text(.02,.97,f"Outside view: model {model_outside}, observed {observed_outside}, Gaussian {control_outside}",transform=ax_wealth.transAxes,va="top",fontsize=7)
    ax_wealth.legend(fontsize=7.8,loc="lower left")

    # Normalize by the full sample count, including values beyond the view.
    # density=True on truncated bins would silently renormalize those away.
    edges = np.linspace(-15,15,61)
    centers = .5*(edges[:-1]+edges[1:])
    for rows,color,label in [(observed[None],OBSERVED_COLOR,"Observed"),(replicated,MODEL_COLOR,"Model replicas"),(gaussian,GAUSSIAN_COLOR,"Fitted IID Gaussian")]:
        if not len(rows):
            continue
        frequency,_ = np.histogram(100*rows.ravel(),bins=edges)
        frequency = frequency/rows.size
        ax_distribution.step(centers,frequency,where="mid",color=color,lw=1.4,label=label)
    ax_distribution.set(xlim=(-15,15),ylim=(0,1),xlabel="Daily log return (%)",ylabel="Fraction per 0.5-point bin",title="Return frequencies across days")
    ax_distribution.legend(fontsize=7.8)

    cutoffs = np.linspace(0,15,81)
    for rows,color,label in [(replicated,MODEL_COLOR,"Model median / 90%"),(gaussian,GAUSSIAN_COLOR,"Fitted IID Gaussian median / 90%")]:
        if not len(rows):
            continue
        frequency = np.stack([np.mean(np.abs(100*rows)>cutoff,axis=1) for cutoff in cutoffs],axis=1)
        lo,med,hi = np.quantile(frequency,[.05,.5,.95],axis=0)
        ax_tail.fill_between(cutoffs,lo,hi,color=color,alpha=.13)
        ax_tail.plot(cutoffs,med,color=color,lw=1.3,label=label)
    actual = np.array([np.mean(np.abs(100*observed)>cutoff) for cutoff in cutoffs])
    ax_tail.plot(cutoffs,actual,color=OBSERVED_COLOR,lw=1.6,label="Observed")
    # Symlog retains exact zero frequencies instead of dropping them on a log axis.
    ax_tail.set(xlim=(0,15),ylim=(0,1),yscale="symlog",xlabel="Absolute daily log-return threshold (%)",
                ylabel="Fraction of days above threshold",title="How often do large moves occur?")
    ax_tail.set_yscale("symlog",linthresh=1/length)
    ticks = np.unique([0.,1/length,*[p for p in (.001,.01,.1,1.) if p>2/length]])
    ax_tail.set_yticks(ticks,labels=["0" if p==0 else f"1/{length}" if p==1/length else f"{p:g}" for p in ticks])
    ax_tail.legend(fontsize=7.8)
    for ax in axes.flat:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.15)
    name = "Prior predictive" if c["kind"] == "prior_predictive" else "Posterior predictive replicas"
    note = "Gaussian is a data-fitted control" if len(gaussian) else "Gaussian fit unavailable: short or constant observations"
    figure.suptitle(f"{name} · {length} observed days\n{note}",fontsize=10)
    return figure


def feature_rows(comparison):
    """Finite-envelope readout, with undefined short/constant ACFs explicit."""
    c = comparison
    rows = []
    for key,label,percent in [("large_move_fraction",f"Days beyond observed mean ± 3 SD ({100*c['large_move_threshold']:.2g} log %)",True),
                              ("absolute_lag_one","Lag-1 dependence in absolute returns",False)]:
        observed = c["features"]["observed"][key][0]
        row = {"Feature":label,"Observed":"Unavailable" if not np.isfinite(observed) else f"{100*observed:.2f}%" if percent else f"{observed:.3f}"}
        for name,heading in [("model","Model central 90%"),("gaussian","Fitted IID Gaussian central 90%")]:
            values = c["features"][name][key]
            valid = np.isfinite(values)
            if not valid.any():
                row[heading] = f"Unavailable (0/{len(values)} histories)"
            else:
                lo,hi = np.quantile(values[valid],[.05,.95])
                if percent:
                    lo,hi=100*lo,100*hi
                unit="%" if percent else ""
                row[heading]=f"{lo:.3g}–{hi:.3g}{unit} ({valid.sum()}/{len(values)})"
        rows.append(row)
    return rows


def timing(result, comparison, *, show_shuffle=False):
    """Existing timing definitions with an IID finite-sample simulation reference."""
    figure, axes = plt.subplots(1,2,figsize=(7.6,3.7),layout="constrained")
    ordering = result["ordering"]
    lags = ordering["lags"]
    for ax,key,title in zip(axes,("signed","absolute"),("Signed-return dependence","Absolute-return dependence"),strict=True):
        data = ordering[key]
        control = (sample_acf(comparison["gaussian"],absolute=(key=="absolute"))
                   if len(comparison["gaussian"]) else {"available_count":0,"count":0})
        ax.set(xlim=(.5,max(4.5,len(lags)+.5)),ylim=(-1,1),xlabel="Lag (trading days)",ylabel="Sample autocorrelation",title=title)
        ax.axhline(0,color="#94a3b8",lw=.8)
        if not len(lags):
            ax.text(.5,.55,"At least 21 days are needed\nfor this dependence illustration.",transform=ax.transAxes,ha="center",va="center",fontsize=9)
        if data["available_count"]:
            curves=data["replicated"][data["available"]]
            if data["count"]<=32:
                for curve in curves:
                    ax.plot(lags,curve,color=MODEL_COLOR,alpha=.17,lw=.6)
            else:
                lo,hi=np.quantile(curves,[.05,.95],axis=0)
                ax.fill_between(lags,lo,hi,color=MODEL_COLOR,alpha=.16)
            ax.plot(lags,np.median(curves,axis=0),color=MODEL_COLOR,lw=1.5,label="Model replicas")
        if control["available_count"]:
            curves=control["values"][control["available"]]
            lo,hi=np.quantile(curves,[.05,.95],axis=0)
            ax.fill_between(lags,lo,hi,color=GAUSSIAN_COLOR,alpha=.12,label="Fitted IID Gaussian: 90%")
        if data["observed_available"]:
            ax.plot(lags,data["observed"],color=OBSERVED_COLOR,lw=1.8,label="Observed prefix")
        if show_shuffle and data["shuffled_available"]:
            ax.plot(lags,data["shuffled"],color="#c47a25",linestyle="--",lw=1.2,label="Shuffled observation")
        ax.text(.02,.03,f"ACF available: model {data['available_count']}/{data['count']}\nGaussian {control['available_count']}/{control['count']}",transform=ax.transAxes,fontsize=7,va="bottom")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7.8,loc="upper right")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.15)
    return figure
