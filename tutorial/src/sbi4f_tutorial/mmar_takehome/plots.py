"""Small original figures for mechanisms, diagnostics and financial outcomes."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t

from . import model

PURPLE, TEAL, INK = "#7653ac", "#167e80", "#283447"
COLORS = {4: "#2f7f78", 8: "#7653ac", 128: "#ba703d"}


def _finish(fig):
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)
    fig.tight_layout()
    return fig


def observed(returns, dates):
    values = np.asarray(returns)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(np.arange(1, 257), 100*values, color=INK, linewidth=.8)
    axes[0].set(xlabel="Trading day in fixed window", ylabel="Daily simple return (%)", title="Large moves arrive in clusters")
    ordered = np.sort(np.abs(values))
    ecdf_tail = (len(values)-np.arange(len(values)))/len(values)
    axes[1].step(100*ordered, ecdf_tail, where="post", color=INK, label="Observed")
    x = np.linspace(.00001, max(ordered)*1.1, 250)
    mu, sd = values.mean(), values.std()
    axes[1].plot(100*x, norm.sf(x, mu, sd)+norm.cdf(-x, mu, sd), color=TEAL,
                 label="Constant-variance Gaussian")
    axes[1].set(xlabel="Absolute daily return (%)", ylabel="Fraction exceeding this magnitude", yscale="log", ylim=(.001,1), title="A reference for marginal tails")
    axes[1].legend(fontsize=8)
    return _finish(fig)


def components(q=.7, nu=4., d=8, seed=1):
    values = model.simulate_parameters(np.array([[0., .015, q, nu]]), d, seed=seed)
    intensity = values["trading_time_increments"][0]
    returns = values["returns"][0]
    innovation = returns/(.015*np.sqrt(intensity))
    wealth = np.r_[1., np.cumprod(1+returns)]
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 8), sharex=True)
    specifications = [(intensity, (0, 8), "Intensity θ", TEAL),
                      (innovation, (-8, 8), "Innovation ε", INK),
                      (100*returns, (-12, 12), "Daily simple return (%)", PURPLE),
                      (wealth, (0, 3), "Wealth from $1", TEAL)]
    for ax, (y, lim, label, color) in zip(axes, specifications, strict=True):
        x = np.arange(len(y)) if len(y) == 257 else np.arange(1,257)
        ax.plot(x, y, color=color, linewidth=.9)
        outside = (y < lim[0]) | (y > lim[1])
        if outside.any():
            ax.scatter(x[outside], np.clip(y[outside], *lim), marker="x", s=20, color="#be4f3b", clip_on=False)
        ax.set(ylabel=label, ylim=lim, xlim=(0,256))
        ax.text(.99, .93, f"{outside.sum()} outside fixed view", ha="right", va="top", transform=ax.transAxes, fontsize=8)
    axes[-1].set_xlabel("Trading day")
    axes[0].set_title(f"One simulated realization · q={q:.2f}, ν={nu:.2f}, d={d} ({256//d}-day blocks)")
    return _finish(fig)


def tail_mechanism(nu):
    x = np.linspace(-6, 6, 500)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot(x, norm.pdf(x), color=INK, label="Variance-one Gaussian")
    ax.plot(x, t.pdf(x, nu, scale=np.sqrt((nu-2)/nu)), color=PURPLE, label=f"Variance-one Student-t, ν={nu:.2f}")
    ax.set(yscale="log", ylim=(1e-5, 2), xlabel="Standardized innovation", ylabel="Density", title="Changing ν changes the innovation law")
    ax.legend(fontsize=8)
    return _finish(fig)


def check_comparison(observed_summaries, replicated_summaries, observed_acf, replicated_acf, title):
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2))
    axes = axes.ravel()
    labels = ["Daily return SD (%)", "95% |return| quantile (%)", "Maximum drawdown (%)"]
    for j, ax in enumerate(axes[:3]):
        values = replicated_summaries[:,j]
        ax.hist(values, bins=22, color=PURPLE, alpha=.65)
        ax.axvline(np.ravel(observed_summaries)[j], color=INK, label="Observed")
        ax.set(xlabel=labels[j], ylabel="Replicated histories")
    lags = np.arange(1, replicated_acf.shape[1]+1)
    available = np.isfinite(replicated_acf).all(axis=1) if len(lags) else np.zeros(len(replicated_acf), dtype=bool)
    if available.any():
        lo, med, hi = np.quantile(replicated_acf[available], [.05,.5,.95], axis=0)
        axes[3].fill_between(lags,lo,hi,color=PURPLE,alpha=.2,label="Central 90% simulations")
        axes[3].plot(lags,med,color=PURPLE)
    axes[3].plot(lags,np.ravel(observed_acf),color=INK,label="Observed")
    axes[3].set(xlabel="Lag (trading days)",ylabel="ACF of |log returns|", ylim=(-1,1))
    axes[3].set_title(f"ACF available: {available.sum()}/{len(available)}", fontsize=9)
    axes[3].legend(fontsize=7)
    fig.suptitle(title, fontsize=12)
    return _finish(fig)


def posterior(draws):
    fig, axes = plt.subplots(2,2,figsize=(8.2,5.4))
    axes = axes.ravel()
    for j, ax in enumerate(axes):
        x = np.linspace(-.006,.006,300) if j==0 else np.linspace(*model.BOUNDS[j],300)
        prior = (norm.pdf(x,0,.0015) if j==0 else 1/((x-2)*np.log(10/.05)) if j==3 else np.full_like(x,1/(model.BOUNDS[j][1]-model.BOUNDS[j][0])))
        ax.plot(x,prior,color=INK,linestyle="--",label="Declared prior")
        ax.hist(draws[:,j],bins=28,density=True,color=PURPLE,alpha=.55,label="Posterior")
        ax.set(xlabel=model.LABELS[j],ylabel="Density")
    axes[0].legend(fontsize=8)
    return _finish(fig)


def diagnostics_figure(data,d=8,parameter=2):
    fig, axes = plt.subplots(1,2,figsize=(10,3.6))
    truth = data[f"diagnostic_{d}_truth"][:,parameter]
    median = data[f"diagnostic_{d}_median"][:,parameter]
    low,high = data[f"diagnostic_{d}_lower90"][:,parameter],data[f"diagnostic_{d}_upper90"][:,parameter]
    axes[0].vlines(truth,low,high,color=PURPLE,alpha=.2)
    axes[0].scatter(truth,median,color=PURPLE,s=12,alpha=.7)
    a,b=min(truth.min(),median.min()),max(truth.max(),median.max())
    axes[0].plot([a,b],[a,b],color=INK,linestyle="--")
    axes[0].set(xlabel="True "+model.PARAMETERS[parameter],ylabel="Posterior median and central 90%",title=f"Recovery at d={d}")
    for j,name in enumerate(model.PARAMETERS):
        delta=data[f"diagnostic_{d}_rank_deviations"][j]
        axes[1].plot(np.arange(len(delta))/(len(delta)-1),delta,label=name,alpha=.8)
    axes[1].axhline(0,color=INK,linewidth=.8)
    axes[1].set(xlabel="Finite posterior rank / 512",ylabel="Rank ECDF minus finite-rank null",title="Calibration is a different question")
    axes[1].legend(fontsize=8)
    return _finish(fig)


def drawdown(wealth,mdd,point_mdd,observed_wealth,threshold=.2):
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    axes[0].plot(wealth.T,color=PURPLE,alpha=.25,linewidth=.8)
    axes[0].plot(observed_wealth,color=INK,linewidth=1.5,label="Observed history")
    axes[0].set(xlabel="Trading day from fresh origin",ylabel="Wealth from $1",title="Independent full-window replicas",ylim=(0,3),xlim=(0,256))
    outside = (wealth < 0) | (wealth > 3)
    if outside.any():
        row,day = np.nonzero(outside)
        axes[0].scatter(day,np.clip(wealth[row,day],0,3),marker="x",s=9,color="#be4f3b",clip_on=False)
    observed_outside = (observed_wealth < 0) | (observed_wealth > 3)
    if observed_outside.any():
        axes[0].scatter(np.flatnonzero(observed_outside),np.clip(observed_wealth[observed_outside],0,3),marker="x",color=INK,clip_on=False)
    axes[0].text(.99,.97,f"{outside.sum()} simulated / {observed_outside.sum()} observed values outside view",transform=axes[0].transAxes,ha="right",va="top",fontsize=7)
    axes[0].legend(fontsize=8)
    for values,color,label in [(mdd,PURPLE,"Paired posterior parameters"),(point_mdd,TEAL,"One posterior-mean vector")]:
        axes[1].step(100*np.sort(values),np.arange(1,len(values)+1)/len(values),where="post",color=color,label=label)
    axes[1].axvline(100*threshold,color=INK,linestyle="--")
    axes[1].set(xlabel="Maximum drawdown (%)",ylabel="Fraction at or below",xlim=(0,100),ylim=(0,1),title="Both retain process randomness")
    axes[1].legend(fontsize=8)
    return _finish(fig)


def sensitivity(data):
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for i,d in enumerate(model.CAPSTONE_D):
        lo,median,hi=np.quantile(data[f"posterior_{d}"][:,2],[.05,.5,.95])
        axes[0].errorbar(i,median,yerr=[[median-lo],[hi-median]],fmt="o",color=COLORS[d],capsize=4)
        values=np.sort(data[f"mdd_{d}"])
        axes[1].step(100*values,np.arange(1,len(values)+1)/len(values),where="post",color=COLORS[d],label=f"d={d}: {256//d}-day blocks")
    axes[0].set(xticks=np.arange(3),xticklabels=[str(d) for d in model.CAPSTONE_D],xlabel="Known number of blocks d",ylabel="q median and central 90%",ylim=(.5,.92),title="Parameter interpretation")
    axes[1].set(xlabel="Replicated maximum drawdown (%)",ylabel="Fraction at or below",xlim=(0,100),ylim=(0,1),title="Financial outcome")
    axes[1].legend(fontsize=8)
    return _finish(fig)
