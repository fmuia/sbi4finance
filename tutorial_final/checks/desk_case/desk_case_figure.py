"""Static figures of the desk case, from the cached results in this folder.

    python desk_case_figure.py

desk_case.png: the quotes and the products, the profile in H, the posterior against Gauss–Newton, the reserves.
desk_case_validation.png (when validate_*.json exist): breach rates of the 90% reserves on simulated thin chains.
desk_case_market.png: the forward start under the front-month methods against the full SPY surface.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import desk_figures as F  # noqa: E402
from sbi4finance.calibration.plots import plt, style  # noqa: E402

style()


def main():
    data = F.load()
    fig = plt.figure(figsize=(13.5, 11.5), constrained_layout=True)
    top, middle, bottom = fig.subfigures(3, 1, height_ratios=[1, 1, 1.15])
    a = top.subplots(1, 3, gridspec_kw={"width_ratios": [1.1, 0.9, 1.2]})
    F.quotes_panel(a[0], data)
    F.timeline_panel(a[1])
    F.profile_panel(a[2], data)
    b = middle.subplots(1, 3)
    for i in range(3):
        F.marginal_panel(b[i], data, i)
    c = bottom.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 1.25]})
    F.joint_panel(c[0], data, 0, 2)
    F.joint_panel(c[1], data, 0, 1)
    F.reserves_panel(c[2], data)
    k = F.key_numbers(data)
    fig.suptitle(f"Desk case: rough Bergomi from SPY's front three monthly smiles (27 quotes, 21 Oct 2026). Exact posterior "
                 f"by importance sampling ({k['proposals']:,} Monte Carlo surfaces, k̂ {k['khat']:.2f}) against "
                 f"Gauss–Newton", fontsize=9.5, x=0.01, ha="left")
    fig.savefig(HERE / "desk_case.png", dpi=140)
    print("saved", HERE / "desk_case.png")
    fig, ax = plt.subplots(figsize=(8.0, 3.8), constrained_layout=True)
    F.market_panel(ax, data, 1, 1.0)
    fig.savefig(HERE / "desk_case_market.png", dpi=140)
    print("saved", HERE / "desk_case_market.png")
    if F.validation_summary(data) is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), constrained_layout=True, sharey=True)
        F.validation_panel(axes, data)
        v = F.validation_summary(data)
        fig.suptitle(f"Breach rates of one-sided 90% reserves on {v['replicas']} simulated thin chains (truths from the "
                     f"front-month posterior; a correct reserve is breached 10% of the time)", fontsize=9, x=0.01,
                     ha="left")
        fig.savefig(HERE / "desk_case_validation.png", dpi=140)
        print("saved", HERE / "desk_case_validation.png")


if __name__ == "__main__":
    main()
