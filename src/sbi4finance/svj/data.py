"""Return windows for the SVJ pipeline: the last T daily log returns of one name, in percent."""

from pathlib import Path

import numpy as np
import pandas as pd

from .model import WINDOW


def load_returns(csv, ticker, kind="simple"):
    """Daily log returns in percent from a CSV with a date column and one column of returns per ticker.

    kind="simple" converts simple returns r to 100 log(1 + r); kind="log" scales log returns by 100.
    """
    frame = pd.read_csv(Path(csv), index_col=0, parse_dates=True)
    series = frame[ticker].dropna().astype(float)
    if kind == "simple":
        series = 100 * np.log1p(series)
    elif kind == "log":
        series = 100 * series
    else:
        raise ValueError(f"unknown return kind {kind!r}")
    return series.sort_index()


def window(returns, end=None, n_days=WINDOW):
    """The n_days returns up to and including `end` (default: the last day)."""
    if end is not None:
        returns = returns.loc[: pd.Timestamp(end)]
    if len(returns) < n_days:
        raise ValueError(f"need {n_days} returns, have {len(returns)}")
    return returns.iloc[-n_days:]


def rolling_ends(returns, step=5, n_days=WINDOW):
    """End dates of rolling windows, one every `step` trading days, ending at the last day."""
    ends = returns.index[n_days - 1 :]
    return ends[::-1][::step][::-1]
