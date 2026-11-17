"""Small return-path statistics with explicit undefined cases and stable arithmetic.

Inputs are one history ``[time]`` or a batch ``[history, time]``. Drawdown takes
log returns; callers with simple returns must first use ``numpy.log1p``.
"""

import numpy as np


def _rows(values):
    if np.iscomplexobj(values) or np.ma.isMaskedArray(values):
        raise ValueError("real unmasked return histories are required")
    values = np.asarray(values)
    if (values.ndim not in (1, 2) or not values.size or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all()):
        raise ValueError("finite return histories with shape [time] or [history,time] are required")
    values = np.asarray(values, dtype=np.float64)
    return np.atleast_2d(values), values.ndim == 1


def sample_acf(values, *, absolute=False):
    """Sample ACF at lags 1..min(20,n//5), centered over the whole prefix.

    Scale before centering and squaring. The denominator is the whole-prefix
    sum of squared deviations, including for nonzero lags. Below 21 days, or
    for a constant series, the statistic is unavailable rather than zero.
    Every input row remains present; its unavailable values are explicit NaNs.
    """
    rows, _ = _rows(values)
    if not isinstance(absolute, (bool, np.bool_)):
        raise ValueError("absolute must be a boolean")
    count, length = rows.shape
    lags = np.arange(1, min(20, length // 5) + 1) if length >= 21 else np.array([], dtype=int)
    result = np.full((count, len(lags)), np.nan)
    available = np.zeros(count, dtype=bool)
    if len(lags):
        scale = np.max(np.abs(rows), axis=1, keepdims=True)
        scaled = rows / np.where(scale > 0, scale, 1.)
        if absolute:
            scaled = np.abs(scaled)
        centered = scaled - np.mean(scaled, axis=1, keepdims=True)
        denominator = np.sum(centered * centered, axis=1)
        available = denominator > 0
        for index, lag in enumerate(lags):
            numerator = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1)
            np.divide(numerator, denominator, out=result[:, index], where=available)
        if not np.isfinite(result[available]).all():
            raise FloatingPointError("sample autocorrelation failed; no histories were removed")
    return {"lags": lags, "values": result, "available": available,
            "available_count": int(available.sum()), "count": count, "length": length,
            "minimum_length": 21, "absolute": bool(absolute)}


def cumulative_log_wealth(log_returns):
    """Cumulative log wealth, including initial wealth one; reject overflow."""
    rows, single = _rows(log_returns)
    try:
        with np.errstate(over="raise", invalid="raise"):
            cumulative = np.cumsum(rows, axis=1)
    except FloatingPointError as error:
        raise FloatingPointError("cumulative log wealth overflowed; no histories were removed") from error
    if not np.isfinite(cumulative).all():
        raise FloatingPointError("cumulative log wealth is nonfinite; no histories were removed")
    result = np.column_stack([np.zeros(len(rows)), cumulative])
    return result[0] if single else result


def maximum_drawdown(log_returns):
    """Positive maximum drawdown in [0,1], calculated without exponentiating wealth.

    ``W_0=1`` is part of the path. A one-day loss is therefore a drawdown.
    Large negative distances may round to complete loss; no paths are clipped,
    rejected, redrawn or omitted to make a statistic finite.
    """
    wealth = cumulative_log_wealth(log_returns)
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            distances = wealth - np.maximum.accumulate(wealth, axis=-1)
            result = np.max(-np.expm1(distances), axis=-1)
    except FloatingPointError as error:
        raise FloatingPointError("drawdown arithmetic overflowed; no histories were removed") from error
    if not np.isfinite(result).all():
        raise FloatingPointError("maximum drawdown is nonfinite; no histories were removed")
    return result
