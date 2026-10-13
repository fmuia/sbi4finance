"""Explicit fixed Q panels and independent quote error, shared by model banks.

One observation is an entire ordered panel of currency prices. The contracts,
curves and noise scale are known synthetic context; simulator truth is separate.
"""

import numpy as np


def validate_context(context):
    if set(context) != {"spot", "rate", "dividend", "contracts", "quote_noise_std"}:
        raise ValueError("context needs spot, rate, dividend, contracts, quote_noise_std")
    for name in ("spot", "rate", "dividend", "quote_noise_std"):
        x = context[name]
        if isinstance(x, bool) or np.ndim(x) != 0 or not np.isfinite(x):
            raise ValueError(f"{name} must be a finite scalar")
    if context["spot"] <= 0 or context["quote_noise_std"] <= 0:
        raise ValueError("spot and quote_noise_std must be positive")
    contracts = context["contracts"]
    if not isinstance(contracts, list) or not contracts:
        raise ValueError("contracts must be a nonempty ordered list")
    for contract in contracts:
        if set(contract) != {"strike", "maturity", "option_type"}:
            raise ValueError("each contract needs strike, maturity, option_type")
        for name in ("strike", "maturity"):
            x = contract[name]
            if isinstance(x, bool) or np.ndim(x) != 0 or not np.isfinite(x) or x <= 0:
                raise ValueError(f"{name} must be a positive finite scalar")
        if contract["option_type"] not in ("put", "call"):
            raise ValueError("only European puts and calls are supported")


def noisy_panel(reference_prices, context, rng):
    """Add the declared Gaussian observation model, without clipping/redrawing.

    A noisy quote may be negative or violate arbitrage. This explicit synthetic
    Gaussian model is not a claim about cleaned market quotes. Numerical pricing
    error must be assessed separately from this deliberately added quote error.
    """
    validate_context(context)
    prices = np.asarray(reference_prices, dtype=np.float64)
    if prices.shape != (len(context["contracts"]),) or not np.isfinite(prices).all():
        raise ValueError("one finite reference estimate per ordered contract required")
    return (prices + context["quote_noise_std"] * rng.standard_normal(prices.shape))[:, None]
