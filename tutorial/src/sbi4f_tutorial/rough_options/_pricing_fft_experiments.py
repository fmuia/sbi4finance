"""Opt-in research FFT adapters for the conditional rough-pricing benchmark.

No production caller selects these adapters. The canonical driver and its
three-channel input validation supply the numerical model. These variants
reuse block setup, multiply spectra in place, and optionally change only the
zero-padding length of the same discrete causal convolution.
"""
from functools import lru_cache

import numpy as np
from scipy.fft import next_fast_len

from .model import _normals
from .simulator import hybrid_weights, rough_driver


class PreparedHybridDriver:
    """Serial research driver with cached setup for repeated chunks of a block.

    ``cached_exact`` keeps the canonical power-of-two FFT length;
    ``fast_length`` chooses a shorter FFT-friendly length >=2*n-1. The latter
    changes floating-point roundoff, not the discrete convolution or grid.
    Returned driver variances are shared cached arrays and must not be mutated.
    """

    VARIANTS = ("cached_exact", "fast_length")

    def __init__(self, variant):
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}")
        self.variant = variant
        self.setup_calls = 0
        self.driver_calls = 0

    @lru_cache(maxsize=32)
    def _setup(self, hurst, dt, steps, variant):
        self.setup_calls += 1
        weights, bridge = hybrid_weights(hurst, dt, steps)
        size = (next_fast_len(2*steps-1, real=True) if variant == "fast_length"
                else 1 << (2*steps-1).bit_length())
        spectrum = np.fft.rfft(weights, n=size)
        variance = np.cumsum(weights**2)+bridge**2
        return bridge, size, spectrum, variance

    def __call__(self, hurst, dt_years, innovations, *, method="hybrid"):
        self.driver_calls += 1
        if method != "hybrid":
            return rough_driver(hurst, dt_years, innovations, method=method)
        z = _normals(innovations)
        steps = z.shape[1]
        bridge, size, kernel, variance = self._setup(hurst, dt_years, steps, self.variant)
        spectrum = np.fft.rfft(z[..., 0], n=size, axis=1)
        spectrum *= kernel
        convolution = np.fft.irfft(spectrum, n=size, axis=1)[:, :steps]
        driver = convolution+bridge*z[..., 1]
        return driver, variance
