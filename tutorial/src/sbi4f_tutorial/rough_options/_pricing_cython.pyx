# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""Research-only Cython equivalent of the opt-in Numba terminal kernel.

Build and load through the lab benchmark; no extension is installed or selected
by the participant runtime. The Python validation/payoff/fit pipeline is reused.
"""
import numpy as np
from libc.math cimport exp, sqrt, isfinite
from libc.stdint cimport int64_t


def terminal_moments(const double[:, ::1] driver,
                     const double[::1] driver_variance,
                     const double[:, :, ::1] z,
                     const double[::1] xi,
                     double dt, double eta,
                     const int64_t[::1] indices):
    """Same loop order and checks as the Numba candidate, with the GIL released."""
    cdef Py_ssize_t m = driver.shape[0], n = driver.shape[1], count = indices.shape[0]
    cdef Py_ssize_t side, path, step, selected
    cdef double iv, iw, sign, d, dv, v, increment, eta_squared = eta*eta
    cdef bint finite = True
    if (driver_variance.shape[0] != n or z.shape[0] != m or z.shape[1] != n
            or z.shape[2] != 3 or xi.shape[0] != n+1 or count < 1):
        raise ValueError("matching driver/normal/curve/indices shapes required")
    for selected in range(count):
        if indices[selected] < 1 or indices[selected] > n or (selected > 0 and indices[selected] <= indices[selected-1]):
            raise ValueError("sorted unique in-range maturity indices required")
    iv_array = np.empty((2, m, count))
    iw_array = np.empty((2, m, count))
    variance_array = np.zeros(2)
    integrated_array = np.zeros(2)
    cdef double[:, :, ::1] ivs = iv_array, iws = iw_array
    cdef double[::1] maximum_variance = variance_array, maximum_integrated = integrated_array
    with nogil:
        for side in range(2):
            sign = 1.0 if side == 0 else -1.0
            for path in range(m):
                iv = 0.0; iw = 0.0; selected = 0
                for step in range(n+1):
                    d = 0.0 if step == 0 else sign*driver[path, step-1]
                    dv = 0.0 if step == 0 else driver_variance[step-1]
                    v = xi[step]*exp(eta*d-0.5*eta_squared*dv)
                    if not isfinite(v):
                        finite = False
                    if v > maximum_variance[side]:
                        maximum_variance[side] = v
                    if step < n:
                        increment = v*dt
                        iv += increment
                        iw += sqrt(increment)*(sign*z[path, step, 0])
                        if not isfinite(iv) or not isfinite(iw):
                            finite = False
                        if iv > maximum_integrated[side]:
                            maximum_integrated[side] = iv
                        if selected < count and step+1 == indices[selected]:
                            ivs[side, path, selected] = iv
                            iws[side, path, selected] = iw
                            selected += 1
    return iv_array, iw_array, variance_array, integrated_array, bool(finite)
