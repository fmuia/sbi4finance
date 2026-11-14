"""Training-only full-covariance ZCA preprocessing with portable fitted state.

Fit on noisy training conditions, then reuse the same transform for validation
and inference. Every feature is retained, including when small covariance
eigenvalues are floored. This module depends only on NumPy.
"""

from dataclasses import dataclass, field
from collections.abc import Mapping
from numbers import Real

import numpy as np


_FORMAT = "full_covariance_zca_whitening_v1"
_FIELDS = {"format", "training_rows", "dimension", "relative_eigenvalue_floor",
           "mean", "whitening_matrix", "inverse_matrix", "eigenvalues", "safe_eigenvalues"}


def _real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real numeric values")
    array = np.array(array, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class FullCovarianceWhitening:
    """Immutable fitted ZCA state; caller-owned metadata binds source and data."""

    mean: np.ndarray = field(repr=False)
    whitening_matrix: np.ndarray = field(repr=False)
    inverse_matrix: np.ndarray = field(repr=False)
    eigenvalues: np.ndarray = field(repr=False)
    safe_eigenvalues: np.ndarray = field(repr=False)
    training_rows: int
    relative_eigenvalue_floor: float = 1e-8

    def __post_init__(self):
        if type(self.training_rows) is not int or self.training_rows < 2:
            raise ValueError("training_rows must be an integer of at least two")
        floor = self.relative_eigenvalue_floor
        if (not isinstance(floor, Real) or isinstance(floor, (bool, np.bool_))
                or not np.isfinite(floor) or not 0 < floor <= 1):
            raise ValueError("relative_eigenvalue_floor must be finite and in (0, 1]")
        arrays = {name: _real_array(getattr(self, name), name) for name in (
            "mean", "whitening_matrix", "inverse_matrix", "eigenvalues", "safe_eigenvalues")}
        dimension = arrays["mean"].size
        if (arrays["mean"].shape != (dimension,) or dimension < 1
                or any(arrays[name].shape != (dimension, dimension) for name in ("whitening_matrix", "inverse_matrix"))
                or any(arrays[name].shape != (dimension,) for name in ("eigenvalues", "safe_eigenvalues"))):
            raise ValueError("whitening state dimensions do not match")
        eigenvalues, safe = arrays["eigenvalues"], arrays["safe_eigenvalues"]
        # Sample covariance is positive semidefinite; eigh may return tiny
        # negative eigenvalues for rank-deficient data at roundoff precision.
        tolerance = 64 * np.finfo(np.float64).eps * dimension * max(abs(eigenvalues[-1]), np.finfo(np.float64).tiny)
        if (eigenvalues[-1] <= 0 or np.any(np.diff(eigenvalues) < -tolerance)
                or eigenvalues[0] < -tolerance):
            raise ValueError("raw covariance eigenvalues must be ordered and positive semidefinite")
        expected_safe = np.maximum(eigenvalues, eigenvalues[-1] * float(floor))
        if np.any(safe <= 0) or not np.array_equal(safe, expected_safe):
            raise ValueError("safe eigenvalues do not match the declared relative floor")
        matrix, inverse = arrays["whitening_matrix"], arrays["inverse_matrix"]
        for name in ("whitening_matrix", "inverse_matrix"):
            value = arrays[name]
            if not np.allclose(value, value.T, rtol=1e-11, atol=1e-12 * float(np.max(np.abs(value)))):
                raise ValueError(f"{name} must be symmetric")
            try:
                np.linalg.cholesky(value)
            except np.linalg.LinAlgError as error:
                raise ValueError(f"{name} must be positive definite") from error
        if not np.allclose(matrix @ inverse, np.eye(dimension), rtol=5e-10, atol=5e-10):
            raise ValueError("whitening and inverse matrices are inconsistent")
        if not np.allclose(np.linalg.eigvalsh(matrix), np.sort(1 / np.sqrt(safe)), rtol=2e-8, atol=0.):
            raise ValueError("whitening matrix spectrum does not match the safe covariance eigenvalues")
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "relative_eigenvalue_floor", float(floor))

    @property
    def dimension(self):
        return len(self.mean)

    @property
    def eigenvalue_floor(self):
        return float(self.eigenvalues[-1] * self.relative_eigenvalue_floor)

    def _values(self, values):
        values = _real_array(values, "conditions")
        if values.ndim < 1 or values.shape[-1] != self.dimension or values.size == 0:
            raise ValueError("conditions must retain the fitted feature dimension")
        return values

    def transform(self, values):
        """Center and whiten in float64, preserving every leading dimension."""
        values = self._values(values)
        result = (values - self.mean) @ self.whitening_matrix
        if not np.isfinite(result).all():
            raise ValueError("whitened conditions are nonfinite")
        return result

    def inverse(self, values):
        """Undo this fitted transform in float64 without reducing dimension."""
        values = self._values(values)
        result = values @ self.inverse_matrix + self.mean
        if not np.isfinite(result).all():
            raise ValueError("inverse-whitened conditions are nonfinite")
        return result

    def to_dict(self):
        """Return JSON-compatible fitted numerical state, with no closures."""
        return {"format": _FORMAT, "training_rows": self.training_rows, "dimension": self.dimension,
                "relative_eigenvalue_floor": self.relative_eigenvalue_floor,
                **{name: getattr(self, name).tolist() for name in (
                    "mean", "whitening_matrix", "inverse_matrix", "eigenvalues", "safe_eigenvalues")}}

    @classmethod
    def from_dict(cls, record):
        """Load only complete, finite and internally consistent fitted state."""
        if not isinstance(record, Mapping) or set(record) != _FIELDS or record["format"] != _FORMAT:
            raise ValueError("unsupported or incomplete whitening state")
        if type(record["dimension"]) is not int or record["dimension"] < 1:
            raise ValueError("whitening dimension must be a positive integer")
        result = cls(**{name: record[name] for name in _FIELDS - {"format", "dimension"}})
        if result.dimension != record["dimension"]:
            raise ValueError("declared whitening dimension does not match fitted state")
        return result


def fit_whitening(training_conditions, relative_eigenvalue_floor=1e-8):
    """Fit empirical noisy-training ZCA with sample covariance (ddof=1).

    The mean, covariance, eigendecomposition and matrix construction all use
    float64. The caller may cast the transformed result to the network dtype.
    """
    values = _real_array(training_conditions, "training_conditions")
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("training_conditions must be a nonempty feature matrix with at least two rows")
    floor = relative_eigenvalue_floor
    if (not isinstance(floor, Real) or isinstance(floor, (bool, np.bool_))
            or not np.isfinite(floor) or not 0 < floor <= 1):
        raise ValueError("relative_eigenvalue_floor must be finite and in (0, 1]")
    if np.all(values == values[0]):
        raise ValueError("training covariance must contain positive variance")
    mean = values.mean(axis=0, dtype=np.float64)
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    if not np.isfinite(covariance).all():
        raise ValueError("training covariance must be finite")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[-1] <= 0:
        raise ValueError("training covariance must contain positive variance")
    safe = np.maximum(eigenvalues, eigenvalues[-1] * float(floor))
    matrix = (eigenvectors / np.sqrt(safe)) @ eigenvectors.T
    inverse = (eigenvectors * np.sqrt(safe)) @ eigenvectors.T
    return FullCovarianceWhitening(mean, matrix, inverse, eigenvalues, safe,
                                   int(values.shape[0]), float(floor))
