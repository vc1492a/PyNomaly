# Authors: Valentino Constantinou <vc@valentino.io>
# License: Apache 2.0

"""Input validation helpers for LocalOutlierProbability."""

from __future__ import annotations

import warnings
from functools import wraps
from typing import Union

import numpy as np

from PyNomaly._compat import check_array
from PyNomaly.exceptions import ClusterSizeError, MissingValuesError

try:
    from scipy.sparse import issparse as _issparse

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover

    _SCIPY_AVAILABLE = False

    def _issparse(x):
        return False


def _is_sparse_like(obj) -> bool:
    """Return True for scipy sparse matrices (with or without scipy installed)."""
    if _issparse(obj):
        return True
    obj_type = type(obj)
    module = getattr(obj_type, "__module__", "") or ""
    return module.startswith("scipy.sparse")


def _require_scipy_for_sparse(obj) -> None:
    """Raise a clear error when sparse input is passed without scipy."""
    if _is_sparse_like(obj) and not _SCIPY_AVAILABLE:
        raise ImportError(
            "Sparse matrix input requires scipy. "
            "Install it with: pip install scipy"
        )


class ValidationMixin:
    """Mixin providing input validation methods for LocalOutlierProbability."""

    def _validate_data_matrix(
        self,
        X,
        *,
        ensure_2d: bool = True,
        reset: bool = False,
    ) -> np.ndarray:
        """
        Validate and densify feature matrices for fit / predict / stream.

        Sparse inputs are accepted and converted to dense arrays. This keeps
        the public API sparse-capable while the LoOP kernels operate on dense
        float64 data.
        """
        if X is None:
            raise ValueError("Input data must be provided.")

        # Preserve DataFrame column names before densifying / converting.
        feature_names = None
        if hasattr(X, "columns"):
            try:
                feature_names = np.asarray(X.columns, dtype=object)
            except Exception:
                feature_names = None

        if _is_sparse_like(X):
            _require_scipy_for_sparse(X)
            X = X.toarray()

        # Historical LoOP API accepts 1-d series and reshapes to a single feature.
        arr_probe = np.asarray(X)
        if getattr(arr_probe, "ndim", None) == 1:
            if arr_probe.__class__.__name__ != "ndarray" or not isinstance(X, np.ndarray):
                warnings.warn(
                    "Provided data or distance matrix must be in ndarray or DataFrame.",
                    UserWarning,
                )
            X = np.asarray(arr_probe, dtype=float).reshape(-1, 1)

        # Prefer PyNomaly's MissingValuesError over sklearn's generic ValueError.
        try:
            probe = np.asarray(X, dtype=float)
        except (ValueError, TypeError):
            probe = None
        if probe is not None and (np.any(np.isnan(probe)) or np.any(np.isinf(probe))):
            raise MissingValuesError(
                "Input contains NaN, infinity or a value too large for "
                "dtype('float64')."
            )

        arr = check_array(
            X,
            accept_sparse=False,
            dtype="numeric",
            ensure_2d=ensure_2d,
            ensure_min_samples=1,
            ensure_min_features=1,
        )
        arr = np.asarray(arr, dtype=float)

        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise MissingValuesError(
                "Input contains NaN, infinity or a value too large for "
                "dtype('float64')."
            )

        if reset:
            self.n_features_in_ = arr.shape[1] if arr.ndim == 2 else 1
            if feature_names is not None and len(feature_names) == self.n_features_in_:
                self.feature_names_in_ = feature_names

        return arr

    @staticmethod
    def _convert_to_array(obj: Union["pd.DataFrame", np.ndarray]) -> np.ndarray:
        """
        Converts the input data to a numpy array if it is a Pandas DataFrame
        or validates it is already a numpy array.
        """
        if _is_sparse_like(obj):
            _require_scipy_for_sparse(obj)
            obj = obj.toarray()

        if obj.__class__.__name__ in ("DataFrame", "Series"):
            arr = np.asarray(obj.values)
        elif obj.__class__.__name__ == "ndarray":
            arr = obj
        else:
            warnings.warn(
                "Provided data or distance matrix must be in ndarray or DataFrame.",
                UserWarning,
            )
            arr = np.asarray(obj, dtype=float)
            if arr.ndim == 0:
                arr = np.array([obj], dtype=float)

        if arr.size == 0:
            if arr.ndim >= 2 and arr.shape[1] == 0:
                raise ValueError(
                    f"Found array with 0 feature(s) (shape={arr.shape}) while a "
                    f"minimum of 1 is required."
                )
            raise ValueError(
                f"Found array with 0 sample(s) (shape={arr.shape}) while a "
                f"minimum of 1 is required."
            )

        if np.iscomplexobj(arr):
            raise ValueError("Complex data not supported.")

        return arr.astype(float, copy=False)

    @staticmethod
    def _convert_observation(obj: Union["pd.DataFrame", np.ndarray]) -> np.ndarray:
        """
        Convert a single observation to a 1-D float vector.

        Accepts dense arrays, pandas objects, lists, and scipy sparse rows.
        Sparse matrices are densified (requires scipy).
        """
        arr = ValidationMixin._convert_to_array(obj)
        if arr.ndim == 2 and arr.shape[0] == 1:
            return arr.ravel()
        return arr

    def _validate_inputs(self):
        """
        Validates that either data or a distance/neighbor matrix pair is set.
        """
        _data = getattr(self, "data_", getattr(self, "data", None))
        _dist = getattr(
            self, "distance_matrix_", getattr(self, "distance_matrix", None)
        )
        _neigh = getattr(
            self, "neighbor_matrix_", getattr(self, "neighbor_matrix", None)
        )

        if all(v is None for v in [_data, _dist]):
            warnings.warn("Data or a distance matrix must be provided.", UserWarning)
            return False
        if all(v is not None for v in [_data, _dist]):
            warnings.warn(
                "Only one of the following may be provided: data or a "
                "distance matrix (not both).",
                UserWarning,
            )
            return False

        if _data is not None:
            points_vector = self._validate_data_matrix(_data, ensure_2d=True, reset=True)
            self.data_ = points_vector
            return points_vector, _dist, _neigh

        if all(matrix is not None for matrix in [_neigh, _dist]):
            dist_vector = self._convert_to_array(_dist)
            neigh_vector = self._convert_to_array(_neigh)
        else:
            warnings.warn(
                "A neighbor index matrix and distance matrix must both be "
                "provided when not using raw input data.",
                UserWarning,
            )
            return False

        if _dist.shape != _neigh.shape:
            warnings.warn(
                "The shape of the distance and neighbor index matrices must match.",
                UserWarning,
            )
            return False
        if (_dist.shape[1] != self.n_neighbors) or (
            _neigh.shape[1] != self.n_neighbors
        ):
            warnings.warn(
                "The shape of the distance or neighbor index matrix does not "
                "match the number of neighbors specified.",
                UserWarning,
            )
            return False

        self.distance_matrix_ = dist_vector
        self.neighbor_matrix_ = neigh_vector
        return _data, dist_vector, neigh_vector

    def _check_cluster_size(self) -> None:
        """Raises ClusterSizeError if any cluster is too small for n_neighbors."""
        n_neighbors = getattr(self, "n_neighbors_", self.n_neighbors)
        c_labels = self._cluster_labels()
        for cluster_id in set(c_labels):
            c_size = np.where(c_labels == cluster_id)[0].shape[0]
            if c_size <= n_neighbors:
                raise ClusterSizeError(
                    "Number of neighbors specified larger than smallest "
                    "cluster. Specify a number of neighbors smaller than "
                    "the smallest cluster size (observations in smallest "
                    "cluster minus one)."
                )

    def _check_n_neighbors(self) -> bool:
        """
        Validate n_neighbors without mutating the public parameter.

        The effective value used during fit is stored on ``n_neighbors_``.
        """
        n_neighbors = self.n_neighbors
        n_obs = self._n_observations()

        if not n_neighbors > 0:
            warnings.warn(
                "n_neighbors must be greater than 0. Fit with 10 instead.",
                UserWarning,
            )
            n_neighbors = 10
            ok = False
        else:
            ok = True

        if n_neighbors >= n_obs:
            adjusted = max(n_obs - 1, 1)
            warnings.warn(
                "n_neighbors must be less than the number of observations."
                f" Fit with {adjusted} instead.",
                UserWarning,
            )
            n_neighbors = adjusted
            ok = False

        self.n_neighbors_ = n_neighbors
        return ok

    def _check_extent(self) -> bool:
        """Validate that extent is 1, 2, or 3."""
        if self.extent not in [1, 2, 3]:
            warnings.warn("extent parameter (lambda) must be 1, 2, or 3.", UserWarning)
            return False
        return True

    def _check_missing_values(self) -> None:
        """Raise MissingValuesError if fitted data contains NaN/Inf."""
        _data = getattr(self, "data_", getattr(self, "data", None))
        if _data is not None:
            arr = self._convert_to_array(_data)
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                raise MissingValuesError(
                    "Input contains NaN, infinity or a value too large for "
                    "dtype('float64')."
                )

    def _check_is_fit(self) -> bool:
        """Return whether the estimator has been successfully fit."""
        if getattr(self, "is_fit_", False) is False:
            warnings.warn(
                "Must fit on historical data by calling fit() prior to "
                "calling stream(x).",
                UserWarning,
            )
            return False
        return True

    def _check_no_cluster_labels(self) -> bool:
        """Return False when multi-cluster labels are present (stream unsupported)."""
        if len(set(self._cluster_labels())) > 1:
            warnings.warn(
                "Stream approach does not support clustered data. "
                "Automatically refit using single cluster of points.",
                UserWarning,
            )
            return False
        return True

    def _effective_n_neighbors(self) -> int:
        """Return the neighbor count used by the current fit."""
        return getattr(self, "n_neighbors_", self.n_neighbors)


def validate_init_types(*types):
    """
    Soft type checker used at fit-time for legacy constructor parameters.
    """

    def decorator(f):
        @wraps(f)
        def new_f(self, *args, **kwds):
            attr_names = [
                "extent",
                "n_neighbors",
                "use_numba",
                "n_jobs",
                "progress_bar",
                "data",
                "distance_matrix",
                "neighbor_matrix",
                "cluster_labels",
            ]
            for attr, expected_type in zip(attr_names, types[1:]):
                val = getattr(self, attr, None)
                if val is not None:
                    if type(val).__name__ == "DataFrame":
                        val = np.array(val)
                    if not isinstance(val, expected_type):
                        warnings.warn(
                            "Argument %r is not of type %s." % (attr, expected_type),
                            UserWarning,
                        )
            return f(self, *args, **kwds)

        return new_f

    return decorator
