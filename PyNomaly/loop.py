# Authors: Valentino Constantinou <vc@valentino.io>
# License: Apache 2.0

"""Local Outlier Probability (LoOP) estimator."""

from __future__ import annotations

import sys
import warnings

import numpy as np

from PyNomaly._compat import (
    BaseEstimator,
    OutlierMixin,
    check_array,
    check_is_fitted,
)
from PyNomaly._distance import DistanceMixin
from PyNomaly._pipeline import PipelineMixin
from PyNomaly._validation import ValidationMixin, validate_init_types

__author__ = "Valentino Constantinou"
__version__ = "1.0.0"
__license__ = "Apache License, Version 2.0"


class LocalOutlierProbability(
    OutlierMixin, ValidationMixin, DistanceMixin, PipelineMixin, BaseEstimator
):
    """Local Outlier Probability (LoOP) estimator.

    Parameters
    ----------
    extent : int, default=3
        Statistical extent (lambda); must be 1, 2, or 3.
    n_neighbors : int, default=10
        Number of neighbors considered for each sample.
    use_numba : bool, default=False
        Whether to use Numba JIT acceleration for distance computation.
    n_jobs : int, default=1
        Numba parallelism. ``-1`` uses all cores. Only effective with
        ``use_numba=True``.
    progress_bar : bool, default=False
        Whether to display a progress bar during distance computation.

    Notes
    -----
    Fitted attributes end with ``_`` (scikit-learn convention). Legacy
    read aliases without the trailing underscore remain available for
    *output* attributes such as ``local_outlier_probabilities``. Init /
    configuration attributes (e.g. ``distance_matrix``) are never aliased
    to fitted state — that would break ``get_params`` / ``check_estimator``.

    Sparse matrices (``scipy.sparse``) are accepted for ``fit``, ``predict``,
    ``decision_function``, and ``stream``. They are converted to dense arrays
    internally before distance computation. **scipy** is a soft dependency for
    sparse input — install it with ``pip install scipy`` (included in the
    ``PyNomaly[all]`` extra).

    References
    ----------
    .. [1] Breunig M., Kriegel H.-P., Ng R., Sander, J. LOF: Identifying
           Density-based Local Outliers. ACM SIGMOD (2000).
    .. [2] Kriegel H.-P., Kröger P., Schubert E., Zimek A. LoOP: Local Outlier
           Probabilities. CIKM (2009).
    """

    _DATA_PARAMS = ("data", "distance_matrix", "neighbor_matrix", "cluster_labels")

    # ------------------------------------------------------------------
    # Backward-compatible read aliases for *fitted output* attributes.
    # These are NOT constructor parameters, so they do not affect
    # get_params() / set_params() / check_estimator param comparisons.
    # ------------------------------------------------------------------
    @property
    def points_vector(self):
        return getattr(self, "points_vector_", None)

    @property
    def prob_distances(self):
        return getattr(self, "prob_distances_", None)

    @property
    def prob_distances_ev(self):
        return getattr(self, "prob_distances_ev_", None)

    @property
    def norm_prob_local_outlier_factor(self):
        return getattr(self, "norm_prob_local_outlier_factor_", None)

    @property
    def local_outlier_probabilities(self):
        return getattr(self, "local_outlier_probabilities_", None)

    @property
    def is_fit(self):
        return getattr(self, "is_fit_", False)

    def __init__(
        self,
        extent=3,
        n_neighbors=10,
        use_numba=False,
        n_jobs=1,
        progress_bar=False,
        data=None,
        distance_matrix=None,
        neighbor_matrix=None,
        cluster_labels=None,
    ) -> None:
        # Configuration parameters only — never mutate these in fit().
        self.extent = extent
        self.n_neighbors = n_neighbors
        self.use_numba = use_numba
        self.n_jobs = n_jobs
        self.progress_bar = progress_bar

        # Deprecated data-in-constructor path (backward compatible).
        _locals = {
            "data": data,
            "distance_matrix": distance_matrix,
            "neighbor_matrix": neighbor_matrix,
            "cluster_labels": cluster_labels,
        }
        for param in self._DATA_PARAMS:
            if _locals[param] is not None:
                warnings.warn(
                    f"Passing '{param}' to __init__ is deprecated. "
                    "Pass it to fit() instead. This will raise an error "
                    "in a future version.",
                    FutureWarning,
                    stacklevel=2,
                )

        self.data = data
        self.distance_matrix = distance_matrix
        self.neighbor_matrix = neighbor_matrix
        self.cluster_labels = cluster_labels

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # Sparse inputs are densified in validation; tag as supported.
        tags.input_tags.sparse = True
        return tags

    def _reset_state(self) -> None:
        """Clear fitted state so the estimator can be re-fit."""
        self.points_vector_ = None
        self.prob_distances_ = None
        self.prob_distances_ev_ = None
        self.norm_prob_local_outlier_factor_ = None
        self.local_outlier_probabilities_ = None
        self.data_ = None
        self.distance_matrix_ = None
        self.neighbor_matrix_ = None
        self.cluster_labels_ = None
        self.n_neighbors_ = self.n_neighbors
        self._objects_ = {}
        self.is_fit_ = False

    def _resolve_runtime_flags(self):
        """Return (use_numba, n_jobs) without mutating public params."""
        use_numba = self.use_numba
        if use_numba is True and "numba" not in sys.modules:
            warnings.warn(
                "Numba is not available, falling back to pure python mode.",
                UserWarning,
            )
            use_numba = False

        n_jobs = self.n_jobs
        if n_jobs < -1 or n_jobs == 0:
            warnings.warn(
                "n_jobs must be -1 or a positive integer. Defaulting to 1.",
                UserWarning,
            )
            n_jobs = 1
        return use_numba, n_jobs

    @validate_init_types(
        object,
        (int, np.integer),
        (int, np.integer),
        bool,
        (int, np.integer),
        bool,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list,
    )
    def fit(
        self,
        X=None,
        y=None,
        data=None,
        distance_matrix=None,
        neighbor_matrix=None,
        cluster_labels=None,
    ) -> "LocalOutlierProbability":
        """
        Fit the LoOP model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), optional
            Training data (scikit-learn style). Sparse matrices are densified.
        y : ignored
            Present for API consistency. For backward compatibility only,
            a 2-d ``y`` with more than one column is treated as a deprecated
            positional ``distance_matrix`` when no raw data (``X`` / ``data``)
            is given.
        data, distance_matrix, neighbor_matrix, cluster_labels
            Explicit keyword alternatives to ``X`` / precomputed neighbors.
        """
        if X is not None:
            data = X

        # Legacy positional fit(data, distance_matrix) support. Only applies
        # when no raw data is given and y cannot be a label vector, so that
        # sklearn-style fit(X, y) with (n, 1) labels is not hijacked.
        if y is not None and distance_matrix is None and data is None:
            y_array = np.asarray(y)
            if y_array.ndim == 2 and y_array.shape[1] > 1:
                distance_matrix = y
                y = None
                warnings.warn(
                    "Passing 'distance_matrix' as the second positional "
                    "argument is deprecated. Use keyword arguments instead.",
                    FutureWarning,
                )

        use_numba, n_jobs = self._resolve_runtime_flags()
        self._fit_use_numba = use_numba
        self._fit_n_jobs = n_jobs

        self._check_extent()
        self._reset_state()

        self.data_ = data if data is not None else self.data
        self.distance_matrix_ = (
            distance_matrix if distance_matrix is not None else self.distance_matrix
        )
        self.neighbor_matrix_ = (
            neighbor_matrix if neighbor_matrix is not None else self.neighbor_matrix
        )
        self.cluster_labels_ = (
            cluster_labels if cluster_labels is not None else self.cluster_labels
        )

        if self._validate_inputs() is False:
            return self

        n_obs = self._n_observations()
        if n_obs < 2:
            raise ValueError(
                f"Found array with {n_obs} sample(s) while a minimum of 2 "
                f"is required by LocalOutlierProbability."
            )

        self._check_n_neighbors()
        self._check_cluster_size()

        _data = getattr(self, "data_", None)
        if _data is not None:
            self._check_missing_values()

        store = self._store()
        if _data is not None:
            self._distances(progress_bar=self.progress_bar)

        store = self._assign_distances(store)
        store = self._ssd(store)
        store = self._standard_distances(store)
        store = self._prob_distances(store)
        self.prob_distances_ = np.asarray(store[:, 5], dtype=float)
        store = self._prob_distances_ev(store)
        store = self._prob_local_outlier_factors(store)
        store = self._prob_local_outlier_factors_ev(store)
        store = self._norm_prob_local_outlier_factors(store)
        self.norm_prob_local_outlier_factor_ = float(
            np.asarray(store[:, 9], dtype=float).max()
        )
        store = self._local_outlier_probabilities(store)
        self.local_outlier_probabilities_ = np.asarray(store[:, 10], dtype=float)

        if _data is not None and self.points_vector_ is not None:
            self.n_features_in_ = (
                self.points_vector_.shape[1]
                if self.points_vector_.ndim == 2
                else 1
            )
        elif self.distance_matrix_ is not None:
            self.n_features_in_ = 1

        # sklearn outlier API: decision_function = score_samples - offset_
        # score_samples = -LoOP probability (higher is more normal)
        # offset_ = -0.5 so decision >= 0 iff probability <= 0.5
        self.offset_ = -0.5
        self.is_fit_ = True
        return self

    def stream(self, x: np.ndarray) -> float:
        """Score a single observation against the fitted model."""
        if self._check_is_fit() is False:
            self.fit()

        _data = getattr(self, "data_", getattr(self, "data", None))
        _dist = getattr(
            self, "distance_matrix_", getattr(self, "distance_matrix", None)
        )
        _clust = getattr(self, "cluster_labels_", getattr(self, "cluster_labels", None))

        orig_cluster_labels = None
        if self._check_no_cluster_labels() is False:
            orig_cluster_labels = _clust
            if getattr(self, "points_vector_", None) is not None:
                self.cluster_labels_ = np.array([0] * self.points_vector_.shape[0])

        point_vector = self._convert_observation(x)
        n_neighbors = self._effective_n_neighbors()
        distances = np.full([1, n_neighbors], 9e10, dtype=float)

        if _data is not None:
            matrix = getattr(self, "points_vector_", None)
        else:
            matrix = _dist
            if point_vector.size == 1:
                point_vector = float(point_vector.flat[0])

        if matrix is None:
            from PyNomaly._compat import NotFittedError

            raise NotFittedError(
                "This LocalOutlierProbability instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this "
                "estimator."
            )

        for p in range(0, matrix.shape[0]):
            if _data is not None:
                ref_point = matrix[p] if matrix.ndim == 1 else matrix[p, :]
                d = float(self._euclidean(ref_point, point_vector))
            else:
                d = float(point_vector)

            idx_max = distances[0].argmax()
            if d < distances[0][idx_max]:
                distances[0][idx_max] = d

        ssd = np.power(distances, 2).sum()
        std_dist = np.sqrt(np.divide(ssd, n_neighbors))
        prob_dist = self._prob_distance(self.extent, std_dist)
        plof = self._prob_outlier_factor(
            np.array(prob_dist), np.array(self.prob_distances_ev_.mean())
        )
        loop_score = self._local_outlier_probability(
            plof, self.norm_prob_local_outlier_factor_
        )

        if orig_cluster_labels is not None:
            self.cluster_labels_ = orig_cluster_labels

        return float(np.asarray(loop_score, dtype=float).ravel()[0])

    def _score_matrix(self, X) -> np.ndarray:
        """
        Return LoOP probabilities for ``X``.

        Training rows are matched to fitted scores so predict / decision
        function stay order- and subset-invariant. Unseen rows use ``stream``.
        """
        check_is_fitted(self, ["is_fit_"])

        if _issparse_safe(X):
            from PyNomaly._validation import _require_scipy_for_sparse

            _require_scipy_for_sparse(X)
            X = X.toarray()

        X_probe = np.asarray(X)
        if X_probe.ndim == 1 and getattr(self, "n_features_in_", None) == 1:
            X = X_probe.reshape(-1, 1)

        X_array = check_array(
            X,
            accept_sparse=False,
            dtype="numeric",
            ensure_2d=True,
        )
        X_array = np.asarray(X_array, dtype=float)

        if np.any(np.isnan(X_array)) or np.any(np.isinf(X_array)):
            raise ValueError(
                "Input contains NaN, infinity or a value too large for "
                "dtype('float64')."
            )

        if hasattr(self, "n_features_in_"):
            n_features = X_array.shape[1]
            if n_features != self.n_features_in_:
                raise ValueError(
                    f"X has {n_features} features, but {type(self).__name__} "
                    f"is expecting {self.n_features_in_} features as input."
                )

        train = getattr(self, "points_vector_", None)
        fitted = getattr(self, "local_outlier_probabilities_", None)
        scores = np.empty(X_array.shape[0], dtype=float)

        if train is not None and fitted is not None and train.ndim == 2:
            for i, row in enumerate(X_array):
                matches = np.where(np.all(np.isclose(train, row), axis=1))[0]
                if matches.size:
                    scores[i] = fitted[matches[0]]
                else:
                    scores[i] = float(self.stream(row))
            return scores

        for i, row in enumerate(X_array):
            scores[i] = float(self.stream(row))
        return scores

    def decision_function(self, X):
        """
        Signed outlier score (higher is more normal).

        Equivalent to ``score_samples(X) - offset_``. Inliers satisfy
        ``decision_function(X) >= 0``.
        """
        return self.score_samples(X) - self.offset_

    def score_samples(self, X):
        """
        Opposite of the LoOP probability (higher is more normal).
        """
        return -1.0 * self._score_matrix(X)

    def predict(self, X):
        """
        Predict inliers (1) and outliers (-1).

        Labels are derived from ``decision_function(X) >= 0``.
        """
        decision = self.decision_function(X)
        return np.where(decision >= 0, 1, -1)


def _issparse_safe(x) -> bool:
    try:
        from scipy.sparse import issparse

        return issparse(x)
    except ImportError:
        return False


LoOP = LocalOutlierProbability
