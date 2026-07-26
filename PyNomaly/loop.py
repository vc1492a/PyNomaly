# Authors: Valentino Constantinou <vc@valentino.io>
# License: Apache 2.0

import sys
import warnings

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin

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

    :param extent: an integer value [1, 2, 3] that controls the statistical
    extent, e.g. lambda times the standard deviation from the mean (optional,
    default 3)
    :param n_neighbors: the total number of neighbors to consider w.r.t. each
    sample (optional, default 10)
    :param use_numba: whether to use Numba JIT acceleration for distance
    computation (optional, default False)
    :param n_jobs: controls Numba thread-level parallelism via prange.
    Use -1 to use all available CPU cores, or 1 for sequential processing.
    Only effective when use_numba=True (optional, default 1)
    :param progress_bar: whether to display a progress bar during distance
    computation (optional, default False)

    Based on the work of Kriegel, Kröger, Schubert, and Zimek (2009) in LoOP:
    Local Outlier Probabilities.
    ----------

    References
    ----------
    .. [1] Breunig M., Kriegel H.-P., Ng R., Sander, J. LOF: Identifying
           Density-based Local Outliers. ACM SIGMOD
           International Conference on Management of Data (2000).
    .. [2] Kriegel H.-P., Kröger P., Schubert E., Zimek A. LoOP: Local Outlier
           Probabilities. 18th ACM conference on
           Information and knowledge management, CIKM (2009).
    .. [3] Goldstein M., Uchida S. A Comparative Evaluation of Unsupervised
           Anomaly Detection Algorithms for Multivariate Data. PLoS ONE 11(4):
           e0152173 (2016).
    .. [4] Hamlet C., Straub J., Russell M., Kerlin S. An incremental and
           approximate local outlier probability algorithm for intrusion
           detection and its evaluation. Journal of Cyber Security Technology
           (2016).
    """

    _DATA_PARAMS = ("data", "distance_matrix", "neighbor_matrix", "cluster_labels")

    @property
    def points_vector(self):
        return getattr(self, "points_vector_", None)

    @points_vector.setter
    def points_vector(self, value):
        self.points_vector_ = value

    @property
    def prob_distances(self):
        return getattr(self, "prob_distances_", None)

    @prob_distances.setter
    def prob_distances(self, value):
        self.prob_distances_ = value

    @property
    def prob_distances_ev(self):
        return getattr(self, "prob_distances_ev_", None)

    @prob_distances_ev.setter
    def prob_distances_ev(self, value):
        self.prob_distances_ev_ = value

    @property
    def norm_prob_local_outlier_factor(self):
        return getattr(self, "norm_prob_local_outlier_factor_", None)

    @norm_prob_local_outlier_factor.setter
    def norm_prob_local_outlier_factor(self, value):
        self.norm_prob_local_outlier_factor_ = value

    @property
    def local_outlier_probabilities(self):
        return getattr(self, "local_outlier_probabilities_", None)

    @local_outlier_probabilities.setter
    def local_outlier_probabilities(self, value):
        self.local_outlier_probabilities_ = value

    @property
    def is_fit(self):
        return getattr(self, "is_fit_", False)

    @is_fit.setter
    def is_fit(self, value):
        self.is_fit_ = value

    @property
    def _objects(self):
        return getattr(self, "_objects_", {})

    @_objects.setter
    def _objects(self, value):
        self._objects_ = value

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
        self.extent = extent
        self.n_neighbors = n_neighbors
        self.use_numba = use_numba
        self.n_jobs = n_jobs
        self.progress_bar = progress_bar

        # Emit deprecation warnings for data params passed to __init__
        _locals = {
            "data": data,
            "distance_matrix": distance_matrix,
            "neighbor_matrix": neighbor_matrix,
            "cluster_labels": cluster_labels,
        }
        for param in self._DATA_PARAMS:
            if _locals[param] is not None:
                warnings.warn(
                    "Passing '{}' to __init__ is deprecated. "
                    "Pass it to fit() instead. This will raise an error "
                    "in a future version.".format(param),
                    FutureWarning,
                    stacklevel=2,
                )

        self.data = data
        self.distance_matrix = distance_matrix
        self.neighbor_matrix = neighbor_matrix
        self.cluster_labels = cluster_labels

    def _reset_state(self) -> None:
        """Resets computed state to allow re-fitting with new data."""
        self.points_vector_ = None
        self.prob_distances_ = None
        self.prob_distances_ev_ = None
        self.norm_prob_local_outlier_factor_ = None
        self.local_outlier_probabilities_ = None

        # Reset storage attributes
        self.data_ = None
        self.distance_matrix_ = None
        self.neighbor_matrix_ = None
        self.cluster_labels_ = None

        self._objects_ = {}
        self.is_fit_ = False

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

        if X is not None:
            data = X

        if y is not None and distance_matrix is None:
            y_array = np.asarray(y)
            if y_array.ndim == 2:
                distance_matrix = y
                y = None
                warnings.warn(
                    "Passing 'distance_matrix' as the second positional argument is deprecated. "
                    "Use keyword arguments instead.",
                    FutureWarning,
                )

        if self.use_numba is True and "numba" not in sys.modules:
            self.use_numba = False
            warnings.warn(
                "Numba is not available, falling back to pure python mode.", UserWarning
            )

        if self.n_jobs < -1 or self.n_jobs == 0:
            warnings.warn(
                "n_jobs must be -1 or a positive integer. Defaulting to 1.", UserWarning
            )
            self.n_jobs = 1

        self._check_extent()
        self._reset_state()

        # Capture state strictly into Scikit-Learn compliant fitted attributes
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

        self._check_n_neighbors()
        self._check_cluster_size()

        _data = getattr(self, "data_", getattr(self, "data", None))
        if _data is not None:
            self._check_missing_values()

        store = self._store()
        if _data is not None:
            self._distances(progress_bar=self.progress_bar)

        store = self._assign_distances(store)
        store = self._ssd(store)
        store = self._standard_distances(store)
        store = self._prob_distances(store)
        self.prob_distances_ = store[:, 5]
        store = self._prob_distances_ev(store)
        store = self._prob_local_outlier_factors(store)
        store = self._prob_local_outlier_factors_ev(store)
        store = self._norm_prob_local_outlier_factors(store)
        self.norm_prob_local_outlier_factor_ = store[:, 9].max()
        store = self._local_outlier_probabilities(store)
        self.local_outlier_probabilities_ = store[:, 10]

        if _data is not None and hasattr(self, "points_vector_") and self.points_vector_ is not None:
            self.n_features_in_ = self.points_vector_.shape[1] if self.points_vector_.ndim == 2 else 1
            if hasattr(_data, "columns"):
                self.feature_names_in_ = np.array(_data.columns, dtype=object)
        elif getattr(self, "distance_matrix_", None) is not None:
            self.n_features_in_ = self.distance_matrix_.shape[1]

        self.is_fit_ = True
        return self

    def stream(self, x: np.ndarray) -> np.ndarray:
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

        point_vector = self._convert_to_array(x)
        distances = np.full([1, self.n_neighbors], 9e10, dtype=float)

        if _data is not None:
            matrix = getattr(self, "points_vector_", None)
        else:
            matrix = _dist
            if point_vector.size == 1:
                point_vector = float(point_vector.flat[0])

        if matrix is None:
            from sklearn.exceptions import NotFittedError
            raise NotFittedError(
                "This LocalOutlierProbability instance is not fitted yet. Call 'fit' "
                "with appropriate arguments before using this estimator."
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
        std_dist = np.sqrt(np.divide(ssd, self.n_neighbors))
        prob_dist = self._prob_distance(self.extent, std_dist)
        plof = self._prob_outlier_factor(
            np.array(prob_dist), np.array(self.prob_distances_ev_.mean())
        )

        # Removed the float() cast to preserve the np.ndarray return type
        loop_score = self._local_outlier_probability(
            plof, self.norm_prob_local_outlier_factor_
        )

        if orig_cluster_labels is not None:
            self.cluster_labels_ = orig_cluster_labels

        return loop_score

    def decision_function(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, ["is_fit_"])

        X_array = self._convert_to_array(X)

        if np.any(np.isnan(X_array)) or np.any(np.isinf(X_array)):
            raise ValueError(
                "Input contains NaN, infinity or a value too large for dtype('float64')."
            )
        
        # Scikit-Learn feature consistency check
        if hasattr(self, "n_features_in_"):
            n_features = X_array.shape[1] if X_array.ndim == 2 else 1
            if n_features != self.n_features_in_:
                raise ValueError(
                    f"X has {n_features} features, but {self.__class__.__name__} "
                    f"is expecting {self.n_features_in_} features as input."
                )
        if hasattr(self, "points_vector_") and self.points_vector_ is not None:
            if X_array.shape == self.points_vector_.shape and np.allclose(X_array, self.points_vector_):
                return -1.0 * self.local_outlier_probabilities_
        
        probabilities = np.array([self.stream(x) for x in X_array])
        return -1.0 * probabilities

    def predict(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, ["is_fit_"])

        X_array = self._convert_to_array(X)

        if np.any(np.isnan(X_array)) or np.any(np.isinf(X_array)):
            raise ValueError(
                "Input contains NaN, infinity or a value too large for dtype('float64')."
            )
        
        # Scikit-Learn feature consistency check
        if hasattr(self, "n_features_in_"):
            n_features = X_array.shape[1] if X_array.ndim == 2 else 1
            if n_features != self.n_features_in_:
                raise ValueError(
                    f"X has {n_features} features, but {self.__class__.__name__} "
                    f"is expecting {self.n_features_in_} features as input."
                )
        
        if hasattr(self, "points_vector_") and self.points_vector_ is not None:
            if X_array.shape == self.points_vector_.shape and np.allclose(X_array, self.points_vector_):
                probabilities = self.local_outlier_probabilities_
                return np.where(probabilities >= 0.5, -1, 1)

        probabilities = np.array([self.stream(x) for x in X_array])
        return np.where(probabilities >= 0.5, -1, 1)

LoOP = LocalOutlierProbability
