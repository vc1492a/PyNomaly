# Authors: Valentino Constantinou <vc@valentino.io>
# License: Apache 2.0

import numpy as np
import sys
import warnings

from PyNomaly.exceptions import (
    PyNomalyError,
    ValidationError,
    ClusterSizeError,
    MissingValuesError,
)
from PyNomaly._utils import Utils
from PyNomaly._validation import ValidationMixin, accepts
from PyNomaly._distance import DistanceMixin
from PyNomaly._pipeline import PipelineMixin

__author__ = "Valentino Constantinou"
__version__ = "1.0.0"
__license__ = "Apache License, Version 2.0"


class LocalOutlierProbability(ValidationMixin, DistanceMixin, PipelineMixin):
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

    _DATA_PARAMS = ("data", "distance_matrix", "neighbor_matrix",
                    "cluster_labels")

    @accepts(
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
        _locals = {"data": data, "distance_matrix": distance_matrix,
                   "neighbor_matrix": neighbor_matrix,
                   "cluster_labels": cluster_labels}
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

        self.points_vector = None
        self.prob_distances = None
        self.prob_distances_ev = None
        self.norm_prob_local_outlier_factor = None
        self.local_outlier_probabilities = None
        self._objects = {}
        self.is_fit = False

        if self.use_numba is True and "numba" not in sys.modules:
            self.use_numba = False
            warnings.warn(
                "Numba is not available, falling back to pure python mode.", UserWarning
            )

        if self.n_jobs < -1 or self.n_jobs == 0:
            warnings.warn(
                "n_jobs must be -1 or a positive integer. Defaulting to 1.",
                UserWarning,
            )
            self.n_jobs = 1

        self._check_extent()

    def _reset_state(self) -> None:
        """Resets computed state to allow re-fitting with new data."""
        self.points_vector = None
        self.prob_distances = None
        self.prob_distances_ev = None
        self.norm_prob_local_outlier_factor = None
        self.local_outlier_probabilities = None
        self._objects = {}
        self.is_fit = False

    def fit(
        self,
        data=None,
        distance_matrix=None,
        neighbor_matrix=None,
        cluster_labels=None,
    ) -> "LocalOutlierProbability":
        """
        Calculates the local outlier probability for each observation in the
        input data according to the input parameters extent, n_neighbors, and
        cluster_labels.
        :param data: a Pandas DataFrame or Numpy array of float data
        (optional, default None)
        :param distance_matrix: a precomputed distance matrix of shape
        (n_observations, n_neighbors) (optional, default None)
        :param neighbor_matrix: a precomputed neighbor index matrix of shape
        (n_observations, n_neighbors) (optional, default None)
        :param cluster_labels: a numpy array or list of cluster assignments
        w.r.t. each sample (optional, default None)
        :return: self, which contains the local outlier probabilities as
        self.local_outlier_probabilities.
        :raises ClusterSizeError: if any cluster is smaller than n_neighbors.
        :raises MissingValuesError: if data contains missing values.
        """

        self._reset_state()

        if data is not None:
            self.data = data
            self.distance_matrix = None
            self.neighbor_matrix = None
        if distance_matrix is not None:
            self.distance_matrix = distance_matrix
        if neighbor_matrix is not None:
            self.neighbor_matrix = neighbor_matrix
        if cluster_labels is not None:
            self.cluster_labels = cluster_labels

        if self._validate_inputs() is False:
            return self
        self._check_n_neighbors()
        self._check_cluster_size()
        if self.data is not None:
            self._check_missing_values()

        store = self._store()
        if self.data is not None:
            self._distances(progress_bar=self.progress_bar)
        store = self._assign_distances(store)
        store = self._ssd(store)
        store = self._standard_distances(store)
        store = self._prob_distances(store)
        self.prob_distances = store[:, 5]
        store = self._prob_distances_ev(store)
        store = self._prob_local_outlier_factors(store)
        store = self._prob_local_outlier_factors_ev(store)
        store = self._norm_prob_local_outlier_factors(store)
        self.norm_prob_local_outlier_factor = store[:, 9].max()
        store = self._local_outlier_probabilities(store)
        self.local_outlier_probabilities = store[:, 10]

        self.is_fit = True

        return self

    def stream(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the local outlier probability for an individual sample
        according to the input parameters extent, n_neighbors, and
        cluster_labels after first calling fit(). Observations are assigned
        a local outlier probability against the mean of expected values of
        probabilistic distance and the normalized probabilistic outlier
        factor from the earlier model, provided when calling fit().
        distance
        :param x: an observation to score for its local outlier probability.
        :return: the local outlier probability of the input observation.
        """

        orig_cluster_labels = None
        if self._check_no_cluster_labels() is False:
            orig_cluster_labels = self.cluster_labels
            self.cluster_labels = np.array([0] * len(self.data))

        if self._check_is_fit() is False:
            self.fit()

        point_vector = self._convert_to_array(x)
        distances = np.full([1, self.n_neighbors], 9e10, dtype=float)
        if self.data is not None:
            matrix = self.points_vector
        else:
            matrix = self.distance_matrix
            # When using distance matrix mode, x is a scalar distance value.
            # Extract scalar from array to avoid NumPy assignment errors.
            if point_vector.size == 1:
                point_vector = float(point_vector.flat[0])
        for p in range(0, matrix.shape[0]):
            if self.data is not None:
                d = self._euclidean(matrix[p, :], point_vector)
            else:
                d = point_vector
            idx_max = distances[0].argmax()
            if d < distances[0][idx_max]:
                distances[0][idx_max] = d

        ssd = np.power(distances, 2).sum()
        std_dist = np.sqrt(np.divide(ssd, self.n_neighbors))
        prob_dist = self._prob_distance(self.extent, std_dist)
        plof = self._prob_outlier_factor(
            np.array(prob_dist), np.array(self.prob_distances_ev.mean())
        )
        loop = self._local_outlier_probability(
            plof, self.norm_prob_local_outlier_factor
        )

        if orig_cluster_labels is not None:
            self.cluster_labels = orig_cluster_labels

        return loop


LoOP = LocalOutlierProbability
