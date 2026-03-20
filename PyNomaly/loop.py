from concurrent.futures import ProcessPoolExecutor, as_completed
from math import erf, sqrt
import numpy as np
import os
from python_utils.terminal import get_terminal_size
import sys
from typing import Tuple, Union
import warnings

try:
    from scipy.spatial.distance import cdist as _scipy_cdist
except ImportError:
    _scipy_cdist = None

try:
    from scipy.special import erf as _scipy_erf
except ImportError:
    _scipy_erf = None

try:
    import numba

    def _numba_distance_kernel_parallel(clust_points_vector, n_neighbors):
        n = clust_points_vector.shape[0]
        d_features = clust_points_vector.shape[1]
        local_distances = np.full((n, n_neighbors), 9e10, dtype=np.float64)
        local_indexes = np.full((n, n_neighbors), 0, dtype=np.int64)
        for i in numba.prange(n):
            for j in range(n):
                if i == j:
                    continue
                d = 0.0
                for f in range(d_features):
                    diff_val = clust_points_vector[i, f] - clust_points_vector[j, f]
                    d += diff_val * diff_val
                d = d ** 0.5
                idx_max = 0
                for k in range(1, n_neighbors):
                    if local_distances[i, k] > local_distances[i, idx_max]:
                        idx_max = k
                if d < local_distances[i, idx_max]:
                    local_distances[i, idx_max] = d
                    local_indexes[i, idx_max] = j
        return local_distances, local_indexes

    _numba_kernel_parallel = numba.jit(
        _numba_distance_kernel_parallel, nopython=True, parallel=True, cache=True
    )

    def _numba_distance_kernel_sequential(clust_points_vector, n_neighbors):
        n = clust_points_vector.shape[0]
        d_features = clust_points_vector.shape[1]
        local_distances = np.full((n, n_neighbors), 9e10, dtype=np.float64)
        local_indexes = np.full((n, n_neighbors), 0, dtype=np.int64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = 0.0
                for f in range(d_features):
                    diff_val = clust_points_vector[i, f] - clust_points_vector[j, f]
                    d += diff_val * diff_val
                d = d ** 0.5
                idx_max = 0
                for k in range(1, n_neighbors):
                    if local_distances[i, k] > local_distances[i, idx_max]:
                        idx_max = k
                if d < local_distances[i, idx_max]:
                    local_distances[i, idx_max] = d
                    local_indexes[i, idx_max] = j
        return local_distances, local_indexes

    _numba_kernel_sequential = numba.jit(
        _numba_distance_kernel_sequential, nopython=True, cache=True
    )

except ImportError:
    pass

__author__ = "Valentino Constantinou"
__version__ = "0.3.5"
__license__ = "Apache License, Version 2.0"


def _cluster_distances_worker(args):
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    clust_points_vector, global_indices, n_neighbors = args
    n = clust_points_vector.shape[0]

    if clust_points_vector.ndim == 1:
        clust_points_vector = clust_points_vector.reshape(-1, 1)

    local_distances = np.full((n, n_neighbors), 9e10, dtype=float)
    local_indexes = np.full((n, n_neighbors), 9e10, dtype=float)

    chunk_size = min(256, n)
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        chunk = clust_points_vector[chunk_start:chunk_end]

        if _scipy_cdist is not None:
            dist = _scipy_cdist(chunk, clust_points_vector, metric="euclidean")
        else:
            diff = chunk[:, np.newaxis, :] - clust_points_vector[np.newaxis, :, :]
            dist = np.sqrt((diff ** 2).sum(axis=2))

        row_idx = np.arange(chunk_end - chunk_start)
        dist[row_idx, row_idx + chunk_start] = np.inf

        knn_idx = np.argpartition(dist, n_neighbors, axis=1)[:, :n_neighbors]
        knn_dists = np.take_along_axis(dist, knn_idx, axis=1)

        local_distances[chunk_start:chunk_end] = knn_dists
        local_indexes[chunk_start:chunk_end] = global_indices[knn_idx]

    return local_distances, local_indexes, global_indices


# Custom Exceptions
class PyNomalyError(Exception):
    """Base exception for PyNomaly."""
    pass


class ValidationError(PyNomalyError):
    """Raised when input validation fails."""
    pass


class ClusterSizeError(ValidationError):
    """Raised when cluster size is smaller than n_neighbors."""
    pass


class MissingValuesError(ValidationError):
    """Raised when data contains missing values."""
    pass


class Utils:
    @staticmethod
    def emit_progress_bar(progress: str, index: int, total: int) -> str:
        """
        A progress bar that is continuously updated in Python's standard
        out.
        :param progress: a string printed to stdout that is updated and later
        returned.
        :param index: the current index of the iteration within the tracked
        process.
        :param total: the total length of the tracked process.
        :return: progress string.
        """

        w, h = get_terminal_size()
        sys.stdout.write("\r")
        if total < w:
            block_size = int(w / total)
        else:
            block_size = int(total / w)
        if index % block_size == 0:
            progress += "="
        percent = index / total
        sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
        sys.stdout.flush()
        return progress


class LocalOutlierProbability(object):
    """
    :param data: a Pandas DataFrame or Numpy array of float data
    :param extent: an integer value [1, 2, 3] that controls the statistical 
    extent, e.g. lambda times the standard deviation from the mean (optional, 
    default 3)
    :param n_neighbors: the total number of neighbors to consider w.r.t. each 
    sample (optional, default 10)
    :param cluster_labels: a numpy array of cluster assignments w.r.t. each 
    sample (optional, default None)
    :param n_jobs: the number of parallel workers for distance computation.
    Use -1 to use all available CPU cores, or 1 for sequential processing
    (optional, default 1)
    :return:
    """ """

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

    """
    Validation methods.
    These methods validate inputs and raise exceptions or warnings as appropriate.
    """

    @staticmethod
    def _convert_to_array(obj: Union["pd.DataFrame", np.ndarray]) -> np.ndarray:
        """
        Converts the input data to a numpy array if it is a Pandas DataFrame
        or validates it is already a numpy array.
        :param obj: user-provided input data.
        :return: a vector of values to be used in calculating the local
        outlier probability.
        """
        if obj.__class__.__name__ == "DataFrame":
            points_vector = obj.values
            return points_vector
        elif obj.__class__.__name__ == "ndarray":
            points_vector = obj
            return points_vector
        else:
            warnings.warn(
                "Provided data or distance matrix must be in ndarray "
                "or DataFrame.",
                UserWarning,
            )
            if isinstance(obj, list):
                points_vector = np.array(obj)
                return points_vector
            points_vector = np.array([obj])
            return points_vector

    def _validate_inputs(self):
        """
        Validates the inputs provided during initialization to ensure
        that the needed objects are provided.
        :return: a tuple of (data, distance_matrix, neighbor_matrix) or
        raises a warning for invalid inputs.
        """
        if all(v is None for v in [self.data, self.distance_matrix]):
            warnings.warn(
                "Data or a distance matrix must be provided.", UserWarning
            )
            return False
        elif all(v is not None for v in [self.data, self.distance_matrix]):
            warnings.warn(
                "Only one of the following may be provided: data or a "
                "distance matrix (not both).",
                UserWarning,
            )
            return False
        if self.data is not None:
            points_vector = self._convert_to_array(self.data)
            return points_vector, self.distance_matrix, self.neighbor_matrix
        if all(
            matrix is not None
            for matrix in [self.neighbor_matrix, self.distance_matrix]
        ):
            dist_vector = self._convert_to_array(self.distance_matrix)
            neigh_vector = self._convert_to_array(self.neighbor_matrix)
        else:
            warnings.warn(
                "A neighbor index matrix and distance matrix must both be "
                "provided when not using raw input data.",
                UserWarning,
            )
            return False
        if self.distance_matrix.shape != self.neighbor_matrix.shape:
            warnings.warn(
                "The shape of the distance and neighbor "
                "index matrices must match.",
                UserWarning,
            )
            return False
        elif (self.distance_matrix.shape[1] != self.n_neighbors) or (
            self.neighbor_matrix.shape[1] != self.n_neighbors
        ):
            warnings.warn(
                "The shape of the distance or "
                "neighbor index matrix does not "
                "match the number of neighbors "
                "specified.",
                UserWarning,
            )
            return False
        return self.data, dist_vector, neigh_vector

    def _check_cluster_size(self) -> None:
        """
        Validates the cluster labels to ensure that the smallest cluster
        size (number of observations in the cluster) is larger than the
        specified number of neighbors.
        :raises ClusterSizeError: if any cluster is too small.
        """
        c_labels = self._cluster_labels()
        for cluster_id in set(c_labels):
            c_size = np.where(c_labels == cluster_id)[0].shape[0]
            if c_size <= self.n_neighbors:
                raise ClusterSizeError(
                    "Number of neighbors specified larger than smallest "
                    "cluster. Specify a number of neighbors smaller than "
                    "the smallest cluster size (observations in smallest "
                    "cluster minus one)."
                )

    def _check_n_neighbors(self) -> bool:
        """
        Validates the specified number of neighbors to ensure that it is
        greater than 0 and that the specified value is less than the total
        number of observations.
        :return: a boolean indicating whether validation has passed without
        adjustment.
        """
        if not self.n_neighbors > 0:
            self.n_neighbors = 10
            warnings.warn(
                "n_neighbors must be greater than 0."
                " Fit with " + str(self.n_neighbors) + " instead.",
                UserWarning,
            )
            return False
        elif self.n_neighbors >= self._n_observations():
            self.n_neighbors = self._n_observations() - 1
            warnings.warn(
                "n_neighbors must be less than the number of observations."
                " Fit with " + str(self.n_neighbors) + " instead.",
                UserWarning,
            )
        return True

    def _check_extent(self) -> bool:
        """
        Validates the specified extent parameter to ensure it is either 1,
        2, or 3.
        :return: a boolean indicating whether validation has passed.
        """
        if self.extent not in [1, 2, 3]:
            warnings.warn(
                "extent parameter (lambda) must be 1, 2, or 3.", UserWarning
            )
            return False
        return True

    def _check_missing_values(self) -> None:
        """
        Validates the provided data to ensure that it contains no
        missing values.
        :raises MissingValuesError: if data contains NaN values.
        """
        if np.any(np.isnan(self.data)):
            raise MissingValuesError(
                "Method does not support missing values in input data."
            )

    def _check_is_fit(self) -> bool:
        """
        Checks that the model was fit prior to calling the stream() method.
        :return: a boolean indicating whether the model has been fit.
        """
        if self.is_fit is False:
            warnings.warn(
                "Must fit on historical data by calling fit() prior to "
                "calling stream(x).",
                UserWarning,
            )
            return False
        return True

    def _check_no_cluster_labels(self) -> bool:
        """
        Checks to see if cluster labels are attempting to be used in
        stream() and, if so, returns False. As PyNomaly does not accept
        clustering algorithms as input, the stream approach does not
        support clustering.
        :return: a boolean indicating whether single cluster (no labels).
        """
        if len(set(self._cluster_labels())) > 1:
            warnings.warn(
                "Stream approach does not support clustered data. "
                "Automatically refit using single cluster of points.",
                UserWarning,
            )
            return False
        return True

    """
    Decorators.
    """

    def accepts(*types):
        """
        A decorator that facilitates a form of type checking for the inputs
        which can be used in Python 3.4-3.7 in lieu of Python 3.5+'s type
        hints.
        :param types: the input types of the objects being passed as arguments
        in __init__.
        :return: a decorator.
        """

        def decorator(f):
            assert len(types) == f.__code__.co_argcount

            def new_f(*args, **kwds):
                for a, t in zip(args, types):
                    if type(a).__name__ == "DataFrame":
                        a = np.array(a)
                    if isinstance(a, t) is False:
                        warnings.warn(
                            "Argument %r is not of type %s" % (a, t), UserWarning
                        )
                opt_types = {
                    "distance_matrix": {"type": types[2]},
                    "neighbor_matrix": {"type": types[3]},
                    "extent": {"type": types[4]},
                    "n_neighbors": {"type": types[5]},
                    "cluster_labels": {"type": types[6]},
                    "use_numba": {"type": types[7]},
                    "n_jobs": {"type": types[8]},
                    "progress_bar": {"type": types[9]},
                }
                for x in kwds:
                    opt_types[x]["value"] = kwds[x]
                for k in opt_types:
                    try:
                        if (
                            isinstance(opt_types[k]["value"], opt_types[k]["type"])
                            is False
                        ):
                            warnings.warn(
                                "Argument %r is not of type %s."
                                % (k, opt_types[k]["type"]),
                                UserWarning,
                            )
                    except KeyError:
                        pass
                return f(*args, **kwds)

            new_f.__name__ = f.__name__
            return new_f

        return decorator

    @accepts(
        object,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        (int, np.integer),
        (int, np.integer),
        list,
        bool,
        (int, np.integer),
        bool,
    )
    def __init__(
        self,
        data=None,
        distance_matrix=None,
        neighbor_matrix=None,
        extent=3,
        n_neighbors=10,
        cluster_labels=None,
        use_numba=False,
        n_jobs=1,
        progress_bar=False,
    ) -> None:
        self.data = data
        self.distance_matrix = distance_matrix
        self.neighbor_matrix = neighbor_matrix
        self.extent = extent
        self.n_neighbors = n_neighbors
        self.cluster_labels = cluster_labels
        self.use_numba = use_numba
        self.n_jobs = n_jobs
        self.points_vector = None
        self.prob_distances = None
        self.prob_distances_ev = None
        self.norm_prob_local_outlier_factor = None
        self.local_outlier_probabilities = None
        self._objects = {}
        self.progress_bar = progress_bar
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

        self._validate_inputs()
        self._check_extent()

    """
    Private methods.
    """

    @staticmethod
    def _standard_distance(cardinality: float, sum_squared_distance: float) -> float:
        """
        Calculates the standard distance of an observation.
        :param cardinality: the cardinality of the input observation.
        :param sum_squared_distance: the sum squared distance between all
        neighbors of the input observation.
        :return: the standard distance.
        #"""
        division_result = sum_squared_distance / cardinality
        st_dist = sqrt(division_result)
        return st_dist

    @staticmethod
    def _prob_distance(extent: int, standard_distance: float) -> float:
        """
        Calculates the probabilistic distance of an observation.
        :param extent: the extent value specified during initialization.
        :param standard_distance: the standard distance of the input
        observation.
        :return: the probabilistic distance.
        """
        return extent * standard_distance

    @staticmethod
    def _prob_outlier_factor(
        probabilistic_distance: np.ndarray, ev_prob_dist: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the probabilistic outlier factor of an observation.
        :param probabilistic_distance: the probabilistic distance of the
        input observation.
        :param ev_prob_dist:
        :return: the probabilistic outlier factor.
        """
        if np.all(probabilistic_distance == ev_prob_dist):
            return np.zeros(probabilistic_distance.shape)
        else:
            ev_prob_dist[ev_prob_dist == 0.0] = 1.0e-8
            result = np.divide(probabilistic_distance, ev_prob_dist) - 1.0
            return result

    @staticmethod
    def _norm_prob_outlier_factor(
        extent: float, ev_probabilistic_outlier_factor: list
    ) -> list:
        """
        Calculates the normalized probabilistic outlier factor of an
        observation.
        :param extent: the extent value specified during initialization.
        :param ev_probabilistic_outlier_factor: the expected probabilistic
        outlier factor of the input observation.
        :return: the normalized probabilistic outlier factor.
        """
        ev_arr = np.array(ev_probabilistic_outlier_factor, dtype=float)
        return (extent * np.sqrt(ev_arr)).tolist()

    @staticmethod
    def _local_outlier_probability(
        plof_val: np.ndarray, nplof_val: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the local outlier probability of an observation.
        :param plof_val: the probabilistic outlier factor of the input
        observation.
        :param nplof_val: the normalized probabilistic outlier factor of the
        input observation.
        :return: the local outlier probability.
        """
        if np.all(plof_val == nplof_val):
            return np.zeros(plof_val.shape)
        plof_f = np.asarray(plof_val, dtype=float)
        nplof_f = np.asarray(nplof_val, dtype=float)
        if _scipy_erf is not None:
            return np.maximum(0, _scipy_erf(plof_f / (nplof_f * np.sqrt(2.0))))
        erf_vec = np.vectorize(erf)
        return np.maximum(0, erf_vec(plof_f / (nplof_f * np.sqrt(2.0))))

    def _n_observations(self) -> int:
        """
        Calculates the number of observations in the data.
        :return: the number of observations in the input data.
        """
        if self.data is not None:
            return len(self.data)
        return len(self.distance_matrix)

    def _store(self) -> np.ndarray:
        """
        Initializes the storage matrix that includes the input value,
        cluster labels, local outlier probability, etc. for the input data.
        :return: an empty numpy array of shape [n_observations, 3].
        """
        return np.empty([self._n_observations(), 3], dtype=object)

    def _cluster_labels(self) -> np.ndarray:
        """
        Returns a numpy array of cluster labels that corresponds to the
        input labels or that is an array of all 0 values to indicate all
        points belong to the same cluster.
        :return: a numpy array of cluster labels.
        """
        if self.cluster_labels is None:
            if self.data is not None:
                return np.array([0] * len(self.data))
            return np.array([0] * len(self.distance_matrix))
        return np.array(self.cluster_labels)

    @staticmethod
    def _euclidean(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
        """
        Calculates the euclidean distance between two observations in the
        input data.
        :param vector1: a numpy array corresponding to observation 1.
        :param vector2: a numpy array corresponding to observation 2.
        :return: the euclidean distance between the two observations.
        """
        diff = vector1 - vector2
        return np.dot(diff, diff) ** 0.5

    def _assign_distances(self, data_store: np.ndarray) -> np.ndarray:
        """
        Takes a distance matrix, produced by _distances or provided through
        user input, and assigns distances for each observation to the storage
        matrix, data_store.
        :param data_store: the storage matrix that collects information on
        each observation.
        :return: the updated storage matrix that collects information on
        each observation.
        """
        for vec, cluster_id in zip(
            range(self.distance_matrix.shape[0]), self._cluster_labels()
        ):
            data_store[vec][0] = cluster_id
            data_store[vec][1] = self.distance_matrix[vec]
            data_store[vec][2] = self.neighbor_matrix[vec]
        return data_store

    @staticmethod
    def _compute_distance_and_neighbor_matrix(
        clust_points_vector: np.ndarray,
        indices: np.ndarray,
        distances: np.ndarray,
        indexes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        This helper method provides the heavy lifting for the _distances
        method and is only intended for use therein. The code has been
        written so that it can make full use of Numba's jit capabilities if
        desired.
        """
        for i in range(clust_points_vector.shape[0]):
            for j in range(i + 1, clust_points_vector.shape[0]):
                # Global index of the points
                global_i = indices[0][i]
                global_j = indices[0][j]

                # Compute Euclidean distance
                diff = clust_points_vector[i] - clust_points_vector[j]
                d = np.dot(diff, diff) ** 0.5

                # Update distance and neighbor index for global_i
                idx_max = distances[global_i].argmax()
                if d < distances[global_i][idx_max]:
                    distances[global_i][idx_max] = d
                    indexes[global_i][idx_max] = global_j

                # Update distance and neighbor index for global_j
                idx_max = distances[global_j].argmax()
                if d < distances[global_j][idx_max]:
                    distances[global_j][idx_max] = d
                    indexes[global_j][idx_max] = global_i

            yield distances, indexes, i

    def _distances_vectorized(
        self, clusters, distances, indexes, progress_bar
    ) -> None:
        """Vectorized kNN distance computation with chunked progress."""
        progress = "="
        total_points = sum(cv.shape[0] for cv, _ in clusters)
        completed = 0

        for clust_points_vector, global_indices in clusters:
            n = clust_points_vector.shape[0]

            if clust_points_vector.ndim == 1:
                clust_points_vector = clust_points_vector.reshape(-1, 1)

            chunk_size = min(256, n)
            for chunk_start in range(0, n, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n)
                chunk = clust_points_vector[chunk_start:chunk_end]

                if _scipy_cdist is not None:
                    dist = _scipy_cdist(
                        chunk, clust_points_vector, metric="euclidean"
                    )
                else:
                    diff = (
                        chunk[:, np.newaxis, :]
                        - clust_points_vector[np.newaxis, :, :]
                    )
                    dist = np.sqrt((diff ** 2).sum(axis=2))

                row_idx = np.arange(chunk_end - chunk_start)
                dist[row_idx, row_idx + chunk_start] = np.inf

                knn_idx = np.argpartition(dist, self.n_neighbors, axis=1)[
                    :, : self.n_neighbors
                ]
                knn_dists = np.take_along_axis(dist, knn_idx, axis=1)

                chunk_global = global_indices[chunk_start:chunk_end]
                distances[chunk_global] = knn_dists
                indexes[chunk_global] = global_indices[knn_idx]

                completed += chunk_end - chunk_start
                if progress_bar:
                    progress = Utils.emit_progress_bar(
                        progress, completed, total_points
                    )

    def _distances_numba(
        self, clusters, distances, indexes, progress_bar, parallel=False
    ) -> None:
        """Numba-accelerated distance computation."""
        progress = "="
        kernel = _numba_kernel_parallel if parallel else _numba_kernel_sequential

        for idx, (clust_points_vector, global_indices) in enumerate(clusters):
            if clust_points_vector.ndim == 1:
                clust_points_vector = clust_points_vector.reshape(-1, 1)

            local_dists, local_idxs = kernel(
                clust_points_vector.astype(np.float64), self.n_neighbors
            )

            distances[global_indices] = local_dists
            indexes[global_indices] = global_indices[local_idxs]

            if progress_bar:
                progress = Utils.emit_progress_bar(
                    progress, idx + 1, len(clusters)
                )

    def _distances_parallel(
        self, clusters, distances, indexes, n_jobs, progress_bar
    ) -> None:
        """Parallel distance computation across clusters via multiprocessing."""
        progress = "="
        worker_args = [
            (cv, gi, self.n_neighbors) for cv, gi in clusters
        ]

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(_cluster_distances_worker, args): idx
                for idx, args in enumerate(worker_args)
            }
            completed_clusters = 0
            for future in as_completed(futures):
                local_dists, local_idxs, global_indices = future.result()
                distances[global_indices] = local_dists
                indexes[global_indices] = local_idxs
                completed_clusters += 1
                if progress_bar:
                    progress = Utils.emit_progress_bar(
                        progress, completed_clusters, len(clusters)
                    )

    def _distances(self, progress_bar: bool = False) -> None:
        """
        Provides the distances between each observation and it's closest
        neighbors. When input data is provided, calculates the euclidean
        distance between every observation. Otherwise, the user-provided
        distance matrix is used.
        :return: the updated storage matrix that collects information on
        each observation.
        """
        distances = np.full(
            [self._n_observations(), self.n_neighbors], 9e10, dtype=float
        )
        indexes = np.full(
            [self._n_observations(), self.n_neighbors], 9e10, dtype=float
        )
        self.points_vector = self._convert_to_array(self.data)

        cluster_labels = self._cluster_labels()
        cluster_ids = sorted(set(cluster_labels))

        clusters = []
        for cluster_id in cluster_ids:
            indices = np.where(cluster_labels == cluster_id)
            clust_points_vector = np.array(
                self.points_vector.take(indices, axis=0)[0], dtype=np.float64
            )
            clusters.append((clust_points_vector, indices[0]))

        n_jobs = self.n_jobs
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1
        n_jobs = min(n_jobs, len(clusters))

        if self.use_numba:
            self._distances_numba(
                clusters, distances, indexes, progress_bar,
                parallel=(n_jobs > 1)
            )
        elif n_jobs > 1 and len(clusters) > 1:
            self._distances_parallel(
                clusters, distances, indexes, n_jobs, progress_bar
            )
        else:
            self._distances_vectorized(
                clusters, distances, indexes, progress_bar
            )

        self.distance_matrix = distances
        self.neighbor_matrix = indexes

    def _ssd(self, data_store: np.ndarray) -> np.ndarray:
        """
        Calculates the sum squared distance between neighbors for each
        observation in the input data.
        :param data_store: the storage matrix that collects information on
        each observation.
        :return: the updated storage matrix that collects information on
        each observation.
        """
        self.cluster_labels_u = np.unique(data_store[:, 0])
        ssd_array = np.empty([self._n_observations(), 1])
        for cluster_id in self.cluster_labels_u:
            indices = np.where(data_store[:, 0] == cluster_id)
            cluster_distances = np.take(data_store[:, 1], indices).tolist()
            ssd = np.power(cluster_distances[0], 2).sum(axis=1)
            for i, j in zip(indices[0], ssd):
                ssd_array[i] = j
        data_store = np.hstack((data_store, ssd_array))
        return data_store

    def _standard_distances(self, data_store: np.ndarray) -> np.ndarray:
        """
        Calculated the standard distance for each observation in the input
        data. First calculates the cardinality and then calculates the standard
        distance with respect to each observation.
        :param data_store: the storage matrix that collects information on
        each observation.
        :return: the updated storage matrix that collects information on
        each observation.
        """
        ssd_vals = data_store[:, 3].astype(float)
        std_distances = np.sqrt(ssd_vals / self.n_neighbors)
        return np.hstack((data_store, std_distances.reshape(-1, 1)))

    def _prob_distances(self, data_store: np.ndarray) -> np.ndarray:
        """
        Calculates the probabilistic distance for each observation in the
        input data.
        :param data_store: the storage matrix that collects information on
        each observation.
        :return: the updated storage matrix that collects information on
        each observation.
        """
        prob_distances = self.extent * data_store[:, 4].astype(float)
        return np.hstack((data_store, prob_distances.reshape(-1, 1)))

    def _prob_distances_ev(self, data_store) -> np.ndarray:
        """
        Calculates the expected value of the probabilistic distance for
        each observation in the input data with respect to the cluster the
        observation belongs to.
        :param data_store: the storage matrix that collects information on
        each observation.
        :return: the updated storage matrix that collects information on
        each observation.
        """
        prob_set_distance_ev = np.empty([self._n_observations(), 1])
        for cluster_id in self.cluster_labels_u:
            indices = np.where(data_store[:, 0] == cluster_id)[0]
            for index in indices:
                # Global neighbor indices for the current point
                nbrhood = data_store[index][2].astype(int)  # Ensure global indices
                nbrhood_prob_distances = np.take(data_store[:, 5], nbrhood).astype(
                    float
                )
                nbrhood_prob_distances_nonan = nbrhood_prob_distances[
                    np.logical_not(np.isnan(nbrhood_prob_distances))
                ]
                prob_set_distance_ev[index] = nbrhood_prob_distances_nonan.mean()

        self.prob_distances_ev = prob_set_distance_ev
        return np.hstack((data_store, prob_set_distance_ev))

    def _prob_local_outlier_factors(self, data_store: np.ndarray) -> np.ndarray:
        """
        Calculates the probabilistic local outlier factor for each
        observation in the input data.
        :param data_store: the storage matrix that collects information on
        each observation.
        :return: the updated storage matrix that collects information on
        each observation.
        """
        return np.hstack(
            (
                data_store,
                np.array(
                    [
                        np.apply_along_axis(
                            self._prob_outlier_factor,
                            0,
                            data_store[:, 5],
                            data_store[:, 6],
                        )
                    ]
                ).T,
            )
        )

    def _prob_local_outlier_factors_ev(self, data_store: np.ndarray) -> np.ndarray:
        """
        Calculates the expected value of the probabilistic local outlier factor
        for each observation in the input data with respect to the cluster the
        observation belongs to.
        :param data_store: the storage matrix that collects information on
        each observation.
        :return: the updated storage matrix that collects information on
        each observation.
        """
        prob_local_outlier_factor_ev_dict = {}
        for cluster_id in self.cluster_labels_u:
            indices = np.where(data_store[:, 0] == cluster_id)
            prob_local_outlier_factors = np.take(data_store[:, 7], indices).astype(
                float
            )
            prob_local_outlier_factors_nonan = prob_local_outlier_factors[
                np.logical_not(np.isnan(prob_local_outlier_factors))
            ]
            prob_local_outlier_factor_ev_dict[cluster_id] = np.power(
                prob_local_outlier_factors_nonan, 2
            ).sum() / float(prob_local_outlier_factors_nonan.size)
        data_store = np.hstack(
            (
                data_store,
                np.array(
                    [
                        [
                            prob_local_outlier_factor_ev_dict[x]
                            for x in data_store[:, 0].tolist()
                        ]
                    ]
                ).T,
            )
        )
        return data_store

    def _norm_prob_local_outlier_factors(self, data_store: np.ndarray) -> np.ndarray:
        """
        Calculates the normalized probabilistic local outlier factor for each
        observation in the input data.
        :param data_store: the storage matrix that collects information on
        each observation.
        :return: the updated storage matrix that collects information on
        each observation.
        """
        return np.hstack(
            (
                data_store,
                np.array(
                    [
                        self._norm_prob_outlier_factor(
                            self.extent, data_store[:, 8].tolist()
                        )
                    ]
                ).T,
            )
        )

    def _local_outlier_probabilities(self, data_store: np.ndarray) -> np.ndarray:
        """
        Calculates the local outlier probability for each observation in the
        input data.
        :param data_store: the storage matrix that collects information on
        each observation.
        :return: the updated storage matrix that collects information on
        each observation.
        """
        return np.hstack(
            (
                data_store,
                np.array(
                    [
                        np.apply_along_axis(
                            self._local_outlier_probability,
                            0,
                            data_store[:, 7],
                            data_store[:, 9],
                        )
                    ]
                ).T,
            )
        )

    """
    Public methods
    """

    def fit(self) -> "LocalOutlierProbability":
        """
        Calculates the local outlier probability for each observation in the
        input data according to the input parameters extent, n_neighbors, and
        cluster_labels.
        :return: self, which contains the local outlier probabilities as
        self.local_outlier_probabilities.
        :raises ClusterSizeError: if any cluster is smaller than n_neighbors.
        :raises MissingValuesError: if data contains missing values.
        """

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
