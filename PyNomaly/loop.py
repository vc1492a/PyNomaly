import itertools
from math import erf
import numpy as np
import sys
import warnings

__author__ = 'Valentino Constantinou'
__version__ = '0.2.6'
__license__ = 'Apache License, Version 2.0'


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
    :return:
    """"""

    Based on the work of Kriegel, Kröger, Schubert, and Zimek (2009) in LoOP: 
    Local Outlier Probabilities.
    ----------

    References
    ----------
    .. [1] Breunig M., Kriegel H.-P., Ng R., Sander, J. LOF: Identifying Density-based Local Outliers. ACM SIGMOD
           International Conference on Management of Data (2000).
    .. [2] Kriegel H.-P., Kröger P., Schubert E., Zimek A. LoOP: Local Outlier Probabilities. 18th ACM conference on 
           Information and knowledge management, CIKM (2009).
    .. [3] Goldstein M., Uchida S. A Comparative Evaluation of Unsupervised Anomaly
           Detection Algorithms for Multivariate Data. PLoS ONE 11(4): e0152173 (2016).
    .. [4] Hamlet C., Straub J., Russell M., Kerlin S. An incremental and approximate local outlier probability 
           algorithm for intrusion detection and its evaluation. Journal of Cyber Security Technology (2016). 
    """

    class Validate:

        @staticmethod
        def data(obj):
            if obj.__class__.__name__ == 'DataFrame':
                points_vector = obj.values
                return points_vector
            elif obj.__class__.__name__ == 'ndarray':
                points_vector = obj
                return points_vector
            else:
                warnings.warn(
                    "Provided data or distance matrix must be in ndarray " 
                    "or DataFrame.",
                    UserWarning)
                if isinstance(obj, list):
                    points_vector = np.array(obj)
                    return points_vector
                points_vector = np.array([obj])
                return points_vector

        def inputs(self, obj):
            if all(v is None for v in [obj.data, obj.distance_matrix]):
                warnings.warn(
                    "Data or a distance matrix must be provided."
                )
                return False
            elif all(v is not None for v in [obj.data, obj.distance_matrix]):
                warnings.warn(
                    "Only one of the following may be provided: data or a "
                    "distance matrix (not both)."
                )
                return False
            if obj.data is not None:
                points_vector = self.data(obj.data)
                return points_vector, obj.distance_matrix, obj.neighbor_matrix
            if obj.distance_matrix is not None:
                dist_vector = self.data(obj.distance_matrix)
                if obj.neighbor_matrix is not None:
                    neigh_vector = self.data(obj.neighbor_matrix)
                    if obj.distance_matrix.shape == obj.neighbor_matrix.shape:
                        if (obj.distance_matrix.shape[1] != obj.n_neighbors) \
                                or (obj.neighbor_matrix.shape[1] !=
                                    obj.n_neighbors):
                            warnings.warn("The shape of the distance or "
                                          "neighbor index matrix does not "
                                          "match the number of neighbors "
                                          "specified.")
                            return False
                        return obj.data, dist_vector, neigh_vector
                    else:
                        warnings.warn("The shape of the distance and neighbor "
                                      "index matrices must match.")
                        return False
                else:
                    warnings.warn(
                        "A neighbor index matrix must be provided when using "
                        "a distance matrix."
                    )
                    return False

        @staticmethod
        def cluster_size(obj):
            c_labels = obj._cluster_labels()
            for cluster_id in set(c_labels):
                c_size = np.where(c_labels == cluster_id)[0].shape[0]
                if c_size <= obj.n_neighbors:
                    warnings.warn(
                        "Number of neighbors specified larger than smallest "
                        "cluster. Specify a number of neighbors smaller than "
                        "the smallest cluster size (observations in smallest "
                        "cluster minus one).",
                        UserWarning)
                    return False

        @staticmethod
        def n_neighbors(obj, set_neighbors=False):
            if not obj.n_neighbors > 0:
                warnings.warn("n_neighbors must be greater than 0.",
                              UserWarning)
                return False
            elif obj.n_neighbors >= obj._n_observations():
                if set_neighbors is True:
                    obj.n_neighbors = obj._n_observations() - 1
                warnings.warn(
                    "n_neighbors must be less than the number of observations."
                    " Fit with " + str(obj.n_neighbors) + " instead.",
                    UserWarning)
                return True

        @staticmethod
        def extent(obj):
            if obj.extent not in [1, 2, 3]:
                warnings.warn(
                    "extent parameter (lambda) must be 1, 2, or 3.",
                    UserWarning)
                return False
            else:
                return True

        @staticmethod
        def missing_values(obj):
            if np.any(np.isnan(obj.data)):
                warnings.warn(
                    "Method does not support missing values in input data. ",
                    UserWarning)
                return False
            else:
                return True

        @staticmethod
        def fit(obj):
            if obj.is_fit is False:
                warnings.warn(
                    "Must fit on historical data by calling fit() prior to "
                    "calling stream(x).",
                    UserWarning)
                return False
            else:
                return True

        @staticmethod
        def no_cluster_labels(obj):
            if len(set(obj._cluster_labels())) > 1:
                warnings.warn(
                    "Stream approach does not support clustered data. "
                    "Automatically refit using single cluster of points.",
                    UserWarning)
                return False
            else:
                return True

    def accepts(*types):
        def decorator(f):
            assert len(types) == f.__code__.co_argcount

            def new_f(*args, **kwds):
                for (a, t) in zip(args, types):
                    if type(a).__name__ == 'DataFrame':
                        a = np.array(a)
                    if isinstance(a, t) is False:
                        warnings.warn("Argument %r is not of type %s" % (a, t),
                                      UserWarning)
                opt_types = {
                    'distance_matrix': {
                        'type': types[2]
                    },
                    'neighbor_matrix': {
                        'type': types[3]
                    },
                    'extent': {
                        'type': types[4]
                    },
                    'n_neighbors': {
                        'type': types[5]
                    },
                    'cluster_labels': {
                        'type': types[6]
                    }
                }
                for x in kwds:
                    opt_types[x]['value'] = kwds[x]
                for k in opt_types:
                    try:
                        if isinstance(opt_types[k]['value'],
                                      opt_types[k]['type']) is False:
                            warnings.warn("Argument %r is not of type %s" % (
                                k, opt_types[k]['type']), UserWarning)
                    except KeyError:
                        pass
                return f(*args, **kwds)

            new_f.__name__ = f.__name__
            return new_f

        return decorator

    @accepts(object, np.ndarray, np.ndarray, np.ndarray, (int, np.integer),
             (int, np.integer), list)
    def __init__(self, data=None, distance_matrix=None, neighbor_matrix=None,
                 extent=3, n_neighbors=10, cluster_labels=None):
        self.data = data
        self.distance_matrix = distance_matrix
        self.neighbor_matrix = neighbor_matrix
        self.extent = extent
        self.n_neighbors = n_neighbors
        self.cluster_labels = cluster_labels
        self.points_vector = None
        self.prob_distances = None
        self.prob_distances_ev = None
        self.norm_prob_local_outlier_factor = None
        self.local_outlier_probabilities = None
        self._objects = {}
        self.is_fit = False

        self.Validate().inputs(self)
        self.Validate.extent(self)

    @staticmethod
    def _standard_distance(cardinality, sum_squared_distance):
        st_dist = np.array(
            [np.sqrt(i) for i in np.divide(sum_squared_distance, cardinality)])
        return st_dist

    @staticmethod
    def _prob_distance(extent, standard_distance):
        return extent * standard_distance

    @staticmethod
    def _prob_outlier_factor(probabilistic_distance, ev_prob_dist):
        if np.all(probabilistic_distance == ev_prob_dist):
            return np.zeros(probabilistic_distance.shape)
        else:
            ev_prob_dist[ev_prob_dist == 0.] = 1.e-8
            result = np.divide(probabilistic_distance, ev_prob_dist) - 1.
            return result

    @staticmethod
    def _norm_prob_outlier_factor(extent, ev_probabilistic_outlier_factor):
        ev_probabilistic_outlier_factor = [i for i in
                                           ev_probabilistic_outlier_factor]
        return extent * np.sqrt(ev_probabilistic_outlier_factor)

    @staticmethod
    def _local_outlier_probability(plof_val, nplof_val):
        erf_vec = np.vectorize(erf)
        if np.all(plof_val == nplof_val):
            return np.zeros(plof_val.shape)
        else:
            return np.maximum(0, erf_vec(plof_val / (nplof_val * np.sqrt(2.))))

    def _n_observations(self):
        if self.data is not None:
            return len(self.data)
        return len(self.distance_matrix)

    def _store(self):
        return np.empty([self._n_observations(), 3], dtype=object)

    def _cluster_labels(self):
        if self.cluster_labels is None:
            if self.data is not None:
                return np.array([0] * len(self.data))
            return np.array([0] * len(self.distance_matrix))
        else:
            return np.array(self.cluster_labels)

    @staticmethod
    def _euclidean(vector1, vector2):
        diff = vector1 - vector2
        return np.dot(diff, diff) ** 0.5

    def _distances(self, data_store):
        if self.data is not None:
            distances = np.full([self._n_observations(), self.n_neighbors], 9e10,
                                dtype=float)
            indexes = np.full([self._n_observations(), self.n_neighbors], 9e10,
                              dtype=float)
            self.points_vector = self.Validate.data(self.data)
            for cluster_id in set(self._cluster_labels()):
                indices = np.where(self._cluster_labels() == cluster_id)
                clust_points_vector = self.points_vector.take(indices, axis=0)[0]
                pairs = itertools.permutations(
                    np.ndindex(clust_points_vector.shape[0]), 2)
                for p in pairs:
                    d = self._euclidean(clust_points_vector[p[0]],
                                        clust_points_vector[p[1]])
                    idx = indices[0][p[0]]
                    idx_max = np.argmax(distances[idx])
                    if d < distances[idx][idx_max]:
                        distances[idx][idx_max] = d
                        indexes[idx][idx_max] = p[1][0]
            self.distance_matrix = distances
            self.neighbor_matrix = indexes
        for vec, cluster_id in zip(range(self.distance_matrix.shape[0]),
                                   self._cluster_labels()):
            data_store[vec][0] = cluster_id
            data_store[vec][1] = self.distance_matrix[vec]
            data_store[vec][2] = self.neighbor_matrix[vec]
        return data_store

    def _ssd(self, data_store):
        self.cluster_labels_u = np.unique(data_store[:, 0])
        ssd_array = np.empty([self._n_observations(), 1])
        for cluster_id in self.cluster_labels_u:
            indices = np.where(data_store[:, 0] == cluster_id)
            cluster_distances = np.take(data_store[:, 1], indices).tolist()
            ssd = np.sum(np.power(cluster_distances[0], 2), axis=1)
            for i, j in zip(indices[0], ssd):
                ssd_array[i] = j
        data_store = np.hstack((data_store, ssd_array))
        return data_store

    def _standard_distances(self, data_store):
        cardinality = np.array([self.n_neighbors] * self._n_observations())
        return np.hstack(
            (data_store,
             np.array([np.apply_along_axis(self._standard_distance, 0,
                                           cardinality, data_store[:, 3])]).T))

    def _prob_distances(self, data_store):
        return np.hstack((data_store, np.array(
            [self._prob_distance(self.extent, data_store[:, 4])]).T))

    def _prob_distances_ev(self, data_store):
        prob_set_distance_ev = np.empty([self._n_observations(), 1])
        for cluster_id in self.cluster_labels_u:
            indices = np.where(data_store[:, 0] == cluster_id)[0]
            for index in indices:
                nbrhood = data_store[index][2].astype(int)
                nbrhood_prob_distances = np.take(data_store[:, 5],
                                                 nbrhood).astype(float)
                nbrhood_prob_distances_nonan = nbrhood_prob_distances[
                    np.logical_not(np.isnan(nbrhood_prob_distances))]
                prob_set_distance_ev[index] = np.mean(
                    nbrhood_prob_distances_nonan)
        self.prob_distances_ev = prob_set_distance_ev
        data_store = np.hstack((data_store, prob_set_distance_ev))
        return data_store

    def _prob_local_outlier_factors(self, data_store):
        return np.hstack(
            (data_store,
             np.array([np.apply_along_axis(self._prob_outlier_factor, 0,
                                           data_store[:, 5],
                                           data_store[:, 6])]).T))

    def _prob_local_outlier_factors_ev(self, data_store):
        prob_local_outlier_factor_ev_dict = {}
        for cluster_id in self.cluster_labels_u:
            indices = np.where(data_store[:, 0] == cluster_id)
            prob_local_outlier_factors = np.take(data_store[:, 7],
                                                 indices).astype(float)
            prob_local_outlier_factors_nonan = prob_local_outlier_factors[
                np.logical_not(np.isnan(prob_local_outlier_factors))]
            prob_local_outlier_factor_ev_dict[cluster_id] = np.sum(
                np.power(
                    prob_local_outlier_factors_nonan, 2)
            ) / float(prob_local_outlier_factors_nonan.size)
        data_store = np.hstack(
            (data_store, np.array([[prob_local_outlier_factor_ev_dict[x] for x
                                    in data_store[:, 0].tolist()]]).T))
        return data_store

    def _norm_prob_local_outlier_factors(self, data_store):
        return np.hstack((data_store, np.array([self._norm_prob_outlier_factor(
            self.extent, data_store[:, 8])]).T))

    def _local_outlier_probabilities(self, data_store):
        return np.hstack(
            (data_store,
             np.array([np.apply_along_axis(self._local_outlier_probability, 0,
                                           data_store[:, 7],
                                           data_store[:, 9])]).T))

    def fit(self):

        if self.data is not None:
            self.Validate.n_neighbors(self, set_neighbors=True)
            self.Validate.cluster_size(self)
            if self.Validate.missing_values(self) is False:
                sys.exit()

        store = self._store()
        store = self._distances(store)
        store = self._ssd(store)
        store = self._standard_distances(store)
        store = self._prob_distances(store)
        self.prob_distances = store[:, 5]
        store = self._prob_distances_ev(store)
        store = self._prob_local_outlier_factors(store)
        store = self._prob_local_outlier_factors_ev(store)
        store = self._norm_prob_local_outlier_factors(store)
        self.norm_prob_local_outlier_factor = np.max(store[:, 9])
        store = self._local_outlier_probabilities(store)
        self.local_outlier_probabilities = store[:, 10]

        self.is_fit = True

        return self

    def stream(self, x):

        if self.Validate.fit(self) is False:
            self.fit()

        if self.Validate.no_cluster_labels(self) is False:
            self.cluster_labels = np.array([0] * len(self.data))
            self.fit()

        point_vector = self.Validate.data(x)
        distances = np.full([1, self.n_neighbors], 9e10, dtype=float)
        if self.data is not None:
            matrix = self.points_vector
        else:
            matrix = self.distance_matrix
        for p in range(0, matrix.shape[0]):
            if self.data is not None:
                d = self._euclidean(matrix[p, :], point_vector)
            else:
                d = point_vector
            idx_max = np.argmax(distances[0])
            if d < distances[0][idx_max]:
                distances[0][idx_max] = d

        ssd = np.sum(np.power(distances, 2))
        std_dist = np.sqrt(np.divide(ssd, self.n_neighbors))
        prob_dist = self._prob_distance(self.extent, std_dist)
        plof = self._prob_outlier_factor(prob_dist,
                                         np.array(
                                             np.mean(self.prob_distances_ev))
                                         )
        loop = self._local_outlier_probability(
            plof, self.norm_prob_local_outlier_factor)

        return loop
