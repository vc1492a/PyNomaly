# Authors: Valentino Constantinou <vc@valentino.io>
# License: Apache 2.0

from math import erf, sqrt
import numpy as np

try:
    from scipy.special import erf as _scipy_erf
except ImportError:
    _scipy_erf = None


class PipelineMixin:
    """Mixin providing the LoOP scoring pipeline for LocalOutlierProbability."""

    @staticmethod
    def _standard_distance(cardinality: float, sum_squared_distance: float) -> float:
        """
        Calculates the standard distance of an observation.
        :param cardinality: the cardinality of the input observation.
        :param sum_squared_distance: the sum squared distance between all
        neighbors of the input observation.
        :return: the standard distance.
        """
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
                nbrhood = data_store[index][2].astype(int)
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
