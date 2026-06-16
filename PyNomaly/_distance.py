# Authors: Valentino Constantinou <vc@valentino.io>
# License: Apache 2.0

import numpy as np
import os
import sys
from typing import Tuple
import warnings

from PyNomaly._utils import Utils

try:
    from scipy.spatial.distance import cdist as _scipy_cdist
except ImportError:
    _scipy_cdist = None

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


class DistanceMixin:
    """Mixin providing distance computation methods for LocalOutlierProbability."""

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
                global_i = indices[0][i]
                global_j = indices[0][j]

                diff = clust_points_vector[i] - clust_points_vector[j]
                d = np.dot(diff, diff) ** 0.5

                idx_max = distances[global_i].argmax()
                if d < distances[global_i][idx_max]:
                    distances[global_i][idx_max] = d
                    indexes[global_i][idx_max] = global_j

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

        if self.use_numba:
            self._distances_numba(
                clusters, distances, indexes, progress_bar,
                parallel=(n_jobs > 1)
            )
        else:
            if n_jobs > 1:
                warnings.warn(
                    "n_jobs > 1 requires use_numba=True for parallel "
                    "processing. Install Numba and set use_numba=True "
                    "to enable parallelism. Falling back to sequential.",
                    UserWarning,
                )
            self._distances_vectorized(
                clusters, distances, indexes, progress_bar
            )

        self.distance_matrix = distances
        self.neighbor_matrix = indexes
