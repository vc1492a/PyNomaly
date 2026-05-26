import numpy as np
from PyNomaly import loop
import time


def main():
    # generate a large set of data
    data = np.ones(shape=(10000, 4))

    # Vectorized NumPy (default, no Numba)
    t1 = time.time()
    scores_numpy = loop.LocalOutlierProbability(
        data,
        n_neighbors=3,
        use_numba=False,
        progress_bar=True
    ).fit().local_outlier_probabilities
    t2 = time.time()
    seconds_no_numba = t2 - t1
    print("\nComputation took " + str(seconds_no_numba) + " seconds without Numba JIT.")

    # Numba JIT (sequential)
    t3 = time.time()
    scores_numba = loop.LocalOutlierProbability(
        data,
        n_neighbors=3,
        use_numba=True,
        progress_bar=True
    ).fit().local_outlier_probabilities
    t4 = time.time()
    seconds_numba = t4 - t3
    print("\nComputation took " + str(seconds_numba) + " seconds with Numba JIT.")

    # Multi-cluster parallel example
    np.random.seed(42)
    cluster_a = np.random.randn(5000, 4) + 0
    cluster_b = np.random.randn(5000, 4) + 10
    multi_data = np.vstack([cluster_a, cluster_b])
    cluster_labels = [0] * 5000 + [1] * 5000

    # Sequential (n_jobs=1) with clusters
    t5 = time.time()
    scores_seq = loop.LocalOutlierProbability(
        multi_data,
        n_neighbors=3,
        cluster_labels=cluster_labels,
        n_jobs=1,
        progress_bar=True
    ).fit().local_outlier_probabilities
    t6 = time.time()
    seconds_seq = t6 - t5
    print("\nComputation took " + str(seconds_seq) + " seconds with n_jobs=1 (2 clusters).")

    # Parallel (n_jobs=-1) with clusters
    t7 = time.time()
    scores_par = loop.LocalOutlierProbability(
        multi_data,
        n_neighbors=3,
        cluster_labels=cluster_labels,
        n_jobs=-1,
        progress_bar=True
    ).fit().local_outlier_probabilities
    t8 = time.time()
    seconds_par = t8 - t7
    print("\nComputation took " + str(seconds_par) + " seconds with n_jobs=-1 (2 clusters).")


if __name__ == '__main__':
    main()
