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
    print("\nComputation took " + str(seconds_numba) + " seconds with Numba JIT (sequential).")

    # Numba JIT (parallel, n_jobs=-1)
    t5 = time.time()
    scores_numba_par = loop.LocalOutlierProbability(
        data,
        n_neighbors=3,
        use_numba=True,
        n_jobs=-1,
        progress_bar=True
    ).fit().local_outlier_probabilities
    t6 = time.time()
    seconds_numba_par = t6 - t5
    print("\nComputation took " + str(seconds_numba_par) + " seconds with Numba JIT (parallel, n_jobs=-1).")


if __name__ == '__main__':
    main()
