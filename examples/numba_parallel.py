import multiprocessing as mp
import numpy as np
from PyNomaly import loop
import time

# generate a large set of data
data = np.ones(shape=(25000, 4))

for i in range(1, mp.cpu_count() + 1, 1):

    t5 = time.time()
    scores_numba_parallel = loop.LocalOutlierProbability(
        data,
        n_neighbors=3,
        use_numba=True,
        progress_bar=True,
        parallel=True,
        # TODO: user warning, correct behavior and continue of too many threads are passed
        # TODO: user warning, ignore num_threads if numba and/or parallel is false
        # TODO: add num threads to readme
        num_threads=i
    ).fit().local_outlier_probabilities
    t6 = time.time()
    seconds_numba_parallel = t6 - t5
    print("\nComputation took " + str(seconds_numba_parallel) +
          " seconds with Numba JIT with parallel processing, using " + str(i) + " thread(s).")





