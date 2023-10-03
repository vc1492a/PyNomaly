import numpy as np
import os
from PyNomaly import loop
import time

# generate a large set of data
data = np.ones(shape=(10000, 4))

# first time the process without Numba
# use the progress bar to track progress

# t1 = time.time()
# scores_numpy = loop.LocalOutlierProbability(
#    data,
#    n_neighbors=3,
#    use_numba=False,
#    progress_bar=True
# ).fit().local_outlier_probabilities
# t2 = time.time()
# seconds_no_numba = t2 - t1
# print("\nComputation took " + str(seconds_no_numba) + " seconds without Numba JIT.")

# t3 = time.time()
# scores_numba = loop.LocalOutlierProbability(
#     data,
#     n_neighbors=3,
#     use_numba=True,
#     progress_bar=False,
#     parallel=False
# ).fit().local_outlier_probabilities
# t4 = time.time()
# seconds_numba = t4 - t3
# print("\nComputation took " + str(seconds_numba) + " seconds with Numba JIT.")

t5 = time.time()
scores_numba_parallel = loop.LocalOutlierProbability(
    data,
    n_neighbors=3,
    use_numba=True,
    progress_bar=True,
    parallel=True,
    # TODO: fix sigenv kill on anything 3 or more cores
    # TODO: user warning, correct behavior and continue of too many threads are passed
    # TODO: user warning, ignore num_threads if numba and/or parallel is false
    # TODO: add num threads to readme
    # num_threads=os.cpu_count()
    # TODO: num_threads or num_cores?
    # TODO: flush stdout
    # TODO: changelog, readme
    # TODO: tests?
    num_threads=3 # TODO: if numba true and parallel false, set to 1
).fit().local_outlier_probabilities
t6 = time.time()
seconds_numba_parallel = t6 - t5
print("\nComputation took " + str(seconds_numba_parallel) + " seconds with Numba JIT with parallel processing.")
