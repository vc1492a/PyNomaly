import numpy as np
from PyNomaly import loop
import time

# generate a large set of data
data = np.ones(shape=(10000, 4))

# first time the process without Numba
# use the progress bar to track progress

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
