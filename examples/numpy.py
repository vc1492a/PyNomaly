from PyNomaly import loop
import numpy as np


data = np.array([
    [43.3, 30.2, 90.2],
    [62.9, 58.3, 49.3],
    [55.2, 56.2, 134.2],
    [48.6, 80.3, 50.3],
    [67.1, 60.0, 55.9],
    [421.5, 90.3, 50.0]
])


scores = loop.LocalOutlierProbability(data, n_neighbors=3).fit().local_outlier_probabilities
print(scores)
