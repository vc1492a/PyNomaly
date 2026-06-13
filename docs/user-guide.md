# User Guide

## Choosing Parameters

### extent

The `extent` parameter controls the sensitivity of the scoring in practice. The parameter corresponds to the statistical notion of an outlier defined as an object deviating more than a given lambda (`extent`) times the standard deviation from the mean.

| `extent` | Meaning | Empirical rule |
|---|---|---|
| 1 | 1 standard deviation | ~68% |
| 2 | 2 standard deviations | ~95% |
| 3 | 3 standard deviations | ~99.7% |

A value of 2 implies outliers deviating more than 2 standard deviations from the mean. The appropriate parameter should be selected according to the level of sensitivity needed for the input data and application.

### n_neighbors

The `n_neighbors` parameter defines the number of neighbors to consider about each sample (neighborhood size) when determining its Local Outlier Probability.

The ideal number of neighbors is dependent on the input data. However, the notion of an outlier implies it would be considered as such regardless of the number of neighbors considered. Some approaches for selecting this parameter:

- Use several different neighborhood sizes and average the results for each observation. Those observations which rank highly across varying neighborhood sizes are likely outliers.
- Select a value proportional to the number of observations, such as an odd-valued integer close to `sqrt(n_observations)`.

## Speeding Things Up with Numba

For large datasets, Numba's just-in-time (JIT) compilation can significantly speed up distance computation. Enable it with `use_numba=True`:

```python
from PyNomaly import loop
m = loop.LocalOutlierProbability(data, extent=2, n_neighbors=20, use_numba=True).fit()
scores = m.local_outlier_probabilities
print(scores)
```

To go further, set `n_jobs=-1` to enable Numba's thread-level parallelism (`prange`), which distributes work across all available CPU cores:

```python
from PyNomaly import loop
m = loop.LocalOutlierProbability(
    data, extent=2, n_neighbors=20,
    use_numba=True, n_jobs=-1
).fit()
scores = m.local_outlier_probabilities
print(scores)
```

This provides **2-3x speedups** on multi-core machines (benchmarked on 8 cores). The parallelism works within each cluster's distance computation, so it benefits both single-cluster and multi-cluster data.

- Set `n_jobs=-1` to use all cores, or specify a positive integer for a fixed number of threads.
- The default `n_jobs=1` runs sequentially.
- `n_jobs` only takes effect when `use_numba=True`. If `n_jobs > 1` is set without Numba, PyNomaly will warn and fall back to the sequential vectorized path.
- Parallel processing is only applicable when raw input data is provided -- if a pre-existing distance matrix is provided, the distance computation step is skipped entirely.

Numba must be installed to use JIT compilation. PyNomaly has been tested with Numba versions 0.45.1 through 0.65.1.

## Progress Bars

You may choose to print progress bars _with or without_ the use of Numba by passing `progress_bar=True` to `LocalOutlierProbability()`:

```python
from PyNomaly import loop
m = loop.LocalOutlierProbability(data, use_numba=True, n_jobs=-1, progress_bar=True).fit()
```

Progress bars are supported in both sequential and Numba execution modes.

## Using Numpy Arrays

When using numpy, make sure to use 2-dimensional arrays in tabular format:

```python
import numpy as np
from PyNomaly import loop

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
```

The shape of the input array corresponds to rows (observations) and columns (features):

```python
print(data.shape)
# (6, 3)
```

## Specifying a Distance Matrix

PyNomaly provides the ability to specify a distance matrix so that any distance metric can be used (a neighbor index matrix must also be provided). This can be useful when wanting to use a distance other than the Euclidean.

!!! note
    In order to maintain alignment with the LoOP definition of closest neighbors, an additional neighbor is added when using [scikit-learn's NearestNeighbors](https://scikit-learn.org/1.5/modules/neighbors.html) since `NearestNeighbors` includes the point itself when calculating the closest neighbors (whereas the LoOP method does not include distances to point itself).

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

data = np.array([
    [43.3, 30.2, 90.2],
    [62.9, 58.3, 49.3],
    [55.2, 56.2, 134.2],
    [48.6, 80.3, 50.3],
    [67.1, 60.0, 55.9],
    [421.5, 90.3, 50.0]
])

n_neighbors = 3
neigh = NearestNeighbors(n_neighbors=n_neighbors+1, metric='hamming')
neigh.fit(data)
d, idx = neigh.kneighbors(data, return_distance=True)

# Remove self-distances
indices = np.delete(indices, 0, 1)
distances = np.delete(distances, 0, 1)

m = loop.LocalOutlierProbability(
    distance_matrix=d, neighbor_matrix=idx, n_neighbors=n_neighbors+1
).fit()
scores = m.local_outlier_probabilities
```

## Streaming Data

PyNomaly contains an implementation of Hamlet et al.'s modifications to the original LoOP approach, which may be used for applications involving streaming data or where rapid calculations may be necessary.

First, the standard LoOP algorithm is used on "training" data, with certain attributes of the fitted data stored from the original LoOP approach. Then, as new points are considered, these fitted attributes are called when calculating the score of the incoming streaming data due to the use of averages from the initial fit, such as the use of a global value for the expected value of the probabilistic distance.

Despite the potential for increased error when compared to the standard approach, it may be effective in streaming applications where refitting the standard approach over all points could be computationally expensive.

### Example

Using the Iris dataset, taking the first 120 observations as training data and the remaining 30 as a stream:

```python
iris = iris.sample(frac=1)  # shuffle data
iris_train = iris.iloc[:, 0:4].head(120)
iris_test = iris.iloc[:, 0:4].tail(30)
```

Fit the model on training data:

```python
m_train = loop.LocalOutlierProbability(iris_train, n_neighbors=10)
m_train.fit()
iris_train_scores = m_train.local_outlier_probabilities
```

Score streaming observations individually:

```python
iris_test_scores = []
for index, row in iris_test.iterrows():
    array = np.array([
        row['Sepal.Length'], row['Sepal.Width'],
        row['Petal.Length'], row['Petal.Width']
    ])
    iris_test_scores.append(m_train.stream(array))
iris_test_scores = np.array(iris_test_scores)
```

### Notes

- When calculating the LoOP score of incoming data, the original fitted scores are **not** updated.
- In some applications, it may be beneficial to refit the data periodically.
- The stream functionality assumes that either data or a distance matrix (or value) will be used across both fitting and streaming, with no changes in specification between steps.
- The stream approach does **not** support clustered data. If cluster labels were used during `fit()`, PyNomaly will automatically refit using a single cluster of points when `stream()` is called.

## Exceptions and Error Handling

PyNomaly provides custom exceptions that can be caught and handled in your application code. All exceptions inherit from `PyNomalyError`:

| Exception | Raised when |
|---|---|
| `PyNomalyError` | Base exception for all PyNomaly errors. |
| `ValidationError` | Base class for input validation errors. |
| `ClusterSizeError` | A cluster contains fewer observations than `n_neighbors`. |
| `MissingValuesError` | Input data contains `NaN` values. |

These exceptions are exported from the package and can be imported directly:

```python
from PyNomaly import loop
from PyNomaly.loop import ClusterSizeError, MissingValuesError

try:
    m = loop.LocalOutlierProbability(
        data, n_neighbors=50, cluster_labels=labels
    ).fit()
except ClusterSizeError:
    print("Reduce n_neighbors or use larger clusters.")
except MissingValuesError:
    print("Clean NaN values from your data before fitting.")
```

PyNomaly also issues `UserWarning` in non-fatal situations such as:

- `n_neighbors` being zero or exceeding the number of observations (automatically adjusted)
- `extent` not being 1, 2, or 3
- Numba not being available when `use_numba=True` is set
- `n_jobs > 1` without `use_numba=True` (falls back to sequential processing)
