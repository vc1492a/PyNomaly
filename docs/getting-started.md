# Getting Started

## Dependencies

- Python 3.8 - 3.14
- numpy >= 1.16.3
- python-utils >= 2.3.0
- (optional) numba >= 0.45.1
- (optional) scipy >= 1.3.0

Numba just-in-time (JIT) compiles the function which calculates the Euclidean
distance between observations, providing a reduction in computation time
(significantly when a large number of observations are scored). Numba is not a
requirement and PyNomaly may still be used solely with numpy if desired.

When scipy is available, PyNomaly uses its optimized distance
computation (`scipy.spatial.distance.cdist`) and error function (`scipy.special.erf`)
implementations for additional performance gains.

## Installation

Install from the Python Package Index:

```shell
pip install PyNomaly
```

Or from conda-forge:

```shell
conda install conda-forge::pynomaly
```

## Quick Start

```python
from PyNomaly import LoOP
m = LoOP().fit(data)
scores = m.local_outlier_probabilities
print(scores)
```

where `data` is a NxM (N rows, M columns; 2-dimensional) set of data as either a Pandas DataFrame or Numpy array.

`LocalOutlierProbability` (also available as `LoOP`) sets the `extent` (an integer value of 1, 2, or 3) and `n_neighbors` (must be greater than 0) parameters with the default values of 3 and 10, respectively:

```python
from PyNomaly import LoOP
m = LoOP(extent=2, n_neighbors=20).fit(data)
scores = m.local_outlier_probabilities
print(scores)
```

## Using Cluster Labels

This implementation of LoOP includes an optional `cluster_labels` parameter. This is useful in cases where regions of varying density occur within the same set of data. When using `cluster_labels`, the Local Outlier Probability of a sample is calculated with respect to its cluster assignment.

```python
from PyNomaly import LoOP
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.6, min_samples=50).fit(data)
m = LoOP(extent=2, n_neighbors=20).fit(data, cluster_labels=list(db.labels_))
scores = m.local_outlier_probabilities
print(scores)
```

!!! note
    Unless your data is all the same scale, it may be a good idea to normalize your data with z-scores or another normalization scheme prior to using LoOP, especially when working with multiple dimensions of varying scale. Users must also appropriately handle missing values prior to using LoOP, as LoOP does not support Pandas DataFrames or Numpy arrays with missing values.
