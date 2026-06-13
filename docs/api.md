# API Reference

## `LocalOutlierProbability`

```python
from PyNomaly import loop

clf = loop.LocalOutlierProbability(
    data=None,
    distance_matrix=None,
    neighbor_matrix=None,
    extent=3,
    n_neighbors=10,
    cluster_labels=None,
    use_numba=False,
    n_jobs=1,
    progress_bar=False,
)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | `np.ndarray` or `pd.DataFrame` | `None` | Input data as a 2D array with shape (n_observations, n_features). Mutually exclusive with `distance_matrix`. |
| `distance_matrix` | `np.ndarray` or `pd.DataFrame` | `None` | Pre-computed distance matrix with shape (n_observations, n_neighbors). Must be provided together with `neighbor_matrix`. Mutually exclusive with `data`. |
| `neighbor_matrix` | `np.ndarray` or `pd.DataFrame` | `None` | Pre-computed neighbor index matrix with shape (n_observations, n_neighbors). Required when `distance_matrix` is provided. |
| `extent` | `int` | `3` | Controls scoring sensitivity. Must be 1, 2, or 3. Corresponds to lambda times the standard deviation from the mean (1 = ~68%, 2 = ~95%, 3 = ~99.7%). |
| `n_neighbors` | `int` | `10` | Number of nearest neighbors to consider. Must be greater than 0 and less than the number of observations. Automatically adjusted with a warning if invalid. |
| `cluster_labels` | `list` | `None` | Cluster assignments for each observation. When provided, LoOP scores are computed within each cluster independently. |
| `use_numba` | `bool` | `False` | Enable Numba JIT compilation for distance computation. Falls back to pure Python with a warning if Numba is not installed. |
| `n_jobs` | `int` | `1` | Number of threads for parallel distance computation. Set to `-1` to use all CPU cores. Only effective when `use_numba=True`. |
| `progress_bar` | `bool` | `False` | Display a progress bar during distance computation. |

!!! note
    Either `data` or both `distance_matrix` and `neighbor_matrix` must be provided, but not both `data` and `distance_matrix`.

---

### Methods

#### `fit()`

```python
clf.fit() -> LocalOutlierProbability
```

Calculates the Local Outlier Probability for each observation in the input data.

**Returns**: `self` -- the fitted model instance. Access scores via `clf.local_outlier_probabilities`.

**Raises**:

- `ClusterSizeError` -- if any cluster contains fewer observations than `n_neighbors`.
- `MissingValuesError` -- if the input data contains `NaN` values.

---

#### `stream(x)`

```python
clf.stream(x) -> np.ndarray
```

Calculates the Local Outlier Probability for an individual observation against the fitted model. Must be called after `fit()`.

**Parameters**:

| Parameter | Type | Description |
|---|---|---|
| `x` | `np.ndarray` | A single observation to score. When using raw data mode, this should be a 1D array with the same number of features as the training data. When using distance matrix mode, this should be a scalar distance value. |

**Returns**: `np.ndarray` -- the Local Outlier Probability of the input observation (a value in [0, 1]).

!!! warning
    The stream approach does **not** support clustered data. If `cluster_labels` were provided during `fit()`, PyNomaly will automatically refit using a single cluster and issue a `UserWarning`.

---

### Attributes

Attributes available after calling `fit()`:

| Attribute | Type | Description |
|---|---|---|
| `local_outlier_probabilities` | `np.ndarray` | Array of LoOP scores for each observation, with values in [0, 1]. |
| `prob_distances` | `np.ndarray` | Probabilistic distances for each observation. |
| `prob_distances_ev` | `np.ndarray` | Expected values of probabilistic distances for each observation's neighborhood. |
| `norm_prob_local_outlier_factor` | `float` | Maximum normalized probabilistic local outlier factor across all observations. Used internally by `stream()`. |
| `is_fit` | `bool` | Whether the model has been fit. |
| `n_neighbors` | `int` | The number of neighbors used (may differ from the value passed to the constructor if it was adjusted). |

---

## Exceptions

All exceptions are importable from `PyNomaly.loop` or directly from `PyNomaly`:

```python
from PyNomaly import ClusterSizeError, MissingValuesError
# or
from PyNomaly.loop import PyNomalyError, ValidationError, ClusterSizeError, MissingValuesError
```

| Exception | Parent | Description |
|---|---|---|
| `PyNomalyError` | `Exception` | Base exception for all PyNomaly errors. |
| `ValidationError` | `PyNomalyError` | Base exception for input validation failures. |
| `ClusterSizeError` | `ValidationError` | Raised when a cluster has fewer observations than `n_neighbors`. |
| `MissingValuesError` | `ValidationError` | Raised when input data contains `NaN` values. |
