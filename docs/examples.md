# Examples

## Iris Data Example

We'll be using the well-known Iris dataset to show LoOP's capabilities. You'll need:

- matplotlib 2.0.0 or greater
- PyDataset 0.2.0 or greater
- scikit-learn 0.18.1 or greater

First, let's import the packages and libraries we will need.

```python
from PyNomaly import loop
import pandas as pd
from pydataset import data
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```

Create two sets of Iris data for scoring; one with clustering and the other without.

```python
iris = pd.DataFrame(data('iris').drop(columns=['Species']))
```

Cluster the data using DBSCAN and generate two sets of scores. In both cases, we will use the default values for both `extent` (3) and `n_neighbors` (10).

```python
db = DBSCAN(eps=0.9, min_samples=10).fit(iris)
m = loop.LocalOutlierProbability(iris).fit()
scores_noclust = m.local_outlier_probabilities
m_clust = loop.LocalOutlierProbability(iris, cluster_labels=list(db.labels_)).fit()
scores_clust = m_clust.local_outlier_probabilities
```

Organize the data into two separate Pandas DataFrames.

```python
iris_clust = pd.DataFrame(iris.copy())
iris_clust['scores'] = scores_clust
iris_clust['labels'] = db.labels_
iris['scores'] = scores_noclust
```

Visualize the scores provided by LoOP in both cases (with and without clustering).

```python
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(iris['Sepal.Width'], iris['Petal.Width'], iris['Sepal.Length'],
c=iris['scores'], cmap='seismic', s=50)
ax.set_xlabel('Sepal.Width')
ax.set_ylabel('Petal.Width')
ax.set_zlabel('Sepal.Length')
plt.show()
```

**LoOP Scores without Clustering**

![LoOP Scores without Clustering](https://github.com/vc1492a/PyNomaly/blob/main/images/scores.png?raw=true)

**LoOP Scores with Clustering**

![LoOP Scores with Clustering](https://github.com/vc1492a/PyNomaly/blob/main/images/scores_clust.png?raw=true)

**DBSCAN Cluster Assignments**

![DBSCAN Cluster Assignments](https://github.com/vc1492a/PyNomaly/blob/main/images/cluster_assignments.png?raw=true)

Note the differences between using `LocalOutlierProbability` with and without clustering. In the example without clustering, samples are scored according to the distribution of the entire data set. In the example with clustering, each sample is scored according to the distribution of each cluster. Which approach is suitable depends on the use case.

!!! note
    Data was not normalized in this example, but it's probably a good idea to do so in practice.

## Distance Metric Comparison

The example in `examples/iris_dist_grid.py` demonstrates scoring with several different distance metrics by providing custom distance and neighbor matrices:

**LoOP Scores by Distance Metric**

![LoOP Scores by Distance Metric](https://github.com/vc1492a/PyNomaly/blob/main/images/scores_by_distance_metric.png?raw=true)

## Streaming Data

The example in `examples/stream.py` demonstrates the streaming approach using the Iris dataset. See the [User Guide](user-guide.md#streaming-data) for a complete walkthrough.

**LoOP Scores using Stream Approach with n=10**

![LoOP Scores using Stream Approach](https://github.com/vc1492a/PyNomaly/blob/main/images/scores_stream.png?raw=true)

## Additional Examples

The following example scripts are available in the [`examples/`](https://github.com/vc1492a/PyNomaly/tree/main/examples) directory of the repository:

| Script | Description |
|---|---|
| `iris.py` | Iris dataset with and without clustering |
| `iris_dist_grid.py` | Comparing multiple distance metrics |
| `stream.py` | Streaming data approach |
| `numba_speed_diff.py` | Numba vs. pure Python speed comparison |
| `parallel_benchmark.py` | Parallel processing benchmarks with `n_jobs` |
| `multiple_gaussian_2d.py` | 2D Gaussian mixture data |
| `1d_time_series.py` | 1-dimensional time series anomaly detection |
| `cluster_labels_flipped.py` | Flipped cluster labels consistency check |
