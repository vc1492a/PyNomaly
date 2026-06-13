# How It Works

This page explains the LoOP (Local Outlier Probabilities) algorithm as implemented in PyNomaly. For full mathematical detail, see the [original paper](http://www.dbs.ifi.lmu.de/Publikationen/Papers/LoOP1649.pdf) by Kriegel, Kröger, Schubert, and Zimek (2009).

## Algorithm Pipeline

LoOP scores are computed through a pipeline of steps that transform raw distances into a probability in the range [0, 1].

### 1. Nearest Neighbor Distances

For each observation, the distances to its *k* nearest neighbors are computed using Euclidean distance. If cluster labels are provided, neighbors are found only within the same cluster, so that each observation is scored relative to its own local region of data.

When a custom distance matrix is provided, this step is skipped and the user-supplied distances are used directly.

### 2. Standard Distance

The **standard distance** of an observation is the root mean square of its *k* nearest neighbor distances:

```
standard_distance(o) = sqrt( (1/k) * sum(dist(o, neighbor_i)^2) )
```

This captures, on average, how far away an observation's neighbors are. Points in dense regions will have small standard distances; points in sparse regions will have large ones.

### 3. Probabilistic Distance

The **probabilistic distance** scales the standard distance by the `extent` parameter (lambda):

```
pdist(o) = extent * standard_distance(o)
```

The `extent` parameter corresponds to the statistical notion of an outlier deviating more than lambda standard deviations from the mean:

| `extent` | Statistical meaning | Approximate coverage |
|---|---|---|
| 1 | 1 standard deviation | ~68% |
| 2 | 2 standard deviations | ~95% |
| 3 | 3 standard deviations | ~99.7% |

A higher extent makes scoring more conservative (fewer points flagged as outliers).

### 4. Probabilistic Local Outlier Factor (PLOF)

The **PLOF** compares an observation's probabilistic distance to the expected (mean) probabilistic distance of its neighbors:

```
PLOF(o) = pdist(o) / E[pdist(neighbors(o))] - 1
```

- A PLOF near **zero** means the observation's local density is similar to its neighbors.
- A **large positive** PLOF means the observation sits in a sparser region relative to its neighbors, suggesting it may be an outlier.

### 5. Normalized PLOF (nPLOF)

The **nPLOF** is a normalization constant that accounts for the overall variability of PLOF values within a cluster:

```
nPLOF = extent * sqrt( E[PLOF^2] )
```

where the expectation is taken over all observations in the same cluster. This ensures that the final LoOP scores are comparable across clusters with different density characteristics.

### 6. Local Outlier Probability (LoOP)

The final **LoOP score** applies the Gaussian error function to produce a value in [0, 1]:

```
LoOP(o) = max(0, erf( PLOF(o) / (nPLOF * sqrt(2)) ))
```

The error function maps the normalized PLOF ratio onto a probability scale:

- **LoOP = 0**: The observation is consistent with its neighborhood density.
- **LoOP close to 1**: The observation is very likely an outlier relative to its local neighborhood.

Because scores are true probabilities, practitioners can apply thresholds according to their application requirements without needing to interpret arbitrary score magnitudes.

## Clustering

When `cluster_labels` are provided, each of the above steps is computed *within* each cluster independently. This is important when the data contains regions of naturally varying density -- without clustering, points in a globally sparse but locally dense region might be incorrectly flagged as outliers.

## Streaming

PyNomaly also supports a streaming mode based on Hamlet et al.'s modifications to the original LoOP approach. In streaming mode:

1. The standard LoOP algorithm is first `fit()` on a training dataset.
2. Key statistics from the fit (expected probabilistic distances, normalized PLOF) are stored.
3. New observations are scored individually via `stream()` using these stored statistics, without refitting the entire model.

This trades some accuracy for computational efficiency, making it suitable for applications where data arrives incrementally. See the [User Guide](user-guide.md#streaming-data) for details.
