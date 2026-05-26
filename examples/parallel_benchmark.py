"""
Benchmark of PyNomaly's Numba parallel processing (prange).

Tests across multiple dimensions:
  - Data sizes (total observations)
  - Number of clusters
  - Sequential vs parallel Numba (n_jobs=1 vs n_jobs=-1)
  - Comparison against the default vectorized NumPy path

Run:
    python examples/parallel_benchmark.py
"""

import numpy as np
import os
import time
from PyNomaly import loop

N_FEATURES = 4
N_NEIGHBORS = 10
N_REPEATS = 3

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def generate_clustered_data(n_per_cluster, n_clusters, n_features=N_FEATURES, seed=42):
    rng = np.random.RandomState(seed)
    clusters = []
    labels = []
    for i in range(n_clusters):
        center = rng.uniform(-20, 20, size=n_features)
        points = rng.randn(n_per_cluster, n_features) + center
        clusters.append(points)
        labels.extend([i] * n_per_cluster)
    data = np.vstack(clusters)
    return data, labels


def time_fit(data, cluster_labels, n_neighbors, use_numba, n_jobs):
    clf = loop.LocalOutlierProbability(
        data, n_neighbors=n_neighbors,
        cluster_labels=cluster_labels,
        use_numba=use_numba,
        n_jobs=n_jobs
    )
    t0 = time.perf_counter()
    clf.fit()
    return time.perf_counter() - t0


def run_benchmark(data, labels, n_neighbors, label, configs):
    n_total = len(data)
    n_clusters = len(set(labels))

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  {n_total:,} observations, {n_clusters} clusters, "
          f"{n_total // n_clusters:,} per cluster, {data.shape[1]} features")
    print(f"{'='*70}")
    print(f"  {'Mode':<35} {'Best':>8} {'Mean':>8} {'Speedup':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")

    baseline_best = None

    for mode_name, use_numba, n_jobs in configs:
        if use_numba and not HAS_NUMBA:
            continue

        if use_numba:
            small_data, small_labels = generate_clustered_data(50, n_clusters, seed=99)
            time_fit(small_data, small_labels, min(n_neighbors, 10), use_numba, n_jobs)

        times = []
        for _ in range(N_REPEATS):
            t = time_fit(data, labels, n_neighbors, use_numba, n_jobs)
            times.append(t)

        best = min(times)
        mean = sum(times) / len(times)

        if baseline_best is None:
            baseline_best = best

        speedup = baseline_best / best if best > 0 else float('inf')
        print(f"  {mode_name:<35} {best:>7.3f}s {mean:>7.3f}s {speedup:>7.2f}x")


def main():
    cpu_count = os.cpu_count() or 1
    print(f"PyNomaly Parallel Benchmark (Numba prange)")
    print(f"CPU cores: {cpu_count}")
    print(f"Numba available: {HAS_NUMBA}")
    print(f"Repeats per config: {N_REPEATS}")

    if not HAS_NUMBA:
        print("\nNumba is not installed. Install it with: pip install numba")
        print("Only vectorized (baseline) results will be shown.\n")

    configs = [
        ("Vectorized NumPy (baseline)",     False, 1),
        ("Numba sequential (n_jobs=1)",      True,  1),
        ("Numba parallel (n_jobs=-1)",       True,  -1),
    ]

    data, labels = generate_clustered_data(500, 8)
    run_benchmark(data, labels, N_NEIGHBORS,
        "Scenario 1: Small clusters (500 pts x 8 clusters = 4,000 total)",
        configs)

    data, labels = generate_clustered_data(2000, 4)
    run_benchmark(data, labels, N_NEIGHBORS,
        "Scenario 2: Medium clusters (2,000 pts x 4 clusters = 8,000 total)",
        configs)

    data, labels = generate_clustered_data(2000, 8)
    run_benchmark(data, labels, N_NEIGHBORS,
        "Scenario 3: Large (2,000 pts x 8 clusters = 16,000 total)",
        configs)

    # Single cluster scaling — shows prange benefit within a cluster
    print(f"\n{'='*70}")
    print(f"  Scenario 4: Single cluster scaling (Numba prange within one cluster)")
    print(f"{'='*70}")
    print(f"  {'Points':>8} {'Vectorized':>10} {'Numba Seq':>10} {'Numba Par':>10} {'Speedup':>8}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for n in [5000, 10000, 20000]:
        data, labels = generate_clustered_data(n, 1)
        vec_t = min(time_fit(data, labels, N_NEIGHBORS, False, 1) for _ in range(N_REPEATS))

        if HAS_NUMBA:
            small_data, small_labels = generate_clustered_data(50, 1, seed=99)
            time_fit(small_data, small_labels, 10, True, 1)
            time_fit(small_data, small_labels, 10, True, -1)

            nb_s = min(time_fit(data, labels, N_NEIGHBORS, True, 1) for _ in range(N_REPEATS))
            nb_p = min(time_fit(data, labels, N_NEIGHBORS, True, -1) for _ in range(N_REPEATS))
            speedup = vec_t / nb_p
            print(f"  {n:>7,} {vec_t:>9.2f}s {nb_s:>9.2f}s {nb_p:>9.2f}s {speedup:>7.2f}x")
        else:
            print(f"  {n:>7,} {vec_t:>9.2f}s {'N/A':>10} {'N/A':>10} {'N/A':>8}")

    print(f"\n{'='*70}")
    print("  Benchmark complete.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
