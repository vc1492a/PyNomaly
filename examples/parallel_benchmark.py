"""
Thorough benchmark of PyNomaly's parallel processing capabilities.

Tests across multiple dimensions:
  - Data sizes (total observations)
  - Number of clusters (parallelism granularity)
  - Execution modes: vectorized, multiprocessing (n_jobs), Numba sequential, Numba parallel
  - Number of workers (1, 2, 4, all cores)

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


def time_fit(data, cluster_labels, n_neighbors, use_numba, n_jobs, warmup=False):
    """Time a single fit call. Returns elapsed seconds."""
    clf = loop.LocalOutlierProbability(
        data, n_neighbors=n_neighbors,
        cluster_labels=cluster_labels,
        use_numba=use_numba,
        n_jobs=n_jobs
    )
    t0 = time.perf_counter()
    clf.fit()
    elapsed = time.perf_counter() - t0
    return elapsed


def run_benchmark(data, labels, n_neighbors, label, configs):
    """Run a set of configurations and print results."""
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
    results = []

    for mode_name, use_numba, n_jobs in configs:
        if use_numba and not HAS_NUMBA:
            continue

        # Warmup for Numba JIT compilation
        if use_numba:
            small_data, small_labels = generate_clustered_data(50, n_clusters, seed=99)
            time_fit(small_data, small_labels, min(n_neighbors, 10), use_numba, n_jobs, warmup=True)

        times = []
        for _ in range(N_REPEATS):
            t = time_fit(data, labels, n_neighbors, use_numba, n_jobs)
            times.append(t)

        best = min(times)
        mean = sum(times) / len(times)

        if baseline_best is None:
            baseline_best = best

        speedup = baseline_best / best if best > 0 else float('inf')
        results.append((mode_name, best, mean, speedup))
        print(f"  {mode_name:<35} {best:>7.3f}s {mean:>7.3f}s {speedup:>7.2f}x")

    return results


def main():
    cpu_count = os.cpu_count() or 1
    print(f"PyNomaly Parallel Processing Benchmark")
    print(f"CPU cores: {cpu_count}")
    print(f"Numba available: {HAS_NUMBA}")
    print(f"Repeats per config: {N_REPEATS}")

    # Scenario 1: Small data, many clusters — tests multiprocessing dispatch overhead
    data, labels = generate_clustered_data(500, 8)
    run_benchmark(data, labels, N_NEIGHBORS,
        "Scenario 1: Small clusters (500 pts x 8 clusters = 4,000 total)",
        [
            ("Vectorized (baseline)",       False, 1),
            ("Multiprocessing n_jobs=2",     False, 2),
            ("Multiprocessing n_jobs=4",     False, 4),
            ("Multiprocessing n_jobs=-1",    False, -1),
            ("Numba sequential",             True,  1),
            ("Numba parallel (prange)",      True,  -1),
        ]
    )

    # Scenario 2: Medium data, moderate clusters — practical use case
    data, labels = generate_clustered_data(2000, 4)
    run_benchmark(data, labels, N_NEIGHBORS,
        "Scenario 2: Medium clusters (2,000 pts x 4 clusters = 8,000 total)",
        [
            ("Vectorized (baseline)",       False, 1),
            ("Multiprocessing n_jobs=2",     False, 2),
            ("Multiprocessing n_jobs=4",     False, 4),
            ("Multiprocessing n_jobs=-1",    False, -1),
            ("Numba sequential",             True,  1),
            ("Numba parallel (prange)",      True,  -1),
        ]
    )

    # Scenario 3: Large data, many clusters — where parallelism should help most
    data, labels = generate_clustered_data(2000, 8)
    run_benchmark(data, labels, N_NEIGHBORS,
        "Scenario 3: Large + many clusters (2,000 pts x 8 clusters = 16,000 total)",
        [
            ("Vectorized (baseline)",       False, 1),
            ("Multiprocessing n_jobs=2",     False, 2),
            ("Multiprocessing n_jobs=4",     False, 4),
            ("Multiprocessing n_jobs=8",     False, 8),
            ("Multiprocessing n_jobs=-1",    False, -1),
            ("Numba sequential",             True,  1),
            ("Numba parallel (prange)",      True,  -1),
        ]
    )

    # Scenario 4: Scaling test — fixed cluster size, increasing cluster count
    print(f"\n{'='*70}")
    print(f"  Scenario 4: Scaling test — 1,000 pts/cluster, increasing clusters")
    print(f"{'='*70}")
    print(f"  {'Clusters':>8} {'Total':>8} {'Seq':>8} {'Par':>8} {'Speedup':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for n_clusters in [2, 4, 8, 16]:
        data, labels = generate_clustered_data(1000, n_clusters)

        seq_times = []
        par_times = []
        for _ in range(N_REPEATS):
            seq_times.append(time_fit(data, labels, N_NEIGHBORS, False, 1))
            par_times.append(time_fit(data, labels, N_NEIGHBORS, False, -1))

        seq_best = min(seq_times)
        par_best = min(par_times)
        speedup = seq_best / par_best if par_best > 0 else float('inf')
        total = n_clusters * 1000

        print(f"  {n_clusters:>8} {total:>7,} {seq_best:>7.3f}s {par_best:>7.3f}s {speedup:>7.2f}x")

    # Scenario 5: Single large cluster — no parallelism benefit expected
    data, labels = generate_clustered_data(10000, 1)
    run_benchmark(data, labels, N_NEIGHBORS,
        "Scenario 5: Single cluster (10,000 pts) — no multi-cluster parallelism",
        [
            ("Vectorized (baseline)",       False, 1),
            ("Multiprocessing n_jobs=-1",    False, -1),
            ("Numba sequential",             True,  1),
            ("Numba parallel (prange)",      True,  -1),
        ]
    )

    print(f"\n{'='*70}")
    print("  Benchmark complete.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
