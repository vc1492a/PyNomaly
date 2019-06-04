# Authors: Valentino Constantinou <vc@valentino.io>
# License: Apache 2.0

from PyNomaly import loop

import numpy as np
from numpy.testing import assert_array_equal
import os
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_warns
import time

# load the iris dataset
# and randomly permute it
rng = check_random_state(0)
iris = load_iris()
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]


def test_loop_numba():
    # disable numba and get a pure Python speed
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    r1 = _test_loop_numba()

    # re-enable, run the first time (compilation)
    os.environ["NUMBA_DISABLE_JIT"] = "0"
    _test_loop_numba()

    # now run the second time once it's been compiled and check the difference
    r2 = _test_loop_numba()
    perc_change = (r2 - r1) / r1

    # assert at least a 20% speed improvement is achieved
    assert perc_change <= -0.2


def test_loop():
    # Toy sample (the last two samples are outliers):
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 2], [1, 2], [2, 1], [5, 3],
                  [-4, 2]])

    # Test LocalOutlierProbability:
    clf = loop.LocalOutlierProbability(X, n_neighbors=5)
    score = clf.fit().local_outlier_probabilities
    share_outlier = 2. / 8.
    predictions = [-1 if s > share_outlier else 1 for s in score]
    assert_array_equal(predictions, 6 * [1] + 2 * [-1])

    # Assert smallest outlier score is greater than largest inlier score:
    assert_greater(np.min(score[-2:]), np.max(score[:-2]))

    # Test the DataFrame functionality
    X_df = pd.DataFrame(X)

    # Test LocalOutlierProbability:
    clf = loop.LocalOutlierProbability(X_df, n_neighbors=5)
    score = clf.fit().local_outlier_probabilities
    share_outlier = 2. / 8.
    predictions = [-1 if s > share_outlier else 1 for s in score]
    assert_array_equal(predictions, 6 * [1] + 2 * [-1])

    # Assert smallest outlier score is greater than largest inlier score:
    assert_greater(np.min(score[-2:]), np.max(score[:-2]))


def test_loop_performance():
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)

    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
    X_test = np.r_[X, X_outliers]
    X_labels = np.r_[
        np.repeat(1, X.shape[0]), np.repeat(-1, X_outliers.shape[0])]

    # fit the model
    clf = loop.LocalOutlierProbability(X_test, n_neighbors=X_test.shape[0] - 1)

    # predict scores (the lower, the more normal)
    score = clf.fit().local_outlier_probabilities
    share_outlier = X_outliers.shape[0] / X_test.shape[0]
    X_pred = [-1 if s > share_outlier else 1 for s in score]

    # check that roc_auc is good
    assert_greater(roc_auc_score(X_pred, X_labels), .985)


def test_input_nodata():
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)

    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
    X_test = np.r_[X, X_outliers]

    with pytest.warns(UserWarning) as record:
        # attempt to fit loop without data or a distance matrix
        loop.LocalOutlierProbability(n_neighbors=X_test.shape[0] - 1)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Data or a distance matrix must be provided."


def test_bad_input_argument():
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)

    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
    X_test = np.r_[X, X_outliers]

    with pytest.warns(UserWarning) as record:
        # attempt to fit loop with a string input for n_neighbors
        loop.LocalOutlierProbability(X, n_neighbors=str(X_test.shape[0] - 1))

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Argument 'n_neighbors' is not of type (<class 'int'>, " \
                     "<class 'numpy.integer'>)."


def test_neighbor_zero():
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)

    clf = loop.LocalOutlierProbability(X, n_neighbors=0)

    with pytest.warns(UserWarning) as record:
        # attempt to fit loop with a 0 neighbor count
        clf.fit()

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "n_neighbors must be greater than 0. Fit with 10 instead."


def test_input_distonly():
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)

    # generate distance and neighbor indices
    neigh = NearestNeighbors(metric='euclidean')
    neigh.fit(X)
    d, idx = neigh.kneighbors(X, n_neighbors=10, return_distance=True)

    with pytest.warns(UserWarning) as record:
        # attempt to fit loop with a distance matrix and no neighbor matrix
        loop.LocalOutlierProbability(distance_matrix=d)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "A neighbor index matrix and distance matrix must both " \
                     "be provided when not using raw input data."


def test_input_neighboronly():
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)

    # generate distance and neighbor indices
    neigh = NearestNeighbors(metric='euclidean')
    neigh.fit(X)
    d, idx = neigh.kneighbors(X, n_neighbors=10, return_distance=True)

    with pytest.warns(UserWarning) as record:
        # attempt to fit loop with a neighbor matrix and no distance matrix
        loop.LocalOutlierProbability(neighbor_matrix=idx)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Data or a distance matrix must be provided."


def test_input_too_many():
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)

    # generate distance and neighbor indices
    neigh = NearestNeighbors(metric='euclidean')
    neigh.fit(X)
    d, idx = neigh.kneighbors(X, n_neighbors=10, return_distance=True)

    with pytest.warns(UserWarning) as record:
        # attempt to fit loop with data and a distance matrix
        loop.LocalOutlierProbability(X, distance_matrix=d, neighbor_matrix=idx)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Only one of the following may be provided: data or a " \
                     "distance matrix (not both)."


def test_distance_neighbor_shape_mismatch():
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)

    # generate distance and neighbor indices
    neigh = NearestNeighbors(metric='euclidean')
    neigh.fit(X)
    d, idx = neigh.kneighbors(X, n_neighbors=10, return_distance=True)

    # generate distance and neighbor indices of a different shape
    neigh_2 = NearestNeighbors(metric='euclidean')
    neigh_2.fit(X)
    d_2, idx_2 = neigh.kneighbors(X, n_neighbors=5, return_distance=True)

    with pytest.warns(UserWarning) as record:
        # attempt to fit loop with a mismatch in shapes
        loop.LocalOutlierProbability(
            distance_matrix=d,
            neighbor_matrix=idx_2,
            n_neighbors=5)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "The shape of the distance and neighbor " \
                     "index matrices must match."


def test_input_neighbor_mismatch():
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)

    # generate distance and neighbor indices
    neigh = NearestNeighbors(metric='euclidean')
    neigh.fit(X)
    d, idx = neigh.kneighbors(X, n_neighbors=5, return_distance=True)

    with pytest.warns(UserWarning) as record:
        # attempt to fit loop with a neighbor size mismatch
        loop.LocalOutlierProbability(distance_matrix=d,
                                     neighbor_matrix=idx,
                                     n_neighbors=10)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "The shape of the distance or " \
                     "neighbor index matrix does not " \
                     "match the number of neighbors " \
                     "specified."


def test_loop_dist_matrix():
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)

    # generate distance and neighbor indices
    neigh = NearestNeighbors(metric='euclidean')
    neigh.fit(X)
    d, idx = neigh.kneighbors(X, n_neighbors=10, return_distance=True)

    # fit loop using data and distance matrix
    clf1 = loop.LocalOutlierProbability(X)
    clf2 = loop.LocalOutlierProbability(distance_matrix=d, neighbor_matrix=idx)
    scores1 = clf1.fit().local_outlier_probabilities
    scores2 = clf2.fit().local_outlier_probabilities

    # compare the agreement between the results
    assert_almost_equal(scores1, scores2, decimal=1)


def test_lambda_values():
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)

    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
    X_test = np.r_[X, X_outliers]

    # Fit the model with different extent (lambda) values
    clf1 = loop.LocalOutlierProbability(X_test, extent=1)
    clf2 = loop.LocalOutlierProbability(X_test, extent=2)
    clf3 = loop.LocalOutlierProbability(X_test, extent=3)

    # predict scores (the lower, the more normal)
    score1 = clf1.fit().local_outlier_probabilities
    score2 = clf2.fit().local_outlier_probabilities
    score3 = clf3.fit().local_outlier_probabilities

    # Get the mean of all the scores
    score_mean1 = np.mean(score1)
    score_mean2 = np.mean(score2)
    score_mean3 = np.mean(score3)

    # check that expected the means align with expectation
    assert_greater(score_mean1, score_mean2)
    assert_greater(score_mean2, score_mean3)


def test_parameters():
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)

    # fit the model
    clf = loop.LocalOutlierProbability(X).fit()

    # check that the model has attributes post fit
    assert_true(hasattr(clf, 'n_neighbors') and
                clf.n_neighbors is not None)
    assert_true(hasattr(clf, 'extent') and
                clf.extent is not None)
    assert_true(hasattr(clf, 'cluster_labels') and
                clf._cluster_labels() is not None)
    assert_true(hasattr(clf, 'prob_distances') and
                clf.prob_distances is not None)
    assert_true(hasattr(clf, 'prob_distances_ev') and
                clf.prob_distances_ev is not None)
    assert_true(hasattr(clf, 'norm_prob_local_outlier_factor') and
                clf.norm_prob_local_outlier_factor is not None)
    assert_true(hasattr(clf, 'local_outlier_probabilities') and
                clf.local_outlier_probabilities is not None)


def test_n_neighbors():
    X = iris.data
    clf = loop.LocalOutlierProbability(X, n_neighbors=500).fit()
    assert_equal(clf.n_neighbors, X.shape[0] - 1)

    clf = loop.LocalOutlierProbability(X, n_neighbors=500)
    assert_warns(UserWarning, clf.fit)
    assert_equal(clf.n_neighbors, X.shape[0] - 1)


def test_extent():
    X = np.array([[1, 1], [1, 0]])
    clf = loop.LocalOutlierProbability(X, n_neighbors=2, extent=4)
    assert_warns(UserWarning, clf.fit)


def test_data_format():
    X = [1.3, 1.1, 0.9, 1.4, 1.5, 3.2]
    clf = loop.LocalOutlierProbability(X, n_neighbors=3)
    assert_warns(UserWarning, clf.fit)


def test_missing_values():
    X = np.array([1.3, 1.1, 0.9, 1.4, 1.5, np.nan, 3.2])
    clf = loop.LocalOutlierProbability(X, n_neighbors=3)

    with pytest.raises(SystemExit) as record_a, pytest.warns(
            UserWarning) as record_b:
        clf.fit()

    assert record_a.type == SystemExit

    # check that only one warning was raised
    assert len(record_b) == 1
    # check that the message matches
    assert record_b[0].message.args[
               0] == "Method does not support missing values in input data."


def test_small_cluster_size():
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)

    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
    X = np.r_[X, X_outliers]
    # Generate cluster labels
    a = [0] * 120
    b = [1] * 18
    cluster_labels = a + b

    clf = loop.LocalOutlierProbability(
        X,
        n_neighbors=50,
        cluster_labels=cluster_labels)

    with pytest.raises(SystemExit) as record_a, pytest.warns(
            UserWarning) as record_b:
        clf.fit()

    assert record_a.type == SystemExit

    # check that only one warning was raised
    assert len(record_b) == 1
    # check that the message matches
    assert record_b[0].message.args[
               0] == "Number of neighbors specified larger than smallest " \
                     "cluster. Specify a number of neighbors smaller than " \
                     "the smallest cluster size (observations in smallest " \
                     "cluster minus one)."


def test_stream_fit():
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)

    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
    X = np.r_[X, X_outliers]

    # Fit the model
    X_train = X[0:138]
    X_test = X[139]
    clf = loop.LocalOutlierProbability(X_train)

    with pytest.warns(UserWarning) as record:
        clf.stream(X_test)

    # check that the message matches
    messages = [i.message.args[0] for i in record]
    assert "Must fit on historical data by calling fit() prior to " \
           "calling stream(x)." in messages


def test_stream_distance():
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)

    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
    X = np.r_[X, X_outliers]

    X_train = X[0:100]
    X_test = X[100:140]

    # generate distance and neighbor indices
    neigh = NearestNeighbors(metric='euclidean')
    neigh.fit(X_train)
    d, idx = neigh.kneighbors(X_train, n_neighbors=10, return_distance=True)

    # Fit the models in standard and distance matrix form
    m = loop.LocalOutlierProbability(X_train).fit()
    m_dist = loop.LocalOutlierProbability(distance_matrix=d,
                                          neighbor_matrix=idx).fit()

    # Collect the scores
    X_test_scores = []
    for i in range(X_test.shape[0]):
        X_test_scores.append(m.stream(np.array(X_test[i])))
    X_test_scores = np.array(X_test_scores)

    X_test_dist_scores = []
    for i in range(X_test.shape[0]):
        dd, ii = neigh.kneighbors(np.array([X_test[i]]), return_distance=True)
        X_test_dist_scores.append(m_dist.stream(np.mean(dd)))
    X_test_dist_scores = np.array(X_test_dist_scores)

    # calculate the rmse and ensure score is below threshold
    rmse = np.sqrt(((X_test_scores - X_test_dist_scores) ** 2).mean(axis=None))
    assert_greater(0.075, rmse)


def test_stream_cluster():
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)

    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
    X = np.r_[X, X_outliers]

    # Generate cluster labels
    a = [0] * 120
    b = [1] * 18
    cluster_labels = a + b

    # Fit the model
    X_train = X[0:138]
    X_test = X[139]
    clf = loop.LocalOutlierProbability(X_train,
                                       cluster_labels=cluster_labels).fit()

    with pytest.warns(UserWarning) as record:
        clf.stream(X_test)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Stream approach does not support clustered data. " \
                     "Automatically refit using single cluster of points."


def test_stream_performance():
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)

    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
    X = np.r_[X, X_outliers]

    X_train = X[0:100]
    X_test = X[100:140]

    # Fit the models in standard and stream form
    m = loop.LocalOutlierProbability(X).fit()
    scores_noclust = m.local_outlier_probabilities

    m_train = loop.LocalOutlierProbability(X_train)
    m_train.fit()
    X_train_scores = m_train.local_outlier_probabilities

    X_test_scores = []
    for idx in range(X_test.shape[0]):
        X_test_scores.append(m_train.stream(X_test[idx]))
    X_test_scores = np.array(X_test_scores)

    stream_scores = np.hstack((X_train_scores, X_test_scores))

    # calculate the rmse and ensure score is below threshold
    rmse = np.sqrt(((scores_noclust - stream_scores) ** 2).mean(axis=None))
    assert_greater(0.35, rmse)


def _test_loop_numba():
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(1000, 2)

    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(200, 2))
    X_test = np.r_[X, X_outliers]

    # start timer
    t1 = time.time()

    # fit the model
    clf = loop.LocalOutlierProbability(X_test, n_neighbors=X_test.shape[0] - 1)

    # predict scores (the lower, the more normal)
    clf.fit().local_outlier_probabilities

    # end timer
    t2 = time.time()

    # get the time
    spread = t2 - t1

    return spread

# TODO: wheels and setup.py if wheel fails

# TODO: create some fixtures and classes for the unit tests
# TODO: pytest fixtures and type hints to clean up unit testing

# TODO: pynomaly speed comparison repo. use travis to run analysis code with various versions
