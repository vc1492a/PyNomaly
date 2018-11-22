# Authors: Valentino Constantinou <vc@valentino.io>
# License: Apache 2.0

from PyNomaly import loop

import numpy as np
from numpy.testing import assert_array_equal

import pandas as pd
import pytest

from sklearn.metrics import roc_auc_score

from sklearn.utils import check_random_state
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_warns

from sklearn.datasets import load_iris

import warnings

# load the iris dataset
# and randomly permute it
rng = check_random_state(0)
iris = load_iris()
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]


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

    with pytest.raises(SystemExit) as record:
        clf.fit()

    assert record.type == SystemExit


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

    with pytest.warns(UserWarning) as record:
        warnings.warn(
            "Number of neighbors specified larger than smallest cluster. Specify a number of neighbors smaller than the smallest cluster size (observations in smallest cluster minus one).",
            UserWarning)

    loop.LocalOutlierProbability(X, n_neighbors=50, cluster_labels=cluster_labels)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Number of neighbors specified larger than smallest cluster. Specify a number of neighbors smaller than the smallest cluster size (observations in smallest cluster minus one)."


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

    with pytest.raises(SystemExit) as record:
        clf.stream(X_test)

    assert record.type == SystemExit


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
        warnings.warn(
            "Stream approach does not support clustered data. Automatically refit using single cluster of points.",
            UserWarning)

    clf.stream(X_test)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Stream approach does not support clustered data. Automatically refit using single cluster of points."


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
