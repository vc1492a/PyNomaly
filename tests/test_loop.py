# Authors: Valentino Constantinou <vc@valentino.io>
# License: Apache 2.0

from PyNomaly import loop

import logging
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# flag to enable or disable NUMBA
NUMBA = False

if NUMBA is False:
    logging.info("Numba is disabled. Coverage statistics are reflective of "
                 "testing native Python code. Consider also testing with numba"
                 " enabled.")
else:
    logging.warning(
        "Numba is enabled. Coverage statistics will be impacted (reduced) to"
        " due the just-in-time compilation of native Python code.")

# load the iris dataset
# and randomly permute it
rng = check_random_state(0)
iris = load_iris()
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]


# fixtures
@pytest.fixture()
def X_n8() -> np.ndarray:
    """
    Fixture that generates a small Numpy array with two anomalous values
    (last two observations).
    :return: a Numpy array.
    """
    # Toy sample (the last two samples are outliers):
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 2], [1, 2], [2, 1], [5, 3],
                  [-4, 2]])
    return X


@pytest.fixture()
def X_n120() -> np.ndarray:
    """
    Fixture that generates a Numpy array with 120 observations. Each
    observation contains two float values.
    :return: a Numpy array.
    """
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)
    return X


@pytest.fixture()
def X_n140_outliers(X_n120) -> np.ndarray:
    """
    Fixture that generates a Numpy array with 140 observations, where the
    first 120 observations are "normal" and the last 20 considered anomalous.
    :param X_n120: A pytest Fixture that generates the first 120 observations.
    :return: A Numpy array.
    """
    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
    X = np.r_[X_n120, X_outliers]
    return X


@pytest.fixture()
def X_n1000() -> np.ndarray:
    """
    Fixture that generates a Numpy array with 1000 observations.
    :return: A Numpy array.
    """
    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(1000, 2)
    return X


def test_loop(X_n8) -> None:
    """
    Tests the basic functionality and asserts that the anomalous observations
    are detected as anomalies. Tests the functionality using inputs
    as Numpy arrays and as Pandas dataframes.
    :param X_n8: A pytest Fixture that generates the 8 observations.
    :return: None
    """
    # Test LocalOutlierProbability:
    clf = loop.LocalOutlierProbability(X_n8, n_neighbors=5, use_numba=NUMBA)
    score = clf.fit().local_outlier_probabilities
    share_outlier = 2. / 8.
    predictions = [-1 if s > share_outlier else 1 for s in score]
    assert_array_equal(predictions, 6 * [1] + 2 * [-1])

    # Assert smallest outlier score is greater than largest inlier score:
    assert np.min(score[-2:]) > np.max(score[:-2])

    # Test the DataFrame functionality
    X_df = pd.DataFrame(X_n8)

    # Test LocalOutlierProbability:
    clf = loop.LocalOutlierProbability(X_df, n_neighbors=5, use_numba=NUMBA)
    score = clf.fit().local_outlier_probabilities
    share_outlier = 2. / 8.
    predictions = [-1 if s > share_outlier else 1 for s in score]
    assert_array_equal(predictions, 6 * [1] + 2 * [-1])

    # Assert smallest outlier score is greater than largest inlier score:
    assert np.min(score[-2:]) > np.max(score[:-2])


def test_loop_performance(X_n120) -> None:
    """
    Using a set of known anomalies (labels), tests the performance (using
    ROC / AUC score) of the software and ensures it is able to capture most
    anomalies under this basic scenario.
    :param X_n120: A pytest Fixture that generates the 120 observations.
    :return: None
    """
    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
    X_test = np.r_[X_n120, X_outliers]
    X_labels = np.r_[
        np.repeat(1, X_n120.shape[0]), np.repeat(-1, X_outliers.shape[0])]

    # fit the model
    clf = loop.LocalOutlierProbability(
        X_test,
        n_neighbors=X_test.shape[0] - 1,
        # test the progress bar
        progress_bar=True,
        use_numba=NUMBA
    )

    # predict scores (the lower, the more normal)
    score = clf.fit().local_outlier_probabilities
    share_outlier = X_outliers.shape[0] / X_test.shape[0]
    X_pred = [-1 if s > share_outlier else 1 for s in score]

    # check that roc_auc is good
    assert roc_auc_score(X_pred, X_labels) >= .98


def test_input_nodata(X_n140_outliers) -> None:
    """
    Test to ensure that the proper warning is issued if no data is
    provided.
    :param X_n140_outliers: A pytest Fixture that generates 140 observations.
    :return: None
    """
    with pytest.warns(UserWarning) as record:
        # attempt to fit loop without data or a distance matrix
        loop.LocalOutlierProbability(n_neighbors=X_n140_outliers.shape[0] - 1,
                                     use_numba=NUMBA)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Data or a distance matrix must be provided."


def test_input_incorrect_type(X_n140_outliers) -> None:
    """
    Test to ensure that the proper warning is issued if the type of an
    argument is the incorrect type.
    :param X_n140_outliers: A pytest Fixture that generates 140 observations.
    :return: None
    """
    with pytest.warns(UserWarning) as record:
        # attempt to fit loop with a string input for n_neighbors
        loop.LocalOutlierProbability(X_n140_outliers,
                                     n_neighbors=str(
                                         X_n140_outliers.shape[0] - 1),
                                     use_numba=NUMBA
                                     )

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Argument 'n_neighbors' is not of type (<class 'int'>, " \
                     "<class 'numpy.integer'>)."


def test_input_neighbor_zero(X_n120) -> None:
    """
    Test to ensure that the proper warning is issued if the neighbor size
    is specified as 0 (must be greater than 0).
    :param X_n120: A pytest Fixture that generates 120 observations.
    :return: None
    """
    clf = loop.LocalOutlierProbability(X_n120, n_neighbors=0, use_numba=NUMBA)

    with pytest.warns(UserWarning) as record:
        # attempt to fit loop with a 0 neighbor count
        clf.fit()

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "n_neighbors must be greater than 0. Fit with 10 instead."


def test_input_distonly(X_n120) -> None:
    """
    Test to ensure that the proper warning is issued if only a distance
    matrix is provided (without a neighbor matrix).
    :param X_n120: A pytest Fixture that generates 120 observations.
    :return: None
    """
    # generate distance and neighbor indices
    neigh = NearestNeighbors(metric='euclidean')
    neigh.fit(X_n120)
    d, idx = neigh.kneighbors(X_n120, n_neighbors=10, return_distance=True)

    with pytest.warns(UserWarning) as record:
        # attempt to fit loop with a distance matrix and no neighbor matrix
        loop.LocalOutlierProbability(distance_matrix=d, use_numba=NUMBA)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "A neighbor index matrix and distance matrix must both " \
                     "be provided when not using raw input data."


def test_input_neighboronly(X_n120) -> None:
    """
    Test to ensure that the proper warning is issued if only a neighbor
    matrix is provided (without a distance matrix).
    :param X_n120: A pytest Fixture that generates 120 observations.
    :return: None
    """
    # generate distance and neighbor indices
    neigh = NearestNeighbors(metric='euclidean')
    neigh.fit(X_n120)
    d, idx = neigh.kneighbors(X_n120, n_neighbors=10, return_distance=True)

    with pytest.warns(UserWarning) as record:
        # attempt to fit loop with a neighbor matrix and no distance matrix
        loop.LocalOutlierProbability(neighbor_matrix=idx, use_numba=NUMBA)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Data or a distance matrix must be provided."


def test_input_too_many(X_n120) -> None:
    """
    Test to ensure that the proper warning is issued if both a data matrix
    and a distance matrix are provided (can only be data matrix).
    :param X_n120: A pytest Fixture that generates 120 observations.
    :return: None
    """
    # generate distance and neighbor indices
    neigh = NearestNeighbors(metric='euclidean')
    neigh.fit(X_n120)
    d, idx = neigh.kneighbors(X_n120, n_neighbors=10, return_distance=True)

    with pytest.warns(UserWarning) as record:
        # attempt to fit loop with data and a distance matrix
        loop.LocalOutlierProbability(X_n120, distance_matrix=d,
                                     neighbor_matrix=idx, use_numba=NUMBA)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Only one of the following may be provided: data or a " \
                     "distance matrix (not both)."


def test_distance_neighbor_shape_mismatch(X_n120) -> None:
    """
    Test to ensure that the proper warning is issued if there is a mismatch
    between the shape of the provided distance and neighbor matrices.
    :param X_n120: A pytest Fixture that generates 120 observations.
    :return: None
    """
    # generate distance and neighbor indices
    neigh = NearestNeighbors(metric='euclidean')
    neigh.fit(X_n120)
    d, idx = neigh.kneighbors(X_n120, n_neighbors=10, return_distance=True)

    # generate distance and neighbor indices of a different shape
    neigh_2 = NearestNeighbors(metric='euclidean')
    neigh_2.fit(X_n120)
    d_2, idx_2 = neigh.kneighbors(X_n120, n_neighbors=5, return_distance=True)

    with pytest.warns(UserWarning) as record:
        # attempt to fit loop with a mismatch in shapes
        loop.LocalOutlierProbability(
            distance_matrix=d,
            neighbor_matrix=idx_2,
            n_neighbors=5,
            use_numba=NUMBA
        )

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "The shape of the distance and neighbor " \
                     "index matrices must match."


def test_input_neighbor_mismatch(X_n120) -> None:
    """
    Test to ensure that the proper warning is issued if the supplied distance
    (and neighbor) matrix and specified number of neighbors do not match.
    :param X_n120: A pytest Fixture that generates 120 observations.
    :return: None
    """
    # generate distance and neighbor indices
    neigh = NearestNeighbors(metric='euclidean')
    neigh.fit(X_n120)
    d, idx = neigh.kneighbors(X_n120, n_neighbors=5, return_distance=True)

    with pytest.warns(UserWarning) as record:
        # attempt to fit loop with a neighbor size mismatch
        loop.LocalOutlierProbability(distance_matrix=d,
                                     neighbor_matrix=idx,
                                     n_neighbors=10,
                                     use_numba=NUMBA)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "The shape of the distance or " \
                     "neighbor index matrix does not " \
                     "match the number of neighbors " \
                     "specified."


def test_loop_dist_matrix(X_n120) -> None:
    """
    Tests to ensure the proper results are returned when supplying the
    appropriate format distance and neighbor matrices.
    :param X_n120: A pytest Fixture that generates 120 observations.
    :return: None
    """
    # generate distance and neighbor indices
    neigh = NearestNeighbors(metric='euclidean')
    neigh.fit(X_n120)
    d, idx = neigh.kneighbors(X_n120, n_neighbors=10, return_distance=True)

    # fit loop using data and distance matrix
    clf1 = loop.LocalOutlierProbability(X_n120, use_numba=NUMBA)
    clf2 = loop.LocalOutlierProbability(distance_matrix=d, neighbor_matrix=idx,
                                        use_numba=NUMBA)
    scores1 = clf1.fit().local_outlier_probabilities
    scores2 = clf2.fit().local_outlier_probabilities

    # compare the agreement between the results
    assert np.abs(scores2 - scores1).all() <= 0.1


def test_lambda_values(X_n140_outliers) -> None:
    """
    Test to ensure results are returned which correspond to what is expected
    when varying the extent parameter (we expect larger extent values to
    result in more constrained scores).
    :param X_n140_outliers: A pytest Fixture that generates 140 observations.
    :return: None
    """
    # Fit the model with different extent (lambda) values
    clf1 = loop.LocalOutlierProbability(X_n140_outliers, extent=1,
                                        use_numba=NUMBA)
    clf2 = loop.LocalOutlierProbability(X_n140_outliers, extent=2,
                                        use_numba=NUMBA)
    clf3 = loop.LocalOutlierProbability(X_n140_outliers, extent=3,
                                        use_numba=NUMBA)

    # predict scores (the lower, the more normal)
    score1 = clf1.fit().local_outlier_probabilities
    score2 = clf2.fit().local_outlier_probabilities
    score3 = clf3.fit().local_outlier_probabilities

    # Get the mean of all the scores
    score_mean1 = np.mean(score1)
    score_mean2 = np.mean(score2)
    score_mean3 = np.mean(score3)

    # check that expected the means align with expectation
    assert score_mean1 > score_mean2
    assert score_mean2 > score_mean3


def test_parameters(X_n120) -> None:
    """
    Test to ensure that the model object contains the needed attributes after
    the model is fit. This is important in the context of the streaming
    functionality.
    :param X_n120: A pytest Fixture that generates 120 observations.
    :return: None
    """
    # fit the model
    clf = loop.LocalOutlierProbability(X_n120, use_numba=NUMBA).fit()

    # check that the model has attributes post fit
    assert (hasattr(clf, 'n_neighbors') and
            clf.n_neighbors is not None)
    assert (hasattr(clf, 'extent') and
            clf.extent is not None)
    assert (hasattr(clf, 'cluster_labels') and
            clf._cluster_labels() is not None)
    assert (hasattr(clf, 'prob_distances') and
            clf.prob_distances is not None)
    assert (hasattr(clf, 'prob_distances_ev') and
            clf.prob_distances_ev is not None)
    assert (hasattr(clf, 'norm_prob_local_outlier_factor') and
            clf.norm_prob_local_outlier_factor is not None)
    assert (hasattr(clf, 'local_outlier_probabilities') and
            clf.local_outlier_probabilities is not None)


def test_n_neighbors() -> None:
    """
    Tests the functionality of providing a large number of neighbors that
    is greater than the number of observations (software defaults to the
    data input size and provides a UserWarning).
    :return: None
    """
    X = iris.data
    clf = loop.LocalOutlierProbability(X, n_neighbors=500,
                                       use_numba=NUMBA).fit()
    assert clf.n_neighbors == X.shape[0] - 1

    clf = loop.LocalOutlierProbability(X, n_neighbors=500, use_numba=NUMBA)

    with pytest.warns(UserWarning) as record:
        clf.fit()

    # check that only one warning was raised
    assert len(record) == 1

    assert clf.n_neighbors == X.shape[0] - 1


def test_extent() -> None:
    """
    Test to ensure that a UserWarning is issued when providing an invalid
    extent parameter value (can be 1, 2, or 3).
    :return: None
    """
    X = np.array([[1, 1], [1, 0]])
    clf = loop.LocalOutlierProbability(X, n_neighbors=2, extent=4,
                                       use_numba=NUMBA)

    with pytest.warns(UserWarning) as record:
        clf.fit()

    # check that only one warning was raised
    assert len(record) == 1


def test_data_format() -> None:
    """
    Test to ensure that a UserWarning is issued when the shape of the input
    data is not explicitly correct. This is corrected by the software when
    possible.
    :return: None
    """
    X = [1.3, 1.1, 0.9, 1.4, 1.5, 3.2]
    clf = loop.LocalOutlierProbability(X, n_neighbors=3, use_numba=NUMBA)

    with pytest.warns(UserWarning) as record:
        clf.fit()

    # check that at least one warning was raised
    assert len(record) >= 1


def test_missing_values() -> None:
    """
    Test to ensure that the program exits of a missing value is encountered
    in the input data, as this is not allowable.
    :return: None
    """
    X = np.array([1.3, 1.1, 0.9, 1.4, 1.5, np.nan, 3.2])
    clf = loop.LocalOutlierProbability(X, n_neighbors=3, use_numba=NUMBA)

    with pytest.raises(SystemExit) as record_a, pytest.warns(
            UserWarning) as record_b:
        clf.fit()

    assert record_a.type == SystemExit

    # check that only one warning was raised
    assert len(record_b) == 1
    # check that the message matches
    assert record_b[0].message.args[
               0] == "Method does not support missing values in input data."


def test_small_cluster_size(X_n140_outliers) -> None:
    """
    Test to ensure that the program exits when the specified number of neighbors
    is larger than the smallest cluster size in the input data.
    :param X_n140_outliers: A pytest Fixture that generates 140 observations.
    :return: None
    """
    # Generate cluster labels
    a = [0] * 120
    b = [1] * 18
    cluster_labels = a + b

    clf = loop.LocalOutlierProbability(
        X_n140_outliers,
        n_neighbors=50,
        cluster_labels=cluster_labels,
        use_numba=NUMBA
    )

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


def test_stream_fit(X_n140_outliers) -> None:
    """
    Test to ensure that the proper warning is issued if the user attempts
    to use the streaming approach prior to the classical approach being fit.
    :param X_n140_outliers: A pytest Fixture that generates 140 observations.
    :return: None
    """
    # Fit the model
    X_train = X_n140_outliers[0:138]
    X_test = X_n140_outliers[139]
    clf = loop.LocalOutlierProbability(X_train, use_numba=NUMBA)

    with pytest.warns(UserWarning) as record:
        clf.stream(X_test)

    # check that the message matches
    messages = [i.message.args[0] for i in record]
    assert "Must fit on historical data by calling fit() prior to " \
           "calling stream(x)." in messages


def test_stream_distance(X_n140_outliers) -> None:
    """
    Test to ensure that the streaming approach functions as desired when
    providing matrices for use and that the returned results are within some
    margin of error when compared to the classical approach (using the RMSE).
    :param X_n140_outliers: A pytest Fixture that generates 140 observations.
    :return: None
    """
    X_train = X_n140_outliers[0:100]
    X_test = X_n140_outliers[100:140]

    # generate distance and neighbor indices
    neigh = NearestNeighbors(metric='euclidean')
    neigh.fit(X_train)
    d, idx = neigh.kneighbors(X_train, n_neighbors=10, return_distance=True)

    # Fit the models in standard and distance matrix form
    m = loop.LocalOutlierProbability(X_train, use_numba=NUMBA).fit()
    m_dist = loop.LocalOutlierProbability(distance_matrix=d,
                                          neighbor_matrix=idx,
                                          use_numba=NUMBA).fit()

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
    assert 0.075 >= rmse


def test_stream_cluster(X_n140_outliers) -> None:
    """
    Test to ensure that the proper warning is issued if the streaming approach
    is called on clustered data, as the streaming approach does not support
    this functionality.
    :param X_n140_outliers: A pytest Fixture that generates 140 observations.
    :return: None
    """
    # Generate cluster labels
    a = [0] * 120
    b = [1] * 18
    cluster_labels = a + b

    # Fit the model
    X_train = X_n140_outliers[0:138]
    X_test = X_n140_outliers[139]
    clf = loop.LocalOutlierProbability(X_train,
                                       cluster_labels=cluster_labels,
                                       use_numba=NUMBA).fit()

    with pytest.warns(UserWarning) as record:
        clf.stream(X_test)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Stream approach does not support clustered data. " \
                     "Automatically refit using single cluster of points."


def test_stream_performance(X_n140_outliers) -> None:
    """
    Test to ensure that the streaming approach works as desired when using
    a regular set of input data (no distance and neighbor matrices) and that
    the result is within some expected level of error when compared to the
    classical approach.
    :param X_n140_outliers: A pytest Fixture that generates 140 observations.
    :return:
    """
    X_train = X_n140_outliers[0:100]
    X_test = X_n140_outliers[100:140]

    # Fit the models in standard and stream form
    m = loop.LocalOutlierProbability(X_n140_outliers, use_numba=NUMBA).fit()
    scores_noclust = m.local_outlier_probabilities

    m_train = loop.LocalOutlierProbability(X_train, use_numba=NUMBA)
    m_train.fit()
    X_train_scores = m_train.local_outlier_probabilities

    X_test_scores = []
    for idx in range(X_test.shape[0]):
        X_test_scores.append(m_train.stream(X_test[idx]))
    X_test_scores = np.array(X_test_scores)

    stream_scores = np.hstack((X_train_scores, X_test_scores))

    # calculate the rmse and ensure score is below threshold
    rmse = np.sqrt(((scores_noclust - stream_scores) ** 2).mean(axis=None))
    assert 0.35 > rmse


def test_progress_bar(X_n8) -> None:
    """
    Tests the progress bar functionality on a small number of observations,
    when the number of observations is less than the width of the console
    window.
    :param X_n8: a numpy array with 8 observations.
    :return: None
    """

    # attempt to use the progress bar on a small number of observations
    loop.LocalOutlierProbability(X_n8, use_numba=NUMBA,
                                 progress_bar=True).fit()
