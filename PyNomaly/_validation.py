# Authors: Valentino Constantinou <vc@valentino.io>
# License: Apache 2.0

import numpy as np
from typing import Union
import warnings

from PyNomaly.exceptions import ClusterSizeError, MissingValuesError


class ValidationMixin:
    """Mixin providing input validation methods for LocalOutlierProbability."""

    @staticmethod
    def _convert_to_array(obj: Union["pd.DataFrame", np.ndarray]) -> np.ndarray:
        """
        Converts the input data to a numpy array if it is a Pandas DataFrame
        or validates it is already a numpy array.
        :param obj: user-provided input data.
        :return: a vector of values to be used in calculating the local
        outlier probability.
        """
        if obj.__class__.__name__ == "DataFrame":
            points_vector = obj.values
            return points_vector
        elif obj.__class__.__name__ == "ndarray":
            points_vector = obj
            return points_vector
        else:
            warnings.warn(
                "Provided data or distance matrix must be in ndarray "
                "or DataFrame.",
                UserWarning,
            )
            if isinstance(obj, list):
                points_vector = np.array(obj)
                return points_vector
            points_vector = np.array([obj])
            return points_vector

    def _validate_inputs(self):
        """
        Validates the inputs provided during initialization to ensure
        that the needed objects are provided.
        :return: a tuple of (data, distance_matrix, neighbor_matrix) or
        raises a warning for invalid inputs.
        """
        if all(v is None for v in [self.data, self.distance_matrix]):
            warnings.warn(
                "Data or a distance matrix must be provided.", UserWarning
            )
            return False
        elif all(v is not None for v in [self.data, self.distance_matrix]):
            warnings.warn(
                "Only one of the following may be provided: data or a "
                "distance matrix (not both).",
                UserWarning,
            )
            return False
        if self.data is not None:
            points_vector = self._convert_to_array(self.data)
            return points_vector, self.distance_matrix, self.neighbor_matrix
        if all(
            matrix is not None
            for matrix in [self.neighbor_matrix, self.distance_matrix]
        ):
            dist_vector = self._convert_to_array(self.distance_matrix)
            neigh_vector = self._convert_to_array(self.neighbor_matrix)
        else:
            warnings.warn(
                "A neighbor index matrix and distance matrix must both be "
                "provided when not using raw input data.",
                UserWarning,
            )
            return False
        if self.distance_matrix.shape != self.neighbor_matrix.shape:
            warnings.warn(
                "The shape of the distance and neighbor "
                "index matrices must match.",
                UserWarning,
            )
            return False
        elif (self.distance_matrix.shape[1] != self.n_neighbors) or (
            self.neighbor_matrix.shape[1] != self.n_neighbors
        ):
            warnings.warn(
                "The shape of the distance or "
                "neighbor index matrix does not "
                "match the number of neighbors "
                "specified.",
                UserWarning,
            )
            return False
        return self.data, dist_vector, neigh_vector

    def _check_cluster_size(self) -> None:
        """
        Validates the cluster labels to ensure that the smallest cluster
        size (number of observations in the cluster) is larger than the
        specified number of neighbors.
        :raises ClusterSizeError: if any cluster is too small.
        """
        c_labels = self._cluster_labels()
        for cluster_id in set(c_labels):
            c_size = np.where(c_labels == cluster_id)[0].shape[0]
            if c_size <= self.n_neighbors:
                raise ClusterSizeError(
                    "Number of neighbors specified larger than smallest "
                    "cluster. Specify a number of neighbors smaller than "
                    "the smallest cluster size (observations in smallest "
                    "cluster minus one)."
                )

    def _check_n_neighbors(self) -> bool:
        """
        Validates the specified number of neighbors to ensure that it is
        greater than 0 and that the specified value is less than the total
        number of observations.
        :return: a boolean indicating whether validation has passed without
        adjustment.
        """
        if not self.n_neighbors > 0:
            self.n_neighbors = 10
            warnings.warn(
                "n_neighbors must be greater than 0."
                " Fit with " + str(self.n_neighbors) + " instead.",
                UserWarning,
            )
            return False
        elif self.n_neighbors >= self._n_observations():
            self.n_neighbors = self._n_observations() - 1
            warnings.warn(
                "n_neighbors must be less than the number of observations."
                " Fit with " + str(self.n_neighbors) + " instead.",
                UserWarning,
            )
        return True

    def _check_extent(self) -> bool:
        """
        Validates the specified extent parameter to ensure it is either 1,
        2, or 3.
        :return: a boolean indicating whether validation has passed.
        """
        if self.extent not in [1, 2, 3]:
            warnings.warn(
                "extent parameter (lambda) must be 1, 2, or 3.", UserWarning
            )
            return False
        return True

    def _check_missing_values(self) -> None:
        """
        Validates the provided data to ensure that it contains no
        missing values.
        :raises MissingValuesError: if data contains NaN values.
        """
        if np.any(np.isnan(self.data)):
            raise MissingValuesError(
                "Method does not support missing values in input data."
            )

    def _check_is_fit(self) -> bool:
        """
        Checks that the model was fit prior to calling the stream() method.
        :return: a boolean indicating whether the model has been fit.
        """
        if self.is_fit is False:
            warnings.warn(
                "Must fit on historical data by calling fit() prior to "
                "calling stream(x).",
                UserWarning,
            )
            return False
        return True

    def _check_no_cluster_labels(self) -> bool:
        """
        Checks to see if cluster labels are attempting to be used in
        stream() and, if so, returns False. As PyNomaly does not accept
        clustering algorithms as input, the stream approach does not
        support clustering.
        :return: a boolean indicating whether single cluster (no labels).
        """
        if len(set(self._cluster_labels())) > 1:
            warnings.warn(
                "Stream approach does not support clustered data. "
                "Automatically refit using single cluster of points.",
                UserWarning,
            )
            return False
        return True


def accepts(*types):
    """
    A decorator that facilitates a form of type checking for the inputs
    which can be used in Python 3.4-3.7 in lieu of Python 3.5+'s type
    hints.
    :param types: the input types of the objects being passed as arguments
    in __init__.
    :return: a decorator.
    """

    def decorator(f):
        assert len(types) == f.__code__.co_argcount

        def new_f(*args, **kwds):
            for a, t in zip(args, types):
                if type(a).__name__ == "DataFrame":
                    a = np.array(a)
                if isinstance(a, t) is False:
                    warnings.warn(
                        "Argument %r is not of type %s" % (a, t), UserWarning
                    )
            opt_types = {
                "extent": {"type": (int, np.integer)},
                "n_neighbors": {"type": (int, np.integer)},
                "use_numba": {"type": bool},
                "n_jobs": {"type": (int, np.integer)},
                "progress_bar": {"type": bool},
                "data": {"type": np.ndarray},
                "distance_matrix": {"type": np.ndarray},
                "neighbor_matrix": {"type": np.ndarray},
                "cluster_labels": {"type": (list, np.ndarray)},
            }
            for x in kwds:
                if x in opt_types:
                    v = kwds[x]
                    if type(v).__name__ == "DataFrame":
                        v = np.array(v)
                    opt_types[x]["value"] = v
            for k in opt_types:
                try:
                    if (
                        isinstance(opt_types[k]["value"], opt_types[k]["type"])
                        is False
                    ):
                        warnings.warn(
                            "Argument %r is not of type %s."
                            % (k, opt_types[k]["type"]),
                            UserWarning,
                        )
                except KeyError:
                    pass
            return f(*args, **kwds)

        new_f.__name__ = f.__name__
        return new_f

    return decorator
