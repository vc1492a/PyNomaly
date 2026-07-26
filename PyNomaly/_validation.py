# Authors: Valentino Constantinou <vc@valentino.io>
# License: Apache 2.0

import warnings
from functools import wraps
from typing import Union

import numpy as np

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
        if obj.__class__.__name__ == "DataFrame" or obj.__class__.__name__ == "Series":
            arr = obj.values
        elif obj.__class__.__name__ == "ndarray":
            arr = obj
        else:
            warnings.warn(
                "Provided data or distance matrix must be in ndarray or DataFrame.",
                UserWarning,
            )
            # Let native NumPy exceptions (ValueError/TypeError) bubble up
            arr = np.asarray(obj, dtype=float)

            if arr.ndim == 0:
                arr = np.array([obj], dtype=float)

        # For scikit-learn compliance
        if arr.size == 0:
            if arr.ndim >= 2 and arr.shape[1] == 0:
                raise ValueError(
                    f"Found array with 0 feature(s) (shape={arr.shape}) while a "
                    f"minimum of 1 is required."
                )
            else:
                raise ValueError(
                    f"Found array with 0 sample(s) (shape={arr.shape}) while a "
                    f"minimum of 1 is required."
                )
        # For scikit-learn compliance
        if np.iscomplexobj(arr):
            raise ValueError("Complex data not supported.")

        # Let native NumPy exceptions (ValueError/TypeError) bubble up
        arr = arr.astype(float)

        return arr

    def _validate_inputs(self):
        """
        Validates the inputs provided during initialization to ensure
        that the needed objects are provided.
        :return: a tuple of (data, distance_matrix, neighbor_matrix) or
        raises a warning for invalid inputs.
        """
        _data = getattr(self, "data_", getattr(self, "data", None))
        _dist = getattr(
            self, "distance_matrix_", getattr(self, "distance_matrix", None)
        )
        _neigh = getattr(
            self, "neighbor_matrix_", getattr(self, "neighbor_matrix", None)
        )

        if all(v is None for v in [_data, _dist]):
            warnings.warn("Data or a distance matrix must be provided.", UserWarning)
            return False
        elif all(v is not None for v in [_data, _dist]):
            warnings.warn(
                "Only one of the following may be provided: data or a "
                "distance matrix (not both).",
                UserWarning,
            )
            return False

        if _data is not None:
            points_vector = self._convert_to_array(_data)
            return points_vector, _dist, _neigh

        if all(matrix is not None for matrix in [_neigh, _dist]):
            dist_vector = self._convert_to_array(_dist)
            neigh_vector = self._convert_to_array(_neigh)
        else:
            warnings.warn(
                "A neighbor index matrix and distance matrix must both be "
                "provided when not using raw input data.",
                UserWarning,
            )
            return False

        if _dist.shape != _neigh.shape:
            warnings.warn(
                "The shape of the distance and neighbor index matrices must match.",
                UserWarning,
            )
            return False
        elif (_dist.shape[1] != self.n_neighbors) or (
            _neigh.shape[1] != self.n_neighbors
        ):
            warnings.warn(
                "The shape of the distance or neighbor index matrix does not "
                "match the number of neighbors specified.",
                UserWarning,
            )
            return False

        return _data, dist_vector, neigh_vector

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
            warnings.warn("extent parameter (lambda) must be 1, 2, or 3.", UserWarning)
            return False
        return True

    def _check_missing_values(self) -> None:
        """
        Validates the provided data to ensure that it contains no
        missing values.
        :raises MissingValuesError: if data contains NaN values.
        """
        _data = getattr(self, "data_", getattr(self, "data", None))
        if _data is not None:
            arr = self._convert_to_array(_data)
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                raise MissingValuesError(
                    "Input contains NaN, infinity or a value too large for dtype('float64')."
                )

    def _check_is_fit(self) -> bool:
        """
        Checks that the model was fit prior to calling the stream() method.
        :return: a boolean indicating whether the model has been fit.
        """
        if getattr(self, "is_fit_", False) is False:
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


def validate_init_types(*types):
    """
    A decorator that facilitates a form of type checking for the inputs
    which can be used in Python 3.4-3.7 in lieu of Python 3.5+'s type
    hints.
    :param types: the input types of the objects being passed as arguments
    in __init__.
    :return: a decorator.
    """

    def decorator(f):
        @wraps(f)
        def new_f(self, *args, **kwds):
            # Map the expected types passed to @accepts to the class attributes
            attr_names = [
                "extent",
                "n_neighbors",
                "use_numba",
                "n_jobs",
                "progress_bar",
                "data",
                "distance_matrix",
                "neighbor_matrix",
                "cluster_labels",
            ]

            # types[1:] skips the first 'object' type which was originally meant for 'self'
            for attr, expected_type in zip(attr_names, types[1:]):
                val = getattr(self, attr, None)

                if val is not None:
                    if type(val).__name__ == "DataFrame":
                        val = np.array(val)
                    if not isinstance(val, expected_type):
                        warnings.warn(
                            "Argument %r is not of type %s." % (attr, expected_type),
                            UserWarning,
                        )

            return f(self, *args, **kwds)

        return new_f

    return decorator
