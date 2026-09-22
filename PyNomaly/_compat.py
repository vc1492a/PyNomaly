# Authors: Valentino Constantinou <vc@valentino.io>
# License: Apache 2.0

"""
Optional scikit-learn integration.

When scikit-learn is installed, LocalOutlierProbability inherits from
``BaseEstimator`` and ``OutlierMixin`` for full ecosystem compatibility
(pipelines, grid search, ``check_estimator``).

When it is not installed, lightweight fallbacks provide the same public
surface (``get_params``, ``set_params``, ``fit_predict``, ``check_is_fitted``)
so PyNomaly works as a standalone package.

Install the optional dependency with::

    pip install PyNomaly[sklearn]
"""

from __future__ import annotations

import inspect

try:
    from sklearn.base import BaseEstimator, OutlierMixin
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_array, check_is_fitted

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without sklearn
    _SKLEARN_AVAILABLE = False

    class NotFittedError(ValueError, AttributeError):
        """Raised when an estimator method is used before fit."""

    class BaseEstimator:
        """Minimal BaseEstimator providing get_params / set_params."""

        def get_params(self, deep=True):
            params = {}
            for name, param in inspect.signature(self.__init__).parameters.items():
                if name == "self":
                    continue
                params[name] = getattr(self, name, param.default)
            return params

        def set_params(self, **params):
            if not params:
                return self
            valid = self.get_params(deep=True)
            for key, value in params.items():
                if key not in valid:
                    raise ValueError(
                        f"Invalid parameter {key!r} for estimator "
                        f"{type(self).__name__}. Valid parameters are: "
                        f"{sorted(valid)!r}."
                    )
                setattr(self, key, value)
            return self

        def __repr__(self):
            params = self.get_params(deep=False)
            filtered = {
                k: v
                for k, v in params.items()
                if v is not inspect.Parameter.empty
            }
            inner = ", ".join(f"{k}={v!r}" for k, v in filtered.items())
            return f"{type(self).__name__}({inner})"

    class OutlierMixin:
        """Minimal OutlierMixin providing fit_predict."""

        def fit_predict(self, X, y=None, **fit_params):
            return self.fit(X, y, **fit_params).predict(X)

    def check_is_fitted(estimator, attributes=None, *, msg=None):
        if attributes is None:
            fitted = [
                a
                for a in vars(estimator)
                if a.endswith("_") and not a.startswith("__")
            ]
            if not fitted:
                raise NotFittedError(
                    msg
                    or (
                        f"This {type(estimator).__name__} instance is not "
                        f"fitted yet. Call 'fit' with appropriate arguments "
                        f"before using this estimator."
                    )
                )
            return

        if isinstance(attributes, str):
            attributes = [attributes]
        missing = [a for a in attributes if not hasattr(estimator, a)]
        if missing:
            raise NotFittedError(
                msg
                or (
                    f"This {type(estimator).__name__} instance is not fitted "
                    f"yet. Call 'fit' with appropriate arguments before using "
                    f"this estimator."
                )
            )

    def check_array(
        array,
        *,
        accept_sparse=False,
        dtype="numeric",
        ensure_2d=True,
        ensure_min_samples=1,
        ensure_min_features=1,
        copy=False,
        **kwargs,
    ):
        """Fallback input validator approximating sklearn.utils.check_array."""
        import numpy as np

        try:
            from scipy.sparse import issparse
        except ImportError:

            def issparse(x):
                return False

        if issparse(array):
            if accept_sparse is False or accept_sparse == []:
                raise TypeError(
                    "A sparse matrix was passed, but dense data is required. "
                    "Use X.toarray() to convert to a dense numpy array."
                )
            array = array.toarray()

        if hasattr(array, "values") and hasattr(array, "columns"):
            array = array.values

        arr = np.asarray(array)
        if copy:
            arr = np.array(arr, copy=True)

        if ensure_2d and arr.ndim == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead:\n"
                f"array={arr}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) if "
                "it contains a single sample."
            )
        if ensure_2d and arr.ndim != 2:
            raise ValueError(
                f"Expected 2D array, got array with ndim={arr.ndim} instead."
            )

        if arr.size == 0:
            n_samples = arr.shape[0] if arr.ndim >= 1 else 0
            n_features = arr.shape[1] if arr.ndim >= 2 else 0
            if n_samples < ensure_min_samples:
                raise ValueError(
                    f"Found array with {n_samples} sample(s) "
                    f"(shape={arr.shape}) while a minimum of "
                    f"{ensure_min_samples} is required."
                )
            if n_features < ensure_min_features:
                raise ValueError(
                    f"Found array with {n_features} feature(s) "
                    f"(shape={arr.shape}) while a minimum of "
                    f"{ensure_min_features} is required."
                )

        if np.iscomplexobj(arr):
            raise ValueError("Complex data not supported")

        if dtype == "numeric":
            try:
                arr = arr.astype(float, copy=False)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    "Unable to convert array of strings to floating numbers."
                ) from exc

        if ensure_2d:
            if arr.shape[0] < ensure_min_samples:
                raise ValueError(
                    f"Found array with {arr.shape[0]} sample(s) "
                    f"(shape={arr.shape}) while a minimum of "
                    f"{ensure_min_samples} is required."
                )
            if arr.shape[1] < ensure_min_features:
                raise ValueError(
                    f"Found array with {arr.shape[1]} feature(s) "
                    f"(shape={arr.shape}) while a minimum of "
                    f"{ensure_min_features} is required."
                )

        return arr
