# Authors: Valentino Constantinou <vc@valentino.io>
# License: Apache 2.0


class PyNomalyError(Exception):
    """Base exception for PyNomaly."""
    pass


class ValidationError(PyNomalyError):
    """Raised when input validation fails."""
    pass


class ClusterSizeError(ValidationError):
    """Raised when cluster size is smaller than n_neighbors."""
    pass


class MissingValuesError(ValidationError):
    """Raised when data contains missing values."""
    pass
