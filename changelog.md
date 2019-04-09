# Changelog
All notable changes to PyNomaly will be documented in this Changelog.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) 
and adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## 0.2.7
### Changed
- Integrated various performance enhancements as described in 
[pull request #30](https://github.com/vc1492a/PyNomaly/pull/30) that 
increase PyNomaly's performance by at least up to 50% in some cases.
- The Validate classes functions from public to private, as they are only 
used in validating specification and data input into PyNomaly.
### Added
- [Issue #27](https://github.com/vc1492a/PyNomaly/issues/27) - Added 
docstring to key functions in PyNomaly to ease future development and 
provide additional information.
- Additional unit tests to raise code coverage from 96% to 100%.

## 0.2.6
### Fixed
- [Issue #25](https://github.com/vc1492a/PyNomaly/issues/25) - Fixed an issue
that caused zero division errors when all the values in a neighborhood are
duplicate samples.
### Changed
- The error behavior when attempting to use the stream approach
before calling `fit`. While the previous implementation resulted in a
warning and system exit, PyNomaly now attempts to `fit` (assumes data or a
distance matrix is available) and then later calls `stream`. If no
data or distance matrix is provided, a warning is raised.
### Added
- [Issue #24](https://github.com/vc1492a/PyNomaly/issues/24) - Added
the ability to use one's own distance matrix,
provided a neighbor index matrix is also provided. This ensures
PyNomaly can be used with distances other than the euclidean.
See the file `iris_dist_grid.py` for examples.
- [Issue #23](https://github.com/vc1492a/PyNomaly/issues/23) - Added
Python 3.7 to the tested distributions in Travis CI and passed tests.
- Unit tests to monitor the issues and features covered
in issues [24](https://github.com/vc1492a/PyNomaly/issues/24) and
[25](https://github.com/vc1492a/PyNomaly/issues/25).


## 0.2.5
### Fixed
- [Issue #20](https://github.com/vc1492a/PyNomaly/issues/20) - Fixed
a bug that inadvertently used global means of the probabilistic distance
as the expected value of the probabilistic distance, as opposed to the
expected value of the probabilistic distance within a neighborhood of
a point.
- Integrated [pull request #21](https://github.com/vc1492a/PyNomaly/pull/21) -
This pull request addressed the issue noted above.
### Changed
- Changed the default behavior to strictly not supporting the
use of missing values in the input data, as opposed to the soft enforcement
(a simple user warning) used in the previous behavior.

## 0.2.4
### Fixed
- [Issue #17](https://github.com/vc1492a/PyNomaly/issues/17) - Fixed
a bug that allowed for a column of empty values in the primary data store.
- Integrated [pull request #18](https://github.com/vc1492a/PyNomaly/pull/18) -
Fixed a bug that was not causing dependencies such as numpy to skip
installation when installing PyNomaly via pip.

## 0.2.3
### Fixed
- [Issue #14](https://github.com/vc1492a/PyNomaly/issues/14) - Fixed an issue
that was causing a ZeroDivisionError when the specified neighborhood size
is larger than the total number of observations in the smallest cluster.

## 0.2.2
### Changed
- This implementation to align more closely with the specification of the
approach in the original paper. The extent parameter now takes an integer
value of 1, 2, or 3 that corresponds to the lambda parameter specified
in the paper. See the [readme](https://github.com/vc1492a/PyNomaly/blob/master/readme.md) for more details.
- Refactored the code base and created the Validate class, which includes
checks for data type, correct specification, and other dependencies.
### Added
- Automated tests to ensure the desired functionality is being met can now be
found in the `PyNomaly/tests` directory.
- Code for the examples in the readme can now be found in the `examples` directory.
- Additional information for parameter selection in the [readme](https://github.com/vc1492a/PyNomaly/blob/master/readme.md).

## 0.2.1
### Fixed
- [Issue #10](https://github.com/vc1492a/PyNomaly/issues/10) - Fixed error on line
142 which was causing the class to fail. More explicit examples
were also included in the readme for using numpy arrays.

### Added
- An improvement to the Euclidean distance calculation by [MichaelSchreier](https://github.com/MichaelSchreier)
which brings a over a 50% reduction in computation time.

## 0.2.0
### Added
- Added new functionality to PyNomaly by integrating a modified LoOP
approach introduced by Hamlet et al. which can be used for streaming
data applications or in the case where computational expense is a concern.
Data is first fit to a "training set", with any additional observations
considered for outlierness against this initial set.

## 0.1.8
### Fixed
- Fixed an issue which allowed the number of neighbors considered to exceed the number of observations. Added a check
to ensure this is no longer possible.

## 0.1.7
### Fixed
- Fixed an issue inadvertently introduced in 0.1.6 that caused distance calculations to be incorrect, 
thus resulting in incorrect LoOP values.  

## 0.1.6
### Fixed
- Updated the distance calculation such that the euclidean distance calculation has been separated from 
the main distance calculation function.
- Fixed an error in the calculation of the standard distance. 

### Changed
- .fit() now returns a fitted object instead of local_outlier_probabilities. Local outlier probabilities can 
be now be retrieved by calling .local_outlier_probabilities. See the readme for an example. 
- Some private functions have been renamed. 

## 0.1.5
### Fixed
- [Issue #4](https://github.com/vc1492a/PyNomaly/issues/4) - Separated parameter type checks 
from checks for invalid parameter values.
    - @accepts decorator verifies LocalOutlierProbability parameters are of correct type.
    - Parameter value checks moved from .fit() to init.
- Fixed parameter check to ensure extent value is in the range (0., 1.] instead of [0, 1] (extent cannot be zero). 
- [Issue #1](https://github.com/vc1492a/PyNomaly/issues/1) -  Added type check using @accepts decorator for cluster_labels.    

## 0.1.4
### Fixed
- [Issue #3](https://github.com/vc1492a/PyNomaly/issues/3) - .fit() fails if the sum of squared distances sums to 0.
    - Added check to ensure the sum of square distances is greater than zero.
    - Added UserWarning to increase the neighborhood size if all neighbors in n_neighbors are 
    zero distance from an observation. 
- Added UserWarning to check for integer type n_neighbor conditions versus float type.
- Changed calculation of the probabilistic local outlier factor expected value to Numpy operation
    from base Python. 

## 0.1.3
### Fixed
- Altered the distance matrix computation to return a triangular matrix instead of a 
fully populated matrix. This was made to ensure no duplicate neighbors were present 
in computing the neighborhood distance for each observation. 

## 0.1.2
### Added
- LICENSE.txt file of Apache License, Version 2.0.
- setup.py, setup.cfg files configured for release to PyPi.
- Changed name throughout code base from PyLoOP to PyNomaly.

### Other
- Initial release to PyPi.

## 0.1.1
### Other
- A bad push to PyPi necessitated the need to skip a version number. 
    - Chosen name of PyLoOP not present on test index but present on production PyPi index. 
    - Issue not known until push was made to the test index.
    - Skipped version number to align test and production PyPi indices.

## 0.1.0 - 2017-05-19
### Added
- readme.md file documenting methodology, package dependencies, use cases, 
how to contribute, and acknowledgements.
- Initial open release of PyNomaly codebase on Github. 