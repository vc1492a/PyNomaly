# Changelog
All notable changes to PyNomaly will be documented in this Changelog.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) 
and adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [0.2.1]
### Fixed
- [Issue #10](https://github.com/vc1492a/PyNomaly/issues/10) - Fixed error on line
142 which was causing the class to fail. More explicit examples
were also included in the readme for using numpy arrays.

## [0.2.0](https://github.com/vc1492a/PyNomaly/commit/9e1996f08da3d151461adbb2b86c5d9447aaafa4)
### Added
- Added new functionality to PyNomaly by integrating a modified LoOP
approach introduced by Hamlet et al. which can be used for streaming
data applications or in the case where computational expense is a concern.
Data is first fit to a "training set", with any additional observations
considered for outlierness against this initial set.

## [0.1.8](https://github.com/vc1492a/PyNomaly/commit/da203acdb50a013667ba5e57dd2facc7a7e4b8a5)
### Fixed
- Fixed an issue which allowed the number of neighbors considered to exceed the number of observations. Added a check
to ensure this is no longer possible.

## [0.1.7](https://github.com/vc1492a/PyNomaly/commit/8df501ab5c5605873c2812f6d8fe8730e2586975)
### Fixed
- Fixed an issue inadvertently introduced in 0.1.6 that caused distance calculations to be incorrect, 
thus resulting in incorrect LoOP values.  

## [0.1.6](https://github.com/vc1492a/PyNomaly/commit/2526879b1f941c887eeb24a267b5ea010e20d5d7) - 2017-12-17
### Fixed
- Updated the distance calculation such that the euclidean distance calculation has been separated from 
the main distance calculation function.
- Fixed an error in the calculation of the standard distance. 

### Changed
- .fit() now returns a fitted object instead of local_outlier_probabilities. Local outlier probabilities can 
be now be retrieved by calling .local_outlier_probabilities. See the readme for an example. 
- Some private functions have been renamed. 

## [0.1.5](https://github.com/vc1492a/PyNomaly/commit/d203c402dd657e8240365d538c723f831237326e) - 2017-07-30
### Fixed
- [Issue #4](https://github.com/vc1492a/PyNomaly/issues/4) - Separated parameter type checks 
from checks for invalid parameter values.
    - @accepts decorator verifies LocalOutlierProbability parameters are of correct type.
    - Parameter value checks moved from .fit() to init.
- Fixed parameter check to ensure extent value is in the range (0., 1.] instead of [0, 1] (extent cannot be zero). 
- [Issue #1](https://github.com/vc1492a/PyNomaly/issues/1) -  Added type check using @accepts decorator for cluster_labels.    

## [0.1.4](https://github.com/vc1492a/PyNomaly/commit/8f5a640c7b7ecfd824113dbba77fff19cc153424) - 2017-06-29
### Fixed
- [Issue #3](https://github.com/vc1492a/PyNomaly/issues/3) - .fit() fails if the sum of squared distances sums to 0.
    - Added check to ensure the sum of square distances is greater than zero.
    - Added UserWarning to increase the neighborhood size if all neighbors in n_neighbors are 
    zero distance from an observation. 
- Added UserWarning to check for integer type n_neighbor conditions versus float type.
- Changed calculation of the probabilistic local outlier factor expected value to Numpy operation
    from base Python. 

## [0.1.3](https://github.com/vc1492a/PyNomaly/commit/ae4692b6f2d0871130a02b9ee54049321b854524) - 2017-06-10
### Fixed
- Altered the distance matrix computation to return a triangular matrix instead of a 
fully populated matrix. This was made to ensure no duplicate neighbors were present 
in computing the neighborhood distance for each observation. 

## [0.1.2](https://pypi.python.org/pypi?:action=display&name=PyNomaly&version=0.1.2) - 2017-06-01
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