# Changelog
All notable changes to PyNomaly will be documented in this Changelog.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) 
and adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

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