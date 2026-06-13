# PyNomaly

PyNomaly is a Python 3 implementation of LoOP (Local Outlier Probabilities).
LoOP is a local density based outlier detection method by Kriegel, Kröger, Schubert, and Zimek which provides outlier
scores in the range of [0,1] that are directly interpretable as the probability of a sample being an outlier.

PyNomaly is a core library of [deepchecks](https://github.com/deepchecks/deepchecks), [OmniDocBench](https://github.com/opendatalab/OmniDocBench) and [pysad](https://github.com/selimfirat/pysad).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPi](https://img.shields.io/badge/pypi-0.4.0-blue.svg)](https://pypi.python.org/pypi/PyNomaly/0.4.0)
[![Total Downloads](https://static.pepy.tech/badge/pynomaly)](https://pepy.tech/projects/pynomaly)
[![Monthly Downloads](https://static.pepy.tech/badge/pynomaly/month)](https://pepy.tech/projects/pynomaly)
![Tests](https://github.com/vc1492a/PyNomaly/actions/workflows/tests.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/vc1492a/PyNomaly/badge.svg?branch=main)](https://coveralls.io/github/vc1492a/PyNomaly?branch=main)
[![JOSS](http://joss.theoj.org/papers/f4d2cfe680768526da7c1f6a2c103266/status.svg)](http://joss.theoj.org/papers/f4d2cfe680768526da7c1f6a2c103266)

## Overview

The outlier score of each sample is called the Local Outlier Probability.
It measures the local deviation of density of a given sample with
respect to its neighbors as Local Outlier Factor (LOF), but provides normalized
outlier scores in the range [0,1]. These outlier scores are directly interpretable
as a probability of an object being an outlier. Since Local Outlier Probabilities provides scores in the
range [0,1], practitioners are free to interpret the results according to the application.

Like LOF, it is local in that the anomaly score depends on how isolated the sample is
with respect to the surrounding neighborhood. Locality is given by k-nearest neighbors,
whose distance is used to estimate the local density. By comparing the local density of a sample to the
local densities of its neighbors, one can identify samples that lie in regions of lower
density compared to their neighbors and thus identify samples that may be outliers according to their Local
Outlier Probability.

The authors' 2009 paper detailing LoOP's theory, formulation, and application is provided by
Ludwig-Maximilians University Munich - Institute for Informatics;
[LoOP: Local Outlier Probabilities](http://www.dbs.ifi.lmu.de/Publikationen/Papers/LoOP1649.pdf).

## Quick Links

- [How It Works](how-it-works.md) -- understand the algorithm
- [Getting Started](getting-started.md) -- installation and first steps
- [User Guide](user-guide.md) -- parameters, performance, streaming, and error handling
- [API Reference](api.md) -- full class and method documentation
- [Examples](examples.md) -- worked examples with visualizations

## Research

If citing PyNomaly, use the following:

```bibtex
@article{Constantinou2018,
  doi = {10.21105/joss.00845},
  url = {https://doi.org/10.21105/joss.00845},
  year  = {2018},
  month = {oct},
  publisher = {The Open Journal},
  volume = {3},
  number = {30},
  pages = {845},
  author = {Valentino Constantinou},
  title = {{PyNomaly}: Anomaly detection using Local Outlier Probabilities ({LoOP}).},
  journal = {Journal of Open Source Software}
}
```

## References

1. Breunig M., Kriegel H.-P., Ng R., Sander, J. LOF: Identifying Density-based Local Outliers. ACM SIGMOD International Conference on Management of Data (2000). [PDF](http://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf).
2. Kriegel H., Kröger P., Schubert E., Zimek A. LoOP: Local Outlier Probabilities. 18th ACM conference on Information and knowledge management, CIKM (2009). [PDF](http://www.dbs.ifi.lmu.de/Publikationen/Papers/LoOP1649.pdf).
3. Goldstein M., Uchida S. A Comparative Evaluation of Unsupervised Anomaly Detection Algorithms for Multivariate Data. PLoS ONE 11(4): e0152173 (2016).
4. Hamlet C., Straub J., Russell M., Kerlin S. An incremental and approximate local outlier probability algorithm for intrusion detection and its evaluation. Journal of Cyber Security Technology (2016). [DOI](http://www.tandfonline.com/doi/abs/10.1080/23742917.2016.1226651?journalCode=tsec20).

## Acknowledgements

- The authors of LoOP (Local Outlier Probabilities)
    - Hans-Peter Kriegel
    - Peer Kröger
    - Erich Schubert
    - Arthur Zimek
- [NASA Jet Propulsion Laboratory](https://jpl.nasa.gov/)
    - [Kyle Hundman](https://github.com/khundman)
    - [Ian Colwell](https://github.com/iancolwell)
