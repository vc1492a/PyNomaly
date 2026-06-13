# Contributing

Please use the [issue tracker](https://github.com/vc1492a/PyNomaly/issues) to report any erroneous behavior or desired feature requests.

If you would like to contribute to development, please fork the repository and make any changes to a branch which corresponds to an open issue. Hot fixes and bug fixes can be represented by branches with the prefix `fix/` versus `feature/` for new capabilities or code improvements. Pull requests will then be made from these branches into the repository's `dev` branch prior to being pulled into `main`.

## Commit Messages

**Your commit messages are important.** PyNomaly uses the [Conventional Commits](https://www.conventionalcommits.org/) specification. Conventional Commits provides an easy set of rules for creating an explicit commit history, which dovetails with [Semantic Versioning](http://semver.org/) by describing the features, fixes, and breaking changes made in commit messages. You can check out examples [here](https://www.conventionalcommits.org/en/v1.0.0/#examples). Make a best effort to use the specification when contributing, as it dramatically eases the documentation around releases and their features, breaking changes, bug fixes, and documentation updates.

## Tests

When contributing, please ensure to run unit tests and add additional tests as necessary if adding new functionality. To run the unit tests, use `pytest`:

```shell
python3 -m pytest --cov=PyNomaly -s -v
```

To run the tests with Numba enabled, simply set the flag `NUMBA` in `test_loop.py` to `True`. Note that a drop in coverage is expected due to portions of the code being compiled upon code execution. Additionally, there are dedicated Numba-specific tests (`test_numba_*`) that run automatically when Numba is installed and are skipped otherwise.

## Versioning

[Semantic versioning](http://semver.org/) is used for this project. If contributing, please conform to semantic versioning guidelines when submitting a pull request.

## License

This project is licensed under the Apache 2.0 license.
