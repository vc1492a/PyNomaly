from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='PyNomaly',
    packages=['PyNomaly'],
    version='0.3.4',
    description='A Python 3 implementation of LoOP: Local Outlier '
                'Probabilities, a local density based outlier detection '
                'method providing an outlier score in the range of [0,1].',
    author='Valentino Constantinou',
    author_email='vc@valentino.io',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/vc1492a/PyNomaly',
    download_url='https://github.com/vc1492a/PyNomaly/archive/0.3.4.tar.gz',
    keywords=['outlier', 'anomaly', 'detection', 'machine', 'learning',
              'probability'],
    classifiers=[],
    license='Apache License, Version 2.0',
    install_requires=['numpy', 'python-utils']
)
