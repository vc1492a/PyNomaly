from setuptools import setup

setup(
    name='PyNomaly',
    packages=['PyNomaly'],
    version='0.2.2',
    description='A Python 3 implementation of LoOP: Local Outlier Probabilities, a local density based outlier detection method providing an outlier score in the range of [0,1].',
    author='Valentino Constantinou',
    author_email='vc@valentino.io',
    url='https://github.com/vc1492a/PyNomaly',
    download_url='https://github.com/vc1492a/PyNomaly/archive/0.2.2.tar.gz',
    keywords=['outlier', 'anomaly', 'detection', 'machine', 'learning', 'probability'],
    classifiers=[],
    license='Apache License, Version 2.0',
    setup_requires=['numpy']
)
