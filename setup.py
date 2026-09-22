from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='PyNomaly',
    packages=['PyNomaly'],
    version='1.0.0',
    description='A Python 3 implementation of LoOP: Local Outlier '
                'Probabilities, a local density based outlier detection '
                'method providing an outlier score in the range of [0,1].',
    author='Valentino Constantinou',
    author_email='vc@valentino.io',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/vc1492a/PyNomaly',
    download_url='https://github.com/vc1492a/PyNomaly/archive/1.0.0.tar.gz',
    keywords=['outlier', 'anomaly', 'detection', 'machine', 'learning',
              'probability'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='>=3.9',
    license='Apache License, Version 2.0',
    install_requires=['numpy', 'python-utils'],
    extras_require={
        'sklearn': ['scikit-learn>=1.0', 'scipy>=1.3.0'],
        'sparse': ['scipy>=1.3.0'],
        'numba': ['numba'],
        'all': ['scikit-learn>=1.0', 'numba', 'scipy>=1.3.0'],
    },
)
