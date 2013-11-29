#!/usr/bin/env python
from ez_setup import use_setuptools
use_setuptools()

import os

from setuptools import setup, find_packages, Extension
from distutils.command import build_ext
import numpy

VERSION = '0.0.1'
DESCRIPTION = "Recommendation Engines based on Matrix methods"
LONG_DESCRIPTION = """
"""

CLASSIFIERS = filter(None, map(str.strip,
"""
Intended Audience :: Developers
License :: OSI Approved :: MIT License
Programming Language :: Python
Operating System :: OS Independent
Topic :: Utilities
Topic :: Database :: Database Engines/Servers
Topic :: Software Development :: Libraries :: Python Modules
""".splitlines()))

setup(
    name="mfrec",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    classifiers=CLASSIFIERS,
    keywords=('recommendation', 'wrmf', 'matrix', 'factorization','svd'),
    author="Martin Laprise",
    author_email="mlaprise@gmail.com",
    url="https://github.com/mlaprise/mfrec",
    license="MIT License",
    packages=find_packages(exclude=['ez_setup']),
    platforms=['any'],
    zip_safe=False,
    install_requires=['numpy', 'cython'],
    ext_modules = [
                   Extension("mfrec.lib.als_implicit", ["mfrec/lib/als_implicit.c"], include_dirs=[numpy.get_include()]),
                   Extension("mfrec.lib.gd_estimator", ["mfrec/lib/gd_estimator.c"], include_dirs=[numpy.get_include()]),
                   Extension("mfrec.lib.kmf_train", ["mfrec/lib/kmf_train.c"], include_dirs=[numpy.get_include()]),
                  ],
)
