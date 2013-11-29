from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("gd_estimator", ["gd_estimator.pyx"], include_dirs=[numpy.get_include()]),
                   Extension("als_implicit", ["als_implicit.pyx"], include_dirs=[numpy.get_include()]),
                   Extension("kmf_train", ["kmf_train.pyx"], include_dirs=[numpy.get_include()])]
)
