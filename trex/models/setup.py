import os
from os.path import join
import numpy

from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup
from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
    config = Configuration('models', parent_package, top_path)

    # newrand wrappers
    config.add_extension('_newrand',
                         sources=['_newrand.pyx'],
                         include_dirs=[numpy.get_include(),
                                       join('src', 'newrand')],
                         depends=[join('src', 'newrand', 'newrand.h')],
                         language='c++',
                         # Use C++11 random number generator fix
                         extra_compile_args=['-std=c++11']
                         )

    # liblinear module
    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    # config.add_extension("_cython_blas",
    #                      sources=["_cython_blas.pyx"],
    #                      include_dirs=[numpy.get_include()],
    #                      libraries=libraries,
    #                      extra_compile_args=["-O3"])

    # precompile liblinear to use C++11 flag
    config.add_library('liblinear-skl',
                       sources=[join('src', 'liblinear', 'linear.cpp'),
                                join('src', 'liblinear', 'tron.cpp')],
                       depends=[join('src', 'liblinear', 'linear.h'),
                                join('src', 'liblinear', 'tron.h'),
                                join('src', 'newrand', 'newrand.h')],
                       # Force C++ linking in case gcc is picked up instead
                       # of g++ under windows with some versions of MinGW
                       extra_link_args=['-lstdc++'],
                       # Use C++11 to use the random number generator fix
                       extra_compiler_args=['-std=c++11'],
                       )

    liblinear_sources = ['_liblinear.pyx']
    liblinear_depends = [join('src', 'liblinear', '*.h'),
                         join('src', 'newrand', 'newrand.h'),
                         join('src', 'liblinear', 'liblinear_helper.c')]

    config.add_extension('_liblinear',
                         sources=liblinear_sources,
                         libraries=['liblinear-skl'] + libraries,
                         include_dirs=[join('.', 'src', 'liblinear'),
                                       join('.', 'src', 'newrand'),
                                       # join('.'),
                                       numpy.get_include()],
                         depends=liblinear_depends,
                         # extra_compile_args=['-O0 -fno-inline'],
                         )

    # end liblinear module

    config.ext_modules = cythonize(
        config.ext_modules,
        compiler_directives={'language_level': 3},
        annotate=True
    )

    return config


if __name__ == '__main__':
    setup(**configuration(top_path='').todict())
