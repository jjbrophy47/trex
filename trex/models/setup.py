import os

import numpy
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup
from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):

    config = Configuration('models', parent_package, top_path)

    # liblinear module
    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    # precompile liblinear to use C++11 flag
    config.add_library('liblinear-skl',
                       sources=[os.path.join('src', 'linear.cpp'),
                                os.path.join('src', 'tron.cpp')],
                       depends=[os.path.join('src', 'linear.h'),
                                os.path.join('src', 'tron.h')],
                       language='c++',
                       # Force C++ linking in case gcc is picked up instead
                       # of g++ under windows with some versions of MinGW
                       extra_link_args=['-lstdc++'],
                       )

    liblinear_sources = ['_liblinear.pyx']
    liblinear_depends = [os.path.join('src', '*.h'),
                         os.path.join('src', 'liblinear_helper.c')]

    config.add_extension('_liblinear',
                         sources=liblinear_sources,
                         libraries=['liblinear-skl'] + libraries,
                         include_dirs=[os.path.join('.', 'src'),
                                       numpy.get_include()],
                         depends=liblinear_depends
                         )

    config.ext_modules = cythonize(
        config.ext_modules,
        compiler_directives={'language_level': 3},
        annotate=True
    )

    return config


if __name__ == '__main__':
    setup(**configuration(top_path='').todict())
