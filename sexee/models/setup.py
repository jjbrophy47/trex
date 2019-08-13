"""
Setup to extend custom modifications to liblinear.
"""
import os
from setuptools import setup
from distutils.core import Extension

here = os.path.abspath(os.path.dirname(__file__))

sources = [os.path.join('liblinear', 'train.c'),
           os.path.join('liblinear', 'linear.cpp'),
           os.path.join('liblinear', 'tron.cpp')]
           # os.path.join('liblinear', 'predict.c')]
depends = [os.path.join('liblinear', '*.h')]
include_dirs = [os.path.join('liblinear')]
liblinear = Extension('mymodule', sources=sources, depends=depends, include_dirs=include_dirs)

setup(
    name='mymodule',
    version='0.1',
    ext_modules=[liblinear]
)
