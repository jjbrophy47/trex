"""
Reference: https://github.com/slundberg/shap/blob/master/setup.py
"""
import os
import re
import codecs
from setuptools import setup

import distutils.cmd
import distutils.log
import setuptools
import subprocess

here = os.path.abspath(os.path.dirname(__file__))


class LiblinearMake(distutils.cmd.Command):
    """A custom command to run Pylint on all Python source files."""

    description = 'compiles liblinear'
    user_options = [
        # The format is (long option, short option, description).
        # ('pylint-rcfile=', None, 'path to Pylint config file'),
    ]

    def initialize_options(self):
        """Set default values for options."""
    #   # Each user option must be listed here with their default value.
    #   self.pylint_rcfile = ''

    def finalize_options(self):
        """Post-process options."""
    #   if self.pylint_rcfile:
    #     assert os.path.exists(self.pylint_rcfile), (
    #         'Pylint config file %s does not exist.' % self.pylint_rcfile)

    def run(self):
        """
        Run command.
        """
        # command = ['pushd trex/models/liblinear/ && make clean && make && popd']
        # command = ['cd', 'trex/models/liblinear/', '&&', 'make', 'clean', '&&', 'make', '&&', 'cd', '-']
        command = 'cd trex/models/liblinear/ && make clean && make && cd -'
        # if self.pylint_rcfile:
        #     command.append('--rcfile=%s' % self.pylint_rcfile)
        #     command.append(os.getcwd())
        self.announce('Running command: %s' % str(command), level=distutils.log.INFO)
        # subprocess.check_call(command)

        os.system(command)


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def run_setup(test_xgboost=True, test_lightgbm=True, test_catboost=True):

    tests_require = ['nose']
    if test_xgboost:
        tests_require += ['xgboost']
    if test_lightgbm:
        tests_require += ['lightgbm']
    if test_catboost:
        tests_require += ['catboost']

    setup(
        name='trex',
        version=find_version("trex", "__init__.py"),
        description='Instance-based explanations for tree ensembles',
        url='',
        author='Jonathan Brophy',
        author_email='jbrophy@cs.uoregon.edu',
        license='MIT',
        packages=['trex', 'trex.models', 'trex.utility', 'trex.models.liblinear'],
        install_requires=['numpy', 'scipy', 'scikit-learn'],
        test_suite='nose.collector',
        tests_require=tests_require,
        classifiers=[
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
        ],
        cmdclass={
            'liblinearmake': LiblinearMake,
        },
        zip_safe=False
    )


def try_run_setup(**kwargs):
    """
    Fails gracefully when various install steps don't work.
    """

    try:
        run_setup(**kwargs)

    except Exception as e:
        print(str(e))

        if "xgboost" in str(e).lower():
            kwargs["test_xgboost"] = False
            print("Couldn't install XGBoost for testing!")
            try_run_setup(**kwargs)

        elif "lightgbm" in str(e).lower():
            kwargs["test_lightgbm"] = False
            print("Couldn't install LightGBM for testing!")
            try_run_setup(**kwargs)

        elif "catboost" in str(e).lower():
            kwargs["test_catboost"] = False
            print("Couldn't install CatBoost for testing!")
            try_run_setup(**kwargs)

        else:
            print("ERROR: Failed to build!")


if __name__ == "__main__":
    try_run_setup(test_xgboost=True, test_lightgbm=True, test_catboost=True)
