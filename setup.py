"""
Reference: https://github.com/slundberg/shap/blob/master/setup.py
"""
import os
import re
import codecs
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def run_setup(with_binary=True, test_xgboost=True, test_lightgbm=True, test_catboost=True):

    tests_require = ['nose']
    if test_xgboost:
        tests_require += ['xgboost']
    if test_lightgbm:
        tests_require += ['lightgbm']
    if test_catboost:
        tests_require += ['catboost']

    setup(
        name='sexee',
        version=find_version("sexee", "__init__.py"),
        description='Instance-based explanations for tree ensembles',
        url='',
        author='Jonathan Brophy',
        author_email='jbrophy@cs.uoregon.edu',
        license='MIT',
        packages=['sexee'],
        install_requires=['numpy', 'scipy', 'scikit-learn'],
        test_suite='nose.collector',
        tests_require=tests_require,
        classifiers=[
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
        ],
        zip_safe=False
    )


# setup(name='util',
#       version='0.1',
#       description='project library',
#       url='',
#       author='Jonathan Brophy',
#       author_email='jbrophy@cs.uoregon.edu',
#       license='MIT',
#       packages=['util'],
#       zip_safe=False)


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

        elif kwargs["with_binary"]:
            kwargs["with_binary"] = False
            print("WARNING: The C extension could not be compiled, sklearn tree models not supported.")
            try_run_setup(**kwargs)

        else:
            print("ERROR: Failed to build!")


if __name__ == "__main__":
    try_run_setup(with_binary=True, test_xgboost=True, test_lightgbm=True)
