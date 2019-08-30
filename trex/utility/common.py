"""
A place to record import errors.
Rference: https://github.com/slundberg/shap/blob/master/shap/common.py
"""

import_errors = {}


def assert_import(package_name):
    global import_errors
    if package_name in import_errors:
        msg, e = import_errors[package_name]
        print(msg)
        raise e


def record_import_error(package_name, msg, e):
    global import_errors
    import_errors[package_name] = (msg, e)
