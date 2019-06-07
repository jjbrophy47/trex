"""
Utility methods for sexee modules.
"""
import numpy as np


def validate_model(model):
    """Make sure the model is a supported model type."""

    model_type = str(model).split('(')[0]
    if 'RandomForestClassifier' in str(model):
        model_type = 'RandomForestClassifier'
    elif 'LGBMClassifier' in str(model):
        model_type = 'LGBMClassifier'
    elif 'CatBoostClassifier' in str(model):
        model_type = 'CatBoostClassifier'
    else:
        exit('{} model not currently supported!'.format(str(model)))

    return model_type


def parse_rf_model(model, leaves_per_tree=False, nodes_per_tree=False):
    """Returns low-level information about sklearn's RandomForestClassifier."""

    result = None

    if leaves_per_tree:
        result = np.array([tree.get_n_leaves() for tree in model.estimators_])

    elif nodes_per_tree:
        result = np.array([tree.tree_.node_count for tree in model.estimators_])

    return result


def parse_lgb_model(model, leaves_per_tree=False):
    """Returns the low-level information about the lgb model."""

    result = None

    if leaves_per_tree:
        model_dict = model.booster_.dump_model()
        result = np.array([tree_dict['num_leaves'] for tree_dict in model_dict['tree_info']])

    return result
