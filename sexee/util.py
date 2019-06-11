"""
Utility methods for sexee modules.
"""
import numpy as np


def validate_model(model):
    """Make sure the model is a supported model type."""

    model_type = str(model).split('(')[0]

    if 'RandomForestClassifier' in str(model):
        model_type = 'RandomForestClassifier'
    elif 'GradientBoostingClassifier' in str(model):
        model_type = 'GradientBoostingClassifier'
    elif 'LGBMClassifier' in str(model):
        model_type = 'LGBMClassifier'
    elif 'CatBoostClassifier' in str(model):
        model_type = 'CatBoostClassifier'
    elif 'XGBClassifier' in str(model):
        model_type = 'XGBClassifier'
    else:
        exit('{} model not currently supported!'.format(str(model)))

    return model_type


def parse_xgb(model, nodes_per_tree=False, leaf_values=False):
    """Parses the xgb raw string data for leaf information."""

    assert validate_model(model) == 'XGBClassifier'

    if nodes_per_tree:
        dump = model._Booster.get_dump()
        nodes_per_tree = [len(tree.strip().replace('\t', '').split('\n')) for tree in dump]
        result = nodes_per_tree

    elif leaf_values:

        trees = {}
        for i, tree in enumerate(model._Booster.get_dump()):
            nodes = tree.strip().replace('\t', '').split('\n')
            trees[i] = np.array([float(node.split('=')[1]) if 'leaf' in node else 0.0 for node in nodes])

        assert len(trees) == model.n_estimators * model.n_classes_, 'trees len does not equal num total trees!'
        return trees

    else:
        exit('No info specified to extract!')

    return result
