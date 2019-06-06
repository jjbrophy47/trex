"""
Feature representation extractor for different tree ensemble models.
"""
import time
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class TreeExtractor:

    def __init__(self, model, encoding='tree_path'):
        """
        Extracts model-specific feature representations for data instances from a trained tree ensemble.

        Parameters
        ----------
        model: object
            Trained tree ensemble. Supported: RandomForestClassifier, LightGBM.
            Unsupported: XGBoost, CatBoost.
        encoding: str, {'tree_path', 'tree_output'}, (default='tree_path')
            Type of feature representation to extract from the tree ensemble.
        """
        model_type_ = str(model).split('(')[0]

        assert model_type_ in ['RandomForestClassifier', 'LGBMClassifier'], '{} not supported!'.format(model_type_)
        assert encoding in ['tree_path', 'tree_output'], '{} encoding not supported!'.format(encoding)

        self.model = model
        self.encoding = encoding

    def fit_transform(self, X):
        assert X.ndim == 2, 'X is not 2d!'

        if self.encoding == 'tree_path':
            X_feature, self.path_enc_ = tree_path_encoding(self.model, X)

        elif self.encoding == 'tree_output':
            X_feature = tree_output_encoding(self.model, X)

        return X_feature

    def transform(self, X):
        assert X.ndim == 2, 'X is not 2d!'

        if self.encoding == 'tree_path':
            assert self.path_enc_ is not None, 'path_enc_ is not fitted!'
            X_feature, _ = tree_path_encoding(self.model, X, one_hot_enc=self.path_enc_)

        elif self.encoding == 'tree_output':
            X_feature = tree_output_encoding(self.model, X)

        return X_feature


# utility methods
def tree_path_encoding(model, X, one_hot_enc=None, to_dense=True, timeit=False):
    """
    Encodes each x in X as a binary vector whose length is equal to the number of
    leaves or nodes in the ensemble, with 1's representing the instance ending at that leaf,
    0 otherwise.
    """
    assert X.ndim == 2, 'X is not 2d!'
    start = time.time()

    # get the leaf ids and num leaves or nodes of each tree for all instances
    if str(model).startswith('RandomForestClassifier'):
        leaves = model.apply(X)
        leaves_per_tree = _parse_rf_model(model, nodes_per_tree=True)  # actually using nodes, could refine

    elif str(model).startswith('LGBMClassifier'):
        leaves = model.predict_proba(X, pred_leaf=True)
        leaves_per_tree = _parse_lgb_model(model, leaves_per_tree=True)

    if one_hot_enc is None:

        # make sure all leaves have been seen at least once
        assert np.all(np.max(leaves, axis=0) + 1 == leaves_per_tree), 'lgb leaves do not match max leaves found'
        one_hot_enc = OneHotEncoder(categories='auto').fit(leaves)

    encoding = one_hot_enc.transform(leaves)
    if to_dense:
        encoding = np.array(encoding.todense())

    if timeit:
        print('path encoding time: {:.3f}'.format(time.time() - start))

    return encoding, one_hot_enc


def tree_output_encoding(model, X):
    """
    Encodes each x in X as a concatenation of one-hot encodings, one for each tree.
    Each one-hot encoding represents the class or output at the leaf x traversed to.
    All one-hot encodings are concatenated, to get a vector of size n_trees * n_classes.
    """
    assert X.ndim == 2, 'X is not 2d!'
    start = time.time()

    if str(model).startswith('RandomForestClassifier'):
        one_hot_preds = [tree.predict_proba(X) for tree in model.estimators_]
        encoding = np.hstack(one_hot_preds)

    # DO NOT USE! An instance can be more similar with other instances than itself, since these are leaf values
    elif str(model).startswith('LGBMClassifier'):
        exit('tree output encoding for LGB not good, do not use!')
        leaves = model.predict_proba(X, pred_leaf=True)
        encoding = np.zeros(leaves.shape)

        for i in range(leaves.shape[0]):  # per instance
            for j in range(leaves.shape[1]):  # per tree
                encoding[i][j] = model.booster_.get_leaf_output(j, leaves[i][j])

    else:
        exit('model {} not implemented!'.format(str(model)))

    print('output encoding time: {:.3f}'.format(time.time() - start))
    return encoding


# private methods
def _parse_rf_model(model, leaves_per_tree=False, nodes_per_tree=False):
    """Returns low-level information about sklearn's RandomForestClassifier."""

    result = None

    if leaves_per_tree:
        result = np.array([tree.get_n_leaves() for tree in model.estimators_])

    elif nodes_per_tree:
        result = np.array([tree.tree_.node_count for tree in model.estimators_])

    return result


def _parse_lgb_model(model, leaves_per_tree=False):
    """Returns the low-level information about the lgb model."""

    result = None

    if leaves_per_tree:
        model_dict = model.booster_.dump_model()
        result = np.array([tree_dict['num_leaves'] for tree_dict in model_dict['tree_info']])

    return result
