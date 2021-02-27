"""
Feature representation extractor for different tree ensemble models.
"""
import os
import json

import scipy
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from .util import assert_import
from .util import record_import_error
from .util import tree_model

try:
    import lightgbm
except ImportError as e:
    record_import_error("lightgbm", "LightGBM could not be imported!", e)

try:
    import xgboost
except ImportError as e:
    record_import_error("xgboost", "XGBoost could not be imported!", e)

try:
    import catboost
except ImportError as e:
    record_import_error("catboost", "CatBoost could not be imported!", e)


class TreeExtractor:

    def __init__(self, model, tree_kernel='leaf_output', sparse=False):
        """
        Extracts model-specific feature representations for data instances from a trained tree ensemble.

        Parameters
        ----------
        model: object
            Trained tree ensemble. Supported: RandomForestClassifier, GradientBoostingClassifier,
            LightGBM, XGBoost, CatBoost.
        tree_kernel: str, {'leaf_path', 'feature_path', 'leaf_output'}, (default='leaf_path')
            Type of feature representation to extract from the tree ensemble.
        sparse: bool (default=False)
            If True, feature representations are returned in a sparse format if possible.
        """
        self.model = model
        self.tree_kernel = tree_kernel
        self.sparse = sparse
        self.validate(self.tree_kernel, self.model)

    def fit_transform(self, X):
        """
        Outputs a feature representation for each x in X.

        X: 2d array-like
            Each row in X is a separate instance.

        Returns a transformation of X with the same number of rows as X.
        If encoding='leaf_path', also returns the fitted one_hot_encoder.
        """
        assert X.ndim == 2, 'X is not 2d!'

        if self.tree_kernel == 'leaf_path':
            X_feature, self.path_enc_ = self.leaf_path(X)

        elif self.tree_kernel == 'tree_output':
            X_feature = self.tree_output(X)

        elif self.tree_kernel == 'leaf_output':
            X_feature = self.leaf_output(X)

        elif self.tree_kernel == 'feature_path':
            X_feature = self.feature_path(X)

        return X_feature

    def transform(self, X):
        """
        Outputs a feature representation for each x in X.

        X: 2d array-like
            Each row in X is a separate instance.

        Returns a transformation of X with the same number of rows as X.
        """
        assert X.ndim == 2, 'X is not 2d!'

        if self.tree_kernel == 'leaf_path':
            assert self.path_enc_ is not None, 'path_enc_ is not fitted!'
            X_feature, _ = self.leaf_path(X, one_hot_enc=self.path_enc_)

        elif self.tree_kernel == 'tree_output':
            X_feature = self.tree_output(X)

        elif self.tree_kernel == 'leaf_output':
            X_feature = self.leaf_output(X)

        elif self.tree_kernel == 'feature_path':
            X_feature = self.feature_path(X)

        return X_feature

    # private
    def feature_path(self, X, one_hot_enc=None, timeit=False):
        """
        Transforms each x in X as follows:
            -A concatenation of one-hot vectors.
            -For each vector, a 1 in position i represents x traversed through node i.

        Returns a 2D array of shape (no. samples, no. nodes in ensemble).
        """
        assert X.ndim == 2, 'X is not 2d!'

        # get the leaf ids and num leaves or nodes of each tree for all instances
        if self.model_type_ == 'RandomForestClassifier':
            encoding = self.model.decision_path(X)[0].todense().astype(np.float32)

        elif self.model_type_ == 'CatBoostClassifier':
            assert_import('catboost')
            self.model.save_model('.cb.json', format='json')
            cb_dump = json.load(open('.cb.json', 'r'))
            cb_model = tree_model.CBModel(cb_dump)
            encoding = cb_model.decision_path(X, sparse=self.sparse)
            os.system('rm .cb.json')

        elif self.model_type_ == 'GradientBoostingClassifier':
            encoding = scipy.sparse.hstack([t.decision_path(X) for est in self.model.estimators_ for t in est])

        elif self.model_type_ == 'LGBMClassifier':
            assert_import('lightgbm')
            lgb_model = tree_model.LGBModel(self.model.booster_.dump_model())
            encoding = lgb_model.decision_path(X, sparse=self.sparse)

        elif self.model_type_ == 'XGBClassifier':
            assert_import('xgboost')
            xgb_model = tree_model.XGBModel(self.model._Booster.get_dump())
            encoding = xgb_model.decision_path(X, sparse=self.sparse)

        return encoding

    def feature_output(self, X, one_hot_enc=None, timeit=False):
        """
        Transforms each x in X as follows:
            -A concatenation of one-hot vectors.
            -For each vector, a 1 in position i represents x traversed through node i.
            -Replace the  at each leaf position with the actual leaf value.

        Returns a 2D array of shape (no. samples, no. nodes in ensemble).
        """
        assert X.ndim == 2, 'X is not 2d!'

        # get the leaf ids and num leaves or nodes of each tree for all instances
        if self.model_type_ == 'RandomForestClassifier':
            encoding = self.model.decision_path(X)[0].todense().astype(np.float32)  # (no. samples, no. total nodes)
            leaf_indices = self.model.apply(X)  # (no. samples, no. trees), same as `leaf_vals`
            leaf_values = np.hstack([t.predict_proba[:, 1].reshape(-1, 1) for t in self.model.estimators_])

            # substitute leaf value into the feature path encoding
            for i in range(leaf_indices.shape[0]):  # per instance
                n_prev_nodes = 0  # reset for this new instance

                for j in range(leaf_indices.shape[1]):  # per tree
                    encoding[i, n_prev_nodes + leaf_indices[i, j]] = leaf_values[i, j]
                    n_prev_nodes += self.model.estimators_[j].tree_.node_count_

        # TODO
        elif self.model_type_ == 'CatBoostClassifier':
            exit(0)
            assert_import('catboost')
            self.model.save_model('.cb.json', format='json')
            cb_dump = json.load(open('.cb.json', 'r'))
            cb_model = tree_model.CBModel(cb_dump)
            encoding = cb_model.decision_path(X, sparse=self.sparse)
            os.system('rm .cb.json')

        elif self.model_type_ == 'GradientBoostingClassifier':
            exit('{} not currently supported!'.format(str(self.model)))

        elif self.model_type_ == 'LGBMClassifier':
            exit('{} not currently supported!'.format(str(self.model)))

        elif self.model_type_ == 'XGBClassifier':
            exit('{} not currently supported!'.format(str(self.model)))

        return encoding

    def leaf_path(self, X, one_hot_enc=None):
        """
        Transforms each x in X as follows:
            -A concatenation of one-hot vectors.
            -For each vector, a 1 in position i represents x traversed to leaf i.

        Returns 2D array of shape (no. samples, no. leaves in the ensemble).
        """
        assert X.ndim == 2, 'X is not 2d!'

        # get the leaf ids and no. leaves or nodes of each tree for all instances
        if self.model_type_ == 'RandomForestClassifier':  # valid, just uses redundant features, 2x features
            leaves = self.model.apply(X)
            leaves_per_tree = np.array([tree.tree_.node_count for tree in self.model.estimators_])  # actually nodes

        elif self.model_type_ == 'CatBoostClassifier':
            assert_import('catboost')
            X_pool = catboost.Pool(self.model.numpy_to_cat(X), cat_features=self.model.get_cat_indices())
            leaves = self.model.calc_leaf_indexes(X_pool)
            leaves_per_tree = self.model.get_tree_leaf_counts()

        elif self.model_type_ == 'GradientBoostingClassifier':
            leaves = self.model.apply(X)
            leaves = leaves.reshape(leaves.shape[0], leaves.shape[1] * leaves.shape[2])  # (n_samples, n_est * n_class)
            leaves_per_tree = np.array([t.tree_.node_count for est in self.model.estimators_ for t in est])  # nodes

        elif self.model_type_ == 'LGBMClassifier':
            assert_import('lightgbm')
            leaves = self.model.predict_proba(X, pred_leaf=True)
            leaves_per_tree = np.array([tree['num_leaves'] for tree in self.model.booster_.dump_model()['tree_info']])

        elif self.model_type_ == 'XGBClassifier':
            assert_import('xgboost')
            leaves = self.model.apply(X)
            leaves_per_tree = [len(t.strip().replace('\t', '').split('\n')) for t in self.model._Booster.get_dump()]

        self.num_trees_ = len(leaves_per_tree)

        # encode leaf positions
        if one_hot_enc is None:
            categories = [np.arange(n_leaves) for n_leaves in leaves_per_tree]
            one_hot_enc = OneHotEncoder(categories=categories).fit(leaves)
        encoding = one_hot_enc.transform(leaves)

        # convert from sparse to dense
        if not self.sparse:
            encoding = np.array(encoding.todense())

        return encoding, one_hot_enc

    def leaf_output(self, X, timeit=False):
        """
        Transforms each x in X as follows:
            -A concatenation of one-hot vectors, one for each tree.
            -For each vector, a 1 in position i represents x traversed to leaf i.
            -Replace the 1 in each vector with the actual leaf value.

        Returns 2D array of shape (no. samples, no. leaves in the ensemble).
        """
        assert X.ndim == 2, 'X is not 2d!'

        if self.model_type_ == 'RandomForestClassifier':

            # get leaf ids per prediction and number of leaves per tree
            leaves = self.model.apply(X)
            leaves_per_tree = np.array([tree.tree_.node_count for tree in self.model.estimators_])  # actually nodes

            # create leaf_path encoding
            categories = [np.arange(n_leaves) for n_leaves in leaves_per_tree]
            one_hot_enc = OneHotEncoder(categories=categories).fit(leaves)
            encoding = np.array(one_hot_enc.transform(leaves).todense())

            # get leaf value of each tree for each sample
            tree_preds = np.hstack([tree.predict_proba(X)[:, 1].reshape(-1, 1) for tree in self.model.estimators_])

            # replace leaf positions with leaf values
            for i in range(encoding.shape[0]):
                tree_cnt = 0
                for j in range(encoding.shape[1]):
                    if encoding[i][j] == 1:
                        encoding[i][j] *= tree_preds[i][tree_cnt]
                        tree_cnt += 1

            self.num_trees_ = tree_preds.shape[1]

        elif self.model_type_ == 'GradientBoostingClassifier':
            exit('{} not currently supported!'.format(str(self.model)))

            one_hot_preds = [tree.predict(X) for est in self.model.estimators_ for tree in est]
            encoding = np.vstack(one_hot_preds).T
            self.num_trees_ = len(one_hot_preds)

        elif self.model_type_ == 'LGBMClassifier':
            assert_import('lightgbm')

            leaf_pos = self.model.predict_proba(X, pred_leaf=True)
            self.num_trees = leaf_pos.shape[1]

            tree_info = self.model.booster_.dump_model()['tree_info']
            n_leaves_per_tree = [tree['num_leaves'] for tree in tree_info]
            n_total_leaves = np.sum(n_leaves_per_tree)

            encoding = np.zeros((X.shape[0], n_total_leaves))

            for i in range(X.shape[0]):  # per instance
                start = 0
                for j, n_leaves in enumerate(n_leaves_per_tree):  # per tree
                    encoding[i][start + leaf_pos[i][j]] = self.model.booster_.get_leaf_output(j, leaf_pos[i][j])
                    start += n_leaves

        elif self.model_type_ == 'CatBoostClassifier':
            assert_import('catboost')

            # for multiclass, leaf_values has n_classes times leaf_counts.sum() values, why is this?
            # we only use the first segment of leaf_values: leaf_values[:leaf_counts.sum()]
            X_pool = catboost.Pool(self.model.numpy_to_cat(X), cat_features=self.model.get_cat_indices())
            leaf_pos = self.model.calc_leaf_indexes(X_pool)  # 2d (n_samples, n_trees)
            leaf_values = self.model.get_leaf_values()  # leaf values for all trees in a 1d array
            leaf_counts = self.model.get_tree_leaf_counts()  # 1d array of leaf counts for each tree

            n_total_leaves = np.sum(leaf_counts)

            encoding = np.zeros((X.shape[0], n_total_leaves))

            for i in range(X.shape[0]):  # per instance
                for j in range(leaf_pos.shape[1]):  # per tree
                    ndx = leaf_counts[:j].sum() + leaf_pos[i][j]
                    encoding[i][ndx] = leaf_values[ndx]

        elif self.model_type_ == 'XGBClassifier':
            exit('{} not currently supported!'.format(str(self.model)))

            assert_import('xgboost')
            leaves = self.model.apply(X)
            leaf_values = tree_model.parse_xgb(self.model, leaf_values=True)
            self.num_trees_ = leaves.shape[1]

            encoding = np.zeros(leaves.shape)
            for i in range(leaves.shape[0]):  # per instance
                for j in range(leaves.shape[1]):  # per tree
                    encoding[i][j] = leaf_values[j][leaves[i][j]]

        else:
            exit('leaf output encoding not supported for {}'.format(self.model_type_))

        if self.model_type_ == 'RandomForestClassifier' and self.sparse:
            encoding = scipy.sparse.csr_matrix(encoding)

        return encoding

    def tree_output(self, X, timeit=False):
        """
        Transforms each x in X as follows:
            -A vector of concatenated leaf values, one for each tree.

        Returns 2D array of shape (no. samples, no. trees).
        """
        assert X.ndim == 2, 'X is not 2d!'

        if self.model_type_ == 'RandomForestClassifier':
            probas = [tree.predict_proba(X)[:, 1].reshape(-1, 1) for tree in self.model.estimators_]
            encoding = np.hstack(probas)
            self.num_trees_ = len(probas)

        elif self.model_type_ == 'CatBoostClassifier':
            assert_import('catboost')

            # for multiclass, leaf_values has n_classes times leaf_counts.sum() values, why is this?
            # we only use the first segment of leaf_values: leaf_values[:leaf_counts.sum()]
            X_pool = catboost.Pool(self.model.numpy_to_cat(X), cat_features=self.model.get_cat_indices())
            leaves = self.model.calc_leaf_indexes(X_pool)  # 2d (n_samples, n_trees)
            leaf_values = self.model.get_leaf_values()  # leaf values for all trees in a 1d array
            leaf_counts = self.model.get_tree_leaf_counts()  # 1d array of leaf counts for each tree

            encoding = np.zeros(leaves.shape)
            for i in range(leaves.shape[0]):  # per instance
                for j in range(leaves.shape[1]):  # per tree
                    leaf_ndx = leaf_counts[:j].sum() + leaves[i][j]
                    encoding[i][j] = leaf_values[leaf_ndx]

        elif self.model_type_ == 'GradientBoostingClassifier':
            probas = [tree.predict(X) for est in self.model.estimators_ for tree in est]
            encoding = np.vstack(probas).T
            self.num_trees_ = len(probas)

        elif self.model_type_ == 'LGBMClassifier':
            assert_import('lightgbm')
            leaves = self.model.predict_proba(X, pred_leaf=True)
            encoding = np.zeros(leaves.shape)
            self.num_trees = leaves.shape[1]

            for i in range(leaves.shape[0]):  # per instance
                for j in range(leaves.shape[1]):  # per tree
                    encoding[i][j] = self.model.booster_.get_leaf_output(j, leaves[i][j])

        elif self.model_type_ == 'XGBClassifier':
            assert_import('xgboost')
            leaves = self.model.apply(X)
            leaf_values = tree_model.parse_xgb(self.model, leaf_values=True)
            self.num_trees_ = leaves.shape[1]

            encoding = np.zeros(leaves.shape)
            for i in range(leaves.shape[0]):  # per instance
                for j in range(leaves.shape[1]):  # per tree
                    encoding[i][j] = leaf_values[j][leaves[i][j]]

        else:
            exit('leaf output encoding not supported for {}'.format(self.model_type_))

        if self.model_type_ == 'RandomForestClassifier' and self.sparse:
            encoding = scipy.sparse.csr_matrix(encoding)

        return encoding

    def validate(self, tree_kernel, model):
        """
        Validate model inputs.
        """

        # check tre kernel
        tree_kernels = ['leaf_path', 'tree_output', 'leaf_output', 'feature_path']
        assert self.tree_kernel in tree_kernels, '{} not supported!'.format(self.tree_kernel)

        # check model
        model_types = ['RandomForestClassifier', 'GradientBoostingClassifier',
                       'LGBMClassifier', 'CatBoostClassifier', 'XGBClassifier']
        valid = False
        for model_type in model_types:
            if model_type in str(self.model):
                self.model_type_ = model_type
                valid = True
        assert valid, '{} not currently supported!'.format(str(self.model))
