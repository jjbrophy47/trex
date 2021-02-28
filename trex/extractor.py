"""
Feature representation extractor for different tree ensemble models.
"""
import os
import json

import numpy as np
import catboost

from .util import tree_model


class TreeExtractor:

    def __init__(self, model, tree_kernel='leaf_output'):
        """
        Extracts model-specific feature representations for data instances from a trained tree ensemble.

        Parameters
        ----------
        model: object
            Learned tree ensemble; models currently supported: RandomForestClassifier and CatBoost.
        tree_kernel: str, {'feature_path', 'feature_output', 'leaf_path', 'leaf_output', 'tree_output'}.
                     (default='leaf_path')
            Type of feature representation to extract from the tree ensemble.
        """
        self.model = model
        self.tree_kernel = tree_kernel
        self.validate(self.tree_kernel, self.model)

    def transform(self, X):
        """
        Outputs a feature representation for each x in X.

        X: 2d array-like
            Each row in X is a separate instance.

        Returns a transformation of X with the same number of rows as X.
        """
        assert X.ndim == 2, 'X is not 2d!'

        if self.tree_kernel == 'feature_path':
            X_feature = self.feature_path(X)

        elif self.tree_kernel == 'feature_output':
            X_feature = self.feature_output(X)

        elif self.tree_kernel == 'leaf_path':
            X_feature = self.leaf_path(X)

        elif self.tree_kernel == 'leaf_output':
            X_feature = self.leaf_output(X)

        elif self.tree_kernel == 'tree_output':
            X_feature = self.tree_output(X)

        else:
            raise ValueError('tree_kernel {} unknown!'.format(self.tree_kernel))

        return X_feature

    # private
    def feature_path(self, X):
        """
        Transforms each x in X as follows:
            -A concatenation of one-hot vectors.
            -For each vector, a 1 in position i represents x traversed through node i.

        Returns a 2D array of shape (no. samples, no. nodes in ensemble).
        """
        assert X.ndim == 2, 'X is not 2d!'

        # RF
        if self.model_type_ == 'RandomForestClassifier':
            encoding = self.model.decision_path(X)[0].todense()

        # CatBoost
        # WARNING: Should only be used if NOT using cat. features.
        #          CatBoost transforms categorical features to numeric internally,
        #          but does not provide any APIs that do this transformation;
        #          thus, the raw JSON representation cannot be used with cat.
        #          features since it assumes transformed cat. features.
        elif self.model_type_ == 'CatBoostClassifier':
            cb_model = self.get_cb_model(self.model)
            encoding, _, _ = cb_model.decision_path(X)
            self.cleanup_cb_model()

        # GBM
        elif self.model_type_ == 'GradientBoostingClassifier':
            exit('{} not currently supported!'.format(str(self.model)))

        # LightGBM
        elif self.model_type_ == 'LGBMClassifier':
            exit('{} not currently supported!'.format(str(self.model)))

        # XGBoost
        elif self.model_type_ == 'XGBClassifier':
            exit('{} not currently supported!'.format(str(self.model)))

        # convert to np.float32
        encoding = encoding.astype(np.float32)

        return encoding

    def feature_output(self, X):
        """
        Transforms each x in X as follows:
            -A concatenation of one-hot vectors.
            -For each vector, a 1 in position i represents x traversed through node i.
            -Replace the  at each leaf position with the actual leaf value.

        Returns a 2D array of shape (no. samples, no. nodes in ensemble).
        """
        assert X.ndim == 2, 'X is not 2d!'

        # RF
        if self.model_type_ == 'RandomForestClassifier':

            # FeaturePath encoding
            encoding = self.feature_path(X)

            # extract leaf indices traversed to and all leaf value
            leaf_indices = self.model.apply(X)  # in terms of all nodes in each tree, (no. samples, no. trees)
            leaf_values = np.hstack([t.predict_proba(X)[:, 1].reshape(-1, 1) for t in self.model.estimators_])

            # substitute leaf value into the feature path encoding
            for i in range(leaf_indices.shape[0]):  # per instance
                n_prev_nodes = 0  # reset for this new instance

                for j in range(leaf_indices.shape[1]):  # per tree
                    encoding[i, n_prev_nodes + leaf_indices[i, j]] = leaf_values[i, j]
                    n_prev_nodes += self.model.estimators_[j].tree_.node_count

        # CatBoost
        # WARNING: Should only be used if NOT using cat. features.
        #          CatBoost transforms categorical features to numeric internally,
        #          but does not provide any APIs that do this transformation;
        #          thus, the raw JSON representation cannot be used with cat.
        #          features since it assumes transformed cat. features.
        elif self.model_type_ == 'CatBoostClassifier':
            cb_model = self.get_cb_model(self.model)

            # extract FeaturePath encoding, and node indices traversed to, and no. nodes per tree
            encoding, leaf_indices, node_counts = cb_model.decision_path(X)

            # extract leaf indices traversed to, all leaf values, and no. leaves per tree
            leaf_value_indices = self.model.calc_leaf_indexes(X)  # in terms of only leaf nodes in each tree
            leaf_values = self.model.get_leaf_values()
            leaf_counts = self.model.get_tree_leaf_counts()

            # substitute leaf value into the feature path encoding
            for i in range(leaf_indices.shape[0]):  # per instance
                n_prev_nodes = 0  # reset no. nodes
                n_prev_leaves = 0  # reset no. leaves

                for j in range(leaf_indices.shape[1]):  # per tree
                    leaf_val = leaf_values[n_prev_leaves + leaf_value_indices[i, j]]
                    encoding[i, n_prev_nodes + leaf_indices[i, j]] = leaf_val

                    # update trackers
                    n_prev_nodes += node_counts[j]
                    n_prev_leaves += leaf_counts[j]

            self.cleanup_cb_model()

        elif self.model_type_ == 'GradientBoostingClassifier':
            exit('{} not currently supported!'.format(str(self.model)))

        elif self.model_type_ == 'LGBMClassifier':
            exit('{} not currently supported!'.format(str(self.model)))

        elif self.model_type_ == 'XGBClassifier':
            exit('{} not currently supported!'.format(str(self.model)))

        # convert to np.float32
        encoding = encoding.astype(np.float32)

        return encoding

    def leaf_path(self, X):
        """
        Transforms each x in X as follows:
            -A concatenation of one-hot vectors.
            -For each vector, a 1 in position i represents x traversed to leaf i.

        Returns 2D array of shape (no. samples, no. leaves in the ensemble).
        """
        assert X.ndim == 2, 'X is not 2d!'

        # RF
        # NOTE: Shape=(no. samples, no. nodes in the ensemble), still valid though, uses 2x redundant features
        if self.model_type_ == 'RandomForestClassifier':
            leaf_indices = self.model.apply(X)  # in terms of all nodes in each tree
            node_counts = np.array([tree.tree_.node_count for tree in self.model.estimators_])

            # construct encoding
            encoding = np.zeros((X.shape[0], node_counts.sum()))

            # substitute leaves traversed to with a 1
            for i in range(encoding.shape[0]):  # per instance
                n_prev_nodes = 0

                for j in range(leaf_indices.shape[1]):  # per tree
                    encoding[i][n_prev_nodes + leaf_indices[i][j]] = 1
                    n_prev_nodes += node_counts[j]

        # CatBoost
        elif self.model_type_ == 'CatBoostClassifier':

            # extract leaf indices traversed to and no. leaves per tree
            X_pool = catboost.Pool(self.model.numpy_to_cat(X), cat_features=self.model.get_cat_indices())
            leaf_indices = self.model.calc_leaf_indexes(X_pool)  # in terms of only leaf nodes in each tree
            leaf_counts = self.model.get_tree_leaf_counts()

            # construct encoding
            encoding = np.zeros((X.shape[0], leaf_counts.sum()))

            # replace leaves traversed to with a 1
            for i in range(encoding.shape[0]):  # per instance
                n_prev_leaves = 0

                for j in range(leaf_indices.shape[1]):  # per tree
                    encoding[i][n_prev_leaves + leaf_indices[i][j]] = 1
                    n_prev_leaves += leaf_counts[j]

        # GBM
        elif self.model_type_ == 'GradientBoostingClassifier':
            exit('{} not currently supported!'.format(str(self.model)))

        # LightGBM
        elif self.model_type_ == 'LGBMClassifier':
            exit('{} not currently supported!'.format(str(self.model)))

        # XGBoost
        elif self.model_type_ == 'XGBClassifier':
            exit('{} not currently supported!'.format(str(self.model)))

        # convert to np.float32
        encoding = encoding.astype(np.float32)

        return encoding

    def leaf_output(self, X, timeit=False):
        """
        Transforms each x in X as follows:
            -A concatenation of one-hot vectors, one for each tree.
            -For each vector, a 1 in position i represents x traversed to leaf i.
            -Replace the 1 in each vector with the actual leaf value.

        Returns 2D array of shape (no. samples, no. leaves in the ensemble).
        """
        assert X.ndim == 2, 'X is not 2d!'

        # RF
        if self.model_type_ == 'RandomForestClassifier':

            # LeafPath encoding
            encoding = self.leaf_path(X)

            # get leaf value of each tree for each sample
            probas = np.hstack([tree.predict_proba(X)[:, 1].reshape(-1, 1) for tree in self.model.estimators_])

            # replace leaf positions with leaf values
            for i in range(encoding.shape[0]):  # per instance
                tree_cnt = 0  # reset tree count

                for j in range(encoding.shape[1]):  # per tree
                    if encoding[i][j] == 1:
                        encoding[i][j] *= probas[i][tree_cnt]
                        tree_cnt += 1

        # CatBoost
        elif self.model_type_ == 'CatBoostClassifier':

            # extract leaf indices traversed to, all leaf values, and no. leaves per tree
            X_pool = catboost.Pool(self.model.numpy_to_cat(X), cat_features=self.model.get_cat_indices())
            leaf_value_indices = self.model.calc_leaf_indexes(X_pool)  # 2d, in terms of only leaf nodes in each tree
            leaf_values = self.model.get_leaf_values()  # 1d, leaf values for all trees in a 1d array
            leaf_counts = self.model.get_tree_leaf_counts()  # 1d, leaf counts for each tree

            # construct result
            encoding = np.zeros((X.shape[0], leaf_counts.sum()))

            # substitute leaf value in the LeafPath encoding
            for i in range(X.shape[0]):  # per instance
                n_prev_leaves = 0  # reset no. leaves

                for j in range(leaf_value_indices.shape[1]):  # per tree
                    ndx = n_prev_leaves + leaf_value_indices[i][j]
                    encoding[i][ndx] = leaf_values[ndx]
                    n_prev_leaves += leaf_counts[j]

        # GBM
        elif self.model_type_ == 'GradientBoostingClassifier':
            exit('{} not currently supported!'.format(str(self.model)))

        # LightGBM
        elif self.model_type_ == 'LGBMClassifier':
            exit('{} not currently supported!'.format(str(self.model)))

        # XGBoost
        elif self.model_type_ == 'XGBClassifier':
            exit('{} not currently supported!'.format(str(self.model)))

        else:
            exit('leaf output encoding not supported for {}'.format(self.model_type_))

        # convert to np.float32
        encoding = encoding.astype(np.float32)

        return encoding

    def tree_output(self, X):
        """
        Transforms each x in X as follows:
            -A vector of concatenated leaf values, one for each tree.

        Returns 2D array of shape (no. samples, no. trees).
        """
        assert X.ndim == 2, 'X is not 2d!'

        # RF
        if self.model_type_ == 'RandomForestClassifier':
            encoding = np.hstack([tree.predict_proba(X)[:, 1].reshape(-1, 1) for tree in self.model.estimators_])

        # CatBoost
        elif self.model_type_ == 'CatBoostClassifier':

            # extract leaf indices traversed to, all leaf values, and no. leaves per tree
            X_pool = catboost.Pool(self.model.numpy_to_cat(X), cat_features=self.model.get_cat_indices())
            leaf_value_indices = self.model.calc_leaf_indexes(X_pool)  # 2d (no. samples, no. trees)
            leaf_values = self.model.get_leaf_values()  # leaf values for all trees in a 1d array
            leaf_counts = self.model.get_tree_leaf_counts()  # 1d array of leaf counts for each tree

            # substitue leaf value for each leaf of each tree
            encoding = np.zeros(leaf_value_indices.shape)

            for i in range(encoding.shape[0]):  # per instance
                n_prev_leaves = 0  # reset no. leaves

                for j in range(encoding.shape[1]):  # per tree
                    encoding[i][j] = leaf_values[n_prev_leaves + leaf_value_indices[i, j]]
                    n_prev_leaves += leaf_counts[j]

        # GBM
        elif self.model_type_ == 'GradientBoostingClassifier':
            exit('{} not currently supported!'.format(str(self.model)))

        # LightGBM
        elif self.model_type_ == 'LGBMClassifier':
            exit('{} not currently supported!'.format(str(self.model)))

        # XGBoost
        elif self.model_type_ == 'XGBClassifier':
            exit('{} not currently supported!'.format(str(self.model)))

        else:
            exit('leaf output encoding not supported for {}'.format(self.model_type_))

        # convert to np.float32
        encoding = encoding.astype(np.float32)

        return encoding

    def validate(self, tree_kernel, model):
        """
        Validate model inputs.
        """

        # check tree kernel
        tree_kernels = ['feature_path', 'feature_output', 'leaf_path', 'leaf_output', 'tree_output']
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

    def get_cb_model(self, model, fn='.cb.json'):
        """
        Return custom representation of a CatBoost model.
        """
        self.model.save_model(fn, format='json')
        cb_dump = json.load(open(fn, 'r'))
        cb_model = tree_model.CBModel(cb_dump)
        return cb_model

    def cleanup_cb_model(self, fn='.cb.json'):
        """
        Remove CatBoost model JSON file.
        """
        os.remove(fn)
