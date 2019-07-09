"""
Feature representation extractor for different tree ensemble models.
"""
import time
import operator

import scipy
import catboost
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from . import util


class TreeExtractor:

    def __init__(self, model, encoding='tree_path', sparse=False):
        """
        Extracts model-specific feature representations for data instances from a trained tree ensemble.

        Parameters
        ----------
        model: object
            Trained tree ensemble. Supported: RandomForestClassifier, LightGBM. Unsupported: XGBoost, CatBoost.
        encoding: str, {'tree_path', 'tree_output'}, (default='tree_path')
            Type of feature representation to extract from the tree ensemble.
        sparse: bool (default=False)
            If True, feature representations are returned in a sparse format if possible.
        """
        self.model_type_ = util.validate_model(model)
        assert encoding in ['tree_path', 'tree_output'], '{} encoding not supported!'.format(encoding)

        self.model = model
        self.encoding = encoding
        self.sparse = sparse

    def fit_transform(self, X):
        """
        Outputs a feature representation for each x in X.

        Parameters
        ----------
        X: 2d array-like
            Each row in X is a separate instance.

        Returns
        -------
        Feature representation of X with the same number of rows as X.
        If encoding='tree_path', also returns the fitted one_hot_encoder.
        """
        assert X.ndim == 2, 'X is not 2d!'

        if self.encoding == 'tree_path':
            X_feature, self.path_enc_ = self._tree_path_encoding(X)

        elif self.encoding == 'tree_output':
            X_feature = self._tree_output_encoding(X)

        return X_feature

    def transform(self, X):
        """
        Outputs a feature representation for each x in X.

        Parameters
        ----------
        X: 2d array-like
            Each row in X is a separate instance.

        Returns
        -------
        Feature representation of X with the same number of rows as X.
        """
        assert X.ndim == 2, 'X is not 2d!'

        if self.encoding == 'tree_path':
            assert self.path_enc_ is not None, 'path_enc_ is not fitted!'
            X_feature, _ = self._tree_path_encoding(X, one_hot_enc=self.path_enc_)

        elif self.encoding == 'tree_output':
            X_feature = self._tree_output_encoding(X)

        return X_feature

    def _tree_path_encoding(self, X, one_hot_enc=None, timeit=False):
        """
        Encodes each x in X as a binary vector whose length is equal to the number of
        leaves or nodes in the ensemble, with 1's representing the instance ending at that leaf,
        0 otherwise.
        """
        assert X.ndim == 2, 'X is not 2d!'
        start = time.time()

        # get the leaf ids and num leaves or nodes of each tree for all instances
        if self.model_type_ == 'RandomForestClassifier':
            leaves = self.model.apply(X)
            leaves_per_tree = np.array([tree.tree_.node_count for tree in self.model.estimators_])  # actually nodes

        elif self.model_type_ == 'GradientBoostingClassifier':
            leaves = self.model.apply(X)
            leaves = leaves.reshape(leaves.shape[0], leaves.shape[1] * leaves.shape[2])  # (n_samples, n_est * n_class)
            leaves_per_tree = np.array([t.tree_.node_count for est in self.model.estimators_ for t in est])  # nodes

        elif self.model_type_ == 'LGBMClassifier':
            leaves = self.model.predict_proba(X, pred_leaf=True)
            leaves_per_tree = np.array([tree['num_leaves'] for tree in self.model.booster_.dump_model()['tree_info']])

        elif self.model_type_ == 'CatBoostClassifier':
            leaves = self.model.calc_leaf_indexes(catboost.Pool(X))
            leaves_per_tree = self.model.get_tree_leaf_counts()

        elif self.model_type_ == 'XGBClassifier':
            leaves = self.model.apply(X)
            leaves_per_tree = [len(t.strip().replace('\t', '').split('\n')) for t in self.model._Booster.get_dump()]

        self.num_trees_ = len(leaves_per_tree)

        if one_hot_enc is None:
            categories = [np.arange(n_leaves) for n_leaves in leaves_per_tree]
            one_hot_enc = OneHotEncoder(categories=categories).fit(leaves)

        encoding = one_hot_enc.transform(leaves)
        if not self.sparse:
            encoding = np.array(encoding.todense())

        if timeit:
            print('leaf_path encoding time: {:.3f}'.format(time.time() - start))

        return encoding, one_hot_enc

    def _feature_path_encoding(self, X, one_hot_enc=None, timeit=False):
        """
        Encodes each x in X as a binary vector whose length is equal to the number of
        nodes in the ensemble, with 1's representing the instance traversed through that node,
        0 otherwise.
        Returns
        -------
        2D array of shape (n_samples, n_nodes) where n_nodes is the number of total nodes across all trees.
        """
        assert X.ndim == 2, 'X is not 2d!'
        start = time.time()

        # get the leaf ids and num leaves or nodes of each tree for all instances
        if self.model_type_ == 'RandomForestClassifier':
            encoding = self.model.decision_path(X)[0]

        elif self.model_type_ == 'GradientBoostingClassifier':
            encoding = scipy.sparse.hstack([t.decision_path(X) for est in self.model.estimators_ for t in est])

        elif self.model_type_ == 'LGBMClassifier':
            lgb_model = LGBModel(self.model.booster_.dump_model())
            encoding = lgb_model.decision_path(X, sparse=self.sparse)
            print(encoding)

        elif self.model_type_ == 'CatBoostClassifier':
            # leaves = self.model.calc_leaf_indexes(catboost.Pool(X))
            # leaves_per_tree = self.model.get_tree_leaf_counts()
            pass

        elif self.model_type_ == 'XGBClassifier':
            # leaves = self.model.apply(X)
            # leaves_per_tree = [len(t.strip().replace('\t', '').split('\n')) for t in self.model._Booster.get_dump()]
            pass

        # self.num_trees_ = len(leaves_per_tree)

        # if one_hot_enc is None:
        #     categories = [np.arange(n_leaves) for n_leaves in leaves_per_tree]
        #     one_hot_enc = OneHotEncoder(categories=categories).fit(leaves)

        # encoding = one_hot_enc.transform(leaves)
        # if not self.sparse:
        #     encoding = np.array(encoding.todense())

        # if timeit:
        #     print('feature_path encoding time: {:.3f}'.format(time.time() - start))

        # return encoding, one_hot_enc

    def _tree_output_encoding(self, X, timeit=False):
        """
        Encodes each x in X as a concatenation of one-hot encodings, one for each tree.
        Each one-hot encoding represents the class or output at the leaf x traversed to.
        All one-hot encodings are concatenated, to get a vector of size n_trees * n_classes.
        Returns
        -------
        2D array of shape (n_samples, n_trees) where n_trees is multiplied by n_classes if multiclass.
        """
        assert X.ndim == 2, 'X is not 2d!'
        start = time.time()

        if self.model_type_ == 'RandomForestClassifier':
            one_hot_preds = [tree.predict_proba(X) for tree in self.model.estimators_]
            encoding = np.hstack(one_hot_preds)
            self.num_trees_ = len(one_hot_preds)

        elif self.model_type_ == 'GradientBoostingClassifier':
            one_hot_preds = [tree.predict(X) for est in self.model.estimators_ for tree in est]
            encoding = np.vstack(one_hot_preds).T
            self.num_trees_ = len(one_hot_preds)

        elif self.model_type_ == 'LGBMClassifier':
            leaves = self.model.predict_proba(X, pred_leaf=True)
            encoding = np.zeros(leaves.shape)
            self.num_trees = leaves.shape[1]

            for i in range(leaves.shape[0]):  # per instance
                for j in range(leaves.shape[1]):  # per tree
                    encoding[i][j] = self.model.booster_.get_leaf_output(j, leaves[i][j])

        elif self.model_type_ == 'CatBoostClassifier':

            # for multiclass, leaf_values has n_classes times leaf_counts.sum() values, why is this?
            # we only use the first segment of leaf_values: leaf_values[:leaf_counts.sum()]
            leaves = self.model.calc_leaf_indexes(catboost.Pool(X))  # 2d (n_samples, n_trees)
            leaf_values = self.model.get_leaf_values()  # leaf values for all trees in a 1d array
            leaf_counts = self.model.get_tree_leaf_counts()  # 1d array of leaf counts for each tree

            encoding = np.zeros(leaves.shape)
            for i in range(leaves.shape[0]):  # per instance
                for j in range(leaves.shape[1]):  # per tree
                    leaf_ndx = leaf_counts[:j].sum() + leaves[i][j]
                    encoding[i][j] = leaf_values[leaf_ndx]

        elif self.model_type_ == 'XGBClassifier':
            leaves = self.model.apply(X)
            leaf_values = util.parse_xgb(self.model, leaf_values=True)
            self.num_trees_ = leaves.shape[1]

            encoding = np.zeros(leaves.shape)
            for i in range(leaves.shape[0]):  # per instance
                for j in range(leaves.shape[1]):  # per tree
                    encoding[i][j] = leaf_values[j][leaves[i][j]]

        else:
            exit('tree output encoding not supported for {}'.format(self.model_type_))

        if self.model_type_ == 'RandomForestClassifier' and self.sparse:
            encoding = scipy.sparse.csr_matrix(encoding)

        if timeit:
            print('output encoding time: {:.3f}'.format(time.time() - start))

        return encoding


class LGBModel:
    """
    Creates a data structure from a dictionary representation of a LightGBM mdoel.
    """

    def __init__(self, model_dump):
        self.trees_ = [LGBTree(tree_dict) for tree_dict in model_dump['tree_info']]

    def __str__(self):
        s = ''
        for i, tree in enumerate(self.trees_):
            s += '\n\n\nTree ({})'.format(i)
            s += tree.__str__()
        return s

    def decision_path(self, X, sparse=False):
        """
        X : 2d array-like
            Input with shape (n_samples, n_features)
        sparse : bool (default=False)
            If True, returns a sparse matrix of the result.
        Returns
        -------
        2d array-like of encoding paths of each instance through all trees
        with shape (n_samples, n_nodes) where n_nodes is the total number of
        nodes in all trees.
        """
        assert X.ndim == 2, 'X is not 2d!'
        tree_encodings = [tree.decision_path(X) for tree in self.trees_]
        model_encoding = np.hstack(tree_encodings)
        if sparse:
            model_encoding = scipy.sparse.csr_matrix(model_encoding)
        return model_encoding


class LGBTree:
    """
    Creates a data structure from a dictionary representation of a LightGBM tree.
    """

    def __init__(self, tree_dump):
        self.root_, self.n_nodes_ = self._parse_tree(tree_dump['tree_structure'])

    def decision_path(self, X):
        """
        X : 2d array-like
            Input with shape (n_samples, n_features)
        Returns
        -------
        2d array-like of encoding paths of each instance through the tree
        with shape (n_samples, n_nodes) where n_nodes is the number of nodes in the tree.
        """
        assert X.ndim == 2, 'X is not 2d!'
        node_ndx = 0
        encoding = np.zeros((len(X), self.n_nodes_))
        encoding[:, node_ndx] = 1  # all instances go through the root node

        if self.root_.node_type == 'leaf':
            return encoding

        left_indices, right_indices = self._node_indices(X, self.root_)
        traverse = [(self.root_.right, right_indices), (self.root_.left, left_indices)]

        while len(traverse) > 0:
            node_ndx += 1
            node, indices = traverse.pop()
            encoding[indices, node_ndx] = 1

            if node.node_type == 'leaf':
                continue

            # split indices based on threshold
            # TODO: could be more efficient if X were indexed
            left_indices, right_indices = self._node_indices(X, node)
            left_indices = np.intersect1d(left_indices, indices)
            right_indices = np.intersect1d(right_indices, indices)
            traverse.append((node.right, right_indices))
            traverse.append((node.left, left_indices))

        return encoding

    def _node_indices(self, X, node):
        indices = np.arange(len(X))
        left_indices = np.where(node.op(X[:, node.feature], node.threshold))[0]
        right_indices = np.setxor1d(indices, left_indices)
        return left_indices, right_indices

    def _parse_tree(self, structure):
        root = self._get_node(structure)
        n_nodes = 1

        # tree does not have any splits
        if root.node_type == 'leaf':
            return root

        traverse = [(root, structure['right_child'], 'right'), (root, structure['left_child'], 'left')]

        while len(traverse) > 0:
            n_nodes += 1
            parent_node, child_structure, child_position = traverse.pop()
            child_node = self._get_node(child_structure)
            parent_node.set_child(child_node, child_position)

            if child_node.node_type == 'split':
                traverse.append((child_node, child_structure['right_child'], 'right'))
                traverse.append((child_node, child_structure['left_child'], 'left'))

        return root, n_nodes

    def __str__(self):

        node_ndx = 0
        s = '\n\nRoot ({})'.format(node_ndx)
        s += self.root_.__str__()

        if self.root_.node_type == 'split':
            traverse = [(self.root_.right, 'right'), (self.root_.left, 'left')]
            while len(traverse) > 0:
                child_node, child_position = traverse.pop()
                node_ndx += 1
                s += '\n\nNode ({}, {})'.format(child_position, node_ndx)
                s += child_node.__str__()

                if child_node.node_type == 'split':
                    traverse.append((child_node.right, 'right'))
                    traverse.append((child_node.left, 'left'))

        return s

    def _get_node(self, structure):

        if 'split_index' in structure:
            feature = int(structure['split_feature'])
            threshold = float(structure['threshold'])
            decision_type = self._get_operator(structure['decision_type'])
            node = Node(node_type='split', feature=feature, threshold=threshold, decision_type=decision_type)

        elif 'leaf_index' in structure:
            node = Node(node_type='leaf')

        else:
            assert 'leaf_value' in structure
            node = Node(node_type='leaf')

        return node

    def _get_operator(self, decision_type):

        if decision_type == '<=':
            result = operator.le
        else:
            exit('unknown decision_type: {}'.format(decision_type))

        return result


class Node:

    def __init__(self, node_type, feature=None, threshold=None, decision_type=None):
        self.node_type = node_type
        self.feature = feature
        self.threshold = threshold
        self.op = decision_type

    def __str__(self):
        s = '\ntype: {}'.format(self.node_type)
        if self.node_type == 'split':
            s += '\nsplit feature: {}'.format(self.feature)
            s += '\nthrehsold: {}'.format(self.threshold)
            s += '\noperator: {}'.format(self.op)
        return s

    def set_child(self, child_node, child_position):

        if child_position == 'left':
            self.left = child_node
        elif child_position == 'right':
            self.right = child_node
        else:
            exit('unrecognized position: {}'.format(child_position))
