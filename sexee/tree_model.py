"""
Data structures that parses raw dictionary or text versions of different tree ensemble models,
and makes them easy to extract features from.
"""
import re
import operator

import scipy
import numpy as np


class LGBModel:
    """
    Creates a data structure from a dictionary representation of a LightGBM model.
    """

    def __init__(self, model_dump):
        self.trees_ = [Tree(tree_dict, tree_type='lgb') for tree_dict in model_dump['tree_info']]

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


class XGBModel:
    """
    Creates a data structure from a dictionary representation of an XGBoost model.
    """

    def __init__(self, model_dump):
        self.trees_ = [Tree(tree_str, tree_type='xgb') for tree_str in model_dump]

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


class CBModel:
    """
    Creates a data structure from a dictionary representation of an XGBoost model.
    """

    def __init__(self, model_dump):
        self.trees_ = [Tree(tree_str, tree_type='cb') for tree_str in model_dump['oblivious_trees']]

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


class Tree:
    """
    Creates a data structure from a raw representation of a tree ensemble tree.
    """

    def __init__(self, tree_dump, tree_type):
        if tree_type == 'cb':
            parse_func = self._parse_cb_tree
        elif tree_type == 'lgb':
            parse_func = self._parse_lgb_tree
        elif tree_type == 'xgb':
            parse_func = self._parse_xgb_tree
        else:
            exit('{} tree_type not supported'.format(tree_type))

        self.root_, self.n_nodes_ = parse_func(tree_dump)

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

            # split indices based on threshold, could be more efficient if X were indexed
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

    # LightGBM
    def _parse_lgb_tree(self, tree_info):
        structure = tree_info['tree_structure']
        root = self._get_node(structure)
        n_nodes = 1

        # tree does not have any splits
        if root.node_type == 'leaf':
            return root, n_nodes

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

    # CatBoost
    def _parse_cb_tree(self, tree_dict):
        splits = [self._get_cb_node(split_dict) for split_dict in tree_dict['splits']][::-1]
        root = splits[0]
        n_nodes = 2 ** (len(splits) + 1) - 1

        assert root.node_type == 'split', 'root node is not a split node!'
        if n_nodes == 2:
            root.set_child(Node(node_type='leaf'), 'left')
            root.set_child(Node(node_type='leaf'), 'right')
            return root, n_nodes

        traverse = [(root, 1, splits[1], 'right'), (root, 1, splits[1], 'left')]

        while len(traverse) > 0:
            parent_node, child_ndx, child_node, child_position = traverse.pop()
            parent_node.set_child(child_node, child_position)

            if child_ndx == len(splits) - 1:
                child_node.set_child(Node(node_type='leaf'), 'left')
                child_node.set_child(Node(node_type='leaf'), 'right')

            else:
                traverse.append((child_node, child_ndx + 1, splits[child_ndx + 1], 'right'))
                traverse.append((child_node, child_ndx + 1, splits[child_ndx + 1], 'left'))

        return root, n_nodes

    def _get_cb_node(self, split_dict):
        feature = int(split_dict['float_feature_index'])
        threshold = float(split_dict['border'])
        decision_type = self._get_operator('>')
        node = Node(node_type='split', feature=feature, threshold=threshold, decision_type=decision_type)
        return node

    # XGBoost
    def _parse_xgb_tree(self, tree_str):
        nodes = [self._get_xgb_node(node_str.strip()) for node_str in tree_str.split('\n') if node_str.strip() != '']
        root = nodes[0]
        n_nodes = len(nodes)

        # tree does not have any splits
        if root.node_type == 'leaf':
            return root, n_nodes

        traverse = [(root, nodes[root.right_ndx], 'right'), (root, nodes[root.left_ndx], 'left')]

        while len(traverse) > 0:
            parent_node, child_node, child_position = traverse.pop()
            parent_node.set_child(child_node, child_position)

            if child_node.node_type == 'split':
                traverse.append((child_node, nodes[child_node.right_ndx], 'right'))
                traverse.append((child_node, nodes[child_node.left_ndx], 'left'))

        return root, n_nodes

    def _get_xgb_node(self, node_str):

        split_regex = r'(\d+):\[f(\d+)(\D+)(\S+)\] yes=(\d+),no=(\d+)'

        if 'leaf' in node_str:
            node = Node(node_type='leaf')

        else:
            matches = re.match(split_regex, node_str)
            node_ndx, feature, decision_type, threshold, left_ndx, right_ndx = matches.groups()
            feature = int(feature)
            threshold = float(threshold)
            decision_type = self._get_operator(decision_type)
            left_ndx, right_ndx = int(left_ndx), int(right_ndx)
            node = Node(node_type='split', feature=feature, threshold=threshold, decision_type=decision_type,
                        left_ndx=left_ndx, right_ndx=right_ndx)

        return node

    def _get_operator(self, decision_type):

        if decision_type == '<':
            result = operator.lt
        elif decision_type == '>':
            result = operator.gt
        elif decision_type == '<=':
            result = operator.le
        else:
            exit('unknown decision_type: {}'.format(decision_type))

        return result


class Node:

    def __init__(self, node_type, feature=None, threshold=None, decision_type=None,
                 left_ndx=None, right_ndx=None):
        self.node_type = node_type
        self.feature = feature
        self.threshold = threshold
        self.op = decision_type
        self.left_ndx = left_ndx
        self.right_ndx = right_ndx

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