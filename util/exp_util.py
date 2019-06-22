"""
Utility methods used by different experiments.
"""
import tqdm
import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd())  # for influence_boosting
from collections import defaultdict
from copy import deepcopy

from influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble


def avg_impact(explainer, test_indices, X_test, progress=False):
    """
    Returns avg impacts of train instances over a given set of test instances.
    Parameters
    ----------
    explainer : sexee.TreeExplainer
        SVM explainer for the tree ensemble.
    test_indices : 1d array-like
        Indexes to compute the average impact over.
    X_test : 2d array-like
        Test instances in original feature space.
    """

    result = defaultdict(float)

    # compute average impact of support vectors over test indices
    for test_ndx in tqdm.tqdm(test_indices, disable=not progress):
        impact_list = explainer.train_impact(X_test[test_ndx].reshape(1, -1))

        # update the train instance impacts
        for train_ndx, train_impact in impact_list:
            result[train_ndx] += train_impact

    # divide by the number of test instances
    for train_ndx, train_val in result.items():
        result[train_ndx] = train_val / len(test_indices)

    return result


def log_loss_increase(yhat1, yhat2, y_true, sort='ascending', k=50):
    """
    Returns the instances with the largest log loss increases.
    Negative numbers mean a log loss increase, while positive are log loss reductions.
    Parameters
    ----------
    y1 and y2 : 1d array-like
        Probabilties for each instance.
    y_true : 1d array-like
        Ground-truth labels.
    sorted : str (default='descending')
        Returns instance indexes in decreasing order of increased log loss.
    k : int (default=50)
        Number of instances to return.
    """

    assert len(yhat1) == len(yhat2) == len(y_true), 'yhat, yhat2, and y_true are not the same length!'

    yhat1_ll = instance_log_loss(yhat1, y_true)
    yhat2_ll = instance_log_loss(yhat2, y_true)
    diff = yhat1_ll - yhat2_ll
    diff_ndx = np.argsort(diff)
    if sort == 'descending':
        diff_ndx = diff_ndx[::-1]
    return diff_ndx[:k]


def instance_log_loss(y_hat, y_true):
    """Returns the log loss per instance."""
    return np.log(1 - np.abs(y_hat - y_true))


def get_influence_explainer(model, X_train, y_train, inf_k):
    """
    Returns a CBLeafInfluenceEnsemble explainer.
    Parameters
    ----------
    model : object
        Learned CatBoost tree ensemble.
    X_train : 2d array-like
        Train data.
    y_train : 1d array-like
        Train labels.
    k : int
        Number of leaves to use in explainer.
    """

    model_path = '.model.json'
    model.save_model(model_path, format='json')

    if inf_k == -1:
        update_set = 'AllPoints'
    elif inf_k == 0:
        update_set = 'SinglePoint'
    else:
        update_set = 'TopKLeaves'

    leaf_influence = CBLeafInfluenceEnsemble(model_path, X_train, y_train, learning_rate=model.learning_rate_,
                                             update_set=update_set, k=inf_k)

    return leaf_influence


def influence_explain_instance(explainer, test_ndx, X_train, X_test, y_test):
    """
    Explains a single test instance using fastleafinfluence by Sharchilev et al.
    Parameters
    ----------
    explainer : CBLeafInfluenceEnsemble
        Leaf influence explainer to compute derivatives.
    test_ndx : int
        Index of test instance to explain.
    X_train : 2d array-like
        Train data.
    X_test : 2d array-like
        Test data.
    y_test : 1d array-like
        Test labels.
    """
    assert X_train.shape[1] == X_test.shape[1], 'X_train and X_test do not have equal num features!'
    assert len(X_test) == len(y_test), 'X_test and y_test are not the same length!'
    assert test_ndx < len(y_test), 'test_ndx is out of bounds of y_test!'

    influence_scores = []
    buf = deepcopy(explainer)
    for i in tqdm.tqdm(range(len(X_train))):
        explainer.fit(removed_point_idx=i, destination_model=buf)
        influence_scores.append(buf.loss_derivative(X_test[test_ndx], y_test[test_ndx])[0])
    influence_scores = np.array(influence_scores)

    return influence_scores
