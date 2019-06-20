"""
Utility methods specific to different experiments.
"""
import tqdm
import numpy as np
from collections import defaultdict


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
