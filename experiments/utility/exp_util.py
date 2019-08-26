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
from . import model_util


def sort_impact(sv_ndx, impact, ascending=False):
    """Sorts support vectors by absolute impact values in descending order."""

    if impact.ndim == 2:
        impact = np.sum(impact, axis=1)
    impact_list = zip(sv_ndx, impact)
    impact_list = sorted(impact_list, key=lambda tup: abs(tup[1]), reverse=not ascending)

    sv_ndx, impact = zip(*impact_list)
    sv_ndx = np.array(sv_ndx)
    return sv_ndx, impact


# TODO: not really needed anymore
def avg_impact(explainer, X_test, test_indices=None, progress=False):
    """
    Returns avg impacts of train instances over a given set of test instances.
    Parameters
    ----------
    explainer : sexee.TreeExplainer
        SVM explainer for the tree ensemble.
    X_test : 2d array-like
        Test instances in original feature space.
    test_indices : 1d array-like
        Indexes to compute the average impact over.
    """

    result = defaultdict(float)

    if test_indices is not None:
        X_test_instances = X_test[test_indices]
    else:
        X_test_instances = X_test

    # compute average impact of support vectors over test indices
    for x_test in tqdm.tqdm(X_test_instances, disable=not progress):
        impact_list = explainer.train_impact(x_test)

        # update the train instance impacts
        for train_ndx, train_impact in impact_list:
            result[train_ndx] += train_impact

    # divide by the number of test instances
    for train_ndx, train_val in result.items():
        result[train_ndx] = train_val / len(X_test_instances)

    result = list(result.items())
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

    yhat1_ll = instance_loss(yhat1, y_true, logloss=True)
    yhat2_ll = instance_loss(yhat2, y_true, logloss=True)
    diff = yhat1_ll - yhat2_ll
    diff_ndx = np.argsort(diff)
    if sort == 'descending':
        diff_ndx = diff_ndx[::-1]
    return diff_ndx[:k]


def instance_loss(y_hat, y_true=None, logloss=False):
    """
    Returns the loss per instance.

    Parameters
    ----------
    y_hat : 1d or 2d array-like
        If 1d, probability of positive class.
        If 2d, array of 1d arrays denoting the probability of each class.
    y_true : 1d array-like (default=None)
        True labels.
    logloss : bool (default=False)
        If True, returns loss values in log form.

    Returns
    -------
    1d array-like of loss values, one for each instance.
    """

    if y_hat.ndim == 2 and y_true is not None:
        assert len(y_hat) == len(y_true), 'y_hat is not the same len as y_true!'
        assert y_hat.shape[1] == len(np.unique(y_true)), 'number of classes do not match!'
        y_hat = positive_class_proba(y_true, y_hat)

    if logloss:
        result = np.log(1 - y_hat)
    else:
        result = 1 - y_hat

    return result


def positive_class_proba(labels, probas):
    """
    Given the predicted label of each sample and the probabilities for each class for each sample,
    return the probabilities of the positive class for each sample.
    """

    assert labels.ndim == 1, 'labels is not 1d!'
    assert probas.ndim == 2, 'probas is not 2d!'
    assert len(labels) == len(probas), 'num samples do not match between labels and probas!'
    y_pred = probas[np.arange(len(labels)), labels]
    assert y_pred.ndim == 1, 'y_pred is not 1d!'
    return y_pred


def make_multiclass(yhat):
    """
    Turn a 1d array of porbabilities for a positive class into a 2d array of
    probabilities for each class.
    Parameters
    ----------
    yhat : 1d array-like
        Array of probabilities for the positive class.
    Returns
    -------
    2d array of shape (n_samples, n_classes) where column 0 are probabilities
    for the negative class and column 1 are probabilities for the positive class.
    """
    assert yhat.ndim == 1, 'yhat is not 1d!'

    yhat = yhat.reshape(-1, 1)
    yhat = np.hstack([1 - yhat, yhat])
    return yhat


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
