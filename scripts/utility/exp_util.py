"""
Utility methods used by different experiments.
"""
import time
import os
import sys
import pickle
sys.path.insert(0, os.getcwd())  # for influence_boosting
from copy import deepcopy

import tqdm
import numpy as np
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import log_loss

from baselines.influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble


# def get_val_data(X_train, val_frac, seed):
#     """
#     Use part of the training data as validation data.
#     """
#     if val_frac < 1.0 and val_frac > 0.0:
#         n_val = int(X_train.shape[0] * val_frac)
#         np.random.seed(seed)
#         val_indices = np.random.choice(X_train.shape[0], size=n_val, replace=False)
#         X_val = X_train[val_indices].copy()
#     return X_val


# def save_model(model, model_path):
#     """
#     Saves model for later use.
#     """
#     with open(model_path, 'wb') as f:
#         f.write(pickle.dumps(model))


# def load_model(model_path):
#     """
#     Loads a previously saved model.
#     """
#     assert os.path.exists(model_path)
#     with open(model_path, 'rb') as f:
#         return pickle.loads(f.read())


# def tune_knn(tree, X_train, X_train_feature, y_train, val_frac, seed, logger=None, cv=5):
#     """
#     Tunes KNN by choosing hyperparameters that give the best pearson
#     correlation to the tree predictions.
#     """
#     n_neighbors_grid = [3, 5, 7, 9, 11, 13, 15, 31, 45, 61]

#     if not val_frac:
#         knn_clf = KNeighborsClassifier(n_neighbors=3, weights='uniform')
#         knn_clf = knn_clf.fit(X_train_feature, y_train)

#     tune_start = time.time()

#     # select a fraction of the training data
#     n_samples = int(X_train.shape[0] * val_frac)
#     np.random.seed(seed)
#     val_indices = np.random.choice(np.arange(X_train.shape[0]), size=n_samples)

#     X_val = X_train[val_indices]
#     X_val_feature = X_train_feature[val_indices]
#     y_val = y_train[val_indices]

#     # result containers
#     results = []
#     fold = 0

#     # tune C
#     skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
#     for train_index, test_index in skf.split(X_val_feature, y_val):
#         fold += 1

#         # obtain fold data
#         X_val_train = X_val[train_index]
#         X_val_test = X_val[test_index]
#         X_val_feature_train = X_val_feature[train_index]
#         X_val_feature_test = X_val_feature[test_index]
#         y_val_train = y_val[train_index]

#         # gridsearch n_neighbors
#         correlations = []
#         for n_neighbors in n_neighbors_grid:
#             start = time.time()

#             # fit a tree ensemble and surrogate model
#             m1 = clone(tree).fit(X_val_train, y_val_train)
#             m2 = KNeighborsClassifier(n_neighbors=n_neighbors,
#                                       weights='uniform').fit(X_val_feature_train, y_val_train)

#             # generate predictions
#             m1_proba = m1.predict_proba(X_val_test)[:, 1]
#             m2_proba = m2.predict_proba(X_val_feature_test)[:, 1]

#             # measure correlation
#             correlation = pearsonr(m1_proba, m2_proba)[0]
#             correlations.append(correlation)

#             if logger:
#                 s = '[Fold {}] n_neighbors={:<2}: {:.3f}s; corr={:.3f}'
#                 logger.info(s.format(fold, n_neighbors, time.time() - start, correlation))

#         results.append(correlations)
#     results = np.vstack(results).mean(axis=0)
#     best_ndx = np.argmax(results)
#     best_n_neighbors = n_neighbors_grid[best_ndx]

#     if logger:
#         logger.info('chosen n_neighbors: {}'.format(best_n_neighbors))
#         logger.info('total tuning time: {:.3f}s'.format(time.time() - tune_start))
#         logger.info('training...')

#     train_start = time.time()
#     knn_clf = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights='uniform')
#     knn_clf = knn_clf.fit(X_train_feature, y_train)

#     if logger:
#         logger.info('total training time: {:.3f}s'.format(time.time() - train_start))

#     return knn_clf


# def log_loss_increase(yhat1, yhat2, y_true, sort='ascending', k=50):
#     """
#     Returns the instances with the largest log loss increases.
#     Negative numbers mean a log loss increase, while positive are log loss reductions.
#     Parameters
#     ----------
#     y1 and y2 : 1d array-like
#         Probabilties for each instance.
#     y_true : 1d array-like
#         Ground-truth labels.
#     sorted : str (default='descending')
#         Returns instance indexes in decreasing order of increased log loss.
#     k : int (default=50)
#         Number of instances to return.
#     """

#     assert len(yhat1) == len(yhat2) == len(y_true), 'yhat, yhat2, and y_true are not the same length!'

#     yhat1_ll = instance_loss(yhat1, y_true, logloss=True)
#     yhat2_ll = instance_loss(yhat2, y_true, logloss=True)
#     diff = yhat1_ll - yhat2_ll
#     diff_ndx = np.argsort(diff)
#     if sort == 'descending':
#         diff_ndx = diff_ndx[::-1]
#     return diff_ndx[:k]


def instance_log_loss(y_true, y_proba, labels=[0, 1]):
    """
    Returns log loss per instance assuming binary classification.
    """
    assert y_true.shape[0] == y_proba.shape[0], 'y_true and y_proba different lengths!'
    assert y_true.ndim == y_proba.ndim == 1, 'y_proba or y_true is not 1d!'
    assert np.all(np.unique(y_true) == np.array(labels))

    # result container
    results = []

    # compute log losses
    for i in range(y_true.shape[0]):
        results.append(log_loss(y_true[[i]], y_proba[[i]], labels=labels))

    return np.array(results)


# def instance_loss(y_hat, y_true=None, logloss=False):
#     """
#     Returns the loss per instance assuming binary classification.

#     Parameters
#     ----------
#     y_hat : 1d or 2d array-like
#         If 1d, probability of positive class.
#         If 2d, array of 1d arrays denoting the probability of each class.
#     y_true : 1d array-like (default=None)
#         True labels.
#     logloss : bool (default=False)
#         If True, returns loss values in log form.

#     Returns
#     -------
#     1d array-like of loss values, one for each instance.
#     """

#     if y_hat.ndim == 2 and y_true is not None:
#         assert len(y_hat) == len(y_true), 'y_hat is not the same len as y_true!'
#         assert y_hat.shape[1] == len(np.unique(y_true)), 'number of classes do not match!'
#         y_hat = positive_class_proba(y_true, y_hat)

#     if logloss:
#         result = np.log(1 - y_hat)
#     else:
#         result = 1 - y_hat

#     return result


# def positive_class_proba(labels, probas):
#     """
#     Given the predicted label of each sample and the probabilities for each class for each sample,
#     return the probabilities of the positive class for each sample.
#     """

#     assert labels.ndim == 1, 'labels is not 1d!'
#     assert probas.ndim == 2, 'probas is not 2d!'
#     assert len(labels) == len(probas), 'num samples do not match between labels and probas!'
#     y_pred = probas[np.arange(len(labels)), labels]
#     assert y_pred.ndim == 1, 'y_pred is not 1d!'
#     return y_pred


# def make_multiclass(yhat):
#     """
#     Turn a 1d array of porbabilities for a positive class into a 2d array of
#     probabilities for each class.
#     Parameters
#     ----------
#     yhat : 1d array-like
#         Array of probabilities for the positive class.
#     Returns
#     -------
#     2d array of shape (n_samples, n_classes) where column 0 are probabilities
#     for the negative class and column 1 are probabilities for the positive class.
#     """
#     assert yhat.ndim == 1, 'yhat is not 1d!'

#     yhat = yhat.reshape(-1, 1)
#     yhat = np.hstack([1 - yhat, yhat])
#     return yhat


# def get_influence_explainer(model, X_train, y_train, inf_k):
#     """
#     Returns a CBLeafInfluenceEnsemble explainer.
#     Parameters
#     ----------
#     model : object
#         Learned CatBoost tree ensemble.
#     X_train : 2d array-like
#         Train data.
#     y_train : 1d array-like
#         Train labels.
#     k : int
#         Number of leaves to use in explainer.
#     """

#     model_path = '.model.json'
#     model.save_model(model_path, format='json')

#     if inf_k == -1:
#         update_set = 'AllPoints'
#     elif inf_k == 0:
#         update_set = 'SinglePoint'
#     else:
#         update_set = 'TopKLeaves'

#     leaf_influence = CBLeafInfluenceEnsemble(model_path, X_train, y_train, learning_rate=model.learning_rate_,
#                                              update_set=update_set, k=inf_k)

#     return leaf_influence


# def influence_explain_instance(explainer, test_ndx, X_train, X_test, y_test):
#     """
#     Explains a single test instance using fastleafinfluence by Sharchilev et al.
#     Parameters
#     ----------
#     explainer : CBLeafInfluenceEnsemble
#         Leaf influence explainer to compute derivatives.
#     test_ndx : int
#         Index of test instance to explain.
#     X_train : 2d array-like
#         Train data.
#     X_test : 2d array-like
#         Test data.
#     y_test : 1d array-like
#         Test labels.
#     """
#     assert X_train.shape[1] == X_test.shape[1], 'X_train and X_test do not have equal num features!'
#     assert len(X_test) == len(y_test), 'X_test and y_test are not the same length!'
#     assert test_ndx < len(y_test), 'test_ndx is out of bounds of y_test!'

#     influence_scores = []
#     buf = deepcopy(explainer)
#     for i in tqdm.tqdm(range(len(X_train))):
#         explainer.fit(removed_point_idx=i, destination_model=buf)
#         influence_scores.append(buf.loss_derivative(X_test[[test_ndx]], y_test[[test_ndx]])[0])
#     influence_scores = np.array(influence_scores)

#     return influence_scores
