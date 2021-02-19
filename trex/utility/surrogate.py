"""
Utility methods for surrogate models.
"""
import time
from itertools import product

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from ..models.linear_model import SVM
from ..models.linear_model import KernelLogisticRegression


def train_surrogate(model, surrogate, param_grid, feature_extractor, X_train, y_train,
                    val_frac=0.1, seed=1, cv=5, logger=None):
    """
    Tunes a surrogate model by choosing hyperparameters that provide the best fidelity
    correlation to the tree-ensemble predictions.
    """
    assert val_frac > 0.0 and val_frac <= 1.0

    # n_neighbors_grid = [3, 5, 7, 9, 11, 13, 15, 31, 45, 61]

    # if not val_frac:
    #     knn_clf = KNeighborsClassifier(n_neighbors=3, weights='uniform')
    #     knn_clf = knn_clf.fit(X_train_feature, y_train)

    # tune_start = time.time()

    # randomly select a fraction of the train data to use for tuning
    # n_tune = int(X_train.shape[0] * args.tune_frac)
    # X_val, y_val = X_train[:n_tune], y_train[:n_tune]

    # select a subset of samples from the training data
    rng = default_rng(seed)
    n_val = int(X_train.shape[0] * val_frac)
    val_indices = rng.choice(X_train.shape[0], size=n_samples, replace=False)

    X_val = X_train[val_indices]
    X_val_feature = X_train_feature[val_indices]
    y_val = y_train[val_indices]

    # enumerate cartesion cross-product of hyperparameters
    params_list = cartesian_product(param_grid)

    # result containers
    results = []
    fold = 0

    # start timing
    start = time.time()

    # tune surrogate model using the validation data
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    for fold, (train_index, test_index) in enumerate(skf.split(X_val_feature, y_val)):

        # obtain fold data
        X_val_train = X_val[train_index]
        X_val_test = X_val[test_index]
        X_val_feature_train = X_val_feature[train_index]
        X_val_feature_test = X_val_feature[test_index]
        y_val_train = y_val[train_index]

        # gridsearch n_neighbors
        scores = []
        for params in params_list:
            start = time.time()

            # fit a tree ensemble and surrogate model
            m1 = clone(model).fit(X_val_train, y_val_train)
            m2 = get_surrogate_model(surrogate, params).fit(X_val_alt_train, y_val_train)

            # generate predictions
            m1_proba = m1.predict_proba(X_val_test)[:, 1]
            m2_proba = m2.predict_proba(X_val_alt_test)[:, 1]

            # measure correlation
            score = score_fidelity(metric, m1_proba, m2_proba) pearsonr(m1_proba, m2_proba)[0]
            scores.append(correlation)

            # display progress
            if logger:
                s = '[Fold {}] params={}: {}={:.3f}, {:.3f}s'
                logger.info(s.format(fold, params, metric, score, time.time() - start))

        # add scores to result list
        results.append(scores)

    # compile results
    results = np.vstack(results).mean(axis=0)

    # find hyperparameters with best fidelity score
    best_ndx = np.argmax(results) if metric in ['pearson', 'spearman'] else np.argmin(results)
    best_params = param_grid[best_ndx]

    # display tuning results
    if logger:
        logger.info('chosen params: {}'.format(best_params))
        logger.info('tune time: {:.3f}s'.format(time.time() - start))

    # train the surrogate model on the entire training set
    start = time.time()
    surrogate_model = get_surrogate_model(params=best_params).fit(X_train_alt, y_train)

    # display train results
    if logger:
        logger.info('train time: {:.3f}s'.format(time.time() - start))

    return surrogate_model


# private
def get_surrogate_model(surrogate='trex_lr', params={}, temp_dir='.'):
    """
    Return C implementation of the kernel model.
    """
    if surrogate == 'trex_lr':
        surrogate_model = KernelLogisticRegression(C=params['C'], temp_dir=temp_dir)

    elif surrogate == 'trex_svm':
        surrogate_model = SVM(C=params['C'], temp_dir=temp_dir)

    elif surrogate == 'teknn':
        surrogate_model = KNeighborsClassifier(n_neighbors=params['n_neighbors'], weights=params['weights'])

    else:
        raise ValueError('surrogate {} unknown!'.format(surrogate))

    return surrogate_model


def cartesian_product(my_dict):
    """
    Takes in a dictionary of lists, and returns a cartesian product of those in lists
    in the form of a list of ditionaries.
    """
    return list((dict(zip(my_dict, x)) for x in product(*my_dict.values())))
