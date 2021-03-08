"""
Utility methods for surrogate models.
"""
import time
from itertools import product

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from .extractor import TreeExtractor
from .models import SVM
from .models import KLR
from .models import KNN


def train_surrogate(model, surrogate, X_train, y_train,
                    val_frac=0.0, metric='mse', cv=5, seed=1,
                    params=None, logger=None):
    """
    Train a surrogate model on tree-extracted features. If 0 < `val_frac` <= 1.0, then
    tune the surrogate model as well.
    """

    # train but do not tune
    if val_frac <= 0.0 or val_frac > 1.0:
        assert params is not None, 'params should not be None!'
        start = time.time()

        # transform train data
        tree_extractor = TreeExtractor(model, tree_kernel=params['tree_kernel'])

        # train surrogate
        surrogate_model = get_surrogate_model(tree_extractor, surrogate, params=params, random_state=seed)
        surrogate_model = surrogate_model.fit(X_train, y_train)

        # display train results
        if logger:
            logger.info('train time: {:.3f}s'.format(time.time() - start))

    # tune and train the surrogate model
    else:
        surrogate = tune_and_train_surrogate(model=model,
                                             surrogate=surrogate,
                                             X_train=X_train,
                                             y_train=y_train,
                                             val_frac=val_frac,
                                             metric=metric,
                                             cv=cv,
                                             seed=seed,
                                             logger=logger)

    return surrogate


def tune_and_train_surrogate(model, surrogate, X_train, y_train,
                             val_frac=0.1, metric='mse', cv=5, seed=1, logger=None):
    """
    Tunes a surrogate model by choosing hyperparameters that provide the best fidelity
    correlation to the tree-ensemble predictions.
    """
    assert val_frac > 0.0 and val_frac <= 1.0

    # randomly select a set of samples from the training data
    rng = np.random.default_rng(seed)
    n_val = int(X_train.shape[0] * val_frac)
    val_indices = rng.choice(X_train.shape[0], size=n_val, replace=False)

    # extract validation data
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]

    # enumerate cartesion cross-product of hyperparameters
    param_grid = get_surrogate_params(surrogate)
    params_list = cartesian_product(param_grid)

    # result containers
    results = []
    fold = 0

    # start timing
    begin = time.time()
    if logger:
        logger.info('\ntraining surrogate model...')

    # tune surrogate model using the validation data
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    for fold, (train_index, test_index) in enumerate(skf.split(X_val, y_val)):

        # original train and test data
        X_val_train = X_val[train_index]
        X_val_test = X_val[test_index]

        # labels
        y_val_train = y_val[train_index]

        # perform gridsearch
        scores = []
        for params in params_list:
            start = time.time()

            # fit a tree ensemble and make predictions on the train fold
            m1 = clone(model).fit(X_val_train, y_val_train)

            # transform fold data
            tree_extractor = TreeExtractor(m1, tree_kernel=params['tree_kernel'])

            # train a surrogate model on the predicted labels
            m2 = get_surrogate_model(tree_extractor, surrogate, params, random_state=seed)
            m2 = m2.fit(X_val_train, y_val_train)

            # generate predictions on the test set
            m1_proba = m1.predict_proba(X_val_test)[:, 1]
            m2_proba = m2.predict_proba(X_val_test)[:, 1]

            # measure fidelity
            score = score_fidelity(m1_proba, m2_proba, metric)
            scores.append(score)

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
    best_params = params_list[best_ndx]

    # display tuning results
    if logger:
        logger.info('best params: {}'.format(best_params))
        logger.info('tune time: {:.3f}s'.format(time.time() - begin))

    # train surrogate model
    start = time.time()

    # transform train data
    tree_extractor = TreeExtractor(model, tree_kernel=best_params['tree_kernel'])

    # train surrogate
    surrogate = get_surrogate_model(tree_extractor, surrogate, params=best_params, random_state=seed)
    surrogate = surrogate.fit(X_train, y_train)

    # display train results
    if logger:
        logger.info('train time: {:.3f}s'.format(time.time() - start))

    return surrogate


# private
def get_surrogate_params(surrogate='klr'):
    """
    Return surrogate-specific hyperparameters to search.
    """
    result = {}

    if surrogate in ['klr', 'svm']:
        result['C'] = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]

    elif surrogate == 'knn':
        result['n_neighbors'] = [3, 5, 7, 9, 11, 13, 15, 31, 45, 61]

    result['tree_kernel'] = ['feature_path', 'feature_output', 'leaf_path',
                             'leaf_output', 'tree_output']

    return result


def get_surrogate_model(tree_extractor, surrogate='klr', params={}, random_state=1):
    """
    Return C implementation of the kernel model.
    """
    if surrogate == 'klr':
        surrogate_model = KLR(tree_extractor,
                              C=params['C'],
                              random_state=random_state)

    elif surrogate == 'svm':
        surrogate_model = SVM(tree_extractor,
                              C=params['C'],
                              random_state=random_state)

    elif surrogate == 'knn':
        surrogate_model = KNN(tree_extractor,
                              n_neighbors=params['n_neighbors'],
                              weights='uniform')

    else:
        raise ValueError('surrogate {} unknown!'.format(surrogate))

    return surrogate_model


def score_fidelity(p1, p2, metric='pearson'):
    """
    Returns fidelity score based on the probability
    scores of `p1` and `p2`.
    """
    if metric == 'pearson':
        result, p_value = pearsonr(p1, p2)

    elif metric == 'spearman':
        result, p_value = spearmanr(p1, p2)

    elif metric == 'mse':
        result = mean_squared_error(p1, p2)

    else:
        raise ValueError('metric {} unknown!'.format(metric))

    return result


def cartesian_product(my_dict):
    """
    Takes in a dictionary of lists, and returns a cartesian product of those in lists
    in the form of a list of ditionaries.
    """
    return list((dict(zip(my_dict, x)) for x in product(*my_dict.values())))
