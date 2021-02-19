"""
Utility methods for surrogate models.
"""
import time
from itertools import product

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from ..models.linear_model import SVM
from ..models.linear_model import KernelLogisticRegression


def train_surrogate(model, surrogate, param_grid, X_train, X_train_alt, y_train,
                    val_frac=0.1, metric='pearson', cv=5, seed=1, logger=None):
    """
    Tunes a surrogate model by choosing hyperparameters that provide the best fidelity
    correlation to the tree-ensemble predictions.
    """
    assert val_frac > 0.0 and val_frac <= 1.0

    # randomly select a set of samples from the training data
    rng = np.random.default_rng(seed)
    n_val = int(X_train_alt.shape[0] * val_frac)
    val_indices = rng.choice(X_train_alt.shape[0], size=n_val, replace=False)

    # extract validation data
    X_val = X_train_alt[val_indices]
    X_val_alt = X_train_alt[val_indices]
    y_val = y_train[val_indices]

    # enumerate cartesion cross-product of hyperparameters
    params_list = cartesian_product(param_grid)

    # result containers
    results = []
    fold = 0

    # start timing
    begin = time.time()

    # tune surrogate model using the validation data
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    for fold, (train_index, test_index) in enumerate(skf.split(X_val_alt, y_val)):

        # original train and test data
        X_val_train = X_val[train_index]
        X_val_test = X_val[test_index]

        # transformed train and test data
        X_val_alt_train = X_val_alt[train_index]
        X_val_alt_test = X_val_alt[test_index]

        # labels
        y_val_train = y_val[train_index]

        # perform gridsearch
        scores = []
        for params in params_list:
            start = time.time()

            # fit a tree ensemble and make predictions on the train fold
            m1 = clone(model).fit(X_val_train, y_val_train)
            y_val_train_pred = m1.predict(X_val_train)

            # train a surrogate model on the predicted labels
            m2 = get_surrogate_model(surrogate, params).fit(X_val_alt_train, y_val_train_pred)

            # generate predictions on the test set
            m1_proba = m1.predict_proba(X_val_test)[:, 1]
            m2_proba = m2.predict_proba(X_val_alt_test)[:, 1]

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

    # train the surrogate model on the train set using predicted labels
    start = time.time()
    y_train_pred = model.predict(X_train)
    surrogate_model = get_surrogate_model(surrogate, params=best_params)
    surrogate_model = surrogate_model.fit(X_train_alt, y_train_pred)

    # display train results
    if logger:
        logger.info('train time: {:.3f}s'.format(time.time() - start))

    return surrogate_model


# private
def get_surrogate_model(surrogate='klr', params={}, temp_dir='.'):
    """
    Return C implementation of the kernel model.
    """
    if surrogate == 'klr':
        surrogate_model = KernelLogisticRegression(C=params['C'], temp_dir=temp_dir)

    elif surrogate == 'svm':
        surrogate_model = SVM(C=params['C'], temp_dir=temp_dir)

    elif surrogate == 'knn':
        surrogate_model = KNeighborsClassifier(n_neighbors=params['n_neighbors'], weights='uniform')

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

    # return 1 - mse so that each score can be maximized
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
