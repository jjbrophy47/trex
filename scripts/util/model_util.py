"""
Utility methods to make life easier.
"""
import os
import uuid
import shutil
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit


# private
class CatBoostClassifierWrapper(CatBoostClassifier):
    """
    Wrapper for the CatBoostClassifier that automatically
    converts numpy arrays into Pandas dataframes and changes
    the categorical to have an np.int64 dtype.
    """

    def __del__(self):
        """
        Remove any temporary directories set.
        """
        params = self._init_params.copy()
        if 'train_dir' in params and os.path.exists(params['train_dir']):
            shutil.rmtree(params['train_dir'])

    # override
    def fit(self, X, y):
        X = self.numpy_to_cat(X)
        return super().fit(X, y)

    # override
    def predict(self, X):
        return super().predict(self.numpy_to_cat(X))

    # override
    def predict_proba(self, X):
        return super().predict_proba(self.numpy_to_cat(X))

    # public
    def get_cat_indices(self):
        """
        Returns cat features if present, otherwise returns None.
        """
        params = self._init_params.copy()
        return params['cat_features'] if 'cat_features' in params else None

    # public
    def numpy_to_cat(self, X):
        """
        Convert numpy array of one dtype to a Pandas dataframe
        of multiple dtypes, changes cat. features to np.int64.
        """
        params = self._init_params.copy()

        # convert categorical feature dtypes into int64s
        if 'cat_features' in params:
            cat_indices = params['cat_features']
            X = pd.DataFrame(X)
            X[cat_indices] = X[cat_indices].astype(np.int64)

        return X


# public
def get_model(model,
              n_estimators=20,
              max_depth=5,
              learning_rate=0.03,
              random_state=1,
              cat_indices=None,
              class_weight=None):
    """
    Returns a tree ensemble classifier.
    """

    # LightGBM
    if model == 'lgb':
        import lightgbm
        max_depth = -1 if max_depth is None else max_depth
        clf = lightgbm.LGBMClassifier(random_state=random_state,
                                      num_leaves=32,
                                      n_estimators=n_estimators,
                                      max_depth=max_depth)
    # CatBoost
    elif model == 'cb':
        train_dir = os.path.join('.catboost_info', str(uuid.uuid4()))
        os.makedirs(train_dir, exist_ok=True)
        clf = CatBoostClassifierWrapper(random_state=random_state,
                                        n_estimators=n_estimators,
                                        max_depth=max_depth - 1,  # CB doesn't count root node in depth
                                        verbose=False,
                                        train_dir=train_dir,
                                        cat_features=cat_indices)

    # Random Forest
    elif model == 'rf':
        clf = RandomForestClassifier(random_state=random_state,
                                     n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     class_weight=class_weight)
    # GBM
    elif model == 'gbm':
        max_depth = 3 if max_depth is None else max_depth  # gbm default
        clf = GradientBoostingClassifier(random_state=random_state,
                                         n_estimators=n_estimators,
                                         max_depth=max_depth)
    # XGBoost
    elif model == 'xgb':
        import xgboost
        max_depth = 3 if max_depth is None else max_depth  # xgb default
        clf = xgboost.XGBClassifier(random_state=random_state,
                                    n_estimators=n_estimators,
                                    max_depth=max_depth)
    else:
        exit('{} model not supported!'.format(model))

    return clf


def train_tree_ensemble(model, X_train, y_train,
                        param_grid={'n_estimators': [10, 25, 50, 100, 250],
                                    'max_depth': [3, 5, 10]},
                        scoring='accuracy', tune_frac=0.1, cv=5, seed=1):
    """
    Tune and train a tree-ensemble model.
    """

    # tune on a fraction of the training data
    if tune_frac < 1.0:
        sss = StratifiedShuffleSplit(n_splits=1,
                                     test_size=2,
                                     train_size=tune_frac,
                                     random_state=seed)
        tune_indices, _ = list(sss.split(X_train, y_train))[0]
        X_train_sub, y_train_sub = X_train[tune_indices], y_train[tune_indices]

    # tune using the entire training set
    else:
        X_train_sub, y_train_sub = X_train, y_train

    # tune
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    gs = GridSearchCV(model, param_grid, scoring=scoring, cv=skf, verbose=1)
    gs = gs.fit(X_train_sub, y_train_sub)

    # train on the entire training set
    model = clone(gs.best_estimator_)
    model = model.fit(X_train, y_train)

    return model, gs.best_params_


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def performance(model, X, y, logger=None,
                name='', do_print=False):
    """
    Returns AUROC and accuracy scores.
    """

    # only 1 sample
    if y.shape[0] == 1:
        return

    # generate prediction probabilities
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)[:, 1]

    elif hasattr(model, 'decision_function'):
        y_proba = sigmoid(model.decision_function(X)).reshape(-1, 1)

    else:
        y_proba = None

    # generate predictions
    y_pred = model.predict(X)

    # evaluate
    auc = roc_auc_score(y, y_proba) if y_proba is not None else None
    acc = accuracy_score(y, y_pred)
    ap = average_precision_score(y, y_proba) if y_proba is not None else None
    ll = log_loss(y, y_proba) if y_proba is not None else None

    # get display string
    if y_proba is not None:
        score_str = '[{}] acc: {:.3f}, auc: {:.3f}, ap: {:.3f}, ll: {:.3f}'

    else:
        score_str = '[{}] acc: {}, auc: {:.3f}, ap: {}, ll: {}'

    # print scores
    if logger:
        logger.info(score_str.format(name, acc, auc, ap, ll))

    elif do_print:
        print(score_str.format(name, acc, auc, ll))

    return auc, acc, ap, ll


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


def get_selected_params(dataset, model, surrogate):
    """
    Convenience method to return selected hyperparameters for a given
    dataset / model / surrogate combination
    """

    # selected hyperparameters using the CatBoost model
    cb = {}
    cb['surgical'] = ['tree_output', 1.0, 'leaf_path', 15]
    cb['vaccine'] = ['tree_output', 1.0, 'tree_output', 61]
    cb['amazon'] = ['tree_output', 1.0, 'feature_path', 7]
    cb['bank_marketing'] = ['tree_output', 1.0, 'tree_output', 31]
    cb['adult'] = ['tree_output', 1.0, 'tree_output', 61]
    cb['census'] = ['tree_output', 1.0, 'tree_output', 61]

    # selected hyperparameters using the RF model
    rf = {}
    rf['surgical'] = ['leaf_path', 0.001, 'feature_output', 61]
    rf['vaccine'] = ['leaf_output', 0.001, 'feature_output', 61]
    rf['amazon'] = ['tree_output', 0.01, 'tree_output', 61]
    rf['bank_marketing'] = ['leaf_output', 0.001, 'leaf_path', 61]
    rf['adult'] = ['leaf_output', 0.001, 'feature_output', 61]
    rf['census'] = ['tree_output', 0.001, 'leaf_path', 61]

    # select parameters based on the tree-ensemble
    d = cb if model == 'cb' else rf

    # extracting hyperparameters
    tree_kernel = d[dataset][0] if 'klr' in surrogate else d[dataset][2]
    C = d[dataset][1]
    n_neighbors = d[dataset][3]

    # make custom adjustment
    for kernel in ['feature_path', 'feature_output', 'weighted_leaf_path', 'leaf_path', 'leaf_output', 'tree_output']:
        if kernel in surrogate:
            tree_kernel = kernel
            break

    # make custom adjustment
    c_str = surrogate.split('_')[-1]
    if 'C' in c_str:
        C = float(c_str.split('-')[-1])

    # assemble params
    result = {'C': C, 'n_neighbors': n_neighbors, 'tree_kernel': tree_kernel}

    return result
