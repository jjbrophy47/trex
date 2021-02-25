"""
Utility methods to make life easier.
"""
import os
import uuid
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss


def get_model(model,
              n_estimators=20,
              max_depth=None,
              learning_rate=0.03,
              random_state=1,
              cat_indices=None):
    """
    Returns a tree ensemble classifier.
    """

    # LightGBM
    if model == 'lgb':
        import lightgbm
        max_depth = -1 if max_depth is None else max_depth
        clf = lightgbm.LGBMClassifier(random_state=random_state,
                                      n_estimators=n_estimators,
                                      max_depth=max_depth)
    # CatBoost
    elif model == 'cb':
        import catboost
        train_dir = os.path.join('.catboost_info', 'rs_{}'.format(random_state), str(uuid.uuid4()))
        os.makedirs(train_dir, exist_ok=True)
        clf = catboost.CatBoostClassifier(random_state=random_state,
                                          n_estimators=n_estimators,
                                          max_depth=max_depth,
                                          verbose=False,
                                          train_dir=train_dir,
                                          cat_features=cat_indices)
    # Random Forest
    elif model == 'rf':
        max_depth = None if max_depth == 0 else max_depth
        clf = RandomForestClassifier(random_state=random_state,
                                     n_estimators=n_estimators,
                                     max_depth=max_depth)
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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def performance(model, X, y, logger=None,
                name='', do_print=False):
    """
    Returns AUROC and accuracy scores.
    """

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
        score_str = '[{}] auc: {:.3f}, acc: {:.3f}, ap: {:.3f}, ll: {:.3f}'

    else:
        score_str = '[{}] auc: {}, acc: {:.3f}, ap: {}, ll: {}'

    # print scores
    if logger:
        logger.info(score_str.format(name, auc, acc, ap, ll))

    elif do_print:
        print(score_str.format(name, auc, acc, ll))

    return auc, acc, ap, ll


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
