"""
Utility methods to make life easier.
"""
import os
import uuid
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def get_classifier(model, n_estimators=20, max_depth=None, learning_rate=0.03, random_state=69):
    """Returns a tree ensemble classifier."""

    # create model
    if model == 'lgb':
        import lightgbm
        max_depth = -1 if max_depth is None else max_depth
        clf = lightgbm.LGBMClassifier(random_state=random_state, n_estimators=n_estimators,
                                      max_depth=max_depth)
    elif model == 'cb':
        import catboost
        train_dir = os.path.join('.catboost_info', str(uuid.uuid4()))
        os.makedirs(train_dir, exist_ok=True)
        clf = catboost.CatBoostClassifier(random_state=random_state, n_estimators=n_estimators,
                                          max_depth=max_depth, verbose=False, train_dir=train_dir)
    elif model == 'rf':
        clf = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators,
                                     max_depth=max_depth)
    elif model == 'gbm':
        max_depth = 3 if max_depth is None else max_depth  # gbm default
        clf = GradientBoostingClassifier(random_state=random_state, n_estimators=n_estimators,
                                         max_depth=max_depth)
    elif model == 'xgb':
        import xgboost
        max_depth = 3 if max_depth is None else max_depth  # xgb default
        clf = xgboost.XGBClassifier(random_state=random_state, n_estimators=n_estimators,
                                    max_depth=max_depth)
    else:
        exit('{} model not supported!'.format(model))

    return clf


def fidelity(y1, y2, return_difference=False):
    """Returns an (overlap, difference) tuple."""

    overlap = np.where(y1 == y2)[0]
    difference = np.where(y1 != y2)[0]

    if return_difference:
        result = overlap, difference
    else:
        result = overlap

    return result


def missed_instances(y1, y2, y_true):
    """Returns indexes missed by both y1 and y2."""

    both_ndx = np.where((y1 == y2) & (y1 != y_true))[0]
    return both_ndx


def performance(model, X_train=None, y_train=None, X_test=None, y_test=None,
                validate=False, logger=None):
    """Displays train and test performance for a learned model."""

    if not logger:
        return

    logger.info('')

    if validate:
        model_type = validate_model(model)
        logger.info('model ({})'.format(model_type))

    acc, auc, logloss = -1, -1, -1

    if X_train is not None and y_train is not None:
        y_hat_pred = model.predict(X_train).flatten()
        acc = accuracy_score(y_train, y_hat_pred)

        if hasattr(model, 'predict_proba'):
            y_hat_proba = model.predict_proba(X_train)
            logloss = log_loss(y_train, y_hat_proba)

            if len(np.unique(y_train)) == 2:
                auc = roc_auc_score(y_train, y_hat_proba[:, 1])

        if hasattr(model, 'decision_function') and len(np.unique(y_train)) == 2:
            y_hat_proba = model.decision_function(X_train)
            auc = roc_auc_score(y_train, y_hat_proba)

    s = 'train acc: {:.3f}, auc: {:.3f}, logloss: {:.3f}'
    logger.info(s.format(acc, auc, logloss))

    if X_test is not None and y_test is not None:
        y_hat_pred = model.predict(X_test).flatten()
        acc = accuracy_score(y_test, y_hat_pred)

        if hasattr(model, 'predict_proba'):
            y_hat_proba = model.predict_proba(X_test)
            logloss = log_loss(y_test, y_hat_proba)

            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, y_hat_proba[:, 1])

        if hasattr(model, 'decision_function') and len(np.unique(y_test)) == 2:
            y_hat_proba = model.decision_function(X_test)
            auc = roc_auc_score(y_test, y_hat_proba)

    s = 'test  acc: {:.3f}, auc: {:.3f}, logloss: {:.3f}'
    logger.info(s.format(acc, auc, logloss))


def validate_model(model):
    """Make sure the model is a supported model type."""

    model_type = str(model).split('(')[0]
    if 'RandomForestClassifier' in str(model):
        model_type = 'RandomForestClassifier'
    elif 'GradientBoostingClassifier' in str(model):
        model_type = 'GradientBoostingClassifier'
    elif 'LGBMClassifier' in str(model):
        model_type = 'LGBMClassifier'
    elif 'CatBoostClassifier' in str(model):
        model_type = 'CatBoostClassifier'
    elif 'XGBClassifier' in str(model):
        model_type = 'XGBClassifier'
    elif model_type == 'OneVsRestClassifier':
        model_type = 'OneVsRestClassifier'
    elif model_type == 'SVC':
        model_type = 'SVC'
    elif 'TreeExplainer' in str(model):
        model_type = 'trex'
    else:
        exit('{} model not currently supported!'.format(str(model)))

    return model_type


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
