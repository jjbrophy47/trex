"""
Utility methods to make life easier.
"""
import numpy as np
import catboost
import lightgbm
import xgboost
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss


def get_classifier(model, n_estimators=20, learning_rate=0.03, random_state=69):
    """Returns a tree ensemble classifier."""

    # create model
    if model == 'lgb':
        clf = lightgbm.LGBMClassifier(random_state=random_state, n_estimators=n_estimators)
    elif model == 'cb':
        clf = catboost.CatBoostClassifier(random_state=random_state, n_estimators=n_estimators, verbose=False)
        # clf = catboost.CatBoostClassifier(random_state=random_state, n_estimators=n_estimators,
        #                                   learning_rate=learning_rate, verbose=False)
    elif model == 'rf':
        clf = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators)
    elif model == 'gbm':
        clf = GradientBoostingClassifier(random_state=random_state, n_estimators=n_estimators)
    elif model == 'xgb':
        clf = xgboost.XGBClassifier(random_state=random_state, n_estimators=n_estimators)
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


def performance(model, X_train=None, y_train=None, X_test=None, y_test=None):
    """Displays train and test performance for a learned model."""

    model_type = validate_model(model)

    result = tuple()

    if X_train is not None and y_train is not None:
        y_hat_pred = model.predict(X_train).flatten()
        tree_missed_train = np.where(y_hat_pred != y_train)[0]
        acc_train = accuracy_score(y_train, y_hat_pred)

        print('model ({})'.format(model_type))
        print('train set acc: {:4f}'.format(acc_train))
        print('missed train instances ({})'.format(len(tree_missed_train)))

        if hasattr(model, 'predict_proba'):
            y_hat_proba = model.predict_proba(X_train)
            ll_train = log_loss(y_train, y_hat_proba)
            print('train log loss: {:.5f}'.format(ll_train))

        result += (y_hat_pred,)

    if X_test is not None and y_test is not None:
        y_hat_pred = model.predict(X_test).flatten()
        tree_missed_test = np.where(y_hat_pred != y_test)[0]
        acc_test = accuracy_score(y_test, y_hat_pred)

        print('model ({})'.format(model_type))
        print('test set acc: {:4f}'.format(acc_test))
        print('missed test instances ({})'.format(len(tree_missed_test)))

        if hasattr(model, 'predict_proba'):
            y_hat_proba = model.predict_proba(X_test)
            ll_test = log_loss(y_test, y_hat_proba)
            print('test log loss: {:.5f}'.format(ll_test))

        result += (y_hat_pred,)

    return result


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
    """Given the predicted label of each sample and the probabilities for each class for each sample,
    return the probabilities of the positive class for each sample."""

    assert labels.ndim == 1, 'labels is not 1d!'
    assert probas.ndim == 2, 'probas is not 2d!'
    assert len(labels) == len(probas), 'num samples do not match between labels and probas!'
    y_pred = probas[np.arange(len(labels)), labels]
    assert y_pred.ndim == 1, 'y_pred is not 1d!'
    return y_pred
