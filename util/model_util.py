"""
Utility methods to make life easier.
"""
import numpy as np
import catboost
import lightgbm
import xgboost
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score


def get_classifier(model, n_estimators=20, random_state=69):
    """Returns a tree ensemble classifier."""

    # create model
    if model == 'lgb':
        clf = lightgbm.LGBMClassifier(random_state=random_state, n_estimators=n_estimators)
    elif model == 'cb':
        clf = catboost.CatBoostClassifier(random_state=random_state, n_estimators=n_estimators, verbose=False)
    elif model == 'rf':
        clf = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators)
    elif model == 'gbm':
        clf = GradientBoostingClassifier(random_state=random_state, n_estimators=n_estimators)
    elif model == 'xgb':
        clf = xgboost.XGBClassifier(random_state=random_state, n_estimators=n_estimators)
    else:
        exit('{} model not supported!')

    return clf


def fidelity(y1, y2):
    """Returns an (overlap, difference) tuple."""

    overlap = np.where(y1 == y2)[0]
    difference = np.where(y1 != y2)[0]

    return overlap, difference


def missed_instances(y1, y2, y_true):
    """Returns indexes missed by both y1 and y2."""

    both_ndx = np.where((y1 == y2) & (y1 != y_true))[0]
    return both_ndx


def performance(model, X_train, y_train, X_test=None, y_test=None):
    """Displays train and test performance for a learned model."""

    model_type = validate_model(model)

    y_hat_train = model.predict(X_train).flatten()
    tree_missed_train = np.where(y_hat_train != y_train)[0]
    print('\nModel ({})'.format(model_type))
    print('train set acc: {:4f}'.format(accuracy_score(y_train, y_hat_train)))
    print('missed train instances ({}): {}'.format(len(tree_missed_train), tree_missed_train))
    result = y_hat_train

    if X_test is not None and y_test is not None:
        y_hat_test = model.predict(X_test).flatten()
        tree_missed_test = np.where(y_hat_test != y_test)[0]
        print('test set acc: {:4f}'.format(accuracy_score(y_test, y_hat_test)))
        print('missed test instances ({}): {}'.format(len(tree_missed_test), tree_missed_test))
        result = y_hat_train, y_hat_test

    return result


def validate_model(model):
    """Make sure the model is a supported model type."""

    model_type = str(model).split('(')[0]
    if 'RandomForestClassifier' in str(model):
        model_type = 'RandomForestClassifier'
    if 'GradientBoostingClassifier' in str(model):
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
    else:
        exit('{} model not currently supported!'.format(str(model)))

    return model_type
