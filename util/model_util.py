"""
Utility methods to make life easier.
"""
import numpy as np
from sklearn.metrics import accuracy_score


def performance(model, X_train, y_train, X_test=None, y_test=None):
    """Displays train and test performance for a learned model."""

    model_type = validate_model(model)

    y_hat_train = model.predict(X_train).flatten()
    tree_missed_train = np.where(y_hat_train != y_train)[0]
    print('\nModel ({})'.format(model_type))
    print('train set acc: {:4f}'.format(accuracy_score(y_train, y_hat_train)))
    print('missed train instances ({}): {}'.format(len(tree_missed_train), tree_missed_train))

    if X_test is not None and y_test is not None:
        y_hat_test = model.predict(X_test).flatten()
        tree_missed_test = np.where(y_hat_test != y_test)[0]
        print('test set acc: {:4f}'.format(accuracy_score(y_test, y_hat_test)))
        print('missed test instances ({}): {}'.format(len(tree_missed_test), tree_missed_test))


def validate_model(model):
    """Make sure the model is a supported model type."""

    model_type = str(model).split('(')[0]
    if 'RandomForestClassifier' in str(model):
        model_type = 'RandomForestClassifier'
    elif 'LGBMClassifier' in str(model):
        model_type = 'LGBMClassifier'
    elif 'CatBoostClassifier' in str(model):
        model_type = 'CatBoostClassifier'
    elif model_type == 'OneVsRestClassifier':
        model_type = 'OneVsRestClassifier'
    elif model_type == 'SVC':
        model_type = 'SVC'
    else:
        exit('{} model not currently supported!'.format(str(model)))

    return model_type
