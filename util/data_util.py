"""
Utility methods to make life easier.
"""
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split


def get_data(dataset, test_size=0.2, random_state=69):

    # load dataset
    if dataset == 'iris':
        data = load_iris()
    elif dataset == 'breast':
        data = load_breast_cancer()
    elif dataset == 'wine':
        data = load_wine()

    elif dataset == 'adult':
        train = np.load('data/adult/train.npy')
        test = np.load('data/adult/test.npy')
        label = ['<=50K', '>50k']
        X_train = train[:, :-1]
        y_train = train[:, -1].astype(np.int32)
        X_test = test[:, :-1]
        y_test = test[:, -1].astype(np.int32)
        return X_train, X_test, y_train, y_test, label

    elif dataset == 'medifor':
        train = np.load('data/medifor/NC17_EvalPart1.npy')
        test = np.load('data/medifor/MFC18_EvalPart1.npy')
        label = ['non-manipulated', 'manipulated']
        X_train = train[:, :-1]
        y_train = train[:, -1].astype(np.int32)
        X_test = test[:, :-1]
        y_test = test[:, -1].astype(np.int32)
        return X_train, X_test, y_train, y_test, label

    X = data['data']
    y = data['target']
    label = data['target_names']

    if test_size is None:
        return X, y, label

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
        data = X_train, X_test, y_train, y_test, label
        return data
