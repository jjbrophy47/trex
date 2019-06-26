"""
Utility methods to make life easier.
"""
import os
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split


def get_data(dataset, test_size=0.2, random_state=69, data_dir='data', return_feature=False):
    """Returns a train and test set from the desired dataset."""

    X = None
    y = None
    label = None

    # load dataset
    if dataset == 'iris':
        data = load_iris()
    elif dataset == 'breast':
        data = load_breast_cancer()
    elif dataset == 'wine':
        data = load_wine()

    elif dataset == 'adult':
        train = np.load(os.path.join(data_dir, 'adult/train.npy'))
        test = np.load(os.path.join(data_dir, 'adult/test.npy'))
        label = ['<=50K', '>50k']
        X_train = train[:, :-1]
        y_train = train[:, -1].astype(np.int32)
        X_test = test[:, :-1]
        y_test = test[:, -1].astype(np.int32)
        return X_train, X_test, y_train, y_test, label

    elif dataset == 'amazon':
        train = np.load(os.path.join(data_dir, 'amazon/train.npy'))
        test = np.load(os.path.join(data_dir, 'amazon/test.npy'))
        label = ['0', '1']
        X_train = train[:, 1:]
        y_train = train[:, 0].astype(np.int32)
        X_test = test[:, 1:]
        y_test = test[:, 0].astype(np.int32)
        return X_train, X_test, y_train, y_test, label

    elif dataset == 'hospital':
        train = np.load(os.path.join(data_dir, 'hospital/train.npy'))
        test = np.load(os.path.join(data_dir, 'hospital/test.npy'))
        label = ['not readmitted', 'readmitted']
        X_train = train[:, :-1]
        y_train = train[:, -1].astype(np.int32)
        X_test = test[:, :-1]
        y_test = test[:, -1].astype(np.int32)
        return X_train, X_test, y_train, y_test, label

    elif dataset == 'medifor':
        train = np.load(os.path.join(data_dir, 'medifor/NC17_EvalPart1.npy'))
        test = np.load(os.path.join(data_dir, 'medifor/MFC18_EvalPart1.npy'))
        feature = np.load(os.path.join(data_dir, 'medifor/feature.npy'))
        label = ['non-manipulated', 'manipulated']
        X_train = train[:, :-1]
        y_train = train[:, -1].astype(np.int32)
        X_test = test[:, :-1]
        y_test = test[:, -1].astype(np.int32)
        if return_feature:
            return X_train, X_test, y_train, y_test, label, feature
        else:
            return X_train, X_test, y_train, y_test, label

    elif dataset == 'medifor2':
        data = np.load(os.path.join(data_dir, 'medifor/NC17_EvalPart1.npy'))
        label = ['non-manipulated', 'manipulated']
        X = data[:, :-1]
        y = data[:, -1].astype(np.int32)

    if X is None and y is None and label is None:
        X = data['data']
        y = data['target']
        label = data['target_names']

    if test_size is None:
        return X, y, label

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
        data = X_train, X_test, y_train, y_test, label
        return data


def flip_labels(arr, k=100, random_state=69, return_indices=True):
    """Flips the label of random elements in an array; only for binary arrays."""

    assert arr.ndim == 1, 'arr is not 1d!'
    assert np.all(np.unique(arr) == np.array([0, 1])), 'arr is not binary!'
    assert k <= len(arr), 'k is greater than len(arr)!'

    np.random.seed(random_state)
    indices = np.random.choice(np.arange(len(arr)), size=k, replace=False)

    new_arr = arr.copy()
    for ndx in indices:
        new_arr[ndx] = 0 if new_arr[ndx] == 1 else 1

    if return_indices:
        return new_arr, indices
    else:
        return new_arr


def flip_labels_with_indices(arr, indices):
    """Flips the label of specified elements in an array; only for binary arrays."""

    assert arr.ndim == 1, 'arr is not 1d!'
    assert np.all(np.unique(arr) == np.array([0, 1])), 'arr is not binary!'

    new_arr = arr.copy()
    for ndx in indices:
        new_arr[ndx] = 0 if new_arr[ndx] == 1 else 1
    return new_arr
