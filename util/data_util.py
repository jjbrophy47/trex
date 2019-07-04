"""
Utility methods to make life easier.
"""
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split


def _load_iris(test_size=0.2, random_state=69):
    data = load_iris()
    X, y, label = data['data'], data['target'], data['target_names']

    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test, label
    else:
        return X, y, label


def _load_breast(test_size=0.2, random_state=69):
    data = load_breast_cancer()
    X, y, label = data['data'], data['target'], data['target_names']

    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test, label
    else:
        return X, y, label


def _load_wine(test_size=0.2, random_state=69):
    data = load_wine()
    X, y, label = data['data'], data['target'], data['target_names']

    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test, label
    else:
        return X, y, label


def _load_adult(data_dir='data'):
    train = np.load(os.path.join(data_dir, 'adult/train.npy'))
    test = np.load(os.path.join(data_dir, 'adult/test.npy'))
    label = ['<=50K', '>50k']
    X_train = train[:, :-1]
    y_train = train[:, -1].astype(np.int32)
    X_test = test[:, :-1]
    y_test = test[:, -1].astype(np.int32)
    return X_train, X_test, y_train, y_test, label


def _load_amazon(data_dir='data'):
    train = np.load(os.path.join(data_dir, 'amazon/train.npy'))
    test = np.load(os.path.join(data_dir, 'amazon/test.npy'))
    label = ['0', '1']
    X_train = train[:, 1:]
    y_train = train[:, 0].astype(np.int32)
    X_test = test[:, 1:]
    y_test = test[:, 0].astype(np.int32)
    return X_train, X_test, y_train, y_test, label


def _load_churn(data_dir='data', test_size=0.2, random_state=69):
    data = np.load(os.path.join(data_dir, 'churn/data.npy'))
    label = ['no', 'yes']
    X = data[:, :-1]
    y = data[:, -1].astype(np.int32)

    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test, label
    else:
        return X, y, label


def _load_creditcard(data_dir='data', test_size=0.2, random_state=69):
    data = np.load(os.path.join(data_dir, 'creditcard/data.npy'))
    label = ['0', '1']
    X = data[:, :-1]
    y = data[:, -1].astype(np.int32)

    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test, label
    else:
        return X, y, label


def _load_heart(data_dir='data', test_size=0.2, random_state=69):
    data = np.load(os.path.join(data_dir, 'heart/data.npy'))
    label = ['other', 'Hungary']
    X = data[:, :-1]
    y = data[:, -1].astype(np.int32)

    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test, label
    else:
        return X, y, label


def _load_hospital(data_dir='data'):
    train = np.load(os.path.join(data_dir, 'hospital/train.npy'))
    test = np.load(os.path.join(data_dir, 'hospital/test.npy'))
    label = ['not readmitted', 'readmitted']
    X_train = train[:, :-1]
    y_train = train[:, -1].astype(np.int32)
    X_test = test[:, :-1]
    y_test = test[:, -1].astype(np.int32)
    return X_train, X_test, y_train, y_test, label


def _load_nc17_mfc18(data_dir='data', feature=False, switch=False):
    train = np.load(os.path.join(data_dir, 'nc17_mfc18/NC17_EvalPart1.npy'))
    test = np.load(os.path.join(data_dir, 'nc17_mfc18/MFC18_EvalPart1.npy'))
    label = ['non-manipulated', 'manipulated']
    X_train = train[:, :-1]
    y_train = train[:, -1].astype(np.int32)
    X_test = test[:, :-1]
    y_test = test[:, -1].astype(np.int32)

    if switch:
        X_temp, y_temp = X_train.copy(), y_train.copy()
        X_train, y_train = X_test, y_test
        X_test, y_test = X_temp, y_temp

    if feature:
        feature = np.load(os.path.join(data_dir, 'nc17_mfc18/feature.npy'))
        return X_train, X_test, y_train, y_test, label, feature
    else:
        return X_train, X_test, y_train, y_test, label


def _load_mfc18_mfc19(data_dir='data', feature=False, switch=False):
    train = np.load(os.path.join(data_dir, 'mfc18_mfc19/MFC18_EvalPart1.npy'))
    test = np.load(os.path.join(data_dir, 'mfc18_mfc19/MFC19_EvalPart1.npy'))
    label = ['non-manipulated', 'manipulated']
    X_train = train[:, :-1]
    y_train = train[:, -1].astype(np.int32)
    X_test = test[:, :-1]
    y_test = test[:, -1].astype(np.int32)

    if switch:
        X_temp, y_temp = X_train.copy(), y_train.copy()
        X_train, y_train = X_test, y_test
        X_test, y_test = X_temp, y_temp

    if feature:
        feature = np.load(os.path.join(data_dir, 'mfc18_mfc19/feature.npy'))
        return X_train, X_test, y_train, y_test, label, feature
    else:
        return X_train, X_test, y_train, y_test, label


def _load_medifor(data_dir='data', test_size=0.2, random_state=69, return_feature=False, return_manipulation=False,
                  return_image_id=False, dataset='MFC18_EvalPart1'):

    data = np.load(os.path.join(data_dir, dataset, 'data.npy'))
    label = ['non-manipulated', 'manipulated']
    X = data[:, :-1]
    y = data[:, -1].astype(np.int32)

    result = tuple()

    if test_size is not None:
        train_ndx, test_ndx = train_test_split(np.arange(len(X)), test_size=test_size,
                                               random_state=random_state, stratify=y)
        X_train, y_train = X[train_ndx], y[train_ndx]
        X_test, y_test = X[train_ndx], y[train_ndx]
        result += (X_train, X_test, y_train, y_test, label)

        if return_feature:
            feature = np.load(os.path.join(data_dir, dataset, 'feature.npy'))
            result += (feature,)

        if return_manipulation:
            manipulation = pd.read_csv(os.path.join(data_dir, dataset, 'manipulations.csv'))
            manip_train, manip_test = manipulation[train_ndx], manipulation[test_ndx]
            result += (manip_train, manip_test)

        if return_image_id:
            assert dataset in ['MFC18_EvalPart1', 'MFC19_EvalPart1'], 'image_id not supported for {}'.format(dataset)
            image_name = pd.read_csv(os.path.join(data_dir, dataset, 'reference.csv'))['image_id'].values
            id_train, id_test = image_name[train_ndx], image_name[test_ndx]
            result += (id_train, id_test)

        return result

    else:
        result = (X, y, label)

        if return_feature:
            feature = np.load(os.path.join(data_dir, dataset, 'feature.npy'))
            result += (feature,)

        if return_manipulation:
            manipulation = pd.read_csv(os.path.join(data_dir, dataset, 'reference.csv'))
            result += (manipulation,)

        if return_image_id:
            assert dataset in ['MFC18_EvalPart1', 'MFC19_EvalPart1'], 'image_id not supported for {}'.format(dataset)
            image_id = pd.read_csv(os.path.join(data_dir, dataset, 'reference.csv'))['image_id'].values
            result += (image_id,)

        return result


def get_data(dataset, test_size=0.2, random_state=69, data_dir='data', return_feature=False,
             return_manipulations=False, return_image_id=False):
    """Returns a train and test set from the desired dataset."""

    # load dataset
    if dataset == 'iris':
        return _load_iris()
    elif dataset == 'breast':
        return _load_breast()
    elif dataset == 'wine':
        return _load_wine()
    elif dataset == 'adult':
        return _load_adult(data_dir=data_dir)
    elif dataset == 'amazon':
        return _load_amazon(data_dir=data_dir)
    elif dataset == 'churn':
        return _load_churn(data_dir=data_dir, test_size=test_size, random_state=random_state)
    elif dataset == 'creditcard':
        return _load_creditcard(data_dir=data_dir, test_size=test_size, random_state=random_state)
    elif dataset == 'heart':
        return _load_heart(data_dir=data_dir, test_size=test_size, random_state=random_state)
    elif dataset == 'hospital':
        return _load_hospital(data_dir=data_dir)
    elif dataset == 'nc17_mfc18':
        return _load_nc17_mfc18(data_dir=data_dir, feature=return_feature)
    elif dataset == 'nc17_mfc18_switch':
        return _load_nc17_mfc18(data_dir=data_dir, feature=return_feature, switch=True)
    elif dataset == 'mfc18_mfc19':
        return _load_mfc18_mfc19(data_dir=data_dir, feature=return_feature)
    elif dataset == 'mfc18_mfc19_switch':
        return _load_mfc18_mfc19(data_dir=data_dir, feature=return_feature, switch=True)
    elif dataset == 'NC17_EvalPart1':
        return _load_medifor(data_dir=data_dir, feature=return_feature, image_id=return_image_id, dataset=dataset,
                             test_size=test_size, random_state=random_state)
    elif dataset == 'MFC18_EvalPart1':
        return _load_medifor(data_dir=data_dir, feature=return_feature, manipulations=return_manipulations,
                             image_id=return_image_id, dataset=dataset, test_size=test_size, random_state=random_state)
    elif dataset == 'MFC19_EvalPart1':
        return _load_medifor(data_dir=data_dir, feature=return_feature, manipulations=return_manipulations,
                             image_id=return_image_id, dataset=dataset, test_size=test_size, random_state=random_state)


def flip_labels(arr, k=100, random_state=69, return_indices=True):
    """Flips the label of random elements in an array; only for binary arrays."""

    assert arr.ndim == 1, 'arr is not 1d!'
    assert np.all(np.unique(arr) == np.array([0, 1])), 'arr is not binary!'
    if k <= 1.0:
        assert isinstance(k, float), 'k is not a float!'
        assert k > 0, 'k is less than zero!'
        k = int(len(arr) * k)
    assert k <= len(arr), 'k is greater than len(arr)!'

    np.random.seed(random_state)
    indices = np.random.choice(np.arange(len(arr)), size=k, replace=False)

    new_arr = arr.copy()
    ones_flipped = 0
    zeros_flipped = 0

    for ndx in indices:
        if new_arr[ndx] == 1:
            ones_flipped += 1
        else:
            zeros_flipped += 1
        new_arr[ndx] = 0 if new_arr[ndx] == 1 else 1

    print('sum before: {}'.format(np.sum(arr)))
    print('ones flipped: {}'.format(ones_flipped))
    print('zeros flipped: {}'.format(zeros_flipped))
    print('sum after: {}'.format(np.sum(new_arr)))
    assert np.sum(new_arr) == np.sum(arr) - ones_flipped + zeros_flipped

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
