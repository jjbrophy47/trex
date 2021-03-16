"""
Utility methods to make life easier.
"""
import os

import numpy as np


def get_data(dataset, data_dir='data', preprocessing='categorical', mismatch=False):
    """
    Returns a train and test set from the desired dataset.
    """
    in_dir = os.path.join(data_dir, dataset)

    # append processing directory
    assert preprocessing in ['categorical', 'standard']
    in_dir = os.path.join(in_dir, preprocessing)
    assert os.path.exists(in_dir)

    # load in data
    train_fn = 'train_mismatch.npy' if (dataset == 'adult' and mismatch) else 'train.npy'
    train = np.load(os.path.join(in_dir, train_fn), allow_pickle=True)
    test = np.load(os.path.join(in_dir, 'test.npy'), allow_pickle=True)
    feature = np.load(os.path.join(in_dir, 'feature.npy'))

    # load in catgeorical feature indices
    if preprocessing == 'categorical':
        cat_indices = np.load(os.path.join(in_dir, 'cat_indices.npy'))

    # standard preprocessing
    else:
        cat_indices = None
        train = train.astype(np.float32)
        test = test.astype(np.float32)

    # make sure labels are binary
    assert np.all(np.unique(train[:, -1]) == np.array([0, 1]))
    assert np.all(np.unique(test[:, -1]) == np.array([0, 1]))

    # split feature values and labels
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, X_test, y_train, y_test, feature, cat_indices
