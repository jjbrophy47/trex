"""
Utility methods to make life easier.
"""
import os

import numpy as np


def get_data(dataset, data_dir='data', processing_dir='categorical'):
    """
    Returns a train and test set from the desired dataset.
    """
    in_dir = os.path.join(data_dir, dataset)

    # append processing directory
    assert processing_dir in ['categorical', 'standard']
    in_dir = os.path.join(in_dir, processing_dir)

    # load in data
    assert os.path.exists(in_dir)
    train = np.load(os.path.join(in_dir, 'train.npy'), allow_pickle=True)
    test = np.load(os.path.join(in_dir, 'test.npy'), allow_pickle=True)
    feature = np.load(os.path.join(in_dir, 'feature.npy'))

    # load in catgeorical feature indices
    if processing_dir == 'categorical':
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


# def flip_labels(arr, k=100, seed=1, return_indices=True, logger=None):
#     """
#     Flips the label of random elements in an array; only for binary arrays.
#     """
#     assert arr.ndim == 1, 'arr is not 1d!'
#     assert np.all(np.unique(arr) == np.array([0, 1])), 'arr is not binary!'
#     if k <= 1.0:
#         assert isinstance(k, float), 'k is not a float!'
#         assert k > 0, 'k is less than zero!'
#         k = int(len(arr) * k)
#     assert k <= len(arr), 'k is greater than len(arr)!'

#     np.random.seed(random_state)
#     indices = np.random.choice(np.arange(len(arr)), size=k, replace=False)

#     new_arr = arr.copy()
#     ones_flipped = 0
#     zeros_flipped = 0

#     for ndx in indices:
#         if new_arr[ndx] == 1:
#             ones_flipped += 1
#         else:
#             zeros_flipped += 1
#         new_arr[ndx] = 0 if new_arr[ndx] == 1 else 1

#     if logger:
#         logger.info('sum before: {:,}'.format(np.sum(arr)))
#         logger.info('ones flipped: {:,}'.format(ones_flipped))
#         logger.info('zeros flipped: {:,}'.format(zeros_flipped))
#         logger.info('sum after: {:,}'.format(np.sum(new_arr)))

#     assert np.sum(new_arr) == np.sum(arr) - ones_flipped + zeros_flipped

#     if return_indices:
#         return new_arr, indices
#     else:
#         return new_arr


# def flip_labels_with_indices(arr, indices):
#     """
#     Flips the label of specified elements in an array; only for binary arrays.
#     """

#     assert arr.ndim == 1, 'arr is not 1d!'
#     assert np.all(np.unique(arr) == np.array([0, 1])), 'arr is not binary!'

#     new_arr = arr.copy()
#     for ndx in indices:
#         new_arr[ndx] = 0 if new_arr[ndx] == 1 else 1
#     return new_arr
