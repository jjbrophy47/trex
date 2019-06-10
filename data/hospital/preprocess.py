"""
Preprocess dataset to make it easier to load and work with.
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder


def main():

    # fill NaN's with <UNK> token
    unk_token = '<UNK>'

    # retrieve dataset
    start = time.time()
    if os.path.exists('raw'):
        data = pd.read_csv('raw/diabetic_data.csv')
        train = data[:20000]  # same split as Scharchilev et al.
        test = data[20000:]

    print('time to load hospital: {}'.format(time.time() - start))

    # define columns
    label_col = 'readmitted'
    feature_col = list(train.columns)
    feature_col.remove(label_col)
    data[label_col] = data[label_col].apply(lambda x: 'YES' if x != 'NO' else 'NO')

    # nan rows
    train_nan_rows = train[train.isnull().any(axis=1)]
    test_nan_rows = test[test.isnull().any(axis=1)]
    print('train nan rows: {}'.format(len(train_nan_rows)))
    print('test nan rows: {}'.format(len(test_nan_rows)))

    # fit encoders and fill in NaNs with unknown token or mean value
    encoders = {}
    for col in feature_col:
        if str(data[col].dtype) == 'object':
            train[col] = train[col].fillna(unk_token)
            test[col] = test[col].fillna(unk_token)
            encoders[col] = OrdinalEncoder().fit(data[col].to_numpy().reshape(-1, 1))
        else:
            train[col] = train[col].fillna(int(train[col].mean()))
            test[col] = test[col].fillna(int(test[col].mean()))
    label_encoder = LabelEncoder().fit(data[label_col])

    # transform train dataframe
    new_train = train.copy()
    for col in feature_col:
        if col in encoders:
            new_train[col] = encoders[col].transform(new_train[col].to_numpy().reshape(-1, 1))
    new_train[label_col] = label_encoder.transform(new_train[label_col])

    # transform test dataframe
    new_test = test.copy()
    for col in feature_col:
        if col in encoders:
            new_test[col] = encoders[col].transform(new_test[col].to_numpy().reshape(-1, 1))
    new_test[label_col] = label_encoder.transform(new_test[label_col])

    # show difference
    print('train')
    print(train.head(5))
    print(new_train.head(5))

    print('test')
    print(test.head(5))
    print(new_test.head(5))

    # save to numpy format
    print('saving to train.npy...')
    np.save('train.npy', new_train.to_numpy())
    print('saving to test.npy...')
    np.save('test.npy', new_test.to_numpy())


if __name__ == '__main__':
    main()
