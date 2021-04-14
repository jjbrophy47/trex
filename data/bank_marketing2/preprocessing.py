"""
Preprocesses dataset but keep continuous variables.
"""
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def dataset_specific(random_state):

    # retrieve dataset
    df = pd.read_csv('bank-additional_bank-additional-full.csv', sep=';')

    # categorize attributes
    columns = list(df.columns)
    label = ['y']
    numeric = ['age', 'duration', 'campaign', 'pdays', 'previous',
               'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
               'euribor3m', 'nr.employed']
    categorical = list(set(columns) - set(numeric) - set(label))

    # remove select columns
    remove_cols = []
    if len(remove_cols) > 0:
        df = df.drop(columns=remove_cols)

    # remove nan rows
    nan_rows = df[df.isnull().any(axis=1)]
    print('nan rows: {}'.format(len(nan_rows)))
    df = df.dropna()

    # split into a 60/20/20 train/dist/val/test split
    n_total = len(df)
    n_split = int(0.2 * n_total)
    train_df, test_df = train_test_split(df, test_size=n_split, stratify=df[label], random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=n_split, stratify=train_df[label],
                                        random_state=random_state)
    train_df = train_df[:1000]

    return train_df, val_df, test_df, label, numeric, categorical


def main(random_state=1, out_dir='.'):

    train_df, val_df, test_df, label, numeric, categorical = dataset_specific(random_state=random_state)

    # encode categorical inputs
    ct = ColumnTransformer([('kbd', 'passthrough', numeric),
                            ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical)])
    train = ct.fit_transform(train_df)
    val = ct.fit_transform(val_df)
    test = ct.transform(test_df)

    # binarize outputs
    le = LabelEncoder()
    train_label = le.fit_transform(train_df[label].to_numpy().ravel()).reshape(-1, 1)
    val_label = le.fit_transform(val_df[label].to_numpy().ravel()).reshape(-1, 1)
    test_label = le.transform(test_df[label].to_numpy().ravel()).reshape(-1, 1)

    # add labels
    train = np.hstack([train, train_label]).astype(np.float32)
    val = np.hstack([val, val_label]).astype(np.float32)
    test = np.hstack([test, test_label]).astype(np.float32)

    print('\ntrain:\n{}, dtype: {}'.format(train, train.dtype))
    print('train.shape: {}, label sum: {}'.format(train.shape, train[:, -1].sum()))

    print('\nval:\n{}, dtype: {}'.format(val, val.dtype))
    print('val.shape: {}, label sum: {}'.format(val.shape, val[:, -1].sum()))

    print('\ntest:\n{}, dtype: {}'.format(test, test.dtype))
    print('test.shape: {}, label sum: {}'.format(test.shape, test[:, -1].sum()))

    # save to numpy format
    print('saving...')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'train.npy'), train)
    np.save(os.path.join(out_dir, 'val.npy'), val)
    np.save(os.path.join(out_dir, 'test.npy'), test)


if __name__ == '__main__':
    main()
