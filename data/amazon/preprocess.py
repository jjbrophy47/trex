"""
Preprocess dataset to make it easier to load and work with.
"""
import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
from utility import util


def main(args):

    # create output directory
    if args.processing == 'standard':
        out_dir = 'standard'
    elif args.processing == 'categorical':
        out_dir = 'categorical'
    else:
        raise ValueError('args.processing: {} unknown!'.format(args.processing))
    os.makedirs(os.path.join(out_dir), exist_ok=True)

    # create logger
    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info('{}'.format(args))
    logger.info('\ntimestamp: {}'.format(datetime.now()))

    # retrieve dataset
    start = time.time()
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    logger.info('\ntime to read in data...{:.3f}s'.format(time.time() - start))

    # get features
    columns = list(train_df.columns)

    logger.info('train_df:\n{}\n{}'.format(train_df.head(5), train_df.shape))
    logger.info('test_df:\n{}\n{}'.format(test_df.head(5), test_df.shape))

    # count number of NAN values per column
    logger.info('')
    for c in train_df.columns:
        logger.info('[TRAIN] {}, no. missing: {:,}'.format(c, train_df[c].isna().sum()))
        logger.info('[TEST] {}, no. missing: {:,}'.format(c, test_df[c].isna().sum()))

    # remove select columns
    remove_cols = []
    if len(remove_cols) > 0:
        train_df = train_df.drop(columns=remove_cols)
        test_df = test_df.drop(columns=remove_cols)

    # categorize attributes
    label = ['ACTION']
    numeric_features = []
    categorical_features = list(set(columns) - set(numeric_features) - set(label))

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))]
    )

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))]
    )

    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]
    )

    # time transforms
    start = time.time()

    # perform one-hot encoding for all cat. attributes
    if args.processing == 'standard':

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
            ],
            sparse_threshold=0
        )

        # encode features
        train = preprocessor.fit_transform(train_df)
        test = preprocessor.transform(test_df)

        # encode labels
        le = LabelEncoder()
        train_label = le.fit_transform(train_df[label].values.ravel()).reshape(-1, 1)
        test_label = le.transform(test_df[label].values.ravel()).reshape(-1, 1)

        # dense matrix
        train = np.hstack([train, train_label]).astype(np.float32)
        test = np.hstack([test, test_label]).astype(np.float32)

    # leave categorical attributes as is
    elif args.processing == 'categorical':

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('ord', ordinal_transformer, categorical_features),
            ],
            sparse_threshold=0
        )

        # encode features
        train = preprocessor.fit_transform(train_df)
        test = preprocessor.transform(test_df)

        # encode labels
        le = LabelEncoder()
        train_label = le.fit_transform(train_df[label].values.ravel()).reshape(-1, 1)
        test_label = le.transform(test_df[label].values.ravel()).reshape(-1, 1)

        # add labels
        train = np.hstack([train, train_label])
        test = np.hstack([test, test_label])

    else:
        raise ValueError('args.processing: {} unknown!'.format(args.processing))

    logger.info('transforming features...{:.3f}s'.format(time.time() - start))

    feature_list = util.get_feature_names(preprocessor)
    assert len(feature_list) == train.shape[1] - 1 == test.shape[1] - 1

    # display statistics
    logger.info('\ntrain:\n{}, dtype: {}'.format(train, train.dtype))
    logger.info('train.shape: {}, label sum: {}'.format(train.shape, train[:, -1].sum()))

    logger.info('\ntest:\n{}, dtype: {}'.format(test, test.dtype))
    logger.info('test.shape: {}, label sum: {}'.format(test.shape, test[:, -1].sum()))

    logger.info('\nfeatures:\n{}'.format(feature_list))

    # save to numpy format
    logger.info('\nsaving to {}/...'.format(out_dir))
    np.save(os.path.join(out_dir, 'train.npy'), train)
    np.save(os.path.join(out_dir, 'test.npy'), test)
    np.save(os.path.join(out_dir, 'feature.npy'), feature_list)

    # save categorical feature names and indices
    if args.processing == 'categorical':
        categorical_indices = np.arange(len(numeric_features), len(numeric_features) + len(categorical_features))
        logger.info('\ncategorical feature names:\n{}'.format(categorical_features))
        logger.info('\ncategorical feature indices:\n{}'.format(categorical_indices))
        np.save(os.path.join(out_dir, 'cat_feature.npy'), categorical_features)
        np.save(os.path.join(out_dir, 'cat_indices.npy'), categorical_indices)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processing', type=str, default='categorical', help='regular or categorical.')
    args = parser.parse_args()
    main(args)
