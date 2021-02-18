"""
Utility methods for displaying data.
"""
import os
import sys
import time
import logging
from datetime import datetime

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_logger(filename=''):
    """
    Return a logger object.
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    log_handler = logging.FileHandler(filename, mode='w')
    formatter = logging.Formatter('%(message)s')

    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(log_handler)

    return logger


def get_feature_names(column_transformer, logger=None):
    """
    Extract feature names from a ColumnTransformer object.
    """
    col_name = []

    # the last transformer is ColumnTransformer's 'remainder'
    for transformer_in_columns in column_transformer.transformers_[:-1]:
        if logger:
            logger.info('\n\ntransformer: ', transformer_in_columns[0])

        raw_col_name = list(transformer_in_columns[2])

        # if pipeline, get the last transformer
        if isinstance(transformer_in_columns[1], Pipeline):
            transformer = transformer_in_columns[1].steps[-1][1]

        else:
            transformer = transformer_in_columns[1]

        try:
            if isinstance(transformer, OneHotEncoder):
                names = list(transformer.get_feature_names(raw_col_name))

            elif isinstance(transformer, SimpleImputer) and transformer.add_indicator:
                missing_indicator_indices = transformer.indicator_.features_
                missing_indicators = [raw_col_name[idx] + '_missing_flag' for idx in missing_indicator_indices]
                names = raw_col_name + missing_indicators

            else:
                names = list(transformer.get_feature_names())

        except AttributeError:
            names = raw_col_name

        if logger:
            logger.info('{}'.format(names))

        col_name.extend(names)

    return col_name


def preprocess(train_df, test_df=None, features={}, processing='standard'):
    """
    The bulk of the preprocessing for a dataset goes here.
    """

    # create output directory
    if processing == 'standard':
        out_dir = 'standard'
    elif processing == 'categorical':
        out_dir = 'categorical'
    else:
        raise ValueError('args.processing: {} unknown!'.format(processing))
    os.makedirs(os.path.join(out_dir), exist_ok=True)

    # create logger
    logger = get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info('timestamp: {}'.format(datetime.now()))
    logger.info('processing: {}'.format(processing))

    # display datasets
    logger.info('\ntrain_df:\n{}\n{}'.format(train_df.head(5), train_df.shape))
    logger.info('test_df:\n{}\n{}'.format(test_df.head(5), test_df.shape))

    # count number of NAN values per column
    logger.info('')
    for c in train_df.columns:
        logger.info('[TRAIN] {}, no. missing: {:,}'.format(c, train_df[c].isna().sum()))
        logger.info('[TEST] {}, no. missing: {:,}'.format(c, test_df[c].isna().sum()))

    # display column info
    logger.info('\ntrain_df column info:')
    for c in train_df.columns:
        print(c, train_df[c].dtype, len(train_df[c].unique()))

    # categorize attributes
    label = features['label']
    numeric = features['numeric']
    categorical = features['categorical']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))]
    )

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))]
    )

    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]
    )

    # time transforms
    start = time.time()

    # perform one-hot encoding for all cat. attributes
    if processing == 'standard':

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric),
                ('cat', categorical_transformer, categorical),
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
    elif processing == 'categorical':

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric),
                ('ord', ordinal_transformer, categorical),
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
        raise ValueError('args.processing: {} unknown!'.format(processing))

    logger.info('transforming features...{:.3f}s'.format(time.time() - start))

    # get features
    feature_list = get_feature_names(preprocessor)
    assert len(feature_list) == train.shape[1] - 1 == test.shape[1] - 1

    # display statistics
    logger.info('\ntrain:\n{}, dtype: {}'.format(train, train.dtype))
    logger.info('train.shape: {}, label sum: {}'.format(train.shape, train[:, -1].sum()))

    logger.info('\ntest:\n{}, dtype: {}'.format(test, test.dtype))
    logger.info('test.shape: {}, label sum: {}'.format(test.shape, test[:, -1].sum()))

    logger.info('\nfeatures:\n{}'.format(feature_list))

    # save categorical feature names and indices
    if processing == 'categorical':
        categorical_indices = np.arange(len(numeric), len(numeric) + len(categorical))
        logger.info('\ncategorical feature names:\n{}'.format(categorical))
        logger.info('\ncategorical feature indices:\n{}'.format(categorical_indices))
        np.save(os.path.join(out_dir, 'cat_feature.npy'), categorical)
        np.save(os.path.join(out_dir, 'cat_indices.npy'), categorical_indices)

    # save to numpy format
    logger.info('\nsaving to {}/...'.format(out_dir))
    np.save(os.path.join(out_dir, 'train.npy'), train)
    np.save(os.path.join(out_dir, 'test.npy'), test)
    np.save(os.path.join(out_dir, 'feature.npy'), feature_list)
