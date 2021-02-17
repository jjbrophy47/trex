"""
Utility methods for displaying data.
"""
import sys
import logging

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


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
