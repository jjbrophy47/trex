"""
Preprocess dataset to make it easier to load and work with.
"""
import os
import sys
import time
import argparse

import pandas as pd

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
from utility import util


def main(args):

    # retrieve dataset
    start = time.time()
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    print('\ntime to read in data...{:.3f}s'.format(time.time() - start))

    # get features
    columns = list(train_df.columns)

    # remove select columns
    remove_cols = []
    if len(remove_cols) > 0:
        train_df = train_df.drop(columns=remove_cols)
        test_df = test_df.drop(columns=remove_cols)
        columns = [x for x in columns if x not in remove_cols]

    # categorize attributes
    features = {}
    features['label'] = ['ACTION']
    features['numeric'] = []
    features['categorical'] = list(set(columns) - set(features['numeric']) - set(features['label']))

    util.preprocess(train_df, test_df, features, processing=args.processing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processing', type=str, default='categorical', help='regular or categorical.')
    args = parser.parse_args()
    main(args)
