"""
Preprocess dataset to make it easier to load and work with.
"""
import os
import sys
import time
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
from utility import util


def main(args):

    # retrieve dataset
    start = time.time()
    data_df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print('\ntime to read in data...{:.3f}s'.format(time.time() - start))

    # fix numeric column containing empy strings
    data_df['TotalCharges'] = pd.to_numeric(data_df['TotalCharges'], errors='coerce')

    # split data into train and test
    train_df, test_df = train_test_split(data_df,
                                         test_size=args.test_size,
                                         random_state=args.seed,
                                         stratify=data_df['Churn'])

    # get features
    columns = list(train_df.columns)

    # remove select columns
    remove_cols = ['customerID']
    if len(remove_cols) > 0:
        train_df = train_df.drop(columns=remove_cols)
        test_df = test_df.drop(columns=remove_cols)
        columns = [x for x in columns if x not in remove_cols]

    # categorize attributes
    features = {}
    features['label'] = ['Churn']
    features['numeric'] = ['tenure', 'MonthlyCharges', 'TotalCharges']
    features['categorical'] = list(set(columns) - set(features['numeric']) - set(features['label']))

    util.preprocess(train_df, test_df, features, processing=args.processing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processing', type=str, default='categorical', help='regular or categorical.')
    parser.add_argument('--test_size', type=float, default=0.2, help='frac. of samples to use for testing.')
    parser.add_argument('--seed', type=int, default=1, help='random state.')
    args = parser.parse_args()
    main(args)
