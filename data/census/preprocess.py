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

    # categorize attributes
    columns = ['age', 'workclass', 'industry_code', 'occupation_code', 'education',
               'wage_per_hour', 'enrolled_in_edu', 'marital_status', 'major_industry_code', 'major_occupation_code',
               'race', 'hispanic_origin', 'sex', 'union_member', 'unemployment_reason',
               'employment', 'capital_gain', 'capital_loss', 'dividends', 'tax_staus',
               'prev_region', 'prev_state', 'household_stat', 'household_summary', 'weight',
               'migration_msa', 'migration_reg', 'migration_reg_move', '1year_house', 'prev_sunbelt',
               'n_persons_employer', 'parents', 'father_birth', 'mother_birth', 'self_birth',
               'citizenship', 'income', 'business', 'taxable_income', 'veterans_admin',
               'veterans_benfits', 'label']

    # retrieve dataset
    start = time.time()
    train_df = pd.read_csv('census-income.data', header=None, names=columns)
    test_df = pd.read_csv('census-income.test', header=None, names=columns)
    print('\ntime to read in data...{:.3f}s'.format(time.time() - start))

    # remove select columns
    remove_cols = []
    if len(remove_cols) > 0:
        train_df = train_df.drop(columns=remove_cols)
        test_df = test_df.drop(columns=remove_cols)
        columns = [x for x in columns if x not in remove_cols]

    # categorize attributes
    features = {}
    features['label'] = ['label']
    features['numeric'] = ['age', 'wage_per_hour', 'capital_gain', 'capital_loss',
                           'dividends', 'weight', 'n_persons_employer']
    features['categorical'] = list(set(columns) - set(features['numeric']) - set(features['label']))

    util.preprocess(train_df, test_df, features, processing=args.processing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processing', type=str, default='categorical', help='regular or categorical.')
    args = parser.parse_args()
    main(args)
