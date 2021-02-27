"""
Organize the performance results into a single csv.
"""
import os
import sys
import argparse
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import sem
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util


def get_result(template, in_dir):
    """
    Obtain the results for this baseline method.
    """
    result = template.copy()

    fp = os.path.join(in_dir, 'results.npy')

    if not os.path.exists(fp):
        result = None

    else:
        d = np.load(fp, allow_pickle=True)[()]
        result.update(d)

    return result


def process_utility(gf):
    """
    Processes utility differences BEFORE addition/deletion,
    and averages the results over different random states.
    """
    result = {}

    acc_list = []
    auc_list = []
    ap_list = []
    ll_list = []

    train_time_list = []
    tune_time_list = []

    # get results from each run
    for row in gf.itertuples(index=False):
        acc_list.append(row.acc)
        auc_list.append(row.auc)
        ap_list.append(row.ap)
        ll_list.append(row.ll)
        train_time_list.append(row.train_time)
        tune_time_list.append(row.tune_time)

    # compute mean and std. error for each metric
    result['acc_mean'] = np.mean(acc_list)
    result['acc_sem'] = sem(acc_list)

    result['auc_mean'] = np.mean(auc_list)
    result['auc_sem'] = sem(auc_list)

    result['ap_mean'] = np.mean(ap_list)
    result['ap_sem'] = sem(ap_list)

    result['ll_mean'] = np.mean(ap_list)
    result['ll_sem'] = sem(ap_list)

    result['train_time_mean'] = np.mean(train_time_list)
    result['train_time_std'] = np.std(train_time_list)

    result['tune_time_mean'] = np.mean(tune_time_list)
    result['tune_time_std'] = np.std(tune_time_list)

    return result


def process_results(df):
    """
    Averages utility results over different random states.
    """

    groups = ['dataset', 'model', 'processing']

    main_result_list = []

    for tup, gf in tqdm(df.groupby(groups)):
        main_result = {k: v for k, v in zip(groups, tup)}
        main_result.update(process_utility(gf))
        main_result['num_runs'] = len(gf)
        main_result['max_rss'] = gf['max_rss'].mean()
        if 'cb' == gf.iloc[0]['model']:
            main_result['n_estimators'] = gf['n_estimators'].mode()[0]
            main_result['max_depth'] = gf['max_depth'].mode()[0]
        main_result_list.append(main_result)

    main_df = pd.DataFrame(main_result_list)

    return main_df


def create_csv(args, out_dir, logger):

    logger.info('\nGathering results...')

    experiment_settings = list(product(*[args.dataset,
                                         args.model,
                                         args.processing,
                                         args.rs]))

    results = []
    for dataset, model, processing, rs in tqdm(experiment_settings):
        template = {'dataset': dataset, 'model': model, 'processing': processing, 'rs': rs}
        experiment_dir = os.path.join(args.in_dir,
                                      dataset,
                                      model,
                                      processing,
                                      'rs_{}'.format(rs))

        # skip empty experiments
        if not os.path.exists(experiment_dir):
            continue

        # add results to result dict
        result = get_result(template, experiment_dir)
        if result is not None:
            results.append(result)

    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 180)

    df = pd.DataFrame(results)
    logger.info('\nRaw results:\n{}'.format(df))

    logger.info('\nProcessing results...')
    main_df = process_results(df)
    logger.info('\nProcessed results:\n{}'.format(main_df))

    main_df.to_csv(os.path.join(out_dir, 'results.csv'), index=None)


def main(args):

    out_dir = os.path.join(args.out_dir)

    # create logger
    os.makedirs(out_dir, exist_ok=True)
    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    create_csv(args, out_dir, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--in_dir', type=str, default='output/performance/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/performance/csv/', help='output directory.')

    # experiment settings
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['churn', 'surgical', 'vaccine', 'amazon', 'bank_marketing', 'adult', 'census'],
                        help='dataset.')
    parser.add_argument('--model', type=int, nargs='+', help='model to extract the results for.',
                        default=['cb', 'dt', 'lr', 'svm_linear', 'svm_rbf', 'knn'])
    parser.add_argument('--processing', type=int, nargs='+', default=['standard', 'categorical'], help='processing.')
    parser.add_argument('--rs', type=int, nargs='+', default=list(range(1, 21)), help='random state.')

    args = parser.parse_args()
    main(args)
