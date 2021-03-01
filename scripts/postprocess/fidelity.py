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


def process_results(df):
    """
    Averages utility results over different random states.
    """

    groups = ['dataset', 'model', 'preprocessing', 'surrogate', 'tree_kernel', 'metric']

    main_result_list = []

    for tup, gf in tqdm(df.groupby(groups)):
        main_result = {k: v for k, v in zip(groups, tup)}
        main_result['num_runs'] = len(gf)
        main_result['max_rss'] = gf['max_rss'].mean()
        main_result['n_features_alt'] = gf['n_features_alt'].mean()
        # main_result['pearson_mean'] = gf['pearson'].mean()
        # main_result['pearson_sem'] = sem(gf['pearson'])
        main_result['spearman_mean'] = gf['spearman'].mean()
        main_result['spearman_sem'] = sem(gf['spearman'])
        main_result['mse_mean'] = gf['mse'].mean()
        main_result['mse_sem'] = sem(gf['mse'])
        main_result['train_time_mean'] = gf['train_time'].mean()
        main_result_list.append(main_result)

    main_df = pd.DataFrame(main_result_list)

    return main_df


def create_csv(args, out_dir, logger):

    logger.info('\nGathering results...')

    # experiment variabless
    experiment_settings = list(product(*[args.dataset,
                                         args.model,
                                         args.preprocessing,
                                         args.surrogate,
                                         args.tree_kernel,
                                         args.metric,
                                         args.rs]))

    # organize results
    results = []
    for dataset, model, preprocessing, surrogate, tree_kernel, metric, rs in tqdm(experiment_settings):

        # create result
        template = {'dataset': dataset,
                    'model': model,
                    'preprocessing': preprocessing,
                    'surrogate': surrogate,
                    'tree_kernel': tree_kernel,
                    'metric': metric,
                    'rs': rs}

        # get results directory
        experiment_dir = os.path.join(args.in_dir,
                                      dataset,
                                      model,
                                      preprocessing,
                                      surrogate,
                                      tree_kernel,
                                      metric,
                                      'rs_{}'.format(rs))

        # skip empty experiments
        if not os.path.exists(experiment_dir):
            continue

        # add results to result dict
        result = get_result(template, experiment_dir)
        if result is not None:
            results.append(result)

    # set display settings
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 180)

    # raw results
    df = pd.DataFrame(results)
    logger.info('\nRaw results:\n{}'.format(df))

    # processed results
    main_df = process_results(df)
    logger.info('\nProcessed results:\n{}'.format(main_df))

    # save processed results
    main_df.to_csv(os.path.join(out_dir, 'results.csv'), index=None)


def main(args):

    # create output directory
    out_dir = os.path.join(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # create logger
    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    # organize results
    create_csv(args, out_dir, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--in_dir', type=str, default='output/fidelity/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/fidelity/csv/', help='output directory.')

    # experiment settings
    parser.add_argument('--dataset', type=str, nargs='+', help='dataset.',
                        default=['churn', 'surgical', 'vaccine', 'amazon', 'bank_marketing', 'adult', 'census'])
    parser.add_argument('--model', type=str, nargs='+', default=['cb', 'rf'], help='model to extract the results for.')
    parser.add_argument('--preprocessing', type=str, nargs='+', default=['categorical', 'standard'],
                        help='preprocessing directory.')
    parser.add_argument('--surrogate', type=int, nargs='+', default=['klr', 'svm', 'knn'], help='surrogate model.')
    parser.add_argument('--tree_kernel', type=int, nargs='+', help='tree kernel.',
                        default=['feature_path', 'feature_output', 'leaf_path', 'leaf_output', 'tree_output'])
    parser.add_argument('--metric', type=str, nargs='+', default=['mse', 'spearman'], help='tuning metric.')
    parser.add_argument('--rs', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='random state.')

    args = parser.parse_args()
    main(args)
