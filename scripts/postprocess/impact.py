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
        try:
            d = np.load(fp, allow_pickle=True)[()]
            result.update(d)

        except (OSError, EOFError):
            result = None

    return result


def mean_and_sem(arr_list, drop_arrs=True):
    """
    Computes mean and std. error across multiple arrays on axis 0.

    If the arrays are of differing lengths, a masked array is built
    using the array with the largest length; all NaN values are ignored
    when computing the mean and std. error.
    """
    lens = [len(arr) for arr in arr_list]
    same_len = True if len(np.unique(lens)) == 1 else False

    # all arrays are of the same length
    if same_len:
        mean_vals = np.mean(arr_list, axis=0)
        sem_vals = sem(arr_list, axis=0)

    # drop arrays that do not have the same legnth as the majority of arrays
    elif drop_arrs:
        mode_len = max(set(lens), key=lens.count)
        new_arr_list = [arr_list[i] for i in range(len(arr_list)) if lens[i] == mode_len]
        mean_vals = np.mean(new_arr_list, axis=0)
        sem_vals = sem(new_arr_list, axis=0)

    # arrays have differing lengths and need a masked array
    else:
        arr_ma = np.ma.empty((len(arr_list), max(lens)))
        arr_ma.mask = True

        # fill masked array
        for i, arr in enumerate(arr_list):
            arr_ma[i, :lens[i]] = arr

        mean_vals = np.mean(arr_ma, axis=0).data
        sem_vals = sem(arr_ma, axis=0)

    return mean_vals, sem_vals


def process_results(df):
    """
    Averages utility results over different random states.
    """

    groups = ['dataset', 'model', 'preprocessing', 'method']

    main_result_list = []

    for tup, gf in tqdm(df.groupby(groups)):
        main_result = {k: v for k, v in zip(groups, tup)}
        main_result['num_runs'] = len(gf)
        main_result['max_rss'] = gf['max_rss'].mean()
        main_result['total_time'] = gf['total_time'].mean()

        # compute average accuracy
        proba_diffs = [np.array(x) for x in gf['proba_diff'].values]
        mean_vals, sem_vals = mean_and_sem(proba_diffs)
        main_result['proba_diff_mean'] = mean_vals
        main_result['proba_diff_sem'] = sem_vals
        # main_result['proba_diff_mean'] = np.mean(proba_diffs, axis=0)
        # main_result['proba_diff_sem'] = sem(proba_diffs, axis=0)

        # get removed percentages
        remove_pcts = [np.array(x) for x in gf['remove_pct'].values]
        mean_vals, _ = mean_and_sem(remove_pcts)
        main_result['remove_pct'] = mean_vals
        # main_result['remove_pct'] = np.mean(removed_pcts, axis=0)

        main_result_list.append(main_result)

    main_df = pd.DataFrame(main_result_list)

    return main_df


def create_csv(args, out_dir, logger):

    logger.info('\nGathering results...')

    # experiment variabless
    experiment_settings = list(product(*[args.dataset,
                                         args.model,
                                         args.preprocessing,
                                         args.method,
                                         args.rs]))

    # organize results
    results = []
    for dataset, model, preprocessing, method, rs in tqdm(experiment_settings):

        # create result
        template = {'dataset': dataset,
                    'model': model,
                    'preprocessing': preprocessing,
                    'method': method,
                    'rs': rs}

        # get results directory
        experiment_dir = os.path.join(args.in_dir,
                                      dataset,
                                      model,
                                      preprocessing,
                                      method,
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
    parser.add_argument('--in_dir', type=str, default='output/impact/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/impact/csv/', help='output directory.')

    # experiment settings
    parser.add_argument('--dataset', type=str, nargs='+', help='dataset.',
                        default=['churn', 'surgical', 'vaccine', 'amazon', 'bank_marketing', 'adult', 'census'])
    parser.add_argument('--model', type=int, nargs='+', default=['cb', 'rf'], help='model to extract the results for.')
    parser.add_argument('--preprocessing', type=int, nargs='+', default=['standard'], help='preprocessing directory.')
    parser.add_argument('--method', type=int, nargs='+',
                        default=['random', 'klr', 'svm', 'maple', 'knn', 'leaf_influence'],
                        help='method for sorting train data.')
    parser.add_argument('--rs', type=int, nargs='+', default=list(range(1, 21)), help='random state.')

    args = parser.parse_args()
    main(args)
