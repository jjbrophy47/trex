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


def process_results(df):
    """
    Averages utility results over different random states.
    """

    groups = ['dataset', 'model', 'flip_frac', 'method']

    main_result_list = []

    for tup, gf in tqdm(df.groupby(groups)):
        main_result = {k: v for k, v in zip(groups, tup)}
        main_result['num_runs'] = len(gf)
        main_result['max_rss'] = gf['max_rss'].mean()
        main_result['total_time'] = gf['total_time'].mean()
        main_result['acc_clean'] = gf['acc_clean'].mean()
        main_result['auc_clean'] = gf['auc_clean'].mean()

        # compute average accuracy
        accs = [np.array(x) for x in gf['accs'].values]
        main_result['accs_mean'] = np.mean(accs, axis=0)
        main_result['accs_sem'] = sem(accs, axis=0)

        # compute average AUC
        aucs = [np.array(x) for x in gf['aucs'].values]
        main_result['aucs_mean'] = np.mean(aucs, axis=0)
        main_result['aucs_sem'] = sem(aucs, axis=0)

        # compute average fixed percentages
        fixed_pcts = [np.array(x) for x in gf['fixed_pcts'].values]
        main_result['fixed_pcts_mean'] = np.mean(fixed_pcts, axis=0)
        main_result['fixed_pcts_sem'] = sem(fixed_pcts, axis=0)

        # get checked percentages
        checked_pcts = [np.array(x) for x in gf['checked_pcts'].values]
        main_result['checked_pcts'] = np.mean(checked_pcts, axis=0)

        main_result_list.append(main_result)

    main_df = pd.DataFrame(main_result_list)

    return main_df


def create_csv(args, out_dir, logger):

    logger.info('\nGathering results...')

    # experiment variabless
    experiment_settings = list(product(*[args.dataset,
                                         args.model,
                                         args.flip_frac,
                                         args.method,
                                         args.rs]))

    # organize results
    results = []
    for dataset, model, flip_frac, method, rs in tqdm(experiment_settings):

        # create result
        template = {'dataset': dataset,
                    'model': model,
                    'flip_frac': flip_frac,
                    'method': method,
                    'rs': rs}

        # get results directory
        experiment_dir = os.path.join(args.in_dir,
                                      dataset,
                                      model,
                                      'flip_{}'.format(flip_frac),
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
    parser.add_argument('--in_dir', type=str, default='output/cleaning_new/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/cleaning_new/csv/', help='output directory.')

    # experiment settings
    parser.add_argument('--dataset', type=str, nargs='+', help='dataset.',
                        default=['churn', 'surgical', 'vaccine', 'bank_marketing', 'adult'])
    parser.add_argument('--model', type=int, nargs='+', default=['cb', 'rf'], help='model.')
    parser.add_argument('--flip_frac', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4], help='flip fraction.')
    parser.add_argument('--method', type=int, nargs='+',
                        default=['random',
                                 'tree_loss',
                                 'fast_leaf_influence',
                                 'klr_tree_output_alpha',
                                 'klr_og_tree_output_alpha',
                                 'klr_og_tree_output_sim',
                                 'klr_leaf_path_alpha',
                                 'klr_og_leaf_path_alpha',
                                 'klr_og_leaf_path_sim',
                                 'klr_weighted_leaf_path_alpha',
                                 'klr_og_weighted_leaf_path_alpha',
                                 'klr_og_weighted_leaf_path_sim',
                                 'klr_weighted_feature_path_alpha',
                                 'klr_og_weighted_feature_path_alpha',
                                 'klr_og_weighted_feature_path_sim',
                                 ],
                        help='method for checking train data.')
    parser.add_argument('--rs', type=int, nargs='+', default=list(range(1, 11)), help='random state.')

    args = parser.parse_args()
    main(args)
