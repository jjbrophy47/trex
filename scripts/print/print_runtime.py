"""
This script prints the runtime results.
"""
import os
import sys
import argparse
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)  # true divide
import logging

import numpy as np
from scipy.stats import sem


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


def remove_logger(logger):
    """
    Remove handlers from logger.
    """
    logger.handlers = []


def get_results(dataset, method, args):

    # add tree kernel to specific methods
    method_dir = method
    if method in ['klr', 'svm', 'teknn']:
        method_dir = os.path.join(method_dir, args.tree_kernel)

    # get results from each run
    r = {}
    for i in args.rs:
        res_path = os.path.join(args.in_dir, dataset, args.tree_type,
                                'rs{}'.format(i), method_dir, 'method.npy')

        if not os.path.exists(res_path):
            print(res_path)
            continue

        r[i] = np.load(res_path, allow_pickle=True)[()]

    return r


def get_mean(args, r, name='fine_tune'):
    """
    Return mean value over multiple results.
    """
    result = []
    for i in args.rs:
        if i in r:
            if name in r[i]:
                result.append(r[i][name])

    # error checking
    if len(result) == 0:
        return -1, -1

    # process results
    res_mean = np.mean(result)
    res_std = sem(result)
    return res_mean, res_std


def main(args):
    print(args)

    # make logger
    out_dir = os.path.join(args.out_dir, args.tree_kernel)
    os.makedirs(out_dir, exist_ok=True)

    logger = get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)

    os.makedirs(args.out_dir, exist_ok=True)

    # settings
    method_list = ['klr', 'svm', 'maple', 'leaf_influence', 'teknn']
    s = '[{:15}] fine_tune: {:>11.5f} +/- {:>11.5f}, test_time: {:>11.5f} +/- {:>11.5f}'

    for i, dataset in enumerate(args.dataset):
        logger.info('\n{}'.format(dataset.capitalize()))

        for j, method in enumerate(method_list):
            r = get_results(dataset, method, args)

            fine_tune_mean, fine_tune_std = get_mean(args, r, name='fine_tune')
            test_time_mean, test_time_std = get_mean(args, r, name='test_time')
            logger.info(s.format(method, fine_tune_mean, fine_tune_std,
                        test_time_mean, test_time_std))

    remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, nargs='+', default=['churn', 'amazon', 'adult', 'census'],
                        help='dataset to explain.')
    parser.add_argument('--in_dir', type=str, default='output/runtime/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/prints/runtime/', help='output directory.')

    parser.add_argument('--tree_type', type=str, default='cb', help='tree type.')
    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='tree kernel.')

    parser.add_argument('--rs', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='random state.')
    args = parser.parse_args()
    main(args)
