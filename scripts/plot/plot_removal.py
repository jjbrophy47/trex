"""
Plots results from the removal experiment.
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem


def set_size(width, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return width, height


def get_results(args):
    """
    Return results.
    """

    res_dir = os.path.join(args.in_dir, args.dataset, args.tree_type, args.tree_kernel)
    assert os.path.exists(res_dir)

    res = {}
    for i in args.rs:
        res[i] = np.load(os.path.join(res_dir, 'rs{}'.format(i), 'results.npy'),
                         allow_pickle=True)[()]
    return res


def get_mean(res, name='aucs'):
    """
    Return mean array from each result dictionary.
    """

    vals = []
    for k in res.keys():
        vals.append(res[k][name])

    vals_arr = np.vstack(vals)
    vals_mean = np.mean(vals_arr, axis=0)
    vals_sem = sem(vals_arr, axis=0)

    return vals_mean, vals_sem


def main(args):

    # matplotlib settings
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('axes', labelsize=16)
    plt.rc('axes', titlesize=16)
    plt.rc('legend', fontsize=18)
    plt.rc('legend', title_fontsize=11)
    plt.rc('lines', linewidth=1)
    plt.rc('lines', markersize=6)

    # inches
    width = 5.5  # Neurips 2020
    width, height = set_size(width=width * 1.25, fraction=1, subplots=(1, 1))
    fig, ax = plt.subplots(1, 1, figsize=(width, height / 1.5))

    res = get_results(args)
    n_removed, _ = get_mean(res, name='n_remove')
    score_mean, score_sem = get_mean(res, name=args.metric)
    original_score, _ = get_mean(res, name='original_{}'.format(args.metric))

    # plot results
    ax.errorbar(n_removed, score_mean, yerr=score_sem, fmt='-o', color='green')
    ax.axhline(original_score, linestyle='--', color='k')
    ax.set_xlabel('# train samples removed')
    ax.set_ylabel('Test {}'.format(args.metric.upper()))
    ax.tick_params(axis='both', which='major')

    os.makedirs(args.out_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'plot.{}'.format(args.ext)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='nc17_mfc18')
    parser.add_argument('--in_dir', type=str, default='output/removal/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/removal/', help='output directory.')

    parser.add_argument('--tree_type', type=str, default='lgb', help='tree ensemble.')
    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='tree kernel.')

    parser.add_argument('--rs', type=int, nargs='+', default=list(range(1, 21)), help='random state.')
    parser.add_argument('--metric', type=str, default='auc', help='predictive metric.')
    parser.add_argument('--ext', type=str, default='pdf', help='output image format.')
    args = parser.parse_args()
    main(args)


class Args:

    dataset = 'nc17_mfc18'
    in_dir = 'output/removal/'
    out_dir = 'output/plots/removal/'

    tree_type = 'lgb'
    tree_kernel = 'leaf_output'

    rs = list(range(1, 21))
    metric = 'auc'
    ext = 'pdf'
