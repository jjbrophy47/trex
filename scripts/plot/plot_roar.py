"""
Plots the ROAR results.
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


def get_results(args, dataset, method, score_ndx=0):

    # get results from each run
    res_list = []

    # get results
    method_dir = 'klr' if method == 'random' else method
    method_name = method if method == 'random' else 'method'

    # add tree kernel to specific methods
    if method_dir in ['klr', 'svm', 'teknn']:
        method_dir = os.path.join(method_dir, args.tree_kernel)

    for i in args.rs:
        in_dir = os.path.join(args.in_dir, dataset, args.tree_type,
                              'rs{}'.format(i), method_dir)

        pct_path = os.path.join(in_dir, 'percentages.npy')
        method_path = os.path.join(in_dir, '{}.npy'.format(method_name))

        if not os.path.exists(method_path):
            if args.verbose > 0:
                print(method_path)
            continue

        pcts = np.load(pct_path)
        res = np.load(method_path, allow_pickle=True)[()]
        res_list.append(res[args.metric])

    if len(res_list) == 0:
        result = None

    # process results
    else:
        if len(res_list) > 1:
            res = np.vstack(res_list)
            res_mean = np.mean(res, axis=0)
            res_sem = sem(res, axis=0)
            result = res_mean, res_sem, pcts
        else:
            result = res_list[0], [0] * len(res_list[0]), pcts

    return result


def main(args):
    print(args)

    method_list = ['random', 'maple', 'teknn', 'klr']
    colors = ['red', 'orange', 'purple', 'blue', 'green']
    labels = ['Random', 'MAPLE', 'TEKNN', 'TREX', 'TREX-SVM']
    markers = ['o', 'd', '^', 'x', '2']
    metric_mapping = {'auc': 'AUROC', 'acc': 'Accuracy'}

    # matplotlib settings
    plt.rc('font', family='serif')

    if args.two_col:
        plt.rc('xtick', labelsize=11)
        plt.rc('ytick', labelsize=11)
        plt.rc('axes', labelsize=11)
        plt.rc('axes', titlesize=11)
        plt.rc('legend', fontsize=11)
        plt.rc('legend', title_fontsize=11)
        plt.rc('lines', linewidth=1)
        plt.rc('lines', markersize=3)

        width = 3.25  # Two column style
        width, height = set_size(width=width * 2, fraction=1, subplots=(2, 2))
        fig, axs = plt.subplots(2, 2, figsize=(width, height), sharex='col')

    else:
        plt.rc('xtick', labelsize=17)
        plt.rc('ytick', labelsize=17)
        plt.rc('axes', labelsize=22)
        plt.rc('axes', titlesize=22)
        plt.rc('legend', fontsize=20)
        plt.rc('legend', title_fontsize=11)
        plt.rc('lines', linewidth=2)
        plt.rc('lines', markersize=8)

        width = 5.5  # Neurips 2020
        width, height = set_size(width=width * 3, fraction=1, subplots=(1, 3))
        fig, axs = plt.subplots(1, max(2, len(args.dataset)),
                                figsize=(width, height), sharex=True)
    axs = axs.flatten()

    n_pts = 10

    ylabel = 'Test {}'.format(metric_mapping[args.metric])
    xlabel = '% train data removed'

    lines = []
    lines_ndx = []
    for i, dataset in enumerate(args.dataset):
        ax = axs[i]

        for j, method in enumerate(method_list):
            res = get_results(args, dataset, method)

            if res is not None:
                res_mean, res_sem, pcts = res
                line = ax.errorbar(pcts[:n_pts], res_mean[:n_pts], yerr=res_sem[:n_pts],
                                   marker=markers[j], color=colors[j], label=labels[j])

                if i == 0:
                    lines.append(line[0])
                    lines_ndx.append(j)

        if args.two_col:
            if i % 2 == 0:
                axs[i].set_ylabel(ylabel)
            if i > 1:
                axs[i].set_xlabel(xlabel)

        else:

            if i == 0:
                ax.set_ylabel(ylabel)
                ax.set_xlabel(xlabel)

        if i == 1:
            ax.legend(frameon=False)

        ax.set_title(dataset.capitalize())
        ax.tick_params(axis='both', which='major')

    out_dir = os.path.join(args.out_dir, args.tree_kernel)
    os.makedirs(out_dir, exist_ok=True)

    plt.tight_layout()

    if not args.two_col:
        fig.subplots_adjust(wspace=0.25, hspace=0.05)

    plt.savefig(os.path.join(out_dir, 'plot.{}'.format(args.ext)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, nargs='+', default=['churn', 'amazon', 'adult', 'census'],
                        help='dataset to explain.')
    parser.add_argument('--in_dir', type=str, default='output/roar/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/roar/', help='output directory.')

    parser.add_argument('--tree_type', type=str, default='cb', help='tree type.')
    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='tree kernel.')

    parser.add_argument('--metric', type=str, default='acc', help='predictive metric.')
    parser.add_argument('--two_col', action='store_true', default=False, help='format into two columns.')
    parser.add_argument('--rs', type=int, nargs='+', default=list(range(20)), help='random states.')
    parser.add_argument('--ext', type=str, default='png', help='output image format.')
    parser.add_argument('--verbose', type=int, default=1, help='verbosity level.')

    args = parser.parse_args()
    main(args)
