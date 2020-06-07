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

    for i in args.rs:
        in_dir = os.path.join(args.in_dir, dataset, args.tree_type,
                              args.tree_kernel, 'rs{}'.format(i), method_dir)

        pct_path = os.path.join(in_dir, 'percentages.npy')
        method_path = os.path.join(in_dir, '{}.npy'.format(method_name))

        if not os.path.exists(method_path):
            print(method_path)
            continue

        pcts = np.load(pct_path)
        res = np.load(method_path, allow_pickle=True)[()]
        res_list.append(res[args.metric])

    if len(res_list) == 0:
        result = None

    # process results
    else:
        res = np.vstack(res_list)
        res_mean = np.mean(res, axis=0)
        res_sem = sem(res, axis=0)
        result = res_mean, res_sem, pcts

    return result


def main(args):

    method_list = ['random', 'maple', 'leafinfluence', 'teknn', 'klr']
    colors = ['red', 'orange', 'black', 'purple', 'blue', 'green']
    labels = ['Random', 'MAPLE', 'LeafInfluence', 'TEKNN', 'TREX-KLR', 'TREX-SVM']
    markers = ['*', 'd', 'h', 'o', '1', '2']
    metric_mapping = {'auc': 'AUROC', 'acc': 'Accuracy'}

    # matplotlib settings
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=17)
    plt.rc('ytick', labelsize=17)
    plt.rc('axes', labelsize=22)
    plt.rc('axes', titlesize=22)
    plt.rc('legend', fontsize=20)
    plt.rc('legend', title_fontsize=11)
    plt.rc('lines', linewidth=2)
    plt.rc('lines', markersize=8)

    # inches
    width = 5.5  # Neurips 2020
    width, height = set_size(width=width * 3, fraction=1, subplots=(1, 3))
    fig, axs = plt.subplots(1, max(2, len(args.dataset)),
                            figsize=(width, height), sharex=True)
    axs = axs.flatten()

    n_pts = 10

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

        if i == 0:
            ax.set_ylabel('Test {}'.format(metric_mapping[args.metric]))
        if i == 1:
            ax.legend()
        ax.set_title(dataset.capitalize())
        ax.set_xlabel('% train data removed')
        ax.tick_params(axis='both', which='major')

    os.makedirs(args.out_dir, exist_ok=True)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.25, hspace=0.05)
    plt.savefig(os.path.join(args.out_dir, 'plot.{}'.format(args.ext)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, nargs='+', default=['churn', 'amazon', 'adult', 'census'],
                        help='dataset to explain.')
    parser.add_argument('--in_dir', type=str, default='output/roar/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/roar/', help='output directory.')

    parser.add_argument('--tree_type', type=str, default='cb', help='tree type.')
    parser.add_argument('--tree_kernel', type=str, default='tree_output', help='tree kernel.')

    parser.add_argument('--metric', type=str, default='acc', help='predictive metric.')
    parser.add_argument('--rs', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='dataset to explain.')
    parser.add_argument('--ext', type=str, default='png', help='output image format.')

    args = parser.parse_args()
    main(args)


class Args:

    dataset = ['churn', 'amazon', 'adult', 'census']
    in_dir = 'output/roar/'
    out_dir = 'output/plots/roar/'

    tree_type = 'cb'
    tree_kernel = 'tree_output'

    rs = [1]
    ext = 'png'
