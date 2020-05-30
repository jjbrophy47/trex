"""
Plots the ROAR results.
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

from plot_cleaning import set_size


def get_results(args, dataset, method, score_ndx=0):

    # get results from each run
    res_list = []
    for i in args.rs:
        rs_dir = os.path.join(args.in_dir, dataset, 'rs{}'.format(i))
        pct_path = os.path.join(rs_dir, 'percentages.npy')
        method_path = os.path.join(rs_dir, '{}.npy'.format(method))

        if not os.path.exists(method_path):
            continue

        pcts = np.load(pct_path)
        res = np.load(method_path)[score_ndx]
        res_list.append(res)

    # error checking
    if len(res_list) == 0:
        return None

    # process results
    res = np.vstack(res_list)
    res_mean = np.mean(res, axis=0)
    res_std = sem(res, axis=0)
    return res_mean, res_std, pcts


def main(args):

    our_methods = ['ours_lr_{}'.format(args.encoding), 'ours_svm_{}'.format(args.encoding)]
    method_list = our_methods + ['random', 'maple', 'leafinfluence', 'teknn_{}'.format(args.encoding)]
    titles = ['Churn', 'Amazon', 'Adult', 'Census']
    colors = ['blue', 'blue', 'red', 'orange', 'black', 'purple']
    labels = ['TREX', 'TREX-SVM', 'Random', 'MAPLE', 'LeafInfluence', 'TE-KNN']
    markers = ['1', '2', 'd', 'h', 'o', '*']
    metrics = ['AUC', 'accuracy']

    # matplotlib settings
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=17)
    plt.rc('ytick', labelsize=17)
    plt.rc('axes', labelsize=22)
    plt.rc('axes', titlesize=22)
    plt.rc('legend', fontsize=20)
    plt.rc('legend', title_fontsize=11)
    plt.rc('lines', linewidth=2)
    plt.rc('lines', markersize=11)

    # inches
    width = 5.5  # Neurips 2020
    width, height = set_size(width=width * 3, fraction=1, subplots=(2, 3))
    fig, axs = plt.subplots(2, len(args.dataset), figsize=(width, height / 1.25), sharex=True)

    for h, metric in enumerate(metrics):

        lines = []
        lines_ndx = []
        for i, dataset in enumerate(args.dataset):
            ax = axs[h][i]

            for j, method in enumerate(method_list):
                res = get_results(args, dataset, method, score_ndx=h)

                if res is not None:
                    res_mean, res_std, pcts = res
                    line = ax.errorbar(pcts[:len(res_mean)], res_mean,
                                       marker=markers[j], color=colors[j], label=labels[j])

                    if i == 0:
                        lines.append(line[0])
                        lines_ndx.append(j)

            if i == 0:
                ax.set_ylabel('Test {}'.format(metric))
                if h == 0:
                    ax.legend()
            if h == 0:
                ax.set_title(titles[i])
            if h == 1:
                ax.set_xlabel('% train data removed')
            ax.tick_params(axis='both', which='major')

    os.makedirs(args.out_dir, exist_ok=True)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.25, hspace=0.05)
    plt.savefig(os.path.join(args.out_dir, 'plot.{}'.format(args.format)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, nargs='+', default=['churn', 'amazon', 'adult', 'census'],
                        help='dataset to explain.')
    parser.add_argument('--in_dir', type=str, default='output/roar/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/roar/', help='output directory.')

    parser.add_argument('--encoding', type=str, default='leaf_output', help='tree ensemble.')

    parser.add_argument('--rs', type=int, nargs='+', default=[1], help='dataset to explain.')
    parser.add_argument('--format', type=str, default='png', help='output image format.')
    args = parser.parse_args()
    main(args)
