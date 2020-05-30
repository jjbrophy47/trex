"""
Plots the fidelity experiment results
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from plot_cleaning import set_size


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _plot_graph(args, ax, dataset, method_list, labels, colors, markers, corr='pearson'):

    for i, method in enumerate(method_list):
        method_path = os.path.join(args.in_dir, dataset, method, 'ours_test.npy')
        tree_path = os.path.join(args.in_dir, dataset, method, 'tree_test.npy')

        if not os.path.exists(method_path):
            continue

        method_res = np.load(method_path)
        tree_res = np.load(tree_path)

        if 'svm' in method:
            method_res = _sigmoid(method_res)

        if args.corr == 'pearson':
            corr = pearsonr(tree_res, method_res)[0]
        else:
            corr = spearmanr(tree_res, method_res)[0]

        label = '{}={:.3f}'.format(labels[i], corr)
        ax.scatter(method_res, tree_res, color=colors[i],
                   label=label, rasterized=True, marker=markers[i])

    ax.legend(loc='upper left')
    ax.tick_params(axis='both', which='major')


def main(args):

    top_method_list = ['lr_linear_leaf_output', 'svm_linear_leaf_output', 'teknn_leaf_output']
    bot_method_list = ['lr_linear_leaf_path', 'svm_linear_leaf_path', 'teknn_leaf_path']
    labels = ['KLR', 'SVM', 'KNN']
    colors = ['cyan', 'blue', 'purple']
    markers = ['1', '2', '*']

    # select a tree-ensemble model
    top_method_list = ['{}_{}'.format(args.model, method) for method in top_method_list]
    bot_method_list = ['{}_{}'.format(args.model, method) for method in bot_method_list]

    ylabel = 'GBDT probability'
    xlabel = 'Surrogate probability'

    # matplotlib settings
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=17)
    plt.rc('ytick', labelsize=17)
    plt.rc('axes', labelsize=22)
    plt.rc('axes', titlesize=22)
    plt.rc('legend', fontsize=20)
    plt.rc('legend', title_fontsize=11)
    plt.rc('lines', linewidth=1)
    plt.rc('lines', markersize=6)

    # inches
    width = 5.5  # Neurips 2020
    width, height = set_size(width=width * 3, fraction=1, subplots=(2, 3))
    fig, axs = plt.subplots(2, len(args.dataset), figsize=(width, height), sharey='row', sharex='col')

    # plot top row
    row = 0
    for i, dataset in enumerate(args.dataset):
        _plot_graph(args, axs[row][i], dataset, top_method_list,
                    labels, colors, markers, corr=args.corr)
        axs[row][i].set_title(dataset.capitalize())

    # plot bottom row
    row = 1
    for i, dataset in enumerate(args.dataset):
        _plot_graph(args, axs[row][i], dataset, bot_method_list,
                    labels, colors, markers, corr=args.corr)
        axs[row][i].set_xlabel(xlabel)

    axs[0][0].set_ylabel(ylabel)
    axs[1][0].set_ylabel(ylabel)

    os.makedirs(args.out_dir, exist_ok=True)

    fig.subplots_adjust(wspace=0.005, hspace=0.005)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'plot.{}'.format(args.format)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, nargs='+', default=['churn', 'amazon', 'adult', 'census'],
                        help='dataset to explain.')
    parser.add_argument('--in_dir', type=str, default='output/fidelity/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/fidelity/', help='output directory.')

    parser.add_argument('--model', type=str, default='cb', help='tree ensemble.')
    parser.add_argument('--corr', type=str, default='pearson', help='statistical correlation.')

    parser.add_argument('--format', type=str, default='png', help='output image format.')
    args = parser.parse_args()
    main(args)
