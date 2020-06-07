"""
Plots the fidelity experiment results
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr


def set_size(width, fraction=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return width, height


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _plot_graph(args, ax, dataset, method_list, tree_kernel,
                labels, colors, markers, corr='pearson'):

    in_dir = os.path.join(args.in_dir, dataset, args.tree_type, tree_kernel)

    for i, method in enumerate(method_list):
        tree_path = os.path.join(in_dir, method, 'tree.npy')
        method_path = os.path.join(in_dir, method, 'surrogate.npy')

        if not os.path.exists(method_path):
            print(method_path)
            continue

        tree_res = np.load(tree_path)
        method_res = np.load(method_path)
        if 'svm' in method:
            method_res = _sigmoid(method_res)

        corr_func = pearsonr if args.corr == 'pearson' else spearmanr
        corr = corr_func(tree_res, method_res)[0]

        label = '{}={:.3f}'.format(labels[i], corr)
        ax.scatter(method_res, tree_res, color=colors[i],
                   label=label, rasterized=True, marker=markers[i])

    ax.legend(loc='upper left')
    ax.tick_params(axis='both', which='major')


def main(args):

    method_list = ['klr', 'svm', 'teknn']
    labels = ['KLR', 'SVM', 'KNN']
    colors = ['cyan', 'blue', 'purple']
    markers = ['1', '2', '*']

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
    width, height = set_size(width=width * 3, fraction=1, subplots=(1, 3))
    fig, axs = plt.subplots(1, max(2, len(args.dataset)), figsize=(width, height),
                            sharey='row')
    axs = axs.flatten()

    # plot top row
    for i, dataset in enumerate(args.dataset):
        _plot_graph(args, axs[i], dataset, method_list, args.tree_kernel,
                    labels, colors, markers, corr=args.corr)
        axs[i].set_title(dataset.capitalize())
        axs[i].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)

    os.makedirs(args.out_dir, exist_ok=True)

    fig.subplots_adjust(wspace=0.005, hspace=0.005)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'plot.{}'.format(args.ext)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, nargs='+', default=['churn', 'amazon', 'adult', 'census'],
                        help='dataset to explain.')
    parser.add_argument('--in_dir', type=str, default='output/fidelity/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/fidelity/', help='output directory.')

    parser.add_argument('--tree_type', type=str, default='cb', help='tree ensemble.')
    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='tree kernel.')

    parser.add_argument('--corr', type=str, default='pearson', help='statistical correlation.')
    parser.add_argument('--ext', type=str, default='png', help='output image format.')
    args = parser.parse_args()
    main(args)


class Args:

    dataset = ['churn', 'amazon', 'adult', 'census']
    in_dir = 'output/fidelity/'
    out_dir = 'output/plots/fidelity/'

    tree_type = 'cb'
    tree_kernel = 'leaf_output'

    corr = 'pearson'
    ext = 'png'
