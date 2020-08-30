"""
Plots the fidelity experiment results
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


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
            if args.verbose > 0:
                print(method_path)
            continue

        tree_res = np.load(tree_path)
        method_res = np.load(method_path)

        if args.corr == 'pearson':
            corr = pearsonr(tree_res, method_res)[0]
        elif args.corr == 'spearman':
            corr = spearmanr(tree_res, method_res)[0]
        elif args.corr == 'r2':
            corr = r2_score(tree_res, method_res)
        else:
            raise ValueError('Correlation function {} unknown!'.format(args.corr))

        label = '{}={:.3f}'.format(labels[i], corr)
        ax.scatter(method_res, tree_res, color=colors[i],
                   label=label, rasterized=True, marker=markers[i])

    ax.tick_params(axis='both', which='major')

    leg = ax.legend(loc='upper right', ncol=2, handletextpad=0.05, framealpha=1.0)

    # Get the bounding box of the original legend
    bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)

    # Change to location of the legend.
    yOffset = 0.25
    xOffset = 0.01
    bb.y0 += yOffset
    bb.y1 += yOffset
    bb.x0 += xOffset
    bb.x1 += xOffset
    leg.set_bbox_to_anchor(bb, transform=ax.transAxes)


def main(args):

    method_list = ['klr', 'svm', 'teknn']
    labels = ['KLR', 'SVM', 'KNN']
    colors = ['blue', 'cyan', 'purple']
    markers = ['1', '2', '*']

    ylabel = 'GBDT probability'
    xlabel = 'Surrogate probability'

    # matplotlib settings
    plt.rc('font', family='serif')

    if args.two_col:
        plt.rc('xtick', labelsize=11)
        plt.rc('ytick', labelsize=11)
        plt.rc('axes', labelsize=11)
        plt.rc('axes', titlesize=11)
        plt.rc('legend', fontsize=8)
        plt.rc('legend', title_fontsize=9)
        plt.rc('lines', linewidth=1)
        plt.rc('lines', markersize=3)

        width = 3.45  # Two column style
        width, height = set_size(width=width * 2, fraction=1, subplots=(2, 2))
        fig, axs = plt.subplots(2, 2, figsize=(width, height * 1.25),
                                sharey='row', sharex='col')

    else:
        plt.rc('xtick', labelsize=17)
        plt.rc('ytick', labelsize=17)
        plt.rc('axes', labelsize=22)
        plt.rc('axes', titlesize=22)
        plt.rc('legend', fontsize=20)
        plt.rc('legend', title_fontsize=11)
        plt.rc('lines', linewidth=1)
        plt.rc('lines', markersize=6)

        width = 5.5  # One column style
        width, height = set_size(width=width * 3, fraction=1, subplots=(1, 3))
        fig, axs = plt.subplots(1, max(2, len(args.dataset)), figsize=(width, height),
                                sharey='row')
    axs = axs.flatten()

    # plot top row
    for i, dataset in enumerate(args.dataset):
        _plot_graph(args, axs[i], dataset, method_list, args.tree_kernel,
                    labels, colors, markers, corr=args.corr)
        axs[i].set_title(dataset.capitalize(), loc='left')

        if args.two_col:
            if i % 2 == 0:
                axs[i].set_ylabel(ylabel)
            if i > 1:
                axs[i].set_xlabel(xlabel)
        else:
            axs[i].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)

    plt.tight_layout()

    out_dir = os.path.join(args.out_dir, args.tree_kernel)
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(os.path.join(out_dir, 'fidelity_{}.{}'.format(args.tree_kernel, args.ext)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, nargs='+', default=['churn', 'amazon', 'adult', 'census'],
                        help='dataset to explain.')
    parser.add_argument('--in_dir', type=str, default='output/fidelity/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/fidelity/', help='output directory.')

    parser.add_argument('--tree_type', type=str, default='cb', help='tree ensemble.')
    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='tree kernel.')

    parser.add_argument('--two_col', action='store_true', default=False, help='format into two columns.')
    parser.add_argument('--corr', type=str, default='pearson', help='statistical correlation.')

    parser.add_argument('--ext', type=str, default='png', help='output image format.')
    parser.add_argument('--verbose', type=int, default=1, help='verbosity level.')

    args = parser.parse_args()
    main(args)
