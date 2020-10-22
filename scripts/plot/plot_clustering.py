"""
This script plots the data before and after feature transformations.
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt


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
    if not os.path.exists(res_dir):
        return None

    res = {}
    res['train_neg'] = np.load(os.path.join(res_dir, 'train_negative.npy'))
    res['train_pos'] = np.load(os.path.join(res_dir, 'train_positive.npy'))
    res['train_tree_neg'] = np.load(os.path.join(res_dir, 'train_tree_negative.npy'))
    res['train_tree_pos'] = np.load(os.path.join(res_dir, 'train_tree_positive.npy'))
    # res['test_pos'] = np.load(os.path.join(res_dir, 'test_positive.npy'))
    # res['test_neg'] = np.load(os.path.join(res_dir, 'test_negative.npy'))
    return res


def main(args):

    # matplotlib settings
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=17)
    plt.rc('ytick', labelsize=17)
    plt.rc('axes', labelsize=22)
    plt.rc('axes', titlesize=22)
    plt.rc('legend', fontsize=18)
    plt.rc('legend', title_fontsize=11)
    plt.rc('lines', linewidth=1)
    plt.rc('lines', markersize=6)

    # inches
    width = 5.5  # Neurips 2020
    width, height = set_size(width=width * 3, fraction=1, subplots=(1, 2))
    fig, axs = plt.subplots(1, 2, figsize=(width, height))
    axs = axs.flatten()

    # plot settings
    markers = ['+', '1']
    alpha = 0.1
    s = 100

    result = get_results(args)

    # plot original features
    ax = axs[0]

    ax.scatter(result['train_neg'][:, 0], result['train_neg'][:, 1],
               color='blue', alpha=alpha, label='train (y=0)',
               s=s, rasterized=args.rasterize)

    ax.scatter(result['train_pos'][:, 0], result['train_pos'][:, 1],
               color='red', alpha=alpha, label='train (y=1)',
               s=s, rasterized=args.rasterize)

    ax.set_title('Original')
    ax.set_xlabel('TSNE 1')
    ax.set_ylabel('TSNE 2')

    # plot tree kernel features
    ax = axs[1]

    ax.scatter(result['train_tree_neg'][:, 0], result['train_tree_neg'][:, 1],
               color='blue', alpha=alpha, label='train (y=0)',
               s=s, rasterized=args.rasterize)

    ax.scatter(result['train_tree_pos'][:, 0], result['train_tree_pos'][:, 1],
               color='red', alpha=alpha, label='train (y=1)',
               s=s, rasterized=args.rasterize)

    ax.set_title('Tree Kernel')
    ax.set_xlabel('TSNE 1')

    ax.tick_params(axis='both', which='major')

    # create output directory
    out_dir = os.path.join(args.out_dir, args.dataset, args.tree_type, args.tree_kernel)
    os.makedirs(out_dir, exist_ok=True)

    # save plot
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plot.{}'.format(args.ext)), rasterized=args.rasterize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # I/O settings
    parser.add_argument('--dataset', type=str, default='churn', help='dataset to explain.')
    parser.add_argument('--in_dir', type=str, default='output/clustering/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/clustering/', help='output directory.')

    # Tree settings
    parser.add_argument('--tree_type', type=str, default='cb', help='tree ensemble.')
    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='tree kernel.')

    # Plot settings
    parser.add_argument('--ext', type=str, default='png', help='output image format.')
    parser.add_argument('--rasterize', action='store_true', default=False, help='rasterize image.')

    args = parser.parse_args()
    main(args)
