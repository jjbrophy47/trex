"""
This script plots the cleaning experiment results.
"""
import os
import argparse
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)  # true divide

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


def get_results(dataset, method, args):

    # get results from each run
    res_list = []
    for i in args.rs:
        res_path = os.path.join(args.in_dir, dataset, args.tree_type,
                                args.tree_kernel, 'rs{}'.format(i),
                                '{}.npy'.format(method))

        if not os.path.exists(res_path):
            continue

        res = np.load(res_path)
        res_list.append(np.load(res_path))

    # error checking
    if len(res_list) == 0:
        return None

    # process results
    res = np.vstack(res_list)
    res_mean = np.mean(res, axis=0)
    res_std = sem(res, axis=0)
    return res_mean, res_std


def main(args):

    # settings
    method_list = ['trex_lr', 'trex_svm', 'random',
                   'tree_loss', 'trex_lr_loss', 'trex_svm_loss',
                   'maple', 'leafinfluence', 'teknn',
                   'teknn_loss']
    labels = ['TREX-KLR', 'TREX-SVM', 'Random',
              'Tree Loss', 'KLR Loss', 'SVM Loss',
              'MAPLE', 'LeafInfluence', 'TE-KNN',
              'TE-KNN Loss']
    colors = ['cyan', 'blue', 'red', 'green', 'purple', 'magenta', 'orange',
              'black', 'yellow', '#EEC64F', 'g', 'r']
    markers = ['1', '2', 'o', 'v', '^', '<', '>', '.', '*', 'h', '3', '4']

    # matplotlib settings
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.rc('axes', labelsize=18)
    plt.rc('axes', titlesize=18)
    plt.rc('legend', fontsize=16)
    plt.rc('legend', title_fontsize=11)
    plt.rc('lines', linewidth=1)
    plt.rc('lines', markersize=6)

    # inches
    width = 5.5  # Neurips 2020
    width, height = set_size(width=width * 3, fraction=1, subplots=(1, 3))

    fig, axs = plt.subplots(1, max(2, len(args.dataset)), figsize=(width, height))
    axs = axs.flatten()

    lines = []
    new_labels = []

    for i, dataset in enumerate(args.dataset):
        ax = axs[i]

        rs_dir = os.path.join(args.in_dir, dataset, args.tree_type,
                              args.tree_kernel, 'rs1')
        assert os.path.exists(rs_dir)
        check_pct = np.load(os.path.join(rs_dir, 'check_pct.npy'))
        test_clean = np.load(os.path.join(rs_dir, 'test_clean.npy'))

        check_pct = [x * 100 for x in check_pct]

        for j, method in enumerate(method_list):
            res = get_results(dataset, method, args)

            if res is not None:
                res_mean, res_std = res
                line = ax.errorbar(check_pct, res_mean, yerr=res_std,
                                   marker=markers[j], color=colors[j])

                if i == 0:
                    lines.append(line[0])
                    new_labels.append(labels[j])

        if i == 0:
            ax.set_ylabel('Test accuracy')

        title = 'Census (10%)' if dataset == 'census_0p1' else dataset.capitalize()

        ax.set_xlabel('% train data checked')
        ax.set_title(title)
        ax.tick_params(axis='both', which='major')
        ax.axhline(test_clean, color='k', linestyle='--')

    os.makedirs(args.out_dir, exist_ok=True)
    n_legend_cols = int(len(lines) / 2) if len(lines) > 3 else len(lines)
    fig.legend(tuple(lines), tuple(new_labels), loc='center', ncol=n_legend_cols,
               bbox_to_anchor=(0.525, 0.115))
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.445, wspace=0.275)
    plt.savefig(os.path.join(args.out_dir, 'plot.{}'.format(args.ext)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, nargs='+', default=['churn', 'amazon', 'adult',
                        'census_0p1', 'census'], help='dataset to explain.')
    parser.add_argument('--in_dir', type=str, default='output/cleaning/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/cleaning/', help='output directory.')

    parser.add_argument('--tree_type', type=str, default='cb', help='tree type.')
    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='tree kernel.')
    parser.add_argument('--kernel_model', type=str, default='lr', help='kernel model.')

    parser.add_argument('--rs', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='random state.')
    parser.add_argument('--ext', type=str, default='png', help='output image format.')
    args = parser.parse_args()
    main(args)


class Args:

    dataset = ['churn', 'amazon', 'adult', 'census']
    in_dir = 'output/cleaning/'
    out_dir = 'output/plots/cleaning/'

    tree_type = 'cb'
    tree_kernel = 'leaf_output'
    kernel_model = 'lr'

    rs = [1]
    ext = 'png'
