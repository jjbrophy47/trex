"""
This script plots the cleaning experiment results.
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


def get_results(dataset, method, args):

    # get results from each run
    res_list = []
    for i in args.rs:
        res_path = os.path.join(args.in_dir, dataset, 'rs{}'.format(i),
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
    method_list = ['our_lr_leaf_output', 'our_svm_leaf_output', 'random',
                   'tree_loss', 'lr_leaf_output_loss', 'svm_leaf_output_loss',
                   'maple', 'leafinfluence', 'knn_leaf_output',
                   'knn_leaf_output_loss']
    labels = ['TREX-KLR', 'TREX-SVM', 'Random',
              'Tree Loss', 'KLR Loss', 'SVM Loss',
              'MAPLE', 'LeafInfluence', 'TE-KNN',
              'KNN Loss']
    titles = ['Churn', 'Amazon', 'Adult', 'Census (10%)', 'Census (100%)']
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

    fig, axs = plt.subplots(1, len(args.dataset), figsize=(width, height))
    axs = axs.flatten()

    lines = []
    for i, dataset in enumerate(args.dataset):
        ax = axs[i]

        rs_dir = os.path.join(args.in_dir, dataset, 'rs1')
        print(rs_dir)
        assert os.path.exists(rs_dir)
        check_pct = np.load(os.path.join(rs_dir, 'check_pct.npy'))
        test_clean = np.load(os.path.join(rs_dir, 'test_clean.npy'))

        check_pct = [x * 100 for x in check_pct]

        for j, method in enumerate(method_list):
            res = get_results(dataset, method, args)

            if res is not None:
                res_mean, res_std = res
                line = ax.errorbar(check_pct, res_mean, yerr=res_std,
                                   marker=markers[j], color=colors[j],
                                   label=labels[j])

                if i == 0:
                    lines.append(line[0])

        if i == 0:
            ax.set_ylabel('Test accuracy')
        ax.set_xlabel('% train data checked')
        ax.set_title(titles[i])
        ax.tick_params(axis='both', which='major')
        ax.axhline(test_clean, color='k', linestyle='--')

    os.makedirs(args.out_dir, exist_ok=True)

    fig.legend(tuple(lines), tuple(labels), loc='center', ncol=int(len(lines) / 2),
               bbox_to_anchor=(0.525, 0.115))
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.445, wspace=0.275)
    plt.savefig(os.path.join(args.out_dir, 'plot.{}'.format(args.format)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, nargs='+', default=['churn', 'amazon', 'adult',
                        'census_0p1', 'census'], help='dataset to explain.')
    parser.add_argument('--rs', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='random state.')
    parser.add_argument('--in_dir', type=str, default='output/cleaning/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/cleaning/', help='output directory.')
    parser.add_argument('--format', type=str, default='png', help='output image format.')
    args = parser.parse_args()
    main(args)
