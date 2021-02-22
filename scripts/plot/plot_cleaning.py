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


def main(args):
    print(args)

    # settings
    method_list = ['klr', 'svm', 'random',
                   'tree', 'klr_loss', 'svm_loss',
                   'maple', 'leaf_influence', 'teknn',
                   'teknn_loss', 'proto']

    label_list = ['TREX-KLR', 'TREX-SVM', 'Random',
                  'GBDT Loss', 'TREX-KLR Loss', 'TREX-SVM Loss',
                  'MAPLE', 'LeafInfluence', 'TEKNN',
                  'TEKNN Loss', 'TreeProto']

    color_list = ['blue', 'cyan', 'red', 'green', 'purple', 'magenta', 'orange',
                  'black', '#EEC64F', 'yellow', 'brown']

    marker_list = ['1', '2', 'o', 'v', '^', '<', '>', '.', '*', 'h', 's']

    zorder_list = [11, 10, 9, 3, 2, 1, 7, 1, 6, 5, 8]

    fig, ax = plt.subplots()

    # plot each method
    methods = list(zip(method_list, label_list, color_list, marker_list, zorder_list))
    for method, label, color, marker, zorder in list(zip(method_list, label)):

        # extract results
        temp_df = df[df['method'] == method]
        metric_mean = temp_df['{}_mean'.format(args.metric)]
        metric_sem = temp_df['{}_sem'.format(args.metric)]
        checked_pcts = temp_df['checked_pcts']

        # plot
        line = ax.errorbar(checked_pcts, metric_mean, yerr=metric_sem,
                           marker=marker, color=color, zorder=zorder)

        ax.set_xlabel('% train data checked')
        ax.set_title('Census (10%)' if args.dataset == 'census_0p1' else args.dataset.capitalize())
        ax.tick_params(axis='both', which='major')
        ax.axhline(test_clean, color='k', linestyle='--')

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

    fig, axs = plt.subplots(1, max(2, len(args.dataset)), figsize=(width, height * 1.25))
    axs = axs.flatten()

    lines = []
    new_labels = []

    for i, dataset in enumerate(args.dataset):
        ax = axs[i]

        rs_dir = os.path.join(args.in_dir, dataset, args.tree_type,
                              'rs1', 'klr', args.tree_kernel)
        if not os.path.exists(rs_dir):
            print(rs_dir)
            continue

        check_pct = np.load(os.path.join(rs_dir, 'check_pct.npy'))
        test_clean = np.load(os.path.join(rs_dir, 'test_clean.npy'))

        check_pct = [x * 100 for x in check_pct]

        for j, method in enumerate(method_list):
            res = get_results(dataset, method, args)

            if res is not None:
                res_mean, res_std = res
                line = ax.errorbar(check_pct, res_mean, yerr=res_std,
                                   marker=markers[j], color=colors[j],
                                   zorder=zorders[j], markersize=3.5)

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

    out_dir = os.path.join(args.out_dir, args.tree_kernel)
    os.makedirs(out_dir, exist_ok=True)

    fig.legend(tuple(lines), tuple(new_labels), loc='center', ncol=6,
               bbox_to_anchor=(0.5, 0.125))

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.425, wspace=0.275)
    plt.savefig(os.path.join(out_dir, 'plot.{}'.format(args.ext)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, nargs='+', default=['churn', 'amazon', 'adult',
                        'census_0p1', 'census'], help='dataset to explain.')
    parser.add_argument('--in_dir', type=str, default='output/cleaning/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/cleaning/', help='output directory.')

    parser.add_argument('--tree_type', type=str, default='cb', help='tree type.')
    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='tree kernel.')

    parser.add_argument('--rs', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='random state.')
    parser.add_argument('--ext', type=str, default='png', help='output image format.')
    parser.add_argument('--verbose', type=int, default=1, help='verbosity level.')

    args = parser.parse_args()
    main(args)
