"""
This script plots the cleaning experiment results.
"""
import os
import argparse
import warnings
import pandas as pd
warnings.simplefilter(action='ignore', category=RuntimeWarning)  # true divide

import numpy as np
import matplotlib.pyplot as plt


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
    method_list = ['klr-leaf_output', 'svm-leaf_output',
                   'random', 'tree_loss',
                   'klr_loss-leaf_output', 'svm_loss-leaf_output',
                   'maple', 'leaf_influence', 'tree_prototype',
                   'knn-leaf_output', 'knn_loss-leaf_output']

    label_list = ['TREX-KLR', 'TREX-SVM', 'Random',
                  'GBDT Loss', 'KLR Loss', 'SVM Loss',
                  'MAPLE', 'LeafInfluence', 'TreeProto',
                  'TEKNN', 'KNN Loss']

    color_list = ['blue', 'cyan', 'red', 'green', 'purple', 'magenta', 'orange',
                  'black', '#EEC64F', 'yellow', 'brown']

    marker_list = ['1', '2', 'o', 'v', '^', '<', '>', '.', '*', 'h', 's']

    zorder_list = [11, 10, 9, 3, 2, 1, 7, 1, 6, 5, 8]

    # get results
    df = pd.read_csv(os.path.join(args.in_dir, 'results.csv'))

    # filter results
    df = df[df['dataset'] == args.dataset]
    df = df[df['model'] == args.model]

    # obtain clean results
    if args.metric in ['acc', 'auc']:
        metric_clean = df['{}_clean'.format(args.metric)].mean()

    # result containers
    lines = []
    labels = []

    # plot each method
    fig, ax = plt.subplots()
    methods = list(zip(method_list, label_list, color_list, marker_list, zorder_list))
    for method, label, color, marker, zorder in methods:

        # get method results
        temp_df = df[df['method'] == method]

        if len(temp_df) == 0:
            continue

        # extract performance results
        temp_df = temp_df.iloc[0]
        metric_mean = temp_df['{}s_mean'.format(args.metric)]
        metric_sem = temp_df['{}s_sem'.format(args.metric)]
        checked_pcts = temp_df['checked_pcts']

        # convert results from strings to arrays
        metric_mean = np.fromstring(metric_mean[1: -1], dtype=np.float32, sep=' ')
        metric_sem = np.fromstring(metric_sem[1: -1], dtype=np.float32, sep=' ')
        checked_pcts = np.fromstring(checked_pcts[1: -1], dtype=np.float32, sep=' ')

        # plot
        line = ax.errorbar(checked_pcts, metric_mean, yerr=metric_sem,
                           marker=marker, color=color, zorder=zorder)
        lines.append(line)
        labels.append(label)

        # add metric-specific results
        if args.metric in ['acc', 'auc']:
            label = 'Acc.' if args.metric == 'acc' else 'AUC'
            ax.axhline(metric_clean, color='k', linestyle='--', label='Clean {}'.format(label))
            ax.set_ylabel('Test {}'.format(label))

        elif args.metric == 'fixed_pct':
            ax.set_ylabel('Corrupted labels fixed (%)')

        ax.set_xlabel('Train data checked (%)')
        ax.set_title('Census (10%)' if args.dataset == 'census_0p1' else args.dataset.capitalize())
        ax.tick_params(axis='both', which='major')

    # adjust plot
    fig.legend(tuple(lines), tuple(labels), loc='center', ncol=3, bbox_to_anchor=(0.5, 0.125))
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.425, wspace=0.275)

    # create output diretory
    out_dir = os.path.join(args.out_dir, args.model, args.metric)
    os.makedirs(out_dir, exist_ok=True)

    # save plot
    fp = os.path.join(out_dir, '{}.pdf'.format(args.dataset))
    plt.savefig(fp, bbox_inches='tight')
    print('saving to {}...'.format(fp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir', type=str, default='output/cleaning/csv/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/cleaning/', help='output directory.')

    parser.add_argument('--dataset', type=str, default='churn', help='dataset.')
    parser.add_argument('--model', type=str, default='cb', help='tree-ensemble.')
    parser.add_argument('--metric', type=str, default='acc', help='acc, auc, or fixed_pcts.')

    parser.add_argument('--rs', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='random state.')
    parser.add_argument('--verbose', type=int, default=1, help='verbosity level.')

    args = parser.parse_args()
    main(args)
