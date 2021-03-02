"""
Plots the ROAR results for a single dataset.
"""
import os
import argparse

import pandas as pd
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

    method_list = ['random', 'klr-leaf_output', 'maple', 'knn-leaf_output']
    color_list = ['red', 'blue', 'orange', 'purple']
    label_list = ['Random', 'TREX-KLR', 'MAPLE', 'TEKNN']
    marker_list = ['o', 'd', '^', 'x']

    # get results
    df = pd.read_csv(os.path.join(args.in_dir, 'results.csv'))

    # filter results
    df = df[df['dataset'] == args.dataset]
    df = df[df['model'] == args.model]

    # result containers
    lines = []

    # plot each method
    fig, ax = plt.subplots()
    methods = list(zip(method_list, label_list, color_list, marker_list))

    for method, label, color, marker in methods:

        # get method results
        temp_df = df[df['method'] == method]

        if len(temp_df) == 0:
            continue

        # extract performance results
        temp_df = temp_df.iloc[0]
        metric_mean = temp_df['{}s_mean'.format(args.metric)]
        metric_sem = temp_df['{}s_sem'.format(args.metric)]
        removed_pcts = temp_df['removed_pcts']

        # convert results from strings to arrays
        metric_mean = np.fromstring(metric_mean[1: -1], dtype=np.float32, sep=' ')
        metric_sem = np.fromstring(metric_sem[1: -1], dtype=np.float32, sep=' ')
        removed_pcts = np.fromstring(removed_pcts[1: -1], dtype=np.float32, sep=' ')

        # plot
        line = ax.errorbar(removed_pcts, metric_mean, yerr=metric_sem,
                           marker=marker, color=color, label=label)
        lines.append(line)

        # add metric-specific info
        if args.metric in ['acc', 'auc']:
            label = 'Acc.' if args.metric == 'acc' else 'AUC'
            ax.set_ylabel('Test {}'.format(label))

        elif 'proba_delta' in args.metric:
            ax.set_ylabel(r'Avg. test prob. $\Delta$')

        else:
            raise ValueError('unknown metric: {}'.format(args.metric))

    # adjust plot
    ax.set_xlabel('Train data removed (%)')
    ax.legend()
    plt.tight_layout()

    # create output directory
    out_dir = os.path.join(args.out_dir, args.model, args.metric)
    os.makedirs(out_dir, exist_ok=True)

    # adjust and save plot
    fp = os.path.join(out_dir, '{}.pdf'.format(args.dataset))
    plt.savefig(fp, bbox_inches='tight')
    print('saving to {}...'.format(fp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir', type=str, default='output/roar/csv/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/roar/', help='output directory.')
    parser.add_argument('--dataset', type=str, default='churn', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='cb', help='tree type.')
    parser.add_argument('--metric', type=str, default='acc', help='acc, auc, avg_proba_delta, median_proba_delta')

    args = parser.parse_args()
    main(args)
