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


def get_height(width, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    """
    golden_ratio = 1.618
    height = (width / golden_ratio) * (subplots[0] / subplots[1])
    return height


def main(args):
    print(args)

    # settings
    dataset_list = ['surgical', 'vaccine', 'amazon', 'bank_marketing', 'adult', 'census']

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
    df = df[df['model'] == args.model]

    # matplotlib settings
    plt.rc('font', family='arial')
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.rc('axes', labelsize=13)
    plt.rc('axes', titlesize=13)
    plt.rc('legend', fontsize=12)
    # plt.rc('legend', title_fontsize=11)
    # plt.rc('lines', linewidth=1)
    # plt.rc('lines', markersize=6)

    # inches
    width = 4.8  # Machine Learning journal
    height = get_height(width=width, subplots=(2, 3))
    # fig, axs = plt.subplots(2, 3, figsize=(width * 1.75, height * 2.25))
    fig, axs = plt.subplots(2, 3, figsize=(width * 1.75, height * 2.5))

    # legend containers
    lines = []
    labels = []

    # dataset incrementer
    k = 0

    # plot datasets
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):

            # extract dataset results
            ax = axs[i][j]
            dataset = dataset_list[k]
            temp_df1 = df[df['dataset'] == dataset]

            # # obtain clean results
            line_clean = None
            if args.metric in ['acc', 'auc']:
                metric_clean = temp_df1['{}_clean'.format(args.metric)].mean()
                line_clean = ax.axhline(metric_clean, color='k', linestyle='--')

            # add y-axis
            if j == 0:
                if args.metric in ['acc', 'auc']:
                    label = 'accuracy' if args.metric == 'acc' else 'AUC'
                    ax.set_ylabel('Test {}'.format(label))

                elif args.metric == 'fixed_pct':
                    ax.set_ylabel('Corrupted labels fixed (%)')

            # add x-axis
            if i == 1:
                ax.set_xlabel('Train data checked (%)')

            # add title
            ax.set_title('Census (10%)' if dataset == 'census_0p1' else dataset.capitalize())
            ax.tick_params(axis='both', which='major')

            # plot each method
            methods = list(zip(method_list, label_list, color_list, marker_list, zorder_list))
            for method, label, color, marker, zorder in methods:

                # get method results
                temp_df2 = temp_df1[temp_df1['method'] == method]

                if len(temp_df2) == 0:
                    continue

                # extract performance results
                temp_df2 = temp_df2.iloc[0]
                metric_mean = temp_df2['{}s_mean'.format(args.metric)]
                metric_sem = temp_df2['{}s_sem'.format(args.metric)]
                checked_pcts = temp_df2['checked_pcts']

                # convert results from strings to arrays
                metric_mean = np.fromstring(metric_mean[1: -1], dtype=np.float32, sep=' ')
                metric_sem = np.fromstring(metric_sem[1: -1], dtype=np.float32, sep=' ')
                checked_pcts = np.fromstring(checked_pcts[1: -1], dtype=np.float32, sep=' ')

                # plot
                line = ax.errorbar(checked_pcts, metric_mean, yerr=metric_sem,
                                   marker=marker, color=color, zorder=zorder)

                # save for legend
                if i == 0 and j == 0:
                    lines.append(line)
                    labels.append(label)

            # add reference line
            if i == 0 and j == 0 and line_clean is not None:
                lines.append(line_clean)
                labels.append('Clean')

            # increment dataset
            k += 1

    # create output directory
    out_dir = os.path.join(args.out_dir, args.model, args.metric)
    os.makedirs(out_dir, exist_ok=True)

    # adjust legend
    # fig.legend(tuple(lines), tuple(labels), loc='center', ncol=6, bbox_to_anchor=(0.5, 0.065))
    fig.legend(tuple(lines), tuple(labels), loc='center', ncol=4, bbox_to_anchor=(0.5, 0.0925))

    # adjust figure
    plt.tight_layout()
    # fig.subplots_adjust(bottom=0.25, wspace=0.3)
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    # save figure
    fp = os.path.join(out_dir, 'all_datasets.pdf')
    plt.savefig(fp)
    print('saving to {}...'.format(fp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--in_dir', type=str, default='output/cleaning/csv/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/cleaning/', help='output directory.')

    parser.add_argument('--model', type=str, default='cb', help='tree-ensemble.')
    parser.add_argument('--metric', type=str, default='acc', help='acc, auc, or fixed_pcts.')

    parser.add_argument('--rs', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='random state.')
    parser.add_argument('--verbose', type=int, default=1, help='verbosity level.')

    args = parser.parse_args()
    main(args)
