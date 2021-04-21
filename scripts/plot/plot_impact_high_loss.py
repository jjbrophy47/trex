"""
Plots the impact results.
"""
import os
import sys
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')
import util


def main(args):
    print(args)

    # settings
    dataset_list = ['churn', 'surgical', 'vaccine', 'bank_marketing', 'adult']

    # label, color, marker, linestyle, zorder
    methods = {}

    # slice 'n dice TREX results
    if args.C is None or args.C == 0.001:
        if args.kernel is None or args.kernel == 'to':
            if args.trex_type is None or args.trex_type == 'alpha':
                methods['klr_og_tree_output_alpha_C-0.001'] = ['TREX (OG-TO-Alpha-C_0.001)', 'seagreen', '2', '-', 11]
            if args.trex_type is None or args.trex_type == 'sim':
                methods['klr_og_tree_output_sim_C-0.001'] = ['TREX (OG-TO-Sim-C_0.001)', 'seagreen', '2', '--', 11]
            if args.trex_type is None or args.trex_type == 'alphasim':
                methods['klr_og_tree_output_C-0.001'] = ['TREX (OG-TO-AlphaSim-C_0.001)', 'seagreen', '2', ':', 11]

        if args.kernel is None or args.kernel == 'lp':
            if args.trex_type is None or args.trex_type == 'alpha':
                methods['klr_og_leaf_path_alpha_C-0.001'] = ['TREX (OG-LP-Alpha-C_0.001)', 'blue', '1', '-', 11]
            if args.trex_type is None or args.trex_type == 'sim':
                methods['klr_og_leaf_path_sim_C-0.001'] = ['TREX (OG-LP-Sim-C_0.001)', 'blue', '1', '--', 10]
            if args.trex_type is None or args.trex_type == 'alphasim':
                methods['klr_og_leaf_path_C-0.001'] = ['TREX (OG-LP-AlphaSim-C_0.001)', 'blue', '1', ':', 11]

        if args.kernel is None or args.kernel == 'wlp':
            if args.trex_type is None or args.trex_type == 'alpha':
                methods['klr_og_weighted_leaf_path_alpha_C-0.001'] = ['TREX (OG-WLP-Alpha-C_0.001)', 'purple', '2', '-', 11]
            if args.trex_type is None or args.trex_type == 'sim':
                methods['klr_og_weighted_leaf_path_sim_C-0.001'] = ['TREX (OG-WLP-Sim-C_0.001)', 'purple', '2', '--', 11]
            if args.trex_type is None or args.trex_type == 'alphasim':
                methods['klr_og_weighted_leaf_path_C-0.001'] = ['TREX (OG-WLP-AlphaSim-C_0.001)', 'purple', '2', ':', 11]

    if args.C is None or args.C == 1.0:
        if args.kernel is None or args.kernel == 'to':
            if args.trex_type is None or args.trex_type == 'alpha':
                methods['klr_og_tree_output_alpha_C-1.0'] = ['TREX (OG-TO-Alpha-C_1.0)', 'yellowgreen', '2', '-', 11]
            if args.trex_type is None or args.trex_type == 'sim':
                methods['klr_og_tree_output_sim_C-1.0'] = ['TREX (OG-TO-Sim-C_1.0)', 'yellowgreen', '2', '--', 11]
            if args.trex_type is None or args.trex_type == 'alphasim':
                methods['klr_og_tree_output_C-1.0'] = ['TREX (OG-TO-AlphaSim-C_1.0)', 'yellowgreen', '2', ':', 11]

        if args.kernel is None or args.kernel == 'lp':
            if args.trex_type is None or args.trex_type == 'alpha':
                methods['klr_og_leaf_path_alpha_C-1.0'] = ['TREX (OG-LP-Alpha-C_1.0)', 'cyan', '1', '-', 11]
            if args.trex_type is None or args.trex_type == 'sim':
                methods['klr_og_leaf_path_sim_C-1.0'] = ['TREX (OG-LP-Sim-C_1.0)', 'cyan', '1', '--', 10]
            if args.trex_type is None or args.trex_type == 'alphasim':
                methods['klr_og_leaf_path_C-1.0'] = ['TREX (OG-LP-AlphaSim-C_1.0)', 'cyan', '1', ':', 11]

        if args.kernel is None or args.kernel == 'wlp':
            if args.trex_type is None or args.trex_type == 'alpha':
                methods['klr_og_weighted_leaf_path_alpha_C-1.0'] = ['TREX (OG-WLP-Alpha-C_1.0)', 'magenta', '2', '-', 11]
            if args.trex_type is None or args.trex_type == 'sim':
                methods['klr_og_weighted_leaf_path_sim_C-1.0'] = ['TREX (OG-WLP-Sim-C_1.0)', 'magenta', '2', '--', 11]
            if args.trex_type is None or args.trex_type == 'alphasim':
                methods['klr_og_weighted_leaf_path_C-1.0'] = ['TREX (OG-WLP-AlphaSim-C_1.0)', 'magenta', '2', ':', 11]

    methods['klr_og_weighted_feature_path_sim_C-1.0'] = ['TREX (OG-WFP-AlphaSim-C_1.0)', 'gold', '2', '--', 11]

    methods['random'] = ['Random', 'red', 'o', '-', 9]
    # methods['maple+'] = ['MAPLE', 'orange', '^', '--', 7]
    methods['maple+_og'] = ['MAPLE (OG)', 'orange', '^', ':', 7]
    methods['leaf_influence'] = ['LeafInfluence', 'black', '.', '-', 1]
    methods['fast_leaf_influence'] = ['FastLeafInfluence', 'black', '.', '--', 1]
    # methods['knn_og'] = ['TEKNN (OG)', 'magenta', 'h', '--', 5]
    methods['bacon'] = ['Bacon', 'brown', '2', '--', 5]

    # get results
    df = pd.read_csv(os.path.join(args.in_dir, 'results.csv'))

    # filter results
    df = df[df['model'] == args.model]
    df = df[df['setting'] == args.setting]
    df = df[df['start_pred'] == args.start_pred]
    df = df[df['n_test'] == args.n_test]

    # plot settings
    util.plot_settings(fontsize=13, markersize=5)

    # inches
    width = 4.8  # Machine Learning journal
    height = util.get_height(width=width, subplots=(2, 3))

    fig, axs = plt.subplots(2, 3, figsize=(width * 3, height * 5))

    # legend containers
    lines = []
    labels = []

    # dataset incrementer
    k = 0

    # plot datasets
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):

            if k == 5:
                break

            # extract dataset results
            ax = axs[i][j]
            dataset = dataset_list[k]
            temp_df1 = df[df['dataset'] == dataset]

            #  extract categorical preprocessing results for RF - Amazon
            if args.model == 'rf' and dataset == 'amazon':
                temp_df1 = temp_df1[temp_df1['preprocessing'] == 'categorical']

            # add y-axis
            if j == 0:

                if args.metric == 'proba_diff':
                    ax.set_ylabel(r'|Test prob. $\Delta$|')

                elif args.metric == 'proba':
                    ax.set_ylabel(r'Test prob.')

                elif args.metric == 'avg_loss':
                    ax.set_ylabel(r'L1 loss')

            # add x-axis
            if i == 1:
                ax.set_xlabel('Train data removed (%)')

            # add title
            ax.set_title('Bank Marketing' if dataset == 'bank_marketing' else dataset.capitalize())
            ax.tick_params(axis='both', which='major')

            # plot each method
            for method, (label, color, marker, linestyle, zorder) in methods.items():

                # get method results
                temp_df2 = temp_df1[temp_df1['method'] == method]

                if len(temp_df2) == 0:
                    continue

                n_runs = temp_df2['num_runs'].values[0]

                # extract performance results
                temp_df2 = temp_df2.iloc[0]
                metric_mean = temp_df2['{}_mean'.format(args.metric)]
                metric_sem = temp_df2['{}_sem'.format(args.metric)]
                removed_pcts = temp_df2['remove_pct']

                # convert results from strings to arrays
                metric_mean = np.fromstring(metric_mean[1: -1], dtype=np.float32, sep=' ')
                metric_sem = np.fromstring(metric_sem[1: -1], dtype=np.float32, sep=' ')
                removed_pcts = np.fromstring(removed_pcts[1: -1], dtype=np.float32, sep=' ')

                # plot only the first 10 checkpoints
                if args.view == 'zoom':
                    metric_mean = metric_mean[:10]
                    metric_sem = metric_mean[:10]
                    removed_pcts = removed_pcts[:10]

                # plot
                yerr = metric_sem if args.view == 'normal' else None

                # TEMP
                # yerr = None
                ax_label = 'n={:,}'.format(n_runs)
                line = ax.errorbar(removed_pcts, metric_mean, yerr=yerr, marker=marker,
                                   linestyle=linestyle, color=color, zorder=zorder, label=ax_label)
                ax.legend()

                # save for legend
                if i == 0 and j == 0:
                    lines.append(line)
                    labels.append(label)

            # increment dataset
            k += 1

            # set x-aixs limits
            ax.set_xlim(left=0, right=None)

    # create output directory
    out_dir = os.path.join(args.out_dir,
                           args.model,
                           args.setting,
                           'start_pred_{}'.format(args.start_pred),
                           args.metric)
    os.makedirs(out_dir, exist_ok=True)

    # adjust legend
    fig.legend(tuple(lines), tuple(labels), loc='center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.1))
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.265, wspace=0.3)

    # save figure
    fn = 'n_test_{}_{}'.format(args.n_test, args.view)
    fn += '_{}'.format(args.kernel) if args.kernel is not None else ''
    fn += '_{}'.format(args.trex_type) if args.trex_type is not None else ''
    fn += '_C{}'.format(args.C) if args.C is not None else ''
    fp = os.path.join(out_dir, '{}.pdf'.format(fn))
    plt.savefig(fp)
    print('saving to {}...'.format(fp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir', type=str, default='output/impact_high_loss/csv/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/impact_high_loss/', help='output directory.')
    parser.add_argument('--model', type=str, default='cb', help='tree-ensemble model.')
    parser.add_argument('--metric', type=str, default='avg_loss', help='peformance metric.')
    parser.add_argument('--setting', type=str, default='static', help='evaluation setting.')
    parser.add_argument('--start_pred', type=int, default=-1, help='starting prediction.')
    parser.add_argument('--view', type=str, default='normal', help='normal or zoom.')
    parser.add_argument('--n_test', type=int, default=1, help='no. test instances.')

    # filter settings
    parser.add_argument('--C', type=float, default=None, help='specific C.')
    parser.add_argument('--trex_type', type=str, default=None, help='None, alpha, or sim.')
    parser.add_argument('--kernel', type=str, default=None, help='None, to, lp, or wlp.')

    args = parser.parse_args()
    main(args)
