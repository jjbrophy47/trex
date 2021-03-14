"""
Plots the ROAR special checkpoint results.
"""
import os
import argparse

import pandas as pd
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
    method_list = ['klr', 'maple', 'knn']
    label_list = ['TREX', 'MAPLE', 'TEKNN']

    # get results
    df = pd.read_csv(os.path.join(args.in_dir, 'results.csv'))

    # filter results
    df = df[df['model'] == args.model]

    # plot settings
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', labelsize=14)
    plt.rc('axes', titlesize=14)
    plt.rc('legend', fontsize=14)
    # plt.rc('legend', title_fontsize=11)
    # plt.rc('lines', linewidth=1)
    plt.rc('lines', markersize=9)

    # inches
    width = 4.8  # Machine Learning journal
    height = get_height(width=width, subplots=(1, 3))
    fig, axs = plt.subplots(1, 3, figsize=(width * 2.45, height * 4.15))

    # extract dataset results
    temp_df1 = df[df['dataset'] == args.dataset]

    #  extract categorical preprocessing results for RF - Amazon
    if args.model == 'rf' and args.dataset == 'amazon':
        temp_df1 = temp_df1[temp_df1['preprocessing'] == 'categorical']

    # plot datasets
    for i in range(axs.shape[0]):
        ax = axs[i]

        # method settings
        method = method_list[i]
        label = label_list[i]

        # add title
        ax.set_title(label)
        ax.tick_params(axis='both', which='major')

        # get method results
        temp_df2 = temp_df1[temp_df1['method'] == method]

        if len(temp_df2) == 0:
            continue

        # extract performance results
        temp_df2 = temp_df2.iloc[0]
        model_proba = temp_df2['ckpt_model_proba']
        new_model_proba = temp_df2['ckpt_new_model_proba']
        remove_pct = temp_df2['ckpt_remove_pct']
        y_test = temp_df2['ckpt_y_test']

        # convert results from strings to arrays
        model_proba = np.fromstring(model_proba[1: -1], dtype=np.float32, sep=' ')
        new_model_proba = np.fromstring(new_model_proba[1: -1], dtype=np.float32, sep=' ')
        y_test = np.fromstring(y_test[1: -1], dtype=np.float32, sep=' ')

        # get pos. and neg. test instances
        pos_indices = np.where(y_test == 1)[0]
        neg_indices = np.where(y_test == 0)[0]

        l1 = ax.scatter(model_proba[pos_indices], new_model_proba[pos_indices],
                        marker='+', color='green')
        l2 = ax.scatter(model_proba[neg_indices], new_model_proba[neg_indices],
                        marker='.', color='red', facecolors='none')
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='--', color='black')
        ax.set_xlabel('Original model prob.')
        if i == 0:
            ax.set_ylabel('Updated model prob.')
            print('Remove %: {:.0f}%'.format(remove_pct))
        ax.set_title(label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # adjust legend
    lines = [l1, l2]
    labels = ['Positive label', 'Negative label']
    fig.legend(tuple(lines), tuple(labels), loc='center', ncol=2, bbox_to_anchor=(0.5, 0.055))
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.3)

    # create output directory
    out_dir = os.path.join(args.out_dir, args.model, 'special_ckpt')
    os.makedirs(out_dir, exist_ok=True)

    # save figure
    fp = os.path.join(out_dir, '{}.pdf'.format(args.dataset))
    plt.savefig(fp)
    print('saving to {}...'.format(fp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir', type=str, default='output/roar/csv/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/plots/roar/', help='output directory.')
    parser.add_argument('--model', type=str, default='cb', help='tree-ensemble model.')
    parser.add_argument('--dataset', type=str, default='amazon', help='datasets to analyze.')

    args = parser.parse_args()
    main(args)
