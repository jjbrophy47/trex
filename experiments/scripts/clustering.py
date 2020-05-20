"""
Explanation of missclassified test instances for the NC17_EvalPart1 (train) and
MFC18_EvalPart1 (test) dataset using TREX and SHAP.
"""
import os
import sys
import argparse
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import trex
from utility import model_util
from utility import data_util
from utility import print_util


def experiment(args, logger, out_dir, seed):

    # get model and data
    clf = model_util.get_classifier(args.tree_type,
                                    n_estimators=args.n_estimators,
                                    random_state=seed)
    data = data_util.get_data(args.dataset,
                              random_state=seed,
                              data_dir=args.data_dir,
                              return_feature=True)
    X_train, X_test, y_train, y_test, label, feature = data

    logger.info('train instances: {}'.format(len(X_train)))
    logger.info('test instances: {}'.format(len(X_test)))
    logger.info('no. features: {}'.format(X_train.shape[1]))

    # filter the features to be the same as MFC18
    if args.dataset == 'NC17_EvalPart1':
        reduced_feature = data_util.get_data('nc17_mfc18',
                                             random_state=seed,
                                             data_dir=args.data_dir,
                                             return_feature=True)[-1]

        keep_ndx = []
        for f1 in reduced_feature:
            for i, f2 in enumerate(feature):
                if f1 == f2:
                    keep_ndx.append(i)
        keep_ndx = np.array(keep_ndx)

        feature = feature[keep_ndx]
        X_train = X_train[:, keep_ndx]
        X_test = X_test[:, keep_ndx]

    # train a tree ensemble and explainer
    tree = clone(clf).fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test, y_test)

    # store indexes of different subgroups
    train0 = np.where(y_train == 0)[0]
    train1 = np.where(y_train == 1)[0]
    test0 = np.where(y_test == 0)[0]
    test1 = np.where(y_test == 1)[0]

    # run tsne
    logger.info('concatenating X_train and X_test...')
    X_feature = np.concatenate([X_train, X_test])

    if args.tree_kernel in ['leaf_output', 'leaf_path', 'feature_path']:
        logger.info('transforming features with TREX...')
        explainer = trex.TreeExplainer(tree, X_train, y_train,
                                       tree_kernel=args.tree_kernel,
                                       random_state=seed,
                                       kernel_model=args.kernel_model,
                                       kernel_model_kernel=args.kernel_model_kernel)
        X_feature = explainer.transform(X_feature)

    if X_feature.shape[1] > args.n_pca:
        logger.info('PCA {} to {}...'.format(X_feature.shape[1], args.n_pca))
        X_feature = PCA(n_components=args.n_pca,
                        random_state=seed).fit_transform(X_feature)

    logger.info('embedding with tsne...')
    X_embed = TSNE(verbose=args.verbose, random_state=seed).fit_transform(X_feature)

    # plot results
    n_train = len(y_train)

    fig, ax = plt.subplots()
    ax.scatter(X_embed[:, 0][:n_train][train0], X_embed[:, 1][:n_train][train0],
               color='blue', alpha=0.25, label='train (non-manip)')
    ax.scatter(X_embed[:, 0][:n_train][train1], X_embed[:, 1][:n_train][train1],
               color='red', alpha=0.25, label='train (manip)')
    ax.scatter(X_embed[:, 0][n_train:][test0], X_embed[:, 1][n_train:][test0],
               color='cyan', alpha=0.25, label='test (non-manip)')
    ax.scatter(X_embed[:, 0][n_train:][test1], X_embed[:, 1][n_train:][test1],
               color='orange', alpha=0.25, label='test (manip)')
    ax.set_xlabel('tsne 0')
    ax.set_ylabel('tsne 1')
    ax.set_title(args.dataset.capitalize())
    ax.legend()

    train_negative = X_embed[:n_train][train0]
    train_positive = X_embed[:n_train][train1]
    test_negative = X_embed[n_train:][test0]
    test_positive = X_embed[n_train:][test1]

    np.save(os.path.join(out_dir, 'train_negative'), train_negative)
    np.save(os.path.join(out_dir, 'train_positive'), train_positive)
    np.save(os.path.join(out_dir, 'test_negative'), test_negative)
    np.save(os.path.join(out_dir, 'test_positive'), test_positive)

    plt.savefig(os.path.join(out_dir, 'tsne.png'), bbox_inches='tight')
    plt.show()


def main(args):

    # make logger
    dataset = args.dataset

    out_dir = os.path.join(args.out_dir, dataset, args.tree_kernel, 'rs{}'.format(args.rs))
    os.makedirs(out_dir, exist_ok=True)

    logger = print_util.get_logger(os.path.join(out_dir, '{}.txt'.format(args.dataset)))
    logger.info(args)

    experiment(args, logger, out_dir, seed=args.rs)

    print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='nc17_mfc18', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='dataset to explain.')
    parser.add_argument('--out_dir', type=str, default='output/clustering/', help='output directory.')

    parser.add_argument('--tree_type', type=str, default='lgb', help='model to use.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees.')

    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--kernel_model', type=str, default='lr', help='kernel model to use.')
    parser.add_argument('--kernel_model_kernel', type=str, default='linear', help='similarity kernel')
    parser.add_argument('--true_label', action='store_true', help='train TREX on the true labels.')

    parser.add_argument('--n_pca', type=int, default=50, help='pca components.')
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level.')

    args = parser.parse_args()
    main(args)


class Args:
    dataset = 'nc17_mfc18'
    data_dir = 'data'
    out_dir = 'output/clustering/'

    tree_type = 'lgb'
    n_estimators = 100
    # max_depth = None
    # C = 0.1

    tree_kernel = 'leaf_output'
    kernel_model = 'lr'
    kernel_model_kernel = 'linear'
    true_label = False

    n_pca = 50
    rs = 1
    verbose = 0
