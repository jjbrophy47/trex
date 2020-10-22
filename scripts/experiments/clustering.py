"""
Visualization of the embedded tree kernel space.
"""
import os
import sys
import time
import argparse
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner

import numpy as np
from sklearn.base import clone
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from trex import TreeExtractor
from utility import model_util
from utility import data_util
from utility import print_util


def align_feature(feature, reduced_feature):
    """
    Return the intersection of the two feature sets.
    """
    keep_ndx = []

    for f1 in reduced_feature:
        for i, f2 in enumerate(feature):
            if f1 == f2:
                keep_ndx.append(i)
    keep_ndx = np.array(keep_ndx)

    return keep_ndx


def reduce_and_embed(args, X_train, X_test, logger, init='random'):
    """
    Reduces dimensionality using PCA and then embeds the
    remaining features using TSNE.
    """

    # reduce dimensionality using PCA
    if args.n_pca > 0 and X_train.shape[1] > args.n_pca:
        logger.info('n_features: {:,}, n_pca: {:,}'.format(X_train.shape[1], args.n_pca))
        start = time.time()

        pca = PCA(n_components=args.n_pca, random_state=args.rs)
        X_train = pca.fit_transform(X_train)
        # X_test = pca.transform(X_test)

        logger.info('PCA...{:.3f}s'.format(time.time() - start))

    # embed feature space using TSNE
    start = time.time()

    tsne = TSNE(verbose=args.verbose, random_state=args.rs, init=init)
    X_train = tsne.fit_transform(X_train)
    # X_test = tsne.transform(X_test)

    logger.info('embedding with tsne...{:.3f}s'.format(time.time() - start))

    return X_train, X_test


def experiment(args, logger, out_dir, seed):

    # get model and data
    clf = model_util.get_classifier(args.tree_type,
                                    n_estimators=args.n_estimators,
                                    max_depth=args.max_depth,
                                    random_state=seed)

    # get original feature space
    data = data_util.get_data(args.dataset,
                              random_state=seed,
                              data_dir=args.data_dir,
                              return_feature=True)
    X_train, X_test, y_train, y_test, label, feature = data

    logger.info('\ntrain instances: {}'.format(len(X_train)))
    logger.info('test instances: {}'.format(len(X_test)))
    logger.info('no. features: {}'.format(X_train.shape[1]))

    # filter the features to be the same as MFC18
    mapping = {'NC17_EvalPart1': 'nc17_mfc18',
               'MFC18_EvalPart1': 'mfc18_mfc19',
               'MFC19_EvalPart1': 'mfc19_mfc20'}

    if args.dataset in mapping:
        reduced_feature = data_util.get_data(mapping[args.dataset],
                                             random_state=seed,
                                             data_dir=args.data_dir,
                                             return_feature=True)[-1]

        keep_ndx = align_feature(feature, reduced_feature)
        feature = feature[keep_ndx]
        X_train = X_train[:, keep_ndx]
        X_test = X_test[:, keep_ndx]

    # train a tree ensemble and explainer
    tree = clone(clf).fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test, y_test)

    # store indexes of different subgroups
    train_neg = np.where(y_train == 0)[0]
    train_pos = np.where(y_train == 1)[0]
    # test_neg = np.where(y_test == 0)[0]
    # test_pos = np.where(y_test == 1)[0]

    # transform features to tree kernel space
    logger.info('\ntransforming features into tree kernel space')
    start = time.time()
    extractor = TreeExtractor(tree, tree_kernel=args.tree_kernel)

    X_train_tree = extractor.fit_transform(X_train)
    logger.info('  train transform time: {:.3f}s'.format(time.time() - start))

    X_test_tree = extractor.transform(X_test)
    logger.info('  test transform time: {:.3f}s'.format(time.time() - start))

    # reduce dimensionality on original and tree feature spaces
    logger.info('\nembed original features into a lower dimensional space')
    X_train, X_test = reduce_and_embed(args, X_train, X_test, logger, init='random')

    logger.info('\nembed tree kernel features into a lower dimensional space')
    X_train_tree, X_test_tree = reduce_and_embed(args, X_train_tree, X_test_tree, logger, init='pca')

    # separating embedded points into train and test
    # n_train = len(y_train)
    # train_neg_embed = X_embed[:n_train][train_neg]
    # train_pos_embed = X_embed[:n_train][train_pos]
    # test_neg_embed = X_embed[n_train:][test_neg]
    # test_pos_embed = X_embed[n_train:][test_pos]

    # save original feature space results
    np.save(os.path.join(out_dir, 'train_negative'), X_train[train_neg])
    np.save(os.path.join(out_dir, 'train_positive'), X_train[train_pos])

    # save tree kenel space results
    np.save(os.path.join(out_dir, 'train_tree_negative'), X_train_tree[train_neg])
    np.save(os.path.join(out_dir, 'train_tree_positive'), X_train_tree[train_pos])

    # np.save(os.path.join(out_dir, 'test_negative'), test_negative)
    # np.save(os.path.join(out_dir, 'test_positive'), test_positive)


def main(args):

    # make logger
    dataset = args.dataset

    out_dir = os.path.join(args.out_dir, dataset, args.tree_type, args.tree_kernel)
    os.makedirs(out_dir, exist_ok=True)

    logger = print_util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)

    experiment(args, logger, out_dir, seed=args.rs)

    print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # I/O settings
    parser.add_argument('--dataset', type=str, default='churn', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='dataset directory.')
    parser.add_argument('--out_dir', type=str, default='output/clustering/', help='output directory.')

    # Tree settings
    parser.add_argument('--tree_type', type=str, default='cb', help='model to use.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees.')
    parser.add_argument('--max_depth', type=int, default=3, help='maximum depth.')

    # TREX settings
    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='type of encoding.')

    # Experiment settings
    parser.add_argument('--n_pca', type=int, default=50, help='pca components.')
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--verbose', type=int, default=2, help='verbosity level.')

    args = parser.parse_args()
    main(args)
