"""
Visualization of the embedded tree kernel space.
"""
import os
import sys
import time
import argparse
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning

import numpy as np
from sklearn.base import clone
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner
import trex
import util


def reduce_and_embed(args, X_train, X_test, logger):
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

    tsne = TSNE(verbose=args.verbose, random_state=args.rs)
    X_train = tsne.fit_transform(X_train)
    # X_test = tsne.transform(X_test)

    logger.info('embedding with tsne...{:.3f}s'.format(time.time() - start))

    return X_train, X_test


def experiment(args, logger, out_dir):

    # start timer
    begin = time.time()

    # get data
    data = util.get_data(args.dataset,
                         data_dir=args.data_dir,
                         preprocessing=args.preprocessing)
    X_train, X_test, y_train, y_test, feature, cat_indices = data

    logger.info('\ntrain instances: {:,}'.format(X_train.shape[0]))
    logger.info('test instances: {:,}'.format(X_test.shape[0]))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))

    # get tree-ensemble
    clf = util.get_model(args.model,
                         n_estimators=args.n_estimators,
                         max_depth=args.max_depth,
                         random_state=args.rs,
                         cat_indices=cat_indices)

    # train a tree ensemble
    model = clone(clf).fit(X_train, y_train)
    util.performance(model, X_train, y_train, logger=logger, name='Train')
    util.performance(model, X_test, y_test, logger=logger, name='Test')

    # store indexes of different subgroups
    train_neg = np.where(y_train == 0)[0]
    train_pos = np.where(y_train == 1)[0]
    # test_neg = np.where(y_test == 0)[0]
    # test_pos = np.where(y_test == 1)[0]

    # transform features to tree kernel space
    logger.info('\ntransforming features into tree kernel space...')
    extractor = trex.TreeExtractor(model, tree_kernel=args.tree_kernel)

    start = time.time()
    X_train_alt = extractor.transform(X_train)
    logger.info('train transform time: {:.3f}s'.format(time.time() - start))

    start = time.time()
    X_test_alt = extractor.transform(X_test)
    logger.info('test transform time: {:.3f}s'.format(time.time() - start))

    # reduce dimensionality on original and tree feature spaces
    logger.info('\nembed original features into a lower dimensional space')
    X_train, X_test = reduce_and_embed(args, X_train, X_test, logger)

    logger.info('\nembed tree kernel features into a lower dimensional space')
    X_train_alt, X_test_alt = reduce_and_embed(args, X_train_alt, X_test_alt, logger)

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
    np.save(os.path.join(out_dir, 'train_tree_negative'), X_train_alt[train_neg])
    np.save(os.path.join(out_dir, 'train_tree_positive'), X_train_alt[train_pos])

    # np.save(os.path.join(out_dir, 'test_negative'), test_negative)
    # np.save(os.path.join(out_dir, 'test_positive'), test_positive)


def main(args):

    # create output directory
    out_dir = os.path.join(args.out_dir, args.dataset, args.model, args.tree_kernel)
    os.makedirs(out_dir, exist_ok=True)

    # create logger
    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)

    # run experiment
    experiment(args, logger, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--dataset', type=str, default='churn', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='dataset directory.')
    parser.add_argument('--preprocessing', type=str, default='standard', help='preprocessing directory.')
    parser.add_argument('--out_dir', type=str, default='output/clustering/', help='output directory.')

    # Tree settings
    parser.add_argument('--model', type=str, default='cb', help='model to use.')
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
