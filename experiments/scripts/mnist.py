"""
Exploration: Examine the raw image data from the most impactful train instances for a given test instance.
    Overlay the binary or manipulation mask over the original probe image.
"""
import os
import sys
import argparse
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for TREX

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.decomposition import PCA

import trex
from utility import model_util
from utility import data_util
from utility import print_util
from utility import exp_util


def _sort_impact(sv_ndx, impact):
    """Sorts support vectors by absolute impact values."""

    if impact.ndim == 2:
        impact = np.sum(impact, axis=1)
    impact_list = zip(sv_ndx, impact)
    impact_list = sorted(sorted(impact_list, key=lambda x: x[0]), key=lambda x: abs(x[1]), reverse=True)

    sv_ndx, impact = zip(*impact_list)
    sv_ndx = np.array(sv_ndx)
    return sv_ndx, impact


def _display_image(x, identifier, predicted, actual, ax=None,
                   impact=None, weight=None, similarity=None, linewidth=3):

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(x.reshape(28, 28), cmap='gray')

    s = identifier
    s += '\n{} predicted as {}'.format(actual, predicted)

    if impact is not None:
        s += '\nimpact: {:.3f}'.format(impact)

    if weight is not None:
        s += '\n' + r'$\alpha$: {:.5f}'.format(weight)

    if similarity is not None:
        s += '\nsimilarity: {:.3f}'.format(similarity)

    ax.spines['top'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.tick_params(axis='both', bottom=False, labelbottom=False, left=False, labelleft=False)
    ax.set_title(s, fontsize=10)


def experiment(args, logger, out_dir, seed):

    # get model and data
    clf = model_util.get_classifier(args.tree_type,
                                    n_estimators=args.n_estimators,
                                    max_depth=args.max_depth,
                                    random_state=args.rs)
    data = data_util.get_data(args.dataset,
                              random_state=args.rs,
                              data_dir=args.data_dir,
                              return_image_id=True,
                              test_size=args.test_size)
    X_train, X_test, y_train, y_test, label = data

    print('train instances: {}'.format(len(X_train)))
    print('test instances: {}'.format(len(X_test)))
    print('labels: {}'.format(label))

    if args.pca_components is not None:
        print('reducing features from {} to {} using PCA...'.format(X_train.shape[1],
                                                                    args.pca_components))
        pca = PCA(args.pca_components, random_state=args.rs).fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

    # fit a tree ensemble and an explainer for that tree ensemble
    print('fitting {} model...'.format(args.tree_type))
    tree = clone(clf).fit(X_train_pca, y_train)

    # use part of the test data as validation data
    X_val_pca = X_test_pca.copy()
    if args.val_frac < 1.0 and args.val_frac > 0.0:
        X_val_pca = X_val_pca[int(X_val_pca.shape[0] * args.val_frac):]

    print('fitting tree explainer...')
    explainer = trex.TreeExplainer(tree, X_train_pca, y_train,
                                   tree_kernel=args.tree_kernel,
                                   random_state=seed,
                                   use_predicted_labels=not args.true_label,
                                   kernel_model=args.kernel_model,
                                   kernel_model_kernel=args.kernel_model_kernel,
                                   C=args.C,
                                   X_val=X_val_pca,
                                   verbose=args.verbose,
                                   logger=logger)

    model_util.performance(tree, X_train_pca, y_train, X_test_pca, y_test)

    # pick a random test instance to explain
    if args.random_test:
        np.random.seed(seed)
        test_ndx = np.random.choice(y_test)

    # pick a random mispredicted test instance to explain
    else:
        y_test_label = explainer.le_.transform(y_test)
        test_dist = exp_util.instance_loss(tree.predict_proba(X_test_pca), y_test_label)
        test_dist_ndx = np.argsort(test_dist)[::-1]
        np.random.seed(seed)
        test_ndx = np.random.choice(test_dist_ndx[:50])

    x_test = X_test_pca[test_ndx].reshape(1, -1)
    test_pred = tree.predict(x_test)[0]
    test_actual = y_test[test_ndx]

    # compute the impact of each training instance
    impact = explainer.explain(x_test)[0]
    alpha = explainer.get_weight()[0]
    sim = explainer.similarity(x_test)[0]

    # sort the training instances by impact in descending order
    sort_ndx = np.argsort(impact)[::-1]

    # show the test image
    fig, axs = plt.subplots(1, 1 + args.topk_train * 2, figsize=(16, 3))
    axs = axs.flatten()
    identifier = 'test_id{}'.format(test_ndx)
    _display_image(X_test[test_ndx], identifier=identifier, predicted=test_pred, actual=test_actual, ax=axs[0])
    plt.setp(axs[0].spines.values(), color='blue')

    topk_train = args.topk_train if args.show_negatives else args.topk_train * 2

    # show positive train images
    for i, train_ndx in enumerate(sort_ndx[:topk_train]):
        i += 1
        identifier = 'train_id{}'.format(train_ndx)
        train_pred = tree.predict(X_train_pca[train_ndx].reshape(1, -1))[0]
        similarity = sim[train_ndx] if args.show_similarity else None
        weight = alpha[train_ndx] if args.show_weight else None
        plt.setp(axs[i].spines.values(), color='green')
        _display_image(X_train[train_ndx], ax=axs[i], identifier=identifier, predicted=train_pred,
                       actual=y_train[train_ndx], similarity=similarity, weight=weight)

    # show negative train images
    if args.show_negatives:
        for i, train_ndx in enumerate(sort_ndx[::-1][:topk_train]):
            i += 1 + args.topk_train
            identifier = 'train_id{}'.format(train_ndx)
            train_pred = tree.predict(X_train_pca[train_ndx].reshape(1, -1))[0]
            similarity = sim[train_ndx] if args.show_similarity else None
            weight = alpha[train_ndx] if args.show_weight else None
            plt.setp(axs[i].spines.values(), color='red')
            _display_image(X_train[train_ndx], ax=axs[i], identifier=identifier, predicted=train_pred,
                           actual=y_train[train_ndx], similarity=similarity, weight=weight)

    os.makedirs(args.out_dir, exist_ok=True)
    plt.savefig(os.path.join(args.out_dir, 'mnist.pdf'), format='pdf', bbox_inches='tight')
    plt.show()


def main(args):

    # make logger
    dataset = args.dataset

    out_dir = os.path.join(args.out_dir, dataset, 'rs{}'.format(args.rs))
    os.makedirs(out_dir, exist_ok=True)

    logger = print_util.get_logger(os.path.join(out_dir, '{}.txt'.format(args.dataset)))
    logger.info(args)

    experiment(args, logger, out_dir, seed=args.rs)

    print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist_49', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='dataset to explain.')
    parser.add_argument('--out_dir', type=str, default='output/mnist/', help='dataset to explain.')

    parser.add_argument('--val_frac', type=float, default=0.1, help='validation dataset.')
    parser.add_argument('--random_test', action='store_true', default=False, help='choose random test instance.')

    parser.add_argument('--tree_type', type=str, default='lgb', help='model to use.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees.')
    parser.add_argument('--max_depth', type=float, default=None, help='maximum tree depth.')
    parser.add_argument('--C', type=float, default=0.1, help='penalty parameter.')

    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--kernel_model', type=str, default='lr', help='linear model to use.')
    parser.add_argument('--kernel_model_kernel', type=str, default='linear', help='similarity kernel.')

    parser.add_argument('--test_size', type=float, default=0.2, help='fraction to use for testing.')
    parser.add_argument('--pca_components', type=int, default=50, help='number of pca components.')

    parser.add_argument('--topk_train', default=3, type=int, help='train subset to use.')
    parser.add_argument('--true_label', action='store_true', default=False, help='train on the true labels.')

    parser.add_argument('--show_negatives', action='store_true', default=False, help='show negative samples.')
    parser.add_argument('--show_similarity', action='store_true', default=False, help='show similarity.')
    parser.add_argument('--show_weight', action='store_true', help='show weight.')

    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level.')
    args = parser.parse_args()
    print(args)
    main(args)


class Args:
    dataset = 'mnist_49'
    data_dir = 'data'
    out_dir = 'output/mnist/'

    val_frac = 0.1
    random_test = False

    tree_type = 'lgb'
    n_estimators = 100
    max_depth = None
    C = 0.1

    tree_kernel = 'leaf_output'
    kernel_model = 'lr'
    kernel_model_kernel = 'linear'

    test_size = 0.2
    pca_components = 50

    topk_train = 3
    true_label = False

    show_negatives = False
    show_similarity = False
    show_weight = False

    rs = 1
    verbose = 0
