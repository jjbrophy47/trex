"""
Generates an instance-attribution explanation for a test set, sorts training
instances by influence, then removes and retrains a new tree ensemble on
this new dataset. It then re-predicts on the test set and measures the change in
performance. If these intances are important, than performance should decrease.
"""
import time
import argparse
from copy import deepcopy
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # for influence_boosting
sys.path.insert(0, here + '/../')  # for utility

import tqdm
import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

import trex
from utility import model_util
from utility import data_util
from utility import print_util
from utility import exp_util
from influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble
from maple.MAPLE import MAPLE

trex_explainer = None
maple_explainer = None
teknn_explainer = None
teknn_extractor = None


def _measure_performance(sort_indices, percentages, X_test, y_test, X_train, y_train, clf):
    """
    Measures the change in log loss as training instances are removed.
    """
    r = {}
    aucs = []
    accs = []

    for percentage in tqdm.tqdm(percentages):
        n_samples = int(X_train.shape[0] * (percentage / 100))
        remove_indices = sort_indices[:n_samples]
        new_X_train = np.delete(X_train, remove_indices, axis=0)
        new_y_train = np.delete(y_train, remove_indices)

        if len(np.unique(new_y_train)) == 1:
            print(percentage)
            break

        # remeasure test instance log loss
        new_model = clone(clf).fit(new_X_train, new_y_train)
        X_test_proba = new_model.predict_proba(X_test)[:, 1]
        X_test_pred = new_model.predict(X_test)
        X_test_auc = roc_auc_score(y_test, X_test_proba)
        X_test_acc = accuracy_score(y_test, X_test_pred)
        aucs.append(X_test_auc)
        accs.append(X_test_acc)

    r['auc'] = aucs
    r['acc'] = accs

    return r


def _trex_method(args, tree, X_test, X_train, y_train, seed, logger):

    global trex_explainer

    # train TREX
    if trex_explainer is None:
        trex_explainer = trex.TreeExplainer(tree, X_train, y_train,
                                            tree_kernel=args.tree_kernel,
                                            random_state=seed,
                                            true_label=args.true_label,
                                            kernel_model=args.kernel_model,
                                            verbose=args.verbose,
                                            val_frac=args.val_frac,
                                            logger=logger)

    # sort instances with highest positive influence first
    contributions_sum = np.zeros(X_train.shape[0])

    train_weight = trex_explainer.get_weight()[0]
    for i in tqdm.tqdm(range(X_test.shape[0])):
        train_sim = trex_explainer.similarity(X_test[[i]])[0]
        contributions = train_weight * train_sim
        contributions_sum += contributions

    train_order = np.argsort(contributions_sum)[::-1]
    return train_order


def _maple_method(X_test, args, model, X_train, y_train, logger):

    global maple_explainer

    train_label = y_train if args.true_label else model.predict(X_train)

    if maple_explainer is None:
        maple_explainer = MAPLE(X_train, train_label, X_train, train_label,
                                verbose=args.verbose, dstump=False)

    # order the training instances
    contributions_sum = np.zeros(X_train.shape[0])
    for i in tqdm.tqdm(range(X_test.shape[0])):
        contributions = maple_explainer.get_weights(X_test[i])
        contributions_sum += contributions
    train_order = np.argsort(contributions_sum)[::-1]
    return train_order


def _influence_method(X_test, args, model, X_train, y_train, y_test, logger):

    model_path = '.model.json'
    model.save_model(model_path, format='json')

    if args.inf_k == -1:
        update_set = 'AllPoints'
    elif args.inf_k == 0:
        update_set = 'SinglePoint'
    else:
        update_set = 'TopKLeaves'

    explainer = CBLeafInfluenceEnsemble(model_path, X_train, y_train, k=args.inf_k,
                                        learning_rate=model.learning_rate_,
                                        update_set=update_set)

    contributions_sum = np.zeros(X_train.shape[0])
    for i in tqdm.tqdm(range(X_test.shape[0])):
        contributions = []
        buf = deepcopy(explainer)

        for j in tqdm.tqdm(range(len(X_train))):
            explainer.fit(removed_point_idx=j, destination_model=buf)
            contributions.append(buf.loss_derivative(X_test[[i]], y_test[[i]])[0])

        contributions = np.array(contributions)
        contributions_sum += contributions

    # sort by descending order; the most positive train instances
    # are the ones that decrease the log loss the most, and are the most helpful
    train_order = np.argsort(contributions_sum)[::-1]
    return train_order


def _teknn_method(args, model, X_test, X_train, y_train, y_test, seed, logger):

    global teknn_explainer
    global teknn_extractor

    if teknn_explainer is None:

        # transform the data
        teknn_extractor = trex.TreeExtractor(model, tree_kernel=args.tree_kernel)
        X_train_alt = teknn_extractor.fit_transform(X_train)
        train_label = y_train if args.true_label else model.predict(X_train)

        # tune and train teknn
        teknn_explainer = exp_util.tune_knn(model, X_train, X_train_alt, train_label, args.val_frac,
                                            seed=1, logger=logger)

    # results container
    contributions_sum = np.zeros(X_train.shape[0])

    # compute the contribution of all training samples on each test instance
    for i in tqdm.tqdm(range(X_test.shape[0])):
        x_test_alt = teknn_extractor.transform(X_test[[i]])
        pred_label = int(teknn_explainer.predict(x_test_alt)[0])
        distances, neighbor_ids = teknn_explainer.kneighbors(x_test_alt)

        for neighbor_id in neighbor_ids[0]:
            contribution = 1 if y_train[neighbor_id] == pred_label else -1
            contributions_sum[neighbor_id] += contribution

    train_order = np.argsort(contributions_sum)[::-1]
    return train_order


def experiment(args, logger, out_dir, seed):
    """
    Main method that trains a tree ensemble, flips a percentage of train labels, prioritizes train
    instances using various methods, and computes how effective each method is at cleaning the data.
    """

    # get model and data
    clf = model_util.get_classifier(args.tree_type,
                                    n_estimators=args.n_estimators,
                                    max_depth=args.max_depth,
                                    random_state=1)

    data = data_util.get_data(args.dataset,
                              random_state=1,
                              data_dir=args.data_dir)

    X_train, X_test, y_train, y_test, label = data

    # use part of the train data
    if args.train_frac < 1.0 and args.train_frac > 0.0:
        n_train_samples = int(X_train.shape[0] * args.train_frac)
        train_indices = np.random.choice(X_train.shape[0], size=n_train_samples, replace=False)
        X_train, y_train = X_train[train_indices], y_train[train_indices]

    # use part of the test data for evaluation
    if args.test_frac < 1.0 and args.test_frac > 0.0:
        n_test_samples = int(X_test.shape[0] * args.test_frac)
        np.random.seed(seed)
        test_indices = np.random.choice(X_test.shape[0], size=n_test_samples, replace=False)
        X_test, y_test = X_test[test_indices], y_test[test_indices]

    elif args.n_test is not None:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.n_test, random_state=seed)
        _, test_indices = list(sss.split(X_test, y_test))[0]
        X_test, y_test = X_test[test_indices], y_test[test_indices]

    logger.info('no. train instances: {:,}'.format(len(X_train)))
    logger.info('no. test instances: {:,}'.format(len(X_test)))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))

    # train a tree ensemble
    model = clone(clf).fit(X_train, y_train)
    model_util.performance(model, X_train, y_train, X_test=X_test, y_test=y_test, logger=logger)

    pcts = list(range(0, 100, 10))
    np.save(os.path.join(out_dir, 'percentages.npy'), pcts)

    # random method
    logger.info('\nordering by random...')
    start = time.time()
    np.random.seed(seed)
    train_order = np.random.choice(np.arange(X_train.shape[0]), size=X_train.shape[0], replace=False)
    random_res = _measure_performance(train_order, pcts, X_test, y_test, X_train, y_train, clf)
    logger.info('time: {:3f}s'.format(time.time() - start))
    np.save(os.path.join(out_dir, 'random.npy'), random_res)

    # TREX method
    if args.trex:
        logger.info('\nordering by our method...')
        start = time.time()
        train_order = _trex_method(args, model, X_test, X_train, y_train, seed, logger)
        trex_res = _measure_performance(train_order, pcts, X_test, y_test, X_train, y_train, clf)
        logger.info('time: {:3f}s'.format(time.time() - start))
        np.save(os.path.join(out_dir, 'method.npy'), trex_res)

    # MAPLE method
    if args.maple:
        logger.info('\nordering by MAPLE...')
        start = time.time()
        train_order = _maple_method(X_test, args, model, X_train, y_train, logger)
        maple_res = _measure_performance(train_order, pcts, X_test, y_test, X_train, y_train, clf)
        logger.info('time: {:3f}s'.format(time.time() - start))
        np.save(os.path.join(out_dir, 'method.npy'), maple_res)

    # influence method
    if args.tree_type == 'cb' and args.inf_k is not None:
        logger.info('\nordering by LeafInfluence...')
        start = time.time()
        train_order = _influence_method(X_test, args, model, X_train, y_train, y_test, logger)
        leafinfluence_res = _measure_performance(train_order, pcts, X_test, y_test, X_train, y_train, clf)
        logger.info('time: {:3f}s'.format(time.time() - start))
        np.save(os.path.join(out_dir, 'method.npy'), leafinfluence_res)

    # TEKNN method
    if args.teknn:
        logger.info('\nordering by teknn...')
        start = time.time()
        train_order = _teknn_method(args, model, X_test, X_train, y_train, y_test, seed, logger)
        knn_res = _measure_performance(train_order, pcts, X_test, y_test, X_train, y_train, clf)
        logger.info('time: {:3f}s'.format(time.time() - start))
        np.save(os.path.join(out_dir, 'method.npy'), knn_res)


def main(args):

    # make logger
    dataset = args.dataset

    for rs in args.rs:
        out_dir = os.path.join(args.out_dir, dataset, args.tree_type,
                               'rs{}'.format(rs))

        if args.trex:
            out_dir = os.path.join(out_dir, args.kernel_model, args.tree_kernel)
        elif args.teknn:
            out_dir = os.path.join(out_dir, 'teknn', args.tree_kernel)
        elif args.maple:
            out_dir = os.path.join(out_dir, 'maple')
        elif args.inf_k is not None:
            out_dir = os.path.join(out_dir, 'leaf_influence')

        os.makedirs(out_dir, exist_ok=True)
        logger = print_util.get_logger(os.path.join(out_dir, 'log.txt'))
        logger.info(args)

        logger.info('\nSeed: {}'.format(rs))
        experiment(args, logger, out_dir, seed=rs)
        print_util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # I/O settings
    parser.add_argument('--dataset', type=str, default='adult', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--out_dir', type=str, default='output/roar/', help='directory to save results.')

    # data settings
    parser.add_argument('--train_frac', type=float, default=1.0, help='dataset to explain.')
    parser.add_argument('--val_frac', type=float, default=0.1, help='Amount of data for validation.')
    parser.add_argument('--test_frac', type=float, default=1.0, help='dataset to evaluate on.')
    parser.add_argument('--n_test', type=int, default=50, help='number of test instances.')

    # tree settings
    parser.add_argument('--tree_type', type=str, default='cb', help='model to use.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees.')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth in tree ensemble.')

    # TREX settings
    parser.add_argument('--trex', action='store_true', default=False, help='Use TREX.')
    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--kernel_model', type=str, default='klr', help='kernel model to use.')
    parser.add_argument('--true_label', action='store_true', default=False, help='train TREX on the true labels.')

    # method settings
    parser.add_argument('--teknn', action='store_true', default=False, help='Use KNN on top of TREX features.')
    parser.add_argument('--inf_k', type=int, default=None, help='Number of leaves to use for leafinfluence.')
    parser.add_argument('--maple', action='store_true', default=False, help='Whether to use MAPLE as a baseline.')

    # experiment settings
    parser.add_argument('--rs', type=int, nargs='+', default=[1], help='Random State.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level.')

    args = parser.parse_args()
    main(args)


class Args:
    dataset = 'adult'
    data_dir = 'data'
    out_dir = 'output/roar/'

    train_frac = 1.0
    val_frac = 0.1
    test_frac = 1.0
    n_test = None

    tree_type = 'cb'
    n_estimators = 100
    max_depth = None

    trex = True
    tree_kernel = 'tree_output'
    kernel_model = 'klr'
    true_label = False

    teknn = False
    inf_k = None
    maple = False

    rs = [1]
    verbose = 0
