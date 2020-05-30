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

import trex
from utility import model_util
from utility import data_util
from utility import print_util
from influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble
from maple.MAPLE import MAPLE


def _measure_performance(sort_indices, percentages, X_test, y_test, X_train, y_train, clf):
    """
    Measures the change in log loss as training instances are removed.
    """
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

    return aucs, accs


def _trex_method(X_test, tree, args, X_train, y_train,
                 X_val, seed, logger, model_dir):

    # load previously saved model
    model_path = os.path.join(model_dir, 'trex_{}_{}.pkl'.format(
                              args.kernel_model, args.tree_kernel))

    if args.trex_load and os.path.exists(model_path):
        logger.info('loading model from: {}'.format(model_path))
        explainer = trex.TreeExplainer.load(model_path)

    # train TREX
    else:
        explainer = trex.TreeExplainer(tree, X_train, y_train,
                                       tree_kernel=args.tree_kernel,
                                       random_state=seed,
                                       kernel_model=args.kernel_model,
                                       kernel_model_kernel=args.kernel_model_kernel)
        logger.info('saving model to: {}'.format(model_path))
        explainer.save(model_path)

    # sort instances with highest positive influence first
    contributions_sum = np.zeros(X_train.shape[0])

    for i in tqdm.tqdm(range(X_test.shape[0])):
        contributions = explainer.explain(X_test[i].reshape(1, -1))[0]

        if args.kernel_model == 'svm':
            n_sv = len(np.where(contributions != 0)[0])
            n_pos = len(np.where(contributions > 0)[0])
            sv_pct = (n_sv / X_train.shape[0]) * 100
            logger.info('support vectors: {} ({:.2f}%), positive sv: {}'.format(n_sv, sv_pct, n_pos))

        contributions_sum += contributions
    train_order = np.argsort(contributions_sum)[::-1]
    return train_order


def _maple_method(X_test, args, model, X_train, y_train, logger, model_dir):

    # load previously saved model
    model_path = os.path.join(model_dir, 'maple.pkl')

    if args.maple_load and os.path.exists(model_path):
        logger.info('loading model from: {}'.format(model_path))
        explainer = MAPLE.load(model_path)

    else:
        train_label = y_train if args.true_label else model.predict(X_train)
        explainer = MAPLE(X_train, train_label, X_train, train_label, verbose=args.verbose, dstump=False)
        logger.info('saving model to: {}'.format(model_path))
        explainer.save(model_path)

    # order the training instances
    contributions_sum = np.zeros(X_train.shape[0])
    for i in tqdm.tqdm(range(X_test.shape[0])):
        contributions = explainer.get_weights(X_test[i])
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
                                        learning_rate=model.learning_rate_, update_set=update_set)

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


def _knn_method(X_test, args, model, X_train, y_train, y_test, logger):

    # transform the data
    extractor = trex.TreeExtractor(model, tree_kernel=args.tree_kernel)
    X_train_alt = extractor.fit_transform(X_train)
    X_test_alt = extractor.transform(X_test)

    # setup aggregate data container
    contributions_sum = np.zeros(X_train.shape[0])
    train_label = y_train if args.true_label else model.predict(X_train)

    # compute the contribution of all training samples for each test instance
    for i in tqdm.tqdm(range(X_test.shape[0])):
        distances = np.linalg.norm(X_test_alt[i] - X_train_alt, axis=1)
        contributions = np.divide(1, distances, out=np.zeros_like(distances), where=distances != 0)

        neg_ndx = np.where(train_label != y_test[i])[0]
        contributions[neg_ndx] *= -1
        contributions_sum += contributions

    train_order = np.argsort(contributions_sum)[::-1]
    return train_order


def experiment(args, logger, out_dir, seed):
    """
    Main method that trains a tree ensemble, flips a percentage of train labels, prioritizes train
    instances using various methods, and computes how effective each method is at cleaning the data.
    """

    # crete a models directory
    model_dir = os.path.join(out_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # get model and data
    clf = model_util.get_classifier(args.tree_type,
                                    n_estimators=args.n_estimators,
                                    random_state=seed)
    data = data_util.get_data(args.dataset,
                              random_state=seed,
                              data_dir=args.data_dir)
    X_train, X_test, y_train, y_test, label = data

    # use part of the train data
    if args.train_frac < 1.0 and args.train_frac > 0.0:
        n_samples = int(X_train.shape[0] * args.train_frac)
        X_train, y_train = X_train[:n_samples], y_train[:n_samples]

    # use part of the test data for tuning
    X_val = X_test.copy()
    if args.val_frac < 1.0 and args.val_frac > 0.0:
        X_val = X_val[int(X_val.shape[0] * args.val_frac):]

    # use part of the test data for evaluation
    if args.test_frac < 1.0 and args.test_frac > 0.0:
        n_test_samples = int(X_test.shape[0] * args.test_frac)
        X_test, y_test = X_test[:n_test_samples], y_test[:n_test_samples]

    logger.info('train instances: {:,}'.format(len(X_train)))
    logger.info('val instances: {:,}'.format(len(X_val)))
    logger.info('test instances: {:,}'.format(len(X_test)))
    logger.info('no. features: {:,}'.format(X_train.shape[1]))

    # train a tree ensemble
    model = clone(clf).fit(X_train, y_train)
    model_util.performance(model, X_train, y_train, X_test=X_test, y_test=y_test, logger=logger)

    pcts = list(range(0, 100, 10))

    # random method
    logger.info('ordering by random...')
    start = time.time()
    np.random.seed(seed)
    train_order = np.random.choice(np.arange(X_train.shape[0]), size=X_train.shape[0], replace=False)
    random_res = _measure_performance(train_order, pcts, X_test, y_test, X_train, y_train, clf)
    logger.info('time: {:3f}s'.format(time.time() - start))

    # our method
    if args.trex:
        logger.info('ordering by our method...')
        start = time.time()
        train_order = _trex_method(X_test, model, args, X_train, y_train, X_val, seed, logger, model_dir)
        trex_res = _measure_performance(train_order, pcts, X_test, y_test, X_train, y_train, clf)
        logger.info('time: {:3f}s'.format(time.time() - start))

    # MAPLE method
    if args.maple:
        logger.info('ordering by MAPLE...')
        start = time.time()
        train_order = _maple_method(X_test, args, model, X_train, y_train, logger, model_dir)
        maple_res = _measure_performance(train_order, pcts, X_test, y_test, X_train, y_train, clf)
        logger.info('time: {:3f}s'.format(time.time() - start))

    # influence method
    if args.tree_type == 'cb' and args.inf_k is not None:
        logger.info('ordering by LeafInfluence...')
        start = time.time()
        train_order = _influence_method(X_test, args, model, X_train, y_train, y_test, logger)
        leafinfluence_res = _measure_performance(train_order, pcts, X_test, y_test, X_train, y_train, clf)
        logger.info('time: {:3f}s'.format(time.time() - start))

    # KNN method
    if args.teknn:
        logger.info('ordering by knn...')
        start = time.time()
        train_order = _knn_method(X_test, args, model, X_train, y_train, y_test, logger)
        knn_res = _measure_performance(train_order, pcts, X_test, y_test, X_train, y_train, clf)
        logger.info('time: {:3f}s'.format(time.time() - start))

    # save percentages
    np.save(os.path.join(out_dir, 'percentages.npy'), pcts)

    # random
    np.save(os.path.join(out_dir, 'random.npy'), random_res)

    # trex
    if args.trex:
        np.save(os.path.join(out_dir, 'trex_{}.npy'.format(args.kernel_model)), trex_res)

    # MAPLE
    if args.maple:
        np.save(os.path.join(out_dir, 'maple.npy'), maple_res)

    # TEKNN
    if args.teknn:
        np.save(os.path.join(out_dir, 'teknn.npy'), knn_res)

    # LeafInfluence
    if args.tree_type == 'cb' and args.inf_k is not None:
        np.save(os.path.join(out_dir, 'leafinfluence.npy'), leafinfluence_res)


def main(args):

    # make logger
    dataset = args.dataset

    out_dir = os.path.join(args.out_dir, dataset, args.tree_type, args.tree_kernel)
    os.makedirs(out_dir, exist_ok=True)
    logger = print_util.get_logger(os.path.join(out_dir, '{}.txt'.format(args.dataset)))
    logger.info(args)

    experiment(args, logger, out_dir, seed=args.rs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='adult', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--out_dir', type=str, default='output/roar/', help='directory to save results.')

    parser.add_argument('--train_frac', type=float, default=1.0, help='dataset to explain.')
    parser.add_argument('--val_frac', type=float, default=0.1, help='Amount of data for validation.')
    parser.add_argument('--test_frac', type=float, default=1.0, help='dataset to evaluate on.')

    parser.add_argument('--tree_type', type=str, default='cb', help='model to use.')
    parser.add_argument('--n_estimators', type=int, default=100, help='number of trees.')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth in tree ensemble.')

    parser.add_argument('--trex', action='store_true', default=False, help='Use TREX.')
    parser.add_argument('--trex_load', action='store_true', default=False, help='Load saved model.')
    parser.add_argument('--tree_kernel', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--kernel_model', type=str, default='lr', help='kernel model to use.')
    parser.add_argument('--kernel_model_kernel', type=str, default='linear', help='similarity kernel')
    parser.add_argument('--true_label', action='store_true', default=False, help='train TREX on the true labels.')

    parser.add_argument('--misclassified', action='store_true', default=False, help='Use misclassified instance.')

    parser.add_argument('--teknn', action='store_true', default=False, help='Use KNN on top of TREX features.')
    parser.add_argument('--inf_k', type=int, default=None, help='Number of leaves to use for leafinfluence.')
    parser.add_argument('--maple', action='store_true', default=False, help='Whether to use MAPLE as a baseline.')
    parser.add_argument('--maple_load', action='store_true', default=False, help='Load saved model.')

    parser.add_argument('--rs', type=int, default=1, help='Random State.')
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

    tree_type = 'lgb'
    n_estimators = 100
    max_depth = None

    trex = True
    trex_load = False
    tree_kernel = 'leaf_output'
    kernel_model = 'lr'
    kernel_model_kernel = 'linear'
    true_label = False

    misclassified = False

    knn = False
    inf_k = None
    maple = False
    maple_load = False

    rs = 1
    verbose = 0
