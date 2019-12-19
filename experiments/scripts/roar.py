"""
Experiment: Generates an instance-attribution explanation for a test instance, sorts training
instances by influence, then removes and retrains a new tree ensemble on
this new dataset. It then re-predicts on the test instance and measures the change in
log loss. If these intances are important, than the log loss should increase.
"""
import time
import argparse
from copy import deepcopy
import os
import sys
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # for influence_boosting
sys.path.insert(0, here + '/../')  # for utility

import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import log_loss

import trex
from utility import model_util, data_util, print_util
from influence_boosting.influence.leaf_influence import CBLeafInfluenceEnsemble
from maple.MAPLE import MAPLE


def measure_loglosses(sort_indices, percentages, x_test, x_test_label, X_train, y_train, clf):
    """
    Measures the change in log loss as training instances are removed.
    """
    log_losses = []

    for percentage in tqdm.tqdm(percentages):
        n_samples = int(X_train.shape[0] * (percentage / 100))
        remove_indices = sort_indices[:n_samples]
        new_X_train = np.delete(X_train, remove_indices, axis=0)
        new_y_train = np.delete(y_train, remove_indices)

        # remeasure test instance log loss
        new_model = clone(clf).fit(new_X_train, new_y_train)
        x_test_proba = new_model.predict_proba(x_test)
        x_test_logloss = log_loss([x_test_label], x_test_proba, labels=[0, 1])
        log_losses.append(x_test_logloss)

    return log_losses


def _our_method(x_test, tree, args, X_train, y_train, X_val, seed, logger, model_dir):

    # load previously saved model
    model_path = os.path.join(model_dir, 'ours_{}.pkl'.format(args.linear_model))

    if os.path.exists(model_path):
        logger.info('loading model from: {}'.format(model_path))
        explainer = trex.TreeExplainer.load(model_path)

    else:
        explainer = trex.TreeExplainer(tree, X_train, y_train, encoding=args.encoding,
                                       dense_output=True, logger=logger, X_val=X_val,
                                       random_state=seed, use_predicted_labels=not args.true_label,
                                       kernel=args.kernel, linear_model=args.linear_model,
                                       verbose=args.verbose)
        logger.info('saving model to: {}'.format(model_path))
        explainer.save(model_path)

    # sort instanes with highest positive influence first
    contributions = explainer.explain(x_test)[0]
    if args.linear_model == 'svm':
        n_sv = len(np.where(contributions != 0)[0])
        n_pos = len(np.where(contributions > 0)[0])
        sv_pct = (n_sv / X_train.shape[0]) * 100
        logger.info('support vectors: {} ({:.2f}%), positive sv: {}'.format(n_sv, sv_pct, n_pos))

    if args.trex_absolute:
        contributions = np.abs(contributions)
    train_order = np.argsort(contributions)[::-1]
    return train_order


def _maple_method(x_test, args, model, X_train, y_train):

    train_label = y_train if args.true_label else model.predict(X_train)
    explainer = MAPLE(X_train, train_label, X_train, train_label, verbose=args.verbose, dstump=False)
    train_weight = explainer.get_weights(x_test)
    train_order = np.argsort(train_weight)[::-1]
    return train_order


def _influence_method(x_test, args, model, X_train, y_train, y_test, logger):

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

    influence_scores = []
    buf = deepcopy(explainer)
    for i in tqdm.tqdm(range(len(X_train))):
        explainer.fit(removed_point_idx=i, destination_model=buf)
        influence_scores.append(buf.loss_derivative(x_test, y_test)[0])
    influence_scores = np.array(influence_scores)

    # sort by descending order; the most positive train instances
    # are the ones that decrease the log loss the most, and are the most helpful
    train_order = np.argsort(influence_scores)[::-1]
    return train_order


def _knn_method(x_test, args, model, X_train, y_train, x_test_label, X_val, logger):

    # transform the data
    extractor = trex.TreeExtractor(model, encoding=args.encoding)
    X_train_alt = extractor.fit_transform(X_train)
    x_test_alt = extractor.transform(x_test)

    distances = np.linalg.norm(x_test_alt - X_train_alt, axis=1)

    if args.knn_absolute:
        train_order = np.argsort(distances)[::-1]
    else:
        pos_ndx = np.where(y_train == x_test_label)[0]
        neg_ndx = np.where(y_train != x_test_label)[0]

        distances[neg_ndx] *= -1
        pos_sort_ndx = np.argsort(distances[pos_ndx])
        neg_sort_ndx = np.argsort(distances[neg_ndx])
        train_order = np.concatenate([np.argsort(pos_ndx[pos_sort_ndx]), np.argsort(neg_ndx[neg_sort_ndx])])
    return train_order


def roar(args, logger, out_dir, seed):
    """
    Main method that trains a tree ensemble, flips a percentage of train labels, prioritizes train
    instances using various methods, and computes how effective each method is at cleaning the data.
    """

    # crete a models directory
    model_dir = os.path.join(out_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # get model and data
    clf = model_util.get_classifier(args.model, n_estimators=args.n_estimators, max_depth=args.max_depth,
                                    random_state=args.rs)
    X_train, X_test, y_train, y_test, label = data_util.get_data(args.dataset, random_state=args.rs,
                                                                 data_dir=args.data_dir)

    # use part of the test data as validation data
    X_val = X_test.copy()
    if args.val_frac < 1.0 and args.val_frac > 0.0:
        X_val = X_val[int(X_val.shape[0] * args.val_frac):]

    logger.info('train instances: {}'.format(len(X_train)))
    logger.info('test instances: {}\n'.format(len(X_test)))

    # train a tree ensemble
    model = clone(clf).fit(X_train, y_train)
    model_util.performance(model, X_train, y_train, X_test=X_test, y_test=y_test, logger=logger)

    # pick a test instance to explain
    test_preds = model.predict(X_test)
    if args.misclassified:
        indices = np.where(y_test != test_preds)[0]
    else:
        indices = np.where(y_test == test_preds)[0]

    np.random.seed(seed)
    test_ndx = np.random.choice(indices)
    x_test = X_test[[test_ndx]]
    x_test_label = y_test[test_ndx]

    logger.info('test index: {}'.format(test_ndx))
    logger.info('proba: {:.3f}, actual: {}'.format(model.predict_proba(x_test)[0][1], y_test[test_ndx]))
    pcts = list(range(0, 100, 10))
    # pcts = np.linspace(0.5, 10, int(10 / 0.5))

    # random method
    logger.info('ordering by random...')
    start = time.time()
    np.random.seed(seed)
    train_order = np.random.choice(np.arange(X_train.shape[0]), size=X_train.shape[0], replace=False)
    random_res = measure_loglosses(train_order, pcts, x_test, x_test_label, X_train, y_train, clf)
    logger.info('time: {:3f}s'.format(time.time() - start))

    # our method
    if args.trex:
        logger.info('ordering by our method...')
        start = time.time()
        train_order = _our_method(x_test, model, args, X_train, y_train, X_val, seed, logger, model_dir)
        our_res = measure_loglosses(train_order, pcts, x_test, x_test_label, X_train, y_train, clf)
        logger.info('time: {:3f}s'.format(time.time() - start))

    # MAPLE method
    if args.maple:
        logger.info('ordering by MAPLE...')
        start = time.time()
        train_order = _maple_method(x_test, args, model, X_train, y_train)
        maple_res = measure_loglosses(train_order, pcts, x_test, x_test_label, X_train, y_train, clf)
        logger.info('time: {:3f}s'.format(time.time() - start))

    # influence method
    if args.model == 'cb' and args.inf_k is not None:
        logger.info('ordering by LeafInfluence...')
        start = time.time()
        train_order = _influence_method(x_test, args, model, X_train, y_train, y_test[[test_ndx]], logger)
        leafinfluence_res = measure_loglosses(train_order, pcts, x_test, x_test_label, X_train, y_train, clf)
        logger.info('time: {:3f}s'.format(time.time() - start))

    # KNN method
    if args.knn:
        logger.info('ordering by knn...')
        start = time.time()
        train_order = _knn_method(x_test, args, model, X_train, y_train, x_test_label, X_val, logger)
        knn_res = measure_loglosses(train_order, pcts, x_test, x_test_label, X_train, y_train, clf)
        logger.info('time: {:3f}s'.format(time.time() - start))

    # plot results
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pcts, random_res, color='red', label='Random', marker='d')
    if args.trex:
        ax.plot(pcts, our_res, color='cyan', label='TREX-{}'.format(args.linear_model), marker='1')
    if args.maple:
        ax.plot(pcts, maple_res, color='orange', label='MAPLE', marker='>')
    if args.knn:
        ax.plot(pcts, knn_res, color='purple', label='TEKNN', marker='o')
    if args.model == 'cb' and args.inf_k is not None:
        ax.plot(pcts, leafinfluence_res, color='green', label='LeafInfluence', marker='*')
    ax.set_xlabel('train data removed (%)')
    ax.set_ylabel('log loss')
    ax.set_title('test_{}'.format(test_ndx))
    ax.legend()

    if args.save_results:

        # make seed directory
        rs_dir = os.path.join(out_dir, 'rs{}'.format(seed))
        os.makedirs(rs_dir, exist_ok=True)

        # save plot
        plt.savefig(os.path.join(rs_dir, 'roar.pdf'), bbox_inches='tight')

        # save percentages
        np.save(os.path.join(rs_dir, 'percentages.npy'), pcts)

        # random
        np.save(os.path.join(rs_dir, 'random.npy'), random_res)

        # trex
        if args.trex:
            np.save(os.path.join(rs_dir, 'ours_{}.npy'.format(args.linear_model)), our_res)

        # MAPLE
        if args.maple:
            np.save(os.path.join(rs_dir, 'maple.npy'), maple_res)

        # TEKNN
        if args.knn:
            np.save(os.path.join(rs_dir, 'teknn.npy'), knn_res)

        # LeafInfluence
        if args.model == 'cb' and args.inf_k is not None:
            np.save(os.path.join(rs_dir, 'leafinfluence.npy'), leafinfluence_res)


def main(args):

    # make logger
    dataset = args.dataset
    if args.train_frac < 1.0 and args.train_frac > 0.0:
        dataset += '_{}'.format(str(args.train_frac).replace('.', 'p'))
    out_dir = os.path.join(args.out_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)
    logger = print_util.get_logger(os.path.join(out_dir, '{}.txt'.format(args.dataset)))
    logger.info(args)

    # load saved models

    seed = args.rs
    for i in range(args.repeats):
        logger.info('\nRun {}, seed: {}'.format(i + 1, seed))
        roar(args, logger, out_dir, seed=seed)
        seed += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='adult', help='dataset to explain.')
    parser.add_argument('--out_dir', type=str, default='output/roar', help='directory to save results.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--train_frac', type=float, default=1.0, help='dataset to explain.')
    parser.add_argument('--val_frac', type=float, default=0.1, help='amount of training data to use for validation.')
    parser.add_argument('--model', type=str, default='cb', help='tree model to use.')
    parser.add_argument('--trex', action='store_true', help='Use TREX.')
    parser.add_argument('--trex_absolute', action='store_true', help='Absolute values of TREX contributions.')
    parser.add_argument('--linear_model', type=str, default='lr', help='linear model to use.')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth in tree ensemble.')
    parser.add_argument('--C', type=float, default=0.1, help='kernel model penalty parameter.')
    parser.add_argument('--kernel', default='linear', help='Similarity kernel for the linear model.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=1, help='for reproducibility.')
    parser.add_argument('--save_results', action='store_true', default=False, help='Save cleaning results.')
    parser.add_argument('--inf_k', type=int, default=None, help='Number of leaves to use for leafinfluence.')
    parser.add_argument('--maple', action='store_true', help='Whether to use MAPLE as a baseline.')
    parser.add_argument('--misclassified', action='store_true', help='explain misclassified test instance.')
    parser.add_argument('--repeats', type=int, default=5, help='Number of times to repeat the experiment.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level.')
    parser.add_argument('--true_label', action='store_true', help='Train the SVM on the true labels.')
    parser.add_argument('--knn', action='store_true', default=False, help='Use KNN on top of TREX features.')
    parser.add_argument('--knn_absolute', action='store_true', help='Absolute values of KNN contributions.')
    args = parser.parse_args()
    main(args)
