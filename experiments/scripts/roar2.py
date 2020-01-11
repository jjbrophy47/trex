"""
Experiment: Generates an instance-attribution explanation for a test set, sorts training
instances by influence, then removes and retrains a new tree ensemble on
this new dataset. It then re-predicts on the test set and measures the change in
performance. If these intances are important, than performance should decrease.
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
from sklearn.metrics import roc_auc_score, accuracy_score

import trex
from utility import model_util, data_util, print_util
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


def _our_method(X_test, tree, args, X_train, y_train, X_val, seed, logger, model_dir):

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
    contributions_sum = np.zeros(X_train.shape[0])
    for i in tqdm.tqdm(range(X_test.shape[0])):
        contributions = explainer.explain(X_test[i].reshape(1, -1))[0]
        if args.linear_model == 'svm':
            n_sv = len(np.where(contributions != 0)[0])
            n_pos = len(np.where(contributions > 0)[0])
            sv_pct = (n_sv / X_train.shape[0]) * 100
            logger.info('support vectors: {} ({:.2f}%), positive sv: {}'.format(n_sv, sv_pct, n_pos))

        if args.trex_absolute:
            contributions = np.abs(contributions)

        contributions_sum += contributions
    train_order = np.argsort(contributions_sum)[::-1]
    return train_order


def _maple_method(X_test, args, model, X_train, y_train, logger, model_dir):

    # load previously saved model
    model_path = os.path.join(model_dir, 'maple.pkl')

    if os.path.exists(model_path):
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
    extractor = trex.TreeExtractor(model, encoding=args.encoding)
    X_train_alt = extractor.fit_transform(X_train)
    X_test_alt = extractor.transform(X_test)

    # setup aggregate data container
    contributions_sum = np.zeros(X_train.shape[0])
    train_label = y_train if args.true_label else model.predict(X_train)

    # compute the contribution of all training samples for each test instance
    for i in tqdm.tqdm(range(X_test.shape[0])):
        distances = np.linalg.norm(X_test_alt[i] - X_train_alt, axis=1)
        contributions = np.divide(1, distances, out=np.zeros_like(distances), where=distances != 0)

        if not args.knn_absolute:
            neg_ndx = np.where(train_label != y_test[i])[0]
            contributions[neg_ndx] *= -1

        contributions_sum += contributions

    train_order = np.argsort(contributions_sum)[::-1]
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

    # use part of the train data
    if args.train_frac < 1.0 and args.train_frac > 0.0:
        n_samples = int(X_train.shape[0] * args.train_frac)
        X_train, y_train = X_train[:n_samples], y_train[:n_samples]

    # use part of the test data as validation data
    X_val = X_test.copy()
    if args.val_frac < 1.0 and args.val_frac > 0.0:
        X_val = X_val[int(X_val.shape[0] * args.val_frac):]

    # use part of the test data for evaluation
    if args.test_frac < 1.0 and args.test_frac > 0.0:
        n_test_samples = int(X_test.shape[0] * args.test_frac)
        X_test, y_test = X_test[:n_test_samples], y_test[:n_test_samples]

    logger.info('train instances: {}'.format(len(X_train)))
    logger.info('test instances: {}\n'.format(len(X_test)))

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
        train_order = _our_method(X_test, model, args, X_train, y_train, X_val, seed, logger, model_dir)
        our_res = _measure_performance(train_order, pcts, X_test, y_test, X_train, y_train, clf)
        logger.info('time: {:3f}s'.format(time.time() - start))

    # MAPLE method
    if args.maple:
        logger.info('ordering by MAPLE...')
        start = time.time()
        train_order = _maple_method(X_test, args, model, X_train, y_train, logger, model_dir)
        maple_res = _measure_performance(train_order, pcts, X_test, y_test, X_train, y_train, clf)
        logger.info('time: {:3f}s'.format(time.time() - start))

    # influence method
    if args.model == 'cb' and args.inf_k is not None:
        logger.info('ordering by LeafInfluence...')
        start = time.time()
        train_order = _influence_method(X_test, args, model, X_train, y_train, y_test, logger)
        leafinfluence_res = _measure_performance(train_order, pcts, X_test, y_test, X_train, y_train, clf)
        logger.info('time: {:3f}s'.format(time.time() - start))

    # KNN method
    if args.knn:
        logger.info('ordering by knn...')
        start = time.time()
        train_order = _knn_method(X_test, args, model, X_train, y_train, y_test, logger)
        knn_res = _measure_performance(train_order, pcts, X_test, y_test, X_train, y_train, clf)
        logger.info('time: {:3f}s'.format(time.time() - start))

    # plot results
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    ax = axs[0]
    ax.plot(pcts, random_res[0], color='red', label='Random', marker='d')
    if args.trex:
        ax.plot(pcts[:len(our_res[0])], our_res[0], color='cyan', label='TREX-{}'.format(args.linear_model),
                marker='1')
    if args.maple:
        ax.plot(pcts[:len(maple_res[0])], maple_res[0], color='orange', label='MAPLE', marker='>')
    if args.knn:
        ax.plot(pcts[:len(knn_res[0])], knn_res[0], color='purple', label='TEKNN', marker='o')
    if args.model == 'cb' and args.inf_k is not None:
        ax.plot(pcts[:len(leafinfluence_res[0])], leafinfluence_res[0], color='green', label='LeafInfluence',
                marker='*')
    ax.set_xlabel('train data removed (%)')
    ax.set_ylabel('roc_auc')

    ax = axs[1]
    ax.plot(pcts, random_res[1], color='red', label='Random', marker='d')
    if args.trex:
        ax.plot(pcts[:len(our_res[1])], our_res[1], color='cyan', label='TREX-{}'.format(args.linear_model),
                marker='1')
    if args.maple:
        ax.plot(pcts[:len(maple_res[1])], maple_res[1], color='orange', label='MAPLE', marker='>')
    if args.knn:
        ax.plot(pcts[:len(knn_res[1])], knn_res[1], color='purple', label='TEKNN', marker='o')
    if args.model == 'cb' and args.inf_k is not None:
        ax.plot(pcts[:len(leafinfluence_res[1])], leafinfluence_res[1], color='green', label='LeafInfluence',
                marker='*')
    ax.set_xlabel('train data removed (%)')
    ax.set_ylabel('test accuracy')
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
    parser.add_argument('--out_dir', type=str, default='output/roar2', help='directory to save results.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--train_frac', type=float, default=1.0, help='dataset to explain.')
    parser.add_argument('--val_frac', type=float, default=0.1, help='amount of training data to use for validation.')
    parser.add_argument('--test_frac', type=float, default=1.0, help='dataset to evaluate on.')
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
    parser.add_argument('--repeats', type=int, default=1, help='Number of times to repeat the experiment.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level.')
    parser.add_argument('--true_label', action='store_true', help='Train the SVM on the true labels.')
    parser.add_argument('--knn', action='store_true', default=False, help='Use KNN on top of TREX features.')
    parser.add_argument('--knn_absolute', action='store_true', help='Absolute values of KNN contributions.')
    args = parser.parse_args()
    main(args)
