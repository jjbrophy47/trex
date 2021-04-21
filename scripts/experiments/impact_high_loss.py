"""
Experiment:
    1) Select test instance(s) at random, possibly for a specified class.
    2) Sort training instances by most influential to this test set.
    3) Remove training samples, keep removals the same as train distribution if not removing a specified class.
    4) Train a new tree-ensemble on the reduced training dataset.
    4a) Compute evaluation metric: prob. delta, acc., AUC, etc.
    5) Repeat steps 2-4a for equal increments up to X% of the train data.

Perform steps 3-4a X times (equally spaced) between 0 and 10%.

The best methods choose samples to be removed that have the largest change
or metric evaluation degradation on the test set.
"""
import os
import sys
import time
import uuid
import shutil
import resource
import argparse
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # lgb compiler warning
from copy import deepcopy
from datetime import datetime

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from scipy.stats import mode

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # for influence_boosting
sys.path.insert(0, here + '/../')  # for utility
import trex
import util
from baselines import CBLeafInfluenceEnsemble
from baselines import MAPLE
from baselines import Bacon
from baselines import DShap


def plot_alpha(alpha, y_train, fp):
    """
    Plots alpha values, sorting from most pos. to most negative.
    """
    s = np.argsort(alpha)[::-1]

    # define legend
    colormap = ListedColormap(['blue', 'red'])
    classes = [r'$y=-1$', r'$y=+1$']

    fig, ax = plt.subplots()
    scatter = ax.scatter(np.arange(len(s)), alpha[s], s=1, c=y_train[s], cmap=colormap)
    ax.axhline(0, linestyle='--', color='gray', linewidth=0.25)
    ax.set_title('Alphas (sorted)')
    ax.set_xlabel('No. instances')
    ax.set_ylabel('Alpha')
    ax.legend(handles=scatter.legend_elements()[0], labels=classes, markerscale=0.2)
    plt.savefig(fp, bbox_inches='tight')
    plt.close()


def score(model, X_test, y_test):
    """
    Evaluates the model the on test set and returns metric scores.
    """

    # 1 test sample
    if y_test.shape[0] == 1:
        result = (-1, -1)

    # >1 test sample
    else:
        acc = accuracy_score(y_test, model.predict(X_test))
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        return (acc, auc)

    return result


def score_loss(model, X_test, y_test):
    """
    Evaluates the model the on test set and returns metric scores.
    """
    proba = model.predict_proba(X_test)
    loss = np.abs([proba[i][1] - y_test[i] for i in range(proba.shape[0])])
    avg_loss = np.mean(loss)
    med_loss = np.median(loss)

    return avg_loss, med_loss


def sort_by_distribution(train_indices, y_train):
    """
    Orders `train_indices` based on the distribution in `y_train`.

    I.e. If the distrubtion is 70% positive, then every 100 samples,
         add the 70 most influential positive instances,
         then the 30 most influential negative instances.
    """
    assert train_indices.shape == y_train.shape

    # separate ordering into most sorted most influential positive and negative instances
    indices_pos = train_indices[np.isin(train_indices, np.where(y_train == 1))]
    indices_neg = train_indices[np.isin(train_indices, np.where(y_train == 0))]

    # zipper ordering back together, making sure every 100 samples represents the `y_train` distribution
    pos_count = 0
    neg_count = 0

    # no. instances to add for each class
    pos_frac = y_train.sum() / y_train.shape[0]
    n_add_pos = int(pos_frac * 100)
    n_add_neg = 100 - n_add_pos

    # result container
    new_train_indices = np.array([], dtype=np.int64)

    for i in range(0, y_train.shape[0], 100):
        new_train_indices = np.concatenate([new_train_indices, indices_pos[pos_count: pos_count + n_add_pos]])
        new_train_indices = np.concatenate([new_train_indices, indices_neg[neg_count: neg_count + n_add_neg]])

        pos_count += n_add_pos
        neg_count += n_add_neg

    # add leftovers
    new_train_indices = np.concatenate([new_train_indices, indices_pos[pos_count:]])
    new_train_indices = np.concatenate([new_train_indices, indices_neg[neg_count:]])

    # make sure the new ordering the is the same shape as the original
    assert new_train_indices.shape == train_indices.shape

    return new_train_indices


def filter_out_non_pred_label_indices(model, X_test, y_train, train_indices):
    """
    Removes train indices that do not match the majority predicted label.
    """

    # compute the majority predicted label
    maj_pred_label = mode(model.predict(X_test)).mode[0]
    maj_pred_indices = np.where(y_train == maj_pred_label)[0]

    # result container
    new_train_indices = []

    # filter
    for ndx in train_indices:
        if ndx in maj_pred_indices:
            new_train_indices.append(ndx)

    train_indices = np.array(new_train_indices)
    return train_indices


def measure_performance(args, clf, X_train, y_train, X_test, y_test, rng,
                        out_dir, logger=None):
    """
    Measures the change in predictions as training instances are removed.
    """

    # baseline predicted probability
    model = clone(clf).fit(X_train, y_train)
    base_proba = model.predict_proba(X_test)[:, 1]
    base_proba_avg = base_proba.mean()
    # acc, auc = score(model, X_test, y_test)
    avg_loss, med_loss = score_loss(model, X_test, y_test)

    # display status
    if logger:
        label_avg = y_test.sum() / y_test.shape[0]
        logger.info('\nremoving, retraining, and remeasuring predicted probability...')
        logger.info('test label (avg.): {:.2f}'.format(label_avg))
        s = '[Before] prob: {:.3f}, loss -> avg.: {:.3f}, med.: {:.3f}'
        logger.info(s.format(base_proba_avg, avg_loss, med_loss))

    # result container
    result = {}
    result['remove_pct'] = [0]
    result['avg_loss'] = [avg_loss]
    result['med_loss'] = [med_loss]
    result['proba_diff'] = [0]
    result['proba'] = [base_proba_avg]
    # result['acc'] = [acc]
    # result['auc'] = [auc]

    # only relevant if removing instances from a given class
    max_frac_to_remove = args.train_frac_to_remove
    if args.n_test <= 0 and args.start_pred in [0, 1]:
        max_frac_to_remove = len(np.where(y_train == args.start_pred)[0]) / len(y_train)

    # construct list of removed percentages
    frac_remove_list = np.linspace(0, max_frac_to_remove, 10 + 1)[1:]

    # initial sorting of train indices to be removed
    train_indices = sort_train_instances(args, model, X_train, y_train,
                                         X_test, y_test, rng, out_dir, 0.0, logger=logger)

    # trackers
    frac_deleted_data = 0

    # variables that can be modified
    if args.evaluation == 'remove':
        X_train_mod = X_train.copy()
        y_train_mod = y_train.copy()

    else:
        X_train_mod = np.zeros((0,) + tuple(X_train.shape[1:]))
        y_train_mod = np.zeros(0, int)

    # sort, remove, retrain, remeasure, repeat
    for frac_remove in frac_remove_list:
        start = time.time()
        ckpt = False
        pct_remove = frac_remove * 100

        # compute how many samples should be removed in terms of the original train data
        frac_remove_adjusted = frac_remove - frac_deleted_data
        n_remove_adjusted = int(X_train.shape[0] * frac_remove_adjusted)
        remove_indices = train_indices[:n_remove_adjusted]

        # remove most influential training samples
        if args.evaluation == 'remove':
            new_X_train = np.delete(X_train_mod, remove_indices, axis=0)
            new_y_train = np.delete(y_train_mod, remove_indices)

        # add most influential training samples
        else:
            new_X_train = np.vstack([X_train_mod, X_train[remove_indices]])
            new_y_train = np.concatenate([y_train_mod, y_train[remove_indices]])

        # only samples from one class remain
        if len(np.unique(new_y_train)) == 1:
            logger.info('Only samples from one class remain!')
            break

        # measure change in test instance probability
        new_model = clone(clf).fit(new_X_train, new_y_train)

        # compute metrics
        proba = new_model.predict_proba(X_test)[:, 1]
        proba_avg = proba.mean()
        proba_avg_diff = np.abs(base_proba_avg - proba_avg)
        # acc, auc = score(new_model, X_test, y_test)
        avg_loss, med_loss = score_loss(new_model, X_test, y_test)

        # add to results
        # result['acc'].append(acc)
        # result['auc'].append(auc)
        result['avg_loss'].append(avg_loss)
        result['med_loss'].append(med_loss)
        result['proba_diff'].append(np.abs(base_proba_avg - proba_avg))
        result['proba'].append(proba_avg)
        result['remove_pct'].append(pct_remove)

        # display progress
        if logger:
            s = '[{:.1f}% removed] after prob.: {:.5f}, delta: {:.5f}, avg. loss: {:.3f}, med. loss: {:.3f}...{:.3f}s'
            if ckpt:
                s += ', ckpt!'
            logger.info(s.format(pct_remove, proba_avg, proba_avg_diff, avg_loss, med_loss, time.time() - start))

    return result


def measure_performance2(args, clf, X_train, y_train, X_test, y_test, rng,
                         out_dir, logger=None):
    """
    Measures the change in predictions as training instances are removed.
    """

    # baseline predicted probability
    model = clone(clf).fit(X_train, y_train)
    base_proba = model.predict_proba(X_test)[:, 1]
    base_proba_avg = base_proba.mean()
    acc, auc = score(model, X_test, y_test)

    # display status
    if logger:
        label_avg = y_test.sum() / y_test.shape[0]
        logger.info('\nremoving, retraining, and remeasuring predicted probability...')
        logger.info('test label (avg.): {:.2f}, before prob.: {:.5f}'.format(label_avg, base_proba_avg))

    # result container
    result = {}
    result['proba_diff'] = [0]
    result['remove_pct'] = [0]
    result['proba'] = [base_proba_avg]
    result['acc'] = [acc]
    result['auc'] = [auc]

    # only relevant if removing instances from a given class
    max_frac_to_remove = args.train_frac_to_remove
    if args.n_test <= 0 and args.start_pred in [0, 1]:
        max_frac_to_remove = len(np.where(y_train == args.start_pred)[0]) / len(y_train)

    # construct list of removed percentages
    ckpt_list = np.linspace(0, max_frac_to_remove, 10 + 1)[1:]
    non_ckpt_list = np.linspace(0, 0.1, 10 + 1)[1:]  # take extra measurements at these points
    frac_remove_list = np.unique(np.concatenate([ckpt_list, non_ckpt_list]))

    # initial sorting of train indices to be removed
    train_indices = sort_train_instances(args, model, X_train, y_train,
                                         X_test, y_test, rng, out_dir, 0.0, logger=logger)

    # trackers
    frac_deleted_data = 0

    # variables that can be modified
    if args.evaluation == 'remove':
        X_train_mod = X_train.copy()
        y_train_mod = y_train.copy()

    else:
        X_train_mod = np.zeros((0,) + tuple(X_train.shape[1:]))
        y_train_mod = np.zeros(0, int)

    # sort, remove, retrain, remeasure, repeat
    for frac_remove in frac_remove_list:
        start = time.time()
        ckpt = False
        pct_remove = frac_remove * 100

        # compute how many samples should be removed in terms of the original train data
        frac_remove_adjusted = frac_remove - frac_deleted_data
        n_remove_adjusted = int(X_train.shape[0] * frac_remove_adjusted)
        remove_indices = train_indices[:n_remove_adjusted]

        # remove most influential training samples
        if args.evaluation == 'remove':
            new_X_train = np.delete(X_train_mod, remove_indices, axis=0)
            new_y_train = np.delete(y_train_mod, remove_indices)

        # add most influential training samples
        else:
            new_X_train = np.vstack([X_train_mod, X_train[remove_indices]])
            new_y_train = np.concatenate([y_train_mod, y_train[remove_indices]])

        # only samples from one class remain
        if len(np.unique(new_y_train)) == 1:
            logger.info('Only samples from one class remain!')
            break

        # measure change in test instance probability
        new_model = clone(clf).fit(new_X_train, new_y_train)

        # compute metrics
        proba = new_model.predict_proba(X_test)[:, 1]
        proba_avg = proba.mean()
        proba_avg_diff = np.abs(base_proba_avg - proba_avg)
        acc, auc = score(new_model, X_test, y_test)

        # add to results
        result['acc'].append(acc)
        result['auc'].append(auc)
        result['proba_diff'].append(np.abs(base_proba_avg - proba_avg))
        result['proba'].append(proba_avg)
        result['remove_pct'].append(pct_remove)

        # checkpoint! recompute influence on new dataset and resort train indices
        if frac_remove in ckpt_list and args.setting == 'dynamic':
            ckpt = True
            # logger.info('Checkpoint! Recomputing influence on new training set...')

            # use updated dataset from here on
            X_train_mod = new_X_train.copy()
            y_train_mod = new_y_train.copy()

            # sort new train data
            train_indices = sort_train_instances(args, new_model, X_train_mod, y_train_mod,
                                                 X_test, y_test, rng, out_dir, frac_remove, logger=logger)

            frac_deleted_data = frac_remove

        # display progress
        if logger:
            s = '[{:.1f}% removed] after prob.: {:.5f}, delta: {:.5f}, acc: {:.3f}, auc: {:.3f}...{:.3f}s'
            if ckpt:
                s += ', ckpt!'
            logger.info(s.format(pct_remove, proba_avg, proba_avg_diff, acc, auc, time.time() - start))

    return result


def random_method(args, model, X_train, y_train, X_test, rng):
    """
    Randomly orders the training intances to be removed.
    """

    # remove training instances at random
    if args.method == 'random':

        # remove training instances from a given class
        if args.n_test <= 0 and args.start_pred in [0, 1]:
            indices = np.where(y_train == args.start_pred)[0]
            result = rng.choice(indices, size=indices.shape[0], replace=False)

        # select instances randomly
        else:
            result = rng.choice(np.arange(X_train.shape[0]), size=X_train.shape[0], replace=False)

    # remove ONLY minority class training instances
    elif args.method == 'random_minority':
        majority_label = mode(y_train).mode[0]
        minority_label = 1 if majority_label == 0 else 0
        minority_indices = np.where(y_train == minority_label)[0]
        result = rng.choice(minority_indices, size=minority_indices.shape[0], replace=False)

    # remove ONLY majority class training instances
    elif args.method == 'random_majority':
        majority_label = mode(y_train).mode[0]
        majority_indices = np.where(y_train == majority_label)[0]
        result = rng.choice(majority_indices, size=majority_indices.shape[0], replace=False)

    # removes samples ONLY from the majority predicted label class
    elif args.method == 'random_pred':
        majority_pred = mode(model.predict(X_test)).mode[0]
        pred_indices = np.where(y_train == majority_pred)[0]
        result = rng.choice(pred_indices, size=pred_indices.shape[0], replace=False)

    else:
        raise ValueError('unknown method {}'.format(args.method))

    return result


def trex_method(args, model, X_train, y_train, X_test, out_dir, frac_remove, logger=None):
    """
    Sort training instances by largest 'excitatory' influnce on the test set.

    There is a difference between
    1) the trainnig instances with the largest excitatory (or inhibitory) attributions, and
    2) the training instances with the largest influence on the PREDICTED labels
       (i.e. excitatory or inhibitory attributions w.r.t. the predicted label).
    """

    # train surrogate model
    params = util.get_selected_params(dataset=args.dataset, model=args.model, surrogate=args.method)
    train_label = y_train if 'og' in args.method else model.predict(X_train)

    logger.info('\nparams: {}'.format(params))

    surrogate = trex.train_surrogate(model=model,
                                     surrogate='klr',
                                     X_train=X_train,
                                     y_train=train_label,
                                     val_frac=args.tune_frac,
                                     metric=args.metric,
                                     seed=args.rs,
                                     params=params,
                                     logger=None)

    # save alpha
    if frac_remove == 0.0:
        alpha_dict = {'alpha': surrogate.get_alpha(), 'y_train': y_train}
        np.save(os.path.join(out_dir, 'alpha.npy'.format(frac_remove)), alpha_dict)

    # display status
    if not logger:
        logger.info('\ncomputing influence of each training sample on the test set...')

    # sort by sim
    if 'sim' in args.method:
        attributions = surrogate.similarity(X_test)

        if args.start_pred == -100:
            pass

        # compute influence based on predicted labels
        if args.start_pred == -1:
            pred_label = model.predict(X_test)

            # put positive weight if similar instances have the same label as the predicted test label
            for i in range(X_test.shape[0]):
                attributions[i] = np.where(train_label == pred_label[i], attributions[i], attributions[i] * -1)

        # put positive weight if similar instances have the same label as the predetermined label
        else:
            for i in range(X_test.shape[0]):
                attributions[i] = np.where(train_label == args.start_pred, attributions[i], attributions[i] * -1)

        # sort indices by largest similarity-influence to the test or predetermined label
        train_indices = np.argsort(attributions.sum(axis=0))[::-1]

    # sort by alpha
    elif 'alpha' in args.method:
        alpha = surrogate.get_alpha()

        # sort by largest to smallest magnitude
        if args.start_pred == -1:
            train_indices = np.argsort(np.abs(alpha))[::-1]

            print(alpha[train_indices])

        # sort by most pos.to most neg. if `start_pred` is 1, and vice versa if 0
        else:
            train_indices = np.argsort(alpha) if args.start_pred == 0 else np.argsort(alpha)[::-1]

    # sort by alpha * sim
    else:

        # compute influence based on predicted labels
        if args.start_pred == -1:

            # sort instances with the larget influence on the predicted labels of the test set
            pred = model.predict(X_test)
            attributions = surrogate.pred_influence(X_test, pred).sum(axis=0)
            train_indices = np.argsort(attributions)[::-1]

        # compute influence based on a predetermined label
        else:

            # sort by most excitatory or inhibitory
            attributions = surrogate.compute_attributions(X_test).sum(axis=0)
            train_indices = np.argsort(attributions)[::-1] if args.start_pred == 1 else np.argsort(attributions)

    return train_indices


def maple_method(args, model, X_train, y_train, X_test, logger=None,
                 frac_progress_update=0.1):
    """
    Sort training instances using MAPLE's "local training distribution".
    """

    # train a MAPLE explainer model
    train_label = y_train if 'og' in args.method else model.predict(X_train)
    maple_explainer = MAPLE(X_train, train_label, X_train, train_label,
                            verbose=args.verbose, dstump=False)

    # display status
    if not logger:
        logger.info('\ncomputing influence of each training sample on the test instance...')

    # contributions container
    contributions_sum = np.zeros(X_train.shape[0])

    # get predicted labels for the test set
    pred_label = model.predict(X_test)

    # compute similarity of each training instance to the set set
    for i in range(X_test.shape[0]):
        contributions = maple_explainer.get_weights(X_test[i])

        # consider train instances with the same label as the predicted label as excitatory, else inhibitory
        if '+' in args.method:

            # random test instances
            if args.start_pred == -1:
                contributions = np.where(train_label == pred_label[i], contributions, contributions * -1)

            # predetermined start prediction
            else:

                # gives positive weight to excitatory (w.r.t. to the start prediction) instances
                contributions = np.where(train_label == args.start_pred, contributions, contributions * -1)

        contributions_sum += contributions

    # sort train data based largest similarity (MAPLE) OR largest similarity-influence (MAPLE+) to test set
    train_indices = np.argsort(contributions_sum)[::-1]

    # train_indices = filter_out_non_pred_label_indices(model, X_test, y_train, train_indices)

    return train_indices


def bacon_method(args, model, X_train, y_train, X_test, logger=None,
                 frac_progress_update=0.1):
    """
    Sort training instances using MAPLE's "local training distribution".
    """

    # train a MAPLE explainer model
    bacon_explainer = Bacon(model, X_train, y_train)

    # display status
    if not logger:
        logger.info('\ncomputing influence of each training sample on the test instance...')

    designated_label = None if args.start_pred == -1 else args.start_pred

    # compute similarity of each training instance to the set set
    contributions = np.zeros(X_train.shape[0])
    for i in range(X_test.shape[0]):
        contributions += bacon_explainer.get_weights(X_test[i], y=designated_label)

    # most excitatory to most inhibitory
    train_indices = np.argsort(contributions)[::-1]

    return train_indices


def influence_method(args, model, X_train, y_train, X_test, y_test, logger=None,
                     k=-1, update_set='AllPoints', frac_progress_update=0.1):
    """
    Sort training instances based on their Leaf Influence on the test set.

    Inf.(x_i) := L(y, F(x_t)) - L(y, F w/o x_i (x_t))
    A neg. number means removing x_i increases the loss (i.e. adding x_i decreases loss).
    A pos. number means removing x_i decreases the loss (i.e. adding x_i increases loss).

    Reference:
    https://github.com/kohpangwei/influence-release/blob/master/influence/experiments.py
    """

    # LeafInfluence settings
    if 'fast' in args.method:
        k = 0
        update_set = 'SinglePoint'

    assert args.model == 'cb', 'tree-ensemble is not a CatBoost model!'

    # save CatBoost model
    temp_dir = os.path.join('.catboost_info', 'leaf_influence_{}'.format(str(uuid.uuid4())))
    temp_fp = os.path.join(temp_dir, 'cb.json')
    os.makedirs(temp_dir, exist_ok=True)
    model.save_model(temp_fp, format='json')

    # initialize Leaf Influence
    explainer = CBLeafInfluenceEnsemble(temp_fp,
                                        X_train,
                                        y_train,
                                        k=k,
                                        learning_rate=model.learning_rate_,
                                        update_set=update_set)

    # display status
    if logger:
        logger.info('\ncomputing influence of each training sample on the test set...')

    # contributions container
    start = time.time()

    # predicted test labels
    model_pred = model.predict(X_test)

    inf_list = []
    buf = deepcopy(explainer)

    # compute influence for each training instance
    for j in range(X_train.shape[0]):
        explainer.fit(removed_point_idx=j, destination_model=buf)

        # approximate loss on the predicted label
        if args.start_pred == -1:
            inf_list.append(buf.loss_derivative(X_test, model_pred))

        # approximate loss on a predetermined label
        else:
            inf_list.append(buf.loss_derivative(X_test, np.array([args.start_pred])))

        # display progress
        if logger and j % int(X_train.shape[0] * frac_progress_update) == 0:
            elapsed = time.time() - start
            train_frac_complete = j / X_train.shape[0] * 100
            logger.info('train {:.1f}%...{:.3f}s'.format(train_frac_complete, elapsed))

    # compute avg. influence of each train instance on the test set loss
    influence = np.vstack(inf_list)  # shape=(no. train, no. test)
    influence = influence.mean(axis=1)  # shape=(no. train)

    # sort by most neg. to most pos.
    train_indices = np.argsort(influence)

    # clean up
    shutil.rmtree(temp_dir)

    return train_indices


def teknn_method(args, model, X_train, y_train, X_test, logger=None):
    """
    Sort trainnig instance based on similarity density to the test instances.
    """

    # train surrogate model
    params = util.get_selected_params(dataset=args.dataset, model=args.model, surrogate=args.method)
    train_label = y_train if 'og' in args.method else model.predict(X_train)
    surrogate = trex.train_surrogate(model=model,
                                     surrogate='knn',
                                     X_train=X_train,
                                     y_train=train_label,
                                     val_frac=args.tune_frac,
                                     metric=args.metric,
                                     seed=args.rs,
                                     params=params,
                                     logger=None)

    # sort instances based on largest influence w.r.t. the predicted labels of the test set
    if args.start_pred == -1:
        attributions = surrogate.compute_attributions(X_test).sum(axis=0)

    # sort instances based on largest influence w.r.t. the given label
    else:
        attributions = surrogate.compute_attributions(X_test, start_pred=args.start_pred).sum(axis=0)

    # sort based by largest similarity-influence to the test set
    train_indices = np.argsort(attributions)[::-1]

    return train_indices


def dshap_method(args, model, X_train, y_train, X_test, logger=None):
    """
    Data Shapley method, with the evaluation as predicted probability.
    """
    ds = DShap(model,
               X_train=X_train,
               y_train=y_train,
               X_test=X_test,
               y_test=np.random.randint(2, size=X_test.shape[0]),  # not used since evaluation='proba'
               metric='proba',
               random_state=args.rs)

    # compute marginals of training instances, pos. means excitatory, neg. means inhibitory
    marginals = ds.tmc_shap()

    # flip marginals if predicted label is negative
    model_pred = model.predict(X_test)[0]
    if model_pred == 0:
        marginals *= -1

    # sort by largest positive value, meaning most contributory to the PREDICTED label
    train_indices = np.argsort(marginals)[::-1]

    return train_indices


def sort_train_instances(args, model, X_train, y_train, X_test, y_test, rng, out_dir, frac_remove, logger=None):
    """
    Sorts training instance to be removed using one of several methods.
    """

    # random
    if 'random' in args.method:
        train_indices = random_method(args, model, X_train, y_train, X_test, rng)

    # TREX method
    elif 'klr' in args.method or 'svm' in args.method:
        train_indices = trex_method(args, model, X_train, y_train, X_test, out_dir, frac_remove, logger=logger)

    # Bacon
    elif 'bacon' in args.method:
        train_indices = bacon_method(args, model, X_train, y_train, X_test, logger=logger)

    # MAPLE
    elif 'maple' in args.method:
        train_indices = maple_method(args, model, X_train, y_train, X_test, logger=logger)

    # Leaf Influence (NOTE: can only compute influence of the LOSS, requires label)
    elif 'leaf_influence' in args.method and args.model == 'cb':
        train_indices = influence_method(args, model, X_train, y_train, X_test, y_test, logger=logger)

    # TEKNN
    elif 'knn' in args.method:
        train_indices = teknn_method(args, model, X_train, y_train, X_test, logger=logger)

    # Data Shapley
    elif args.method == 'dshap':
        train_indices = dshap_method(args, model, X_train, y_train, X_test, logger=logger)

    else:
        raise ValueError('method {} unknown!'.format(args.method))

    # alternate between removing positive and negative instances based on the train distribution
    if args.n_test == 1 or (args.n_test > 1 and args.start_pred == -1):
        train_indices = sort_by_distribution(train_indices, y_train)

    return train_indices


def experiment(args, logger, out_dir):
    """
    Main method that removes training instances ordered by
    different methods and measure their impact on a random
    set of test instances.
    """

    # start timer
    begin = time.time()

    # create random number generator
    rng = np.random.default_rng(args.rs)

    # get data
    data = util.get_data(args.dataset,
                         data_dir=args.data_dir,
                         preprocessing=args.preprocessing)
    X_train, X_test, y_train, y_test, feature, cat_indices = data

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

    # compute loss of each test instance
    proba = model.predict_proba(X_test)
    losses = np.abs([proba[i][1] - y_test[i] for i in range(proba.shape[0])])

    # plot highest losses and their corresponding labels
    sns.displot(losses)
    plt.show()

    # select instances with an L1 loss >= 0.9
    test_indices = np.where(losses >= 0.9)[0]
    n_pos = y_test[test_indices].sum()

    logger.info('\nNo. test instances w/ L1 loss >= 0.9: {:,}, no. pos.: {:,}'.format(len(test_indices), n_pos))

    for i, ndx in enumerate(test_indices):
        logger.info('\nTest {}, loss: {:.3f}'.format(ndx, losses[ndx]))

        X_test_sub = X_test[[ndx]]
        y_test_sub = y_test[[ndx]]
        instance_dir = os.path.join(out_dir, 'test_{}'.format(i))

        os.makedirs(instance_dir, exist_ok=True)

        # display dataset statistics
        logger.info('\nno. train instances: {:,}'.format(X_train.shape[0]))
        logger.info('no. test instances: {:,}'.format(X_test_sub.shape[0]))
        logger.info('no. features: {:,}\n'.format(X_train.shape[1]))

        # sort train instances, then remove, retrain, and re-evaluate
        result = measure_performance(args, clf, X_train, y_train,
                                     X_test_sub, y_test_sub, rng, instance_dir, logger=logger)

        # save results
        result['max_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['total_time'] = time.time() - begin
        np.save(os.path.join(instance_dir, 'results.npy'), result)

        # display results
        logger.info('\nResults:\n{}'.format(result))
        logger.info('\nsaving results to {}...'.format(os.path.join(instance_dir, 'results.npy')))


def main(args):

    # define output directory
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           args.model,
                           args.preprocessing,
                           args.method,
                           args.setting,
                           'start_pred_{}'.format(args.start_pred),
                           'n_test_{}'.format(args.n_test))

    # create output directory
    os.makedirs(out_dir, exist_ok=True)
    util.clear_dir(out_dir)

    # create logger
    logger = util.get_logger(os.path.join(out_dir, 'log.txt'))
    logger.info(args)
    logger.info('\ntimestamp: {}'.format(datetime.now()))

    # run experiment
    experiment(args, logger, out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--dataset', type=str, default='churn', help='dataset to explain.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--preprocessing', type=str, default='standard', help='preprocessing directory.')
    parser.add_argument('--out_dir', type=str, default='output/impact_high_loss/', help='directory to save results.')

    # Tree-ensemble settings
    parser.add_argument('--model', type=str, default='cb', help='model to use.')
    parser.add_argument('--n_estimators', type=int, default=10, help='no. of trees.')
    parser.add_argument('--max_depth', type=int, default=3, help='max. depth in tree ensemble.')

    # Tuning settings
    parser.add_argument('--metric', type=str, default='mse', help='metric for tuning surrogate models.')
    parser.add_argument('--tune_frac', type=float, default=0.0, help='amount of data for validation.')
    parser.add_argument('--C', type=float, default=1.0, help='penalty parameters for KLR or SVM.')
    parser.add_argument('--n_neighbors', type=int, default=5, help='no. neighbors to use for KNN.')
    parser.add_argument('--tree_kernel', type=str, default='tree_output', help='tree kernel.')

    # Experiment settings
    parser.add_argument('--method', type=str, default='klr', help='method.')
    parser.add_argument('--setting', type=str, default='static', help='if dynamic, resort at each checkpoint.')
    parser.add_argument('--evaluation', type=str, default='remove', help='remove or add instances for evaluation.')
    parser.add_argument('--n_test', type=int, default=1, help='no. of test instances to evaluate.')
    parser.add_argument('--start_pred', type=int, default=1, help='0, 1, or -1; if -1, randomly picks test instances.')
    parser.add_argument('--train_frac_to_remove', type=float, default=0.1, help='fraction of train data to remove.')
    parser.add_argument('--n_checkpoints', type=int, default=10, help='no. checkpoints to perform retraining.')
    parser.add_argument('--rs', type=int, default=1, help='random state.')

    # Additional settings
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')

    args = parser.parse_args()
    main(args)
