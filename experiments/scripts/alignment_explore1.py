"""
Exploration: Look through a subet of test instances for medifor, find similar train instances through use of
the tree ensemble feature representations and a similarity kernel.
"""
import argparse

import tqdm
import sexee
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone

from util import model_util, data_util


def _similarity(explainer, test_indices, X_test, random_state=69, test_subset=1000, train_subset=None,
                train_indices=None, agg_func=np.mean, verbose=True):

    if test_subset > len(test_indices):
        test_subset = len(test_indices)

    np.random.seed(random_state)
    manip_choice = np.random.choice(test_indices, size=test_subset, replace=False)

    sim_agg = []
    for test_ndx in tqdm.tqdm(manip_choice, disable=not verbose):
        sim = explainer.similarity(X_test[test_ndx], train_indices=train_indices)
        sim_agg.append(agg_func(np.sort(sim[np.nonzero(sim)])[::-1][:train_subset]))

    result = agg_func(sim_agg)
    return result


def feature_similarity(model='lgb', encoding='tree_path', dataset='medifor', n_estimators=100,
                       random_state=69, test_subset=1000, train_subset=None, agg_type='mean',
                       data_dir='data', verbose=True):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir)

    # train a tree ensemble and explainer
    tree = clone(clf).fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test, y_test)
    explainer = sexee.TreeExplainer(tree, X_train, y_train, encoding=encoding, random_state=random_state)

    # choose aggregate function
    if agg_type == 'mean':
        agg_func = np.mean
    elif agg_type == 'max':
        agg_func = np.max
    else:
        exit('agg_type {} not supported'.format(agg_type))

    # Q1: how similar are the test instances to the train instances from the tree ensemble's perspective?
    result1 = _similarity(explainer, np.arange(len(y_test)), X_test, random_state, test_subset, train_subset,
                          agg_func=agg_func, verbose=verbose)
    print('\ntest and train similarity: {:.3f}'.format(result1))

    # Q2: Similarity between manipulated test and train?
    manip_test = np.where(y_test == 1)[0]
    result2 = _similarity(explainer, manip_test, X_test, random_state, test_subset, train_subset,
                          agg_func=agg_func, verbose=verbose)
    print('manip test and train similarity: {:.3f}'.format(result2))

    # Q3: Similarity between nonmanipulated test and train?
    nonmanip_test = np.where(y_test == 0)[0]
    result3 = _similarity(explainer, nonmanip_test, X_test, random_state, test_subset, train_subset,
                          agg_func=agg_func, verbose=verbose)
    print('nonmanip test and train similarity: {:.3f}'.format(result3))

    # Q4: Similarity between manipulated test and manipulated train?
    manip_train = np.where(y_train == 1)[0]
    result4 = _similarity(explainer, manip_test, X_test, random_state, test_subset, train_subset, manip_train,
                          agg_func=agg_func, verbose=verbose)
    print('manip test and manip train similarity: {:.3f}'.format(result4))

    # Q5: Similarity between manipulated test and nonmanipulated train?
    nonmanip_train = np.where(y_train == 0)[0]
    result5 = _similarity(explainer, manip_test, X_test, random_state, test_subset, train_subset, nonmanip_train,
                          agg_func=agg_func, verbose=verbose)
    print('manip test and nonmanip train similarity: {:.3f}'.format(result5))

    # Q6: Similarity between nonmanipulated test and manipulated train?
    result6 = _similarity(explainer, nonmanip_test, X_test, random_state, test_subset, train_subset, manip_train,
                          agg_func=agg_func, verbose=verbose)
    print('nonmanip test and manip train similarity: {:.3f}'.format(result6))

    # Q7: Similarity between nonmanipulated test and nonmanipulated train?
    result7 = _similarity(explainer, nonmanip_test, X_test, random_state, test_subset, train_subset, nonmanip_train,
                          agg_func=agg_func, verbose=verbose)
    print('nonmanip test and nonmanip train similarity: {:.3f}'.format(result7))

    # plot results
    fig, axs = plt.subplots(1, 4, figsize=(18, 5))
    axs[0].bar(0, result1, tick_label=['test v train'])
    axs[0].set_ylabel('similarity ({})'.format(encoding))
    axs[0].set_ylim(0, 1)
    axs[1].bar([0, 1], [result2, result3], tick_label=['manip test v train', 'nonmanip test v train'])
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha='right')
    axs[1].set_ylim(0, 1)
    axs[2].bar([0, 1], [result4, result5], tick_label=['manip test v manip train', 'manip test v nonmanip train'])
    axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=45, ha='right')
    axs[2].set_ylim(0, 1)
    axs[3].bar([0, 1], [result6, result7],
               tick_label=['nonmanip test v manip train', 'nonmanip test v nonmanip train'])
    axs[3].set_xticklabels(axs[3].get_xticklabels(), rotation=45, ha='right')
    axs[3].set_ylim(0, 1)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='medifor', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='tree_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--test_subset', metavar='N', default=1000, type=int, help='test points to sample.')
    parser.add_argument('--train_subset', metavar='N', default=None, type=int, help='train points to compare.')
    parser.add_argument('--agg_type', metavar='TYPE', default='mean', type=str, help='aggregate function to use.')
    args = parser.parse_args()
    print(args)
    feature_similarity(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.test_subset,
                       args.train_subset, args.agg_type)
