"""
Explanations of a tree ensemble trained on the text dataset: 20newsgroups.
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
from sklearn.base import clone
from sklearn.decomposition import PCA

import trex
from utility import model_util, data_util, exp_util


def prediction_explanation(model='lgb', encoding='leaf_output', dataset='20newsgroups', n_estimators=100,
                           random_state=69, topk_train=3, test_size=0.1, show_performance=True,
                           true_label=False, pca_components=None, data_dir='data', kernel='linear',
                           linear_model='lr', categories='alt.atheism_talk.religion.misc',
                           mispredict=False, mispredict_max=100, out_dir='output/20newsgroups'):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    data = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir, categories=categories,
                              return_raw=True)
    X_train, X_test, y_train, y_test, label, X_train_raw, X_test_raw = data

    X_train = X_train.todense()
    X_test = X_test.todense()

    print('train instances: {}'.format(X_train.shape[0]))
    print('test instances: {}'.format(X_test.shape[0]))
    print('n_features: {}'.format(X_train.shape[1]))
    print('labels: {}'.format(label))

    if pca_components is not None:
        print('reducing dimensionality from {} to {} using PCA...'.format(X_train.shape[1], pca_components))
        pca = PCA(pca_components, random_state=random_state).fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

    # fit a tree ensemble and an explainer for that tree ensemble
    print('fitting {} model...'.format(model))
    tree = clone(clf).fit(X_train, y_train)

    print('fitting tree explainer...')
    explainer = trex.TreeExplainer(tree, X_train, y_train, encoding=encoding, dense_output=True,
                                   use_predicted_labels=not true_label, random_state=random_state,
                                   kernel=kernel, linear_model=linear_model)
    print(explainer)

    if show_performance:
        model_util.performance(tree, X_train, y_train, X_test, y_test)

    if mispredict:
        # pick a test instance with large loss to explain
        test_dist = exp_util.instance_loss(tree.predict_proba(X_test), y_test)
        test_dist_ndx = np.argsort(test_dist)[::-1]
        np.random.seed(random_state)
        mispredict_ndx = np.random.choice(mispredict_max)
        test_ndx = test_dist_ndx[mispredict_ndx]

    else:
        # pick a random manipulated test instance to explain
        np.random.seed(random_state)
        test_ndx = np.random.choice(y_test)

    # collect metadata about the chosen test instance
    x_test = X_test[test_ndx].reshape(1, -1)
    test_pred = tree.predict(x_test)[0]
    test_actual = y_test[test_ndx]

    # compute the impact of each training instance
    impact = explainer.explain(x_test)[0]
    alpha = explainer.get_weight()[0]
    sim = explainer.similarity(x_test)[0]

    # sort the training instances by impact in descending order
    sort_ndx = np.argsort(impact)[::-1]

    # show the test sample
    print('\ntest_id{}, predicted={}, actual={}'.format(test_ndx, label[test_pred], label[test_actual]))
    print(X_test_raw[test_ndx])

    # show positive train images
    print('\n\nPOSITIVE SAMPLES')
    for i, train_ndx in enumerate(sort_ndx[:topk_train]):
        train_pred = tree.predict(X_train[train_ndx].reshape(1, -1))[0]
        train_actual = y_train[train_ndx]
        print('\ntrain_id{}, predicted={}, actual={}'.format(train_ndx, label[train_pred], label[train_actual]))
        print('impact={:.3f}, sim={:.3f}, weight={:.3f}'.format(impact[train_ndx], sim[train_ndx], alpha[train_ndx]))
        print(X_train_raw[train_ndx])

    # show negative train images
    print('\n\nNegative SAMPLES')
    for i, train_ndx in enumerate(sort_ndx[::-1][:topk_train]):
        train_pred = tree.predict(X_train[train_ndx].reshape(1, -1))[0]
        train_actual = y_train[train_ndx]
        print('\ntrain_id{}, predicted={}, actual={}'.format(train_ndx, label[train_pred], label[train_actual]))
        print('impact={:.3f}, sim={:.3f}, weight={:.3f}'.format(impact[train_ndx], sim[train_ndx], alpha[train_ndx]))
        print(X_train_raw[train_ndx])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='20newsgroups', help='dataset to explain.')
    parser.add_argument('--categories', type=str, default='talk.religion.misc_sci.space',
                        help='categories')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--linear_model', type=str, default='lr', help='linear model to use.')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--kernel', type=str, default='linear', help='similarity kernel.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in the ensemble.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--topk_train', default=3, type=int, help='train subset to use.')
    parser.add_argument('--true_label', action='store_true', help='train linear model on the true labels.')
    parser.add_argument('--mispredict', action='store_true', help='explains a misclassified test instance.')
    parser.add_argument('--pca_components', type=int, default=None, help='Reduce dimensionality.')
    args = parser.parse_args()
    print(args)
    prediction_explanation(model=args.model, encoding=args.encoding, dataset=args.dataset, kernel=args.kernel,
                           n_estimators=args.n_estimators, random_state=args.rs, topk_train=args.topk_train,
                           linear_model=args.linear_model, true_label=args.true_label, mispredict=args.mispredict,
                           categories=args.categories, pca_components=args.pca_components)
