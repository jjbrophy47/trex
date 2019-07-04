"""
Exploration: Train an svm to predict if an instance came from the train or test set.
Extension: If this model works well, examine instances at the extremes, as well as uncertain instances.
    Compute SHAP values on these instances. The most important features for wrong or uncertain instances may
    be ones we should get rid of, while the most important features for right instances may be ones we
    should keep.
"""
import argparse

import shap
import sexee
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from util import model_util, data_util, print_util


def train_vs_test(model='lgb', encoding='tree_path', dataset='medifor', n_estimators=100, random_state=69,
                  plot=False, train_subset=None, data_dir='data', test_size=0.5, n_features=5, sample_explain=False):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label, feature = data_util.get_data(dataset, random_state=random_state,
                                                                          data_dir=data_dir, return_feature=True)

    # reduce train set to only instances of a certain class
    if train_subset == 'negative':
        subset_ndx = np.where(y_train == 0)[0]
        X_train = X_train[subset_ndx]
        y_train = y_train[subset_ndx]
    elif train_subset == 'positive':
        subset_ndx = np.where(y_train == 1)[0]
        X_train = X_train[subset_ndx]
        y_train = y_train[subset_ndx]

    # create dataset where negative instancs are train instances, and positive instances are test instances
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([np.zeros(len(y_train)), np.ones(len(y_test))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=y)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    label = ['train', 'test']

    # learn a classifier to tell train and test instances apart
    predictor = clone(clf).fit(X_train, y_train)
    model_util.performance(predictor, X_train, y_train, X_test, y_test)

    # plot most important features to the test instances
    shap_explainer = shap.TreeExplainer(predictor)
    test_shap = shap_explainer.shap_values(X_test)
    shap.summary_plot(test_shap, features=X_test, feature_names=feature, max_display=n_features, auto_size_plot=False)

    # explain one of the test instances
    if sample_explain:
        trex_explainer = sexee.TreeExplainer(predictor, X_train, y_train)
        train_ndx, impact, sim, weight = trex_explainer.train_impact(X_test[0], similarity=True, weight=True)
        impact_list = zip(train_ndx, impact, sim, weight)
        impact_list = sorted(impact_list, key=lambda tup: abs(tup[1]), reverse=True)
        svm_pred, pred_label = trex_explainer.decision_function(X_test[0], pred_svm=True)
        print_util.show_test_instance(0, svm_pred, pred_label, y_test=y_test, label=label, X_test=None)
        print_util.show_train_instances(impact_list, y_train, k=5, label=label, X_train=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='medifor', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='tree_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--plot', action='store_true', default=False, help='plots intermediate feaure removals.')
    parser.add_argument('--train_subset', default=None, help='train subset to use.')
    args = parser.parse_args()
    print(args)
    train_vs_test(args.model, args.encoding, args.dataset, args.n_estimators, args.rs, args.plot, args.train_subset)
