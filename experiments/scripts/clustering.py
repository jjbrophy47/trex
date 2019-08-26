"""
Explanation of missclassified test instances for the NC17_EvalPart1 (train) and
MFC18_EvalPart1 (test) dataset using TREX and SHAP.
"""
import os
import sys
import argparse
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner; TODO: remove this dependency

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

import sexee
from utility import model_util, data_util


def feature_clustering(model='lgb', encoding='leaf_output', dataset='nc17_mfc18', n_estimators=100,
                       random_state=69, verbose=0, data_dir='data', pca_components=50,
                       save_results=False, out_dir='output/feature_clustering/',
                       linear_model='svm', kernel='linear', true_label=True):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir)

    # train a tree ensemble and explainer
    tree = clone(clf).fit(X_train, y_train)
    model_util.performance(tree, X_train, y_train, X_test, y_test)

    # store indexes of different subgroups
    train0 = np.where(y_train == 0)[0]
    train1 = np.where(y_train == 1)[0]
    test0 = np.where(y_test == 0)[0]
    test1 = np.where(y_test == 1)[0]

    # run tsne
    print('concatenating X_train and X_test...')
    X_feature = np.concatenate([X_train, X_test])

    if encoding in ['leaf_output', 'leaf_path', 'feature_path']:
        print('exracting tree features...')
        explainer = sexee.TreeExplainer(tree, X_train, y_train, encoding=encoding, random_state=random_state,
                                        linear_model=linear_model, kernel=kernel)
        print(explainer)
        X_feature = explainer.transform(X_feature)

    if X_feature.shape[1] > pca_components:
        if encoding == 'leaf_path':
            print('reducing dimensions from {} to {} with TruncatedSVD...'.format(X_feature.shape[1], pca_components))
            X_feature = TruncatedSVD(n_components=pca_components, random_state=random_state).fit_transform(X_feature)
        else:
            print('reducing dimensions from {} to {} with PCA...'.format(X_feature.shape[1], pca_components))
            X_feature = PCA(n_components=pca_components, random_state=random_state).fit_transform(X_feature)

    print('embedding with tsne...')
    X_embed = TSNE(verbose=verbose, random_state=random_state).fit_transform(X_feature)

    # plot results
    n_train = len(y_train)

    fig, ax = plt.subplots()
    ax.scatter(X_embed[:, 0][:n_train][train0], X_embed[:, 1][:n_train][train0], color='blue', alpha=0.25,
               label='train (non-manip)')
    ax.scatter(X_embed[:, 0][:n_train][train1], X_embed[:, 1][:n_train][train1], color='red', alpha=0.25,
               label='train (manip)')
    ax.scatter(X_embed[:, 0][n_train:][test0], X_embed[:, 1][n_train:][test0], color='cyan', alpha=0.25,
               label='test (non-manip)')
    ax.scatter(X_embed[:, 0][n_train:][test1], X_embed[:, 1][n_train:][test1], color='orange', alpha=0.25,
               label='test (manip)')
    ax.set_xlabel('tsne 0')
    ax.set_ylabel('tsne 1')
    ax.legend()

    if save_results:
        train_negative = X_embed[:n_train][train0]
        train_positive = X_embed[:n_train][train1]
        test_negative = X_embed[n_train:][test0]
        test_positive = X_embed[n_train:][test1]

        plot_dir = os.path.join(out_dir, dataset + '_' + encoding)
        os.makedirs(plot_dir, exist_ok=True)

        print('saving data to {}...'.format(plot_dir))
        np.save(os.path.join(plot_dir, 'train_negative'), train_negative)
        np.save(os.path.join(plot_dir, 'train_positive'), train_positive)
        np.save(os.path.join(plot_dir, 'test_negative'), test_negative)
        np.save(os.path.join(plot_dir, 'test_positive'), test_positive)

        plt.savefig(os.path.join(plot_dir, 'tsne_plot.pdf'), format='pdf', bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='nc17_mfc18', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--linear_model', type=str, default='svm', help='linear model to use.')
    parser.add_argument('--kernel', type=str, default='linear', help='similarity kernel')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--kernel', type=str, default='linear', help='similarity kernel.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in the ensemble.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--pca_components', metavar='NUM', type=int, default=50, help='pca components.')
    parser.add_argument('--true_label', action='store_true', help='train TREX on the true labels.')
    args = parser.parse_args()
    print(args)
    feature_clustering(model=args.model, encoding=args.encoding, dataset=args.dataset, n_estimators=args.n_estimators,
                       random_state=args.rs, linear_model=args.linear_model, kernel=args.kernel,
                       pca_components=args.pca_components, true_label=args.true_label)
