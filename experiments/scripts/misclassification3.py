"""
Explanation of missclassified test instances for the NC17_EvalPart1 (train) and
MFC18_EvalPart1 (test) dataset using TREX and SHAP. Visualize the instances using histograms.
"""
import os
import sys
import argparse
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner; TODO: remove this dependency

import shap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from trex.explainer import TreeExplainer
from utility import model_util, data_util, exp_util


def _get_top_features(x, shap_vals, feature, k=5):
    """
    Parameters
    ----------
    x: 1d array like
        Feature values for this instance.
    shap_vals: 1d array like
        Feature contributions to the prediction.
    feature: 1d array like
        Feature names.
    k: int (default=5)
        Only keep the top k features.

    Returns a list of (feature_name, feature_value, feature_shap) tuples.
    """
    assert len(x) == len(shap_vals) == len(feature)
    shap_sort_ndx = np.argsort(np.abs(shap_vals))[::-1]
    return list(zip(feature[shap_sort_ndx], x[shap_sort_ndx], shap_vals[shap_sort_ndx]))[:k]


def _plot_features(x_test, X_train, target_ndx, alt_ndx, feature, shap_vals, k=5):

    fig, axs = plt.subplots(1, k, figsize=(24, 6))
    axs = axs.flatten()

    x_test = x_test[0]
    shap_sort_ndx = np.argsort(np.abs(shap_vals))[::-1][:k]
    shap_sum = np.abs(shap_vals).sum()

    for i, ndx in enumerate(shap_sort_ndx):
        axs[i].axvline(x_test[ndx], linestyle='--', color='k')
        sns.distplot(X_train[target_ndx][:, ndx], ax=axs[i], norm_hist=True, label='excitatory', color='g')
        sns.distplot(X_train[alt_ndx][:, ndx], ax=axs[i], norm_hist=True, label='inhibitory', color='r')
        axs[i].set_title('{}: {:.1f}%'.format(feature[ndx], shap_vals[ndx] / shap_sum * -100), fontsize=24)

        if feature[ndx] == 'lstmwresampling':
            axs[i].set_xlim(-0.25, 0.75)

        if feature[ndx] == 'tallc':
            axs[i].set_xlim(0.65, 0.75)

        if feature[ndx] == 'contrast_enhance':
            axs[i].set_xlim(0, 1)

        axs[i].set_xlabel('value', fontsize=24)

        if i == 0:
            axs[i].set_ylabel('density', fontsize=24)

        axs[i].tick_params(axis='both', which='major', labelsize=22, pad=15)

    handles = axs[i].lines[1:]

    fig.legend(handles, ('excitatory samples', 'inhibitory samples'), loc='center', ncol=2,
               bbox_to_anchor=(0.5, 0.07), fontsize=24)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.35)
    plt.subplots_adjust(wspace=0.25)


def _shift_plot_right(ax, amt=0.02):
    """
    Shifts the subplot to the right by a specified amount.
    """
    box = ax.get_position()
    box.x0 += amt
    box.x1 += amt
    ax.set_position(box)


def _get_short_names(feature):
    replace = {}
    replace['p_fibberinh_1_0_mediforsystem'] = 'fibberinh'
    replace['p_kitwaredartmouthjpegdimples_0db8e4c_mediforsystem'] = 'jpegdimples'
    replace['dct03_a_baseline_ta1'] = 'dct03'
    replace['block02_baseline_ta1'] = 'block02'
    replace['p_ucrlstmwresamplingwcmm2_1_0_mediforsystem'] = 'lstmwresampling'
    replace['p_uscisigradbased02a_0_2a_mediforsystem'] = 'gradbased'
    replace['p_purdueta11adoublejpegdetection_2_0_mediforsystem'] = 'doublejpeg'
    replace['p_purdueta11acontrastenhancementdetection_1_0_mediforsystem'] = 'contrast_enhance'
    replace['p_sriprita1imgmdlprnubased_1_0_mediforsystem'] = 'prnubased'
    replace['p_ta11c_1_0_mediforsystem'] = 'tallc'

    feature = np.array([replace.get(f) if replace.get(f) is not None else f for f in feature])
    return feature


def _get_top_contributors(contributions, pct=0.5, positive=False):

    if positive and pct is None:
        return np.where(contributions > 0)[0]

    sort_ndx = np.argsort(np.abs(contributions))[::-1]
    contribution_sum = np.abs(contributions).sum()

    total = 0.0
    top_contributors = []

    for i in sort_ndx:

        if positive and contributions[i] < 0:
            continue

        total += abs(contributions[i])
        top_contributors.append(i)

        if total / contribution_sum >= pct:
            break

    return np.array(top_contributors)


def misclassification(model='lgb', encoding='leaf_output', dataset='nc17_mfc18', n_estimators=100, random_state=69,
                      topk_train=4, topk_test=1, data_dir='data', verbose=0, feature_length=20,
                      linear_model='lr', kernel='linear', topk_feature=5, true_label=False, alpha=0.5):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    data = data_util.get_data(dataset, random_state=random_state, data_dir=data_dir, return_feature=True)
    X_train, X_test, y_train, y_test, label, feature = data

    # shorten feature names
    feature = _get_short_names(feature)

    # train a tree ensemble
    tree = clf.fit(X_train, y_train)
    tree_yhat = model_util.performance(tree, X_train, y_train, X_test, y_test)

    # train an svm on learned representations from the tree ensemble
    explainer = TreeExplainer(tree, X_train, y_train, encoding=encoding, random_state=random_state,
                              dense_output=True, linear_model=linear_model, kernel=kernel,
                              use_predicted_labels=not true_label)

    if verbose > 0:
        print(explainer)

    shap_explainer = shap.TreeExplainer(tree)
    test_shap = shap_explainer.shap_values(X_test)
    train_shap = shap_explainer.shap_values(X_train)

    # extract predictions
    tree_yhat_train, tree_yhat_test = tree_yhat
    tree_pred_train = tree.predict(X_train)
    tree_pred_test = tree.predict(X_test)

    # get worst missed test indices
    test_dist = exp_util.instance_loss(tree.predict_proba(X_test), y_test)
    test_dist_ndx = np.argsort(test_dist)[::-1]
    test_dist = test_dist[test_dist_ndx]
    both_missed_test = test_dist_ndx

    test_dist_ndx1 = np.where((y_test == 1) & (tree.predict(X_test) == 0))[0]
    test_dist_ndx2 = np.where((y_test == 0) & (tree.predict(X_test) == 1))[0]

    if verbose > 0:
        print(test_dist_ndx1, test_dist_ndx1.shape)
        print(test_dist_ndx2, test_dist_ndx2.shape)

    # show explanations for missed instances
    test_str = '\ntest_{}\npredicted as {}, actual is {}'
    train_str = 'train_{} predicted as {}, actual is {}, contribution={:.3f}'

    # explain test instances
    for test_ndx in both_missed_test[:topk_test]:
        x_test = X_test[[test_ndx]]

        # find the most impactful features
        shap_list = _get_top_features(x_test[0], test_shap[test_ndx], feature, k=topk_feature)
        shap_sum = np.sum(np.abs(test_shap[test_ndx]))

        # find the most impactful training instances
        contributions = explainer.explain(x_test)[0]
        sort_ndx = np.argsort(np.abs(contributions))[::-1]
        contribution_sum = np.abs(contributions).sum()

        if verbose > 0:
            print(stats.describe(np.abs(contributions)))
            print(tree.predict_proba(x_test))

        pos_contributors = np.where(contributions > 0)[0]
        neg_contributors = np.where(contributions < 0)[0]

        # show stats for positive contributors
        n_contribs = len(pos_contributors)
        pct_contribs = n_contribs / len(X_train) * 100
        pct_contrib = np.abs(contributions[pos_contributors]).sum() / contribution_sum * 100
        contrib_str = 'Positive contributors, {:.1f}% of explanation: {}, {:.1f}% of training'
        print(contrib_str.format(pct_contrib, n_contribs, pct_contribs))

        # show stats for negative contributors
        n_contribs = len(neg_contributors)
        pct_contribs = n_contribs / len(X_train) * 100
        pct_contrib = np.abs(contributions[neg_contributors]).sum() / contribution_sum * 100
        contrib_str = 'Negative contributors, {:.1f}% of explanation: {}, {:.1f}% of training'
        print(contrib_str.format(pct_contrib, n_contribs, pct_contribs))

        # how often do the top contributors share the same predicted label
        # as the test instance's predicted label
        shared_ndx = np.where(tree_pred_train[pos_contributors] == tree_pred_test[test_ndx])[0]
        if verbose > 0:
            print('{} / {}'.format(len(shared_ndx), len(pos_contributors)))

        # plot feature histograms for specific train instances
        _plot_features(x_test, X_train, pos_contributors, neg_contributors, feature, test_shap[test_ndx],
                       k=topk_feature)
        plt.savefig(os.path.join('output', 'misclassification', 'histogram.pdf'), bbox_inches='tight')

        # display test instance
        test_instance_str = test_str.format(test_ndx, tree_pred_test[test_ndx], y_test[test_ndx])
        print(test_instance_str)
        for feature_name, feature_val, feature_shap in shap_list:
            print('\t{}: val={:.3f}, shap={:.3f}'.format(feature_name, feature_val, feature_shap / shap_sum))

        # display training instances
        for i, train_ndx in enumerate(sort_ndx[:topk_train]):

            # find the most impactful features
            shap_list = _get_top_features(X_train[train_ndx], train_shap[train_ndx], feature, k=topk_feature)
            shap_sum = np.sum(np.abs(train_shap[train_ndx]))

            # display train instance
            train_instance_str = train_str.format(train_ndx, tree_pred_train[train_ndx], y_train[train_ndx],
                                                  contributions[train_ndx] / contribution_sum)
            print(train_instance_str)
            for feature_name, feature_val, feature_shap in shap_list:
                print('\t{}: val={:.3f}, shap={:.3f}'.format(feature_name, feature_val, feature_shap / shap_sum))

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='nc17_mfc18', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--linear_model', type=str, default='svm', help='linear model to use.')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--kernel', type=str, default='linear', help='similarity kernel.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in the ensemble.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--topk_train', metavar='NUM', type=int, default=4, help='train instances to show.')
    parser.add_argument('--topk_test', metavar='NUM', type=int, default=1, help='missed test instances to show.')
    parser.add_argument('--topk_feature', metavar='NUM', type=int, default=5, help='features to show.')
    args = parser.parse_args()
    print(args)
    misclassification(model=args.model, encoding=args.encoding, dataset=args.dataset, n_estimators=args.n_estimators,
                      random_state=args.rs, topk_train=args.topk_train, topk_test=args.topk_test,
                      linear_model=args.linear_model, kernel=args.kernel, topk_feature=args.topk_feature)
