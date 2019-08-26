"""
Explanation of missclassified test instances for the NC17_EvalPart1 (train) and
MFC18_EvalPart1 (test) dataset using SEXEE and SHAP.
"""
import os
import sys
import argparse
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../')  # for utility
sys.path.insert(0, here + '/../../')  # for libliner; TODO: remove this dependency

import shap
import numpy as np
import matplotlib.pyplot as plt

from sexee.explainer import TreeExplainer
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


def _plot_instance(instance_str, shap_list, shap_sum=None, ax=None):
    """
    Plot the the most impactful features for the given instance.
    """

    shap_list = shap_list[::-1]
    feature_name, feature_val, feature_shap = zip(*shap_list)
    index = np.arange(len(feature_name))

    pos_ndx = np.where(np.array(feature_shap) > 0)[0]

    bar_vals = np.abs(feature_shap)
    if shap_sum is not None:
        bar_vals /= shap_sum

    barlist = ax.barh(index, bar_vals, color='#ADFF2F')
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_title(instance_str, fontsize=22)
    ax.set_ylabel('feature')
    ax.set_xlabel('impact on prediction', fontsize=22)
    ax.tick_params(axis='x', which='major', labelsize=24)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, (name, val) in enumerate(zip(feature_name, feature_val)):
        ax.text(0, i - 0.1, ' {} = {:.3f}'.format(name, val), color='k', fontsize=22)

    for ndx in pos_ndx:
        barlist[ndx].set_color('#FA8072')  # light red


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
    replace['p_purdueta11acontrastenhancementdetection_1_0_mediforsystem'] = 'contrastenhance'
    replace['p_sriprita1imgmdlprnubased_1_0_mediforsystem'] = 'prnubased'

    feature = np.array([replace.get(f) if replace.get(f) is not None else f for f in feature])
    return feature


def misclassification(model='lgb', encoding='leaf_output', dataset='nc17_mfc18', n_estimators=100, random_state=69,
                      topk_train=4, topk_test=1, data_dir='data', verbose=0, feature_length=20,
                      linear_model='lr', kernel='linear', topk_feature=5, true_label=False):

    # get model and data
    clf = model_util.get_classifier(model, n_estimators=n_estimators, random_state=random_state)
    data = data_util.get_data(dataset, random_state=random_state, return_feature=True, data_dir=data_dir)
    X_train, X_test, y_train, y_test, label, feature = data

    # shorten feature names
    feature = _get_short_names(feature)

    remove_ndx = np.array([2, 4, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 29, 31, 32, 33, 34])

    X_train = np.delete(X_train, remove_ndx, axis=1)
    X_test = np.delete(X_test, remove_ndx, axis=1)
    feature = np.delete(feature, remove_ndx)

    # remove_ndx = np.where(feature == 'lstmwresampling')[0]
    # X_train = np.delete(X_train, remove_ndx, axis=1)
    # X_test = np.delete(X_test, remove_ndx, axis=1)
    # feature = np.delete(feature, remove_ndx)

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
    print(test_dist_ndx1, test_dist_ndx1.shape)

    test_dist_ndx2 = np.where((y_test == 0) & (tree.predict(X_test) == 1))[0]
    print(test_dist_ndx2, test_dist_ndx2.shape)

    # lstm_ndx = np.where(feature == 'lstmwresampling')[0]
    # X_test[test_dist_ndx1, lstm_ndx] = 1.0
    # X_test[test_dist_ndx2, lstm_ndx] = 0.0
    # tree_yhat = model_util.performance(tree, X_train, y_train, X_test, y_test)

    # # get worst missed test indices
    # test_dist = exp_util.instance_loss(tree.predict_proba(X_test), y_test)
    # test_dist_ndx = np.argsort(test_dist)[::-1]
    # test_dist = test_dist[test_dist_ndx]
    # both_missed_test = test_dist_ndx

    # target_ndx = test_dist_ndx[np.where(test_dist >= 0)]
    # print(target_ndx, target_ndx.shape)
    # x_target = X_test[target_ndx][:, 16]
    # print(len(np.where(x_target == -1)[0]), len(x_target))
    # exit(0)

    # show explanations for missed instances
    test_str = '\ntest_{}\npredicted as {}, actual is {}'
    train_str = 'train_{} predicted as {}, actual is {}, contribution={:.3f}'
    train_str2 = 'train_{}\npredicted as {}, actual is {}'

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

        # display test instance
        test_instance_str = test_str.format(test_ndx, tree_pred_test[test_ndx], y_test[test_ndx])
        print(test_instance_str)
        for feature_name, feature_val, feature_shap in shap_list:
            print('\t{}: val={:.3f}, shap={:.3f}'.format(feature_name, feature_val, feature_shap / shap_sum))

        # fig, axs = plt.subplots(1, 5, figsize=(30, 4))
        # _plot_instance(test_instance_str, shap_list, shap_sum=shap_sum, ax=axs[0])

        # display training instances
        for i, train_ndx in enumerate(sort_ndx[:topk_train]):

            # find the most impactful features
            shap_list = _get_top_features(X_train[train_ndx], train_shap[train_ndx], feature, k=topk_feature)
            shap_sum = np.sum(np.abs(train_shap[train_ndx]))

            # display train instance
            train_instance_str = train_str.format(train_ndx, tree_pred_train[train_ndx], y_train[train_ndx],
                                                  contributions[train_ndx] / contribution_sum)
            train_instance_str2 = train_str2.format(train_ndx, tree_pred_train[train_ndx], y_train[train_ndx])
            print(train_instance_str)
            for feature_name, feature_val, feature_shap in shap_list:
                print('\t{}: val={:.3f}, shap={:.3f}'.format(feature_name, feature_val, feature_shap / shap_sum))

            # _plot_instance(train_instance_str2, shap_list, shap_sum=shap_sum, ax=axs[i + 1])

        # # adjust spacing of the subplots
        # plt.subplots_adjust(wspace=0)
        # _shift_plot_right(axs[0], amt=-0.02)

        # out_dir = os.path.join('output', 'misclassification')
        # os.makedirs(out_dir, exist_ok=True)
        # plt.savefig(os.path.join(out_dir, 'misclassification.pdf'), bbox_inches='tight')


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
