"""
Sample explanation for tree ensembles with SEXEE.
"""
import argparse

from sexee.explainer import TreeExplainer
from util import model_util, data_util


def show_test_instance(test_ndx, svm_pred, pred_label, y_test=None, label=None):

    # show test instance
    if y_test is not None and label is not None:
        test_str = '\n\nTest [{}], distance to separator: {:.3f}, prediction: {}, actual: {}'
        print(test_str.format(test_ndx, svm_pred, label[pred_label], label[y_test[test_ndx]]))

    elif y_test is not None:
        test_str = '\n\nTest [{}], distance to separator: {:.3f}, prediction: {}, actual: {}'
        print(test_str.format(test_ndx, svm_pred, pred_label, y_test[test_ndx]))

    else:
        test_str = '\n\nTest [{}], distance to separator: {:.3f}, prediction: {}'
        print(test_str.format(test_ndx, svm_pred, pred_label))


def show_train_instances(impact_list, y_train, k=5, label=None):

    # show most influential train instances
    n_items = len(impact_list[0])

    if n_items == 2:
        train_str = 'Train [{}], impact: {:.3f}, label: {}'
    elif n_items == 4:
        train_str = 'Train [{}], impact: {:.3f}, similarity: {:.3f}, weight: {:.3f}, label: {}'
    else:
        exit('3 train impact items is ambiguous!')

    nonzero_sv = [items[0] for items in impact_list if abs(items[1]) > 0]
    print('\nSupport Vectors: {}'.format(len(impact_list)))
    print('Nonzero Support Vectors: {}'.format(len(nonzero_sv)))

    print('\nMost Impactful Train Instances')
    for items in impact_list[:k]:
        train_label = y_train[items[0]] if label is None else label[y_train[items[0]]]
        items += (train_label,)
        print(train_str.format(*items))


def show_fidelity(both_train, diff_train, y_train, both_test=None, diff_test=None, y_test=None):
    print('\nFidelity')

    n_both, n_diff, n_train = len(both_train), len(diff_train), len(y_train)
    print('train overlap: {} ({:.3f})'.format(n_both, n_both / n_train))
    print('train difference: {} ({:.3f})'.format(n_diff, n_diff / n_train))

    if both_test is not None and diff_test is not None and y_test is not None:
        n_both, n_diff, n_test = len(both_test), len(diff_test), len(y_test)
        print('test overlap: {} ({:.3f})'.format(n_both, n_both / n_test))
        print('test difference: {} ({:.3f})'.format(n_diff, n_diff / n_test))


def main(args):

    # get model and data
    clf = model_util.get_classifier(args.model)
    X_train, X_test, y_train, y_test, label = data_util.get_data(args.dataset, random_state=args.rs)

    # train a tree ensemble
    model = clf.fit(X_train, y_train)
    tree_yhat = model_util.performance(model, X_train, y_train, X_test, y_test)

    # train an svm on learned representations from the tree ensemble
    explainer = TreeExplainer(model, X_train, y_train, encoding=args.encoding, random_state=args.rs,
                              timeit=args.timeit)
    test_feature = explainer.extractor_.transform(X_test)
    svm_yhat = model_util.performance(explainer.get_svm(), explainer.train_feature_, y_train, test_feature, y_test)

    # extract predictions
    tree_yhat_train, tree_yhat_test = tree_yhat
    svm_yhat_train, svm_yhat_test = svm_yhat

    # test fidelity on train and test predictions
    both_train, diff_train = model_util.fidelity(tree_yhat_train, svm_yhat_train)
    both_test, diff_test = model_util.fidelity(tree_yhat_test, svm_yhat_test)
    show_fidelity(both_train, diff_train, y_train, both_test, diff_test, y_test)

    # test instances that tree and svm models missed
    both_missed_test = model_util.missed_instances(tree_yhat_test, svm_yhat_test, y_test)

    # show explanations for missed instances
    for test_ndx in both_missed_test[:args.topk_test]:
        impact_list, (svm_pred, pred_label) = explainer.train_impact(X_test[test_ndx].reshape(1, -1), pred_svm=True,
                                                                     similarity=True, weight=True)
        impact_list = sorted(impact_list, key=lambda tup: abs(tup[1]), reverse=True)
        show_test_instance(test_ndx, svm_pred, pred_label, y_test=y_test, label=label)
        show_train_instances(impact_list, y_train, k=args.topk_train, label=label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='iris', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='tree_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=20, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--topk_train', metavar='NUM', type=int, default=5, help='Train instances to show.')
    parser.add_argument('--topk_test', metavar='NUM', type=int, default=1, help='Missed test instances to show.')
    parser.add_argument('--timeit', action='store_true', default=False, help='Show timing info for explainer.')
    parser.add_argument('--sparse', action='store_true', default=False, help='Use sparse feature representations.')
    args = parser.parse_args()
    print(args)
    main(args)
