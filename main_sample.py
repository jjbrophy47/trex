"""
Sample explanation for tree ensembles with SEXEE.
"""
import argparse
import catboost
import lightgbm
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split

from sexee.explainer import TreeExplainer
from util import model_util


def show_test_instance(test_ndx, svm_pred, pred_label, y_test=None, label=None):

    print(pred_label)
    print(label)
    print(label[pred_label])

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

    print('\nMost Impactful Train Instances')
    for items in impact_list[:k]:
        train_label = y_train[items[0]] if label is None else label[y_train[items[0]]]
        items += (train_label,)
        print(train_str.format(*items))


def main(args):

    # create model
    if args.model == 'lgb':
        clf = lightgbm.LGBMClassifier(random_state=args.rs, n_estimators=args.n_estimators)
    elif args.model == 'cb':
        clf = catboost.CatBoostClassifier(random_state=args.rs, n_estimators=args.n_estimators)
    elif args.model == 'rf':
        clf = RandomForestClassifier(random_state=args.rs, n_estimators=args.n_estimators)

    # load dataset
    if args.dataset == 'iris':
        data = load_iris()
    elif args.dataset == 'breast':
        data = load_breast_cancer()
    elif args.dataset == 'wine':
        data = load_wine()

    X = data['data']
    y = data['target']
    label = data['target_names']
    print(label)

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.rs, stratify=y)

    # train a tree ensemble
    model = clf.fit(X_train, y_train)
    model_util.performance(model, X_train, y_train, X_test, y_test)

    # train an svm on learned representations from the tree ensemble
    explainer = TreeExplainer(model, X_train, y_train, encoding=args.encoding, random_state=args.rs,
                              use_predicted_labels=True)
    test_feature = explainer.extractor_.transform(X_test)
    model_util.performance(explainer.get_svm(), explainer.train_feature_, y_train, test_feature, y_test)

    test_ndx = 2
    impact_list, (svm_pred, pred_label) = explainer.train_impact(X_test[test_ndx].reshape(1, -1), pred_svm=True,
                                                                 similarity=True, weight=True)
    impact_list = sorted(impact_list, key=lambda tup: abs(tup[1]), reverse=True)
    show_test_instance(test_ndx, svm_pred, pred_label, y_test=y_test, label=label)
    show_train_instances(impact_list, y_train, k=args.topk, label=label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='iris', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='tree_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=20, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--topk', metavar='NUM', type=int, default=5, help='Num of similar instances to display.')
    args = parser.parse_args()
    print(args)
    main(args)
