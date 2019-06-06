"""
Sample explanation for tree ensembles with SEXEE.
"""
import argparse
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sexee.explainer import TreeExplainer


def show_most_impactful_train_instances(impact):

    neg_inf_ndx = np.where(influence < 0)[0]
    neg_inf = influence[neg_inf_ndx]
    neg_inf_sv_ndx = self.svm_.support_[neg_inf_ndx]
    neg_inf_list = sorted(zip(neg_inf_sv_ndx, neg_inf), key=lambda tup: tup[1])

    pos_inf_ndx = np.where(influence > 0)[0]
    pos_inf = influence[pos_inf_ndx]
    pos_inf_sv_ndx = self.svm_.support_[pos_inf_ndx]
    pos_inf_list = sorted(zip(pos_inf_sv_ndx, pos_inf), key=lambda tup: tup[1], reverse=True)

    # show test instance
    if y is not None and ndx is not None:
        test_str = '\n\nTest [{}], distance to separator: {:.3f}, prediction: {}, actual: {}'
        print(test_str.format(ndx, prediction, prediction_label, y_train[ndx]))
    else:
        test_str = '\n\nTest [{}], distance to separator: {:.3f}, prediction: {}'
        print(test_str.format(ndx, prediction, prediction_label))

    # show most influential train instances
    train_str = 'Train [{}], impact: {:.3f}, similarity: {:.3f}, weight: {:.3f}, label: {}'

    print('\nPositive Train Instances')
    for train_ndx, inf in pos_inf_list[:topk]:
        train_coef = self.svm_.dual_coef_[0][np.where(self.svm_.support_ == train_ndx)[0][0]]
        print(train_str.format(train_ndx, inf, sim[train_ndx], train_coef, self.y_train[train_ndx]))

    print('\nNegative Train Instances')
    for train_ndx, inf in neg_inf_list[:topk]:
        train_coef = self.svm_.dual_coef_[0][np.where(self.svm_.support_ == train_ndx)[0][0]]
        print(train_str.format(train_ndx, inf, sim[train_ndx], train_coef, self.y_train[train_ndx]))


def main(args):

    # create model
    if args.model == 'lgb':
        clf = lgb.LGBMClassifier(random_state=args.rs, n_estimators=args.n_estimators)
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

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.rs, stratify=y)

    model = clf.fit(X_train, y_train)
    explainer = TreeExplainer(model, X_train, y_train, encoding=args.encoding, random_state=args.rs)

    imp_list = explainer.train_impact(X_test[0].reshape(1, -1))
    print(sorted(imp_list, key=lambda tup: abs(tup[1]), reverse=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='iris', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='tree_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=20, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--topk', metavar='NUM', type=int, default=5, help='Num of similar instances to display.')
    parser.add_argument('--plot_similarity', default=False, action='store_true', help='plot train similarities.')
    args = parser.parse_args()
    print(args)

    main(args)
