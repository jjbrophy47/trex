"""
Analyzes different feature representations from tree ensembles.
"""
import argparse
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, euclidean_distances
from sexee.explainer import TreeExtractor


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
    print(label)

    # train a tree ensemble
    model = clf.fit(X, y)
    extractor_path = TreeExtractor(model, encoding='tree_path')
    extractor_output = TreeExtractor(model, encoding='tree_output')

    X_dist_sim = euclidean_distances(X)
    print(X_dist_sim, X_dist_sim.shape)

    X_path = extractor_path.fit_transform(X)
    X_path_sim = linear_kernel(X_path)
    print(X_path_sim, X_path_sim.shape)

    X_out = extractor_output.fit_transform(X)
    print(X_out, X_out.shape)
    X_out_sim = rbf_kernel(X_out, gamma=10.0)
    print(X_out_sim, X_out_sim.shape)

    # # plot correlation between feature representation similarities
    # fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    # fig.suptitle('Feature Representation Similarity, Dataset: {}'.format(args.dataset))
    # axs[0].scatter(X_dist_sim.flatten(), X_path_sim.flatten())
    # axs[0].set_title('Original vs Tree Path')
    # axs[0].set_xlabel('original (euclidean)')
    # axs[0].set_ylabel('tree path (linear kernel)')
    # axs[1].scatter(X_dist_sim.flatten(), X_out_sim.flatten())
    # axs[1].set_title('Original vs Tree Output')
    # axs[1].set_xlabel('original (euclidean)')
    # axs[1].set_ylabel('tree output (rbf kernel)')
    # axs[2].scatter(X_path_sim.flatten(), X_out_sim.flatten())
    # axs[2].set_title('Tree Path vs Tree Output')
    # axs[2].set_xlabel('tree path (linear kernel)')
    # axs[2].set_ylabel('tree output (rbf kernel)')
    # plt.show()
    # plt.savefig('similarity.pdf', format='pdf', bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='iris', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='lgb', help='model to use.')
    parser.add_argument('--encoding', type=str, default='tree_path', help='type of encoding.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=20, help='number of trees in random forest.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--plot_similarity', default=False, action='store_true', help='plot train similarities.')
    args = parser.parse_args()
    print(args)
    main(args)
