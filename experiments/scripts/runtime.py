"""
Experiment: Compare runtimes for explaining a single test instance for different methods.
"""
import time
import argparse
import os
import sys
import signal
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')  # for influence_boosting
sys.path.insert(0, here + '/../')  # for utility

import numpy as np
from sklearn.base import clone
from maple import MAPLE

import trex
from utility import model_util, data_util, exp_util

ONE_DAY = 86400  # number of seconds in a day
maple_limit_reached = False


class timeout:
    """
    Timeout class to throw a TimeoutError if a piece of code runs for too long.
    """

    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def _our_method(test_ndx, X_test, model, X_train, y_train, encoding='leaf_output', linear_model='svm', C=0.1,
                kernel='rbf', random_state=69):
    """Explains the predictions of each test instance."""

    start = time.time()
    explainer = trex.TreeExplainer(model, X_train, y_train, encoding=encoding, random_state=random_state,
                                   linear_model=linear_model, kernel=kernel, C=C)
    fine_tune = time.time() - start

    start = time.time()
    explainer.explain(X_test[test_ndx].reshape(1, -1))
    test_time = time.time() - start

    return fine_tune, test_time


def _influence_method(model, test_ndx, X_train, y_train, X_test, y_test, inf_k):
    """
    Computes the influence on each test instance if train instance i were upweighted/removed.
    This uses the fastleafinfluence method by Sharchilev et al.
    """

    start = time.time()
    leaf_influence = exp_util.get_influence_explainer(model, X_train, y_train, inf_k)
    fine_tune = time.time() - start

    start = time.time()
    exp_util.influence_explain_instance(leaf_influence, test_ndx, X_train, X_test, y_test)
    test_time = time.time() - start

    return fine_tune, test_time


def _maple_method(model, test_ndx, X_train, y_train, X_test, y_test, dstump=False):
    """
    Produces a train weight distribution for a single test instance.
    """

    with timeout(seconds=ONE_DAY):
        try:
            start = time.time()
            maple = MAPLE.MAPLE(X_train, y_train, X_train, y_train, dstump=dstump)
            fine_tune = time.time() - start
        except:
            print('maple fine-tuning exceeded 24h!')
            global maple_limit_reached
            maple_limit_reached = True
            return None, None

    start = time.time()
    maple.explain(X_test[test_ndx]) if dstump else maple.get_weights(X_test[test_ndx])
    test_time = time.time() - start

    return fine_tune, test_time


def runtime(model_type='cb', linear_model='lr', kernel='linear', encoding='leaf_output', dataset='iris',
            n_estimators=100, max_depth=None, C=0.1, random_state=69, inf_k=None, repeats=10, dstump=False,
            true_label=False, maple=False, data_dir='data', out_dir='output/runtime', save_results=False,
            start_pct=10):
    """
    Main method that trains a tree ensemble, then compares the runtime of different methods to explain
    a random subset of test instances.
    """

    settings = '{}_{}'.format(linear_model, kernel)
    settings += '_true_label' if true_label else ''

    for m in range(start_pct, 110, 10):
        print('\n{}% of the training data:'.format(m))

        our_fine_tune, our_test_time = [], []
        inf_fine_tune, inf_test_time = [], []
        maple_fine_tune, maple_test_time = [], []

        for i in range(repeats):
            print('\nrun {}'.format(i))
            random_state += 10

            # get model and data
            clf = model_util.get_classifier(model_type, n_estimators=n_estimators, max_depth=max_depth,
                                            random_state=random_state)
            X_train, X_test, y_train, y_test, label = data_util.get_data(dataset, random_state=random_state,
                                                                         data_dir=data_dir)

            n_samples = int(X_train.shape[0] * (m / 100))
            X_train, y_train = X_train[:n_samples], y_train[:n_samples]
            X_test, y_test = X_test[:n_samples], y_test[:n_samples]

            print('train instances: {}'.format(len(X_train)))
            print('num features: {}'.format(X_train.shape[1]))

            # train a tree ensemble
            model = clone(clf).fit(X_train, y_train)
            model_util.performance(model, X_test=X_test, y_test=y_test)

            # randomly pick test instances to explain
            np.random.seed(random_state)
            test_ndx = np.random.choice(len(y_test), size=1, replace=False)

            # train on predicted labels (ours and maple methods only)
            train_label = y_train if true_label else model.predict(X_train)

            # our method
            print('ours...')
            fine_tune, test_time = _our_method(test_ndx, X_test, model, X_train, train_label, encoding=encoding, C=C,
                                               linear_model=linear_model, kernel=kernel, random_state=random_state)
            print('fine tune: {:.3f}s'.format(fine_tune))
            print('test time: {:.3f}s'.format(test_time))
            our_fine_tune.append(fine_tune)
            our_test_time.append(test_time)

            # influence method
            if model_type == 'cb' and inf_k is not None:
                print('leafinfluence...')
                fine_tune, test_time = _influence_method(model, test_ndx, X_train, y_train, X_test, y_test, inf_k)
                print('fine tune: {:.3f}s'.format(fine_tune))
                print('test time: {:.3f}s'.format(test_time))
                inf_fine_tune.append(fine_tune)
                inf_test_time.append(test_time)

            if maple and not maple_limit_reached:
                print('maple...')
                fine_tune, test_time = _maple_method(model, test_ndx, X_train, train_label, X_test, y_test,
                                                     dstump=dstump)
                if fine_tune is not None and test_time is not None:
                    print('fine tune: {:.3f}s'.format(fine_tune))
                    print('test time: {:.3f}s'.format(test_time))
                    maple_fine_tune.append(fine_tune)
                    maple_test_time.append(test_time)

        # display results
        our_fine_tune = np.array(our_fine_tune)
        our_test_time = np.array(our_test_time)
        print('\nour')
        print('fine tuning: {:.3f}s +/- {:.3f}s'.format(our_fine_tune.mean(), our_fine_tune.std()))
        print('test time: {:.3f}s +/- {:.3f}s'.format(our_test_time.mean(), our_test_time.std()))

        if model_type == 'cb' and inf_k is not None:
            inf_fine_tune = np.array(inf_fine_tune)
            inf_test_time = np.array(inf_test_time)
            print('\nleafinfluence')
            print('fine tuning: {:.3f}s +/- {:.3f}s'.format(inf_fine_tune.mean(), inf_fine_tune.std()))
            print('test time: {:.3f}s +/- {:.3f}s'.format(inf_test_time.mean(), inf_test_time.std()))

        if maple and not maple_limit_reached:
            maple_fine_tune = np.array(maple_fine_tune)
            maple_test_time = np.array(maple_test_time)
            print('\nmaple')
            print('fine tuning: {:.3f}s +/- {:.3f}s'.format(maple_fine_tune.mean(), maple_fine_tune.std()))
            print('test time: {:.3f}s +/- {:.3f}s'.format(maple_test_time.mean(), maple_test_time.std()))

        # save results
        if save_results:
            exp_dir = os.path.join(out_dir, dataset, '{}_percent'.format(m))
            os.makedirs(exp_dir, exist_ok=True)

            # ours
            np.save(os.path.join(exp_dir, 'ours_{}_fine_tune.npy'.format(settings)), our_fine_tune)
            np.save(os.path.join(exp_dir, 'ours_{}_test_time.npy'.format(settings)), our_test_time)

            # leafinfluence
            if model_type == 'cb' and inf_k is not None:
                np.save(os.path.join(exp_dir, 'influence_fine_tune.npy'), inf_fine_tune)
                np.save(os.path.join(exp_dir, 'influence_test_time.npy'), inf_test_time)

            # MAPLE
            if maple:
                maple_settings = '' if not dstump else 'dstump_'
                np.save(os.path.join(exp_dir, 'maple_{}fine_tune.npy'.format(maple_settings)), maple_fine_tune)
                np.save(os.path.join(exp_dir, 'maple_{}test_time.npy'.format(maple_settings)), maple_test_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature representation extractions for tree ensembles',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='adult', help='dataset to explain.')
    parser.add_argument('--model', type=str, default='cb', help='model to use.')
    parser.add_argument('--linear_model', type=str, default='lr', help='linear model to use.')
    parser.add_argument('--encoding', type=str, default='leaf_output', help='type of encoding.')
    parser.add_argument('--kernel', type=str, default='linear', help='Similarity kernel.')
    parser.add_argument('--n_estimators', metavar='N', type=int, default=100, help='number of trees in tree ensemble.')
    parser.add_argument('--max_depth', type=int, default=None, help='maximum depth in tree ensemble.')
    parser.add_argument('--C', type=float, default=0.1, help='kernel model penalty parameter.')
    parser.add_argument('--rs', metavar='RANDOM_STATE', type=int, default=69, help='for reproducibility.')
    parser.add_argument('--inf_k', default=None, type=int, help='Number of leaves for leafinfluence.')
    parser.add_argument('--maple', action='store_true', help='Run experiment using MAPLE.')
    parser.add_argument('--dstump', action='store_true', help='Enable DSTUMP with Maple.')
    parser.add_argument('--start_pct', default=10, type=int, help='Percentage of training data to start with.')
    parser.add_argument('--repeats', default=5, type=int, help='Number of times to repeat the experiment.')
    parser.add_argument('--save_results', action='store_true', default=False, help='Save cleaning results.')
    parser.add_argument('--true_label', action='store_true', help='Train explainers on true labels.')
    args = parser.parse_args()
    print(args)
    runtime(model_type=args.model, linear_model=args.linear_model, encoding=args.encoding, kernel=args.kernel,
            dataset=args.dataset, n_estimators=args.n_estimators, random_state=args.rs, inf_k=args.inf_k,
            repeats=args.repeats, true_label=args.true_label, maple=args.maple, max_depth=args.max_depth, C=args.C,
            save_results=args.save_results, dstump=args.dstump)
