"""
This is a modified and simplified version of DShap
that only works for binary classification.
"""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


class DShap(object):

    def __init__(self,
                 model,
                 X_train,
                 y_train,
                 X_test,
                 y_test,
                 metric='accuracy',
                 tolerance=0.01,
                 max_iter=1000,
                 random_state=1):
        """
        Args:
            model: Trained model.
            X_train: Train covariates.
            y_train: Train labels.
            X_test: Test covariates.
            y_test: Test labels.
            metric: Evaluation metric.
            tolerance: Tolerance used to truncate TMC-Shapley.
            random_state: Random seed.
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.metric = metric
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_state = random_state

        # derived attributes
        self.sources_ = np.arange(X_train.shape[0])
        self.random_score_ = self.init_score()
        self.mean_score_ = self.compute_mean_score()
        self.mem_tmc_ = np.zeros((0, self.sources_.shape[0]))

    def tmc_shap(self, check_every=10, err_tol=0.1):
        """
        Runs TMC-Shapley algorithm.

        Args:
            check_every: No. iterations to run before checking convergence
            err_tol: Convergence tolerance.

        Returns:
            Average marginal contributions of each training instance.
            A positive number means the training instance increased the score,
            and a negative number means the training instance decreased the score.
        """

        # result container
        marginals_sum = np.zeros(self.X_train.shape[0])

        # run TMC-Shapley for a specified no. iterations
        for iteration in range(self.max_iter):

            # run one iteration of the TMC algorithm
            marginals = self.one_iteration()
            marginals_sum += marginals

            # history of marignals, shape=(no. iterations, no. samples)
            self.mem_tmc_ = np.vstack([self.mem_tmc_, marginals.reshape(1, -1)])

            # check if TMC-Shapley should finish early, only do this every so often
            if (iteration + 1) % check_every == 0:
                error = self.compute_error(self.mem_tmc_)
                print('[ITER {:,}], max. error: {:.3f}'.format(iteration + 1, error))

                # delete old marginals to avoid memory explosion
                if (iteration + 1) > 100:
                    self.mem_tmc_ = self.mem_tmc_[check_every:].copy()

                print(self.mem_tmc_.shape)

                if error < err_tol:
                    break

        if error > err_tol:
            print('Warning: did not converge, try increasing `max_iter`.')

        # compute average marginals
        marginals = marginals_sum / iteration

        return marginals

    # private
    def init_score(self):
        """
        Gives the value of an initial untrained model.
        """
        if self.metric == 'accuracy':
            hist = np.bincount(self.y_test).astype(float) / len(self.y_test)
            return np.max(hist)

        elif self.metric == 'auc':
            return 0.5

        elif self.metric == 'proba':
            return 0.5

        else:
            raise ValueError('unknown metric {}'.format(self.metric))

    def compute_score(self, model, X, y):
        """
        Computes the values of the given model.

        Args:
            model: The model to be evaluated.
        """
        if self.metric == 'accuracy':
            model_pred = model.predict(X)
            result = accuracy_score(model_pred, y)

        elif self.metric == 'auc':
            model_proba = model.predict_proba(X)[:, 1]
            result = roc_auc_score(model_proba, y)

        elif self.metric == 'proba':
            model_proba = model.predict_proba(X)
            if model_proba.shape[1] == 1:
                model_proba = model_proba[0]
            result = np.mean(model_proba)

        else:
            raise ValueError('Invalid metric!')

        return result

    def compute_mean_score(self, num_iter=100):
        """
        Computes the average performance and its error using bagging.
        """
        scores = []
        for _ in range(num_iter):

            # select a subset of bootstrapped samples
            bag_idxs = np.random.choice(self.y_test.shape[0], size=self.y_test.shape[0], replace=True)

            # score this subset
            score = self.compute_score(self.model,
                                       X=self.X_test[bag_idxs],
                                       y=self.y_test[bag_idxs])
            scores.append(score)

        return np.mean(scores)

    def compute_error(self, marginals, n_run=100):
        """
        Checks to see if the the marginals are converging.
        """

        # has not run long enough
        if marginals.shape[0] < n_run:
            return 1.0

        # add up all marginals using axis=0, then divide by their iteration
        all_vals = (np.cumsum(marginals, axis=0) / np.arange(1, len(marginals) + 1).reshape(-1, 1))[-n_run:]

        # diff. between last `n_run` runs and last run, divide by last run, average over all points
        errors = np.mean(np.abs(all_vals[-n_run:] - all_vals[-1:]) / (np.abs(all_vals[-1:]) + 1e-12), axis=1)

        # return max error from one of the points
        return np.max(errors)

    def one_iteration(self):
        """
        Runs one iteration of TMC-Shapley algorithm.
        """

        # shuffle the indices of the data
        idxs = np.random.permutation(self.sources_)

        # result container
        marginal_contribs = np.zeros(self.X_train.shape[0])

        # empty containers
        X_batch = np.zeros((0,) + tuple(self.X_train.shape[1:]))
        y_batch = np.zeros(0, int)

        # trackers
        truncation_counter = 0
        new_score = self.random_score_

        # perform for each data point
        for n, idx in enumerate(idxs):
            old_score = new_score

            # add sample to sample batch
            X_batch = np.vstack([X_batch, self.X_train[idx].reshape(1, -1)])
            y_batch = np.concatenate([y_batch, self.y_train[idx].reshape(1)])

            # train and re-evaluate
            model = clone(self.model)
            model = model.fit(X_batch, y_batch)
            new_score = self.compute_score(model,
                                           X=self.X_test,
                                           y=self.y_test)

            # add normalized contributions
            marginal_contribs[idx] = (new_score - old_score)

            # compute approximation quality
            distance_to_full_score = np.abs(new_score - self.mean_score_)
            if distance_to_full_score <= self.tolerance * self.mean_score_:
                truncation_counter += 1
                if truncation_counter > 5:
                    break

            # approximation is not converging, keep going
            else:
                truncation_counter = 0

        return marginal_contribs
