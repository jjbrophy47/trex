"""
SVM and kernel kernel logistic regression models.
"""
import os

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import linear_kernel

from . import liblinear_util


class SVM(BaseEstimator, ClassifierMixin):
    """
    Multiclass wrapper around sklearn's SVC. This is to unify the API
    for the SVM and Kernel LR models.
    If multiclass, uses a one-vs-rest strategy and fits a
    BinaryKernelLogisticRegression classifier for each class.
    """

    def __init__(self, C=1.0, pred_size=1000, temp_dir='.temp_svm'):
        """
        Parameters
        ----------
        C: float (default=1.0)
            Regularization parameter.
        pred_size: int (default=1000)
            Max number of instancs to predict at one time. A higher number can
            be faster, but requires more memory to create the similarity matrix.
        temp_dir: str (default='.temp_svm')
            Temporary directory.
        """
        self.C = C
        self.pred_size = pred_size
        self.temp_dir = temp_dir

    def fit(self, X, y):
        self.X_train_ = X
        self.n_features_ = X.shape[1]
        estimator = BinarySVM(C=self.C,
                              pred_size=self.pred_size,
                              temp_dir=self.temp_dir)
        self.ovr_ = OneVsRestClassifier(estimator).fit(X, y)
        return self

    def decision_function(self, X):
        return self.ovr_.decision_function(X)

    def predict_proba(self, X):
        return self.ovr_.predict_proba(X)

    def predict(self, X):
        return self.ovr_.predict(X)

    def similarity(self, X, train_indices=None):
        X_train = self.X_train_[train_indices] if train_indices is not None else self.X_train_
        return linear_kernel(X, X_train)

    def get_weight(self):
        """
        Return a matrix of train instance weights.
            If binary, the array has shape (1, n_train_samples).
            If multiclass, the array has shape (n_classes, n_train_samples).
        """
        return np.vstack([estimator.get_weight() for estimator in self.ovr_.estimators_])

    def explain(self, X, y=None):
        """
        Return a sparse matrix of train instance contributions to X. A positive score
        means the training instance contributed towards the predicted label.

        Parameters
        ----------
        X : 2d array-like
            Instances to explain.
        y : 1d array-like
            If not None, a positive score means the training instance contributed
            to the label in y. Must be the same length as X.

        Returns a sparse matrix of shape (len(X), n_train_samples).
        """
        if y is None:
            y = self.predict(X)
        assert len(y) == len(X)

        # handle multiclass and binary slightly differently
        if len(self.ovr_.estimators_) > 1:
            result = np.vstack([self.ovr_.estimators_[y[i]].explain(X[[i]]) for i in range(len(X))])
        else:
            result = np.vstack([self.ovr_.estimators_[0].explain(X[[i]]) for i in range(len(X))])
            result[np.where(y == 0)] *= -1

        return result

    # private
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class BinarySVM(BaseEstimator, ClassifierMixin):
    """
    Wrapper around liblinear. Solves the l2 regularized l2 loss (squared hinge)
    support vector classifier dual problem using a linear kernel.
    Solver number 1: https://github.com/cjlin1/liblinear
    This is equivalent to sklearn's LinearSVC.
    Reference: https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf
    """

    def __init__(self, C=1.0, pred_size=1000, temp_dir='.temp_svm'):
        """
        Parameters
        ----------
        C: float (default=1.0)
            Regularization parameter.
        pred_size: int (default=1000)
            Max number of instancs to predict at one time. A higher number can
            be faster, but requires more memory to create the similarity matrix.
        temp_dir: str (default='.temp_svm')
            Temporary directory.
        """
        self.C = C
        self.pred_size = pred_size
        self.temp_dir = temp_dir

    def fit(self, X, y, n_check=10):

        # store training instances for later use
        self.X_train_ = X
        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2

        # remove any previously stored models
        os.makedirs(self.temp_dir, exist_ok=True)

        # setup path names
        train_data_path = os.path.join(self.temp_dir, 'train_data')
        model_path = os.path.join(self.temp_dir, 'model')
        prediction_path = os.path.join(self.temp_dir, 'prediction')

        # train the model using liblinear
        y_liblinear = np.where(y == 0, -1, 1)  # liblinear works better with -1 instead of 0
        liblinear_util.create_data_file(X, y_liblinear, train_data_path)
        liblinear_util.train_linear_svc(train_data_path, model_path, C=self.C)
        self.coef_ = liblinear_util.parse_model_file(model_path)

        # make sure our decomposition is making the same predictions as liblinear
        liblinear_util.predict_linear_svc(train_data_path, model_path, prediction_path)
        pred_label = liblinear_util.parse_linear_svc_predictions(prediction_path, minus_to_zeros=True)

        if not np.all(pred_label[:n_check] == self.predict(X[:n_check])):
            print('SVM PREDICTIONS NOT ALL CLOSE!')
            print(pred_label[:n_check])
            print(self.predict(X[:n_check]))

        return self

    def decision_function(self, X):
        """
        Returns a 1d array of decision values of size=len(X).
        """
        assert X.ndim == 2

        decisions = []
        for i in range(0, len(X), self.pred_size):
            X_sim = linear_kernel(X[i: i + self.pred_size], self.X_train_)
            decisions.append(np.sum(X_sim * self.coef_, axis=1))

        decision = np.concatenate(decisions)
        return decision

    def predict_proba(self, X):
        """
        Returns a 2d array of probabilities of shape (len(X), n_classes).
        """
        assert X.ndim == 2
        a = self._sigmoid(self.decision_function(X)).reshape(-1, 1)
        return np.hstack([1 - a, a])

    def predict(self, X):
        """
        Returns a 1d array of predicted labels of size=len(X).
        """
        pred_label = np.where(self.decision_function(X) >= 0, 1, 0)
        return pred_label

    def get_weight(self):
        """
        Return a sparse array of train instance weights with shape (1, n_train_samples).
        """
        return self.coef_.copy()

    def explain(self, x):
        """
        Return a sparse matrix of the impact of the training instances on x.
        The resulting array is of shape (1, n_train_samples).
        """
        assert x.shape == (1, self.X_train_.shape[1])
        x_sim = linear_kernel(x, self.X_train_)
        impact = x_sim * self.coef_
        return impact

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


class KernelLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Wrapper around liblinear. Solves the l2 logistic regression dual problem
    using a linear kernel.
    Solver number 7: https://github.com/cjlin1/liblinear
    Reference: https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf
    If multiclass, uses a one-vs-rest strategy and fits a
    BinaryKernelLogisticRegression classifier for each class.
    """

    def __init__(self, C=1.0, pred_size=1000, temp_dir='.temp_klr'):
        """
        Parameters
        ----------
        C: float (default=1.0)
            Regularization parameter, where 0 <= alpha_i <= C.
        pred_size: int (default=1000)
            Max number of instancs to predict at one time. A higher number can
            be faster, but requires more memory to create the similarity matrix.
        temp_dir: str (default='.temp_klr')
            Temporary directory.
        """
        self.C = C
        self.pred_size = pred_size
        self.temp_dir = temp_dir

    def fit(self, X, y):
        self.X_train_ = X
        self.n_features_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))
        estimator = BinaryKernelLogisticRegression(C=self.C,
                                                   pred_size=self.pred_size,
                                                   temp_dir=self.temp_dir)
        self.ovr_ = OneVsRestClassifier(estimator).fit(X, y)
        self.coef_ = np.vstack([estimator.coef_ for estimator in self.ovr_.estimators_])
        return self

    def predict_proba(self, X):
        return self.ovr_.predict_proba(X)

    def predict(self, X):
        return self.ovr_.predict(X)

    def similarity(self, X, train_indices=None):
        X_train = self.X_train_[train_indices] if train_indices is not None else self.X_train_
        return linear_kernel(X, X_train)

    def get_weight(self):
        return np.vstack([estimator.get_weight() for estimator in self.ovr_.estimators_])

    def explain(self, X, y=None):
        """
        Return an array of train instance contributions to X. A positive score
        means the training instance contributed towards the predicted label.

        Parameters
        ----------
        X : 2d array-like
            Instances to explain.
        y : 1d array-like
            If not None, a positive score means the training instance contributed
            to the label in y. Must be the same length as X.

        Returns a sparse matrix of shape (len(X), n_train_samples).
        """
        if y is None:
            y = self.predict(X)
        assert len(y) == len(X)

        # handle multiclass and binary slightly differently
        if len(self.ovr_.estimators_) > 1:
            result = np.vstack([self.ovr_.estimators_[y[i]].explain(X[[i]]) for i in range(len(X))])
        else:
            result = np.vstack([self.ovr_.estimators_[0].explain(X[[i]]) for i in range(len(X))])
            result[np.where(y == 0)] *= -1

        return result


class BinaryKernelLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Wrapper around liblinear. Solves the l2 logistic regression dual problem using a linear kernel.
    Reference: https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf
    """

    def __init__(self, C=1.0, pred_size=1000, temp_dir='.temp_klr'):
        """
        Parameters
        ----------
        C: float (default=1.0)
            Regularization parameter, where 0 <= alpha_i <= C.
        pred_size: int (default=1000)
            Max number of instancs to predict at one time. A higher number can
            be faster, but requires more memory to create the similarity matrix.
        temp_dir: str (default='.temp_klr')
            Temporary directory for storing liblinear models and prediction files.
        """
        self.C = C
        self.pred_size = pred_size
        self.temp_dir = temp_dir

    def fit(self, X, y, n_check=10, atol=1e-4):

        # store training instances for later use
        self.X_train_ = X
        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2

        # remove any previously stored models
        os.makedirs(self.temp_dir, exist_ok=True)

        # setup path names
        train_data_path = os.path.join(self.temp_dir, 'train_data')
        model_path = os.path.join(self.temp_dir, 'model')
        prediction_path = os.path.join(self.temp_dir, 'prediction')

        # train the model using liblinear
        y_liblinear = np.where(y == 0, -1, 1)  # liblinear works better with -1 instead of 0
        liblinear_util.create_data_file(X, y_liblinear, train_data_path)
        liblinear_util.train_lr(train_data_path, model_path, C=self.C)
        self.coef_ = liblinear_util.parse_model_file(model_path)

        # make sure our decomposition is making the same predictions as liblinear
        liblinear_util.predict_lr(train_data_path, model_path, prediction_path)
        pred_label, pred_proba = liblinear_util.parse_lr_predictions(prediction_path, minus_to_zeros=True)

        if not np.allclose(pred_proba[:n_check][:, 1], self.predict_proba(X[:n_check])[:, 1], atol=atol):
            print('KLR PREDICTIONS NOT ALL CLOSE!, ATOL={}'.format(atol))
            print(pred_proba[:n_check][:, 1])
            print(self.predict_proba(X[:n_check][:, 1]))

        return self

    def predict_proba(self, X):
        """
        Returns a 2d array of probabilities of shape (len(X), n_classes).
        """
        assert X.ndim == 2

        pos_probas = []
        for i in range(0, len(X), self.pred_size):
            X_sim = linear_kernel(X[i: i + self.pred_size], self.X_train_)
            pos_probas.append(self._sigmoid(np.sum(X_sim * self.coef_, axis=1)))
        pos_proba = np.concatenate(pos_probas).reshape(-1, 1)
        proba = np.hstack([1 - pos_proba, pos_proba])
        return proba

    def predict(self, X):
        """
        Returns a 1d array of predicted labels of size=len(X).
        """
        pred_label = np.argmax(self.predict_proba(X), axis=1)
        return pred_label

    def get_weight(self):
        """
        Return a 1d array of train instance weights.
        """
        return self.coef_.copy()

    def explain(self, x):
        """
        Return a 2d array of train instance impacts of shape (1, n_train_samples).
        """
        assert x.shape == (1, self.X_train_.shape[1])
        x_sim = linear_kernel(x, self.X_train_)
        impact = x_sim * self.coef_
        return impact

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
