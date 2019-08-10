"""
SVM and kernel kernel logistic regression models.
"""
import os
import shutil

import numpy as np
from scipy import sparse as sps
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel, sigmoid_kernel
from sklearn.svm import SVC

from . import liblinear_util


class SVM(BaseEstimator, ClassifierMixin):
    """
    Multiclass wrapper around sklearn's SVC. This is to unify the API for the SVM and Kernel LR models.
    If multiclass, uses a one-vs-rest strategy and fits a BinaryKernelLogisticRegression classifier for each class.
    """

    def __init__(self, C=1.0, kernel='linear', gamma=None, coef0=0.0, degree=3, random_state=None):
        """
        Parameters
        ----------
        C: float (default=1.0)
            Regularization parameter, where 0 <= alpha_i <= C.
        kernel: str (default='linear')
            Type of kernel to use. Also 'rbf', 'poly', and 'sigmoid'.
        gamma: float (default=None)
            Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
            If None, defaults to 1 / n_features.
        coef0: float (default=0.0)
            Independent term in 'poly' and 'sigmoid'.
        degree: int (default=3)
            Degree of the 'poly' kernel.
        random_state: int (default=None)
            Number for reproducibility.
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.random_state = random_state

    def fit(self, X, y):
        self.X_train_ = X
        self.n_features_ = X.shape[1]
        self._create_kernel_callable()
        estimator = BinarySVM(kernel=self.kernel_func_, random_state=self.random_state)
        self.ovr_ = OneVsRestClassifier(estimator).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.ovr_.predict_proba(X)

    def predict(self, X):
        return self.ovr_.predict(X)

    def similarity(self, X, train_indices=None):
        X_train = self.X_train_[train_indices] if train_indices is not None else self.X_train_
        return self.kernel_func_(X, X_train)

    def get_weight(self):
        """
        Return a sparse matrix of train instance weights.
            If binary, the array has shape (1, n_train_samples).
            If multiclass, the array has shape (n_classes, n_train_samples).
        """
        return sps.vstack([estimator.get_weight() for estimator in self.ovr_.estimators_])

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

        return sps.vstack([self.ovr_.estimators_[y[i]].explain(x.reshape(1, -1)) for i, x in enumerate(X)])

    def _create_kernel_callable(self):
        assert self.kernel in ['rbf', 'poly', 'sigmoid', 'linear']

        if self.kernel == 'rbf':
            self.gamma_ = 1.0 / self.n_features_ if self.gamma is None else self.gamma
            self.kernel_func_ = lambda X1, X2: rbf_kernel(X1, X2, gamma=self.gamma_)
        elif self.kernel == 'poly':
            self.gamma_ = 1.0 / self.n_features_ if self.gamma is None else self.gamma
            self.kernel_func_ = lambda X1, X2: polynomial_kernel(X1, X2, degree=self.degree, gamma=self.gamma_)
        elif self.kernel == 'sigmoid':
            self.gamma_ = 1.0 / self.n_features_ if self.gamma is None else self.gamma
            self.kernel_func_ = lambda X1, X2: sigmoid_kernel(X1, X2, degree=self.degree, gamma=self.gamma_)
        elif self.kernel == 'linear':
            self.kernel_func_ = lambda X1, X2: linear_kernel(X1, X2)


class BinarySVM(BaseEstimator, ClassifierMixin):
    """
    Wrapper around sklearn's SVC. This is to unify the API for the SVM and Kernel LR models.
    """

    def __init__(self, C=1.0, kernel=linear_kernel, random_state=None):
        """
        Parameters
        ----------
        C: float (default=1.0)
            Regularization parameter, where 0 <= alpha_i <= C.
        kernel: str (default='linear')
            Type of kernel to use. Also 'rbf', 'poly', and 'sigmoid'.
        random_state: int (default=None)
            Number for reproducibility.
        """
        self.C = C
        self.kernel = kernel
        self.random_state = random_state
        assert callable(self.kernel)

    def fit(self, X, y):

        # store training instances for later use
        self.X_train_ = X
        self.n_features_ = X.shape[1]

        # train the SVM
        self.model_ = SVC(C=self.C, kernel=self.kernel, random_state=self.random_state).fit(X, y)
        self.coef_ = self.model_.dual_coef_[0]
        self.coef_indices_ = self.model_.support_
        self.intercept_ = self.model_.intercept_[0]

        # make sure our decomposition is making the predictions as the svm
        assert np.allclose(self.model_.predict(X), self.predict(X))
        assert np.allclose(self.model_.decision_function(X), self.decision_function(X))

        return self

    # TODO: if X.shape[0] is large, and self.X_train_.shape[0] is also large, then computing
    # the similarity matrix all at once may be intractable, do them one at a time.
    def decision_function(self, X):
        assert X.ndim == 2
        X_sim = self.kernel(X, self.X_train_[self.coef_indices_])
        decision = np.sum(X_sim * self.coef_, axis=1) + self.intercept_
        return decision

    def predict(self, X):
        pred_label = np.where(self.decision_function(X) >= 0, 1, 0)
        return pred_label

    def get_weight(self):
        """
        Return a sparse array of train instance weights with shape (1, n_train_samples).
        """
        data = self.coef_
        indices = self.coef_indices_
        indptr = np.array([0, len(data)])
        return sps.csr_matrix((data, indices, indptr), shape=(1, len(self.X_train_)))

    def explain(self, x):
        """
        Return a sparse matrix of the impact of the training instances on x.
        The resulting array is of shape (1, n_train_samples).
        """
        assert x.shape == (1, self.X_train_.shape[1])
        x_sim = self.kernel(x, self.X_train_[self.coef_indices_])
        impact = (x_sim * self.coef_)[0]
        indptr = np.array([0, len(impact)])

        print(self.decision_function(x))
        print(x_sim)
        print(self.coef_)
        print(np.sum(impact) + self.intercept_)

        print(impact)
        print(self.coef_indices_)
        print(indptr)

        return sps.csr_matrix((impact, self.coef_indices_, indptr), shape=(1, len(self.X_train_)))


class KernelLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Wrapper around liblinear. Solves the l2 logistic regression dual problem using a linear kernel.
    Reference: https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf
    If multiclass, uses a one-vs-rest strategy and fits a BinaryKernelLogisticRegression classifier for each class.
    """

    def __init__(self, C=1.0):
        """
        Parameters
        ----------
        C: float (default=1.0)
            Regularization parameter, where 0 <= alpha_i <= C.
        random_state: int (default=None)
            Number for reproducibility.
        """
        self.C = C

    def fit(self, X, y):
        self.X_train_ = X
        self.n_features_ = X.shape[1]
        estimator = BinaryKernelLogisticRegression(C=self.C)
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

        return np.vstack([self.ovr_.estimators_[y[i]].explain(x.reshape(1, -1)) for i, x in enumerate(X)])


class BinaryKernelLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Wrapper around liblinear. Solves the l2 logistic regression dual problem using a linear kernel.
    Reference: https://www.csie.ntu.edu.tw/~cjlin/papers/liblinear.pdf
    """

    def __init__(self, C=1.0, temp_dir='.temp_klr'):
        """
        Parameters
        ----------
        C: float (default=1.0)
            Regularization parameter, where 0 <= alpha_i <= C.
        temp_dir: str (default='.temp_klr')
            Temporary directory for storing liblinear models and prediction files.
        """
        self.C = C
        self.temp_dir = temp_dir

    def fit(self, X, y):

        # store training instances for later use
        self.X_train_ = X
        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2

        # remove any previously stored models
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir)

        # setup path names
        train_data_path = os.path.join(self.temp_dir, 'train_data')
        model_path = os.path.join(self.temp_dir, 'model')
        prediction_path = os.path.join(self.temp_dir, 'prediction')

        # train the model using liblinear
        y_liblinear = np.where(y == 0, -1, 1)  # liblinear works better with -1 instead of 0
        liblinear_util.create_data_file(X, y_liblinear, train_data_path)
        liblinear_util.train_model(train_data_path, model_path, C=self.C)
        self.coef_ = liblinear_util.parse_model_file(model_path)

        # make sure our decomposition is making the same predictions as liblinear
        liblinear_util.predict(train_data_path, model_path, prediction_path)
        pred_label, pred_proba = liblinear_util.parse_prediction_file(prediction_path, minus_to_zeros=True)
        assert np.allclose(pred_label, self.predict(X))
        assert np.allclose(pred_proba.flatten(), self.predict_proba(X).flatten())

        return self

    # TODO: if X.shape[0] is large, and self.X_train_.shape[0] is also large, then computing
    # the similarity matrix all at once may be intractable, do them one at a time.
    def predict_proba(self, X):
        assert X.ndim == 2
        X_sim = linear_kernel(X, self.X_train_)
        proba_pos = self._sigmoid(np.sum(X_sim * self.coef_, axis=1)).reshape(-1, 1)
        proba = np.hstack([1 - proba_pos, proba_pos])
        return proba

    def predict(self, X):
        pred_label = np.argmax(self.predict_proba(X), axis=1)
        return pred_label

    def get_weight(self):
        """
        Return an array of train instance weights.
        """
        return self.coef_.copy()

    def explain(self, X):
        """
        Return an array of train instance impacts of shape (len(X), n_train_samples).
        """
        X_sim = linear_kernel(X, self.X_train_)
        print(self.predict_proba(X))
        print(X_sim)
        print(self.coef_)
        impact = X_sim * self.coef_
        print(np.sum(impact))
        print(self._sigmoid(np.sum(impact)))
        return impact

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
