"""
Instance-based explainer for a tree ensemble using an SVM or KLR.
Currently supports: sklearn's RandomForestClassifier and GBMClassifier,
    lightgbm, xgboost, and catboost.
    Is also only compatible with dense dataset inputs.
"""
import os
import time
import uuid

import pickle
import numpy as np
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

from .extractor import TreeExtractor
from .models.linear_model import SVM, KernelLogisticRegression


class TreeExplainer:

    def __init__(self, tree, X_train, y_train,
                 kernel_model='svm',
                 tree_kernel='leaf_output',
                 C=1.0,
                 val_frac=0.1,
                 pred_size=1000,
                 true_label=False,
                 random_state=None,
                 verbose=0,
                 C_grid=[1e-2, 1e-1, 1e0, 1e1, 1e2],
                 logger=None,
                 temp_dir='.trex'):
        """
        Trains an svm on feature representations from a learned tree ensemble.

        Parameters
        ----------
        tree : object
            Learned tree ensemble. Supported: RandomForestClassifier, GBM, LightGBM, CatBoost, XGBoost.
        X_train : 2d array-like
            Train instances in original feature space.
        y_train : 1d array-like (default=None)
            Ground-truth train labels.
        kernel_model : str (default='svm', {'svm', 'klr'})
            Kernel model to approximate the tree ensemble.
        tree_kernel : str (default='leaf_output', {'leaf_output', 'tree_output', leaf_path', 'feature_path'})
            Feature representation to extract from the tree ensemble.
        C : float (default=0.1)
            Regularization parameter for the kernel model. Lower C values result
            in stronger regularization.
        val_frac : float (default=0.1)
            Fraction of training data used to tune the hyperparameters of KLR or SVM.
        pred_size : int (default=1000)
            Break predictions up into chunks of pred_size.
        true_label : bool (default=False)
            If False, predicted labels from the tree ensemble are used to train the kernel model.
        random_state : int (default=None)
            Random state to promote reproducibility.
        temp_dir : str (default='.trex')
        """

        # error checking
        self.tree = tree
        self.X_train = X_train
        self.y_train = y_train
        self.kernel_model = kernel_model
        self.tree_kernel = tree_kernel
        self.C = C
        self.true_label = true_label
        self.random_state = random_state
        self.verbose = verbose
        self.logger = logger
        self.temp_dir = os.path.join(temp_dir, str(uuid.uuid4()))
        self._validate()

        if logger:
            logger.info('TREX:\ntransforming training data...')

        # extract feature representations from the tree ensemble
        self.extractor_ = TreeExtractor(self.tree, tree_kernel=self.tree_kernel)
        self.train_feature_ = self.extractor_.fit_transform(self.X_train)

        # choose ground truth or predicted labels to train the kernel model
        train_label = self.y_train if true_label else self.tree.predict(X_train).flatten()

        # encode class labels into numbers between 0 and n_classes - 1
        self.le_ = LabelEncoder().fit(train_label)
        train_label = self.le_.transform(train_label)

        # train the kernel model on the tree ensemble feature representation
        if val_frac is not None:
            if logger:
                logger.info('tuning...')
            tune_start = time.time()

            # select a fraction of the training data
            n_samples = int(X_train.shape[0] * val_frac)
            np.random.seed(self.random_state)
            val_indices = np.random.choice(np.arange(X_train.shape[0]), size=n_samples)

            X_val = X_train[val_indices]
            X_val_feature = self.train_feature_[val_indices]
            y_val = train_label[val_indices]

            # result containers
            results = []
            fold = 0

            # tune C
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.random_state)
            for train_index, test_index in skf.split(X_val_feature, y_val):
                fold += 1

                # obtain fold data
                X_val_train = X_val[train_index]
                X_val_test = X_val[test_index]
                X_val_feature_train = X_val_feature[train_index]
                X_val_feature_test = X_val_feature[test_index]
                y_val_train = y_val[train_index]

                # gridsearch on C values
                correlations = []
                for C in C_grid:

                    start = time.time()

                    # fit a tree ensemble and surrogate model
                    m1 = clone(tree).fit(X_val_train, y_val_train)
                    m2 = self._get_kernel_model(C=C).fit(X_val_feature_train, y_val_train)

                    # generate predictions
                    m1_proba = m1.predict_proba(X_val_test)[:, 1]
                    m2_proba = m2.predict_proba(X_val_feature_test)[:, 1]

                    # measure correlation
                    correlation = pearsonr(m1_proba, m2_proba)[0]
                    correlations.append(correlation)

                    if self.logger:
                        s = '[Fold {}] C={:<5}: {:.3f}s; corr={:.3f}'
                        self.logger.info(s.format(fold, C, time.time() - start, correlation))

                results.append(correlations)
            results = np.vstack(results).mean(axis=0)
            best_ndx = np.argmax(results)
            self.C = C_grid[best_ndx]

            if self.logger:
                self.logger.info('chosen C: {}'.format(self.C))
                self.logger.info('total tuning time: {:.3f}s'.format(time.time() - tune_start))
                self.logger.info('training...')

        train_start = time.time()
        clf = self._get_kernel_model(C=self.C)
        self.kernel_model_ = clf.fit(self.train_feature_, train_label)

        if self.logger:
            self.logger.info('total training time: {:.3f}s'.format(time.time() - train_start))

    def __str__(self):
        s = '\nTree Explainer:'
        s += '\nextractor: {}'.format(self.extractor_)
        s += '\ntrain_feature: {}'.format(self.train_feature_)
        s += '\nlabel encoder: {}'.format(self.le_)
        s += '\nkernel model: {}'.format(self.kernel_model_)
        s += '\nn_samples: {}'.format(self.n_samples_)
        s += '\nn_feats: {}'.format(self.n_feats_)
        s += '\nn_classes: {}'.format(self.n_classes_)
        s += '\n'
        return s

    def save(self, fn):
        """
        Stores the model for later use.
        """
        with open(fn, 'wb') as f:
            f.write(pickle.dumps(self))

    def load(fn):
        """
        Loads a previously saved model.
        """
        assert os.path.exists(fn)
        with open(fn, 'rb') as f:
            return pickle.loads(f.read())

    def decision_function(self, X):
        """
        Computes the decision values for X using the SVM.

        Parameters
        ----------
        X : 2d array-like
            Instances to make predictions on.

        Returns a 1d array of decision vaues.
        """
        assert self.kernel_model == 'svm', 'decision_function only supports svm!'
        assert X.ndim == 2

        decisions = []
        for i in range(0, len(X), self.pred_size):
            X_sub = self.transform(X[i: i + self.pred_size])
            decisions.append(self.kernel_model_.decision_function(X_sub))
        return np.concatenate(decisions)

    def predict_proba(self, X):
        """
        Computes the probabilities for X using the logistic regression model.

        Parameters
        ----------
        X : 2d array-like
            Instances to make predictions on.

        Returns a 2d array of probabilities of shape (len(X), n_classes).
        """
        probas = []
        for i in range(0, len(X), self.pred_size):
            X_sub = self.transform(X[i: i + self.pred_size])
            probas.append(self.kernel_model_.predict_proba(X_sub))
        return np.vstack(probas)

    def predict(self, X):
        """
        Computes the predicted labels for X using the kernel model.

        Parameters
        ----------
        X : 2d array-like
            Instances to make predictions on.

        Return 1 1d array of predicted labels.
        """
        predictions = []
        for i in range(0, len(X), self.pred_size):
            X_sub = self.transform(X[i: i + self.pred_size])
            predictions.append(self.kernel_model_.predict(X_sub))
        return np.concatenate(predictions)

    # Note: If len(X) is large and the number of training instances is large,
    #       the resulting matrix may be huge.
    def similarity(self, X, train_indices=None):
        """
        Computes the similarity between the instances in X and the training data.

        Parameters
        ----------
        X : 2d array-like
            Instances to compute the similarity to.
        train_indices : 1d array-like (default=None)
            If not None, compute similarities to only these train instances.

        Returns an array of shape (len(X), n_train_samples).
        """
        similarities = []
        for i in range(0, len(X), self.pred_size):
            X_sub = self.transform(X[i: i + self.pred_size])
            similarities.append(self.kernel_model_.similarity(X_sub))
        return np.vstack(similarities)

    def get_weight(self):
        """
        Return an array of training instance weights.
            If binary, returns an array of shape (1, n_train_samples).
            If multiclass, returns an array of shape (n_classes, n_train_samples).
        """
        weight = self.kernel_model_.get_weight()

        if self.n_classes_ == 2:
            assert weight.shape == (1, self.n_samples_)
        else:
            assert weight.shape == (self.n_classes_, self.n_samples_)

        return weight

    def explain(self, X, y=None):
        """
        Computes the contribution of the training instances on X. A positive number means
        the training instance contributed to the predicted label, and a negative number
        means it contributed away from the predicted label.

        Parameters
        ----------
        X: 2d array-like
            Instances to explain.
        y: 1d array-like (default=None)
            Ground-truth labels of instances being explained; impact scores now represent
            the contribution towards the ground-truth label, instead of the predicted label.
            Must have the same length as X.

        Returns a 2d array of contributions of shape (len(X), n_train_samples).
        """
        contributions_list = []
        for i in range(0, len(X), self.pred_size):
            X_sub = self.transform(X[i: i + self.pred_size])
            y_sub = y[i: i + self.pred_size]
            contributions_list.append(self.kernel_model_.explain(X_sub, y=y_sub))

        contributions = np.vstack(contributions_list)
        assert contributions.shape == (len(X), self.n_samples_)

        return contributions

    def transform(self, X):
        """
        Transform X using the tree-ensemble feature extractor.

        Parameters
        ----------
        X : 2d array-like
            Instances to transform.

        Returns an array of shape (len(X), n_tree_features).
        """
        assert X.ndim == 2, 'X is not 2d!'
        assert X.shape[1] == self.n_feats_, 'num features do not match!'

        X_feature = self.extractor_.transform(X)
        assert X_feature.shape[1] == self.train_feature_.shape[1], 'num features do not match!'

        return X_feature

    def get_params(self, deep=False):
        """
        Returns the parameter of this model as a dictionary.
        """
        d = {}
        d['tree_kernel'] = self.tree_kernel
        d['tree_type'] = str(self.tree)
        d['kernel_model'] = self.kernel_model
        d['C'] = self.C
        d['true_label'] = self.true_label
        d['random_state'] = self.random_state
        d['train_shape'] = self.X_train.shape
        return d

    def set_params(self, **params):
        """
        Set the parameters of this model.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _get_kernel_model(self, C=0.1):
        """
        Return C implementation of the kernel model.
        """
        if self.kernel_model == 'svm':
            kernel_model = SVM(C=C, temp_dir=self.temp_dir)
        else:
            kernel_model = KernelLogisticRegression(C=C, temp_dir=self.temp_dir)

        return kernel_model

    def _validate(self):
        """
        Check model and data inputs.
        """

        # check data
        check_X_y(self.X_train, self.y_train)
        if self.y_train.dtype == np.float and not np.all(np.mod(self.y_train, 1) == 0):
            raise ValueError('Unknown label type: ')
        self.n_samples_ = self.X_train.shape[0]
        self.n_feats_ = self.X_train.shape[1]
        self.classes_ = np.unique(self.y_train)
        self.n_classes_ = len(self.classes_)
        self.labels_ = dict(zip(self.classes_, np.arange(self.n_classes_)))

        # check tree_kernel
        tree_types = ['leaf_path', 'tree_output', 'leaf_output', 'feature_path']
        assert self.tree_kernel in tree_types, '{} unsupported!'.format(self.tree_kernel)

        # check model
        if 'RandomForestClassifier' in str(self.tree):
            self.tree_type_ = 'RandomForestClassifier'
        elif 'GradientBoostingClassifier' in str(self.tree):
            self.tree_type_ = 'GradientBoostingClassifier'
        elif 'LGBMClassifier' in str(self.tree):
            self.tree_type_ = 'LGBMClassifier'
        elif 'CatBoostClassifier' in str(self.tree):
            self.tree_type_ = 'CatBoostClassifier'
        elif 'XGBClassifier' in str(self.tree):
            self.tree_type_ = 'XGBClassifier'
        else:
            exit('{} not currently supported!'.format(str(self.tree)))

        # check kernel model
        assert self.kernel_model in ['svm', 'klr'], '{} unsupported'.format(self.kernel_model)
