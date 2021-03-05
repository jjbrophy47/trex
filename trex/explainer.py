"""
Instance-based explainer for a tree ensemble using an SVM or KLR.

Currently supported tree-ensembles:
    -Sklearn's RandomForestClassifier and GradientBoostingClassifier,
    -LightGBMClassifier,
    -XGBClassifier,
    -CatBoostClassifier.

NOTE: Binary classification only.
NOTE: Dense input only.
"""
import time

import numpy as np
from sklearn.utils.validation import check_X_y

from .extractor import TreeExtractor
from .util import train_surrogate


class TreeExplainer:

    def __init__(self,
                 model,
                 X_train,
                 y_train,
                 kernel_model='klr',
                 tree_kernel='leaf_output',
                 val_frac=0.1,
                 param_grid={'C': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]},
                 metric='pearson',
                 weighed=False,
                 weighted=False,
                 pred_size=1000,
                 random_state=None,
                 logger=None):
        """
        Trains an svm on feature representations from a learned tree ensemble.

        Parameters
        ----------
        model : object
            Learned tree ensemble.
            Supported: CatBoost and RandomForestClassifier.
        X_train : 2d array-like
            Train instances in original feature space.
        y_train : 1d array-like (default=None)
            Ground-truth train labels.
        kernel_model : str (default='svm', {'svm', 'klr'})
            Kernel model to approximate the tree ensemble.
        tree_kernel : str (default='leaf_output', {'leaf_output', 'tree_output', leaf_path', 'feature_path'})
            Feature representation to extract from the tree ensemble.
        val_frac : float (default=0.1)
            Fraction of training data used to tune the hyperparameters of the surrogate model.
        param_grid : dict (default={'C': 1e-2, 1e-1, 1e0, 1e1})
            Hyperparameter values to try during tuning.
        metric : str (default='mse')
            Metric to use for scoring during tuning.
        weighted : bool (default=False)
            If True, train surrogate on weighted training samples based on the predicted
            probabilities of the tree ensemble.
        pred_size : int (default=1000)
            Break predictions up into chunks to avoid memory explosion.
        random_state : int (default=None)
            Random state to promote reproducibility.
        logger : obj (default=None)
            Logging object; if not None, shows progress during tuning.
        """
        self.model = model
        self.kernel_model = kernel_model
        self.tree_kernel = tree_kernel
        self.param_grid = param_grid
        self.metric = metric
        self.weighted = weighted
        self.pred_size = pred_size
        self.random_state = random_state
        self.logger = logger

        # extract feature representations from the tree ensemble
        self.feature_extractor_ = TreeExtractor(self.model, tree_kernel=self.tree_kernel)

        # transform train data
        start = time.time()
        self.X_train_alt_ = self.feature_extractor_.transform(X_train)
        if logger:
            logger.info('\ntransforming features...{:.3f}s'.format(time.time() - start))
            logger.info('no. features after transformation: {:,}'.format(self.X_train_alt_.shape[1]))

        # train on predicted labels from the tree-ensemble
        self.y_train_ = self.model.predict(X_train)

        # train surrogate model
        self.surrogate_ = train_surrogate(model,
                                          kernel_model,
                                          param_grid,
                                          X_train,
                                          self.X_train_alt_,
                                          y_train,
                                          val_frac=val_frac,
                                          metric=self.metric,
                                          weighted=self.weighted,
                                          seed=self.random_state,
                                          logger=self.logger)

        # validation checks
        self.validate()

        # record no. original features
        self.n_features_ = X_train.shape[1]
        self.n_features_alt_ = self.X_train_alt_.shape[1]

    def decision_function(self, X):
        """
        Computes the decision values for X using the SVM.

        Parameters
        ----------
        X : 2D numpy array
            Instances to make predictions on.

        Returns a 1d array of decision vaues of shape=(X.shape[0],).
        """
        assert self.kernel_model == 'svm', 'decision_function only supports svm!'
        assert X.ndim == 2

        decisions = []
        for i in range(0, len(X), self.pred_size):
            X_sub = self.transform(X[i: i + self.pred_size])
            decisions.append(self.surrogate_.decision_function(X_sub))
        return np.concatenate(decisions)

    def predict_proba(self, X):
        """
        Computes the probabilities for X using the logistic regression model.

        Parameters
        ----------
        X : 2D numpy array
            Instances to make predictions on.

        Returns a 2d array of probabilities of shape=(X.shape[0], no. classes).
        """
        probas = []
        for i in range(0, len(X), self.pred_size):
            X_sub = self.transform(X[i: i + self.pred_size])
            probas.append(self.surrogate_.predict_proba(X_sub))
        return np.vstack(probas)

    def predict(self, X):
        """
        Computes the predicted labels for X using the kernel model.

        Parameters
        ----------
        X : 2D numpy array
            Instances to make predictions on.

        Return 1 1d array of predicted labels of shape=(X.shape[0],).
        """
        predictions = []
        for i in range(0, len(X), self.pred_size):
            X_sub = self.transform(X[i: i + self.pred_size])
            predictions.append(self.surrogate_.predict(X_sub))
        return np.concatenate(predictions)

    def similarity(self, X):
        """
        Computes the similarity between the instances in X and the training data.

        Parameters
        ----------
        X : 2D numpy array
            Instances to compute the similarity to.

        Returns an array of shape=(X.shape[0], no. train samples).
        """
        similarities = []
        for i in range(0, len(X), self.pred_size):
            X_sub = self.transform(X[i: i + self.pred_size])
            similarities.append(self.surrogate_.similarity(X_sub))
        return np.vstack(similarities)

    def get_alpha(self):
        """
        Return a 1D array of training instance weights of shape=(no. train samples,).
        """
        return self.surrogate_.alpha_

    def compute_attributions(self, X):
        """
        Computes the contribution of the training instances on X. A positive number means
        the training instance contributed to the predicted label, and a negative number
        means it contributed away from the predicted label.

        Parameters
        ----------
        X: 2d array-like
            Instances to compute training instance attributions for.
        y: 1d array-like (default=None)
            Ground-truth labels of instances being explained; impact scores now represent
            the contribution towards the ground-truth label, instead of the predicted label.
            Must have the same length as X.

        Returns a 2d array of contributions of shape (X.shape[0], no. train samples).
        """
        attributions_list = []
        for i in range(0, len(X), self.pred_size):
            X_sub = self.transform(X[i: i + self.pred_size])
            attributions_list.append(self.surrogate_.compute_attributions(X_sub))

        attributions = np.vstack(attributions_list)
        assert attributions.shape == (len(X), self.n_samples_)

        return attributions

    def transform(self, X):
        """
        Transform X using the tree-ensemble feature extractor.

        Parameters
        ----------
        X : 2d array-like
            Instances to transform.

        Returns an array of shape=(X.shape[0[, no. alt. features).
        """
        assert X.ndim == 2, 'X is not 2d!'
        assert X.shape[1] == self.n_features_, 'no. features do not match!'

        X_alt = self.feature_extractor_.transform(X)
        assert X_alt.shape[1] == self.X_train_alt_.shape[1], 'no. features do not match!'

        return X_alt

    def get_params(self, deep=False):
        """
        Returns the parameter of this model as a dictionary.
        """
        d = {}
        d['model'] = str(self.model)
        d['kernel_model'] = self.kernel_model
        d['tree_kernel'] = self.tree_kernel
        d['param_grid'] = self.param_grid
        d['metric'] = self.metric
        d['pred_size'] = self.pred_size
        d['random_state'] = self.random_state
        return d

    def set_params(self, **params):
        """
        Set the parameters of this model.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # private
    def validate(self):
        """
        Check model and data inputs.
        """

        # check data
        check_X_y(self.X_train_alt_, self.y_train_)
        assert np.all(np.unique(self.y_train_) == np.array([0, 1]))

        # record data statistics
        self.n_samples_ = self.X_train_alt_.shape[0]
        self.classes_ = np.unique(self.y_train_)
        self.n_classes_ = self.classes_.shape[0]

        # check model
        model_types = ['RandomForestClassifier', 'GradientBoostingClassifier',
                       'LGBMClassifier', 'CatBoostClassifier', 'XGBClassifier']
        valid = False
        for model_type in model_types:
            if model_type in str(self.model):
                valid = True
        assert valid, '{} not currently supported!'.format(str(self.model))

        # check tree kernel
        tree_types = ['feature_path', 'feature_output', 'leaf_path', 'leaf_output', 'tree_output']
        assert self.tree_kernel in tree_types, '{} unsupported!'.format(self.tree_kernel)

        # check kernel model
        assert self.kernel_model in ['klr', 'svm'], '{} unsupported'.format(self.kernel_model)
