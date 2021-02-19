"""
Instance-based explainer for a tree ensemble using an SVM or KLR.
Currently supports: sklearn's RandomForestClassifier and GBMClassifier,
    lightgbm, xgboost, and catboost.
    Is also only compatible with dense dataset inputs.
"""
import os
import uuid

# import pickle
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
                 param_grid={'C': [1e-2, 1e-1, 1e0]},
                 metric='pearson',
                 use_predicted_label=True,
                 pred_size=1000,
                 random_state=None,
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
        random_state : int (default=5)
            Number of cross-validation folds to use for tuning.
        temp_dir : str (default='.trex')
        """
        self.model = model
        self.kernel_model = kernel_model
        self.tree_kernel = tree_kernel
        self.param_grid = param_grid
        self.metric = metric
        self.pred_size = pred_size
        self.use_predicted_label = use_predicted_label
        self.random_state = random_state
        self.logger = logger
        self.temp_dir = os.path.join(temp_dir, str(uuid.uuid4()))

        # extract feature representations from the tree ensemble
        self.feature_extractor_ = TreeExtractor(self.model, tree_kernel=self.tree_kernel)

        # transform train data
        self.X_train_alt_ = self.feature_extractor_.fit_transform(X_train)
        self.y_train_ = self.model.predict(X_train) if use_predicted_label else y_train

        # train surrogate model
        self.surrogate_ = train_surrogate(model,
                                          kernel_model,
                                          param_grid,
                                          X_train,
                                          self.X_train_alt_,
                                          self.y_train_,
                                          val_frac=val_frac,
                                          metric=self.metric,
                                          seed=self.random_state,
                                          logger=self.logger,
                                          temp_dir=self.temp_dir)

        # validation checks
        self.validate()

        # record no. original features
        self.n_features_ = X_train.shape[1]

    # def __del__(self):
    #     """
    #     Clean up any temporary directories.
    #     """
    #     shutil.rmtree(self.temp_dir)

    # def __str__(self):
    #     s = '\nTree Explainer:'
    #     s += '\nextractor: {}'.format(self.feature_extractor_)
    #     s += '\ntrain_: {}'.format(self.X_train_alt_)
    #     s += '\nkernel model: {}'.format(self.kernel_model_)
    #     s += '\nn_samples: {}'.format(self.n_samples_)
    #     s += '\nn_feats: {}'.format(self.n_feats_)
    #     s += '\nn_classes: {}'.format(self.n_classes_)
    #     s += '\n'
    #     return s

    # def save(self, fn):
    #     """
    #     Stores the model for later use.
    #     """
    #     with open(fn, 'wb') as f:
    #         f.write(pickle.dumps(self))

    # def load(fn):
    #     """
    #     Loads a previously saved model.
    #     """
    #     assert os.path.exists(fn)
    #     with open(fn, 'rb') as f:
    #         return pickle.loads(f.read())

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
            decisions.append(self.surrogate_.decision_function(X_sub))
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
            probas.append(self.surrogate_.predict_proba(X_sub))
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
            predictions.append(self.surrogate_.predict(X_sub))
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
            similarities.append(self.surrogate_.similarity(X_sub))
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
            y_sub = y[i: i + self.pred_size] if y is not None else None
            contributions_list.append(self.surrogate_.explain(X_sub, y=y_sub))

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
        assert X.shape[1] == self.n_features_, 'no. features do not match!'

        X_feature = self.feature_extractor_.transform(X)
        assert X_feature.shape[1] == self.X_train_alt_.shape[1], 'no. features do not match!'

        return X_feature

    def get_params(self, deep=False):
        """
        Returns the parameter of this model as a dictionary.
        """
        d = {}
        d['tree_kernel'] = self.tree_kernel
        # d['tree_type'] = str(self.tree)
        d['kernel_model'] = self.kernel_model
        # d['C'] = self.C
        # d['true_label'] = self.true_label
        d['random_state'] = self.random_state
        # d['train_shape'] = self.X_train.shape
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
        tree_types = ['leaf_path', 'tree_output', 'leaf_output', 'feature_path']
        assert self.tree_kernel in tree_types, '{} unsupported!'.format(self.tree_kernel)

        # check kernel model
        assert self.kernel_model in ['klr', 'svm'], '{} unsupported'.format(self.kernel_model)
