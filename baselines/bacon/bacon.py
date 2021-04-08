import numpy as np


class Bacon:

    def __init__(self, model, X_train, y_train):
        assert 'RandomForestClassifier' in str(model)
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

        self.train_leaf_ids_ = self.model.apply(X_train)

    def get_weights(self, x):
        x = x.reshape(1, -1)

        instance_leaf_ids = self.model.apply(x)[0]

        print(self.train_leaf_ids_.shape)
        print(instance_leaf_ids.shape)

        weights = np.zeros(self.X_train.shape[0])
        for i in range(self.model.n_estimators):
            same_leaf_train_indices = np.where(self.train_leaf_ids_[:, i] == instance_leaf_ids[i])
            weights[same_leaf_train_indices] += 1.0 / len(same_leaf_train_indices[0])

        pred_label = self.model.predict(x)[0]
        weights = np.where(self.y_train == pred_label, weights, weights * -1)
        return weights
