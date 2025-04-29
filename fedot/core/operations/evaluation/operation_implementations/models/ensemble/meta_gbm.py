from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelBinarizer

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class MetaGBMImplementation(ModelImplementation):

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = None

    def fit(self, input_data: InputData):
        self.model.fit(input_data.features, input_data.target)
        return self

    def predict(self, input_data: InputData) -> OutputData:
        labels = self.model.predict(X=input_data.features)
        output_data = self._convert_to_output(input_data=input_data, predict=labels)
        return output_data


class MetaGBMClassifier(MetaGBMImplementation):

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = LogitGBM()

    def predict_proba(self, input_data: InputData) -> OutputData:
        probs = self.model.predict_proba(X=input_data.features)
        output_data = self._convert_to_output(input_data=input_data, predict=probs)
        return output_data


class MetaGBMRegressor(MetaGBMImplementation):

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = LinearGBM()


class LogitGBM:
    def __init__(self, n_trees=100, learning_rate=0.1):
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.models = []
        self.n_classes = None
        self.init_logits = None
        self.label_binarizer = LabelBinarizer()

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _softmax(self, logits):
        logits = np.clip(logits, -500, 500)
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def fit(self, X, y):
        y = np.asarray(y).ravel()
        self.label_binarizer.fit(y)
        Y_bin = self.label_binarizer.transform(y)
        self.n_classes = Y_bin.shape[1]

        n_samples = X.shape[0]
        self.models = [[] for _ in range(self.n_classes)]

        # Инициализируем логиты (log-odds)
        self.init_logits = np.zeros(self.n_classes)
        for k in range(self.n_classes):
            p = np.clip(Y_bin[:, k].mean(), 1e-15, 1 - 1e-15)
            self.init_logits[k] = np.log(p / (1 - p))

        Fm = np.tile(self.init_logits, (n_samples, 1))  # (n_samples, n_classes)

        for _ in range(self.n_trees):
            for k in range(self.n_classes):
                if self.n_classes == 2:
                    preds = self._sigmoid(Fm[:, k])
                else:
                    preds = self._softmax(Fm)[:, k]

                residual = Y_bin[:, k] - preds

                model = LinearRegression(n_jobs=-1)
                model.fit(X, residual)
                update = model.predict(X).ravel()

                Fm[:, k] += self.learning_rate * update
                self.models[k].append(model)

    def _predict_logits(self, X):
        n_samples = X.shape[0]
        Fm = np.tile(self.init_logits, (n_samples, 1))  # (n_samples, n_classes)

        for k in range(self.n_classes):
            for model in self.models[k]:
                update = model.predict(X).ravel()
                Fm[:, k] += self.learning_rate * update

        return Fm

    def predict_proba(self, X):
        logits = self._predict_logits(X)

        if self.n_classes == 2:
            probs_class_1 = self._sigmoid(logits[:, 1])
            probs = np.vstack([1 - probs_class_1, probs_class_1]).T
        else:
            probs = self._softmax(logits)

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.label_binarizer.inverse_transform(probs)


class LinearGBM:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.initial_pred = None

    def fit(self, X, y):
        self.initial_pred = np.mean(y)
        current_pred = np.full_like(y, self.initial_pred, dtype=np.float64)

        for _ in range(self.n_estimators):
            residuals = y - current_pred
            est = LinearRegression()
            est.fit(X, residuals)
            current_pred =+ self.learning_rate * est.predict(X)
            self.estimators.append(est)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.initial_pred, dtype=np.float64)

        for est in self.estimators:
            y_pred += self.learning_rate * est.predict(X)

        return y_pred
