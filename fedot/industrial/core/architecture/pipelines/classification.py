import numpy as np
from fedot.core.pipelines.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

from fedot.industrial.core.architecture.preprocessing.data_convertor import NumpyConverter


class SklearnCompatibleClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper for FedotIndustrial to make it compatible with sklearn.

    Args:
        estimator (Pipeline): FedotIndustrial pipeline.

    """

    def __init__(self, estimator: Pipeline):
        self.estimator = estimator
        self.classes_ = None

    def fit(self, X, y):
        features, target = NumpyConverter(
            X).numpy_data, NumpyConverter(y).numpy_data
        self.estimator.fit(features, target)
        self.classes_ = np.unique(target)
        return self

    def predict(self, X):
        features = NumpyConverter(X).numpy_data
        labels = self.estimator.predict(features).predict
        return labels

    def predict_proba(self, X):
        features = NumpyConverter(X).numpy_data
        probs = self.estimator.predict(features, output_mode='probs').predict
        return probs
