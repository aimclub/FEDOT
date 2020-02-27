import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

from core.models.data import (
    InputData
)
from core.models.model import (
    Model
)


class EvaluationStrategy:
    def __init__(self, model: Model):
        self.model = model
        self.is_train_models = True

    def evaluate(self, data: InputData) -> InputData:
        data.features = EvaluationStrategy.preprocess(data.features)
        if self.is_train_models:
            self.model.fit(data=data)
        return self.model.predict(data=data)

    @staticmethod
    def preprocess(x):
        """Normalize data with sklearn.preprocessing.scale()"""
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(x)
        x = imp.transform(x)
        return preprocessing.scale(x)
