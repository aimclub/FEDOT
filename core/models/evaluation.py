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
        data.features = _preprocess(data.features)
        if self.is_train_models:
            self.model.fit(data=data)
        prediction = self.model.predict(data=data)
        if any([np.isnan(_) for _ in prediction]):
            print("Value error")
        return prediction


def _preprocess(x):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(x)
    x = imp.transform(x)
    return preprocessing.scale(x)
