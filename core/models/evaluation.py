import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

from core.models.data import (
    InputData
)
from core.models.model import (
    Model, train_test_data_setup
)


class EvaluationStrategy:
    def __init__(self, model: Model):
        self.model = model
        self.is_train_models = True
        self.is_fitted = False

    def evaluate(self, data: InputData) -> InputData:
        data.features = _preprocess(data.features)
        if self.is_train_models or not self.is_fitted:
            train_data, _ = train_test_data_setup(data)
            self.model.fit(data=train_data)
            self.is_fitted = True
        prediction = self.model.predict(data=data)
        if any([np.isnan(_) for _ in prediction]):
            print("Value error")
        return prediction


def _preprocess(x):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(x)
    x = imp.transform(x)
    x_float = x.astype(float)
    return preprocessing.scale(x_float)
