from typing import Optional
import lightgbm

from fedot.core.operations.evaluation.\
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.log import Log, default_log


class LightGBMImplementation(ModelImplementation):
    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__()
        self.params = params
        self.model = None

        # Define logger object
        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """

        raise NotImplementedError()

    def predict(self, input_data, is_fit_chain_stage: Optional[bool] = None):
        """ Method for making prediction

        :param input_data: data with features to process
        :param is_fit_chain_stage: is this fit or predict stage for chain
        """

        prediction = self.model.predict(input_data.features)

        return prediction

    def get_params(self):
        """ Method return parameters, which can be optimized for particular
        operation
        """

        return self.model.get_params()


class CustomLightGBMClassImplementation(LightGBMImplementation):
    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__(log)
        _default_params = {
            "num_leaves": 32,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 10,
            "learning_rate": 0.03,
            "n_estimators": 3000,
            "random_state": 42
        }
        if params is not None:
            self.params = {**params, **_default_params}
        else:
            self.params = _default_params
        self.model = lightgbm.LGBMClassifier(**self.params)

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """

        self.model.fit(train_data.features, train_data.target)
        return self.model

    def predict_proba(self, input_data):
        """ Method make prediction with probabilities of classes

        :param input_data: data with features to process
        """

        prediction = self.model.predict_proba(input_data.features)

        return prediction

    @property
    def classes_(self):
        return self.model.classes_


class CustomLightGBMRegImplementation(LightGBMImplementation):
    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__(log)
        _default_params = {
            "num_leaves": 32,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 10,
            "learning_rate": 0.03,
            "n_estimators": 3000,
            "random_state": 42
        }
        if params is not None:
            self.params = {**params, **_default_params}
        else:
            self.params = _default_params
        self.model = lightgbm.LGBMRegressor(**self.params)

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """

        self.model.fit(train_data.features, train_data.target)
        return self.model
