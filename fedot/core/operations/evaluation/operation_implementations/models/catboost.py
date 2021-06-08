from typing import Optional
from catboost import CatBoostRegressor, CatBoostClassifier

from fedot.core.operations.evaluation.\
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.log import Log, default_log


class CatBoostImplementation(ModelImplementation):
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

        self.model.fit(train_data.features, train_data.target)
        return self.model

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

        return self.model.get_all_params()


class CustomCatBoostClassImplementation(CatBoostImplementation):
    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__(log)
        _default_params = {
            "allow_writing_files": False,
            "verbose": False
        }
        if params is not None:
            self.params = {**params, **_default_params}
        else:
            self.params = _default_params
        self.model = CatBoostClassifier(**self.params)

    def predict_proba(self, input_data):
        """ Method make prediction with probabilities of classes

        :param input_data: data with features to process
        """

        prediction = self.model.predict_proba(input_data.features)

        return prediction

    @property
    def classes_(self):
        return self.model.classes_


class CustomCatBoostRegImplementation(CatBoostImplementation):
    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__(log)
        _default_params = {
            "allow_writing_files": False,
            "verbose": False
        }
        if params is not None:
            self.params = {**params, **_default_params}
        else:
            self.params = _default_params
        self.model = CatBoostRegressor(**self.params)
