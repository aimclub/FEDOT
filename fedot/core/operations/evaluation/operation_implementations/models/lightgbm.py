from typing import Optional

import lightgbm
import numpy as np

from fedot.core.data.data import InputData
from fedot.core.log import Log
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    ModelImplementation


class FedotLGBMRegImplementation(ModelImplementation):
    """ Wrapper above LightGBM regression model """

    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__(log)
        self.params = {**{'objective': 'regression'}, **params}
        self.model = None

    def fit(self, train_data: InputData):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """
        lgb_train = lightgbm.Dataset(train_data.features,
                                     np.ravel(train_data.target),
                                     params=self.params,
                                     free_raw_data=True)
        self.model = lightgbm.train(self.params, lgb_train, verbose_eval=False)
        return self.model

    def predict(self, input_data: InputData, is_fit_pipeline_stage: Optional[bool] = None):
        prediction = self.model.predict(input_data.features)
        prediction = self._convert_to_output(input_data, prediction)
        return prediction

    def get_params(self):
        return self.params


class FedotLGBMClassImplementation(ModelImplementation):
    """ Wrapper above LightGBM classification model """

    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__(log)
        self.params = params
        self.model = None
        self.classes = None

    def fit(self, train_data: InputData):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """
        self.classes = np.unique(train_data.target)
        if len(self.classes) <= 2:
            self.params = {**{'objective': 'binary'}, **self.params}
        else:
            self.params = {**{'objective': 'multiclass'}, **self.params}

        if self.params['objective'] == 'binary' and 'class_weight' in self.params.keys():
            del self.params['class_weight']

        lgb_train = lightgbm.Dataset(train_data.features,
                                     label=np.ravel(train_data.target),
                                     params=self.params,
                                     free_raw_data=True)
        self.model = lightgbm.train(self.params, lgb_train, verbose_eval=False)
        return self.model

    def predict(self, input_data, is_fit_pipeline_stage: Optional[bool] = None):
        # TODO always return probabilities, not labels
        prediction = self.model.predict(input_data.features)
        return prediction

    def predict_proba(self, input_data):
        prediction = self.model.predict(input_data.features)
        return prediction

    def get_params(self):
        return self.params

    @property
    def classes_(self):
        return self.classes
