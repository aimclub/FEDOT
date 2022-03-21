from typing import Optional

import lightgbm
import numpy as np
from fedot.core.log import Log
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    ModelImplementation


class FedotLGBMRegImplementation(ModelImplementation):

    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__(log)
        self.params = params
        self.model = None

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """
        params = {'verbose': -1, 'objective': 'regression'}
        lgb_train = lightgbm.Dataset(train_data.features,
                                     np.ravel(train_data.target),
                                     params=params,
                                     free_raw_data=True)
        operation_implementation = lightgbm.train(params, lgb_train,
                                                  verbose_eval=False)

    def predict(self, input_data, is_fit_pipeline_stage: Optional[bool] = None):
        """ Method for making prediction

        :param input_data: data with features to process
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        """
        prediction = self.model.predict(input_data.features)

        return prediction

    def get_params(self):
        return self.params
