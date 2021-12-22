from typing import Optional

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)

from fedot.core.log import Log
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation


class DiscriminantAnalysisImplementation(ModelImplementation):

    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__(log)
        self.params = params
        self.model = None

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """
        self.model.fit(train_data.features, train_data.target)
        return self.model

    def predict(self, input_data, is_fit_pipeline_stage: Optional[bool] = None):
        """ Method make prediction with labels of classes

        :param input_data: data with features to process
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        """
        prediction = self.model.predict(input_data.features)

        prediction = nan_to_num(prediction)

        return prediction

    def predict_proba(self, input_data):
        """ Method make prediction with probabilities of classes

        :param input_data: data with features to process
        """
        prediction = self.model.predict_proba(input_data.features)

        prediction = nan_to_num(prediction)

        return prediction

    def get_params(self):
        """ Method return parameters, which can be optimized for particular
        operation
        """
        return self.model.get_params()

    @property
    def classes_(self):
        return self.model.classes_


class LDAImplementation(DiscriminantAnalysisImplementation):

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        if not params:
            self.model = LinearDiscriminantAnalysis()
        else:
            self.model = LinearDiscriminantAnalysis(**params)
        self.params = params
        self.parameters_changed = False

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """
        try:
            self.model.fit(train_data.features, train_data.target)
        except ValueError:
            # Problem arise when features and target are "ideally" mapping
            # features [[1.0], [0.0], [0.0]] and target [[1], [0], [0]]
            self.parameters_changed = True
            new_solver = 'lsqr'
            self.log.debug(f'Change invalid parameter solver ({self.model.solver}) to {new_solver}')

            self.model.solver = new_solver
            self.params['solver'] = new_solver
            self.model.fit(train_data.features, train_data.target)
        return self.model

    def get_params(self):
        if self.parameters_changed is True:
            return tuple([self.params, ['solver']])
        else:
            return self.params


class QDAImplementation(DiscriminantAnalysisImplementation):

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        if not params:
            self.model = QuadraticDiscriminantAnalysis()
        else:
            self.model = QuadraticDiscriminantAnalysis(**params)
        self.params = params


def nan_to_num(prediction):
    """ Function converts nan values to numerical

    :return prediction: prediction without nans
    """
    if np.array([pd.isna(_) for _ in prediction]).any():
        prediction = np.nan_to_num(prediction)

    return prediction
