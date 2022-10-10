from typing import Optional

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class DiscriminantAnalysisImplementation(ModelImplementation):

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = None

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """
        self.model.fit(train_data.features, train_data.target)
        return self.model

    def predict(self, input_data):
        """ Method make prediction with labels of classes

        :param input_data: data with features to process
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

    @property
    def classes_(self):
        return self.model.classes_


class LDAImplementation(DiscriminantAnalysisImplementation):

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = LinearDiscriminantAnalysis(**self.params.to_dict())

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """

        self.check_and_correct_params()

        try:
            self.model.fit(train_data.features, train_data.target)
        except ValueError:
            # Problem arise when features and target are "ideally" mapping
            # features [[1.0], [0.0], [0.0]] and target [[1], [0], [0]]
            new_solver = 'lsqr'
            self.log.debug(f'Change invalid parameter solver ({self.model.solver}) to {new_solver}')

            self.model.solver = new_solver
            self.params.update(solver=new_solver)
            self.model.fit(train_data.features, train_data.target)
        return self.model

    def check_and_correct_params(self):
        """ Checks if the hyperparameters for the LDA model are correct and fixes them if needed """
        current_solver = self.params.get('solver')
        current_shrinkage = self.params.get('shrinkage')

        is_solver_svd = current_solver is not None and current_solver == 'svd'
        if is_solver_svd and current_shrinkage is not None:
            # Ignore shrinkage
            self.params.update(shrinkage=None)
            self.model.shrinkage = None


class QDAImplementation(DiscriminantAnalysisImplementation):

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = QuadraticDiscriminantAnalysis(**self.params.to_dict())


def nan_to_num(prediction):
    """ Function converts nan values to numerical

    :return prediction: prediction without nans
    """
    if np.array([pd.isna(_) for _ in prediction]).any():
        prediction = np.nan_to_num(prediction)

    return prediction
