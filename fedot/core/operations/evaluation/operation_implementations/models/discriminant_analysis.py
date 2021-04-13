import numpy as np

from typing import Optional
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from fedot.core.operations.evaluation.\
    operation_implementations.implementation_interfaces import ModelImplementation


class DiscriminantAnalysisImplementation(ModelImplementation):

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.params = params
        self.model = None

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """
        self.model.fit(train_data.features, train_data.target)

        return self.model

    def predict(self, input_data, is_fit_chain_stage: Optional[bool] = None):
        """ Method make prediction with labels of classes

        :param input_data: data with features to process
        :param is_fit_chain_stage: is this fit or predict stage for chain
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
    if np.array([np.isnan(_) for _ in prediction]).any():
        prediction = np.nan_to_num(prediction)

    return prediction
