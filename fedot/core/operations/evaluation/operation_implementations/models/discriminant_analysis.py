import numpy as np

from typing import Optional
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)


class DiscriminantAnalysisImplementation:

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.params = params
        self.model = None

    def fit(self, features, target):
        """ Method fit model on a dataset

        :param features: features for model
        :param target: target for model
        """
        self.model.fit(features, target)

        return self.model

    def predict(self, features):
        """ Method make prediction with labels of classes

        :param features: data with features to process
        """
        prediction = self.model.predict(features)

        prediction = nan_to_num(prediction)

        return prediction

    def predict_proba(self, features):
        """ Method make prediction with probabilities of classes

        :param features: data with features to process
        """
        prediction = self.model.predict_proba(features)

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
