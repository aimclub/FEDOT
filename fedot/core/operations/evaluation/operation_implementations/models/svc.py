from typing import Optional

import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class FedotSVCImplementation(ModelImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        if not self.params:
            default_params = {'kernel': 'linear', 'probability': True, 'class_weight': 'balanced'}
            self.params.update(**default_params)
        self.inner_model = SVC(**self.params.to_dict())
        self.model = OneVsRestClassifier(self.inner_model)
        self.classes = None

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """
        self.classes = np.unique(train_data.target)
        self.model.fit(train_data.features, train_data.target)
        return self.model

    def predict(self, input_data):
        """ Method make prediction with labels of classes

        :param input_data: data with features to process
        """
        prediction = self.model.predict(input_data.features)

        return prediction

    def predict_proba(self, input_data):
        """ Method make prediction with probabilities of classes

        :param input_data: data with features to process
        """
        prediction = self.model.predict_proba(input_data.features)

        return prediction

    @property
    def classes_(self):
        return self.classes
