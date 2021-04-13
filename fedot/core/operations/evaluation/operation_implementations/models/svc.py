import numpy as np

from typing import Optional
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from fedot.core.operations.evaluation.\
    operation_implementations.implementation_interfaces import ModelImplementation


class CustomSVCImplementation(ModelImplementation):
    def __init__(self, **params: Optional[dict]):
        super().__init__()
        if not params:
            self.inner_model = SVC(kernel='linear',
                                   probability=True,
                                   class_weight='balanced')
        else:
            self.inner_model = SVC(**params)
        self.params = params
        self.model = OneVsRestClassifier(self.inner_model)
        self.classes = None

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """
        self.classes = np.unique(train_data.target)
        self.model.fit(train_data.features, train_data.target)
        return self.model

    def predict(self, input_data, is_fit_chain_stage: Optional[bool] = None):
        """ Method make prediction with labels of classes

        :param input_data: data with features to process
        :param is_fit_chain_stage: is this fit or predict stage for chain
        """
        prediction = self.model.predict(input_data.features)

        return prediction

    def predict_proba(self, input_data):
        """ Method make prediction with probabilities of classes

        :param input_data: data with features to process
        """
        prediction = self.model.predict_proba(input_data.features)

        return prediction

    def get_params(self):
        """ Method return parameters, which can be optimized for particular
        operation
        """
        return self.model.get_params()

    @property
    def classes_(self):
        return self.classes
