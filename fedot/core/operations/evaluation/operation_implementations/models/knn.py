import numpy as np

from typing import Optional
from sklearn.neighbors import KNeighborsClassifier
from fedot.core.operations.evaluation.\
    operation_implementations.implementation_interfaces import ModelImplementation
import warnings


class CustomKnnImplementation(ModelImplementation):
    def __init__(self, **params: Optional[dict]):
        super().__init__()
        if not params:
            self.model = KNeighborsClassifier()
        else:
            self.model = KNeighborsClassifier(**params)
        self.params = params
        self.classes = None

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """
        self.classes = np.unique(np.array(train_data.target))

        # Improve hyperparameters for model
        self.check_and_correct_neighbors(train_data)
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

    def check_and_correct_neighbors(self, input_data) -> None:
        """ Method check if the amount of neighbors is too big - clip it

        :param input_data: InputData for fit
        """
        current_params = self.model.get_params()
        n_neighbors = current_params.get('n_neighbors')

        if n_neighbors > len(input_data.features):
            # Improve the parameter "n_neighbors": n_neighbors <= n_samples
            current_params.update({'n_neighbors': len(input_data.features)})
            self.model = KNeighborsClassifier(**current_params)

            prefix = "Warning: n_neighbors of K-nn model was changed"
            warnings.warn(f"{prefix} from {n_neighbors} to {len(input_data.features)}")

    @property
    def classes_(self):
        return self.classes
