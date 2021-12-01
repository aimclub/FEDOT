from typing import Callable, Optional

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from fedot.core.log import Log
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation


class KNeighborsImplementation(ModelImplementation):

    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__(log)
        self.parameters_changed = False
        self.params = params
        self.model = None

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """

        raise NotImplementedError()

    def predict(self, input_data, is_fit_pipeline_stage: Optional[bool] = None):
        """ Method for making prediction

        :param input_data: data with features to process
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        """
        prediction = self.model.predict(input_data.features)

        return prediction

    def check_and_correct_k_value(self, input_data, model_impl: Callable):
        """ Method check if the amount of neighbors is too big - clip it

        :param input_data: InputData for fit
        :param model_impl: Model to use
        """
        was_changed = False
        current_params = self.model.get_params()
        n_neighbors = current_params.get('n_neighbors')

        if n_neighbors > len(input_data.features):
            # Improve the parameter "n_neighbors": n_neighbors <= n_samples
            new_k_value = round(len(input_data.features) / 2)
            if new_k_value == 0:
                new_k_value = 1
            current_params.update({'n_neighbors': new_k_value})
            self.model = model_impl(**current_params)

            prefix = "n_neighbors of K-nn model was changed"
            self.log.info(f"{prefix} from {n_neighbors} to {new_k_value}")
            was_changed = True

        return was_changed

    def get_params(self):
        """ Method return parameters, which can be optimized for particular
        operation
        """
        if self.parameters_changed is True:
            params_dict = self.model.get_params()
            return tuple([params_dict, ['n_neighbors']])
        else:
            return self.model.get_params()


class FedotKnnClassImplementation(KNeighborsImplementation):
    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__(log)
        if not params:
            self.model = KNeighborsClassifier()
        else:
            round_n_neighbors(params)
            self.model = KNeighborsClassifier(**params)
        self.params = params
        self.classes = None

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """
        self.classes = np.unique(np.array(train_data.target))

        # Improve hyperparameters for model
        self.parameters_changed = self.check_and_correct_k_value(train_data, KNeighborsClassifier)
        self.model.fit(train_data.features, train_data.target)
        return self.model

    def predict_proba(self, input_data):
        """ Method make prediction with probabilities of classes

        :param input_data: data with features to process
        """
        prediction = self.model.predict_proba(input_data.features)

        return prediction

    @property
    def classes_(self):
        return self.classes


class FedotKnnRegImplementation(KNeighborsImplementation):
    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__(log)
        if not params:
            self.model = KNeighborsRegressor()
        else:
            round_n_neighbors(params)
            self.model = KNeighborsRegressor(**params)
        self.params = params

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """

        # Improve hyperparameters for model
        self.parameters_changed = self.check_and_correct_k_value(train_data, KNeighborsRegressor)
        self.model.fit(train_data.features, train_data.target)
        return self.model


def round_n_neighbors(params):
    """ Convert n_neighbors into integer value. Operation work inplace. """
    if params.get('n_neighbors') is not None:
        n_neighbors = round(params.get('n_neighbors'))
        if n_neighbors == 0:
            n_neighbors = 1
        params['n_neighbors'] = n_neighbors
