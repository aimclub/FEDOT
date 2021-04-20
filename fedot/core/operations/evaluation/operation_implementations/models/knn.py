import numpy as np

from typing import Optional, Callable
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from fedot.core.operations.evaluation.\
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.log import Log, default_log


class KNeighborsImplementation(ModelImplementation):

    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__()
        self.params = params
        self.model = None

        # Define logger object
        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """

        raise NotImplementedError()

    def predict(self, input_data, is_fit_chain_stage: Optional[bool] = None):
        """ Method for making prediction

        :param input_data: data with features to process
        :param is_fit_chain_stage: is this fit or predict stage for chain
        """
        prediction = self.model.predict(input_data.features)

        return prediction

    def check_and_correct_k_value(self, input_data, model_impl: Callable):
        """ Method check if the amount of neighbors is too big - clip it

        :param input_data: InputData for fit
        :param model_impl: Model to use
        """
        current_params = self.model.get_params()
        n_neighbors = current_params.get('n_neighbors')

        if n_neighbors > len(input_data.features):
            # Improve the parameter "n_neighbors": n_neighbors <= n_samples
            new_k_value = round(len(input_data.features) / 2)
            current_params.update({'n_neighbors': new_k_value})
            self.model = model_impl(**current_params)

            prefix = "n_neighbors of K-nn model was changed"
            self.log.info(f"{prefix} from {n_neighbors} to {new_k_value}")

    def get_params(self):
        """ Method return parameters, which can be optimized for particular
        operation
        """
        return self.model.get_params()


class CustomKnnClassImplementation(KNeighborsImplementation):
    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__(log)
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
        self.check_and_correct_k_value(train_data, KNeighborsClassifier)
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


class CustomKnnRegImplementation(KNeighborsImplementation):
    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__(log)
        if not params:
            self.model = KNeighborsRegressor()
        else:
            self.model = KNeighborsRegressor(**params)
        self.params = params

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """

        # Improve hyperparameters for model
        self.check_and_correct_k_value(train_data, KNeighborsRegressor)
        self.model.fit(train_data.features, train_data.target)
        return self.model
