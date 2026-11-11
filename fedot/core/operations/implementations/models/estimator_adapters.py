from typing import Callable, Optional

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC

from fedot.core.operations.implementations.base import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters


class KNeighborsImplementation(ModelImplementation):

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = None

    def predict(self, input_data):
        """ Method for making prediction

        :param input_data: data with features to process
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
            if new_k_value == 0:
                new_k_value = 1
            self.params.update(n_neighbors=new_k_value)
            self.model = model_impl(**self.params.to_dict())

            prefix = "n_neighbors of K-nn model was changed"
            self.log.info(f"{prefix} from {n_neighbors} to {new_k_value}")


class KnnClassificationAdapter(KNeighborsImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        params = round_n_neighbors(self.params.to_dict())
        self.model = KNeighborsClassifier(**params)
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


class KnnRegressionAdapter(KNeighborsImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        params = round_n_neighbors(self.params.to_dict())
        self.model = KNeighborsRegressor(**params)

    def fit(self, train_data):
        """ Method fit model on a dataset

        :param train_data: data to train the model
        """

        # Improve hyperparameters for model
        self.check_and_correct_k_value(train_data, KNeighborsRegressor)
        self.model.fit(train_data.features, train_data.target)
        return self.model


def round_n_neighbors(params: dict) -> dict:
    """ Convert n_neighbors into integer value. Operation work inplace. """
    if 'n_neighbors' in params:
        n_neighbors = round(params['n_neighbors'])
        if n_neighbors == 0:
            n_neighbors = 1
        params['n_neighbors'] = n_neighbors
    return params


class FedotSVCImplementation(ModelImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        if not self.params:
            default_params = {'kernel': 'linear',
                              'probability': True, 'class_weight': 'balanced'}
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

        prediction = np.nan_to_num(prediction)

        return prediction

    def predict_proba(self, input_data):
        """ Method make prediction with probabilities of classes

        :param input_data: data with features to process
        """
        prediction = self.model.predict_proba(input_data.features)

        prediction = np.nan_to_num(prediction)

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
        except (ValueError, IndexError):
            # Problem arise when features and target are "ideally" mapping
            # features [[1.0], [0.0], [0.0]] and target [[1], [0], [0]]
            new_solver = 'lsqr'
            self.log.debug(
                f'Change invalid parameter solver ({self.model.solver}) to {new_solver}')

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
