from typing import Optional

import numpy as np
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

    _MIN_REG_PARAM = 0.01

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = QuadraticDiscriminantAnalysis(**self.params.to_dict())

    def fit(self, train_data):
        """Fit QDA, correcting singular covariance parameters when necessary."""
        try:
            return super().fit(train_data)
        except np.linalg.LinAlgError:
            current_reg_param = self.params.get('reg_param') or 0.0
            regularized_param = max(current_reg_param, self._MIN_REG_PARAM)
            self.log.debug(f'Increase invalid reg_param ({current_reg_param}) to {regularized_param}')
            self.params.update(reg_param=regularized_param)
            self.model = QuadraticDiscriminantAnalysis(**self.params.to_dict())

        try:
            return super().fit(train_data)
        except np.linalg.LinAlgError:
            # Since scikit-learn 1.9, SVD cannot fit when a class has no more
            # samples than features. The eigen solver supports regularization
            # through shrinkage and is the recommended fallback in that case.
            if 'solver' not in self.model.get_params():
                raise
            self.log.debug('Change QDA solver to eigen with automatic shrinkage')
            self.params.update(solver='eigen', shrinkage='auto')
            self.model = QuadraticDiscriminantAnalysis(**self.params.to_dict())
            return super().fit(train_data)
