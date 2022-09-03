from copy import copy
from typing import Optional

import numpy as np
from sklearn.utils import resample

from fedot.core.data.data import InputData, OutputData
from fedot.core.log import default_log
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import (
    DataOperationImplementation
)

GLOBAL_PREFIX = 'sklearn_imbalanced_class:'


# TODO: ResampleImplementation to multi-class imbalanced data


class ResampleImplementation(DataOperationImplementation):
    """ Implementation of imbalanced bin class transformation for
    classification task by using method from sklearn.utils.resample

    :param balance: Data transformation strategy. Balance strategy can be 'expand_minority' or 'reduce_majority'.
    In case of expand_minority elements of minor class are expanding to n_samples.
    In otherwise with reduce_majority elements of major class are reducing to n_samples.
    :param replace: Implements resampling with replacement. If False, this will implement (sliced) random permutations.
    :param n_samples: Value of samples for transformation. The n_samples can take values in the range [0, 2].
    With n_samples = 0 nothing happens and data will remain the same.
    In case of n_samples = 1 means that both classes will be balanced and the shape of both will become equal.
    If n_samples < 1.0 means that the data of one class is getting closer to the shape of opposite class.
    In rest case, when n_samples > 1.0 the data of one class is surpass in shape of opposite class.
    If None numbers of samples will be equal to the shape of opposite selected transformed class.
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()

        self.balance = params.get('balance') if params.get('balance') is not None else 'expand_minority'
        self.replace = params.get('replace') if params.get('replace') is not None else False
        self.n_samples = params.get('n_samples') if params.get('n_samples') is not None else 1.0
        self.parameters_changed = False

        self.log = default_log(self)

    def fit(self, input_data: Optional[InputData]):
        """ Class doesn't support fit operation

        :param input_data: data with features, target and ids to process
        """
        pass

    def get_params(self):
        """ Method return parameters """
        params_dict = {
            'balance': self.balance,
            'replace': self.replace,
            'n_samples': self.n_samples,
        }
        if self.parameters_changed is True:
            return tuple([params_dict, ['replace']])
        else:
            return params_dict

    def transform(self, input_data: InputData) -> OutputData:
        """ Transformed input data via selected balance strategy for predict stage

        :param input_data: data with features, target and ids to process
        :return: output_data: transformed input_data via strategy
        """
        output_data = self._convert_to_output(input_data, input_data.features,
                                              data_type=input_data.data_type)
        return output_data

    def transform_for_fit(self, input_data: InputData) -> OutputData:
        """ Transformed input data via selected balance strategy for fit stage

        :param input_data: data with features, target and ids to process
        :return: output_data: transformed input_data via strategy
        """
        copied_data = copy(input_data)

        if len(np.unique(copied_data.target)) != 2:
            # Imbalanced multi-class balancing is not supported.
            return self._convert_to_output(input_data, input_data.features)

        unique_class, number_of_elements = np.unique(copied_data.target, return_counts=True)

        if number_of_elements[0] == number_of_elements[1]:
            # If number of elements of each class are equal that transformation is not required
            return self._convert_to_output(input_data, input_data.features)

        min_data, maj_data = self._get_data_by_target(copied_data.features, copied_data.target,
                                                      unique_class, number_of_elements)

        self.n_samples = self._convert_to_absolute(min_data, maj_data)

        self.parameters_changed = self._check_and_correct_replace_param(min_data, maj_data)

        if self.balance == 'expand_minority':
            prev_shape = min_data.shape
            min_data = self._resample_data(min_data)
            self.log.debug(f'{GLOBAL_PREFIX} According to {self.balance} data was changed from {prev_shape} to {min_data.shape}')

        elif self.balance == 'reduce_majority':
            prev_shape = maj_data.shape
            maj_data = self._resample_data(maj_data)
            self.log.debug(f'{GLOBAL_PREFIX} According to {self.balance} data was changed from {prev_shape} to {maj_data.shape}')

        self.n_samples = self._convert_to_relative(min_data, maj_data)

        transformed_data = np.concatenate((min_data, maj_data), axis=0).transpose()

        output_data = OutputData(
            idx=np.arange(transformed_data.shape[1]),
            features=input_data.features,
            predict=transformed_data[:-1].transpose(),
            task=input_data.task,
            target=transformed_data[-1],
            data_type=input_data.data_type,
            supplementary_data=input_data.supplementary_data)
        return output_data

    @staticmethod
    def _get_data_by_target(features: np.array, target: np.array, unique: np.array,
                            number_of_elements: np.array) -> np.array:
        """ Unify features and target in one array and split into classes """
        if number_of_elements[0] < number_of_elements[1]:
            min_idx = np.where(target == unique[0])[0]
            maj_idx = np.where(target == unique[1])[0]
        else:
            min_idx = np.where(target == unique[1])[0]
            maj_idx = np.where(target == unique[0])[0]

        minority_data = np.hstack((features[min_idx], target[min_idx].reshape(-1, 1)))
        majority_data = np.hstack((features[maj_idx], target[maj_idx].reshape(-1, 1)))

        return minority_data, majority_data

    def _check_and_correct_replace_param(self, min_data: np.array, maj_data: np.array) -> bool:
        """ Method checks if selected replace parameter is correct

        :param min_data: minority data from input data
        :param maj_data: majority data from input data
        """
        was_changed = False

        if self.replace is False:
            if self.balance == 'expand_minority' and self.n_samples >= min_data.shape[0]:
                self.replace, was_changed = True, True

            elif self.balance == 'reduce_majority' and self.n_samples >= maj_data.shape[0]:
                self.replace, was_changed = True, True

            self.log.debug(f'{GLOBAL_PREFIX} resample operation allow repeats in data')

        return was_changed

    def _convert_to_absolute(self, min_data: np.array, maj_data: np.array) -> float:
        self.log.debug(f'{GLOBAL_PREFIX} n_samples was converted to absolute values')
        difference = maj_data.shape[0] - min_data.shape[0]

        if self.balance == 'expand_minority':
            return round(difference * self.n_samples + min_data.shape[0])

        elif self.balance == 'reduce_majority':
            return round(maj_data.shape[0] - difference * self.n_samples)

    def _convert_to_relative(self, min_data: np.array, maj_data: np.array) -> float:
        self.log.debug(f'{GLOBAL_PREFIX} n_samples was converted to relative values')

        if self.balance == 'expand_minority':
            return round(self.n_samples / maj_data.shape[0], 2)

        elif self.balance == 'reduce_majority':
            return round(self.n_samples / min_data.shape[0], 2)

    def _resample_data(self, data: np.array):
        return resample(data, replace=self.replace, n_samples=self.n_samples)
