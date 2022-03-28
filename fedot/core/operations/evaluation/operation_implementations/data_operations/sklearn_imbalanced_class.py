from copy import copy
from typing import Optional

import numpy as np
from sklearn.utils import resample

from fedot.core.data.data import InputData, OutputData
from fedot.core.log import Log, default_log
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import DataOperationImplementation

GLOBAL_PREFIX = 'sklearn_imbalanced_class:'
# TODO: ResampleImplementation to multi-class imbalanced data


class ResampleImplementation(DataOperationImplementation):
    """ Implementation of imbalanced bin class transformation for
    classification task by using method from sklearn.utils.resample

    :param balance: balance strategy to transformation of data. Balance can be 'expand_minority' or 'reduce_majority'.
    In case of expand_minority elements of minor class are expanding to n_samples or to the shape of major class.
    If reduce_majority, elements of major class are reduce to n_samples or to the shape of minor class.
    :param replace: implements resampling with replacement. If False, this will implement (sliced) random permutations.
    :param n_samples: Number of samples to generate.
    If None number of samples will be equal to the shape of opposite selected transformed class.
    """

    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__()

        self.balance = params.get('balance')
        self.replace = params.get('replace')
        self.n_samples = params.get('n_samples')
        self.parameters_changed = False

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

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
            return tuple([params_dict, ['n_samples']])
        else:
            return params_dict

    def transform(self, input_data: Optional[InputData], is_fit_pipeline_stage: Optional[bool]) -> Optional[OutputData]:
        """ Transformed input data via selected balance strategy

        :param input_data: data with features, target and ids to process
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return: output_data: transformed input_data via strategy
        """

        new_input_data = copy(input_data)

        if is_fit_pipeline_stage:
            features = new_input_data.features
            target = new_input_data.target

            if len(np.unique(target)) != 2:
                # Imbalanced multi-class balancing is not supported.
                return self._return_source_data(input_data)

            unique_class, counts_class = np.unique(target, return_counts=True)

            if counts_class[0] == counts_class[1]:
                # Number of elements from each class are equal. Transformation is not required.
                return self._return_source_data(input_data)

            min_data, maj_data = self._get_data_by_target(features, target,
                                                          unique_class[0], unique_class[1],
                                                          counts_class[0], counts_class[1])

            self.n_samples = self._convert_to_absolute(min_data, maj_data)

            self.parameters_changed = self._check_and_correct_sample_size(min_data, maj_data)

            if self.balance == 'expand_minority':
                min_data = self._resample_data(min_data)

            elif self.balance == 'reduce_majority':
                maj_data = self._resample_data(maj_data)

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

        else:
            output_data = self._return_source_data(input_data)

        return output_data

    @staticmethod
    def _get_data_by_target(features, target, fst_class, snd_class, fst_class_values, snd_class_values):
        """ Unify features and target in one array and split into classes """
        if fst_class_values < snd_class_values:
            min_idx = np.where(target == fst_class)[0]
            maj_idx = np.where(target == snd_class)[0]
        else:
            min_idx = np.where(target == fst_class)[0]
            maj_idx = np.where(target == snd_class)[0]

        minority_data = np.hstack((features[min_idx], target[min_idx]))
        majority_data = np.hstack((features[maj_idx], target[maj_idx]))

        return minority_data, majority_data

    def _check_and_correct_sample_size(self, min_data, maj_data):
        """ Method checks if selected values in n_sample are incorrect - correct it otherwise

        :param min_data: minority data from input data
        :param maj_data: majority data from input data
        """
        prefix = "sklearn_imbalanced_class Warning: n_samples was changed"
        was_changed = False

        if self.replace is False and (self.n_samples > min_data.shape[0] or self.n_samples > maj_data.shape[0]):
            prev_n_samples = self.n_samples
            self.n_samples = self._set_sample_size(min_data, maj_data)
            self.log.info(f'{prefix[0]} from {prev_n_samples} to {self.n_samples}')
            was_changed = True

        return was_changed

    def _convert_to_absolute(self, min_data, maj_data):
        self.log.debug(f'{GLOBAL_PREFIX} n_samples was converted to absolute values')

        if self.balance == 'expand_minority':
            return round(min_data.shape[0] * self.n_samples)

        elif self.balance == 'reduce_majority':
            return round(maj_data.shape[0] * self.n_samples)

    def _convert_to_relative(self, min_data, maj_data):
        self.log.debug(f'{GLOBAL_PREFIX} n_samples was converted to relative values')

        if self.balance == 'expand_minority':
            return round(self.n_samples / min_data.shape[0], 2)

        elif self.balance == 'reduce_majority':
            return round(self.n_samples / maj_data.shape[0], 2)

    def _set_sample_size(self, min_data, maj_data):
        if self.balance == 'expand_minority':
            return maj_data.shape[0]

        elif self.balance == 'reduce_majority':
            return min_data.shape[0]

    def _resample_data(self, data):
        return resample(data, replace=self.replace, n_samples=self.n_samples)

    def _return_source_data(self, input_data):
        return OutputData(
            idx=input_data.idx,
            features=input_data.features,
            predict=input_data.features,
            task=input_data.task,
            target=input_data.target,
            data_type=input_data.data_type,
            supplementary_data=input_data.supplementary_data)
