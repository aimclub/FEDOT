from copy import copy
from typing import Optional

import numpy as np
from sklearn.utils import resample

from fedot.core.data.data import OutputData
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import DataOperationImplementation
from fedot.core.repository.dataset_types import DataTypesEnum


class ResampleImplementation(DataOperationImplementation):
    """"""

    def __init__(self, **params: Optional[dict]):
        super().__init__()

        if not params:
            self.balance = 'minority'
            self.replace = True
            self.n_samples = None
        else:
            self.balance = params.get('balance')
            self.replace = params.get("replace")
            self.n_samples = params.get("n_samples")

    def fit(self, input_data):
        """ Class doesn't support fit operation

        :param input_data: data with features, target and ids to process
        """
        pass

    def get_params(self):
        return {
            'balance': self.balance,
            'replace': self.replace,
            'n_samples': self.n_samples,
        }

    def transform(self, input_data, is_fit_pipeline_stage: Optional[bool]) -> DataTypesEnum.table:
        """

        :param input_data:
        :param is_fit_pipeline_stage:
        :return:
        """
        new_input_data = copy(input_data)

        if is_fit_pipeline_stage:
            features = new_input_data.features
            target = new_input_data.target

            if len(np.unique(target)) != 2:
                raise ValueError()

            unique_class, counts_class = np.unique(target, return_counts=True)

            if counts_class[0] == counts_class[1]:
                raise ValueError()

            elif counts_class[0] > counts_class[1]:
                minority_idx = np.where(target == unique_class[1])[0]
                majority_idx = np.where(target == unique_class[0])[0]
            else:
                minority_idx = np.where(target == unique_class[0])[0]
                majority_idx = np.where(target == unique_class[1])[0]

            minority_data = np.hstack((features[minority_idx], np.expand_dims(target[minority_idx], 1)))
            majority_data = np.hstack((features[majority_idx], np.expand_dims(target[majority_idx], 1)))

            if self.balance == 'minority':
                if self.n_samples is None or self.replace is False and self.n_samples > majority_data.shape[0]:
                    minority_data = resample(minority_data, replace=self.replace, n_samples=majority_data.shape[0])
                else:
                    minority_data = resample(minority_data, replace=self.replace, n_samples=self.n_samples)

            elif self.balance == 'majority':
                if self.n_samples is None or self.replace is False and self.n_samples > majority_data.shape[0]:
                    majority_data = resample(majority_data, replace=self.replace, n_samples=minority_data.shape[0])
                else:
                    majority_data = resample(majority_data, replace=self.replace, n_samples=minority_data.shape[0])
            else:
                raise ValueError()

            resample_data = np.concatenate((minority_data, majority_data), axis=0).transpose()

            output_data = OutputData(
                idx=np.arange(resample_data.shape[1]),
                features=input_data.features,
                predict=resample_data[:-1].transpose(),
                task=input_data.task,
                target=resample_data[-1],
                data_type=input_data.data_type,
                supplementary_data=input_data.supplementary_data)

        else:
            output_data = OutputData(
                idx=input_data.idx,
                features=input_data.features,
                predict=input_data.features,
                task=input_data.task,
                target=input_data.target,
                data_type=input_data.data_type,
                supplementary_data=input_data.supplementary_data)

        return output_data
