from typing import Optional

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations. \
    implementation_interfaces import DataOperationImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.tasks import Task, TaskTypesEnum


class DecomposerImplementation(DataOperationImplementation):
    """ Base class for decomposing target. The idea is to find the difference
    between the actual and predicted values - the residuals. Then the residuals
    replace the original target.
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)

    def fit(self, input_data: InputData):
        """
        The decompose operation doesn't support fit method
        """
        pass

    def transform(self, input_data: InputData) -> OutputData:
        """
        Method for modifying input_data for predict stage
        :param input_data: data with features, target and ids

        :return input_data: data with transformed features attribute
        """
        raise NotImplementedError()

    @staticmethod
    def divide_inputs(input_data: InputData) -> (np.array, np.array):
        """ Method for dividing InputData into parts:
        first came from Model parent and second came from Data parent

        :param input_data: InputData object
        :return prev_prediction: data obtained from "Model parent" at the previous node
        :return prev_features: data obtained from "Data parent" at the previous node
        """

        features = np.array(input_data.features)
        # Array with masks
        features_mask = np.array(input_data.supplementary_data.compound_mask)
        unique_features_masks = np.unique(features_mask)

        if len(unique_features_masks) < 2:
            prefix = 'Decompose operation must have at least two parents nodes'
            raise ValueError(f'{prefix}, but got {len(unique_features_masks)}')

        # Get amount of nodes data already visited
        flow_lengths = input_data.supplementary_data.flow_mask

        # Find minimum and maximum of visited nodes and first indices of them
        min_flow_length_i = np.argmin(flow_lengths)
        max_flow_length_i = np.argmax(flow_lengths)

        # For case when data from "Model parent" and "Data parent" go through equal number of nodes
        if min_flow_length_i == max_flow_length_i:
            # Find data models
            model_parent, data_parent = input_data.supplementary_data.define_parents(unique_features_masks,
                                                                                     task=input_data.task.task_type)
        else:
            model_parent = features_mask[max_flow_length_i]
            data_parent = features_mask[min_flow_length_i]

        # Get prediction from "Model parent"
        prev_prediction_id = np.ravel(np.argwhere(features_mask == model_parent))
        prev_prediction = features[:, prev_prediction_id]

        # Get prediction from "Data parent" - it must be the last parent in parent list
        prev_features_id = np.ravel(np.argwhere(features_mask == data_parent))
        prev_features = features[:, prev_features_id]

        return prev_prediction, prev_features

    def get_params(self) -> OperationParameters:
        return OperationParameters()


class DecomposerRegImplementation(DecomposerImplementation):
    """ Class for decomposing target for regression task """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)

    def transform(self, input_data: InputData) -> OutputData:
        """
        Method for modifying input_data for predict stage
        :param input_data: data with features, target and ids

        :return input_data: data with transformed features attribute
        """
        prev_prediction, prev_features = self.divide_inputs(input_data)
        output_data = self._get_output_data(input_data, prev_features)
        return output_data

    def transform_for_fit(self, input_data: InputData) -> OutputData:
        """
        Method for modifying input_data for fit stage
        :param input_data: data with features, target and ids

        :return input_data: data with transformed features attribute
        """
        prev_prediction, prev_features = self.divide_inputs(input_data)

        # Target must be a column or table, not one-dimensional array
        target = np.array(input_data.target)
        if len(target.shape) < 2:
            target = target.reshape((-1, 1))

        diff = target - prev_prediction
        input_data.target = diff

        output_data = self._get_output_data(input_data, prev_features)
        return output_data

    def _get_output_data(self, input_data: InputData, prev_features: np.array) -> OutputData:
        output_data = self._convert_to_output(input_data, prev_features)
        # We decompose the target, so in the future we need to ignore
        output_data.supplementary_data.is_main_target = False
        return output_data


class DecomposerClassImplementation(DecomposerImplementation):
    """ Class for decomposing target for both binary and multiclass
    classification task
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)

    def transform(self, input_data: InputData) -> OutputData:
        """
        Method for modifying input_data for predict stage
        :param input_data: data with features, target and ids
        :return input_data: data with transformed features attribute
        """
        prev_prediction, prev_features = self.divide_inputs(input_data)
        output_data = self._get_output_data(input_data, prev_features)
        return output_data

    def transform_for_fit(self, input_data: InputData) -> OutputData:
        """
        Method for modifying input_data for fit stage
        :param input_data: data with features, target and ids
        :return input_data: data with transformed features attribute
        """
        prev_prediction, prev_features = self.divide_inputs(input_data)

        target = np.array(input_data.target)
        if len(target.shape) < 2:
            target = target.reshape((-1, 1))

        input_data.target = self._get_difference(target, prev_prediction)
        output_data = self._get_output_data(input_data, prev_features)
        return output_data

    def _get_output_data(self, input_data: InputData, prev_features: np.array) -> OutputData:
        regression_task = Task(TaskTypesEnum.regression)
        output_data = self._convert_to_output(input_data, prev_features)
        output_data.supplementary_data.is_main_target = False
        output_data.task = regression_task
        return output_data

    def _get_difference(self, target: np.array, prev_prediction: np.array) -> np.array:
        """ Calculates difference between predictions (probabilities) and target
        :param target: class labels
        :param prev_prediction: predictions from previous classification model
        :return diff: difference between probabilities of classes
        """
        classes = np.unique(target)
        if len(classes) > 2:
            diff = self._multi_difference(target, prev_prediction)
        else:
            diff = self._binary_difference(classes, target, prev_prediction)
        return diff

    @staticmethod
    def _binary_difference(classes: np.array, target: np.array, prev_prediction: np.array) -> np.array:
        """ Calculates difference between predictions (probabilities) and target
        for binary classification task
        :param classes: which classes are in the target
        :param target: class labels
        :param prev_prediction: predictions from previous classification model
        :return diff: difference between probabilities of classes
        """
        minus_class = np.min(classes)
        plus_class = np.max(classes)

        minus_ids = np.argwhere(target == minus_class)
        plus_ids = np.argwhere(target == plus_class)

        # Replace class labels with probabilities (0.0 or 1.0)
        bin_target = np.copy(target)
        bin_target[minus_ids] = 0.0
        bin_target[plus_ids] = 1.0

        diff = bin_target - prev_prediction

        return diff

    @staticmethod
    def _multi_difference(target: np.array, prev_prediction: np.array) -> np.array:
        """ Calculates difference between predictions (probabilities) and target
        for multiclass classification task

        :param target: class labels
        :param prev_prediction: predictions from previous classification model
        :return diff: difference between probabilities of classes
        """

        # Make one-hot encoding for target
        binary_enc = OneHotEncoder().fit_transform(target)
        probabilities_target = binary_enc.toarray()
        diff = probabilities_target - prev_prediction

        return diff
