import numpy as np

from typing import Optional
from sklearn.preprocessing import OneHotEncoder
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.operations.evaluation.operation_implementations.\
    implementation_interfaces import DataOperationImplementation


class DecomposerImplementation(DataOperationImplementation):
    """ Base class for decomposing target. The idea is to find the difference
    between the actual and predicted values - the residuals. Then the residuals
    replace the original target.
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.params = None

    def fit(self, input_data):
        """
        The decompose operation doesn't support fit method
        """
        pass

    def transform(self, input_data, is_fit_chain_stage: Optional[bool]):
        """
        Method for modifying input_data
        :param input_data: data with features, target and ids
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return input_data: data with transformed features attribute
        """
        raise NotImplementedError()

    @staticmethod
    def divide_inputs(input_data):
        """ Method for dividing InputData into parts:
        first came from Model parent and second came from Data parent

        :param input_data: InputData object
        :return prev_prediction: data obtained from "Model parent" at the previous node
        :return prev_features: data obtained from "Data parent" at the previous node
        """

        features = np.array(input_data.features)
        # Array with masks
        features_mask = np.array(input_data.supplementary_data.get_compound_mask())

        # Get amount of nodes data already visited
        flow_lengths = input_data.supplementary_data.get_flow_mask()

        # Find minimum and maximum of visited nodes and first indices of them
        min_flow_length_i = np.argmin(flow_lengths)
        max_flow_length_i = np.argmax(flow_lengths)

        # Get prediction from "Model parent"
        model_parent = features_mask[max_flow_length_i]
        prev_prediction_id = np.ravel(np.argwhere(features_mask == model_parent))
        prev_prediction = features[:, prev_prediction_id]

        # Get prediction from "Data parent" - it must be the last parent in parent list
        data_parent = features_mask[min_flow_length_i]
        prev_features_id = np.ravel(np.argwhere(features_mask == data_parent))
        prev_features = features[:, prev_features_id]

        return prev_prediction, prev_features

    def get_params(self):
        return None


class DecomposerRegImplementation(DecomposerImplementation):
    """ Class for decomposing target for regression task """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.params = None

    def transform(self, input_data, is_fit_chain_stage: Optional[bool]):
        """
        Method for modifying input_data
        :param input_data: data with features, target and ids
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return input_data: data with transformed features attribute
        """

        # Get inputs from Data and Model parent
        prev_prediction, prev_features = self.divide_inputs(input_data)

        if is_fit_chain_stage:
            # Target must be a column or table, not one-dimensional array
            target = np.array(input_data.target)
            if len(target.shape) < 2:
                target = target.reshape((-1, 1))

            # Calculate difference between prediction of model and current target
            diff = target - prev_prediction

            # Update target
            input_data.target = diff
            # Create OutputData
            output_data = self._convert_to_output(input_data, prev_features)
            # We decompose the target, so in the future we need to ignore
            output_data.supplementary_data.is_main_target = False
        else:
            # For predict stage there is no need to worry about target
            output_data = self._convert_to_output(input_data, prev_features)
            output_data.supplementary_data.is_main_target = False

        return output_data


class DecomposerClassImplementation(DecomposerImplementation):
    """ Class for decomposing target for both binary and multiclass
    classification task
    """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        self.params = None

    def transform(self, input_data, is_fit_chain_stage: Optional[bool]):
        """
        Method for modifying input_data
        :param input_data: data with features, target and ids
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return input_data: data with transformed features attribute
        """

        # Task since that model - regression
        regression_task = Task(TaskTypesEnum.regression)

        # Get inputs from Data and Model parent
        prev_prediction, prev_features = self.divide_inputs(input_data)

        if is_fit_chain_stage:
            # Target must be a column or table, not one-dimensional array
            target = np.array(input_data.target)
            if len(target.shape) < 2:
                target = target.reshape((-1, 1))

            classes = np.unique(target)
            if len(classes) > 2:
                diff = self._multi_difference(target, prev_prediction)
            else:
                # Binary classification task
                diff = self._binary_difference(classes, target, prev_prediction)

            # Update target
            input_data.target = diff
            # Create OutputData
            output_data = self._convert_to_output(input_data, prev_features)
            # We decompose the target, so in the future we need to ignore
            output_data.supplementary_data.is_main_target = False
            output_data.task = regression_task
        else:
            # For predict stage there is no need to worry about target
            output_data = self._convert_to_output(input_data, prev_features)
            output_data.supplementary_data.is_main_target = False
            output_data.task = regression_task

        return output_data

    @staticmethod
    def _binary_difference(classes, target, prev_prediction):
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
    def _multi_difference(target, prev_prediction):
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
