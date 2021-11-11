from typing import Union, List, Optional

import numpy as np
import pandas as pd

from fedot.core.data.data import InputData, data_type_is_table, data_has_missing_values, data_has_categorical_features
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log, default_log
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import \
    ImputationImplementation, OneHotEncodingImplementation
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.pipelines.node import Node

# The allowed percent of empty samples in features.
# Example: 30% objects in features are 'nan', then drop this feature from data.
ALLOWED_NAN_PERCENT = 0.3


class DataPreprocessing:
    """
    Class which contains methods for data preprocessing.
    The class performs two types of preprocessing: obligatory and optional

    obligatory - delete rows where nans in the target, remove features,
    which full of nans, delete extra_spaces
    optional - depends on what operations are in the pipeline, gap-filling
    is applied if there is no imputation operation in the pipeline, categorical
    encoding is applied if there is no encoder in the structure of the pipeline etc.
    """

    def __init__(self, log: Log):
        super().__init__()
        self.process_features = {}
        self.ids_relevant_features = []

        # Cannot be processed due to incorrect types or large amount of nans
        self.ids_incorrect_features = []
        self.log = log

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    def prepare_for_fit(self, pipeline, data: Union[InputData, MultiModalData]):
        """ Launch preprocessing operations (encoding, nan removing or imputation,
        speces deleting etc.) if it is necessary for pipeline fitting

        :param pipeline: pipeline to prepare data for
        :param data: data to preprocess
        """

        # Check if pipeline has encoders in its structure
        has_imputation_operation, has_encoder_operation = self.pipeline_encoders_validation(pipeline)

        if isinstance(data, InputData):
            if data_type_is_table(data):
                data = self._drop_features_full_of_nans(data)
                data = self._drop_rows_with_nan_in_target(data)
                data = self._clean_extra_spaces(data)

        elif isinstance(data, MultiModalData):
            for data_source_name, values in data.items():
                if data_type_is_table(values):
                    data[data_source_name] = self._drop_features_full_of_nans(values)
                    data[data_source_name] = self._drop_rows_with_nan_in_target(values)
                    data[data_source_name] = self._clean_extra_spaces(values)

        if data_has_missing_values(data) and not has_imputation_operation:
            data = self.apply_imputation(data)

        self.process_features = self._encode_data_for_fit(data)
        return data

    def prepare_for_predict(self, pipeline, data: Union[InputData, MultiModalData]):
        """ Launch preprocessing operations (encoding, nan removing or imputation, speces deleting etc.)
        if it is necessary for pipeline predict stage. Preprocessor should already must be fitted.

        :param pipeline: pipeline to prepare data for
        :param data: data to preprocess
        """
        # Check if pipeline has encoders in its structure
        has_imputation_operation, has_encoder_operation = self.pipeline_encoders_validation(
            pipeline)

        if isinstance(data, InputData):
            if data_type_is_table(data):
                self.take_only_correct_features(data)
                data = self._clean_extra_spaces(data)

        elif isinstance(data, MultiModalData):
            for data_source_name, values in data.items():
                if data_type_is_table(values):
                    self.take_only_correct_features(values)
                    data[data_source_name] = self._clean_extra_spaces(values)

        if data_has_missing_values(data) and not has_imputation_operation:
            data = self.apply_imputation(data)

        self._encode_data_for_predict(data, self.process_features)
        return data

    def take_only_correct_features(self, data: InputData):
        """ Take only correct features in the table """
        if len(self.ids_relevant_features) != 0:
            data.features = data.features[:, self.ids_relevant_features]

    def _drop_features_full_of_nans(self, data: InputData) -> InputData:
        """ Dropping features with more than 30% nan's

        :param data: data to transform
        :return: transformed data
        """
        features = data.features
        n_samples, n_columns = features.shape

        for i in range(n_columns):
            feature = features[:, i]
            if np.sum(pd.isna(feature)) / n_samples < ALLOWED_NAN_PERCENT:
                self.ids_relevant_features.append(i)
            else:
                self.ids_incorrect_features.append(i)

        self.take_only_correct_features(data)
        return data

    @staticmethod
    def _drop_rows_with_nan_in_target(data: InputData):
        """
        Drop rows where in target column there are nans
        :param data:
        :return:
        """
        features = data.features
        target = data.target

        if _is_any_nan(target):
            ids_with_nan = np.hstack(np.argwhere(_is_nan(target)))

            data.features = np.delete(features, ids_with_nan, axis=0)
            data.target = np.delete(features, ids_with_nan, axis=0)
            data.idx = np.delete(features, ids_with_nan, axis=0)

        return data

    @staticmethod
    def _clean_extra_spaces(data: InputData):
        """ Remove extra spaces from data.
            Transform cells in columns from ' x ' to 'x'
        """
        features = pd.DataFrame(data.features)
        features = features.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        data.features = np.array(features)
        return data

    @staticmethod
    def pipeline_encoders_validation(pipeline) -> (bool, bool):
        """
        Check whether Imputation and OneHotEncoder operation exist in pipeline.

        :param pipeline: pipeline to check
        :return (bool, bool): has Imputation and OneHotEncoder in pipeline
        """

        has_imputers, has_encoders = [], []

        def _check_imputer_encoder_recursion(root: Optional[Node], has_imputer: bool = False,
                                             has_encoder: bool = False):
            node_type = root.operation.operation_type
            if node_type == 'simple_imputation':
                has_imputer = True
            if node_type == 'one_hot_encoding':
                has_encoder = True

            if has_imputer and has_encoder:
                return has_imputer, has_encoder
            elif root.nodes_from is None:
                return has_imputer, has_encoder

            for node in root.nodes_from:
                answer = _check_imputer_encoder_recursion(node, has_imputer, has_encoder)
                if answer is not None:
                    imputer, encoder = answer
                    has_imputers.append(imputer)
                    has_encoders.append(encoder)

        _check_imputer_encoder_recursion(pipeline.root_node)

        if not has_imputers and not has_encoders:
            return False, False

        has_imputer = all(branch_has_imp is True for branch_has_imp in has_imputers)
        has_encoder = all(branch_has_imp is True for branch_has_imp in has_encoders)
        return has_imputer, has_encoder

    def apply_imputation(self, data: Union[InputData, MultiModalData]) -> Union[InputData, MultiModalData]:
        if isinstance(data, InputData):
            return self._apply_imputation_unidata(data)
        if isinstance(data, MultiModalData):
            for data_source_name, values in data.items():
                data[data_source_name].features = self._apply_imputation_unidata(values)
            return data
        raise ValueError(f"Data format is not supported.")

    @staticmethod
    def _apply_imputation_unidata(data: InputData):
        """ Fill in the gaps in the data inplace.

        :param data: data for fill in the gaps
        """
        imputer = ImputationImplementation()
        output_data = imputer.fit_transform(data)
        transformed = InputData(features=output_data.predict, data_type=output_data.data_type,
                                target=output_data.target, task=output_data.task, idx=output_data.idx)
        return transformed

    def _encode_data_for_fit(self, data: Union[InputData, MultiModalData]) -> \
            Union[List[DataOperationImplementation], DataOperationImplementation]:
        """ Encode categorical features to numerical. In additional,
            save encoders to use later for prediction data.

            :param data: data to transform
            :return encoders: operation preprocessing categorical features or list of it
        """

        encoders = None
        if isinstance(data, InputData):
            transformed, encoder = self._create_encoder(data)
            encoders = encoder
            data.features = transformed
        elif isinstance(data, MultiModalData):
            encoders = {}
            for data_source_name, values in data.items():
                if data_source_name.startswith('data_source_table'):
                    transformed, encoder = self._create_encoder(values)
                    if encoder is not None:
                        encoders[data_source_name] = encoder
                    data[data_source_name].features = transformed

        return encoders

    @staticmethod
    def _encode_data_for_predict(data: Union[InputData, MultiModalData],
                                 encoders: Union[dict, DataOperationImplementation]):
        """ Transformation the prediction data inplace. Use the same transformations as for the training data.

            :param data: data to transformation
            :param encoders: encoders f transformation
        """
        if encoders:
            if isinstance(data, InputData):
                transformed = encoders.transform(data, True).predict
                data.features = transformed
            elif isinstance(data, MultiModalData):
                for data_source_name, encoder in encoders.items():
                    transformed = encoder.transform(data[data_source_name], True).predict
                    data[data_source_name].features = transformed

    @staticmethod
    def _create_encoder(data: InputData):
        """ Fills in the gaps, converts categorical features using OneHotEncoder and create encoder.

        :param data: data to preprocess
        :return tuple(array, Union[OneHotEncodingImplementation, None]): tuple of transformed and [encoder or None]
        """

        encoder = None
        if data_has_categorical_features(data):
            encoder = OneHotEncodingImplementation()
            encoder.fit(data)
            transformed = encoder.transform(data, True).predict
        else:
            transformed = data.features

        return transformed, encoder


def _is_nan(array):
    return np.array([True if x is np.nan else False for x in array])


def _is_any_nan(array):
    if True in _is_nan(array):
        return True
    else:
        return False
