from typing import Union, List, Optional

import numpy as np
import pandas as pd

from fedot.core.data.data import InputData, data_type_is_table, data_has_missing_values, data_has_categorical_features
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log, default_log
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import \
    ImputationImplementation, OneHotEncodingImplementation, replace_inf_with_nans
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation
from fedot.core.pipelines.node import Node
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.preprocessing.categorical import CategoricalPreprocessor

# The allowed percent of empty samples in features.
# Example: 30% objects in features are 'nan', then drop this feature from data.
ALLOWED_NAN_PERCENT = 0.3


class DataPreprocessor:
    """
    Class which contains methods for data preprocessing.
    The class performs two types of preprocessing: obligatory and optional

    obligatory - delete rows where nans in the target, remove features,
    which full of nans, delete extra_spaces
    optional - depends on what operations are in the pipeline, gap-filling
    is applied if there is no imputation operation in the pipeline, categorical
    encoding is applied if there is no encoder in the structure of the pipeline etc.

    TODO refactor for multimodal data preprocessing
    """

    def __init__(self, log: Log = None):
        self.process_features = {}
        self.ids_relevant_features = []

        # Cannot be processed due to incorrect types or large amount of nans
        self.ids_incorrect_features = []
        self.categorical_processor = CategoricalPreprocessor()
        self.log = log

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    def obligatory_prepare_for_fit(self, data: Union[InputData, MultiModalData]):
        """
        Perform obligatory preprocessing for pipeline fit method.
        It includes removing features full of nans, extra spaces in features deleting,
        drop rows where target cells are none
        """
        # TODO add advanced gapfilling for time series

        if isinstance(data, InputData):
            data = self._prepare_unimodal_for_fit(data)

        elif isinstance(data, MultiModalData):
            for data_source_name, values in data.items():
                data[data_source_name] = self._prepare_unimodal_for_fit(values)

        return data

    def obligatory_prepare_for_predict(self, data: Union[InputData, MultiModalData]):
        """ Perform obligatory preprocessing for pipeline predict method """
        if isinstance(data, InputData):
            data = self._prepare_unimodal_for_predict(data)

        elif isinstance(data, MultiModalData):
            for data_source_name, values in data.items():
                data[data_source_name] = self._prepare_unimodal_for_predict(values)

        return data

    def optional_prepare_for_fit(self, pipeline, data: Union[InputData, MultiModalData]):
        """ Launch preprocessing operations if it is necessary for pipeline fitting

        :param pipeline: pipeline to prepare data for
        :param data: data to preprocess
        """

        # Check if pipeline has encoders in its structure
        has_imputation_operation, has_encoder_operation = self.pipeline_encoders_imputers_validation(pipeline)

        if data_has_missing_values(data) and not has_imputation_operation:
            data = self.apply_imputation(data)

        self.process_features = self._encode_data_for_fit(data)
        return data

    def optional_prepare_for_predict(self, pipeline, data: Union[InputData, MultiModalData]):
        """ Launch preprocessing operations if it is necessary for pipeline predict stage.
        Preprocessor should already must be fitted.

        :param pipeline: pipeline to prepare data for
        :param data: data to preprocess
        """
        # Check if pipeline has encoders in its structure
        has_imputation_operation, has_encoder_operation = self.pipeline_encoders_imputers_validation(
            pipeline)

        if data_has_missing_values(data) and not has_imputation_operation:
            data = self.apply_imputation(data)

        self._encode_data_for_predict(data, self.process_features)
        return data

    def take_only_correct_features(self, data: InputData):
        """ Take only correct features in the table """
        if len(self.ids_relevant_features) != 0:
            data.features = data.features[:, self.ids_relevant_features]

    def _prepare_unimodal_for_fit(self, data: InputData) -> InputData:
        """ Method process InputData for pipeline fit method """
        if data.supplementary_data.was_preprocessed is True:
            # Preprocessing was already done - return data
            return data

        # Fix tables / time series sizes
        data = self._correct_shapes(data)

        if data_type_is_table(data):
            data = replace_inf_with_nans(data)
            data = self._drop_features_full_of_nans(data)
            data = self._drop_rows_with_nan_in_target(data)
            data = self._clean_extra_spaces(data)

            # Process categorical features
            self.categorical_processor.fit(data)
            data = self.categorical_processor.transform(data)
        return data

    def _prepare_unimodal_for_predict(self, data: InputData) -> InputData:
        """ Method process InputData for pipeline predict method """

        data = self._correct_shapes(data)
        if data_type_is_table(data):
            data = replace_inf_with_nans(data)
            self.take_only_correct_features(data)

            data = self._clean_extra_spaces(data)
            data = self.categorical_processor.transform(data)

        return data

    def _drop_features_full_of_nans(self, data: InputData) -> InputData:
        """ Dropping features with more than ALLOWED_NAN_PERCENT nan's

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
        """ Drop rows where in target column there are nans  """
        features = data.features
        target = data.target

        # Find indices of nans rows
        bool_target = np.isnan(target)
        number_nans_per_rows = bool_target.sum(axis=1)

        # Ids of rows which doesn't contain nans in target
        non_nan_row_ids = np.ravel(np.argwhere(number_nans_per_rows == 0))

        if len(non_nan_row_ids) == 0:
            raise ValueError('Data contains too much nans in the target column(s)')
        data.features = features[non_nan_row_ids, :]
        data.target = target[non_nan_row_ids, :]
        data.idx = np.array(data.idx)[non_nan_row_ids]

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
    def pipeline_encoders_imputers_validation(pipeline) -> (bool, bool):
        """
        Check whether Imputation or OneHotEncoder operation exist in pipeline.

        :param pipeline: pipeline to check
        :return (bool, bool): has Imputation and OneHotEncoder in pipeline
        """
        has_operations = []
        def _check_operation_recursion(root: Optional[Node],
                                       operation_name: str,
                                       has_operation: bool = False):
            # Check if the operation in the node
            node_type = root.operation.operation_type
            if node_type == operation_name:
                has_operation = True

            if has_operation:
                return has_operation
            elif root.nodes_from is None:
                return has_operation

            for node in root.nodes_from:
                answer = _check_operation_recursion(node, operation_name, has_operation)
                if answer is not None:
                    has_operations.append(answer)

        _check_operation_recursion(pipeline.root_node, operation_name='simple_imputation')
        has_imputer = all(branch_has_imp is True for branch_has_imp in has_operations)
        return has_imputer

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
        """
        Encode categorical features to numerical. In additional,
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
        """
        Transformation the prediction data inplace. Use the same transformations as for the training data.

        :param data: data to transformation
        :param encoders: encoders for transformation
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
        """
        Fills in the gaps, converts categorical features using OneHotEncoder and create encoder.

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

    @staticmethod
    def _correct_shapes(data: InputData) -> InputData:
        """
        Correct shapes of tabular data or time series: tabular must be
        two-dimensional arrays, time series - one-dim array
        """

        if data.data_type is DataTypesEnum.table:
            if len(data.features.shape) < 2:
                data.features = data.features.reshape((-1, 1))
            if data.target is not None and len(data.target.shape) < 2:
                data.target = data.target.reshape((-1, 1))

        elif data.data_type is DataTypesEnum.ts:
            data.features = np.ravel(data.features)
            if data.target is not None:
                data.target = np.ravel(data.target)

        return data
