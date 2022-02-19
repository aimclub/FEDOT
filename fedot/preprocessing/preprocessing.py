from copy import copy
from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from fedot.core.data.data import InputData, data_type_is_table
from fedot.core.data.data import data_type_is_ts, OutputData
from fedot.core.data.data_preprocessing import data_has_categorical_features, data_has_missing_values, \
    replace_inf_with_nans
from fedot.core.data.data_preprocessing import find_categorical_columns
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import Log, default_log
from fedot.core.operations.evaluation.operation_implementations.data_operations.categorical_encoders import \
    OneHotEncodingImplementation, LabelEncodingImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import \
    ImputationImplementation
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.preprocessing.categorical import BinaryCategoricalPreprocessor
from fedot.preprocessing.data_types import TableTypesCorrector, NAME_CLASS_INT
# The allowed percent of empty samples in features.
# Example: 90% objects in features are 'nan', then drop this feature from data.
from fedot.preprocessing.structure import PipelineStructureExplorer, DEFAULT_SOURCE_NAME

ALLOWED_NAN_PERCENT = 0.9


class DataPreprocessor:
    """
    Class which contains methods for data preprocessing.
    The class performs two types of preprocessing: obligatory and optional

    obligatory - delete rows where nans in the target, remove features,
    which full of nans, delete extra_spaces
    optional - depends on what operations are in the pipeline, gap-filling
    is applied if there is no imputation operation in the pipeline, categorical
    encoding is applied if there is no encoder in the structure of the pipeline etc.
    """

    def __init__(self, log: Log = None):
        # There was performed encoding for string target column or not
        self.target_encoders = {}
        self.features_encoders = {}
        self.ids_relevant_features = {}

        # Cannot be processed due to incorrect types or large number of nans
        self.ids_incorrect_features = {}
        # Categorical preprocessor for binary categorical features
        self.binary_categorical_processors = {}
        self.types_correctors = {}
        self.structure_analysis = PipelineStructureExplorer()
        self.main_target_source_name = None
        self.log = log

        if not log:
            self.log = default_log(__name__)
        else:
            self.log = log

    def _init_supplementary_preprocessors(self, data: Union[InputData, MultiModalData]):
        """ Initialize helpers for preprocessor

        :param data: data class with input data for preprocessing
        """
        categorical_sources = list(self.binary_categorical_processors.keys())
        types_sources = list(self.types_correctors.keys())
        if len(categorical_sources) == len(types_sources) and len(types_sources) > 0:
            # Preprocessors have been already initialized
            return None
        self.helpers_were_initialized = True

        self.binary_categorical_processors = {}
        self.types_correctors = {}

        if isinstance(data, InputData):
            self.binary_categorical_processors.update({DEFAULT_SOURCE_NAME: BinaryCategoricalPreprocessor()})
            self.types_correctors.update({DEFAULT_SOURCE_NAME: TableTypesCorrector()})
        elif isinstance(data, MultiModalData):
            for data_source in list(data.keys()):
                self.binary_categorical_processors.update({data_source: BinaryCategoricalPreprocessor()})
                self.types_correctors.update({data_source: TableTypesCorrector()})
        else:
            raise ValueError('Unknown type of data.')

    def _init_main_target_source_name(self, multi_data: MultiModalData):
        """ Define for MultiModal data branches with main target and side ones """
        if self.main_target_source_name is not None:
            # Target name has been already defined
            return None

        for data_source_name, input_data in multi_data.items():
            if input_data.supplementary_data.is_main_target:
                self.main_target_source_name = data_source_name
                break

    def obligatory_prepare_for_fit(self, data: Union[InputData, MultiModalData]):
        """
        Perform obligatory preprocessing for pipeline fit method.
        It includes removing features full of nans, extra spaces in features deleting,
        drop rows where target cells are none
        """
        # TODO add advanced gapfilling for time series and advanced gap-filling
        self._init_supplementary_preprocessors(data)

        if isinstance(data, InputData):
            data = self._prepare_obligatory_unimodal_for_fit(data, source_name=DEFAULT_SOURCE_NAME)

        elif isinstance(data, MultiModalData):
            self._init_main_target_source_name(data)
            for data_source_name, values in data.items():
                data[data_source_name] = self._prepare_obligatory_unimodal_for_fit(values,
                                                                                   source_name=data_source_name)

        self.mark_as_preprocessed(data)
        return data

    def obligatory_prepare_for_predict(self, data: Union[InputData, MultiModalData]):
        """ Perform obligatory preprocessing for pipeline predict method """
        if isinstance(data, InputData):
            data = self._prepare_obligatory_unimodal_for_predict(data, source_name=DEFAULT_SOURCE_NAME)

        elif isinstance(data, MultiModalData):
            for data_source_name, values in data.items():
                data[data_source_name] = self._prepare_obligatory_unimodal_for_predict(values,
                                                                                       source_name=data_source_name)

        self.mark_as_preprocessed(data)
        return data

    def optional_prepare_for_fit(self, pipeline, data: Union[InputData, MultiModalData]):
        """ Launch preprocessing operations if it is necessary for pipeline fitting

        :param pipeline: pipeline to prepare data for
        :param data: data to preprocess
        """
        self._init_supplementary_preprocessors(data)

        if isinstance(data, InputData):
            self._prepare_optional_for_fit(pipeline, data, DEFAULT_SOURCE_NAME)
        else:
            # Multimodal data
            self._init_main_target_source_name(data)
            for data_source_name, values in data.items():
                self._prepare_optional_for_fit(pipeline, values, data_source_name)

        return data

    def optional_prepare_for_predict(self, pipeline, data: Union[InputData, MultiModalData]):
        """ Launch preprocessing operations if it is necessary for pipeline predict stage.
        Preprocessor should already must be fitted.

        :param pipeline: pipeline to prepare data for
        :param data: data to preprocess
        """
        if isinstance(data, InputData):
            self._prepare_optional_for_predict(pipeline, data, DEFAULT_SOURCE_NAME)
        else:
            # Multimodal data
            for data_source_name, values in data.items():
                self._prepare_optional_for_predict(pipeline, values, data_source_name)

        return data

    def take_only_correct_features(self, data: InputData, source_name: str):
        """ Take only correct features in the table """
        current_relevant_ids = self.ids_relevant_features[source_name]
        if len(current_relevant_ids) != 0:
            data.features = data.features[:, current_relevant_ids]

    def _prepare_obligatory_unimodal_for_fit(self, data: InputData, source_name: str) -> InputData:
        """ Method process InputData for pipeline fit method """
        if data.supplementary_data.was_preprocessed is True:
            # Preprocessing was already done - return data
            return data

        # Fix tables / time series sizes
        data = self._correct_shapes(data)

        if data_type_is_table(data):
            replace_inf_with_nans(data)

            # Find incorrect features which must be removed
            self._find_features_full_of_nans(data, source_name)
            self.take_only_correct_features(data, source_name)
            data = self._drop_rows_with_nan_in_target(data)

            # Column types processing - launch after correct features selection
            self.types_correctors[source_name].convert_data_for_fit(data)
            if self.types_correctors[source_name].target_converting_has_errors:
                data = self._drop_rows_with_nan_in_target(data)

            # Train Label Encoder for categorical target if necessary and apply it
            self._train_target_encoder(data, source_name)
            data.target = self._apply_target_encoding(data, source_name)

            data = self._clean_extra_spaces(data)
            # Wrap indices in numpy array
            data.idx = np.array(data.idx)

            # Process categorical features
            self.binary_categorical_processors[source_name].fit(data)
            data = self.binary_categorical_processors[source_name].transform(data)

        return data

    def _prepare_obligatory_unimodal_for_predict(self, data: InputData, source_name: str) -> InputData:
        """ Method process InputData for pipeline predict method """
        if data.supplementary_data.was_preprocessed is True:
            # Preprocessing was already done - return data
            return data

        data = self._correct_shapes(data)
        if data_type_is_table(data):
            replace_inf_with_nans(data)
            self.take_only_correct_features(data, source_name)

            # Perform preprocessing for types - launch after correct features selection
            self.types_correctors[source_name].convert_data_for_predict(data)

            data = self._clean_extra_spaces(data)
            # Wrap indices in numpy array
            data.idx = np.array(data.idx)
            data = self.binary_categorical_processors[source_name].transform(data)

            self._apply_categorical_encoding(data, source_name)
        return data

    def _prepare_optional_for_fit(self, pipeline, data: InputData, source_name: str):
        """ Perform optional preprocessing for unimodal data """
        if not data_type_is_table(data):
            return data

        if data_has_missing_values(data):
            # Data contains missing values
            has_imputer = self.structure_analysis.check_structure_by_tag(pipeline, tag_to_check='imputation',
                                                                         source_name=source_name)
            if has_imputer is False:
                self.apply_imputation(data)

        if data_has_categorical_features(data):
            # Data contains categorical features values
            has_encoder = self.structure_analysis.check_structure_by_tag(pipeline, tag_to_check='encoding',
                                                                         source_name=source_name)
            if has_encoder is False:
                self.one_hot_encoding_for_fit(data, source_name)

    def _prepare_optional_for_predict(self, pipeline, data: InputData, source_name: str):
        """ Perform optional preprocessing for predict stage """
        has_imputer = self.structure_analysis.check_structure_by_tag(pipeline, tag_to_check='imputation',
                                                                     source_name=source_name)
        if data_has_missing_values(data) and not has_imputer:
            data = self.apply_imputation(data)

        self._apply_categorical_encoding(data, source_name)

    def _find_features_full_of_nans(self, data: InputData, source_name: str):
        """ Find features with more than ALLOWED_NAN_PERCENT nan's

        :param data: data to find columns with nan values
        :param source_name: name of data source node
        """
        # Initialize empty lists to fill it with indices
        self.ids_relevant_features.update({source_name: []})
        self.ids_incorrect_features.update({source_name: []})

        features = data.features
        n_samples, n_columns = features.shape

        for i in range(n_columns):
            feature = features[:, i]
            if np.sum(pd.isna(feature)) / n_samples < ALLOWED_NAN_PERCENT:
                self.ids_relevant_features[source_name].append(i)
            else:
                self.ids_incorrect_features[source_name].append(i)

    @staticmethod
    def _drop_rows_with_nan_in_target(data: InputData):
        """ Drop rows where in target column there are nans """
        features = data.features
        target = data.target

        # Find indices of nans rows. Using pd instead of np because it is needed for string columns
        bool_target = np.array(pd.isna(target))
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

    def apply_imputation(self, data: Union[InputData, MultiModalData]) -> Union[InputData, MultiModalData]:
        if isinstance(data, InputData):
            return self._apply_imputation_unidata(data)
        if isinstance(data, MultiModalData):
            for data_source_name, values in data.items():
                data[data_source_name].features = self._apply_imputation_unidata(values)
            return data
        raise ValueError(f"Data format is not supported.")

    def one_hot_encoding_for_fit(self, data: Union[InputData], source_name: str = DEFAULT_SOURCE_NAME):
        """
        Encode categorical features to numerical. In additional,
        save encoders to use later for prediction data.

        :param data: data to transform
        :param source_name: name of data source node
        :return encoder: operation for preprocessing categorical features
        """

        encoder = self._create_onehot_encoder(data)

        encoder_output = encoder.transform(data, True)
        transformed = encoder_output.predict
        data.features = transformed
        data.supplementary_data = encoder_output.supplementary_data

        # Store encoder to make prediction in the future
        self.features_encoders.update({source_name: encoder})

    def label_encoding_for_fit(self, data: Union[InputData], source_name: str = DEFAULT_SOURCE_NAME):
        """
        Encode categorical features to numerical using LabelEncoder. In additional,
        save encoders to use later for prediction data.

        :param data: data to transform
        :param source_name: name of data source node
        :return encoder: operation for preprocessing categorical features
        """
        encoder = self._create_label_encoder(data)

        encoder_output = encoder.transform(data, True)
        transformed = encoder_output.predict
        data.features = transformed
        data.supplementary_data = encoder_output.supplementary_data

        # Store encoder to make prediction in the future
        self.features_encoders.update({source_name: encoder})

    def cut_dataset(self, data: InputData, border: int):
        """ Cutting large dataset based on border (number of objects to remain) """
        self.log.info("Cut dataset due to it size is large")
        data.shuffle()
        data.idx = data.idx[:border]
        data.features = data.features[:border]
        data.target = data.target[:border]

    @staticmethod
    def _apply_imputation_unidata(data: InputData):
        """ Fill in the gaps in the data inplace.

        :param data: data for fill in the gaps
        """
        imputer = ImputationImplementation()
        output_data = imputer.fit_transform(data)
        data.features = output_data.predict
        return data

    def _apply_categorical_encoding(self, data: InputData, source_name: str):
        """
        Transformation the prediction data inplace. Use the same transformations as for the training data.

        :param data: data to transformation
        :param source_name: name of data source node
        """
        if self.features_encoders.get(source_name) is None:
            # No encoding needed for current data
            return data

        # Check if column contains string objects
        features_types = data.supplementary_data.column_types['features']
        categorical_ids, non_categorical_ids = find_categorical_columns(data.features,
                                                                        features_types)
        if len(categorical_ids) > 0:
            # Perform encoding for categorical features
            encoder_output = self.features_encoders[source_name].transform(data, True)
            transformed = encoder_output.predict
            data.features = transformed

            data.supplementary_data = encoder_output.supplementary_data

    def _train_target_encoder(self, data: InputData, source_name: str):
        """ Convert string categorical target into integer column using LabelEncoder """
        categorical_ids, non_categorical_ids = find_categorical_columns(data.target,
                                                                        data.supplementary_data.column_types['target'])

        if len(categorical_ids) > 0:
            # Target is categorical
            target_encoder = LabelEncoder()
            target_encoder.fit(data.target)
            self.target_encoders.update({source_name: target_encoder})

    def _apply_target_encoding(self, data, source_name: str) -> np.array:
        """ Apply trained encoder for target column

        For example, target [['red'], ['green'], ['red']] will be converted into
        [[0], [1], [0]]
        """
        if self.target_encoders.get(source_name) is not None:
            # Target encoders have already been fitted
            data.supplementary_data.column_types['target'] = [NAME_CLASS_INT]
            encoded_target = self.target_encoders[source_name].transform(data.target)
            if len(encoded_target.shape) == 1:
                encoded_target = encoded_target.reshape((-1, 1))
            return encoded_target
        else:
            return data.target

    def apply_inverse_target_encoding(self, column_to_transform: np.array) -> np.array:
        """ Apply inverse Label Encoding operation for target column """
        main_target_source_name = self._determine_target_converter()

        if self.target_encoders.get(main_target_source_name) is not None:
            # Check if column contains string objects
            categorical_ids, non_categorical_ids = find_categorical_columns(column_to_transform)
            if len(categorical_ids) > 0:
                # There is no need to perform converting (it was performed already)
                return column_to_transform
            # It is needed to apply fitted encoder to apply inverse transformation
            transformed = self.target_encoders[main_target_source_name].inverse_transform(column_to_transform)

            # Convert one-dimensional array into column
            if len(transformed.shape) == 1:
                transformed = transformed.reshape((-1, 1))
            return transformed
        else:
            # Return source column
            return column_to_transform

    def _determine_target_converter(self):
        """
        For inverse target transformation there is a need to determine
        which target encoder to use (if there are several targets in
        single MultiModal pipeline).
        """
        # Choose data source node name with main target
        if self.main_target_source_name is None:
            return DEFAULT_SOURCE_NAME
        else:
            return self.main_target_source_name

    @staticmethod
    def _create_onehot_encoder(data: InputData) -> Union[OneHotEncodingImplementation, None]:
        """
        Fills in the gaps, converts categorical features using OneHotEncoder and create encoder.

        :param data: data to preprocess
        """

        encoder = None
        if data_has_categorical_features(data):
            encoder = OneHotEncodingImplementation()
            encoder.fit(data)

        return encoder

    @staticmethod
    def _create_label_encoder(data: InputData) -> Union[LabelEncodingImplementation, None]:
        """
        Fills in the gaps, converts categorical features using LabelEncoder and create encoder.

        :param data: data to preprocess
        :return tuple(array, Union[OneHotEncodingImplementation, None]): tuple of transformed and [encoder or None]
        """

        encoder = None
        if data_has_categorical_features(data):
            encoder = LabelEncodingImplementation()
            encoder.fit(data)

        return encoder

    @staticmethod
    def _correct_shapes(data: InputData) -> InputData:
        """
        Correct shapes of tabular data or time series: tabular must be
        two-dimensional arrays, time series - one-dim array
        """

        if data_type_is_table(data):
            if len(data.features.shape) < 2:
                data.features = data.features.reshape((-1, 1))
            if data.target is not None and len(data.target.shape) < 2:
                data.target = data.target.reshape((-1, 1))

        elif data.data_type == DataTypesEnum.ts:
            data.features = np.ravel(data.features)

        return data

    @staticmethod
    def convert_indexes_for_fit(pipeline, data: Union[InputData, MultiModalData]):
        if isinstance(data, MultiModalData):
            for data_source_name, values in data.items():
                if data_type_is_ts(data[data_source_name]):
                    data[data_source_name] = data[data_source_name].convert_non_int_indexes_for_fit(pipeline)
            return data
        elif data_type_is_ts(data):
            return data.convert_non_int_indexes_for_fit(pipeline)
        else:
            return data

    @staticmethod
    def convert_indexes_for_predict(pipeline, data: Union[InputData, MultiModalData]):
        if isinstance(data, MultiModalData):
            for data_source_name, values in data.items():
                if data_type_is_ts(data[data_source_name]):
                    data[data_source_name] = data[data_source_name].convert_non_int_indexes_for_predict(pipeline)
            return data
        elif data_type_is_ts(data):
            return data.convert_non_int_indexes_for_predict(pipeline)
        else:
            return data

    @staticmethod
    def restore_index(input_data: InputData, result: OutputData):
        if isinstance(input_data, InputData):
            if input_data.supplementary_data.non_int_idx is not None:
                result.idx = copy(input_data.supplementary_data.non_int_idx)
                result.supplementary_data.non_int_idx = copy(input_data.idx)
        return result

    @staticmethod
    def mark_as_preprocessed(data: Union[InputData, MultiModalData]):
        if isinstance(data, InputData):
            data.supplementary_data.was_preprocessed = True
        else:
            # Multimodal data
            for data_source_name, values in data.items():
                values.supplementary_data.was_preprocessed = True


def merge_preprocessors(api_preprocessor: DataPreprocessor,
                        pipeline_preprocessor: DataPreprocessor) -> DataPreprocessor:
    """
    Combining two preprocessor objects. One is the preprocessor from the API,
    the second is the preprocessor from the obtained pipeline
    """
    # Take all obligatory data preprocessing from API
    new_data_preprocessor = api_preprocessor

    # Update optional preprocessing (take it from obtained pipeline)
    new_data_preprocessor.structure_analysis = pipeline_preprocessor.structure_analysis
    if len(new_data_preprocessor.features_encoders.keys()) == 0:
        # Store features encoder from obtained pipeline because in API there are no encoding
        new_data_preprocessor.features_encoders = pipeline_preprocessor.features_encoders
    return new_data_preprocessor


def update_indices_for_time_series(test_data: Union[InputData, MultiModalData]):
    """ Replace indices for time series for predict stage """
    if test_data.task.task_type != TaskTypesEnum.ts_forecasting:
        return test_data

    if isinstance(test_data, MultiModalData):
        # Process multimodal data - change indices in every data block
        for data_source_name, input_data in test_data.items():
            forecast_len = input_data.task.task_params.forecast_length
            if forecast_len < len(input_data.idx):
                # Indices incorrect - there is a need to reassign them
                last_id = len(input_data.idx)
                input_data.idx = np.arange(last_id, last_id + input_data.task.task_params.forecast_length)
    else:
        # Simple input data
        forecast_len = test_data.task.task_params.forecast_length
        if forecast_len < len(test_data.idx):
            last_id = len(test_data.idx)
            test_data.idx = np.arange(last_id, last_id + test_data.task.task_params.forecast_length)
    return test_data
