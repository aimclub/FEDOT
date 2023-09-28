from abc import ABC, abstractmethod
from inspect import stack
from typing import Dict, List, Union, TYPE_CHECKING

import numpy as np
from sklearn.preprocessing import LabelEncoder

from fedot.core.data.data import InputData, OutputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.operations.evaluation.operation_implementations.data_operations.categorical_encoders import (
    LabelEncodingImplementation,
    OneHotEncodingImplementation
)
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import (
    ImputationImplementation
)
from fedot.preprocessing.categorical import BinaryCategoricalPreprocessor
from fedot.preprocessing.data_types import TableTypesCorrector
from fedot.preprocessing.structure import DEFAULT_SOURCE_NAME

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline


class BasePreprocessor(ABC):
    """
    Class which contains abstract methods for data preprocessing.
    """

    def __init__(self):
        # There was performed encoding for string target column or not
        self.target_encoders: Dict[str, LabelEncoder] = {}
        self.features_encoders: Dict[str, Union[OneHotEncodingImplementation, LabelEncodingImplementation]] = {}
        self.use_label_encoder: bool = False
        self.features_imputers: Dict[str, ImputationImplementation] = {}
        self.ids_relevant_features: Dict[str, List[int]] = {}

        # Cannot be processed due to incorrect types or large number of nans
        self.ids_incorrect_features: Dict[str, List[int]] = {}
        # Categorical preprocessor for binary categorical features
        self.binary_categorical_processors: Dict[str, BinaryCategoricalPreprocessor] = {}
        self.types_correctors: Dict[str, TableTypesCorrector] = {}
        self.main_target_source_name = None

    @abstractmethod
    def obligatory_prepare_for_fit(self, data: Union[InputData, MultiModalData]) -> Union[InputData, MultiModalData]:
        """
        Performs obligatory preprocessing for pipeline's fit method.

        Args:
            data: data to be preprocessed

        Returns:
            preprocessed data
        """
        raise NotImplementedError(f'Method {stack()[0][3]} is not implemented in {self.__class__}')

    @abstractmethod
    def obligatory_prepare_for_predict(self, data: Union[InputData, MultiModalData]) -> Union[InputData,
                                                                                              MultiModalData]:
        """
        Performs obligatory preprocessing for pipeline's predict method.

        Args:
            data: data to be preprocessed

        Returns:
            preprocessed data
        """
        raise NotImplementedError(f'Method {stack()[0][3]} is not implemented in {self.__class__}')

    @abstractmethod
    def optional_prepare_for_fit(self, pipeline, data: Union[InputData, MultiModalData]) -> Union[InputData,
                                                                                                  MultiModalData]:
        """
        Launches preprocessing operations if it is necessary for pipeline fitting.

        Args:
            pipeline: pipeline defining whether to make optional preprocessing
            data: data to be preprocessed

        Returns:
            preprocessed data
        """
        raise NotImplementedError(f'Method {stack()[0][3]} is not implemented in {self.__class__}')

    @abstractmethod
    def optional_prepare_for_predict(self, pipeline, data: Union[InputData, MultiModalData]) -> Union[InputData,
                                                                                                      MultiModalData]:
        """
        Launches preprocessing operations if it is necessary for pipeline predict stage.
        Preprocessor must be already fitted.

        Args:
            pipeline: pipeline defining whether to use optional preprocessing
            data: data to be preprocessed

        Returns:
            preprocessed data
        """
        raise NotImplementedError(f'Method {stack()[0][3]} is not implemented in {self.__class__}')

    @abstractmethod
    def label_encoding_for_fit(self, data: InputData, source_name: str = DEFAULT_SOURCE_NAME):
        """
        Encodes categorical features to numerical using LabelEncoder.
        In addition, saves encoders to use later for data prediction.

        Args:
            data: data to transform
            source_name: name of data source node
        """
        raise NotImplementedError(f'Method {stack()[0][3]} is not implemented in {self.__class__}')

    @abstractmethod
    def cut_dataset(self, data: InputData, border: int):
        """
        Cuts large dataset based on border (number of objects to remain).

        Args:
            data: data to be cut
            border: number of objects to keep
        """
        raise NotImplementedError(f'Method {stack()[0][3]} is not implemented in {self.__class__}')

    @abstractmethod
    def apply_inverse_target_encoding(self, column_to_transform: np.ndarray) -> np.ndarray:
        """
        Applies inverse label encoding operation for target column if needed

        Args:
            column_to_transform: column to be encoded

        Returns:
            encoded or untouched column
        """
        raise NotImplementedError(f'Method {stack()[0][3]} is not implemented in {self.__class__}')

    @abstractmethod
    def convert_indexes_for_fit(self, pipeline: 'Pipeline', data: Union[InputData, MultiModalData]) -> \
            Union[InputData, MultiModalData]:
        """
        Converts provided data's and pipeline's indexes for fit

        Args:
            pipeline: pipeline whose indexes should be converted
            data: data whose indexes should be converted

        Returns:
            converted data
        """
        raise NotImplementedError(f'Method {stack()[0][3]} is not implemented in {self.__class__}')

    @abstractmethod
    def convert_indexes_for_predict(self, pipeline, data: Union[InputData, MultiModalData]) -> \
            Union[InputData, MultiModalData]:
        """
        Converts provided data's and pipeline's indexes for predict

        Args:
            pipeline: pipeline whose indexes should be converted
            data: data whose indexes should be converted

        Returns:
            converted data
        """
        raise NotImplementedError(f'Method {stack()[0][3]} is not implemented in {self.__class__}')

    @abstractmethod
    def restore_index(self, input_data: InputData, result: OutputData) -> OutputData:
        """
        restores index from ``input_data`` into ``result``
        Args:
            input_data: data to take the index from
            result: data to store index into

        Returns:
            ``result`` with restored index
        """
        raise NotImplementedError(f'Method {stack()[0][3]} is not implemented in {self.__class__}')

    @abstractmethod
    def update_indices_for_time_series(self, test_data: Union[InputData, MultiModalData]) -> Union[InputData,
                                                                                                   MultiModalData]:
        """
        Replaces indices for time series for predict stage

        Args:
            test_data: data for replacing the indices

        Returns:
            data with the replaced indices
        """
        raise NotImplementedError(f'Method {stack()[0][3]} is not implemented in {self.__class__}')

    @staticmethod
    def mark_as_preprocessed(data: Union[InputData, MultiModalData], *, is_obligatory: bool = True):
        """
        Marks provided ``data`` as preprocessed with ``type`` method,
            so it won't be preprocessed in further steps of an algorithm.

        Args:
            data: data to be marked
            is_obligatory: was the data obligatorily or optionally preprocessed
        """
        values = [data] if isinstance(data, InputData) else data.values()
        for input_data in values:
            if is_obligatory:
                input_data.supplementary_data.obligatorily_preprocessed = True
            else:
                input_data.supplementary_data.optionally_preprocessed = True

    @staticmethod
    def merge_preprocessors(api_preprocessor: 'BasePreprocessor',
                            pipeline_preprocessor: 'BasePreprocessor') -> 'BasePreprocessor':
        """
        Combines two preprocessor's objects.

        Args:
            api_preprocessor: the one from the API
            pipeline_preprocessor: the one from the obtained pipeline

        Returns:
            merged preprocessor
        """
        # Take all obligatory data preprocessing from API
        new_data_preprocessor = api_preprocessor

        # Update optional preprocessing (take it from obtained pipeline)
        if not new_data_preprocessor.features_encoders:
            # Store features encoder from obtained pipeline because in API there are no encoding
            new_data_preprocessor.features_encoders = pipeline_preprocessor.features_encoders
        return new_data_preprocessor
