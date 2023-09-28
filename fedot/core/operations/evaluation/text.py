import warnings
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.data_operations.text_preprocessing import (
    TextCleanImplementation
)
from fedot.core.operations.evaluation.operation_implementations.data_operations.text_pretrained \
    import PretrainedEmbeddingsImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.utilities.random import ImplementationRandomStateHandler

warnings.filterwarnings("ignore", category=UserWarning)


class SkLearnTextVectorizeStrategy(EvaluationStrategy):
    _operations_by_types = {
        'tfidf': TfidfVectorizer,
        'cntvect': CountVectorizer,
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.vectorizer = self._convert_to_operation(operation_type)
        self.vectorizer = self.vectorizer(**self.params_for_fit.to_dict())

    def fit(self, train_data: InputData):
        features_list = self._convert_to_one_dim(train_data.features)

        with ImplementationRandomStateHandler(implementation=self.vectorizer):
            self.vectorizer.fit(features_list)

        return self.vectorizer

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        features_list = self._convert_to_one_dim(predict_data.features)
        predicted = trained_operation.transform(features_list).toarray()

        # Convert prediction to output (if it is required)
        converted = self._convert_to_output(predicted, predict_data)
        return converted

    @staticmethod
    def _convert_to_one_dim(array_with_text):
        """ Method converts array with text into one-dimensional list

        :param array_with_text: numpy array or list with text data
        :return features_list: one-dimensional list with text
        """
        features = np.ravel(np.array(array_with_text, dtype=str))
        features_list = list(features)
        return features_list

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))


class FedotTextPreprocessingStrategy(EvaluationStrategy):
    _operations_by_types = {
        'text_clean': TextCleanImplementation}

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.text_processor = self._convert_to_operation(operation_type)
        self.text_processor = self.text_processor(self.params_for_fit)

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided

        :param InputData train_data: data used for operation training
        :return: trained model
        """
        with ImplementationRandomStateHandler(implementation=self.text_processor):
            self.text_processor.fit(train_data)
        return self.text_processor

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        This method used for prediction of the target data during predict stage.

        :param trained_operation: trained operation object
        :param predict_data: data to predict
        :return OutputData: passed data with new predicted target
        """

        prediction = trained_operation.transform(predict_data)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        This method used for prediction of the target data during fit stage.

        :param trained_operation: trained operation object
        :param predict_data: data to predict
        :return OutputData: passed data with new predicted target
        """
        prediction = trained_operation.transform_for_fit(predict_data)
        converted = self._convert_to_output(prediction, predict_data)
        return converted


class GensimTextVectorizeStrategy(EvaluationStrategy):
    _operations_by_types = {
        'word2vec_pretrained': PretrainedEmbeddingsImplementation
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.vectorizer = self._convert_to_operation(operation_type)
        self.vectorizer = self.vectorizer(self.params_for_fit)

    def fit(self, train_data: InputData):
        """ Class doesn't support fit operation

        :param train_data: data with features, target and ids to process
        """
        with ImplementationRandomStateHandler(implementation=self.vectorizer):
            self.vectorizer.fit(train_data)
        return self.vectorizer

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        prediction = trained_operation.transform(predict_data)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        prediction = trained_operation.transform_for_fit(predict_data)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    @staticmethod
    def _convert_to_one_dim(array_with_text):
        """ Method converts array with text into one-dimensional list

        :param array_with_text: numpy array or list with text data
        :return features_list: one-dimensional list with text
        """
        features = np.ravel(np.array(array_with_text, dtype=str))
        features_list = list(features)
        return features_list

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))
