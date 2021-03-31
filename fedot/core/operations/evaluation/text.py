import warnings

import numpy as np

from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.\
    data_operations.text_preprocessing import TextCleanImplementation

warnings.filterwarnings("ignore", category=UserWarning)


class SkLearnTextVectorizeStrategy(EvaluationStrategy):
    __operations_by_types = {
        'tfidf': TfidfVectorizer,
        'cntvect': CountVectorizer,
    }

    def __init__(self, operation_type: str, params: Optional[dict] = None):
        self.vectorizer = self._convert_to_operation(operation_type)
        self.params = params
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        features_list = self._convert_to_one_dim(train_data.features)

        vectorizer = self.vectorizer().fit(features_list)

        return vectorizer

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool) -> OutputData:

        features_list = self._convert_to_one_dim(predict_data.features)
        predicted = trained_operation.transform(features_list).toarray()

        # Convert prediction to output (if it is required)
        converted = self._convert_to_output(predicted, predict_data)
        return converted

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain TextVectorize strategy for {operation_type}')

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


class CustomTextPreprocessingStrategy(EvaluationStrategy):
    __operations_by_types = {
        'text_clean': TextCleanImplementation}

    def __init__(self, operation_type: str, params: Optional[dict] = None):
        self.text_processor = self._convert_to_operation(operation_type)
        self.params = params
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided

        :param InputData train_data: data used for operation training
        :return: trained model
        """
        if self.params:
            text_processor = self.text_processor(**self.params_for_fit)
        else:
            text_processor = self.text_processor()

        text_processor.fit(train_data)
        return text_processor

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool) -> OutputData:
        """
        This method used for prediction of the target data.

        :param trained_operation: trained operation object
        :param predict_data: data to predict
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return OutputData: passed data with new predicted target
        """

        prediction = trained_operation.transform(predict_data,
                                                 is_fit_chain_stage)
        # Convert prediction to output (if it is required)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain custom text preprocessing strategy for {operation_type}')
