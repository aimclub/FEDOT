import re

import nltk
import numpy as np

from typing import Optional
from datetime import timedelta
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.operations.evaluation.evaluation import EvaluationStrategy
from fedot.core.operations.evaluation.operation_realisations.\
    data_operations.text_preprocessing import TextClean


class TextVectorizeStrategy(EvaluationStrategy):
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

        # Wrap prediction as features for next level
        converted = OutputData(idx=predict_data.idx,
                               features=predict_data.features,
                               predict=predicted,
                               task=predict_data.task,
                               target=predict_data.target,
                               data_type=DataTypesEnum.table)
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


class TextPreprocessingStrategy(EvaluationStrategy):
    __operations_by_types = {
        'text_clean': TextClean}

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
        return prediction

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain TextPreprocessing strategy for {operation_type}')

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))
