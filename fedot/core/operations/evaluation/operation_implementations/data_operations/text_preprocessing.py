import re
from typing import Optional

import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.utilities.requirements_notificator import warn_requirement

try:
    import nltk
except ModuleNotFoundError:
    warn_requirement('nltk')
    nltk = None

from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import (
    DataOperationImplementation
)
from fedot.core.repository.dataset_types import DataTypesEnum


class TextCleanImplementation(DataOperationImplementation):
    """ Class for text cleaning (lemmatization and stemming) operation """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.stemmer = nltk.stem.SnowballStemmer(language=self.language)
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self._download_nltk_resources()

    @property
    def language(self) -> str:
        return self.params.setdefault('language', 'english')

    def fit(self, input_data: InputData):
        """ Class doesn't support fit operation

        :param input_data: data with features, target and ids to process
        """
        pass

    def transform(self, input_data: InputData) -> OutputData:
        """ Method for transformation of the text data for predict stage

        :param input_data: data with features, target and ids to process
        :return output_data: output data with transformed features table
        """

        clean_data = []
        for text in input_data.features:
            text = str(text).lower()
            words = self._word_vectorize(text)
            without_stop_words = self._remove_stop_words(words)
            words = self._lemmatization(without_stop_words)
            words = [word for word in words if word.isalpha()]
            new_text = ' '.join(words)
            new_text = self._clean_html_text(new_text)
            clean_data.append(new_text)
        clean_data = np.array(clean_data)

        output_data = self._convert_to_output(input_data,
                                              clean_data,
                                              data_type=DataTypesEnum.text)
        return output_data

    @staticmethod
    def _download_nltk_resources():
        for resource in ['punkt']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(f'{resource}')
        for resource in ['stopwords', 'wordnet', 'omw-1.4']:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                nltk.download(f'{resource}')

    @staticmethod
    def _word_vectorize(text):
        if isinstance(text, np.ndarray):
            # occurrs when node with text preprocessing is not primary
            text = text[0]
        words = nltk.word_tokenize(text)

        return words

    def _remove_stop_words(self, words: set):
        stop_words = set(nltk.corpus.stopwords.words(self.language))
        cleared_words = [word for word in words if word not in stop_words]

        return cleared_words

    def _stemming(self, words):
        stemmed_words = [self.stemmer.stem(word) for word in words]

        return stemmed_words

    def _lemmatization(self, words):
        # TODO pos
        lemmas = [self.lemmatizer.lemmatize(word) for word in words]

        return lemmas

    @staticmethod
    def _clean_html_text(raw_text):
        clean_pattern = re.compile('<.*?>')
        text = re.sub(clean_pattern, ' ', raw_text)

        return text
