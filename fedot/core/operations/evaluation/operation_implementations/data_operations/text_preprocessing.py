import re

import nltk
import numpy as np

from typing import Optional

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.operations.evaluation.operation_implementations.\
    implementation_interfaces import DataOperationImplementation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


class TextCleanImplementation(DataOperationImplementation):
    """ Class for text cleaning (lemmatization and stemming) operation """

    def __init__(self, **params: Optional[dict]):
        self.stemmer = PorterStemmer()
        self.lemmanizer = WordNetLemmatizer()
        self._download_nltk_resources()

        if not params:
            self.lang = 'english'
        else:
            self.lang = params.get('language')
        super().__init__()

    def fit(self, input_data):
        """ Class doesn't support fit operation

        :param input_data: data with features, target and ids to process
        """
        pass

    def transform(self, input_data, is_fit_chain_stage: Optional[bool]):
        """ Method for transformation of the text data

        :param input_data: data with features, target and ids to process
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return output_data: output data with transformed features table
        """

        clean_data = []
        for text in input_data.features:
            words = set(self._word_vectorize(text))
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
        for resource in ['punkt', 'stopwords', 'wordnet']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(f'{resource}')

    @staticmethod
    def _word_vectorize(text):
        words = nltk.word_tokenize(text)

        return words

    def _remove_stop_words(self, words: set):
        stop_words = set(stopwords.words(self.lang))
        cleared_words = [word for word in words if word not in stop_words]

        return cleared_words

    def _stemming(self, words):
        stemmed_words = [self.stemmer.stem(word) for word in words]

        return stemmed_words

    def _lemmatization(self, words):
        # TODO pos
        lemmas = [self.lemmanizer.lemmatize(word) for word in words]

        return lemmas

    @staticmethod
    def _clean_html_text(raw_text):
        clean_pattern = re.compile('<.*?>')
        text = re.sub(clean_pattern, ' ', raw_text)

        return text

    def get_params(self):
        raise NotImplementedError()
