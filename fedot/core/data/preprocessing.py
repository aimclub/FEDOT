import re

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn import preprocessing
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer

from fedot.core.repository.dataset_types import DataTypesEnum


class PreprocessingStrategy:
    def fit(self, data) -> 'PreprocessingStrategy':
        raise NotImplementedError()

    def apply(self, data):
        raise NotImplementedError()

    def fit_apply(self, data):
        self.fit(data)
        return self.apply(data)


class TextPreprocessingStrategy(PreprocessingStrategy):
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmanizer = WordNetLemmatizer()
        self.lang = 'english'
        self._download_nltk_resources()

    def fit(self, data_to_fit):
        return self

    def apply(self, data):
        clean_data = []
        for text in data:
            words = set(self._word_vectorize(text))
            without_stop_words = self._remove_stop_words(words)
            words = self._lemmatization(without_stop_words)
            words = [word for word in words if word.isalpha()]
            new_text = ' '.join(words)
            new_text = self._clean_html_text(new_text)
            clean_data.append(new_text)
        return np.array(clean_data)

    @staticmethod
    def _download_nltk_resources():
        for resource in ['punkt', 'stopwords', 'wordnet']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(f'{resource}')

    def _word_vectorize(self, text):
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

    def _clean_html_text(self, raw_text):
        clean_pattern = re.compile('<.*?>')
        text = re.sub(clean_pattern, ' ', raw_text)

        return text


def _expand_data(data):
    if len(data.shape) == 1:
        data = data[:, None]
    return data
