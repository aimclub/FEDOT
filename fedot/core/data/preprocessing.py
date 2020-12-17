import re

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

from fedot.core.repository.dataset_types import DataTypesEnum


class PreprocessingStrategy:
    def fit(self, data_to_fit) -> 'PreprocessingStrategy':
        raise NotImplementedError()

    def apply(self, data):
        raise NotImplementedError()


class Scaling(PreprocessingStrategy):
    def __init__(self, with_imputation=True):
        if with_imputation:
            self.default = ImputationStrategy()
        self.with_imputation = with_imputation
        self.scaler = preprocessing.StandardScaler()

    def fit(self, data_to_fit):
        if self.with_imputation:
            self.default.fit(data_to_fit)
            data_to_fit = self.default.apply(data_to_fit)

        data_to_fit = _expand_data(data_to_fit)
        self.scaler.fit(data_to_fit)
        return self

    def apply(self, data):
        if self.with_imputation:
            data = self.default.apply(data)

        data = _expand_data(data)
        resulted = self.scaler.transform(data)
        return resulted


class Normalization(PreprocessingStrategy):
    def __init__(self):
        self.default = ImputationStrategy()

    def fit(self, data_to_fit):
        self.default.fit(data_to_fit)
        return self

    def apply(self, data):
        modified = self.default.apply(data)
        resulted = preprocessing.normalize(modified)

        return resulted


class ImputationStrategy(PreprocessingStrategy):
    def __init__(self):
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    def fit(self, data_to_fit):
        self.imputer.fit(data_to_fit)
        return self

    def apply(self, data):
        modified = self.imputer.transform(data)
        return modified


class EmptyStrategy(PreprocessingStrategy):
    def fit(self, data_to_fit):
        return self

    def apply(self, data):
        result = np.asarray(data)
        if len(result.shape) == 1:
            result = np.expand_dims(result, axis=1)
        return result


class TsScalingStrategy(Scaling):
    def __init__(self):
        # the NaN preservation is important for the lagged ts features and forecasted ts
        super().__init__(with_imputation=False)


class TextPreprocessingStrategy(PreprocessingStrategy):
    def __init__(self):
        self.stemmer = PorterStemmer
        self.lemmanizer = WordNetLemmatizer
        self.lang = 'english'

    def fit(self, data_to_fit):
        return self

    def apply(self, data):
        clean_data = []
        for text in data:
            words = set(self._word_vectorize(text))
            without_stop_words = self._remove_stop_words(words)
            words = self._lemmatization(without_stop_words)
            words = [word for word in words if word.isalpha()]
            new_text = " ".join(words)
            new_text = self._clean_html_text(new_text)
            clean_data.append(new_text)
        return np.array(clean_data)

    def _word_vectorize(self, text):
        words = nltk.word_tokenize(text)

        return words

    def _remove_stop_words(self, words: set):
        stop_words = set(stopwords.words(self.lang))
        cleared_words = [word for word in words if word not in stop_words]

        return cleared_words

    def _stemming(self, words):
        stemmed_words = [self.stemmer().stem(word) for word in words]

        return stemmed_words

    def _lemmatization(self, words):
        # TODO pos
        lemmas = [self.lemmanizer().lemmatize(word) for word in words]

        return lemmas

    def _clean_html_text(self, raw_text):
        clean_pattern = re.compile('<.*?>')
        text = re.sub(clean_pattern, ' ', raw_text)

        return text


_preprocessing_for_input_data = {
    DataTypesEnum.ts: EmptyStrategy,
    DataTypesEnum.table: Scaling,
    DataTypesEnum.ts_lagged_table: TsScalingStrategy,
    DataTypesEnum.forecasted_ts: TsScalingStrategy,
}


def preprocessing_func_for_data(data: 'InputData', node: 'Node'):
    preprocessing_func = EmptyStrategy
    if 'without_preprocessing' not in node.model.metadata.tags:
        if node.manual_preprocessing_func:
            preprocessing_func = node.manual_preprocessing_func
        else:
            preprocessing_func = _preprocessing_for_input_data[data.data_type]
    return preprocessing_func


def _expand_data(data):
    if len(data.shape) == 1:
        data = data[:, None]
    return data
