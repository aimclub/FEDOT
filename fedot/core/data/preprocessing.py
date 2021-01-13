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


class ImputationStrategy(PreprocessingStrategy):
    def __init__(self):
        self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    def fit(self, data):
        self.imputer.fit(data)
        return self

    def apply(self, data):
        try:
            modified = self.imputer.transform(data)
        except NotFittedError:
            modified = self.imputer.fit_transform(data)

        return modified


class EmptyStrategy(PreprocessingStrategy):
    def fit(self, data):
        return self

    def apply(self, data):
        result = np.asarray(data)
        if len(result.shape) == 1:
            result = np.expand_dims(result, axis=1)
        return result


class Scaling(PreprocessingStrategy):
    def __init__(self):
        self.scaler = preprocessing.StandardScaler()

    def fit(self, data):
        data = _expand_data(data)
        self.scaler.fit(data)
        return self

    def apply(self, data):
        data = _expand_data(data)
        try:
            resulted = self.scaler.transform(data)
        except NotFittedError:
            resulted = self.scaler.fit_transform(data)

        return resulted


class ScalingWithImputation(Scaling):
    def __init__(self):
        super(ScalingWithImputation, self).__init__()
        self.default = ImputationStrategy()

    def fit(self, data):
        self.default.fit(data)
        data = self.default.apply(data)
        return super(ScalingWithImputation, self).fit(data)

    def apply(self, data):
        data = self.default.apply(data)
        return super(ScalingWithImputation, self).apply(data)


class Normalization(Scaling):
    def __init__(self):
        super(Normalization, self).__init__()
        self.scaler = preprocessing.MinMaxScaler()


class NormalizationWithImputation(ScalingWithImputation):
    def __init__(self):
        super(NormalizationWithImputation, self).__init__()
        self.default = ImputationStrategy()
        self.scaler = preprocessing.MinMaxScaler()


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


_preprocessing_for_input_data = {
    DataTypesEnum.ts: EmptyStrategy,
    DataTypesEnum.table: ScalingWithImputation,
    DataTypesEnum.ts_lagged_table: Scaling,
    DataTypesEnum.forecasted_ts: Scaling,
}

_label_for_preprocessing_strategy = {
    'empty': EmptyStrategy,
    'normalization_with_imputation': NormalizationWithImputation,
    'normalization': Normalization,
    'scaling_with_imputation': ScalingWithImputation,
    'scaling': Scaling}


def preprocessing_strategy_label_by_class(target_strategy: PreprocessingStrategy):
    for label, strategy in _label_for_preprocessing_strategy.items():
        if isinstance(target_strategy, strategy):
            return label
    return None


def preprocessing_strategy_class_by_label(string: str) -> [PreprocessingStrategy, None]:
    preprocessing_strategy = _label_for_preprocessing_strategy.get(string)

    if preprocessing_strategy:
        return preprocessing_strategy
    return None


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
