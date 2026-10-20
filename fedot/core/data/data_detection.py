import re
import string
from abc import abstractmethod
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from fedot.utilities.custom_errors import AbstractMethodNotImplementError

ALLOWED_NAN_PERCENT = 0.9


class DataDetector:
    """
    Base class for automatic data type detectors.
    """

    @staticmethod
    @abstractmethod
    def prepare_multimodal_data(dataframe: pd.DataFrame, columns: List[str]) -> dict:
        """
        Prepares detected data in a form suitable for MultiModalData.

        :param dataframe: pandas DataFrame to process
        :param columns: list of columns selected by detector
        :return: dictionary with prepared data
        """
        raise AbstractMethodNotImplementError

    @staticmethod
    @abstractmethod
    def new_key_name(data_part_key: str) -> str:
        """
        Creates a source key for detected data.

        :param data_part_key: original data source key
        :return: new data source key
        """
        raise AbstractMethodNotImplementError


class TextDataDetector(DataDetector):
    """
    Class for detecting text data during its import.
    """
    STRUCTURED_THRESHOLD = 0.8
    MIN_UNIQUE_RATIO = 0.3
    MIN_VALUES = 3

    def define_text_columns(self, data_frame: pd.DataFrame) -> List[str]:
        """
        Finds columns containing natural-language text.

        :param data_frame: pandas DataFrame with data
        :return: list of text columns' names
        """
        text_columns = []
        for column_name in data_frame.columns:
            if self._column_contains_text(data_frame[column_name]):
                text_columns.append(column_name)
        return text_columns

    @staticmethod
    def is_full_of_nans(text_data: np.array) -> bool:
        """
        Checks whether text data contains too many missing values.

        :param text_data: numpy array with text data
        :return: True if fraction of NaN values exceeds allowed threshold
        """
        if np.sum(pd.isna(text_data)) / len(text_data) > ALLOWED_NAN_PERCENT:
            return True
        return False

    @staticmethod
    def prepare_multimodal_data(dataframe: pd.DataFrame, columns: List[str]) -> dict:
        """ Prepares MultiModal text data in a form of dictionary

        :param dataframe: pandas DataFrame to process
        :param columns: list of text columns' names

        :return multimodal_text_data: dictionary with numpy arrays of text data
        """
        multi_modal_text_data = {}

        for column_name in columns:
            text_feature = np.array(dataframe[column_name])
            multi_modal_text_data.update({column_name: text_feature})

        return multi_modal_text_data

    @staticmethod
    def new_key_name(data_part_key: str) -> str:
        """
        Creates a source key for text data.

        :param data_part_key: original data source key
        :return: key for text data source
        """
        return f'data_source_text/{data_part_key}'

    @staticmethod
    def _is_serialized_collection(value: str) -> bool:
        """Check whether a value is a serialized collection."""
        value = value.strip()
        return (value.startswith('{') and value.endswith('}')) or (
            value.startswith('[') and value.endswith(']')
        )

    @staticmethod
    def _is_slug(value: str) -> bool:
        return re.fullmatch(
            r'[a-z0-9]+(?:[-_][a-z0-9]+){2,}', value.strip(), re.IGNORECASE
        ) is not None

    @staticmethod
    def _is_person_name(value: str) -> bool:
        parts = value.strip().split()
        if not 2 <= len(parts) <= 3:
            return False
        stopwords = {'the', 'and', 'of', 'for', 'has', 'in', 'at', 'with', 'a', 'an'}
        if any(part.lower() in stopwords for part in parts):
            return False
        return all(re.fullmatch(r"[A-Z][a-zA-Z'-]*", part) for part in parts)

    @classmethod
    def _is_person_list(cls, value: str) -> bool:
        names = [name.strip() for name in value.split(',')]
        if len(names) < 3:
            return False
        return sum(cls._is_person_name(name) for name in names) / len(names) >= 0.8

    def analyze_column(self, column: pd.Series) -> dict:
        """Return the text prediction and features used to make it."""
        if not (pd.api.types.is_object_dtype(column)
                or pd.api.types.is_string_dtype(column)):
            return {'is_text': False, 'reason': 'NOT_STRING', 'features': {}}

        values = column.dropna().astype(str).str.strip()
        values = values[values != '']
        if len(values) < self.MIN_VALUES:
            return {'is_text': False, 'reason': 'TOO_FEW_VALUES', 'features': {}}
        if self._is_float_compatible(column):
            return {'is_text': False, 'reason': 'NUMERIC_STRINGS', 'features': {}}

        features = self.extract_features(values)
        for name in ('url_ratio', 'email_ratio', 'uuid_ratio', 'numeric_ratio',
                     'path_ratio', 'id_ratio'):
            if features[name] >= self.STRUCTURED_THRESHOLD:
                return {
                    'is_text': False,
                    'reason': f'STRUCTURED_{name.upper()}',
                    'features': features,
                }

        collection_ratio = values.map(self._is_serialized_collection).mean()
        slug_ratio = values.map(self._is_slug).mean()
        person_list_ratio = values.map(self._is_person_list).mean()
        person_ratio = values.map(self._is_person_name).mean()
        features.update({
            'collection_ratio': collection_ratio,
            'slug_ratio': slug_ratio,
            'person_list_ratio': person_list_ratio,
            'person_ratio': person_ratio,
        })

        for name, reason in (
            ('collection_ratio', 'SERIALIZED_COLLECTION'),
            ('slug_ratio', 'SLUG_VALUES'),
            ('person_list_ratio', 'PERSON_LIST'),
        ):
            if features[name] >= 0.6:
                return {'is_text': False, 'reason': reason, 'features': features}
        if person_ratio >= 0.8 and features['mean_word_count'] <= 3.2:
            return {'is_text': False, 'reason': 'PERSON_NAMES', 'features': features}

        mean_words = features['mean_word_count']
        mean_length = features['mean_length']
        alphabetic = features['alphabetic_ratio']
        unique_ratio = features['unique_ratio']
        vocabulary = features['vocabulary_size']

        long_text = (
            features['median_word_count'] >= 7
            and features['median_length'] >= 40
            and alphabetic >= 0.50
        ) or (mean_words >= 5 and mean_length >= 50 and alphabetic >= 0.52)
        if long_text:
            return {'is_text': True, 'reason': 'LONG_TEXT', 'features': features}

        short_text = (
            unique_ratio >= 0.45
            and mean_words >= 2.4
            and mean_length >= 12
            and alphabetic >= 0.48
            and vocabulary >= 15
        )
        if short_text:
            return {'is_text': True, 'reason': 'DIVERSE_SHORT_TEXT', 'features': features}

        repeated_text = (
            mean_words >= 3.4
            and mean_length >= 18
            and alphabetic >= 0.65
            and vocabulary >= 40
            and (unique_ratio >= 0.03
                 or features['function_word_ratio'] >= 0.02
                 or mean_words >= 3.8)
        )
        if repeated_text:
            return {
                'is_text': True,
                'reason': 'REPEATED_COMPOSITIONAL_TEXT',
                'features': features,
            }
        return {
            'is_text': False,
            'reason': 'CATEGORICAL_OR_UNCERTAIN',
            'features': features,
        }

    def _column_contains_text(self, column: pd.Series) -> bool:
        """Check whether a column contains natural-language text."""
        return bool(self.analyze_column(column)['is_text'])

    @staticmethod
    def _safe_std(values: pd.Series) -> float:
        """
        Calculates standard deviation without returning NaN.

        :param values: pandas Series with numeric values
        :return: standard deviation or zero if it cannot be calculated
        """
        result = values.std()

        if pd.isna(result):
            return 0.0

        return float(result)

    @staticmethod
    def _structured_ratios(values: pd.Series) -> dict:
        """
        Calculates fractions of structured values in a column.

        :param values: pandas Series with string values
        :return: dictionary with ratios of structured value types
        """
        patterns = {
            "url_ratio": re.compile(
                r"^https?://",
                re.IGNORECASE,
            ),

            "email_ratio": re.compile(
                r"^[\w.\-+]+@"
                r"[\w.\-]+\.\w+$"
            ),

            "uuid_ratio": re.compile(
                r"^[0-9a-fA-F]{8}-"
                r"[0-9a-fA-F]{4}-"
                r"[0-9a-fA-F]{4}-"
                r"[0-9a-fA-F]{4}-"
                r"[0-9a-fA-F]{12}$"
            ),

            "numeric_ratio": re.compile(
                r"^[+-]?"
                r"(?:\d+(?:\.\d*)?"
                r"|\.\d+)$"
            ),

            "path_ratio": re.compile(
                r"^(?:"
                r"[A-Za-z]:[\\/]"
                r"|/"
                r").+"
            ),

            "id_ratio": re.compile(
                r"^[A-Za-z]{0,10}"
                r"[-_]?"
                r"\d+"
                r"[A-Za-z0-9_-]*$"
            ),
        }

        result = {}

        for name, pattern in patterns.items():
            result[name] = values.apply(
                lambda value: bool(
                    pattern.match(value)
                )
            ).mean()

        return result

    @staticmethod
    def _normalize_pattern(value: str) -> str:
        """
        Normalizes variable parts of a string value.

        :param value: string value to normalize
        :return: normalized string pattern
        """
        value = value.lower()

        value = re.sub(
            r"[0-9a-f]{8}-"
            r"[0-9a-f]{4}-"
            r"[0-9a-f]{4}-"
            r"[0-9a-f]{4}-"
            r"[0-9a-f]{12}",
            "<uuid>",
            value,
        )

        value = re.sub(
            r"\d+",
            "<num>",
            value,
        )

        return value

    def extract_features(self, values: pd.Series) -> dict:
        """
        Extracts statistical and structural features from a text column.

        :param values: pandas Series with string values
        :return: dictionary with extracted column features
        """

        lengths = values.str.len()

        word_lists = values.apply(
            lambda value: re.findall(
                r"\b\w+\b",
                value.lower(),
            )
        )

        word_counts = word_lists.apply(
            len
        )

        tokens = [
            token
            for words in word_lists
            for token in words
        ]

        unique_tokens = set(tokens)

        full_text = "".join(
            values.tolist()
        )

        total_chars = max(
            len(full_text),
            1,
        )

        alphabetic_chars = sum(
            char.isalpha()
            for char in full_text
        )

        digit_chars = sum(
            char.isdigit()
            for char in full_text
        )

        whitespace_chars = sum(
            char.isspace()
            for char in full_text
        )

        punctuation_chars = sum(
            char in string.punctuation
            for char in full_text
        )

        features = {
            "rows":
                len(values),

            "unique_ratio":
                values.nunique()
                / len(values),

            "mean_length":
                lengths.mean(),

            "median_length":
                lengths.median(),

            "std_length":
                self._safe_std(lengths),

            "mean_word_count":
                word_counts.mean(),

            "median_word_count":
                word_counts.median(),

            "std_word_count":
                self._safe_std(word_counts),

            "single_word_ratio":
                (word_counts == 1).mean(),

            "multi_word_ratio":
                (word_counts > 1).mean(),

            "alphabetic_ratio":
                alphabetic_chars
                / total_chars,

            "digit_ratio":
                digit_chars
                / total_chars,

            "whitespace_ratio":
                whitespace_chars
                / total_chars,

            "punctuation_ratio":
                punctuation_chars
                / total_chars,

            "vocabulary_size":
                len(unique_tokens),

            "vocabulary_ratio":
                (
                    len(unique_tokens)
                    / len(tokens)
                    if tokens
                    else 0.0
                ),

            "mean_token_length":
                (
                    np.mean([
                        len(token)
                        for token in tokens
                    ])
                    if tokens
                    else 0.0
                ),

            "common_token_ratio":
                self._common_token_ratio(
                    tokens
                ),

            "pattern_unique_ratio":
                self._pattern_unique_ratio(
                    values
                ),

            "function_word_ratio":
                self._function_word_ratio(
                    tokens
                ),
        }

        features.update(
            self._structured_ratios(
                values
            )
        )

        features[
            "tfidf_vocabulary_size"
        ] = self._tfidf_vocabulary_size(
            values
        )

        return features

    @classmethod
    def _pattern_unique_ratio(cls, values: pd.Series) -> float:
        """
        Calculates fraction of unique normalized patterns.

        :param values: pandas Series with string values
        :return: fraction of unique normalized patterns
        """
        patterns = values.apply(
            cls._normalize_pattern
        )

        return (
                patterns.nunique()
                / len(patterns)
        )

    @staticmethod
    def _common_token_ratio(tokens: list) -> float:
        """
        Calculates fraction of repeated tokens in a column.

        :param tokens: list of tokens from column values
        :return: fraction of tokens repeated in the column
        """
        if not tokens:
            return 0.0

        counts = Counter(tokens)

        repeated = sum(
            count
            for count in counts.values()
            if count > 1
        )

        return (
                repeated
                / len(tokens)
        )

    @staticmethod
    def _function_word_ratio(tokens: list) -> float:
        """
        Calculates fraction of common function words.

        :param tokens: list of tokens from column values
        :return: fraction of function words
        """
        if not tokens:
            return 0.0

        function_words = {
            "the", "a", "an",
            "and", "or",
            "of", "to", "in",
            "for", "with",
            "on", "at",
            "from", "by",
            "is", "are",
            "was", "were",
            "this", "that",

            "и", "или",
            "в", "во",
            "на", "с", "со",
            "для", "из",
            "от", "до",
            "по", "к", "у",
            "о", "об",
            "что", "как",
            "это", "не",
        }

        count = sum(
            token in function_words
            for token in tokens
        )

        return count / len(tokens)

    @staticmethod
    def _tfidf_vocabulary_size(values: pd.Series) -> int:
        """
        Calculates size of TF-IDF vocabulary.

        :param values: pandas Series with string values
        :return: number of tokens in TF-IDF vocabulary
        """
        try:
            vectorizer = (
                TfidfVectorizer()
            )

            vectorizer.fit(values)

            return len(
                vectorizer.vocabulary_
            )

        except ValueError:
            return 0

    @staticmethod
    def _is_float_compatible(column: pd.Series) -> bool:
        """
        :param column: pandas series with data
        :return: True if column contains only float or nan values
        """
        nans_number = column.isna().sum()
        converted_column = pd.to_numeric(column, errors='coerce')
        result_nans_number = converted_column.isna().sum()
        failed_objects_number = result_nans_number - nans_number
        non_nan_all_objects_number = len(column) - nans_number
        failed_ratio = failed_objects_number / non_nan_all_objects_number
        return failed_ratio < 0.5

    def find_link_columns(
            self,
            data_frame: pd.DataFrame,
    ) -> List[str]:
        """
        Finds columns that mainly contain links.

        :param data_frame: pandas DataFrame to process
        :return: list of link columns' names
        """
        return [
            column_name
            for column_name in data_frame.columns
            if self._column_contains_links(
                data_frame[column_name]
            )
        ]

    @classmethod
    def _column_contains_links(
            cls,
            column: pd.Series,
    ) -> bool:
        """
        Checks whether a column mainly contains links.

        :param column: pandas Series with data
        :return: True if column contains links, False otherwise
        """
        if not (
                pd.api.types.is_object_dtype(column)
                or pd.api.types.is_string_dtype(column)
        ):
            return False

        values = (
            column
            .dropna()
            .astype(str)
            .str.strip()
        )

        values = values[
            values != ""
            ]

        if len(values) == 0:
            return False

        url_pattern = re.compile(
            r"^https?://",
            re.IGNORECASE,
        )

        url_ratio = values.apply(
            lambda value: bool(
                url_pattern.match(value)
            )
        ).mean()

        return (
                url_ratio
                >= cls.STRUCTURED_THRESHOLD
        )

    @staticmethod
    def find_sparse_columns(data_frame: pd.DataFrame) -> List[str]:
        """Find string columns with too many missing values."""
        sparse_columns = []
        for column_name in data_frame.columns:
            column = data_frame[column_name]
            if not (pd.api.types.is_object_dtype(column)
                    or pd.api.types.is_string_dtype(column)):
                continue
            if len(column) and column.isna().sum() / len(column) > ALLOWED_NAN_PERCENT:
                sparse_columns.append(column_name)
        return sparse_columns


class TimeSeriesDataDetector(DataDetector):
    """
    Class for detecting time series data during its import.
    """

    @staticmethod
    def prepare_multimodal_data(dataframe: pd.DataFrame, columns: List[str]) -> dict:
        """ Prepares MultiModal data for time series forecasting task in a form of dictionary

        :param dataframe: pandas DataFrame to process
        :param columns: column names, which should be used as features in forecasting

        :return multi_modal_ts_data: dictionary with numpy arrays
        """
        multi_modal_ts_data = {}
        for column_name in columns:
            feature_ts = np.array(dataframe[column_name])

            # Will be the same
            multi_modal_ts_data.update({column_name: feature_ts})

        return multi_modal_ts_data

    @staticmethod
    def new_key_name(data_part_key: str) -> str:
        """
        Creates a source key for time series data.

        :param data_part_key: original data source key
        :return: key for time series data source
        """
        if data_part_key == 'idx':
            return 'idx'
        return f'data_source_ts/{data_part_key}'

