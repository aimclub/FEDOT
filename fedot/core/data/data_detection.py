import re
from abc import abstractmethod
from typing import List

import numpy as np
import pandas as pd

from fedot.utilities.custom_errors import AbstractMethodNotImplementError


ALLOWED_NAN_PERCENT = 0.9

URL_PATTERN = re.compile(
    r'^(?:(?:https?://|www\.)'
    r'(?:[a-z0-9-]+\.)+[a-z]{2,63}'
    r'(?::\d{2,5})?(?:[/?#]\S*)?'
    r'|(?:[a-z0-9-]+\.)+[a-z]{2,63}/\S+)$',
    re.IGNORECASE,
)


class DataDetector:
    """Base class for automatic data type detectors."""

    @staticmethod
    @abstractmethod
    def prepare_multimodal_data(
            dataframe: pd.DataFrame,
            columns: List[str],
    ) -> dict:
        """Prepare data for a multimodal source."""
        raise AbstractMethodNotImplementError

    @staticmethod
    @abstractmethod
    def new_key_name(data_part_key: str) -> str:
        """Return the name of the data source."""
        raise AbstractMethodNotImplementError


class TextDataDetector(DataDetector):
    """Detect columns containing natural-language text."""

    STRUCTURED_THRESHOLD = 0.8
    COLLECTION_THRESHOLD = 0.6
    MIN_VALUES = 3
    SHORT_UNIQUE_RATIO = 0.45
    SHORT_MIN_VOCABULARY = 15

    def define_text_columns(
            self,
            data_frame: pd.DataFrame,
    ) -> List[str]:
        """Find text columns in a dataframe."""
        return [
            name
            for name in data_frame.columns
            if self._column_contains_text(data_frame[name])
        ]

    @staticmethod
    def is_full_of_nans(text_data: np.ndarray) -> bool:
        """Check whether the fraction of missing values is too high."""
        if len(text_data) == 0:
            return False

        return bool(
            pd.isna(text_data).mean() > ALLOWED_NAN_PERCENT
        )

    @staticmethod
    def prepare_multimodal_data(
            dataframe: pd.DataFrame,
            columns: List[str],
    ) -> dict:
        """Return text columns as separate data sources."""
        return {
            name: np.array(dataframe[name])
            for name in columns
        }

    @staticmethod
    def new_key_name(data_part_key: str) -> str:
        """Return the name of a text data source."""
        return f'data_source_text/{data_part_key}'

    @staticmethod
    def _is_serialized_collection(value: str) -> bool:
        """Check whether a value looks like a serialized collection."""
        value = value.strip()

        return (
            (value.startswith('{') and value.endswith('}'))
            or (value.startswith('[') and value.endswith(']'))
        )

    @staticmethod
    def _is_slug(value: str) -> bool:
        return re.fullmatch(
            r'[a-z0-9]+(?:[-_][a-z0-9]+){2,}',
            value.strip(),
            re.IGNORECASE,
        ) is not None

    @staticmethod
    def _looks_like_name(value: str) -> bool:
        """Check the shape of a name without assuming a specific script."""
        words = value.strip().split()

        if not 2 <= len(words) <= 5:
            return False

        stopwords = {
            'the', 'and', 'of', 'for',
            'in', 'at', 'with',
        }

        if any(word.lower() in stopwords for word in words):
            return False

        for word in words:
            normalized = (
                word.replace('-', '')
                .replace("'", '')
                .replace('’', '')
                .replace('.', '')
            )

            if not normalized.isalpha() or not word[0].isupper():
                return False

        return True

    @classmethod
    def _is_person_list(cls, value: str) -> bool:
        """Check whether a value resembles a list of person names."""
        names = [
            name.strip()
            for name in value.split(',')
        ]

        if len(names) < 3:
            return False

        name_ratio = sum(
            cls._looks_like_name(name)
            for name in names
        )

        return name_ratio / len(names) >= 0.8

    def analyze_column(self, column: pd.Series) -> dict:
        """Return the prediction, rejection reason and column features."""
        if not (
            pd.api.types.is_object_dtype(column)
            or pd.api.types.is_string_dtype(column)
        ):
            return {
                'is_text': False,
                'reason': 'NOT_STRING',
                'features': {},
            }

        values = column.dropna().astype(str).str.strip()
        values = values[values != '']

        if len(values) < self.MIN_VALUES:
            return {
                'is_text': False,
                'reason': 'TOO_FEW_VALUES',
                'features': {},
            }

        if self._is_float_compatible(column):
            return {
                'is_text': False,
                'reason': 'NUMERIC_STRINGS',
                'features': {},
            }

        features = self.extract_features(values)

        for name in (
            'url_ratio',
            'email_ratio',
            'uuid_ratio',
            'numeric_ratio',
            'path_ratio',
            'id_ratio',
        ):
            if features[name] >= self.STRUCTURED_THRESHOLD:
                return {
                    'is_text': False,
                    'reason': f'STRUCTURED_{name.upper()}',
                    'features': features,
                }

        features.update({
            'collection_ratio': values.map(
                self._is_serialized_collection
            ).mean(),
            'slug_ratio': values.map(self._is_slug).mean(),
            'person_list_ratio': values.map(
                self._is_person_list
            ).mean(),
        })

        for name, reason in (
            ('collection_ratio', 'SERIALIZED_COLLECTION'),
            ('slug_ratio', 'SLUG_VALUES'),
            ('person_list_ratio', 'PERSON_LIST'),
        ):
            if features[name] >= self.COLLECTION_THRESHOLD:
                return {
                    'is_text': False,
                    'reason': reason,
                    'features': features,
                }

        mean_words = features['mean_word_count']
        mean_length = features['mean_length']
        alphabetic = features['alphabetic_ratio']
        unique_ratio = features['unique_ratio']
        vocabulary = features['vocabulary_size']

        long_text = (
            (
                features['median_word_count'] >= 7
                and features['median_length'] >= 40
                and alphabetic >= 0.50
            )
            or (
                mean_words >= 5
                and mean_length >= 50
                and alphabetic >= 0.52
            )
        )

        if long_text:
            return {
                'is_text': True,
                'reason': 'LONG_TEXT',
                'features': features,
            }

        short_text = (
            unique_ratio >= self.SHORT_UNIQUE_RATIO
            and mean_words >= 2.4
            and mean_length >= 12
            and alphabetic >= 0.48
            and vocabulary >= self.SHORT_MIN_VOCABULARY
        )

        if short_text:
            return {
                'is_text': True,
                'reason': 'DIVERSE_SHORT_TEXT',
                'features': features,
            }

        repeated_text = (
            mean_words >= 3.4
            and mean_length >= 18
            and alphabetic >= 0.65
            and vocabulary >= 40
            and (
                unique_ratio >= 0.03
                or features['function_word_ratio'] >= 0.02
                or mean_words >= 3.8
            )
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
    def _structured_ratios(values: pd.Series) -> dict:
        """Calculate proportions of structured values in a column."""
        patterns = {
            'url_ratio': URL_PATTERN,
            'email_ratio': re.compile(
                r'[\w.+-]+@[\w.-]+\.\w+'
            ),
            'uuid_ratio': re.compile(
                r'[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}',
                re.IGNORECASE,
            ),
            'numeric_ratio': re.compile(
                r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)'
            ),
            'path_ratio': re.compile(
                r'(?:[A-Za-z]:[\\/]|/).+'
            ),
            'id_ratio': re.compile(
                r'[A-Za-z]{0,10}[-_]?\d+[A-Za-z0-9_-]*'
            ),
        }

        return {
            name: values.map(
                lambda value: pattern.fullmatch(value) is not None
            ).mean()
            for name, pattern in patterns.items()
        }

    def extract_features(self, values: pd.Series) -> dict:
        """Calculate statistics used by the text detection rules."""
        lengths = values.str.len()

        word_lists = values.map(
            lambda value: re.findall(
                r'\b\w+\b',
                value.lower(),
            )
        )

        word_counts = word_lists.map(len)

        tokens = [
            token
            for words in word_lists
            for token in words
        ]

        text = ''.join(values.tolist())
        total_chars = max(len(text), 1)

        features = {
            'unique_ratio': values.nunique() / len(values),
            'mean_length': lengths.mean(),
            'median_length': lengths.median(),
            'mean_word_count': word_counts.mean(),
            'median_word_count': word_counts.median(),
            'alphabetic_ratio': (
                sum(char.isalpha() for char in text)
                / total_chars
            ),
            'vocabulary_size': len(set(tokens)),
            'function_word_ratio': self._function_word_ratio(
                tokens
            ),
        }

        features.update(self._structured_ratios(values))

        return features

    @staticmethod
    def _function_word_ratio(tokens: list) -> float:
        """Calculate the fraction of common function words."""
        if not tokens:
            return 0.0

        function_words = {
            'the', 'a', 'an',
            'and', 'or',
            'of', 'to', 'in',
            'for', 'with',
            'on', 'at',
            'from', 'by',
            'is', 'are',
            'was', 'were',
            'this', 'that',
            'и', 'или',
            'в', 'во',
            'на', 'с', 'со',
            'для', 'из',
            'от', 'до',
            'по', 'к', 'у',
            'о', 'об',
            'что', 'как',
            'это', 'не',
        }

        count = sum(
            token in function_words
            for token in tokens
        )

        return count / len(tokens)

    @staticmethod
    def _is_float_compatible(column: pd.Series) -> bool:
        """Check whether the column mostly contains numeric values."""
        non_missing = column.dropna()

        if non_missing.empty:
            return False

        converted = pd.to_numeric(
            non_missing,
            errors='coerce',
        )

        return bool(converted.isna().mean() < 0.5)

    def find_link_columns(
            self,
            data_frame: pd.DataFrame,
    ) -> List[str]:
        """Find columns whose values are predominantly URLs."""
        return [
            name
            for name in data_frame.columns
            if self._column_contains_links(data_frame[name])
        ]

    @classmethod
    def _column_contains_links(
            cls,
            column: pd.Series,
    ) -> bool:
        """Check whether a column predominantly contains URLs."""
        if not (
            pd.api.types.is_object_dtype(column)
            or pd.api.types.is_string_dtype(column)
        ):
            return False

        values = column.dropna().astype(str).str.strip()
        values = values[values != '']

        if values.empty:
            return False

        url_ratio = values.map(
            lambda value: URL_PATTERN.fullmatch(value) is not None
        ).mean()

        return bool(url_ratio >= cls.STRUCTURED_THRESHOLD)

    @staticmethod
    def find_sparse_columns(data_frame: pd.DataFrame) -> List[str]:
        """Find string columns with too many missing values."""
        sparse_columns = []

        for name in data_frame.columns:
            column = data_frame[name]

            if not (
                pd.api.types.is_object_dtype(column)
                or pd.api.types.is_string_dtype(column)
            ):
                continue

            if (
                len(column)
                and column.isna().mean() > ALLOWED_NAN_PERCENT
            ):
                sparse_columns.append(name)

        return sparse_columns


class TimeSeriesDataDetector(DataDetector):
    """Detect time series data during import."""

    @staticmethod
    def prepare_multimodal_data(
            dataframe: pd.DataFrame,
            columns: List[str],
    ) -> dict:
        """Return time series columns as separate data sources."""
        return {
            name: np.array(dataframe[name])
            for name in columns
        }

    @staticmethod
    def new_key_name(data_part_key: str) -> str:
        """Return the name of a time series source."""
        if data_part_key == 'idx':
            return 'idx'

        return f'data_source_ts/{data_part_key}'
