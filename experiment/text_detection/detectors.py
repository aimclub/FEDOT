"""Detectors used in the text-column benchmark.

The two historical baselines approximate the algorithms, not their exact
implementations from the corresponding FEDOT revisions.
"""

from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from fedot.core.data.data_detection import TextDataDetector


class CurrentFedotDetector:
    """Approximation of the old uniqueness-based FEDOT detector."""

    @staticmethod
    def _is_numeric(column: pd.Series) -> bool:
        values = column.dropna()
        if values.empty:
            return False
        return pd.to_numeric(values, errors='coerce').isna().mean() < 0.5

    def define_text_columns(self, dataframe: pd.DataFrame) -> List[str]:
        detected = []
        for name in dataframe.columns:
            column = dataframe[name]
            if column.dtype != object or self._is_numeric(column):
                continue

            values = column.dropna()
            if not values.empty and values.nunique() / len(values) > 0.95:
                detected.append(name)
        return detected


class Pr903Detector:
    """Approximation of PR #903: uniqueness, TF-IDF, and link filtering."""

    def define_text_columns(self, dataframe: pd.DataFrame) -> List[str]:
        detected = []
        for name in dataframe.columns:
            column = dataframe[name]
            if column.dtype != object or CurrentFedotDetector._is_numeric(column):
                continue

            values = column.dropna().astype(str)
            if values.empty or values.nunique() / len(values) <= 0.60:
                continue
            if values.str.match(r'^https?://', case=False).mean() > 0.5:
                continue

            try:
                vectorizer = TfidfVectorizer()
                vectorizer.fit(values)
            except ValueError:
                continue

            if len(vectorizer.vocabulary_) > 20:
                detected.append(name)
        return detected


class TextDataDetectorV1(TextDataDetector):
    """Text detector used before the short-text and repeated-text changes."""

    def _column_contains_text(self, column: pd.Series) -> bool:
        if not (pd.api.types.is_object_dtype(column)
                or pd.api.types.is_string_dtype(column)):
            return False
        values = column.dropna().astype(str).str.strip()
        values = values[values != '']
        if len(values) < self.MIN_VALUES or self._is_float_compatible(column):
            return False
        features = self.extract_features(values)
        if any(features[name] >= self.STRUCTURED_THRESHOLD for name in (
                'url_ratio', 'email_ratio', 'uuid_ratio',
                'path_ratio', 'numeric_ratio', 'id_ratio')):
            return False
        if features['unique_ratio'] < self.MIN_UNIQUE_RATIO:
            return False
        return (
            features['median_word_count'] >= 7
            and features['median_length'] >= 35
            and features['alphabetic_ratio'] >= 0.55
        ) or (
            features['mean_word_count'] >= 5
            and features['mean_length'] >= 25
            and features['alphabetic_ratio'] >= 0.60
            and features['digit_ratio'] < 0.25
            and features['pattern_unique_ratio'] >= 0.15
        ) or (
            features['mean_word_count'] >= 4
            and features['mean_length'] >= 20
            and features['alphabetic_ratio'] >= 0.70
            and features['digit_ratio'] < 0.15
            and features['function_word_ratio'] >= 0.04
        )


# The integrated detector is the source of truth for V2.
TextDataDetectorV2 = TextDataDetector
