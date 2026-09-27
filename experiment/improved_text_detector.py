import numpy as np
import pandas as pd
import string
import re
from collections import Counter

from fedot.core.data.data_detection import TextDataDetector
from sklearn.feature_extraction.text import TfidfVectorizer


class NewTextDataDetector(TextDataDetector):
    STRUCTURED_THRESHOLD = 0.8
    MIN_UNIQUE_RATIO = 0.3
    MIN_VALUES = 3

    def _column_contains_text(
        self,
        column: pd.Series,
    ) -> bool:

        # только строковые колонки
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

        if len(values) < self.MIN_VALUES:
            return False

        # строки, которые на самом деле являются числами
        if self._is_float_compatible(column):
            return False

        features = self.extract_features(
            values
        )

        # явные структурированные данные
        structured_features = [
            "url_ratio",
            "email_ratio",
            "uuid_ratio",
            "path_ratio",
            "numeric_ratio",
            "id_ratio",
        ]

        for feature in structured_features:
            if (
                features[feature]
                >= self.STRUCTURED_THRESHOLD
            ):
                return False

        # обычная категориальная колонка
        if (
            features["unique_ratio"]
            < self.MIN_UNIQUE_RATIO
        ):
            return False

        # длинный естественный текст
        if (
            features["median_word_count"] >= 7
            and features["median_length"] >= 35
            and features["alphabetic_ratio"] >= 0.55
        ):
            return True

        # обычные предложения / описания / отзывы
        if (
            features["mean_word_count"] >= 5
            and features["mean_length"] >= 25
            and features["alphabetic_ratio"] >= 0.60
            and features["digit_ratio"] < 0.25
            and features["pattern_unique_ratio"] >= 0.15
        ):
            return True

        # короткий естественный текст:
        # заголовки, bio и т.п.
        if (
            features["mean_word_count"] >= 4
            and features["mean_length"] >= 20
            and features["alphabetic_ratio"] >= 0.70
            and features["digit_ratio"] < 0.15
            and features["function_word_ratio"] >= 0.04
        ):
            return True

        return False

    def extract_features(
        self,
        values: pd.Series,
    ) -> dict:

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

    @staticmethod
    def _safe_std(
        values: pd.Series,
    ) -> float:

        result = values.std()

        if pd.isna(result):
            return 0.0

        return float(result)

    @staticmethod
    def _structured_ratios(
        values: pd.Series,
    ) -> dict:

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
    def _normalize_pattern(
        value: str,
    ) -> str:

        value = value.lower()

        # uuid
        value = re.sub(
            r"[0-9a-f]{8}-"
            r"[0-9a-f]{4}-"
            r"[0-9a-f]{4}-"
            r"[0-9a-f]{4}-"
            r"[0-9a-f]{12}",
            "<uuid>",
            value,
        )

        # числа
        value = re.sub(
            r"\d+",
            "<num>",
            value,
        )

        return value

    @classmethod
    def _pattern_unique_ratio(
        cls,
        values: pd.Series,
    ) -> float:

        patterns = values.apply(
            cls._normalize_pattern
        )

        return (
            patterns.nunique()
            / len(patterns)
        )

    @staticmethod
    def _common_token_ratio(
        tokens: list,
    ) -> float:

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
    def _function_word_ratio(
        tokens: list,
    ) -> float:

        if not tokens:
            return 0.0

        # небольшой набор служебных слов
        # нужен только как дополнительный сигнал
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
    def _tfidf_vocabulary_size(
        values: pd.Series,
    ) -> int:

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