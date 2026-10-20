
import re

import pandas as pd

from fedot.core.data.data_detection import TextDataDetector


class TextDataDetectorV2(TextDataDetector):
    """
    Experimental text detector.

    Detects semantic text using statistical
    and structural properties of column values.
    """

    MIN_NON_EMPTY = 3
    STRUCTURE_FRACTION = 0.60

    @staticmethod
    def _sample_ratio(
        values: pd.Series,
        predicate,
    ) -> float:
        return sum(
            bool(predicate(value))
            for value in values
        ) / len(values)

    @staticmethod
    def _is_serialized_collection(value: str) -> bool:
        """
        Detects serialized lists and dictionaries.
        """
        text = value.strip()

        return (
            (
                text.startswith("{")
                and text.endswith("}")
            )
            or (
                text.startswith("[")
                and text.endswith("]")
            )
        )

    @staticmethod
    def _is_slug(value: str) -> bool:
        """
        Detects slug-like identifiers.
        """
        return bool(
            re.fullmatch(
                r"[a-z0-9]+(?:[-_][a-z0-9]+){2,}",
                value.strip(),
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _is_person_name(value: str) -> bool:
        """
        Detects simple person-name patterns.
        """
        parts = value.strip().split()

        if not 2 <= len(parts) <= 3:
            return False

        stopwords = {
            "the", "and", "of", "for",
            "has", "in", "at", "with",
            "a", "an",
        }

        if any(
            part.lower() in stopwords
            for part in parts
        ):
            return False

        return all(
            bool(
                re.fullmatch(
                    r"[A-Z][a-zA-Z'-]*",
                    part,
                )
            )
            for part in parts
        )

    @classmethod
    def _is_person_list(cls, value: str) -> bool:
        """
        Detects comma-separated lists of person names.
        """

        parts = [
            part.strip()
            for part in value.split(",")
        ]

        if len(parts) < 3:
            return False

        person_ratio = sum(
            cls._is_person_name(part)
            for part in parts
        ) / len(parts)

        return person_ratio >= 0.80

    def analyze_column(self, column: pd.Series) -> dict:
        """
        Returns text detection result and diagnostic data.
        """

        if not (
            pd.api.types.is_object_dtype(column)
            or pd.api.types.is_string_dtype(column)
        ):
            return {
                "is_text": False,
                "reason": "NOT_STRING",
                "features": {},
            }

        values = (
            column.dropna()
            .astype(str)
            .str.strip()
        )

        values = values[values != ""]

        if len(values) < self.MIN_NON_EMPTY:
            return {
                "is_text": False,
                "reason": "TOO_FEW_VALUES",
                "features": {},
            }

        if self._is_float_compatible(column):
            return {
                "is_text": False,
                "reason": "NUMERIC_STRINGS",
                "features": {},
            }

        # Reuse FEDOT feature extraction.
        features = self.extract_features(values)

        # ================================================
        # 1. Reject explicit structured values
        # ================================================

        structured_features = (
            "url_ratio",
            "email_ratio",
            "uuid_ratio",
            "numeric_ratio",
            "path_ratio",
            "id_ratio",
        )

        for feature in structured_features:
            if (
                features[feature]
                >= self.STRUCTURED_THRESHOLD
            ):
                return {
                    "is_text": False,
                    "reason": f"STRUCTURED_{feature.upper()}",
                    "features": features,
                }

        # ================================================
        # 2. Detect serialized or structured string data
        # ================================================

        collection_ratio = self._sample_ratio(
            values,
            self._is_serialized_collection,
        )

        slug_ratio = self._sample_ratio(
            values,
            self._is_slug,
        )

        person_list_ratio = self._sample_ratio(
            values,
            self._is_person_list,
        )

        person_ratio = self._sample_ratio(
            values,
            self._is_person_name,
        )

        features.update({
            "collection_ratio": collection_ratio,
            "slug_ratio": slug_ratio,
            "person_list_ratio": person_list_ratio,
            "person_ratio": person_ratio,
        })

        if collection_ratio >= self.STRUCTURE_FRACTION:
            return {
                "is_text": False,
                "reason": "SERIALIZED_COLLECTION",
                "features": features,
            }

        if slug_ratio >= self.STRUCTURE_FRACTION:
            return {
                "is_text": False,
                "reason": "SLUG_VALUES",
                "features": features,
            }

        if person_list_ratio >= self.STRUCTURE_FRACTION:
            return {
                "is_text": False,
                "reason": "PERSON_LIST",
                "features": features,
            }

        if (
            person_ratio >= 0.80
            and features["mean_word_count"] <= 3.2
        ):
            return {
                "is_text": False,
                "reason": "PERSON_NAMES",
                "features": features,
            }

        # ================================================
        # 3. Get statistical features
        # ================================================

        mean_words = features["mean_word_count"]
        mean_length = features["mean_length"]
        alphabetic = features["alphabetic_ratio"]
        unique_ratio = features["unique_ratio"]
        vocabulary = features["vocabulary_size"]

        function_words = features[
            "function_word_ratio"
        ]

        # ================================================
        # 4. Long natural-language text
        # ================================================

        long_text = (
            (
                features["median_word_count"] >= 7
                and features["median_length"] >= 40
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
                "is_text": True,
                "reason": "LONG_TEXT",
                "features": features,
            }

        # ================================================
        # 5. Short semantic text
        # ================================================

        short_text = (
            unique_ratio >= 0.45
            and mean_words >= 2.4
            and mean_length >= 12
            and alphabetic >= 0.48
            and vocabulary >= 15
        )

        if short_text:
            return {
                "is_text": True,
                "reason": "DIVERSE_SHORT_TEXT",
                "features": features,
            }

        # ================================================
        # 6. Repeated but semantically rich text
        # ================================================

        repeated_text = (
            mean_words >= 3.4
            and mean_length >= 18
            and alphabetic >= 0.65
            and vocabulary >= 40
            and (
                unique_ratio >= 0.03
                or function_words >= 0.02
                or mean_words >= 3.8
            )
        )

        if repeated_text:
            return {
                "is_text": True,
                "reason": "REPEATED_COMPOSITIONAL_TEXT",
                "features": features,
            }

        # ================================================
        # 7. Otherwise keep as non-text
        # ================================================

        return {
            "is_text": False,
            "reason": "CATEGORICAL_OR_UNCERTAIN",
            "features": features,
        }

    def _column_contains_text(
        self,
        column: pd.Series,
    ) -> bool:
        return self.analyze_column(column)["is_text"]
