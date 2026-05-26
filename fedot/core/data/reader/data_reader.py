import numpy as np
import torch
from dataclasses import dataclass

from fedot.core.common.registry import Registry
from fedot.core.data.tensor_data.rules import (
    DataReaderNotFoundError,
    build_tabular_file_load_plan,
)
from fedot.core.data.tensor_data.tools import get_values_from_df
from fedot.core.data.reader.tools import get_df_from_csv, read_arff_file
from fedot.core.data.common.types import (
    ARRAY_RUNTIME_TYPES,
    PANDAS_RUNTIME_TYPES,
    ArrayType,
    PandasType,
    IndexType,
    TensorLike,
)
from fedot.core.data.tensor_data.data_spec import DataSpec


POSSIBLE_TABULAR_IDX_KEYWORDS = ['idx', 'index', 'id', 'unnamed: 0']


@dataclass
class DataReaderResult:
    """Container returned by :meth:`DataReader.read` and registered reader functions.

    Holds raw feature values read from the source (array, dataframe, or file) and
    optional column/attribute names. Callers such as :class:`~fedot.core.data.tensor_data.tensor_data_creator.TensorDataCreator`
    merge these fields into a :class:`~fedot.core.data.tensor_data.data_spec.DataSpec`
    (``features``, ``features_names``); target handling stays in the creator pipeline.

    Attributes:
        features: Feature matrix or tensor in reader-specific layout (e.g. NumPy array
            for CSV/ARFF after ``get_values_from_df`` / ``read_arff_file``).
        features_names: Column names for tabular sources, ARFF attribute names, or
            ``None`` when the source does not provide names (e.g. plain NumPy).
    """

    features: TensorLike
    features_names: IndexType = None


class DataReader(Registry):
    """
    Registry of lightweight readers from raw inputs to arrays and names.

    ``DataReader`` does not build :class:`~fedot.core.data.tensor_data.tensor_data.TensorData`.
    It dispatches ``source_data`` to the first registered predicate whose reader
    returns a :class:`DataReaderResult`. :class:`~fedot.core.data.tensor_data.tensor_data_creator.TensorDataCreator`
    assigns ``result.features`` / ``result.features_names`` onto its ``DataSpec`` and
    runs preprocessing and tensor conversion.

    Some readers also set ``spec.features`` for backward compatibility; the
    authoritative read output is always the returned :class:`DataReaderResult`.
    """

    not_found_error = DataReaderNotFoundError
    not_found_message = 'No reading function registered for data type: {source_type}'

    @classmethod
    def read(cls, source_data, spec: DataSpec) -> DataReaderResult:
        """
        Dispatch ``source_data`` to the matching registered reader.

        Args:
            source_data: Raw input (tensor, array, dataframe, or file path).
            spec: Creation specification; readers may read options from it
                (delimiter, ``index_col``, etc.) and some mutate ``spec.features``.

        Returns:
            DataReaderResult: Loaded ``features`` and ``features_names`` for the caller
            to copy into a :class:`~fedot.core.data.tensor_data.data_spec.DataSpec`.

        Raises:
            DataReaderNotFoundError: If no registered reader matches
                ``source_data``.
        """
        creator = cls.resolve_creator(source_data)
        return creator(source_data, spec)


@DataReader.register_creator(lambda x: isinstance(x, torch.Tensor))
def from_torch(features: torch.Tensor, spec: DataSpec) -> DataReaderResult:
    """
    Read an already materialized torch tensor.

    Args:
        features: Input feature tensor.
        spec: Creation specification (``spec.features`` is set for compatibility).

    Returns:
        DataReaderResult: The same tensor and optional ``spec.features_names``.
    """
    spec.features = features
    return DataReaderResult(features=features, features_names=spec.features_names)


@DataReader.register_creator(
    lambda x: isinstance(x, ARRAY_RUNTIME_TYPES)
)
def from_numpy(features: ArrayType, spec: DataSpec) -> DataReaderResult:
    """
    Read a NumPy or CuPy array.

    Args:
        features: Input feature array.
        spec: Creation specification (``spec.features`` is set for compatibility).

    Returns:
        DataReaderResult: The array and optional ``spec.features_names``.
    """
    spec.features = features

    return DataReaderResult(features=features, features_names=spec.features_names)


@DataReader.register_creator(
    lambda x: isinstance(x, PANDAS_RUNTIME_TYPES)
)
def from_pandas(
        features: PandasType,
        spec: DataSpec) -> DataReaderResult:
    """
    Read a pandas or cuDF dataframe/series.

    Args:
        features: Input dataframe or series.
        spec: Creation specification passed through for options.

    Returns:
        DataReaderResult: Array values from ``get_values_from_df`` and column names.
    """

    cols = features.columns
    features_names = np.asarray(
        cols.to_numpy() if hasattr(cols, 'to_numpy') else cols)

    features = get_values_from_df(features)

    return DataReaderResult(features=features, features_names=features_names)


@DataReader.register_creator(
    lambda x: isinstance(x, str) and (x.endswith(".csv") or x.endswith(".tsv"))
)
def from_csv_tsv(
    file_path: str, spec: DataSpec
) -> DataReaderResult:
    """
    Read a CSV or TSV file path.

    The delimiter is inferred from the extension unless ``spec.delimiter`` overrides it.
    Index columns can be set via ``spec.index_col`` or detected from
    ``spec.possible_idx_keywords`` (with defaults from this module).

    Args:
        file_path: Path to the CSV/TSV file.
        spec: Creation specification with file loading options.

    Returns:
        DataReaderResult: Numeric feature matrix and column names from the file.
    """

    file_load_plan = build_tabular_file_load_plan(
        file_path=file_path,
        delimiter=spec.delimiter,
        possible_idx_keywords=spec.possible_idx_keywords,
        default_keywords=POSSIBLE_TABULAR_IDX_KEYWORDS,
    )

    data = get_df_from_csv(
        file_path=file_load_plan.file_path,
        delimiter=file_load_plan.delimiter,
        index_col=spec.index_col,
        possible_idx_keywords=file_load_plan.possible_idx_keywords,
        columns_to_drop=spec.columns_to_drop,
        nrows=spec.max_rows
    )

    cols = data.columns
    features_names = np.asarray(
        cols.to_numpy() if hasattr(cols, 'to_numpy') else cols)

    features = get_values_from_df(data)

    return DataReaderResult(features=features, features_names=features_names)


@DataReader.register_creator(
    lambda x: isinstance(x, str) and x.endswith(".arff")
)
def from_arff(
    source: str, spec: DataSpec
) -> DataReaderResult:
    """
    Read an ARFF file path.

    Args:
        source: Path to the ARFF file.
        spec: Reserved for future reader options (currently unused).

    Returns:
        DataReaderResult: Stacked attribute matrix from :func:`~fedot.core.data.reader.tools.read_arff_file`
        and attribute (field) names, or ``None`` names when the header has no attributes.
    """

    features, features_names = read_arff_file(source)

    return DataReaderResult(features=features, features_names=features_names)
