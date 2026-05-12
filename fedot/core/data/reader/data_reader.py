from typing import Any, Callable, ClassVar, List, Tuple
import numpy as np
import torch

from fedot.core.data.tensor_data.rules import (
    build_tabular_file_load_plan,
    resolve_registered_creator,
)
from fedot.core.data.tensor_data.tools import get_values_from_df
from fedot.core.data.reader.tools import get_df_from_csv, read_arff_file
from fedot.core.data.common.types import (
    ARRAY_RUNTIME_TYPES,
    PANDAS_RUNTIME_TYPES,
    ArrayType,
    PandasType,
)
from fedot.core.data.tensor_data.data_spec import DataSpec


POSSIBLE_TABULAR_IDX_KEYWORDS = ['idx', 'index', 'id', 'unnamed: 0']


class DataReader:
    """
    Registry-based reader that fills `DataSpec` from supported input sources.

    `DataReader` does not build `TensorData` directly. It selects a registered
    reader function for the input type and returns the updated `DataSpec`; the
    creator then performs preprocessing and constructs the final `TensorData`.
    """

    _creators: ClassVar[List[Tuple[Callable, Callable]]] = []

    @classmethod
    def register_creator(cls, predicate: Callable[[Any], bool]) -> Callable[[Callable], Callable]:
        """
        Register a reader function for a source type.

        Registered functions are checked in registration order. A reader function
        must accept `(source_data, spec)` and return the updated `DataSpec`.

        Args:
            predicate: Function that returns `True` if the reader can handle the
                given input.

        Returns:
            Callable[[Callable], Callable]: Decorator that registers the reader.
        """
        def decorator(func):
            cls._creators.append((predicate, func))
            return func
        return decorator

    @classmethod
    def read(cls, source_data, spec: DataSpec):
        """
        Read raw input into `spec`.

        Args:
            source_data: Raw input such as a tensor, array, dataframe, or file path.
            spec: Creation specification to update.

        Returns:
            DataSpec: Updated specification with source data stored in `features`,
            `target`, and related metadata.
        """
        creator = resolve_registered_creator(cls._creators, source_data)
        return creator(source_data, spec)


@DataReader.register_creator(lambda x: isinstance(x, torch.Tensor))
def from_torch(features: torch.Tensor, spec: DataSpec) -> DataSpec:
    """
    Read an already materialized torch tensor.

    Args:
        features: Input feature tensor.
        spec: Creation specification to update.

    Returns:
        DataSpec: Specification with `features` set to the tensor.
    """
    spec.features = features
    return spec


@DataReader.register_creator(
    lambda x: isinstance(x, ARRAY_RUNTIME_TYPES)
)
def from_numpy(features: ArrayType, spec: DataSpec) -> DataSpec:
    """
    Read a numpy or cupy array.

    Args:
        features: Input feature array.
        spec: Creation specification to update.

    Returns:
        DataSpec: Specification with `features` set to the array.
    """
    spec.features = features

    return spec


@DataReader.register_creator(
    lambda x: isinstance(x, PANDAS_RUNTIME_TYPES)
)
def from_pandas(
        features: PandasType,
        spec: DataSpec) -> DataSpec:
    """
    Read a pandas or cudf dataframe/series.

    Args:
        features: Input dataframe or series.
        spec: Creation specification to update.

    Returns:
        DataSpec: Specification with array values and detected feature names.
    """

    cols = features.columns
    spec.features_names = np.asarray(cols.to_numpy() if hasattr(cols, 'to_numpy') else cols)

    spec.features = get_values_from_df(features)

    if spec.target is not None:
        spec.target = get_values_from_df(spec.target)

    return spec


@DataReader.register_creator(
    lambda x: isinstance(x, str) and (x.endswith(".csv") or x.endswith(".tsv"))
)
def from_csv_tsv(
    file_path: str, spec: DataSpec
) -> DataSpec:
    """
    Read a CSV or TSV file path.

    The delimiter is inferred from the extension unless it is explicitly provided
    in `spec`. Index columns can be provided via `index_col` or detected from
    `possible_idx_keywords`.

    Args:
        file_path: Path to the CSV/TSV file.
        spec: Creation specification with file loading options.

    Returns:
        DataSpec: Specification with loaded values and detected feature names.
    """

    file_load_plan = build_tabular_file_load_plan(
        file_path=file_path,
        delimiter=spec.delimiter,
        possible_idx_keywords=spec.possible_idx_keywords,
        default_keywords=POSSIBLE_TABULAR_IDX_KEYWORDS,
    )
    spec.possible_idx_keywords = file_load_plan.possible_idx_keywords

    spec.features = get_df_from_csv(
        file_path=file_load_plan.file_path,
        delimiter=file_load_plan.delimiter,
        index_col=spec.index_col,
        possible_idx_keywords=file_load_plan.possible_idx_keywords,
        columns_to_drop=spec.columns_to_drop,
        nrows=spec.max_rows
    )

    cols = spec.features.columns
    spec.features_names = np.asarray(cols.to_numpy() if hasattr(cols, 'to_numpy') else cols)

    spec.features = get_values_from_df(spec.features)

    return spec


@DataReader.register_creator(
    lambda x: isinstance(x, str) and x.endswith(".arff")
)
def from_arff(
    source: str, spec: DataSpec
) -> DataSpec:
    """
    Read an ARFF file path.

    Args:
        source: Path to the ARFF file.
        spec: Creation specification with target extraction options.

    Returns:
        DataSpec: Specification with loaded features and target.
    """

    spec.features, spec.target = read_arff_file(source, target_idx=spec.target_idx)

    return spec
