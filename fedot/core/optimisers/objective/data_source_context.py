from dataclasses import dataclass
from enum import Enum
from typing import Optional

from fedot.core.data.tensor_data import TensorData
from fedot.core.optimisers.objective.data_objective_eval import TensorDataSource
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter


class ComposerDataSourceMode(Enum):
    internal_split = 'internal_split'
    external_holdout = 'external_holdout'


@dataclass(frozen=True)
class ComposerTensorDataSourceContext:
    mode: ComposerDataSourceMode
    data_producer: TensorDataSource
    validation_blocks: Optional[int]


def build_internal_composer_tensor_data_source_context(
        train_data: TensorData,
        cv_folds: Optional[int]) -> ComposerTensorDataSourceContext:
    data_splitter = DataSourceSplitter(cv_folds, shuffle=True)
    data_producer = data_splitter.build(train_data)
    return ComposerTensorDataSourceContext(
        mode=ComposerDataSourceMode.internal_split,
        data_producer=data_producer,
        validation_blocks=data_splitter.validation_blocks,
    )


def build_external_holdout_composer_tensor_data_source_context(
        train_data: TensorData,
        validation_data: TensorData) -> ComposerTensorDataSourceContext:
    return ComposerTensorDataSourceContext(
        mode=ComposerDataSourceMode.external_holdout,
        data_producer=DataSourceSplitter.build_holdout_producer_from_split(train_data, validation_data),
        validation_blocks=None,
    )
