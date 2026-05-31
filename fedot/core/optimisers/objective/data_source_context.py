from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.optimisers.objective.data_objective_eval import DataSource
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter


class ComposerDataSourceMode(Enum):
    internal_split = 'internal_split'
    external_holdout = 'external_holdout'


@dataclass(frozen=True)
class ComposerDataSourceContext:
    mode: ComposerDataSourceMode
    data_producer: DataSource
    validation_blocks: Optional[int]


def build_internal_composer_data_source_context(
        train_data: Union[InputData, MultiModalData],
        cv_folds: Optional[int]) -> ComposerDataSourceContext:
    data_splitter = DataSourceSplitter(cv_folds, shuffle=True)
    data_producer = data_splitter.build(train_data)
    return ComposerDataSourceContext(
        mode=ComposerDataSourceMode.internal_split,
        data_producer=data_producer,
        validation_blocks=data_splitter.validation_blocks,
    )


def build_external_holdout_composer_data_source_context(
        train_data: InputData,
        validation_data: InputData) -> ComposerDataSourceContext:
    return ComposerDataSourceContext(
        mode=ComposerDataSourceMode.external_holdout,
        data_producer=DataSourceSplitter.build_holdout_producer_from_split(train_data, validation_data),
        validation_blocks=None,
    )
