from dataclasses import dataclass
from typing import Any, Optional, Union, Callable

from fedot.core.data.data_compatibility_rules import build_data_type_compatibility
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.industrial.core.repository.constanst_repository import FEDOT_DATA_TYPE
from fedot.industrial.core.architecture.abstraction.delegator import DelegatorFactory
from torch.utils.data import DataLoader

@dataclass(frozen=True)
class IndustrialDataTypePlan:
    requested_name: str
    fedot_data_type: DataTypesEnum
    tensor_canonical_data_type: DataTypesEnum
    input_compatible_data_type: DataTypesEnum


@dataclass(frozen=True)
class IndustrialTaskPlan:
    task: Task
    have_predict_horizon: bool


@dataclass(frozen=True)
class IndustrialLearningStrategyFlags:
    strategy_name: Optional[str]
    is_big_data: bool
    is_default_fedot_context: bool


def resolve_industrial_data_type_plan(strategy_params: Optional[dict]) -> IndustrialDataTypePlan:
    requested_name = 'tensor' if strategy_params is None else strategy_params.get('data_type', 'tensor')
    if requested_name not in FEDOT_DATA_TYPE:
        raise ValueError(f'Unsupported industrial data_type: {requested_name}')
    fedot_data_type = FEDOT_DATA_TYPE[requested_name]
    compatibility = build_data_type_compatibility(fedot_data_type)
    return IndustrialDataTypePlan(
        requested_name=requested_name,
        fedot_data_type=fedot_data_type,
        tensor_canonical_data_type=compatibility.tensor_canonical,
        input_compatible_data_type=compatibility.input_compatible,
    )


def should_use_predict_horizon(strategy_params: Optional[dict]) -> bool:
    return bool(
        strategy_params is not None
        and strategy_params.get('data_type') == 'time_series'
        and 'detection_window' in strategy_params
    )


def build_industrial_task_plan(task_name: str, strategy_params: Optional[dict]) -> IndustrialTaskPlan:
    have_predict_horizon = should_use_predict_horizon(strategy_params)
    if have_predict_horizon:
        detection_window = strategy_params['detection_window']
        task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=detection_window))
    elif task_name == 'classification':
        task = Task(TaskTypesEnum.classification)
    elif task_name == 'regression':
        task = Task(TaskTypesEnum.regression)
    elif task_name == 'ts_forecasting':
        forecast_length = None if strategy_params is None else strategy_params.get('forecast_length')
        task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=forecast_length or 1))
    else:
        raise ValueError(f'Unsupported industrial task: {task_name}')
    return IndustrialTaskPlan(task=task, have_predict_horizon=have_predict_horizon)


def resolve_learning_strategy_flags(strategy_params: Optional[dict]) -> IndustrialLearningStrategyFlags:
    strategy_name = None if strategy_params is None else strategy_params.get('learning_strategy')
    return IndustrialLearningStrategyFlags(
        strategy_name=strategy_name,
        is_big_data=bool(strategy_name and 'big' in strategy_name),
        is_default_fedot_context=bool(strategy_name and 'tabular' in strategy_name),
    )

def _X2Xy(collate_fn):
    def wrapped(batch, *args, **kwargs):
        return collate_fn(batch, *args, **kwargs), [None] * len(batch)

    return wrapped
class DataLoaderHandler:
    __non_included_kwargs = {'check_worker_number_rationality'}

    @staticmethod
    def limited_generator(gen, max_batches, enumerate=False):
        i = 0
        for elem in gen:
            if i >= max_batches:
                break
            yield (i, elem) if enumerate else elem
            i += 1

    collate_modes = {
        'X2Xy': _X2Xy,
        'pass': lambda x: x
    }

    @classmethod
    def __clean_dict(cls, d: dict, is_iterable=False):
        d = {attr: val for attr, val in d.items() if not (attr.startswith('_') or attr in cls.__non_included_kwargs)}

        if is_iterable:
            d.pop('sampler', None)

        if any((d.get(k, False) for k in ('batch_size', 'shuffle', 'sampler', 'drop_last'))):
            d.pop('batch_sampler', None)
        if any((d.get(k, False) for k in ('batch_size', 'shuffle', 'sampler', 'drop_last'))):
            d.pop('batch_sampler', None)

        if d.get('batch_size', None):
            d.pop('drop_last', None)
        if d.get('drop_last', False):
            d.pop('batch_size', None)
        return d

    @classmethod
    def check_convert(cls, dataloader: DataLoader, mode: Union[None, str, Callable] = None, max_batches: int = None,
                      enumerate=False) -> DataLoader:
        batch = cls.__get_batch_sample(dataloader)
        dl_params = {attr: getattr(dataloader, attr) for attr in dir(dataloader)}
        dl_params = cls.__clean_dict(dataloader.__dict__, hasattr(dataloader.dataset, '__iter__'))
        modified1, dl_params = cls.__substitute_collate_fn(dl_params, batch, mode)
        if modified1:
            dataloader = DataLoader(**dl_params)
        if max_batches or enumerate:
            dataloader = cls.limit_batches(dataloader, max_batches, enumerate)
        return dataloader

    @classmethod
    def __get_batch_sample(cls, dataloader: DataLoader):
        for b in dataloader:
            return b

    @classmethod
    def __substitute_collate_fn(cls, dl_params: dict, batch: Any, mode: Union[None, str, Callable]):
        modified = True
        type_ = mode
        if isinstance(mode, Callable):
            collate_fn = mode
        elif isinstance(mode, str):
            collate_fn = cls.collate_modes[mode]
        else:
            type_ = cls.__check_type(batch)
            collate_fn = cls.collate_modes[type_]
        if type_ == 'pass':
            modified = False
        dl_params['collate_fn'] = collate_fn(dl_params['collate_fn'])
        return modified, dl_params

    @staticmethod
    def limit_batches(dataloader, max_batches, enumerate=False):
        if max_batches is None and not enumerate:
            return dataloader
        return DataLoaderHandler.__substitute_iter(dataloader, max_batches, enumerate)

    @staticmethod
    def __substitute_iter(iterable, max_batches=None, enumerate=False):
        max_batches = max_batches or float('inf')

        def newiter(_):
            return iter(DataLoaderHandler.limited_generator(iterable, max_batches, enumerate))

        return DelegatorFactory.create_delegator_inst(iterable, {'__iter__': newiter})

    @staticmethod
    def __check_type(batch) -> str:
        return 'pass' if isinstance(batch, (tuple, list)) else 'X2Xy'