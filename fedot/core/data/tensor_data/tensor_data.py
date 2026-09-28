from dataclasses import dataclass, field, fields
from copy import deepcopy
import sys
from typing import Optional, Union, Dict, Any, Tuple
from fedot.core.data.common.types import IndexType
from fedot.core.data.common.enums import StateEnum, TSOrientationEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task
import numpy as np
import torch
import logging
from fedot.core.data.tensor_data.tools import get_device_from_str, tensor_memory_usage, td_values_equal
from fedot.core.data.common.types import TensorLike
from fedot.core.data.tensor_data.contracts import PreparationState, validate_runtime_data

logger = logging.getLogger(__name__)


@dataclass
class TensorData:
    """
    Tensor-based data container used by FEDOT data processing and model nodes.

    `TensorData` stores already prepared features, target values, task metadata,
    preprocessing metadata, and dataloader options. Data reading, target extraction,
    preprocessing, and backend synchronization are performed by
    :class:`TensorDataCreator`; this class is the resulting runtime container.

    Most fields are copied from `DataSpec` after
    `TensorDataCreator.obligatory_preprocess()` has normalized index references,
    processed time-series layouts, extracted the target, applied obligatory
    tabular transformations, converted data to torch tensors, and inferred
    categorical/numerical feature indices.

    Attributes:
        task: FEDOT task descriptor used by downstream operations.
        data_type: FEDOT data type after alias normalization. Tabular aliases are
            normalized to `DataTypesEnum.tabular`; time-series aliases are
            normalized to `DataTypesEnum.ts`.
        state: Data processing state, usually `StateEnum.FIT`. It controls whether
            pipeline is fitted or reused for transform-like processing.
        idx: One label per sample on axis 0. Creation preserves labels when rows
            with missing targets are removed; omitted labels are row positions.
        features: Prepared feature tensor. This field is required. Raw arrays/dataframes/files are read by
            `DataReader`, cleaned, optionally reshaped for time series, transformed
            by obligatory services, converted to `torch.Tensor`, and finally moved
            to the selected backend device.
        target: Prepared target tensor or `None`. It can be provided explicitly or
            extracted from `features` using `target_idx`; rows with missing target
            values are dropped before tensor conversion.
        predict: Optional externally provided predictions. The current creator
            passes this field through only when it is set manually on the instance.
        target_idx: Target column index/name after index-reference normalization.
            Used during creation to split `target` from `features`.
        categorical_idx: Indices of categorical/preprocessed feature columns after
            obligatory tabular preprocessing. User-provided categorical indices are
            merged with indices used by preprocessing plans.
        numerical_idx: Indices of final feature columns not listed in
            `categorical_idx`.
        features_names: Names of prepared columns in output order. Source names
            are retained in `preparation_state.schema` for selector resolution.
        idx_mapping: Mapping from current feature column to source feature column.
            This is not a row mapping. Expanded columns may share one source.
        ts_orientation: Time-series layout hint, for example `"long"` or `"wide"`.
        ts_terms_idx: Index/name of the term column for long-format time series.
            It can be normalized or updated during time-series preprocessing.
        ts_forecast_horizon: Optional forecast horizon used by time-series
            preprocessing.
        ts_init_shape: Original time-series shape captured during time-series
            preprocessing and reused when converting data to tensors.
        dataloader_kwargs: Options for future dataloader construction, such as
            batch size, shuffling, worker count, and `drop_last`.
        fingerprint: Fingerprint of the data used for caching and tracing.
        trace_uuid: UUID of the trace used for tracing.

    Examples:
        Prefer the public helper:

        >>> from fedot import create_data
        >>> train = create_data(array, target=y)
        >>> test = create_data(x_test, from_data=train)

        Or extract a target column from a CSV / dataframe:

        >>> train = create_data('path/to/file.csv', target='target')
    """
    task: Union[Task, str]
    data_type: Union[DataTypesEnum, str]
    features: torch.Tensor

    state: Union[str, StateEnum] = StateEnum.FIT
    idx: IndexType = None
    target: Optional[torch.Tensor] = None
    predict: TensorLike = None
    target_idx: IndexType = None
    categorical_idx: IndexType = field(default_factory=list)
    numerical_idx: IndexType = field(default_factory=list)
    features_names: IndexType = None
    idx_mapping: dict[int, int] = field(default_factory=dict)
    ts_orientation: Union[TSOrientationEnum, str] = None
    ts_terms_idx: Optional[Union[int, str]] = None
    ts_forecast_horizon: Optional[int] = None
    ts_init_shape: Optional[Tuple[int]] = None

    dataloader_kwargs: Dict[str, Any] = field(default_factory=dict)

    # hashes
    fingerprint: Optional[str] = None
    trace_uuid: Optional[str] = None
    preparation_state: Optional[PreparationState] = field(
        default=None, repr=False, compare=False)

    def __post_init__(self):
        # Direct construction borrows tensor storage; create_data owns its output
        # tensors. Mutable metadata is always private to the new container.
        for name in ('task', 'idx', 'target_idx', 'categorical_idx', 'numerical_idx',
                     'features_names', 'idx_mapping', 'dataloader_kwargs', 'ts_terms_idx'):
            value = getattr(self, name)
            setattr(self, name, value.clone() if isinstance(
                value, torch.Tensor) else deepcopy(value))
        self.validate()

    def validate(self):
        """Check row alignment and tensor devices; return self without mutation."""
        validate_runtime_data(self)
        return self

    @property
    def device(self):
        """Device owned by prepared tensor storage, independent of Backend()."""
        return self.features.device if isinstance(self.features, torch.Tensor) else None

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TensorData):
            return False

        return all(
            td_values_equal(getattr(self, field.name),
                            getattr(other, field.name))
            for field in fields(self) if field.compare
        )

    @property
    def memory_usage(self) -> Dict[str, int]:
        """
        Estimate tensor memory usage in bytes.

        The estimate works for CPU and GPU torch tensors because it is based on
        tensor element size and number of elements. Metadata is estimated with
        `sys.getsizeof`, so nested containers are only shallowly counted.

        Returns:
            Dict[str, int]: Memory usage by tensor field and the total value:
                `features`, `target`, `predict`, `metadata`, and `total`.
        """
        usage = {
            'features': tensor_memory_usage(self.features),
            'target': tensor_memory_usage(self.target),
            'predict': tensor_memory_usage(self.predict),
            'metadata': self._metadata_memory_usage(),
        }
        usage['total'] = sum(usage.values())
        return usage

    def _metadata_memory_usage(self) -> int:
        metadata_fields = (
            self.task,
            self.data_type,
            self.state,
            self.idx,
            self.target_idx,
            self.categorical_idx,
            self.numerical_idx,
            self.features_names,
            self.idx_mapping,
            self.ts_orientation,
            self.ts_terms_idx,
            self.ts_forecast_horizon,
            self.ts_init_shape,
            self.dataloader_kwargs,
        )
        return sum(sys.getsizeof(field) for field in metadata_fields if field is not None)

    def to(self, device: Union[str, torch.device]):
        """
        Move tensor fields to the given device in place.

        This explicitly mutating operation moves features, target, predictions,
        and tensor row labels together. It does not change the global Backend or
        the fitted handlers' preparation backend. If a transfer fails, no fields
        are replaced. Directly constructed tensor storage may alias caller data;
        create_data results do not share source tensor storage.

        Args:
            device (Union[str, torch.device]): Target device, for example `"cpu"`
                or `"cuda"`.

        Returns:
            TensorData: The same instance with tensor fields moved to `device`.
        """
        device = get_device_from_str(device)
        moved = {name: getattr(self, name).to(device)
                 for name in ('features', 'target', 'predict', 'idx')
                 if isinstance(getattr(self, name), torch.Tensor)}
        for name, value in moved.items():
            setattr(self, name, value)
        return self
