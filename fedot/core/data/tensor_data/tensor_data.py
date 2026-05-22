from dataclasses import dataclass, field
import sys
from typing import Optional, Union, Dict, Any, Tuple
from fedot.core.data.common.types import IndexType
from fedot.core.data.common.enums import StateEnum, TSOrientationEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task
import torch
import logging
from fedot.core.data.tensor_data.tools import get_device_from_str, tensor_memory_usage
from fedot.core.data.common.types import TensorLike

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
        idx: Sample or feature index metadata. During current creator preprocessing
            it is initialized from the final feature tensor width.
        features: Prepared feature tensor. Raw arrays/dataframes/files are read by
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
        encoding_strategy: Optional categorical encoding strategy used by
            obligatory tabular preprocessing.
        embedding_strategy: Optional text embedding strategy used by obligatory
            tabular preprocessing.
        custom_strategy: Optional custom preprocessing strategy passed to
            obligatory tabular preprocessing.
        features_names: Source feature names used to resolve string indices such as
            `target_idx`, `categorical_idx`, and `ts_terms_idx`.
        idx_mapping: Mapping between original row positions and rows kept after
            preprocessing. It is created before target extraction and updated by
            obligatory tabular services.
        ts_orientation: Time-series layout hint, for example `"long"` or `"wide"`.
        ts_terms_idx: Index/name of the term column for long-format time series.
            It can be normalized or updated during time-series preprocessing.
        ts_forecast_horizon: Optional forecast horizon used by time-series
            preprocessing.
        ts_init_shape: Original time-series shape captured during time-series
            preprocessing and reused when converting data to tensors.
        dataloader_kwargs: Options for future dataloader construction, such as
            batch size, shuffling, worker count, and `drop_last`.

    Examples:
        Create `TensorData` from a numpy array using the creator:

        >>> from fedot.core.data.tensor_data.td_creator import TensorDataCreator
        >>> td = TensorDataCreator.create(array, backend_name='cpu')

        Create `TensorData` from a CSV file and extract a target column:

        >>> td = TensorDataCreator.create(
        ...     'path/to/file.csv',
        ...     backend_name='cpu',
        ...     target_idx='target',
        ... )
    """
    task: Union[Task, str]
    data_type: Union[DataTypesEnum, str]

    state: Union[str, StateEnum] = StateEnum.FIT
    idx: IndexType = None
    features: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None
    predict: TensorLike = None
    target_idx: IndexType = None
    categorical_idx: IndexType = field(default_factory=list)
    # TODO romankuklo: delete strategies
    numerical_idx: IndexType = field(default_factory=list)
    encoding_strategy: Optional[Union[Dict]] = None
    embedding_strategy: Optional[Union[Dict]] = None
    custom_strategy: Optional[Dict] = None
    features_names: IndexType = None
    idx_mapping: dict[int, int] = field(default_factory=dict)
    ts_orientation: Union[TSOrientationEnum, str] = None
    ts_terms_idx: Optional[Union[int, str]] = None
    ts_forecast_horizon: Optional[int] = None
    ts_init_shape: Optional[Tuple[int]] = None

    dataloader_kwargs: Dict[str, Any] = field(default_factory=dict)

    # hashes 
    raw_fingerprint: Optional[str] = None
    ready_fingerprint: Optional[str] = None
    parent_raw_fingerprint: Optional[str] = None
    parent_ready_fingerprint: Optional[str] = None

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
            self.encoding_strategy,
            self.embedding_strategy,
            self.custom_strategy,
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

        This method mutates `features` and `target` when they are torch tensors and
        returns `self` to support chained calls.

        Args:
            device (Union[str, torch.device]): Target device, for example `"cpu"`
                or `"cuda"`.

        Returns:
            TensorData: The same instance with tensor fields moved to `device`.
        """
        device = get_device_from_str(device)
        if isinstance(self.features, torch.Tensor):
            self.features = self.features.to(device)
        if self.target is not None:
            self.target = self.target.to(device)
        return self
