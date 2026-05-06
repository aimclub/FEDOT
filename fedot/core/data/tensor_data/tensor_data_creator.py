import logging
import torch

from fedot.core.data.common.enums import StateEnum
from fedot.core.data.tensor_data.rules import (
    build_creation_failure,
    build_creation_request,
    build_device_sync_plan,
    build_raw_conversion_plan,
    normalize_array_target_reference,
    normalize_tensordata_identity,
)
from fedot.core.data.tensor_data.tools import (
    convert_idx_to_list, replace_missing_with_nan, get_target_and_features,
    transform_to_tensor, _drop_rows_with_nan_in_target)
from fedot.preprocessing.ts_preprocessing import process_ts_data
from fedot.core.backend.backend import Backend, torch_to_xp
from fedot.core.data.common.compatibility_rules import autodetect_tensor_data_type
from fedot.preprocessing.tools.index_mapping_tools import create_index_mapping
from fedot.preprocessing.service.tabular_obligatory_service import ObligatoryTabularService
from fedot.preprocessing.tools.tools import get_used_idx_from_plan
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.reader.data_reader import DataReader
from fedot.core.data.tensor_data.data_spec import DataSpec
from fedot.core.data.tensor_data.lazy_tensor import LazyTensor


logger = logging.getLogger(__name__)


class TensorDataCreator:
    """
    Facade for creating `TensorData` from raw sources.

    `TensorDataCreator` coordinates the full creation pipeline:
    reading raw data into `DataSpec`, normalizing task/data metadata, applying
    obligatory preprocessing, converting values to torch tensors, and moving the
    result to the requested FEDOT backend.

    Supported sources are defined by `DataReader` registrations and include torch
    tensors, numpy/cupy arrays, pandas/cudf dataframes, CSV/TSV files, and ARFF
    files.
    """

    def __init__(self):
        """
        Initialize an empty creator.

        The `spec` field is assigned inside :meth:`create` before reading and
        preprocessing starts.
        """
        self.spec = None

    def obligatory_preprocess(self):
        """
        Run preprocessing steps that are required before building `TensorData`.

        The method mutates `self.spec`: normalizes column/index references,
        replaces missing values, processes time-series layouts, extracts target
        columns, drops rows with missing target values, applies obligatory tabular
        services, converts features/target to torch tensors, and fills categorical
        and numerical feature indices.
        """

        self.spec.target_idx = convert_idx_to_list(self.spec.target_idx)
        self.spec.categorical_idx = convert_idx_to_list(self.spec.categorical_idx)
        self.spec.features_names = convert_idx_to_list(self.spec.features_names)
        self.spec.ts_terms_idx = convert_idx_to_list(self.spec.ts_terms_idx)

        self.spec.features = replace_missing_with_nan(self.spec.features)

        # TODO romankuklo: think how to add it to obligatory ts preprocessing
        self.spec.features, self.spec.target, self.spec.ts_init_shape, self.spec.ts_terms_idx = process_ts_data(self.spec.features,
                                                                                            self.spec.target,
                                                                                            self.spec.features_names,
                                                                                            self.spec.state,
                                                                                            self.spec.ts_orientation,
                                                                                            self.spec.ts_terms_idx,
                                                                                            self.spec.ts_forecast_horizon,
                                                                                            self.spec.data_type)

        self.spec.idx_mapping = create_index_mapping(self.spec.features, self.spec.ts_init_shape)

        self.spec.features, self.spec.target, self.spec.idx_mapping = get_target_and_features(self.spec.features,
                                                                               self.spec.target,
                                                                               self.spec.features_names,
                                                                               self.spec.target_idx,
                                                                               self.spec.state,
                                                                               self.spec.data_type,
                                                                               self.spec.idx_mapping)

        self.spec.features, self.spec.target = _drop_rows_with_nan_in_target(self.spec.features, self.spec.target)

        service = ObligatoryTabularService()

        service_params = {
            "encoding_strategy": self.spec.encoding_strategy,
            "embedding_strategy": self.spec.embedding_strategy,
            "custom_strategy": self.spec.custom_strategy,
            "features_names": self.spec.features_names,
            "idx_mapping": self.spec.idx_mapping,
            "data_type": self.spec.data_type
        }

        if self.spec.state == StateEnum.FIT:
            prepared_data = service.fit_transform(
                self.spec.features,
                self.spec.target,
                service_params
            )
        else:
            prepared_data = service.transform(
                self.spec.features,
                self.spec.target,
                self.spec.idx_mapping
            )

        self.spec.features = prepared_data.features
        self.spec.target = prepared_data.target
        self.spec.idx_mapping = prepared_data.idx_mapping

        # TODO romankuklo: how to save steps?
        self.spec.features, self.spec.target = transform_to_tensor(self.spec.features,
                                                         self.spec.target,
                                                         self.spec.ts_init_shape)

        self.spec.idx = torch.arange(self.spec.features.shape[1], dtype=torch.int32)

        preprocessed_idx = get_used_idx_from_plan(service.plan)
        self.spec.categorical_idx = list(set(self.spec.categorical_idx) | set(preprocessed_idx))
        self.spec.numerical_idx = list(set(range(self.spec.features.shape[1])) - set(self.spec.categorical_idx))

    def preprocess_data(self):
        """
        Normalize metadata and prepare raw arrays for obligatory preprocessing.

        If `data_type` is omitted, it is inferred from the task. Task, data type,
        and state values are normalized to FEDOT runtime objects. Raw features and
        target are converted to the active backend array module before
        `obligatory_preprocess` converts them to torch tensors.

        When the GPU backend cannot represent raw object/string data as a cupy
        array, preprocessing is performed under the CPU backend and the resulting
        tensors are moved to the requested backend later.
        """
        if self.spec.data_type is None:
            self.spec.data_type = autodetect_tensor_data_type(self.spec.task)
        
        self.spec.target, self.spec.target_idx = normalize_array_target_reference(
            target=self.spec.target,
            target_idx=self.spec.target_idx,
            feature_width=self.spec.features.shape[1],
        )

        identity = normalize_tensordata_identity(
            task=self.spec.task,
            data_type=self.spec.data_type,
            state=self.spec.state,
        )
        self.spec.task = identity.task
        self.spec.data_type = identity.data_type
        self.spec.state = identity.state

        raw_conversion_plan = build_raw_conversion_plan(Backend().name)
        if isinstance(self.spec.features, torch.Tensor):
            self.spec.features = torch_to_xp(self.spec.features, Backend().xp)
            self.spec.target = torch_to_xp(self.spec.target, Backend().xp)
        else:
            try:
                self.spec.features = Backend().xp.array(self.spec.features)
                if self.spec.target is not None:
                    self.spec.target = Backend().xp.array(self.spec.target)
            except Exception:
                if not raw_conversion_plan.should_try_cpu_fallback:
                    raise
                with Backend().override("cpu"):
                    logger.info("Turning to cpu backend to get TensorData due to failed to convert features to cupy array")
                    self.spec.features = Backend().xp.array(self.spec.features)
                    if self.spec.target is not None:
                        self.spec.target = Backend().xp.array(self.spec.target)
                    self.obligatory_preprocess()
                    raw_conversion_plan.preprocessing_done = True
        
        if not raw_conversion_plan.preprocessing_done:
            self.obligatory_preprocess()
            raw_conversion_plan.preprocessing_done = True

    def to_tensor_data(self) -> "TensorData":
        """
        Build `TensorData` from the current preprocessed `DataSpec`.

        Returns:
            TensorData: Runtime data container populated with fields from `spec`.
        """
        return TensorData(
            features=self.spec.features,
            target=self.spec.target,
            task=self.spec.task,
            data_type=self.spec.data_type,
            state=self.spec.state,
            features_names=self.spec.features_names,
            target_idx=self.spec.target_idx,
            categorical_idx=self.spec.categorical_idx,
            encoding_strategy=self.spec.encoding_strategy,
            embedding_strategy=self.spec.embedding_strategy,
            custom_strategy=self.spec.custom_strategy,
            ts_orientation=self.spec.ts_orientation,
            ts_terms_idx=self.spec.ts_terms_idx,
            ts_forecast_horizon=self.spec.ts_forecast_horizon,
            dataloader_kwargs=self.spec.dataloader_kwargs,
        )

    def to_backend(self, tensor_data: "TensorData") -> "TensorData":
        """
        Synchronize `TensorData` tensors with the active backend device.

        Args:
            tensor_data: Tensor data produced by :meth:`to_tensor_data`.

        Returns:
            TensorData: The same tensor data object, moved to the backend device
            when needed.
        """
        device_sync_plan = build_device_sync_plan(
            features_device_type=tensor_data.features.device.type,
            backend_device_type=Backend().device.type,
        )
        if device_sync_plan.should_move_to_backend:
            tensor_data.to(Backend().device)

        return tensor_data

    @classmethod
    def create(cls, source_data, backend_name, **kwargs):
        """
        Create `TensorData` from a raw data source.

        Args:
            source_data: Input data supported by `DataReader` registrations.
            backend_name: Target backend name, currently `"cpu"` or `"gpu"`.
            **kwargs: Options used to initialize `DataSpec`, such as `target`,
                `target_idx`, `data_type`, `state`, `encoding_strategy`,
                `ts_orientation`, and file loading options.

        Returns:
            TensorData: Materialized tensor data on the requested backend.

        Raises:
            ValueError: If reading or preprocessing fails. The error message
            includes the source type and backend name.
        """
        creator = cls()

        creation_request = build_creation_request(backend_name)
        Backend().set(creation_request.backend_name)
        creator.spec = DataSpec(**kwargs)

        reader = DataReader()

        try:
            creator.spec = reader.read(source_data, creator.spec)
            creator.preprocess_data()
            tensor_data = creator.to_tensor_data()
            tensor_data = creator.to_backend(tensor_data)
            return tensor_data
        except Exception as e:
            failure = build_creation_failure(source_data, creation_request.backend_name, e)
            raise ValueError(failure.message) from e
    
    @classmethod
    def create_lazy(cls, source_data, backend_name, **kwargs):
        """
        Create a lazy `TensorData` wrapper.

        The returned `LazyTensor` delays the same work as :meth:`create` until
        `LazyTensor.get()` or `LazyTensor.to(...)` is called.

        Args:
            source_data: Input data supported by `DataReader` registrations.
            backend_name: Target backend name, currently `"cpu"` or `"gpu"`.
            **kwargs: Options forwarded to :meth:`create`.

        Returns:
            LazyTensor: Lazy wrapper that caches the created `TensorData`.
        """

        def _create():
            return cls.create(source_data, backend_name, **kwargs)

        return LazyTensor(_create)
