"""Effect shell for source reading, preparation, tracing and device transfer."""
from copy import deepcopy
from dataclasses import replace

from fedot.core.backend.backend import Backend
from fedot.core.caching.cacher import Cacher
from fedot.core.caching.hasher import Hasher
from fedot.core.data.common.compatibility_rules import autodetect_tensor_data_type
from fedot.core.data.common.enums import StateEnum
from fedot.core.data.reader.data_reader import DataReader
from fedot.core.data.tensor_data.contracts import TensorDataContractError, validate_runtime_data
from fedot.core.data.tensor_data.data_spec import DataSpec
from fedot.core.data.tensor_data.lazy_tensor import LazyTensor
from fedot.core.data.tensor_data.preparation import PreparationRuntime, prepared_column_metadata
from fedot.core.data.tensor_data.rules import (
    build_creation_failure, build_creation_request, normalize_array_target_reference,
    normalize_tensordata_identity,
)
from fedot.core.data.tensor_data.source_layout import owned_array, prepare_source
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tools import transform_to_tensor


class TensorDataCreator:
    """Create the existing TensorData container without owning global backend state.

    Raw inputs and option collections are borrowed read-only. Prepared tensors
    and metadata are owned by the result. Fitted preparation is retained for
    create_data(..., from_data=train) and remains available through traces.
    """

    def __init__(self):
        self.spec = None

    def obligatory_preprocess(self):
        schema = prepare_source(self.spec)
        runtime = PreparationRuntime()
        if self.spec.state == StateEnum.FIT:
            params = {name: getattr(self.spec, name) for name in (
                'encoding_strategy', 'embedding_strategy', 'custom_strategy',
                'features_names', 'idx_mapping', 'data_type')}
            prepared, state, raw_hash = runtime.fit(
                self.spec.features, self.spec.target, schema, params, self.spec.use_cache)
        else:
            state = self.spec.preparation_state
            if state is None:
                state = runtime.from_trace(self.spec.trace_uuid, schema)
            raw_hash = Hasher.hash(self.spec.features, target=self.spec.target)
            prepared = runtime.transform(
                self.spec.features, self.spec.target, state, self.spec.idx_mapping)
        self.spec.preparation_state = state
        self.spec.plan_hash = state.plan_hash
        self.spec.raw_fingerprint = raw_hash
        if isinstance(prepared, TensorData):
            # Cached arrays are reusable, but request metadata is never borrowed
            # from a different call (row IDs, task and loader settings can differ).
            return replace(
                prepared, task=self.spec.task, data_type=self.spec.data_type,
                state=self.spec.state, idx=self.spec.idx,
                target_idx=self.spec.target_idx, dataloader_kwargs=self.spec.dataloader_kwargs,
                ts_orientation=self.spec.ts_orientation, ts_terms_idx=self.spec.ts_terms_idx,
                ts_forecast_horizon=self.spec.ts_forecast_horizon, ts_init_shape=self.spec.ts_init_shape,
                trace_uuid=self.spec.trace_uuid or prepared.trace_uuid, preparation_state=state,
            )
        self.spec.idx_mapping = prepared.idx_mapping
        self.spec.features, self.spec.target = transform_to_tensor(
            prepared.features, prepared.target, self.spec.ts_init_shape)
        self.spec.features_names, self.spec.categorical_idx, self.spec.numerical_idx = prepared_column_metadata(
            schema, state, prepared.idx_mapping, self.spec.features.shape[1])
        return None

    def preprocess_data(self):
        """Resolve identity, then run owned arrays on the fitted preparation backend."""
        if self.spec.data_type is None:
            self.spec.data_type = autodetect_tensor_data_type(self.spec.task)
        identity = normalize_tensordata_identity(
            self.spec.task, self.spec.data_type, self.spec.state)
        self.spec.task, self.spec.data_type, self.spec.state = identity.task, identity.data_type, identity.state
        features = self.spec.features
        width = features.shape[1] if features.ndim > 1 else 1
        self.spec.target, self.spec.target_idx = normalize_array_target_reference(
            self.spec.target, self.spec.target_idx, width)
        original_features, original_target = self.spec.features, self.spec.target
        state = self.spec.preparation_state
        backend = state.backend_name if state is not None else Backend().name
        with Backend().override(backend):
            try:
                self.spec.features = owned_array(
                    original_features, Backend().xp)
                self.spec.target = owned_array(original_target, Backend().xp)
            except (TypeError, ValueError):
                if Backend().name == 'cpu' or state is not None:
                    raise
                # Only raw object/string conversion may choose CPU fallback;
                # failures inside fitted handlers must remain visible.
                with Backend().override('cpu'):
                    self.spec.features = owned_array(
                        original_features, Backend().xp)
                    self.spec.target = owned_array(
                        original_target, Backend().xp)
                    return self.obligatory_preprocess()
            return self.obligatory_preprocess()

    def read_features(self, source_data):
        result = DataReader().read(source_data, self.spec)
        self.spec.features = result.features
        self.spec.features_names = deepcopy(result.features_names)

    def read_target(self):
        if self.spec.target is not None and self.spec.target_idx is None:
            # The historical integer shorthand is resolved after feature reading.
            if isinstance(self.spec.target, int) and not isinstance(self.spec.target, bool):
                return
            result = DataReader().read(self.spec.target, DataSpec())
            self.spec.target = result.features

    def to_tensor_data(self) -> TensorData:
        return TensorData(
            task=self.spec.task, data_type=self.spec.data_type, state=self.spec.state,
            idx=self.spec.idx, features=self.spec.features, target=self.spec.target,
            target_idx=self.spec.target_idx, categorical_idx=self.spec.categorical_idx,
            numerical_idx=self.spec.numerical_idx, features_names=self.spec.features_names,
            idx_mapping=self.spec.idx_mapping, ts_orientation=self.spec.ts_orientation,
            ts_terms_idx=self.spec.ts_terms_idx, ts_forecast_horizon=self.spec.ts_forecast_horizon,
            ts_init_shape=self.spec.ts_init_shape, dataloader_kwargs=self.spec.dataloader_kwargs,
            trace_uuid=self.spec.trace_uuid, preparation_state=self.spec.preparation_state,
        )

    def to_backend(self, tensor_data: TensorData) -> TensorData:
        # Move every tensor field, including CUDA ordinal and predictions.
        return tensor_data.to(Backend().device)

    @classmethod
    def create(cls, source_data, backend_name, **kwargs):
        """Read and prepare data; restore the caller's backend even on failure.

        Invalid input raises TensorDataContractError (a ValueError). Unexpected
        reader/handler failures keep their original exception as __cause__.
        Explicit idx labels align with samples, not feature columns.
        """
        request = build_creation_request(backend_name)
        creator = cls()
        creator.spec = DataSpec(**kwargs)
        if creator.spec.preparation_state is not None and creator.spec.state != StateEnum.PREDICT:
            raise TensorDataContractError(
                'invalid_state', 'state', 'fitted preparation can only be reused in predict state')
        try:
            with Backend().override(request.backend_name):
                creator.read_features(source_data)
                creator.read_target()
                tensor_data = creator.preprocess_data()
                if tensor_data is None:
                    tensor_data = creator.to_tensor_data()
                else:
                    creator.to_backend(tensor_data)
                    validate_runtime_data(tensor_data)
                    tensor_data.fingerprint = Hasher.hash(tensor_data)
                    return tensor_data
                creator.to_backend(tensor_data)
                validate_runtime_data(tensor_data)
                output_hash = Hasher.hash(tensor_data)
                tensor_data.fingerprint = output_hash
                Cacher(use_cache=creator.spec.use_cache).cache_tensor_data(
                    output_data=tensor_data, output_hash=output_hash,
                    input_hash=creator.spec.raw_fingerprint, operation_hash=creator.spec.plan_hash,
                    state=creator.spec.state.value,
                    trace_stage='obligatory_preprocessing' if creator.spec.state == StateEnum.FIT else None,
                )
                return tensor_data
        except TensorDataContractError:
            raise
        except Exception as error:
            failure = build_creation_failure(
                source_data, request.backend_name, error)
            raise ValueError(f'{failure.message}: {error}') from error

    @classmethod
    def create_lazy(cls, source_data, backend_name, **kwargs):
        """Delay reading; snapshot options but borrow source until get().

        Like an ordinary reader, the factory observes source changes made before
        materialization. Creation itself never mutates that source.
        """
        options = deepcopy(kwargs)
        return LazyTensor(lambda: cls.create(source_data, backend_name, **options))
