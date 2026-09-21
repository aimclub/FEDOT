"""Fitted preparation owned by the existing TensorData creation boundary.

The planner and handlers remain the existing preprocessing implementations.
This adapter isolates their mutations and retains fitted state for from_data.
"""
from copy import deepcopy
from collections import Counter

import cloudpickle

from fedot.core.backend.backend import Backend, torch_to_xp
from fedot.core.caching.cacher import Cacher
from fedot.core.caching.cache_loader import Loader
from fedot.core.caching.hasher import Hasher
from fedot.core.caching.tracer import TraceBuilder
from fedot.core.data.prepared_data.prepared_data import PreparedData
from fedot.core.data.tensor_data.contracts import (
    FittedPreparationStep, PreparationState, TensorDataContractError,
)
from fedot.preprocessing.planner.obligatory_planner import build_obligatory_plan
from fedot.preprocessing.service.tabular_obligatory_service import ObligatoryTabularService
from fedot.preprocessing.tools.preprocessor_types import PreprocessingStepEnum
from fedot.preprocessing.tools.tools import update_handler_mapping


def _updated_mapping(mapping, indices, width, new_cols):
    """Preserve source ownership even when expansion keeps the same width."""
    if new_cols is None:
        if width == len(mapping):
            return dict(mapping)
        remaining = [source for pos, source in sorted(
            mapping.items()) if pos not in indices]
        if width != len(remaining):
            raise TensorDataContractError(
                'feature_alignment', 'idx_mapping',
                'a width-changing handler must report new_cols_dict')
        return dict(enumerate(remaining))
    if set(new_cols) != set(indices) or any(not isinstance(n, int) or n < 0 for n in new_cols.values()):
        raise TensorDataContractError(
            'feature_alignment', 'idx_mapping', 'invalid handler column expansion')
    sources = [source for pos, source in sorted(
        mapping.items()) if pos not in indices]
    for pos in indices:
        sources.extend([mapping[pos]] * new_cols[pos])
    if len(sources) != width:
        raise TensorDataContractError(
            'feature_alignment', 'idx_mapping', 'handler expansion does not match output width')
    return dict(enumerate(sources))


def _apply_handler(data, handler, indices, step, *, fit):
    count = data.features.shape[0]
    if (step.value if hasattr(step, 'value') else step) == PreprocessingStepEnum.target_encoding.value:
        if data.target is not None:
            target_data = PreparedData(features=data.target)
            result = handler.fit_transform(
                target_data, indices) if fit else handler.transform(target_data)
            data.target = result.features
        return data
    previous_mapping = dict(data.idx_mapping)
    data.new_cols_dict = None
    result = handler.fit_transform(
        data, indices) if fit else handler.transform(data)
    if result.features.shape[0] != count:
        raise TensorDataContractError(
            'row_alignment', 'features', 'preprocessing handlers must preserve sample rows')
    result.idx_mapping = _updated_mapping(
        previous_mapping, indices, result.features.shape[1], result.new_cols_dict)
    return result


def _feature_indices(step, mapping):
    source_indices = set(step.features_idx)
    indices = [pos for pos, source in sorted(
        mapping.items()) if source in source_indices]
    if source_indices - set(mapping.values()):
        raise TensorDataContractError(
            'unknown_column', 'features_idx', 'step refers to a removed source column')
    return indices


class PreparationRuntime:
    """Thin interpreter of existing preprocessing plans and fitted handlers."""

    def fit(self, features, target, schema, params, use_cache):
        plan = build_obligatory_plan(features, target, params)
        for step in plan.steps:
            if step.step != PreprocessingStepEnum.target_encoding:
                step.features_idx = [dict(schema.mapping)[pos]
                                     for pos in step.features_idx]
        raw_hash = Hasher.hash(features, target=target)
        # Cache lookup is sampled upstream. Fitted-state ownership needs the
        # complete source, including rows not visited by that lookup fingerprint.
        training_hash = Hasher.hash(features, target=target,
                                    min_rows=len(features), max_rows=len(features))
        plan_hash = Hasher.hash(plan)
        cacher = Cacher(use_cache=use_cache)
        # Only new, schema-bound artifacts are safe to reuse. Legacy artifacts
        # lack row/column ownership and an independently reusable fitted state.
        cached = cacher.load_tensor_data(features, plan, target)
        if cached.success:
            state = getattr(cached.data, 'preparation_state', None)
            if (state is not None and state.schema == schema
                    and state.backend_name == Backend().name and state.input_hash == training_hash):
                return cached.data, deepcopy(state), raw_hash

        cacher.cache_preprocessing_plan(plan=plan, plan_hash=plan_hash)
        service = ObligatoryTabularService(use_cache=use_cache)
        mapping = update_handler_mapping(plan, service.handler_mapping)
        prepared = PreparedData(features=features, target=target,
                                idx_mapping=dict(schema.mapping), ts_shape=features.shape)
        fitted_steps = []
        for order, step in enumerate(plan.steps):
            is_target = step.step == PreprocessingStepEnum.target_encoding
            indices = list(range(target.shape[1])) if is_target else _feature_indices(
                step, prepared.idx_mapping)
            handler = mapping[step.step][step.method](**step.step_args)
            prepared = _apply_handler(
                prepared, handler, indices, step.step, fit=True)
            fitted_steps.append(FittedPreparationStep(
                step.step.value, tuple(indices), cloudpickle.dumps(handler)))
            service._cache_fitted_model(
                cacher=cacher, input_hash=raw_hash, model=handler,
                operation_hash=plan_hash, step_order=order, step_name=step.step.value,
                method=step.method.value if hasattr(
                    step.method, 'value') else str(step.method),
                features_idx=indices,
            )
        state = PreparationState(schema, tuple(fitted_steps), cloudpickle.dumps(plan), plan_hash,
                                 Backend().name, training_hash)
        return prepared, state, raw_hash

    def transform(self, features, target, state, idx_mapping=None):
        prepared = PreparedData(features=features, target=target,
                                idx_mapping=dict(
                                    state.schema.mapping) if idx_mapping is None else dict(idx_mapping),
                                ts_shape=features.shape)
        for fitted in state.steps:
            prepared = _apply_handler(prepared, cloudpickle.loads(fitted.handler_state), list(fitted.indices),
                                      fitted.step, fit=False)
        return prepared

    @staticmethod
    def inverse_target(predict, state: PreparationState):
        """Decode labels with a private fitted handler, without disk lookup.

        Snapshot bytes belong to trusted training state, not an external data
        format. Decoder errors must remain visible rather than falling back to
        potentially unrelated cache entries.
        """
        if predict is None:
            return predict
        fitted = next((step for step in state.steps if
                       step.step == PreprocessingStepEnum.target_encoding.value), None)
        if fitted is None:
            return predict
        handler = cloudpickle.loads(fitted.handler_state)
        with Backend().override(state.backend_name):
            xp = Backend().xp
            values = torch_to_xp(predict, xp) if hasattr(
                predict, 'detach') else predict
            values = xp.array(values, copy=True)
            squeeze = values.ndim == 1
            features = values.reshape(-1, 1) if squeeze else values
            decoded = handler.inverse_transform(
                PreparedData(features=features)).features
            return decoded.reshape(-1) if squeeze or decoded.shape[1] == 1 else decoded

    def from_trace(self, trace_uuid, schema):
        """Compatibility path for callers supplying only an explicit trace."""
        if trace_uuid is None:
            raise TensorDataContractError(
                'missing_preparation', 'trace_uuid',
                'trace_uuid is required for predict without from_data preparation')
        trace = TraceBuilder.from_trace_uuid(trace_uuid)
        stage = ObligatoryTabularService._get_train_obligatory_stage(trace)
        plan = Loader.load(stage.operation_path, kind='preprocessing_plan')
        if plan is None:
            raise TensorDataContractError(
                'missing_preparation', 'trace_uuid', 'fitted plan is unavailable')
        references = sorted(stage.models, key=lambda ref: ref.step_order)
        if [ref.step_order for ref in references] != list(range(len(plan.steps))):
            raise TensorDataContractError(
                'missing_preparation', 'trace_uuid', 'fitted handler set is incomplete')
        fitted = []
        for ref in references:
            step = plan.steps[ref.step_order]
            handler = Loader.load(ref.model_path, kind='preprocessing_model')
            if handler is None:
                raise TensorDataContractError(
                    'missing_preparation', 'trace_uuid', 'fitted handler is unavailable')
            indices = ref.features_idx if ref.features_idx is not None else step.features_idx
            fitted.append(FittedPreparationStep(step.step.value,
                          tuple(indices), cloudpickle.dumps(handler)))
        return PreparationState(schema, tuple(fitted), cloudpickle.dumps(plan), stage.operation_hash,
                                Backend().name, stage.input_hash)


def prepared_column_metadata(schema, state, mapping, width):
    """Final feature labels and categorical ownership, in output order."""
    categorical_sources = set(schema.categorical)
    for step in state.plan.steps:
        if step.step != PreprocessingStepEnum.target_encoding:
            categorical_sources.update(step.features_idx)
    categorical = [pos for pos, source in sorted(
        mapping.items()) if source in categorical_sources]
    numerical = [pos for pos in range(width) if pos not in categorical]
    names = None
    if schema.names is not None:
        source_names = {source: name for (
            _, source), name in zip(schema.mapping, schema.names)}
        counts = Counter(mapping.values())
        reserved = {source_names[source]
                    for source, count in counts.items() if count == 1}
        occurrences = Counter()
        names = []
        for _, source in sorted(mapping.items()):
            name = source_names[source]
            if counts[source] > 1:
                while True:
                    candidate = f'{name}__{occurrences[source]}'
                    occurrences[source] += 1
                    if candidate not in reserved:
                        name = candidate
                        break
            reserved.add(name)
            names.append(name)
    return names, categorical, numerical
