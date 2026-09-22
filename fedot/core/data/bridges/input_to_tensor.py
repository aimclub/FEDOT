from fedot.core.data.bridges.input_to_tensor_rules import build_input_data_tensor_bridge_plan
from fedot.core.data.input_data.input_data_descriptor import build_input_data_descriptor
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.core.data.common.enums import StateEnum
from fedot.core.data.tensor_data.contracts import TensorDataContractError
from fedot.core.repository.dataset_types import DataTypesEnum


def input_data_to_tensordata(input_data, backend_name: str, state=StateEnum.FIT):
    """Bridge sample-indexed data without reinterpreting temporal indices.

    Legacy 1D InputData forecasting uses an index per timestep; TensorData uses
    one label per sample. That temporal-index layout is not a supported bridge
    round trip and must be prepared explicitly with create_data instead.
    """
    descriptor = build_input_data_descriptor(input_data)
    if (descriptor.tensor_canonical_data_type == DataTypesEnum.ts
            and input_data.features.ndim == 1 and input_data.idx is not None
            and len(input_data.idx) != 1):
        raise TensorDataContractError(
            'temporal_index_bridge', 'idx',
            'per-timestep InputData indices cannot be used as TensorData sample labels; '
            'prepare the series explicitly with create_data')
    plan = build_input_data_tensor_bridge_plan(
        descriptor=descriptor,
        target=input_data.target,
        state=state,
    )

    return TensorDataCreator.create(
        input_data.features,
        backend_name=backend_name,
        task=plan.task,
        data_type=plan.data_type,
        state=plan.state,
        target=plan.target,
        features_names=plan.features_names,
        categorical_idx=plan.categorical_idx,
        idx=input_data.idx,
    )
