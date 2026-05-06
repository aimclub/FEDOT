from fedot.core.data.bridges.input_to_tensor_rules import build_input_data_tensor_bridge_plan
from fedot.core.data.input_data.input_data_descriptor import build_input_data_descriptor
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.core.data.common.enums import StateEnum


def input_data_to_tensordata(input_data, backend_name: str, state=StateEnum.FIT):
    descriptor = build_input_data_descriptor(input_data)
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
    )
