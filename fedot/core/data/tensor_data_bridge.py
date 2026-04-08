from fedot.core.data.data import InputData
from fedot.core.data.tensor_data_bridge_rules import build_tensordata_input_bridge_plan


def tensordata_to_input_data(tensor_data) -> InputData:
    plan = build_tensordata_input_bridge_plan(tensor_data)

    return InputData(
        idx=plan.idx,
        features=plan.features,
        target=plan.target,
        task=plan.task,
        data_type=plan.data_type,
        features_names=plan.features_names,
        categorical_idx=plan.categorical_idx,
    )
