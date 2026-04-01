from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Optional

import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum


INDUSTRIAL_INCEPTION_NN_OPERATION = 'industrial_inception_nn'
INDUSTRIAL_RESNET_NN_OPERATION = 'industrial_resnet_nn'


@dataclass(frozen=True)
class IndustrialNNBridgeSpec:
    operation_type: str
    module_name: str
    class_name: str
    default_params: dict[str, Any]


_INDUSTRIAL_NN_BRIDGE_SPECS = {
    INDUSTRIAL_INCEPTION_NN_OPERATION: IndustrialNNBridgeSpec(
        operation_type=INDUSTRIAL_INCEPTION_NN_OPERATION,
        module_name='fedot.industrial.core.models.nn.network_impl.common_model.inception',
        class_name='InceptionTimeModel',
        default_params={
            'epochs': 5,
            'batch_size': 32,
            'learning_rate': 0.001,
        },
    ),
    INDUSTRIAL_RESNET_NN_OPERATION: IndustrialNNBridgeSpec(
        operation_type=INDUSTRIAL_RESNET_NN_OPERATION,
        module_name='fedot.industrial.core.models.nn.network_impl.common_model.resnet',
        class_name='ResNetModel',
        default_params={
            'epochs': 5,
            'batch_size': 32,
            'learning_rate': 0.001,
            'model_name': 'ResNet18',
        },
    ),
}


def resolve_industrial_nn_bridge_spec(operation_type: str) -> IndustrialNNBridgeSpec:
    try:
        return _INDUSTRIAL_NN_BRIDGE_SPECS[operation_type]
    except KeyError as exc:
        supported = ', '.join(sorted(_INDUSTRIAL_NN_BRIDGE_SPECS))
        raise ValueError(
            f'Unsupported industrial NN bridge operation: {operation_type}. '
            f'Expected one of: {supported}'
        ) from exc


def resolve_industrial_nn_model_class(operation_type: str):
    spec = resolve_industrial_nn_bridge_spec(operation_type)
    module = import_module(spec.module_name)
    return getattr(module, spec.class_name)


def build_industrial_nn_bridge_params(operation_type: str,
                                      user_params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    spec = resolve_industrial_nn_bridge_spec(operation_type)
    return {
        **spec.default_params,
        **dict(user_params or {}),
    }


def normalize_industrial_nn_prediction(prediction: Any,
                                       reference_data: InputData,
                                       output_mode: str) -> OutputData:
    if isinstance(prediction, OutputData):
        base_output = prediction
    else:
        base_output = OutputData(
            idx=reference_data.idx,
            features=reference_data.features,
            predict=prediction,
            task=reference_data.task,
            target=reference_data.target,
            data_type=DataTypesEnum.table,
            supplementary_data=reference_data.supplementary_data,
        )

    if reference_data.task.task_type is not TaskTypesEnum.classification:
        return base_output

    normalized_predict = _normalize_classification_predict(base_output.predict, output_mode=output_mode)
    return OutputData(
        idx=base_output.idx,
        features=base_output.features,
        predict=normalized_predict,
        task=base_output.task,
        target=base_output.target,
        data_type=base_output.data_type,
        supplementary_data=base_output.supplementary_data,
    )


def _normalize_classification_predict(predict: Any, output_mode: str):
    predict_array = np.array(predict)

    if output_mode == 'labels':
        if predict_array.ndim > 1:
            return np.argmax(predict_array, axis=1)
        return (predict_array > 0.5).astype(int)

    if output_mode in ('default', 'probs') and predict_array.ndim > 1 and predict_array.shape[1] == 2:
        return predict_array[:, 1]

    return predict_array
