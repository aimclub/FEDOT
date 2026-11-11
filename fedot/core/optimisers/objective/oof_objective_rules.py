from dataclasses import dataclass

import numpy as np
from fedot.core.backend.array_conversions import as_torch

from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.data.tensor_data import TensorData


@dataclass(frozen=True)
class OOFMetricData:
    reference: TensorData
    predicted: TensorData


def build_oof_metric_data(reference: InputData, predicted: OutputData) -> OOFMetricData:
    """Adapt legacy OOF execution results to the TensorData metric contract."""
    predicted_target = None
    if predicted.target is not None:
        predicted_target = as_torch(predicted.target)

    return OOFMetricData(
        reference=TensorData(
            idx=np.asarray(reference.idx),
            features=as_torch(reference.features),
            target=as_torch(reference.target),
            task=reference.task,
            data_type=reference.data_type,
        ),
        predicted=TensorData(
            idx=np.asarray(predicted.idx),
            features=as_torch(predicted.features),
            target=predicted_target,
            predict=as_torch(predicted.predict),
            task=predicted.task,
            data_type=predicted.data_type,
        ),
    )
