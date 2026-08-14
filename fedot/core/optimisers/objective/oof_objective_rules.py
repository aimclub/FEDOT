from dataclasses import dataclass

import numpy as np
import torch

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
        predicted_target = torch.as_tensor(np.asarray(predicted.target))

    return OOFMetricData(
        reference=TensorData(
            idx=np.asarray(reference.idx),
            features=torch.as_tensor(np.asarray(reference.features)),
            target=torch.as_tensor(np.asarray(reference.target)),
            task=reference.task,
            data_type=reference.data_type,
        ),
        predicted=TensorData(
            idx=np.asarray(predicted.idx),
            features=torch.as_tensor(np.asarray(predicted.features)),
            target=predicted_target,
            predict=torch.as_tensor(np.asarray(predicted.predict)),
            task=predicted.task,
            data_type=predicted.data_type,
        ),
    )
