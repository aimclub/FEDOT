from dataclasses import replace

import torch

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.implementations.base import TensorDataOperationImplementation
from fedot.core.repository.tasks import Task, TaskTypesEnum


class DecomposerImplementation(TensorDataOperationImplementation):
    """Residual objective over a prepared feature branch and baseline prediction.

    Graph input preparation supplies the feature parent's schema in features
    and the model parent's prediction in predict. No parent masks or graph
    metadata are stored in TensorData or retained by the fitted operation.
    """

    def fit(self, data: TensorData):
        if data.predict is None:
            raise ValueError('Decompose requires a baseline model prediction.')
        return self

    def transform(self, data: TensorData) -> TensorData:
        # Preserve the historical inference convention: a provided target is
        # passed through, but residuals are only computed during training.
        return replace(data, predict=None, fingerprint=None)

    def transform_for_fit(self, data: TensorData) -> TensorData:
        target = data.target.reshape(len(data.features), -1)
        prediction = data.predict.reshape(len(data.features), -1)
        if target.shape != prediction.shape:
            raise ValueError('Regression decompose requires matching target and prediction columns.')
        return replace(self.transform(data), target=target - prediction)


class DecomposerRegImplementation(DecomposerImplementation):
    """Regression residuals, including multiple target columns."""


class DecomposerClassImplementation(DecomposerImplementation):
    """Classification probability residuals become a regression objective."""

    def fit(self, data: TensorData):
        super().fit(data)
        target = data.target.reshape(len(data.features), -1)
        self.classes_ = tuple(torch.unique(target[:, column], sorted=True) for column in range(target.shape[1]))
        if any(len(classes) < 2 for classes in self.classes_):
            raise ValueError('Classification decompose requires at least two observed classes per target.')
        return self

    def transform(self, data: TensorData) -> TensorData:
        return replace(super().transform(data), task=Task(TaskTypesEnum.regression))

    def transform_for_fit(self, data: TensorData) -> TensorData:
        target = data.target.reshape(len(data.features), -1)
        prediction = data.predict
        classes = [values.to(target.device) for values in self.classes_]
        if len(classes) == 1 and len(classes[0]) > 2:
            prediction = prediction.reshape(len(target), -1)
            if prediction.shape[1] != len(classes[0]):
                raise ValueError('Probability columns must match the fitted decompose class vocabulary.')
            encoded = (target == classes[0]).to(prediction.dtype)
        else:
            if any(len(values) != 2 for values in classes):
                raise ValueError('Multioutput multiclass decompose requires a separate probability-axis contract.')
            if len(classes) == 1:
                prediction = prediction.reshape(len(target), -1)
                if prediction.shape[1] == 2:
                    prediction = prediction[:, 1:2]
            elif prediction.ndim == 3 and prediction.shape[-1] == 2:
                prediction = prediction[..., 1]
            prediction = prediction.reshape(len(target), -1)
            if prediction.shape != target.shape:
                raise ValueError('Binary probability columns must match the number of targets.')
            positive = torch.stack([values[1] for values in classes])
            encoded = (target == positive).to(prediction.dtype)
        return replace(self.transform(data), target=encoded - prediction)
