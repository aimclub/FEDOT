from core.models.data import InputData, OutputData
from core.models.evaluation.evaluation import EvaluationStrategy
from core.repository.model_types_repository import ModelTypesIdsEnum
import numpy as np


def predict_data(predict_data: InputData):
    return predict_data.features


def predict_diff(predict_data: InputData):
    if predict_data.features.shape[1] != 1:
        raise ValueError('Too many inputs for the differential model')
    return predict_data.features[:, 0] - predict_data.target


def predict_add(predict_data: InputData):
    if predict_data.features.shape[1] != 2:
        raise ValueError('Too many inputs for the additive model')
    return np.sum(predict_data.features, axis=1)


class DataStrategy(EvaluationStrategy):
    _model_functions_by_type = {
        ModelTypesIdsEnum.datamodel: predict_data,
        ModelTypesIdsEnum.diff_data_model: predict_diff,
        ModelTypesIdsEnum.add_data_model: predict_add
    }

    def __init__(self, model_type: ModelTypesIdsEnum):
        self._model_specific_predict = self._model_functions_by_type[model_type]

    def fit(self, train_data: InputData):
        return None

    def predict(self, trained_model, predict_data: InputData):
        return self._model_specific_predict(predict_data)

    def tune(self, model, data_for_tune: InputData):
        return model
