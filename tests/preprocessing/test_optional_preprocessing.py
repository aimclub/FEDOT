import numpy as np
import torch

from fedot.preprocessing.mapping import PREPROCESSING_OPTIONAL_MAPPING
from fedot.preprocessing.preprocessor_types import PreprocessingStepEnum, ImputationMethodEnum
from fedot.preprocessing.service import OtionalPreprocessingService
from fedot.core.data.prepared_data import PreparedData
from fedot.core.data.tensordata import TensorData
from fedot.preprocessing.planner import build_optional_plan, PreprocessingPlan
from fedot.preprocessing.preprocessor_types import PreprocessingStepEnum, ImputationMethodEnum


def test_build_optional_plan():
    tensor = torch.Tensor([[1, float('nan'), 3], [4, 5, 6]])
    data = TensorData.create(tensor, "cpu")
    pipeline = None

    optional_steps ={
        PreprocessingStepEnum.imputation: [
            {
                'method': ImputationMethodEnum.mean, 
                'features_idx': [0],
                'step_args': None
            }
        ]
    }

    optional_plan = build_optional_plan(data, pipeline, optional_steps)
    
    assert isinstance(optional_plan, PreprocessingPlan)
    assert len(optional_plan.steps) == 1


def test_mean_imputation():
    X = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ])

    td = TensorData.create(X, backend_name="cpu")
    preprocessor = PREPROCESSING_OPTIONAL_MAPPING[PreprocessingStepEnum.imputation][ImputationMethodEnum.mean]()
    preprocessed_data = preprocessor.fit_transform(td, [1])
    assert preprocessed_data.features[1, 1] == 5


def test_preprocessing_plan_imputation():
    X = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ])

    td = TensorData.create(X, backend_name="cpu")
    service = OtionalPreprocessingService()
    preprocessed_data = service.fit_transform(td, None, {PreprocessingStepEnum.imputation: None})
    assert isinstance(preprocessed_data, PreparedData)
    assert preprocessed_data.features[1, 1] == 5


def test_preprocessing_plan_mode_imputation():
    X = np.array([
        [1, 2, 3],
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ])

    td = TensorData.create(X, backend_name="cpu")
    service = OtionalPreprocessingService()
    preprocessed_data = service.fit_transform(td, None, {
        PreprocessingStepEnum.imputation: [{"method": ImputationMethodEnum.mode, 
                                           "features_idx": [1],
                                           "step_args": None}]})
    assert isinstance(preprocessed_data, PreparedData)
    assert preprocessed_data.features[2, 1] == 2


def test_preprocessing_plan_mean_imputation():
    X = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ])

    td = TensorData.create(X, backend_name="cpu")
    service = OtionalPreprocessingService()
    preprocessed_data = service.fit_transform(td, None, {
        PreprocessingStepEnum.imputation: [{"method": ImputationMethodEnum.mean, 
                                           "features_idx": [1],
                                           "step_args": None}]})
    assert isinstance(preprocessed_data, PreparedData)
    assert preprocessed_data.features[1, 1] == 5


def test_preprocessing_plan_constant_imputation():
    X = np.array([
        [1, 2, 3],
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ])

    td = TensorData.create(X, backend_name="cpu")
    service = OtionalPreprocessingService()
    preprocessed_data = service.fit_transform(td, None, {
        PreprocessingStepEnum.imputation: [{"method": ImputationMethodEnum.constant, 
                                           "features_idx": [1],
                                           "step_args": {"constant": 3}}]})
    assert isinstance(preprocessed_data, PreparedData)
    assert preprocessed_data.features[2, 1] == 3


def test_preprocessing_plan_delete_raw_imputation():
    X = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9]
    ])

    td = TensorData.create(X, backend_name="cpu")
    service = OtionalPreprocessingService()
    preprocessed_data = service.fit_transform(td, None, {
        PreprocessingStepEnum.imputation: [{"method": ImputationMethodEnum.delete_raw, 
                                           "features_idx": [1],
                                           "step_args": None}]})
    assert isinstance(preprocessed_data, PreparedData)
    assert preprocessed_data.features.shape[0] == 2
    assert preprocessed_data.target.shape[0] == 2
