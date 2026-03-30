import torch

from fedot.preprocessing.planner import build_optional_plan, PreprocessingPlan
from fedot.core.data.tensordata import TensorData
from fedot.preprocessing.preprocessor_types import PreprocessingStepEnum, ImputationMethodEnum


def test_build_optional_plan():
    tensor = torch.Tensor([[1, float('nan'), 3], [4, 5, 6]])
    data = TensorData.create(tensor, "cpu")
    pipeline = None

    optional_steps ={
        PreprocessingStepEnum.imputation: [
            {
                'method': ImputationMethodEnum.simple, 
            }
        ]
    }

    optional_plan = build_optional_plan(data, pipeline, optional_steps)
    
    assert isinstance(optional_plan, PreprocessingPlan)
    assert len(optional_plan.steps) == 1

if __name__ == "__main__":
    test_build_optional_plan()