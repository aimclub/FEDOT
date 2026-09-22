import numpy as np

from fedot.api.api_utils.api_run_planner import plan_chunked_ensemble
from fedot.core.data.input_data.data import InputData
from fedot.core.pipelines.ensembling.utils import prepare_chunked_ensemble_validation
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def _make_classification_data():
    return InputData(
        idx=np.arange(10),
        features=np.arange(20).reshape(10, 2),
        target=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )


def test_prepare_chunked_ensemble_validation_uses_holdout_split():
    data = _make_classification_data()
    plan = plan_chunked_ensemble(
        should_run_sampling_stage=True,
        strategy_kind='chunking',
        task_type=TaskTypesEnum.classification,
    )

    prepared = prepare_chunked_ensemble_validation(data, plan)

    assert len(prepared.train_data.idx) == 8
    assert len(prepared.validation_data.idx) == 2
    assert set(prepared.train_data.idx).isdisjoint(set(prepared.validation_data.idx))
    assert set(prepared.class_representatives.keys()) == {0, 1}
