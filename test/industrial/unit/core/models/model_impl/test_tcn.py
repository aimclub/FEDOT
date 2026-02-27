import numpy as np
import pytest
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TsForecastingParams, Task, TaskTypesEnum

from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels


@pytest.fixture(scope='session')
def ts():
    horizon = 5
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=horizon))
    ts = np.random.rand(100)
    train_input = InputData(idx=np.arange(0, len(ts)),
                            features=ts,
                            target=ts,
                            task=task,
                            data_type=DataTypesEnum.ts)
    return train_test_data_setup(train_input, validation_blocks=None)


def test_tsc_model(ts):
    train, test = ts
    with IndustrialModels():
        ppl = PipelineBuilder().add_node('tcn_model', params={'epochs': 10}).build()
        ppl.fit(train)
        predict = ppl.predict(test)

        assert predict.predict.size == 5
