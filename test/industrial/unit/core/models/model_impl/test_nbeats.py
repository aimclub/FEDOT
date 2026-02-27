import numpy as np
import pytest
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TsForecastingParams, Task, TaskTypesEnum

from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels


@pytest.fixture(scope='session')
def ts_input_data():
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

# def test_nbeats_model(ts_input_data):


def nbeats_model(ts_input_data):
    IndustrialModels().setup_repository()
    train, test = ts_input_data
    model = PipelineBuilder().add_node('nbeats_model', params=dict(
        backcast_length=10,
        forecast_length=5,
        epochs=10
    )).build()

    model.fit(train)
    forecast = model.predict(test)

    assert len(forecast.predict) == 5
