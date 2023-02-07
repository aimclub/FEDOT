from examples.simple.time_series_forecasting.ts_pipelines import cgru_pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from test.unit.tasks.test_forecasting import get_ts_data


def test_cgru_forecasting():
    horizon = 5
    n_steps = 100
    train_data, test_data = get_ts_data(n_steps=n_steps + horizon, forecast_length=horizon)

    pipeline = PipelineBuilder().add_node('lagged').add_node('cgru').build()
    pipeline.fit(train_data)
    predicted = pipeline.predict(test_data).predict[0]

    assert len(predicted) == horizon


def test_cgru_in_pipeline():
    horizon = 5
    n_steps = 100
    train_data, test_data = get_ts_data(n_steps=n_steps + horizon, forecast_length=horizon)

    pipeline = cgru_pipeline()
    pipeline.fit(train_data)
    predicted = pipeline.predict(test_data).predict[0]

    assert len(predicted) == horizon
