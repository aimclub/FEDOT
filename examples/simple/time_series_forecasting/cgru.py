import numpy as np
from sklearn.metrics import mean_absolute_error

from examples.advanced.time_series_forecasting.composing_pipelines import visualise, get_border_line_info
from examples.simple.time_series_forecasting.api_forecasting import get_ts_data
from fedot.core.composer.metrics import root_mean_squared_error
from fedot.core.pipelines.pipeline_builder import PipelineBuilder


def cgru_forecasting():
    """ Example of cgru pipeline serialization """
    horizon = 12
    window_size = 200
    train_data, test_data, _ = get_ts_data('beer', horizon)

    pipeline = (PipelineBuilder()
                .add_node("lagged", params={'window_size': window_size})
                .add_node("cgru")
                .build())

    pipeline.fit(train_data)
    prediction = pipeline.predict(test_data).predict

    plot_info = [
        {'idx': np.concatenate([train_data.idx, test_data.idx]),
         'series': np.concatenate([test_data.features, test_data.target]),
         'label': 'Actual time series'},
        {'idx': test_data.idx,
         'series': np.ravel(prediction),
         'label': 'prediction'},
        get_border_line_info(test_data.idx[0],
                             prediction,
                             np.ravel(np.concatenate([test_data.features, test_data.target])),
                             'Border line')
    ]

    rmse = root_mean_squared_error(test_data.target, prediction)
    mae = mean_absolute_error(test_data.target, prediction)
    print(f'RMSE - {rmse:.4f}')
    print(f'MAE - {mae:.4f}')

    visualise(plot_info)


if __name__ == '__main__':
    cgru_forecasting()
