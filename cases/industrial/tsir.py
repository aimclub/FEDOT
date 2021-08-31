import sys
import traceback

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from fedot.api.main import Fedot
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
from fedot.core.repository.tasks import TsForecastingParams


def call_timeseries(problem='ts_forecasting', column='sepal length (cm)'):
    df_el, _ = load_iris(return_X_y=True, as_frame=True)

    features = df_el[column].values
    target = features

    try:

        auto_model = Fedot(problem=problem, timeout=0.2, task_params=TsForecastingParams(forecast_length=14),
                           composer_params={'metric': 'rmse'}, preset='light_tun', verbose_level=4)

        pipeline = auto_model.fit(features=features, target=target)

        max_lag = 28
        for n in pipeline.nodes:
            if str(n.operation) == 'lagged':
                if n.custom_params['window_size'] > max_lag:
                    n.custom_params['window_size'] = max_lag
        pipeline.print_structure()
        pipeline.fit_from_scratch(auto_model.train_data)

        auto_model_prediction = in_sample_ts_forecast(pipeline, auto_model.train_data,
                                                      horizon=round(target.shape[0] - max_lag - 14))

        plt.plot(auto_model_prediction)
        plt.show()
    except Exception as e:
        print(traceback.format_exc())
        raise e.with_traceback(sys.exc_info()[2])


if __name__ == '__main__':
    call_timeseries()
