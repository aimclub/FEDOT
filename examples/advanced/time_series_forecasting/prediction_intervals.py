import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fedot.api.main import Fedot
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import TsForecastingParams, Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root

# class for making prediction intervals
from fedot.core.pipelines.prediction_intervals.main import PredictionIntervals
from fedot.core.pipelines.prediction_intervals.params import PredictionIntervalsParams
# metrics to evaluate results
from fedot.core.pipelines.prediction_intervals.metrics import interval_score, picp


def build_pred_ints(start=5000, end=7000, horizon=200):

    d = pd.read_csv(f'{fedot_project_root()}/examples/data/ts/ts_long.csv')
    init_series = d[d['series_id'] == 'temp']['value'].to_numpy()

    ts = init_series[start:end]
    ts_test = init_series[end:end + horizon]

    fig, ax = plt.subplots()
    ax.plot(range(len(ts)), ts)
    ax.plot(range(len(ts), len(ts) + len(ts_test)), ts_test)

    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=horizon))
    idx = np.arange(len(ts))
    train_input = InputData(idx=idx,
                            features=ts,
                            target=ts,
                            task=task,
                            data_type=DataTypesEnum.ts)
    model = Fedot(problem='ts_forecasting',
              task_params=task.task_params,
              timeout=3,
              preset='ts',
              show_progress=False)

    model.fit(train_input)
    model.forecast()

    # initilize PredictionIntervals instance
    params = PredictionIntervalsParams(number_mutations=50, show_progress=False, mutations_choice='different')
    pred_ints = PredictionIntervals(model=model,
                                    horizon=horizon,
                                    method='mutation_of_best_pipeline',
                                    params=params)

    pred_ints.fit(train_input)

    x = pred_ints.forecast()
    pred_ints.plot(ts_test=ts_test)
    pred_ints.get_base_quantiles(train_input)
    pred_ints.plot_base_quantiles()

    # Evaluate results using metrcis picp (predicition interval coverage probability) and interval_score,
    # see  https://arxiv.org/pdf/2007.05709.pdf

    print(f'''Evaluate results using metrcis picp (predicition interval coverage probability) and interval_score,
see https://arxiv.org/pdf/2007.05709.pdf
interval_score: {interval_score(ts_test,up=x['up_int'],low=x['low_int'])}
picp: {picp(ts_test,low = x['low_int'],up=x['up_int'])}
''')


if __name__ == '__main__':
    build_pred_ints()
