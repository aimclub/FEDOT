from random import seed

import numpy as np

from cases.oil.oil_pipeline import prepare_data, get_simple_pipeline, plot_lines
from fedot.api.main import Fedot
from fedot.core.repository.tasks import TsForecastingParams

seed(0)
np.random.seed(0)


def run_oil_modelling(well_id=0, with_automl=True):
    train_data, test_data = prepare_data(well_id)
    train_data_full, test_data_full = prepare_data(well_id, modify_target=False)

    initial_pipeline = get_simple_pipeline(train_data, well_id)

    if with_automl:
        automl = Fedot(problem='ts_forecasting', composer_params={'timeout': 5,
                                                                  'initial_pipeline': initial_pipeline},
                       preset='ts_tun',
                       task_params=TsForecastingParams(forecast_length=3), verbose_level=0)
        pipeline = automl.fit(train_data)
        pipeline.print_structure()
    else:
        pipeline = initial_pipeline
        pipeline.fit_from_scratch(train_data)

    train_data, test_data = prepare_data(well_id)

    pipeline.fit_from_scratch(train_data)
    predicted_test = pipeline.predict(test_data)
    pipeline.show()

    train_data_full = train_data_full[f'data_source_ts/inj_0']
    test_data_full = test_data_full[f'data_source_ts/inj_0']

    plot_lines(well_id, train_data_full, test_data_full, predicted_test)


if __name__ == '__main__':
    well_id = 0
    print('Without AutoML')
    run_oil_modelling(well_id=well_id, with_automl=False)
    print('With AutoML')
    run_oil_modelling(well_id=well_id, with_automl=True)
