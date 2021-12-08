from fedot.core.repository.tasks import TsForecastingParams
from fedot.api.main import Fedot

main_params = {'problem': 'ts_forecasting', 'preset': None, 'seed': None, 'timeout': 0.5}
composer_params = {'with_tuning': False}
fit_params = {'features': 'C:\\Users\\yulas\\Documents\\fedot\\cases\\data\\metocean\\metocean_data_train.csv', 'target': 'sea_height'}
main_params['composer_params'] = composer_params
main_params['task_params'] = TsForecastingParams(forecast_length=30)
model = Fedot(**main_params)
model.fit(**fit_params)