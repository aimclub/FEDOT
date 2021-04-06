import pandas as pd

from fedot.api.main import Fedot

train_data = pd.read_csv(f'./data/ssh.data')
test_data = pd.read_csv(f'./data/ssh.data')
problem = 'ts_forecasting'

auto_model = Fedot(problem=problem, learning_time=1, preset='light',
                   composer_params={'metric': ['rmse', 'node_num']}, seed=42)
auto_model.fit(features=train_data, target='class')

auto_model.best_models.show()
