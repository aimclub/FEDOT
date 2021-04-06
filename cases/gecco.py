from fedot.api.main import Fedot

# train_data = pd.read_csv(f'./data/ssh.csv')
# test_data = pd.read_csv(f'./data/ssh.csv')
problem = 'ts_forecasting'

auto_model = Fedot(problem=problem, learning_time=30, preset='light',
                   composer_params={'metric': ['rmse', 'node_num']}, seed=42, verbose_level=3)
auto_model.fit(features='./data/ssh.csv')

auto_model.best_models.show()
