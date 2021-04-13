from fedot.api.main import Fedot

# train_data = pd.read_csv(f'./data/ssh.csv')
# test_data = pd.read_csv(f'./data/ssh.csv')
problem = 'ts_forecasting'

auto_model = Fedot(problem=problem, learning_time=2,
                   composer_params={'available_model_types': ['linear', 'lasso', 'ridge', 'xgbreg',
                                                              'knnreg'],
                                    'metric': ['rmse', 'node_num']}, seed=42, verbose_level=3)
auto_model.fit(features='./data/ssh.csv')

auto_model.best_models.show()

print(auto_model.best_models)

for num, chain in enumerate(auto_model.best_models):
    chain.save(f'{num}')
