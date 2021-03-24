import pandas as pd
from sklearn.model_selection import train_test_split
from fedot.api.main import Fedot
from fedot.core.utils import project_root


def run_classification_example():
    train_data_path = f'{project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{project_root()}/cases/data/scoring/scoring_test.csv'

    problem = 'classification'

    baseline_model = Fedot(problem=problem)
    baseline_model.fit(features=train_data_path, target='target', predefined_model='xgboost')

    baseline_model.predict_proba(features=test_data_path)
    print(baseline_model.get_metrics())

    auto_model = Fedot(problem=problem, seed=42)
    auto_model.fit(features=train_data_path, target='target')
    prediction = auto_model.predict_proba(features=test_data_path)
    print(auto_model.get_metrics())

    return prediction


def run_regression_example():
    data_path = f'{project_root()}/cases/data/cholesterol/cholesterol.csv'
    chol_data = pd.read_csv(data_path, sep=',')

    chol_data_train, chol_data_test = train_test_split(chol_data, test_size=0.3)

    problem = 'regression'

    baseline_model = Fedot(problem=problem)
    baseline_model.fit(features=chol_data_train, target='target', predefined_model='xgboost')

    baseline_model.predict_proba(features=chol_data_test)
    print(baseline_model.get_metrics())

    auto_model = Fedot(problem=problem, seed=42)
    auto_model.fit(features=chol_data_train, target='target')
    prediction = auto_model.predict_proba(features=chol_data_test)
    print(auto_model.get_metrics())

    return prediction


def run_ts_forecasting_example(with_plot=True):
    train_data_path = f'{project_root()}/notebooks/jupyter_media/intro/salaries.csv'

    # init model for the time series forecasting
    model = Fedot(problem='ts_forecasting')

    # run AutoML model design in the same way
    chain = model.fit(features=train_data_path, target='target')
    chain.show()

    # use model to obtain forecast
    forecast = model.forecast(pre_history=train_data_path, forecast_length=30)

    # plot forecasting result
    if with_plot:
        model.plot_prediction()

    return forecast

def run_classification_multiobj_example(with_plot=True):
    train_data = pd.read_csv(f'./data/Hill_Valley_with_noise_Training.data')
    test_data = pd.read_csv(f'./data/Hill_Valley_with_noise_Testing.data')
    target = test_data['class']
    del test_data['class']
    problem = 'classification'

    auto_model = Fedot(problem=problem, learning_time=10, preset='light',
                       composer_params={'metric': ['f1', 'node_num']}, seed=42)
    auto_model.fit(features=train_data, target='class')
    prediction = auto_model.predict_proba(features=test_data)
    print(auto_model.get_metrics(target))

    if with_plot:
        auto_model.best_models.show()

    return prediction


if __name__ == '__main__':
    run_classification_example()
    run_classification_multiobj_example()
    run_ts_forecasting_example()
    run_regression_example()
