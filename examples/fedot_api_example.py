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


if __name__ == '__main__':
    run_classification_example()

    run_ts_forecasting_example()
