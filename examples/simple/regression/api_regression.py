import numpy as np

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.utils import fedot_project_root


def run_regression_example():
    data_path = f'{fedot_project_root()}/cases/data/cholesterol/cholesterol.csv'

    data = InputData.from_csv(data_path)
    train, test = train_test_data_setup(data)
    problem = 'regression'

    composer_params = {'history_folder': 'custom_history_folder'}
    baseline_model = Fedot(problem=problem, composer_params=composer_params,
                           preset='stable')
    baseline_model.fit(features=train)

    baseline_model.predict(features=test)
    print(baseline_model.get_metrics())
    auto_model = Fedot(problem=problem, seed=42, timeout=1)
    auto_model.fit(features=train, target='target')
    prediction = auto_model.predict(features=test)
    print(auto_model.get_metrics())
    auto_model.plot_prediction()

    return prediction


if __name__ == '__main__':
    run_regression_example()
