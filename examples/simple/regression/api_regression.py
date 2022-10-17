import logging

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.utils import fedot_project_root


def run_regression_example(visualise=False):
    data_path = f'{fedot_project_root()}/cases/data/cholesterol/cholesterol.csv'

    data = InputData.from_csv(data_path,
                              task=Task(TaskTypesEnum.regression))
    train, test = train_test_data_setup(data)
    problem = 'regression'

    composer_params = {'history_folder': 'custom_history_folder', 'preset': 'auto'}
    auto_model = Fedot(problem=problem, seed=42, timeout=2, logging_level=logging.INFO,
                       **composer_params)

    auto_model.fit(features=train, target='target')
    prediction = auto_model.predict(features=test)
    if visualise:
        auto_model.history.save('saved_regression_history.json')
        auto_model.plot_prediction()
    print(auto_model.get_metrics())
    return prediction


if __name__ == '__main__':
    run_regression_example(visualise=True)
