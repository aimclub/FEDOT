from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.utils import fedot_project_root


def run_regression_example():
    data_path = f'{fedot_project_root()}/cases/data/cholesterol/cholesterol.csv'

    data = InputData.from_csv(data_path,
                              task=Task(TaskTypesEnum.regression))
    train, test = train_test_data_setup(data)
    problem = 'regression'

    composer_params = {'history_folder': 'custom_history_folder'}
    auto_model = Fedot(problem=problem, seed=42, composer_params=composer_params,
                       preset='auto',
                       timeout=2, verbose_level=1)

    auto_model.fit(features=train, target='target')
    auto_model.history.save('saved_regression_history.json')
    prediction = auto_model.predict(features=test)
    print(auto_model.get_metrics())
    auto_model.plot_prediction()

    return prediction


if __name__ == '__main__':
    run_regression_example()
