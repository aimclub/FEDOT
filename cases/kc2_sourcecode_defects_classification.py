from os.path import join

from fedot import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.utils import fedot_project_root


def get_kc2_data():
    file_path = 'cases/data/kc2/kc2.csv'
    full_path = join(str(fedot_project_root()), file_path)
    task = Task(TaskTypesEnum.classification)
    data = InputData.from_csv(full_path, task=task, target_columns='problems')

    target = data.target
    encoded = (target == 'yes').astype(int)
    data.target = encoded

    train, test = train_test_data_setup(data, shuffle=True)

    return train, test


def run_classification(train_data, test_data,
                       timeout: float = 5, visualize=False):
    auto_model = Fedot(problem='classification',
                       timeout=timeout, n_jobs=8,
                       early_stopping_iterations=None, )
    auto_model.fit(features=train_data.features, target=train_data.target)
    prediction = auto_model.predict(features=test_data.features)
    metrics = auto_model.get_metrics(target=test_data.target)
    print(metrics)
    if visualize:
        auto_model.current_pipeline.show()
        auto_model.plot_prediction()
    return prediction


def run_classification_baseline(train_data, test_data, timeout: float = 5):
    baseline_model = Fedot(problem='classification', timeout=timeout)
    baseline_model.fit(features=train_data.features, target=train_data.target,
                       predefined_model='rf')
    baseline_model.predict(features=test_data.features)
    metrics = baseline_model.get_metrics(target=test_data.target)
    print(metrics)


if __name__ == '__main__':
    train_data, test_data = get_kc2_data()
    run_classification_baseline(train_data, test_data)
    run_classification(train_data, test_data,
                       timeout=5.0, visualize=True)
