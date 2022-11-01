import pandas as pd

from fedot.api.main import Fedot
from fedot.core.utils import fedot_project_root


def run_classification_multiobj_example(visualization=False, timeout=1):
    train_data = pd.read_csv(f'{fedot_project_root()}/examples/data/Hill_Valley_with_noise_Training.data')
    test_data = pd.read_csv(f'{fedot_project_root()}/examples/data/Hill_Valley_with_noise_Testing.data')
    target = test_data['class']
    del test_data['class']
    problem = 'classification'

    metric_names = ['f1', 'node_num']
    auto_model = Fedot(problem=problem, timeout=timeout, preset='best_quality', seed=42,
                       metric=metric_names,
                       with_tuning=False)
    auto_model.fit(features=train_data, target='class')
    prediction = auto_model.predict_proba(features=test_data)
    print(auto_model.get_metrics(target))

    if visualization:
        auto_model.plot_prediction()
        auto_model.plot_pareto()

    return prediction


if __name__ == '__main__':
    run_classification_multiobj_example(visualization=True)
