import numpy as np
from sklearn.datasets import load_iris

from fedot.api.main import Fedot


def run_gpu_example():
    synthetic_data = load_iris()
    features = np.asarray(synthetic_data.data).astype(np.float32)
    features_test = np.asarray(synthetic_data.data).astype(np.float32)
    target = synthetic_data.target

    problem = 'classification'

    baseline_model = Fedot(problem=problem, preset='gpu')
    baseline_model.fit(features=features_test, target=target, predefined_model='rf')

    baseline_model.predict(features=features)
    print(baseline_model.get_metrics())

    if False:
        auto_model = Fedot(problem=problem, seed=42, learning_time=1)
        auto_model.fit(features=features, target=target)
        prediction = auto_model.predict_proba(features=features_test)
        print(auto_model.get_metrics())


if __name__ == '__main__':
    run_gpu_example()
