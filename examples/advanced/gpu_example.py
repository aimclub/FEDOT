"""The same model ID and preset on CPU or CUDA; requires the matching optional GPU stack."""
import argparse

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from fedot import Fedot


def run_example(device='cuda:0'):
    features, target = make_moons(n_samples=1000, noise=0.1, random_state=137)
    train_x, test_x, train_y, test_y = train_test_split(features, target, random_state=137, stratify=target)
    model = Fedot(problem='classification', preset='fast_train', timeout=1,
                  tensor_data_config={'backend_name': device})
    train = model.create_data(train_x, target=train_y)
    model.fit(train, predefined_model='rf')
    test = model.create_data(test_x, target=test_y, from_data=train)
    prediction = model.predict(test)
    print(model.current_pipeline.root_node.fitted_operation._fedot_execution.resolution)
    print(model.get_metrics(test))
    return prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0', help='cpu or a CUDA device such as cuda:0')
    run_example(parser.parse_args().device)
