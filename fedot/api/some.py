import numpy as np
import torch
from dataclasses import replace

from fedot import Fedot


def _to_numpy(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


if __name__ == '__main__':
    features = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ], dtype=np.float32)
    target = np.array([0, 1, 0, 1], dtype=np.int64)

    model = Fedot(problem='classification', use_input_preprocessing=False)
    pipeline = model.fit_tensor_data(
        features=features,
        target=target,
        predefined_model='torch_linear',
    )

    fitted = pipeline.root_node.fitted_operation
    print('pipeline.is_fitted:', pipeline.is_fitted)
    print('fitted weights shape:', tuple(fitted.module.weight.shape))

    train_probs = model.predict_tensordata(model.train_data)
    train_labels = pipeline.predict_tensordata(
        model.train_data,
        output_mode='labels',
    ).predict
    print('probs:', _to_numpy(train_probs))
    print('labels:', _to_numpy(train_labels))

    test_features = np.array([
        [2.0, 3.0],
        [6.0, 7.0],
    ], dtype=np.float32)
    test_data = replace(
        model.train_data,
        features=torch.tensor(test_features, dtype=torch.float32),
        target=None,
        predict=None,
    )
    test_probs = model.predict_tensordata(test_data)
    test_labels = pipeline.predict_tensordata(
        test_data,
        output_mode='labels',
    ).predict
    print('test probabilities:', _to_numpy(test_probs))
    print('test labels:', _to_numpy(test_labels))
