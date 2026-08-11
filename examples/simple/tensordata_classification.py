"""Minimal Fedot classification on TensorData with a predefined torch model.

This example builds a small tabular classification dataset as ``TensorData``,
fits a predefined ``torch_linear`` model through the Fedot API, and prints
label / probability predictions.
"""

from copy import deepcopy

import numpy as np

from fedot import Fedot, create_data


def _to_numpy(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def run_tensordata_classification_example():
    """Fit ``torch_linear`` on TensorData and predict labels / probabilities.

    Returns:
        Tuple containing:
            - fitted pipeline;
            - prediction with class labels;
            - prediction with class probabilities.
    """
    features = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        dtype=np.float32,
    )
    target = np.array([0, 1, 0, 1], dtype=np.int64)

    # Public factory: backend='cpu' by default.
    # Equivalent when a Fedot instance already exists (injects problem /
    # tensor_data_config):
    #   train_data = model.create_data(features, target=target)
    #   test_data = model.create_data(test_features, from_data=train_data)
    train_data = create_data(features, target=target)

    model = Fedot(problem='classification')
    pipeline = model.fit(
        train_data,
        predefined_model='torch_linear',
    )

    fitted = pipeline.root_node.fitted_operation
    print('pipeline.is_fitted:', pipeline.is_fitted)
    print('fitted weights shape:', tuple(fitted.module.weight.shape))
    print('train target:', _to_numpy(train_data.target))

    test_features = features.copy()
    test_features[0] += 1.0
    test_features[-1] -= 1.0

    test_data = create_data(test_features, from_data=train_data)

    # Classification output modes (labels/probs) are handled at the operation layer.
    labels = pipeline.root_node.predict(
        tensor_data=deepcopy(test_data),
        output_mode='labels',
    )
    probabilities = pipeline.root_node.predict(
        tensor_data=deepcopy(test_data),
        output_mode='probs',
    )

    print('test features:\n', test_features)
    print('test labels:', _to_numpy(labels.predict))
    print('test probabilities:', _to_numpy(probabilities.predict))
    return pipeline, labels, probabilities


if __name__ == '__main__':
    run_tensordata_classification_example()
