import numpy as np

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
    print('train target:', _to_numpy(model.train_data.target))

    test_features = features.copy()
    test_features[0] += 1.0
    test_features[-1] -= 1.0

    test_prediction = model.predict_tensordata(test_features)
    test_labels = model.predict_tensordata(test_features, output_mode='labels')
    print('test features:\n', test_features)
    print('test probabilities:', _to_numpy(test_prediction.predict))
    print('test labels:', _to_numpy(test_labels.predict))
