import numpy as np

from fedot import Fedot, create_data


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

    model = Fedot(problem='classification')
    tensor_data = model.create_data(features, target=target)
    pipeline = model.fit(
        tensor_data,
        predefined_model='torch_linear',
    )

    fitted = pipeline.root_node.fitted_operation
    print('pipeline.is_fitted:', pipeline.is_fitted)
    print('fitted weights shape:', tuple(fitted.module.weight.shape))
    print('train target:', _to_numpy(model.train_data.target))

    test_features = features.copy()
    test_features[0] += 1.0
    test_features[-1] -= 1.0

    test_tensor_data = create_data(test_features, from_data=tensor_data)
    test_prediction = model.predict(test_tensor_data)
    print('test features:\n', test_features)
    print('test probabilities:', _to_numpy(test_prediction.predict))
