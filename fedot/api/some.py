import numpy as np

from fedot import Fedot
from fedot.core.data.common.enums import StateEnum
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator

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

    tensor_data = TensorDataCreator.create(
        features, 
        target=target, 
        backend_name='cpu',
    )
    model = Fedot(problem='classification', use_input_preprocessing=True)
    pipeline = model.fit_tensor_data(
        tensor_data=tensor_data,
        predefined_model='torch_linear',
    )

    fitted = pipeline.root_node.fitted_operation
    print('pipeline.is_fitted:', pipeline.is_fitted)
    print('fitted weights shape:', tuple(fitted.module.weight.shape))
    print('train target:', _to_numpy(model.train_data.target))

    test_features = features.copy()
    test_features[0] += 1.0
    test_features[-1] -= 1.0

    test_tensor_data = TensorDataCreator.create(
        test_features,
        backend_name='cpu',
        task=tensor_data.task,
        state=StateEnum.PREDICT,
        trace_uuid=tensor_data.trace_uuid,
    )

    test_prediction = model.predict_tensordata(test_tensor_data)
    test_labels = model.predict_tensordata(test_tensor_data, output_mode='labels')
    print('test features:\n', test_features)
    print('test probabilities:', _to_numpy(test_prediction.predict))
    print('test labels:', _to_numpy(test_labels.predict))
