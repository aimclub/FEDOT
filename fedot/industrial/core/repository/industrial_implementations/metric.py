from fedot.core.composer.metrics import from_maximised_metric
from fedot.core.data.input_data.data import InputData, OutputData
from sklearn.metrics import accuracy_score, f1_score

from fedot.industrial.core.architecture.settings.computational import backend_methods as np


@from_maximised_metric
def metric_f1(reference: InputData, predicted: OutputData) -> float:
    n_classes = reference.num_classes
    binary_averaging_mode = 'binary'
    multiclass_averaging_mode = 'weighted'
    if n_classes > 2:
        additional_params = {'average': multiclass_averaging_mode}
    else:
        u, count = np.unique(np.ravel(reference.target), return_counts=True)
        count_sort_ind = np.argsort(count)
        pos_label = u[count_sort_ind[0]].item()
        additional_params = {
            'average': binary_averaging_mode, 'pos_label': pos_label}
    if len(predicted.predict.shape) > 1:
        predicted.predict = np.argmax(predicted.predict, axis=1)
    elif len(predicted.predict.shape) >= 2:
        predicted.predict = predicted.predict.squeeze()
        reference.target = reference.target.squeeze()
    return f1_score(y_true=reference.target, y_pred=predicted.predict,
                    **additional_params)


@from_maximised_metric
def metric_acc(reference: InputData, predicted: OutputData) -> float:
    try:
        if len(predicted.predict.shape) >= 2:
            if len(
                    reference.target.shape) <= 2 <= len(
                    predicted.predict.shape):
                predicted.predict = np.argmax(predicted.predict, axis=1)
            else:
                predicted.predict = predicted.predict.squeeze()
                reference.target = reference.target.squeeze()
        elif len(predicted.predict.shape) <= 2 and predicted.predict.dtype.name in ['float', 'float64']:
            predicted.predict = np.round(predicted.predict)

        return accuracy_score(
            y_true=reference.target,
            y_pred=predicted.predict)
    except Exception:
        _ = 1
