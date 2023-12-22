import warnings
from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy, SkLearnEvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.data_operations.decompose \
    import DecomposerRegImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_filters \
    import IsolationForestRegImplementation
from fedot.core.operations.evaluation.operation_implementations. \
    data_operations.sklearn_filters import LinearRegRANSACImplementation, NonLinearRegRANSACImplementation
from fedot.core.operations.evaluation.operation_implementations. \
    data_operations.sklearn_selectors import LinearRegFSImplementation, NonLinearRegFSImplementation
from fedot.core.operations.evaluation.operation_implementations.models.atomized.atomized_ts_differ import \
    AtomizedTimeSeriesDiffer
from fedot.core.operations.evaluation.operation_implementations.models.atomized.atomized_ts_sampler import \
    AtomizedTimeSeriesSampler
from fedot.core.operations.evaluation.operation_implementations.models.atomized.atomized_ts_scaler import \
    AtomizedTimeSeriesScaler
from fedot.core.operations.evaluation.operation_implementations.models.atomized.atomized_ts_transform_to_time import \
    AtomizedTimeSeriesToTime
from fedot.core.operations.evaluation.operation_implementations.models.knn import FedotKnnRegImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.utilities.random import ImplementationRandomStateHandler

warnings.filterwarnings("ignore", category=UserWarning)


class FedotAtomizedStrategy(EvaluationStrategy):
    _operations_by_types = {
        'atomized_ts_differ': AtomizedTimeSeriesDiffer,
        'atomized_ts_scaler': AtomizedTimeSeriesScaler,
        'atomized_ts_sampler': AtomizedTimeSeriesSampler,
        'atomized_ts_to_time': AtomizedTimeSeriesToTime,
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        model = self.operation_impl(self.params_for_fit.get('pipeline'))
        return model.fit(train_data)

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        prediction = trained_operation.predict(predict_data)
        return prediction

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        prediction = trained_operation.predict_for_fit(predict_data)
        return prediction
