import warnings
from typing import Optional

import cudf
from cuml import KMeans
from cuml import Ridge, LogisticRegression, Lasso
from cuml.ensemble import RandomForestClassifier, RandomForestRegressor

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum


class CuMLEvaluationStrategy(EvaluationStrategy):
    """
    This class defines the certain operation implementation for the GPU-based CuML operations
    defined in operation repository
    :param str operation_type: str type of the operation defined in operation or
    data operation repositories
    :param dict params: hyperparameters to fit the operation with
    """
    __operations_by_types = {
        'ridge': Ridge,
        'lasso': Lasso,
        'logit': LogisticRegression,
        'rf': RandomForestClassifier,
        'rfr': RandomForestRegressor,
        'kmeans': KMeans,
    }

    def __init__(self, operation_type: str, params: Optional[dict] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        self.operation_id = operation_type
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided
        :param InputData train_data: data used for operation training
        :return: trained cuML operation
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # if self.params_for_fit:
        #    operation_implementation = self.operation_impl(**self.params_for_fit)
        # else:
        operation_implementation = self.operation_impl()

        # If model doesn't support multi-output and current task is ts_forecasting
        current_task = train_data.task.task_type
        models_repo = OperationTypesRepository()
        non_multi_models, _ = models_repo.suitable_operation(task_type=current_task,
                                                             tags=['non_multi'])
        is_model_not_support_multi = self.operation_type in non_multi_models
        features = cudf.DataFrame(train_data.features)
        target = cudf.Series(train_data.target.flatten())

        if is_model_not_support_multi and current_task == TaskTypesEnum.ts_forecasting:
            raise NotImplementedError('Not supported for GPU yet')
            # Manually wrap the regressor into multi-output model
            # operation_implementation = convert_to_multivariate_model(operation_implementation,
            #                                                         train_data)
        else:
            operation_implementation.fit(features, target)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool) -> OutputData:
        """
        This method used for prediction of the target data.
        :param trained_operation: operation object
        :param predict_data: data to predict
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return OutputData: passed data with new predicted target
        """
        raise NotImplementedError()

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain cuML strategy for {operation_type}')

    def _find_operation_by_impl(self, impl):
        for operation, operation_impl in self.__operations_by_types.items():
            if operation_impl == impl:
                return operation

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))
