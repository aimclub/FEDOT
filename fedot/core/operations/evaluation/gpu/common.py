import warnings
from typing import Optional

import cudf
import cuml
from cuml import KMeans, Ridge, LogisticRegression, Lasso, ElasticNet, \
    MBSGDClassifier, MBSGDRegressor, CD
from cuml.ensemble import RandomForestClassifier, RandomForestRegressor
from cuml.svm import SVC
from cuml.neighbors import KNeighborsClassifier as CuMlknnClassifier, \
    KNeighborsRegressor as CuMlknnRegressor
from cuml import LinearRegression as CuMlLinReg, SGD as CuMlSGD, \
    MultinomialNB as CuMlMultinomialNB

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import SkLearnEvaluationStrategy
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum


class CuMLEvaluationStrategy(SkLearnEvaluationStrategy):
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
        'linear': CuMlLinReg,
        'rf': RandomForestClassifier,
        'rfr': RandomForestRegressor,
        'svc': SVC,
        'knn': CuMlknnClassifier,
        'knnreg': CuMlknnRegressor,
        'sgd': CuMlSGD,
        'multinb': CuMlMultinomialNB,
        'elasticnet': ElasticNet,
        'mbsgdclass': MBSGDClassifier,
        'mbsgdcregr': MBSGDRegressor,
        'cd': CD


    }

    def __init__(self, operation_type: str, params: Optional[dict] = None):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)
        self.operation_id = operation_type
        cuml.set_global_output_type('numpy')

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided
        :param InputData train_data: data used for operation training
        :return: trained cuML operation
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.params_for_fit:
            operation_implementation = self.operation_impl(**self.params_for_fit)
        else:
            operation_implementation = self.operation_impl()

        # If model doesn't support multi-output and current task is ts_forecasting
        current_task = train_data.task.task_type
        models_repo = OperationTypesRepository()
        non_multi_models, _ = models_repo.suitable_operation(task_type=current_task,
                                                             tags=['non_multi'])
        is_model_not_support_multi = self.operation_type in non_multi_models
        features = cudf.DataFrame(train_data.features.astype('float32'))
        target = cudf.Series(train_data.target.flatten().astype('float32'))

        if is_model_not_support_multi and current_task == TaskTypesEnum.ts_forecasting:
            raise NotImplementedError('Not supported for GPU yet')
            # TODO Manually wrap the regressor into multi-output model
        else:
            operation_implementation.fit(features, target)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_pipeline_stage: bool) -> OutputData:
        """
        This method used for prediction of the target data.
        :param trained_operation: operation object
        :param predict_data: data to predict
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return OutputData: passed data with new predicted target
        """
        raise NotImplementedError()

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain cuML strategy for {operation_type}')
