from fedot.core.data.data import InputData
from fedot.core.log import Log
from fedot.core.repository.operation_types_repository import \
    OperationMetaInfo, OperationTypesRepository
from fedot.core.repository.tasks import Task
from fedot.core.operations.operation import Operation, _eval_strategy_for_task, DEFAULT_PARAMS_STUB


class DataOperation(Operation):
    """
    Class with fit/predict methods defining the evaluation strategy for the task

    :param operation_type: str type of the data operation defined in data_operation
    repository
    :param log: Log object to record messages
    """

    def __init__(self, operation_type: str, log: Log = None):
        super().__init__(operation_type, log)

    def _init(self, task: Task, **kwargs):
        operations_repo = OperationTypesRepository(repository_name='data_operation_repository.json')
        params_for_fit = None
        if self.params != DEFAULT_PARAMS_STUB:
            params_for_fit = self.params

        try:
            self._eval_strategy = _eval_strategy_for_task(self.operation_type,
                                                          task.task_type,
                                                          operations_repo)(
                self.operation_type,
                params_for_fit)
        except Exception as ex:
            self.log.error(f'Can not find evaluation strategy because of {ex}')
            raise ex

        if 'output_mode' in kwargs:
            self._eval_strategy.output_mode = kwargs['output_mode']

    @property
    def acceptable_task_types(self):
        operations_repo = OperationTypesRepository(repository_name='data_operation_repository.json')
        operation_info = operations_repo.operation_info_by_id(self.operation_type)
        return operation_info.task_type

    @property
    def metadata(self) -> OperationMetaInfo:
        operations_repo = OperationTypesRepository(repository_name='data_operation_repository.json')
        operation_info = operations_repo.operation_info_by_id(self.operation_type)
        if not operation_info:
            raise ValueError(f'Data operation {self.operation_type} not found')
        return operation_info

    def fit(self, data: InputData, is_fit_chain_stage: bool = True):
        """
        This method is used for defining and running of the evaluation strategy
        to train the operation with the data provided

        :param data: data used for operation training
        :return: tuple of trained operation and prediction on train data
        :param is_fit_chain_stage: is this fit or predict stage for chain
        """
        self._init(data.task)

        fitted_operation = self._eval_strategy.fit(train_data=data)

        predict_train = self.predict(fitted_operation, data, is_fit_chain_stage)

        return fitted_operation, predict_train

    def predict(self, fitted_operation, data: InputData,
                is_fit_chain_stage: bool, output_mode: str = 'default'):
        """
        This method is used for defining and running of the evaluation strategy
        to predict with the data provided

        :param fitted_operation: trained operation object
        :param data: data used for prediction
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :param output_mode: string with information about output of operation,
        for example, is the operation predict probabilities or class labels
        """
        self._init(data.task, output_mode=output_mode)

        prediction = self._eval_strategy.predict(trained_operation=fitted_operation,
                                                 predict_data=data,
                                                 is_fit_chain_stage=is_fit_chain_stage)

        return prediction
