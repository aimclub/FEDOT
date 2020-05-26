from core.models.data import (
    InputData,
)
from core.models.evaluation.data_evaluation_strategies import DataModellingStrategy
from core.models.evaluation.evaluation import AutoMLEvaluationStrategy, EvaluationStrategy, KerasForecastingStrategy, \
    SkLearnClassificationStrategy, SkLearnClusteringStrategy, SkLearnRegressionStrategy, StatsModelsForecastingStrategy
from core.repository.dataset_types import DataTypesEnum
from core.repository.model_types_repository import ModelMetaInfo, ModelTypesIdsEnum, ModelTypesRepository
from core.repository.tasks import Task, TaskTypesEnum, compatible_task_types
import numpy as np

DEFAULT_PARAMS_STUB = 'default_params'


class Model:
    def __init__(self, model_type: ModelTypesIdsEnum):
        self.model_type = model_type
        self._eval_strategy, self._data_preprocessing = None, None
        self.params = DEFAULT_PARAMS_STUB

        # additional params that can be changed for test and debug purposes (num of fitting epoch, etc)
        self.external_params = {}

    @property
    def acceptable_task_types(self):
        _, model_info = ModelTypesRepository().search_models(
            desired_ids=[self.model_type])
        return model_info[0].task_type

    def compatible_task_type(self, base_task_type: TaskTypesEnum):
        # if the model can't be used directly for the task type from data
        if base_task_type not in self.acceptable_task_types:
            # search the supplementary task types, that can be included in chain which solves original task
            globally_compatible_task_types = compatible_task_types(base_task_type)
            compatible_task_types_acceptable_for_model = list(set(self.acceptable_task_types).intersection
                                                              (set(globally_compatible_task_types)))
            if len(compatible_task_types_acceptable_for_model) == 0:
                raise ValueError(f'Model {self.model_type} can not be used as a part of {base_task_type}.')
            task_type_for_model = compatible_task_types_acceptable_for_model[0]
            return task_type_for_model
        return base_task_type

    @property
    def metadata(self) -> ModelMetaInfo:
        _, model_info = ModelTypesRepository().search_models(
            desired_ids=[self.model_type])
        return model_info[0]

    def output_datatype(self, input_datatype: DataTypesEnum) -> DataTypesEnum:
        output_types = self.metadata.output_types
        if input_datatype in output_types:
            return input_datatype
        else:
            return output_types[0]

    @property
    def description(self):
        model_type = self.model_type
        model_params = self.params
        return f'n_{model_type}_{model_params}'

    def _init(self, task: Task):
        self._eval_strategy = self._eval_strategy_for_task(task)
        self._eval_strategy = _insert_external_params(self._eval_strategy, self.external_params)

    def fit(self, data: InputData):
        self._init(data.task)

        fitted_model = self._eval_strategy.fit(train_data=data)
        predict_train = self._eval_strategy.predict(trained_model=fitted_model,
                                                    predict_data=data)
        return fitted_model, predict_train

    def predict(self, fitted_model, data: InputData):
        self._init(data.task)

        prediction = self._eval_strategy.predict(trained_model=fitted_model,
                                                 predict_data=data)

        if np.array([np.isnan(_) for _ in prediction]).any():
            return np.nan_to_num(prediction)

        return prediction

    def fine_tune(self, data: InputData, iterations: int = 30):
        self._init(data.task)

        try:
            fitted_model, tuned_params = self._eval_strategy.fit_tuned(train_data=data,
                                                                       iterations=iterations)
            self.params = tuned_params
            if self.params is None:
                self.params = DEFAULT_PARAMS_STUB
        except Exception as ex:
            print(f'Tuning failed because of {ex}')
            fitted_model = self._eval_strategy.fit(train_data=data)
            self.params = DEFAULT_PARAMS_STUB

        predict_train = self._eval_strategy.predict(trained_model=fitted_model,
                                                    predict_data=data)
        return fitted_model, predict_train

    def __str__(self):
        return f'{self.model_type.name}'

    def _eval_strategy_for_task(self, task: Task):
        # TODO refactor
        strategies_for_tasks = {
            TaskTypesEnum.classification: [SkLearnClassificationStrategy, AutoMLEvaluationStrategy,
                                           DataModellingStrategy],
            TaskTypesEnum.regression: [SkLearnRegressionStrategy, DataModellingStrategy],
            TaskTypesEnum.ts_forecasting: [StatsModelsForecastingStrategy, DataModellingStrategy,
                                           KerasForecastingStrategy],
            TaskTypesEnum.clustering: [SkLearnClusteringStrategy]
        }

        models_for_strategies = {
            SkLearnClassificationStrategy: [ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.knn, ModelTypesIdsEnum.logit,
                                            ModelTypesIdsEnum.dt, ModelTypesIdsEnum.rf, ModelTypesIdsEnum.mlp,
                                            ModelTypesIdsEnum.lda, ModelTypesIdsEnum.qda, ModelTypesIdsEnum.svc,
                                            ModelTypesIdsEnum.bernb],
            AutoMLEvaluationStrategy: [ModelTypesIdsEnum.tpot, ModelTypesIdsEnum.h2o],
            SkLearnClusteringStrategy: [ModelTypesIdsEnum.kmeans],
            SkLearnRegressionStrategy: [ModelTypesIdsEnum.linear, ModelTypesIdsEnum.ridge,
                                        ModelTypesIdsEnum.lasso, ModelTypesIdsEnum.rfr,
                                        ModelTypesIdsEnum.linear, ModelTypesIdsEnum.ridge, ModelTypesIdsEnum.lasso,
                                        ModelTypesIdsEnum.xgbreg, ModelTypesIdsEnum.adareg, ModelTypesIdsEnum.gbr,
                                        ModelTypesIdsEnum.knnreg, ModelTypesIdsEnum.dtreg, ModelTypesIdsEnum.treg,
                                        ModelTypesIdsEnum.svr, ModelTypesIdsEnum.sgdr],
            KerasForecastingStrategy: [ModelTypesIdsEnum.lstm],
            StatsModelsForecastingStrategy: [ModelTypesIdsEnum.ar, ModelTypesIdsEnum.arima],
            DataModellingStrategy: [ModelTypesIdsEnum.direct_datamodel,
                                    ModelTypesIdsEnum.trend_data_model,
                                    ModelTypesIdsEnum.residual_data_model,
                                    ModelTypesIdsEnum.additive_data_model]
        }

        eval_strategies = strategies_for_tasks[self.compatible_task_type(task.task_type)]

        for strategy in eval_strategies:
            if self.model_type in models_for_strategies[strategy]:
                eval_strategy = strategy(self.model_type)
                return eval_strategy

        raise ValueError(f'Strategy for the {self.model_type} in {task.task_type} not found')


def _insert_external_params(strategy: EvaluationStrategy, ext_params: dict):
    if 'epochs' in ext_params:
        strategy.epochs = ext_params['epochs']
    if 'max_run_time_sec' in ext_params:
        strategy.max_time_min = ext_params['max_run_time_sec'] / 60
    return strategy
