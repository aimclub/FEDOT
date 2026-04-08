import os
import warnings
from copy import deepcopy
from functools import partial
from typing import Union, Optional

import numpy as np
import pandas as pd
from fedot.api.main import Fedot
from fedot.core.data.data import OutputData, InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.visualisation.pipeline_specific_visuals import PipelineHistoryVisualizer
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from pymonad.either import Either
from sklearn import model_selection as skms
from sklearn.calibration import CalibratedClassifierCV

from fedot.industrial.api.main_rules import (
    build_industrial_explain_plan,
    build_industrial_finetune_plan,
    build_industrial_fit_plan,
    build_industrial_history_visualization_plan,
    build_industrial_load_plan,
    build_industrial_metrics_plan,
    build_industrial_metrics_request_plan,
    build_industrial_predict_plan,
    build_industrial_predict_proba_plan,
    build_industrial_save_plan,
    normalize_industrial_prediction,
    trim_industrial_forecast,
)
from fedot.industrial.api.utils.api_init import ApiManager
from fedot.industrial.api.utils.checkers_collections import DataCheck
from fedot.industrial.core.architecture.abstraction.decorators import DaskServer, exception_handler
from fedot.industrial.core.architecture.pipelines.classification import (
    SklearnCompatibleClassifier,
)
from fedot.industrial.core.repository.constanst_repository import FEDOT_GET_METRICS
from fedot.industrial.core.repository.industrial_implementations.abstract import build_tuner
from fedot.industrial.core.repository.initializer_industrial_models import IndustrialModels

warnings.filterwarnings("ignore")


class FedotIndustrial(Fedot):
    """Main class for Industrial API. It provides a high-level interface for working with the
    Fedot framework. The class allows you to train, predict, and evaluate models for time series.
    All arguments are passed as keyword arguments and handled by the ApiManager class.

    Args:
        problem: str. The type of task to solve. Available options: 'ts_forecasting', 'ts_classification', 'ts_regression'.
        timeout: int. Time for model design (in minutes): ``None`` or ``-1`` means infinite time.
                logging_level: logging levels are the same as in
            `built-in logging library <https://docs.python.org/3/library/logging.html>`_.

            .. details:: Possible options:

                - ``50`` -> critical
                - ``40`` -> error
                - ``30`` -> warning
                - ``20`` -> info
                - ``10`` -> debug
                - ``0`` -> nonset
        backend_method: str. Default `cpu`. The method for backend. Available options: 'cpu', 'dask'.
        initial_assumption: Pipeline = None. The initial pipeline for the model.
        optimizer_params: dict = None.
        task_params: dict = None.
        strategy: str = None.
        strategy_params: dict = None.
        available_operations: list = None.
        output_folder: str = './output'.

    Example:
        First, configure experiment and instantiate FedotIndustrial class::

            from fedot.industrial.api.main import FedotIndustrial
            from fedot.industrial.tools.loader import DataLoader


            industrial = FedotIndustrial(problem='ts_classification',
                                         timeout=15,
                                         n_jobs=2,
                                         logging_level=20)

        Next, download data from UCR archive::

            train_data, test_data = DataLoader(dataset_name='ItalyPowerDemand').load_data()

        Finally, fit the model and get predictions::

            model = industrial.fit(train_features=train_data[0], train_target=train_data[1])
            labels = industrial.predict(test_features=test_data[0])
            probs = industrial.predict_proba(test_features=test_data[0])
            metric = industrial.get_metrics(target=test_data[1], metric_names=['f1', 'roc_auc'])

    """

    def __init__(self, **kwargs):
        super(Fedot, self).__init__()
        self.manager = ApiManager().build(kwargs)
        self.logger = self.manager.logger

    def __init_industrial_backend(self, input_data: Optional[Union[InputData, np.array]] = None):
        self.logger.info('-' * 50)
        self.logger.info('Initialising Industrial Repository')
        if self.manager.industrial_config.is_default_fedot_context:
            self.logger.info(f'-------------------------------------------------')
            self.logger.info('Initialising Fedot Evolutionary Optimisation params')
            self.repo = IndustrialModels().setup_default_repository()
            self.manager.automl_config.optimisation_strategy = self.manager.optimisation_agent['Fedot']
        else:
            self.logger.info(f'-------------------------------------------------')
            self.logger.info('Initialising Industrial Evolutionary Optimisation params')
            self.repo = IndustrialModels().setup_repository(backend=self.manager.compute_config.backend)
            optimisation_agent = self.manager.automl_config.optimisation_strategy['optimisation_agent']
            optimisation_params = self.manager.automl_config.optimisation_strategy['optimisation_strategy']
            self.manager.automl_config.optimisation_strategy = partial(
                self.manager.optimisation_agent[optimisation_agent],
                optimisation_params=optimisation_params)
        return input_data

    def __init_solver(self, input_data: Optional[Union[InputData, np.array]] = None):
        self.logger.info('-' * 50)
        self.logger.info('Initialising Dask Server')
        if self.manager.automl_config.config['initial_assumption'] is None:
            self.manager.automl_config.config['initial_assumption'] = \
                self.manager.industrial_config.config['initial_assumption'].build()
        else:
            self.manager.automl_config.config['initial_assumption'] = \
                self.manager.automl_config.config['initial_assumption'].build()
        dask_server = DaskServer(self.manager.compute_config.distributed)
        self.manager.dask_client = dask_server.client
        self.manager.dask_cluster = dask_server.cluster
        self.logger.info(f'Link Dask Server - {self.manager.dask_client.dashboard_link}')
        self.logger.info('-' * 50)
        self.logger.info('Initialising solver')
        self.manager.solver = Fedot(
            **self.manager.learning_config.config['learning_strategy_params'],
            metric=self.manager.learning_config.config['optimisation_loss'],
            problem=self.manager.automl_config.config['task'],
            task_params=self.manager.industrial_config.task_params
            if self.manager.industrial_config.is_forecasting_context else self.manager.automl_config.config
            ['task_params'], optimizer=self.manager.automl_config.optimisation_strategy,
            available_operations=self.manager.automl_config.config['available_operations'],
            initial_assumption=self.manager.automl_config.config['initial_assumption'])
        return input_data

    def _process_input_data(self, input_data):
        train_data, self.target_encoder = Either.insert(input_data).then(lambda data: deepcopy(data)). \
            then(lambda data: DataCheck(input_data=data, task=self.manager.automl_config.config['task'],
                                        task_params=self.manager.automl_config.config['task_params'], fit_stage=True,
                                        industrial_task_params=self.manager.industrial_config.strategy_params)). \
            then(lambda data_cls: (data_cls.check_input_data(), data_cls.get_target_encoder())).value
        train_data.features = train_data.features.squeeze() if self.manager.industrial_config.is_default_fedot_context \
            else train_data.features
        return train_data

    def __calibrate_probs(self, probability_model, predict_data):
        model_sklearn = SklearnCompatibleClassifier(probability_model)
        train_idx, test_idx = skms.train_test_split(self.train_data.idx,
                                                    train_size=0.8,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle=True)
        X_train, y_train = self.train_data.features[train_idx, :, :], self.train_data.target[train_idx]
        X_val, y_val = self.train_data.features[test_idx, :, :], self.train_data.target[test_idx]
        train_data_for_calibration = (X_train, y_train)
        val_data = (X_val, y_val)
        model_sklearn.fit(train_data_for_calibration[0], train_data_for_calibration[1])
        cal_clf = CalibratedClassifierCV(model_sklearn, method="sigmoid", cv="prefit")
        cal_clf.fit(val_data[0], val_data[1])
        # calibrated prediction
        calibrated_proba = cal_clf.predict_proba(predict_data.features)
        return calibrated_proba

    def __predict_for_ensemble(self):
        predict = self.manager.industrial_config.strategy.predict(self.predict_data, 'probs')
        ensemble_strat = self.manager.industrial_config.strategy.ensemble_strategy
        predict = {strategy: np.argmax(self.manager.industrial_config.strategy.ensemble_predictions(
            predict, strategy), axis=1) for strategy in ensemble_strat}
        return predict

    def __abstract_predict(self, predict_data, predict_mode):
        solver_is_fedot = self.manager.condition_check.solver_is_fedot_class(self.manager.solver)
        solver_is_pipeline = self.manager.condition_check.solver_is_pipeline_class(self.manager.solver)
        have_encoder = self.manager.condition_check.solver_have_target_encoder(self.target_encoder)
        predict_plan = build_industrial_predict_plan(
            predict_mode=predict_mode,
            solver_is_fedot_class=solver_is_fedot,
            solver_is_pipeline_class=solver_is_pipeline,
            has_target_encoder=have_encoder,
            predict_task=predict_data.task,
        )

        def _inverse_encoder_transform(predict):
            predicted_labels = self.target_encoder.inverse_transform(predict)
            self.predict_data.target = self.target_encoder.inverse_transform(self.predict_data.target)
            return predicted_labels

        def predict_func(predict_from_solver):
            if predict_plan.use_pipeline_predict_mode:
                return self.manager.solver.predict(predict_from_solver, predict_mode)
            if predict_plan.labels_output:
                return self.manager.solver.predict(predict_from_solver)
            return self.manager.solver.predict_proba(predict_from_solver)

        raw_predict = Either(value=predict_data,
                             monoid=[predict_data, predict_plan.custom_predict]).either(
            left_function=lambda predict_from_solver: predict_func(predict_from_solver),
            right_function=lambda predict_from_custom: self.manager.solver.predict(predict_from_custom))
        normalized_predict = normalize_industrial_prediction(raw_predict)
        if have_encoder:
            normalized_predict = _inverse_encoder_transform(normalized_predict)
        return trim_industrial_forecast(normalized_predict, predict_plan.forecast_length)

    def _metric_evaluation_loop(self,
                                target,
                                predicted_labels,
                                predicted_probs,
                                problem,
                                metric_names,
                                rounding_order,
                                train_data,
                                seasonality):
        metrics_plan = build_industrial_metrics_plan(
            target=target,
            predicted_labels=predicted_labels,
            has_target_encoder=self.manager.condition_check.solver_have_target_encoder(self.target_encoder),
        )
        if metrics_plan.prediction_is_mapping:
            metric_dict = {model_name: FEDOT_GET_METRICS[problem](target=target,
                                                                  metric_names=metric_names,
                                                                  rounding_order=rounding_order,
                                                                  labels=model_result,
                                                                  probs=predicted_probs) for model_name, model_result
                           in predicted_labels.items()}
            return metric_dict

        if metrics_plan.use_target_encoder:
            new_target = self.target_encoder.transform(target.flatten())
            labels = self.target_encoder.transform(predicted_labels).reshape(metrics_plan.valid_shape)
        else:
            new_target = target.flatten()
            labels = predicted_labels.reshape(metrics_plan.valid_shape)

        return FEDOT_GET_METRICS[problem](target=new_target,
                                          metric_names=metric_names,
                                          rounding_order=rounding_order,
                                          labels=labels,
                                          probs=predicted_probs,
                                          train_data=train_data,
                                          seasonality=seasonality)

    def fit(self,
            input_data: tuple,
            **kwargs):
        """
        Method for training Industrial model.

        Args:
            input_data: tuple with train_features and train_target
            **kwargs: additional parameters

        """
        fit_plan = build_industrial_fit_plan(self.manager.industrial_config.strategy)

        with exception_handler(Exception, on_exception=self.shutdown, suppress=False):
            train_data = self._process_input_data(input_data)
            train_data = self.__init_industrial_backend(train_data)
            train_data = self.__init_solver(train_data)
            if fit_plan.use_solver_fit:
                self.manager.solver.fit(train_data)
            else:
                self.manager.industrial_config.strategy.fit(train_data)

    def predict(self,
                predict_data: tuple,
                predict_mode: str = 'labels',
                **kwargs):
        """
        Method to obtain prediction labels from trained Industrial model.

        Args:
            predict_mode: ``default='default'``. Defines the mode of prediction. Could be 'default' or 'probs'.
            predict_data: tuple with test_features and test_target

        Returns:
            the array with prediction values

        """
        self.repo = IndustrialModels().setup_repository(backend=self.manager.compute_config.backend)
        processed_input = self._process_input_data(predict_data)
        self.manager.predict_data = processed_input
        self.manager.predicted_labels = self.__abstract_predict(processed_input, predict_mode)

        return self.manager.predicted_labels

    def predict_proba(self,
                      predict_data: tuple,
                      predict_mode: str = 'probs',
                      calibrate_probs: bool = False,
                      **kwargs):
        """
        Method to obtain prediction probabilities from trained Industrial model.

        Args:
            predict_mode: ``default='default'``. Defines the mode of prediction. Could be 'default' or 'probs'.
            predict_data: tuple with test_features and test_target
            calibrate_probs: ``default=False``. If True, calibrate probabilities

        Returns:
            the array with prediction probabilities

        """
        self.repo = IndustrialModels().setup_repository(backend=self.manager.compute_config.backend)
        predict_proba_plan = build_industrial_predict_proba_plan(
            predict_mode=predict_mode,
            is_regression_task_context=self.manager.industrial_config.is_regression_task_context,
        )
        processed_input = self._process_input_data(predict_data)
        self.manager.predicted_probs = self.__abstract_predict(processed_input, predict_proba_plan.normalized_mode)

        return self.manager.predicted_probs

    def finetune(self,
                 train_data: Union[InputData, dict, tuple],
                 tuning_params: Optional[dict] = None,
                 model_to_tune: Optional[Pipeline] = None,
                 return_only_fitted: bool = False):
        """Method to obtain prediction probabilities from trained Industrial model.

            Args:
                model_to_tune: model to fine-tune
                train_data: raw train data
                tuning_params: dictionary with tuning parameters
                return_only_fitted: ``default=False``. Defines what to return.

            """

        def _fit_pipeline(data_dict):
            data_dict['model_to_tune'].fit(data_dict['train_data'])
            return data_dict

        is_fedot_datatype = self.manager.condition_check.input_data_is_fedot_type(train_data)
        finetune_plan = build_industrial_finetune_plan(
            is_fedot_datatype=is_fedot_datatype,
            task_name=self.manager.automl_config.config['task'],
            tuning_params=tuning_params,
        )

        with exception_handler(Exception, on_exception=self.shutdown, suppress=False):
            processed_train_data = train_data
            if finetune_plan.should_process_input:
                processed_train_data = self._process_input_data(processed_train_data)
            processed_train_data = self.__init_industrial_backend(processed_train_data)
            tuning_context = {
                'train_data': processed_train_data,
                'model_to_tune': model_to_tune.build(),
                'tuning_params': finetune_plan.normalized_tuning_params,
            }
            tuned_model = _fit_pipeline(tuning_context)['model_to_tune'] if return_only_fitted else build_tuner(
                self, **tuning_context)

        self.manager.is_finetuned = True
        self.manager.solver = tuned_model

    def get_metrics(self,
                    labels: np.ndarray,
                    probs: np.ndarray,
                    target: Union[list, np.array] = None,
                    metric_names: tuple = None,
                    rounding_order: int = 3,
                    train_data: Union[list, np.array] = None,
                    seasonality: int = 1) -> pd.DataFrame:
        """
        Method to calculate metrics for Industrial model.

        Available metrics for classification task: 'f1', 'accuracy', 'precision', 'roc_auc', 'logloss'.

        Available metrics for regression task: 'r2', 'rmse', 'mse', 'mae', 'median_absolute_error',
        'explained_variance_score', 'max_error', 'd2_absolute_error_score', 'msle', 'mape'.

        Args:
            target: target values
            metric_names: list of metric names
            rounding_order: rounding order for metrics

        Returns:
            pandas DataFrame with calculated metrics

        """
        problem = self.manager.automl_config.task
        metrics_request_plan = build_industrial_metrics_request_plan(
            problem=problem,
            probs=probs,
            metric_names=metric_names,
        )
        if metrics_request_plan.warn_missing_probabilities:
            self.logger.info('Predicted probabilities are not available. Use `predict_proba()` method first')

        self.metric_dict = self._metric_evaluation_loop(
            target=target,
            problem=problem,
            predicted_labels=labels,
            predicted_probs=probs,
            rounding_order=rounding_order,
            metric_names=metric_names,
            train_data=train_data,
            seasonality=seasonality)
        return self.metric_dict

    def save(self, mode: str = 'all', **kwargs):
        save_plan = build_industrial_save_plan(
            mode=mode,
            is_fedot_solver=self.manager.condition_check.solver_is_fedot_class(self.manager.solver),
        )

        def save_model(api_manager):
            return Either(value=api_manager.solver,
                          monoid=[api_manager.solver,
                                  api_manager.condition_check.solver_is_fedot_class(
                                      api_manager.solver)]). \
                either(left_function=lambda pipeline: pipeline.save(path=api_manager.compute_config.output_folder,
                                                                    create_subdir=True, is_datetime_in_path=True),
                       right_function=lambda solver: solver.current_pipeline.save(
                           path=api_manager.compute_config.output_folder,
                           create_subdir=True,
                           is_datetime_in_path=True))

        def save_opt_hist(api_manager):
            return self.manager.solver.history.save(
                f"{self.manager.compute_config.output_folder}/optimization_history.json")

        def save_metrics(api_manager):
            return self.metric_dict.to_csv(
                f'{self.manager.compute_config.output_folder}/metrics.csv')

        def save_preds(api_manager):
            return pd.DataFrame(api_manager.predicted_labels).to_csv(
                f'{self.manager.compute_config.output_folder}/labels.csv')

        method_dict = {'metrics': save_metrics, 'model': save_model, 'opt_hist': save_opt_hist,
                       'prediction': save_preds}
        self.manager.create_folder(self.manager.compute_config.output_folder)
        if not save_plan.include_opt_hist:
            del method_dict['opt_hist']

        def save_all(api_manager):
            for method in method_dict.values():
                try:
                    method(api_manager)
                except Exception as ex:
                    self.manager.logger.info(f'Error during saving. Exception - {ex}')

        Either(value=self.manager, monoid=[self.manager, save_plan.save_all]). \
            either(left_function=lambda api_manager: method_dict[save_plan.selected_mode](self.manager),
                   right_function=lambda api_manager: save_all(api_manager))

    def load(self, path):
        """Loads saved Industrial model from disk

        Args:
            path (str): path to the model

        """
        self.repo = IndustrialModels().setup_repository()
        dir_list = os.listdir(path)
        load_plan = build_industrial_load_plan(path=path, dir_list=dir_list)
        if load_plan.load_multiple_pipelines:
            return [Pipeline().load(f'{load_plan.resolved_path}/{p}/0_pipeline_saved') for p in dir_list]
        return Pipeline().load(load_plan.resolved_path)

    def explain(self, explaing_config: dict = {}):
        """Explain model's prediction via time series points perturbation

            Args:
                explaing_config: Additional arguments for explanation. These arguments control the
                         number of samples, window size, metric, threshold, and dataset name.
                         See the function implementation for detailed information on
                         supported arguments.
        """
        explain_plan = build_industrial_explain_plan(explaing_config)

        explainer = self.manager.industrial_config.explain_methods[explain_plan.method](
            model=self,
            features=self.manager.predict_data.features.squeeze(),
            target=self.manager.predict_data.target
        )

        explainer.explain(n_samples=explain_plan.samples, window=explain_plan.window, method=explain_plan.metric)
        explainer.visual(metric=explain_plan.metric, threshold=explain_plan.threshold, name=explain_plan.name)

    def return_report(self) -> pd.DataFrame:
        return self.manager.solver.return_report() if isinstance(self.manager.solver, Fedot) else None

    def vis_optimisation_history(self, opt_history_path: str = None,
                                 mode: str = 'all',
                                 return_history: bool = False):
        """ The function runs visualization of the composing history and the best pipeline. """
        history = OptHistory.load(opt_history_path + 'optimization_history.json') \
            if isinstance(opt_history_path, str) else opt_history_path
        history_visualizer = PipelineHistoryVisualizer(history)
        vis_func = {
            'fitness': (
                history_visualizer.fitness_box, dict(
                    save_path='fitness_by_generation.png', best_fraction=1)),
            'models': (
                history_visualizer.operations_animated_bar, dict(
                    save_path='operations_animated_bar.gif', show_fitness=True)),
            'diversity': (
                history_visualizer.diversity_population, dict(
                    save_path='diversity_population.gif', fps=1))}
        history_plan = build_industrial_history_visualization_plan(mode)

        def plot_func(selected_mode):
            return vis_func[selected_mode][0](**vis_func[selected_mode][1])

        Either(value=vis_func,
               monoid=[history_plan.selected_mode, history_plan.visualize_all]).either(
            left_function=lambda _vis_func: plot_func(history_plan.selected_mode),
            right_function=lambda _vis_func: [func(**params) for func, params in vis_func.values()]
        )
        return history_visualizer.history if return_history else None

    def shutdown(self):
        """Shutdown Dask client"""
        if self.manager.dask_client is not None:
            self.manager.dask_client.close()
            del self.manager.dask_client
        if self.manager.dask_cluster is not None:
            self.manager.dask_cluster.close()
            del self.manager.dask_cluster
