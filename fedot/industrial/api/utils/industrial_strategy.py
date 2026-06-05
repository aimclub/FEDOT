import logging
import math
from copy import deepcopy

import numpy as np
from fedot import Fedot
from fedot.core.data.input_data.data import InputData
from fedot.core.data.split.data_split import train_test_data_setup
from fedot.core.data.multimodal.multi_modal import MultiModalData
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum
from pymonad.maybe import Maybe

from fedot.industrial.core.architecture.abstraction.client import use_default_fedot_client
from fedot.industrial.core.ensemble.kernel_ensemble import KernelEnsembler
from fedot.industrial.core.ensemble.random_automl_forest import RAFEnsembler
from fedot.industrial.core.operation.decomposition.matrix_decomposition.method_impl.column_sampling_decomposition import \
    CURDecomposition
from fedot.industrial.core.repository.constanst_repository import BATCH_SIZE_FOR_FEDOT_WORKER, FEDOT_WORKER_NUM, \
    FEDOT_WORKER_TIMEOUT_PARTITION, FEDOT_TS_FORECASTING_ASSUMPTIONS, FEDOT_TASK
from fedot.industrial.core.repository.industrial_implementations.abstract import build_tuner
from fedot.industrial.api.utils.industrial_strategy_rules import (
    build_federated_runtime_plan,
    build_industrial_kernel_finetune_plan,
    build_sampling_iteration_plans,
    build_sampling_predict_plan,
    resolve_industrial_strategy_dispatch,
)


class IndustrialStrategy:
    """
    Class for industrial strategy implementation

    Args:
        industrial_strategy_params: dict
            Parameters for industrial strategy
        industrial_strategy: str
            Industrial strategy name
        api_config: dict
            Configuration for API
    """

    def __init__(self, industrial_strategy_params, industrial_strategy, api_config):
        self.industrial_strategy_params = industrial_strategy_params or {}
        self.finetune = self.industrial_strategy_params.get('finetune', False)
        self.finetune_params = self.industrial_strategy_params.get(
            'tuning_params', {})
        self.industrial_strategy = industrial_strategy

        self.sampling_algorithm = {'CUR': self.__cur_sampling,
                                   'Random': self.__random_sampling}
        self.ensemble_strategy_dict = {'MeanEnsemble': np.mean,
                                       'MedianEnsemble': np.median,
                                       'MinEnsemble': np.min,
                                       'MaxEnsemble': np.max,
                                       'ProductEnsemble': np.prod}

        self.ensemble_strategy = list(self.ensemble_strategy_dict.keys())
        self.random_label = None
        self.config = api_config
        self.logger = logging.getLogger('IndustrialStrategy')
        self.kernel_ensembler = KernelEnsembler
        self.RAF_workers = None
        self.solver = None

    def __cur_sampling(self, tensor, target, sampling_rate=0.7):
        projection_rank = math.ceil(max(tensor.shape) * sampling_rate)
        decomposer = CURDecomposition(rank=projection_rank)
        sampled_tensor, sampled_target = decomposer.fit_transform(
            tensor, target)
        return decomposer, sampled_tensor, sampled_target

    def __random_sampling(self, tensor, target, sampling_rate=0.7):
        projection_rank = math.ceil(max(tensor.shape) * sampling_rate)
        tensor = tensor.squeeze()
        selected_rows = np.random.choice(
            tensor.shape[0], size=projection_rank, replace=False)
        sampled_tensor, sampled_target = tensor[selected_rows,
                                                :], target[selected_rows, :]
        return selected_rows, sampled_tensor, sampled_target

    def fit(self, input_data):
        dispatch_plan = resolve_industrial_strategy_dispatch(
            self.industrial_strategy)
        getattr(self, dispatch_plan.fit_method_name)(input_data)
        return self.solver

    def predict(self, input_data, predict_mode):
        dispatch_plan = resolve_industrial_strategy_dispatch(
            self.industrial_strategy)
        return getattr(self, dispatch_plan.predict_method_name)(input_data, predict_mode)

    def _federated_strategy(self, input_data):

        n_samples = input_data.features.shape[0]
        runtime_plan = build_federated_runtime_plan(
            n_samples=n_samples,
            batch_size_threshold=BATCH_SIZE_FOR_FEDOT_WORKER,
            requested_workers=self.RAF_workers,
            timeout=self.config['timeout'],
            timeout_partition=FEDOT_WORKER_TIMEOUT_PARTITION,
            default_workers=FEDOT_WORKER_NUM,
        )
        if runtime_plan.use_raf:
            self.logger.info('RAF algorithm was applied')
            self.RAF_workers = runtime_plan.raf_workers
            self.config['timeout'] = runtime_plan.timeout

            self.logger.info(
                f'Batch_size - {runtime_plan.batch_size}. Number of batches - {self.RAF_workers}'
            )

            self.solver = RAFEnsembler(composing_params=self.config,
                                       n_splits=self.RAF_workers,
                                       batch_size=runtime_plan.batch_size)
            self.logger.info(
                f'Number of AutoMl models in ensemble - {self.solver.n_splits}')

            self.solver.fit(input_data)

        else:
            self.logger.info(f'RAF algorithm is not applicable: n_samples={n_samples} < {BATCH_SIZE_FOR_FEDOT_WORKER}. '
                             f'FEDOT algorithm was applied')
            self.solver = Fedot(**self.config)
            self.solver.fit(input_data)

    def _forecasting_strategy(self, input_data):
        self.logger.info('TS forecasting algorithm was applied')
        self.solver = {}
        kernel_data = {
            model_name: input_data for model_name in FEDOT_TS_FORECASTING_ASSUMPTIONS.keys()}
        kernel_model = {model_name: model_impl.build() if 'build' in dir(model_impl) else model_impl(
            {}).fit(input_data) for model_name, model_impl in FEDOT_TS_FORECASTING_ASSUMPTIONS.items()}
        self.solver = self._finetune_loop(
            kernel_model, kernel_data, self.finetune_params)
        # for model_name, init_assumption in FEDOT_TS_FORECASTING_ASSUMPTIONS.items():
        #     self.config['initial_assumption'] = init_assumption.build()
        #     industrial = Fedot(**self.config)
        #     Maybe(
        #         value=industrial.fit(input_data),
        #         monoid=True).maybe(
        #         default_value=self.logger.info(f'Failed during fit stage - {model_name}'),
        #         extraction_function=lambda fitted_model: self.solver.update({model_name: industrial}))

    def _sampling_strategy(self, input_data):
        self.logger.info('Sampling strategy algorithm was applied')
        self.solver = {}
        self.sampler = {}
        algorithm = self.industrial_strategy_params['sampling_algorithm']
        sampling_plans = build_sampling_iteration_plans(
            sampling_algorithm=algorithm,
            sampling_range=self.industrial_strategy_params['sampling_range'],
        )
        base_features = deepcopy(input_data.features)
        base_target = deepcopy(input_data.target)
        for sampling_plan in sampling_plans:
            decomposer, sampled_features, sampled_target = self.sampling_algorithm[algorithm](
                tensor=base_features,
                target=base_target,
                sampling_rate=sampling_plan.sampling_rate,
            )
            sampled_input = deepcopy(input_data)
            sampled_input.features = sampled_features
            sampled_input.target = sampled_target
            sampled_input.idx = np.arange(len(sampled_input.features))
            industrial = Fedot(**self.config)
            Maybe(
                value=industrial.fit(sampled_input),
                monoid=True).maybe(
                default_value=self.logger.info(
                    f'Failed during fit stage - {algorithm}'),
                extraction_function=lambda fitted_model: self.solver.update(
                    {sampling_plan.result_key: industrial}))
            self.sampler.update({sampling_plan.result_key: decomposer})

    def _forecasting_exogenous_strategy(self, input_data):
        self.logger.info('TS exogenous forecasting algorithm was applied')
        self.solver = {}
        init_assumption = PipelineBuilder().add_node('lagged', 0)
        task = FEDOT_TASK[self.config['problem']]
        train_lagged, predict_lagged = train_test_data_setup(InputData(idx=np.arange(len(input_data.features)),
                                                                       features=input_data.features,
                                                                       target=input_data.features,
                                                                       task=task,
                                                                       data_type=DataTypesEnum.ts), 2)
        dataset_dict = {'lagged': train_lagged}
        exog_variable = self.industrial_strategy_params['exog_variable']
        init_assumption.add_node('exog_ts', 1)

        # Exogenous time series
        train_exog, predict_exog = train_test_data_setup(InputData(idx=np.arange(len(exog_variable)),
                                                                   features=exog_variable,
                                                                   target=input_data.features,
                                                                   task=task,
                                                                   data_type=DataTypesEnum.ts), 2)
        dataset_dict.update({'exog_ts': train_exog})

        train_dataset = MultiModalData(dataset_dict)
        init_assumption = init_assumption.join_branches('ridge')
        self.config['initial_assumption'] = init_assumption.build()

        industrial = Fedot(**self.config)
        industrial.fit(train_dataset)
        self.solver = {'exog_model': industrial}

    def _finetune_loop(self,
                       kernel_ensemble: dict,
                       kernel_data: dict,
                       tuning_params: dict = {}):
        tuned_models = {}
        finetune_plan = build_industrial_kernel_finetune_plan(
            self.config['problem'], tuning_params)
        for generator, kernel_model in kernel_ensemble.items():
            model_to_tune = deepcopy(kernel_model)
            pipeline_tuner, solver = build_tuner(
                self, model_to_tune, finetune_plan.normalized_tuning_params, kernel_data[generator], 'head')
            tuned_models.update({generator: solver})
        return tuned_models

    def _kernel_strategy(self, input_data):
        self.kernel_ensembler = KernelEnsembler(
            self.industrial_strategy_params)
        kernel_ensemble, kernel_data = self.kernel_ensembler.transform(
            input_data).predict
        self.solver = self._finetune_loop(kernel_ensemble, kernel_data)

    def _lora_strategy(self, input_data):
        self.lora_model = PipelineBuilder().add_node(
            'lora_model', params=self.industrial_strategy_params).build()
        self.lora_model.fit(input_data)

    def _federated_predict(self,
                           input_data,
                           mode: str = 'labels'):
        valid_nodes = self.solver.current_pipeline.root_node.nodes_from
        self.predicted_branch_probs = [
            x.predict(input_data).predict for x in valid_nodes]

        # reshape if binary
        if len(self.predicted_branch_probs[0].shape) < 2:
            self.predicted_branch_probs = [
                np.array([x, 1 - x]).T for x in self.predicted_branch_probs]

        self.predicted_branch_labels = [
            np.argmax(x, axis=1) for x in self.predicted_branch_probs]

        n_samples = self.predicted_branch_probs[0].shape[0]
        n_channels = len(self.predicted_branch_probs)

        head_model = deepcopy(self.solver.current_pipeline.root_node)
        head_model.nodes_from = []
        input_data.features = np.hstack(self.predicted_branch_labels).reshape(n_samples,
                                                                              n_channels,
                                                                              1)
        if mode == 'labels':
            return head_model.predict(input_data, 'labels').predict
        else:
            return head_model.predict(input_data).predict

    def _forecasting_predict(self,
                             input_data,
                             mode: str = True):
        @use_default_fedot_client
        def _predict_function(forecasting_model):
            if isinstance(forecasting_model, dict):
                predict_by_component = {
                    model_name: model['composite_pipeline'].predict(
                        model['train_fold_data'], mode).predict
                    for model_name, model in forecasting_model.items()}
                return np.sum(list(predict_by_component.values()), axis=0)
            else:
                return forecasting_model.predict(input_data, mode).predict

        labels_dict = {
            forecasting_strategy: _predict_function(forecasting_model) for forecasting_strategy,
            forecasting_model in self.solver.items()}

        return labels_dict

    def _lora_predict(self,
                      input_data,
                      mode: str = True):
        labels_dict = {
            k: v.predict(
                features=input_data,
                in_sample=mode) for k,
            v in self.solver.items()}
        return labels_dict

    def _kernel_predict(self,
                        input_data,
                        mode: str = 'labels'):
        labels_dict = {
            k: v.predict(
                input_data,
                mode).predict for k,
            v in self.solver.items()}
        return labels_dict

    def _sampling_predict(self,
                          input_data,
                          mode: str = 'labels'):
        labels_dict = {}
        predict_plan = build_sampling_predict_plan(
            mode=mode,
            sampling_algorithm=self.industrial_strategy_params['sampling_algorithm'],
        )
        for sampling_rate, solver in self.solver.items():
            copy_input = deepcopy(input_data)
            feature_space = self.sampler[sampling_rate].column_indices if predict_plan.use_cur_feature_space else None
            squeezed_features = input_data.features.squeeze()
            copy_input.features = squeezed_features if feature_space is None else squeezed_features[
                :, feature_space]
            labels_dict.update({sampling_rate: solver.predict(copy_input, mode) if predict_plan.labels_output
                                else solver.predict_proba(copy_input)})
            del copy_input
        return labels_dict

    def _check_predictions(self, predictions):
        """Check if the predictions array has the correct size.

        Args:
            predictions: array of shape (n_samples, n_classifiers). The votes obtained by each classifier
            for each sample.

        Returns:
            predictions: array of shape (n_samples, n_classifiers). The votes obtained by each classifier
            for each sample.

        Raises:
            ValueError: if the array do not contain exactly 3 dimensions: [n_samples, n_classifiers, n_classes]

        """

        list_proba = [predictions[model_preds] for model_preds in predictions]
        transformed = []
        if self.random_label is None:
            self.random_label = {
                class_by_gen: np.random.choice(
                    self.kernel_ensembler.classes_misses_by_generator[class_by_gen])
                for class_by_gen in self.kernel_ensembler.classes_described_by_generator}
        for prob_by_gen, class_by_gen in zip(
                list_proba, self.kernel_ensembler.classes_described_by_generator):
            converted_probs = np.zeros(
                (prob_by_gen.shape[0], len(
                    self.kernel_ensembler.all_classes)))
            for true_class, map_class in self.kernel_ensembler.mapper_dict[class_by_gen].items(
            ):
                converted_probs[:, true_class] = prob_by_gen[:, map_class]
            random_label = self.random_label[class_by_gen]
            converted_probs[:, random_label] = prob_by_gen[:, -1]
            transformed.append(converted_probs)

        return np.array(transformed).transpose((1, 0, 2))

    def ensemble_predictions(self, prediction_dict, strategy):
        transformed_predictions = self._check_predictions(prediction_dict)
        average_proba_predictions = self.ensemble_strategy_dict[strategy](
            transformed_predictions, axis=1)

        if average_proba_predictions.shape[1] == 1:
            average_proba_predictions = np.concatenate(
                [average_proba_predictions, 1 - average_proba_predictions], axis=1)

        return average_proba_predictions
