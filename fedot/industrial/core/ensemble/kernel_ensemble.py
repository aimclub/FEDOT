from copy import deepcopy
from typing import Any, Optional

import pandas as pd
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from MKLpy.callbacks import EarlyStopping
from MKLpy.scheduler import ReduceOnWorsening
from scipy.spatial.distance import pdist, squareform
from sklearn.svm import SVC

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.models.base_extractor import BaseExtractor
from fedot.industrial.core.repository.constanst_repository import (
    KERNEL_ALGO,
    KERNEL_BASELINE_FEATURE_GENERATORS,
    KERNEL_BASELINE_NODE_LIST,
    KERNEL_DISTANCE_METRIC,
    get_default_industrial_model_params,
)


class KernelEnsembler(BaseExtractor):
    """
    Class for kernel ensembling. This class implements a kernel-based ensemble method for feature
    extraction and classification. It supports both one-stage and two-stage kernel learning
    strategies and can handle multiclass classification problems.

    Args:
        params (Optional[OperationParameters]): Parameters of the operation

    Attributes:
        distance_metric (str): The distance metric used to calculate the Gram matrix
        kernel_strategy (str): The kernel learning strategy used by the model
        learning_strategy (str): The learning strategy used by the model
        head_model (str): The head model used by the model
        feature_extractor (List[str]): The feature extractors used by the model

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.distance_metric = params.get('distance_metric', KERNEL_DISTANCE_METRIC['default_metric'])
        self.kernel_strategy = params.get('kernel_strategy ', 'one_step_cka')
        self.learning_strategy = params.get('learning_strategy', 'all_classes')
        self.head_model = params.get('head_model', 'rf')
        self.feature_extractor = params.get('feature_extractor', list(
            KERNEL_BASELINE_FEATURE_GENERATORS.keys()))

        self._mapping_dict = {k: v for k, v in enumerate(self.feature_extractor)}
        self.lr = params.get('learning_rate', 0.1)
        self.patience = params.get('patience', 5)
        self.epoch = params.get('epoch', 500)
        self.optimisation_metric = params.get('optimisation_metric', 'roc_auc')

        self.algo_impl_dict = {'one_step': self.__one_stage_kernel,
                               'two_step': self.__two_stage_kernel}

        self.feature_matrix_train = []
        self.feature_matrix_test = []

    def __convert_weights(self, kernel_model):
        kernels_weights_by_class = []
        if not self.multiclass:
            kernels_weights_by_class.append(abs(
                kernel_model.solution.weights.cpu().detach().numpy()))
        else:
            for n_class in self.n_classes:
                kernels_weights_by_class.append(
                    abs(kernel_model.solution[n_class].weights.cpu().detach().numpy()))
        kernel_df = pd.DataFrame(kernels_weights_by_class)
        # kernel_df.columns = self.feature_extractor
        return kernel_df

    def __multiclass_check(self, target):
        self.n_classes = np.unique(target)
        if self.n_classes.shape[0] > 2:
            self.multiclass_strategy = 'ova'
            self.multiclass = True
        else:
            self.multiclass_strategy = 'ovr'
            self.multiclass = False

    def _select_top_feature_generators(self, kernel_weight_matrix):
        self.all_classes = kernel_weight_matrix.index.values.tolist()
        kernel_weight_matrix['best_generator_by_class'] = kernel_weight_matrix.apply(
            lambda row: self._mapping_dict[np.where(np.isclose(row.values,
                                                               max(row)))[0][0]], axis=1)
        top_n_generators = kernel_weight_matrix['best_generator_by_class'].value_counts(
        ).head(2).index.values.tolist()

        self.classes_described_by_generator = {gen: kernel_weight_matrix[kernel_weight_matrix['best_generator_by_class']
                                                                         == gen].index.values.tolist()
                                               for gen in top_n_generators}
        self.classes_misses_by_generator = {gen: [i for i in self.all_classes if
                                                  i not in self.classes_described_by_generator[gen]]
                                            for gen in top_n_generators}
        self.mapper_dict = {
            gen: {
                k: v for k, v in zip(
                    self.classes_described_by_generator[gen], np.arange(
                        0, len(
                            self.classes_described_by_generator[gen]) + 1))} for gen in top_n_generators}
        return top_n_generators, self.classes_described_by_generator

    def _map_target_for_generator(self, entry, mapper_dict):
        return mapper_dict[entry] if entry in mapper_dict else entry

    def _create_kernel_data(self, input_data, classes_described_by_generator, gen):
        train_fold = deepcopy(input_data)
        if self.learning_strategy != 'all_classes':
            described_idx, _ = np.where(
                train_fold.target == classes_described_by_generator[gen])
            not_described_idx = [i for i in np.arange(
                0, train_fold.target.shape[0]) if i not in described_idx]
            mp = np.vectorize(self._map_target_for_generator)
            train_fold.target = mp(
                entry=train_fold.target, mapper_dict=self.mapper_dict[gen])
            train_fold.target[not_described_idx] = max(
                list(self.mapper_dict[gen].values())) + 1
        return train_fold

    def _create_pipeline(self, gen):
        basis, generator = KERNEL_BASELINE_NODE_LIST[gen]
        model = PipelineBuilder().add_node(
            basis, params=get_default_industrial_model_params(basis)).add_node(
            generator, params=get_default_industrial_model_params(generator)).add_node(
            self.head_model).build() if basis is not None else PipelineBuilder().add_node(
            generator, params=get_default_industrial_model_params(generator)).add_node(
            self.head_model).build()
        return model

    def _create_kernel_ensemble(
            self,
            input_data,
            top_n_generators,
            classes_described_by_generator):
        kernel_ensemble = {}
        kernel_data = {}
        for i, gen in enumerate(top_n_generators):
            kernel_data.update({gen: self._create_kernel_data(input_data, classes_described_by_generator, gen)})
            kernel_ensemble.update({gen: self._create_pipeline(gen)})
        return kernel_ensemble, kernel_data

    def _transform(self, input_data: InputData) -> np.array:
        """
        Method for feature generation for all series
        """
        self.__multiclass_check(input_data.target)
        grammian_list = self.generate_grammian(input_data)

        if self.kernel_strategy.__contains__('one'):
            kernel_weight_matrix = self.__one_stage_kernel(grammian_list, input_data.target)

        else:
            kernel_weight_matrix = self.__two_stage_kernel(grammian_list, input_data.target)

        top_n_generators, classes_described_by_generator = self._select_top_feature_generators(kernel_weight_matrix)

        self.predict = self._create_kernel_ensemble(
            input_data,
            top_n_generators,
            classes_described_by_generator
        )

        return self.predict

    def generate_grammian(self, input_data) -> list[Any]:
        for model in self.feature_extractor:
            model = KERNEL_BASELINE_FEATURE_GENERATORS[model].build()
            self.feature_matrix_train.append(model.fit(input_data).predict)
        self.feature_matrix_train = [
            x.reshape(
                x.shape[0],
                x.shape[1] * x.shape[2]
            ) for x in self.feature_matrix_train]
        KLtr = [squareform(pdist(X=feature, metric=self.distance_metric))
                for feature in self.feature_matrix_train]
        return KLtr

    def __one_stage_kernel(self, grammian_list, target):
        mkl = KERNEL_ALGO[self.kernel_strategy](
            multiclass_strategy=self.multiclass_strategy).fit(grammian_list, target)
        kernel_weight_matrix = self.__convert_weights(mkl)
        return kernel_weight_matrix

    def __two_stage_kernel(self, grammian_list, target):
        earlystop = EarlyStopping(
            grammian_list,
            target,  # validation data, KL is a validation kernels list
            patience=self.patience,  # max number of acceptable negative steps
            cooldown=5,  # how ofter we run a measurement, 1 means every optimization step
            metric=self.optimisation_metric,  # the metric we monitor
        )

        mkl = KERNEL_ALGO[self.kernel_strategy](multiclass_strategy='ovr',
                                                max_iter=self.epoch,
                                                learner=SVC(C=1000),
                                                learning_rate=self.lr,
                                                callbacks=[earlystop],
                                                scheduler=ReduceOnWorsening()).fit(grammian_list, target)
        kernel_weight_matrix = self.__convert_weights(mkl)
        return kernel_weight_matrix
