from copy import deepcopy

from fedot.core.data.input_data.data import InputData
from fedot.core.data.multimodal.multi_modal import MultiModalData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.repository.constanst_repository import FEDOT_ATOMIZE_OPERATION, FEDOT_HEAD_ENSEMBLE, FEDOT_TASK
from fedot.industrial.core.repository.model_repository import SKLEARN_CLF_MODELS, SKLEARN_REG_MODELS


class RAFEnsembler:
    """Class for ensemble of random automl forest

    Args:
        composing_params: dict with parameters for ensemble
        n_splits: number of splits for ensemble
        batch_size: size of batch for ensemble

    """

    def __init__(self,
                 composing_params,
                 n_splits: int = None,
                 batch_size: int = 1000):

        self.current_pipeline = None
        self.problem = composing_params['problem']
        self.task = FEDOT_TASK[composing_params['problem']]
        self.atomized_automl = FEDOT_ATOMIZE_OPERATION[composing_params['problem']]
        self.head = FEDOT_HEAD_ENSEMBLE[composing_params['problem']]

        self.ensemble_method = self._raf_ensemble
        self.atomized_automl_params = composing_params
        self.n_splits = n_splits
        self.batch_size = batch_size

    def _decompose_pipeline(self):
        batch_pipe = [automl_branch.fitted_operation.model.current_pipeline.root_node for automl_branch in
                      self.current_pipeline.nodes if automl_branch.name in FEDOT_ATOMIZE_OPERATION.values()]
        self.ensemble_branches = batch_pipe
        self.ensemble_head = self.current_pipeline.nodes[0]
        self.ensemble_head.nodes_from = self.ensemble_branches
        self.current_pipeline = Pipeline(self.ensemble_head)

    def fit(self, train_data):
        if self.n_splits is None:
            self.n_splits = round(
                train_data.features.shape[0] / self.batch_size)

        new_features = np.array_split(train_data.features,
                                      self.n_splits)
        new_target = np.array_split(train_data.target,
                                    self.n_splits)

        self.current_pipeline = self.ensemble_method(new_features,
                                                     new_target,
                                                     n_splits=self.n_splits)
        self._decompose_pipeline()

    def predict(self, test_data, output_mode: str = 'labels'):
        return self.current_pipeline.predict(test_data, output_mode).predict

    def _raf_ensemble(self, features, target, n_splits):
        raf_ensemble = PipelineBuilder()
        data_dict = {}
        for i, data_fold_features, data_fold_target in zip(range(n_splits), features, target):

            train_fold = InputData(idx=np.arange(0, len(data_fold_features)),
                                   features=data_fold_features,
                                   target=data_fold_target,
                                   task=self.task,
                                   data_type=DataTypesEnum.image)

            raf_ensemble.add_node(operation_type=f'data_source_img/{i}',
                                  branch_idx=i)\
                .add_node(self.atomized_automl,
                          params=self.atomized_automl_params,
                          branch_idx=i)

            data_dict.update({f'data_source_img/{i}': train_fold})
        train_multimodal = MultiModalData(data_dict)
        head_automl_params = deepcopy(self.atomized_automl_params)

        head_automl_params['available_operations'] = [
            operation for operation in head_automl_params['available_operations'] if operation in list(
                SKLEARN_CLF_MODELS.keys()) or operation in list(
                SKLEARN_REG_MODELS.keys())]

        raf_ensemble = raf_ensemble.join_branches(self.head).build()
        raf_ensemble.fit(input_data=train_multimodal)
        return raf_ensemble
