from typing import List, Union

from fedot.core.data.data import data_has_categorical_features, InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode, Node
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum

NOT_FITTED_ERR_MSG = 'Model not fitted yet'


class ApiInitialAssumptionsHelper:
    def get_initial_assumption(self,
                               data: Union[InputData, MultiModalData],
                               task: Task) -> Pipeline:

        has_categorical_features = data_has_categorical_features(data)

        if isinstance(data, MultiModalData):
            node_final = self.create_multidata_pipeline(task, data, has_categorical_features)
        elif isinstance(data, InputData):
            node_final = self.create_unidata_pipeline(task, has_categorical_features)
        else:
            raise NotImplementedError(f"Don't handle {type(data)}")

        init_pipeline = Pipeline(node_final)

        return init_pipeline

    def create_unidata_pipeline(self,
                                task: Task,
                                has_categorical_features: bool) -> Node:
        node_imputation = PrimaryNode('simple_imputation')
        if task.task_type == TaskTypesEnum.ts_forecasting:
            node_lagged = SecondaryNode('lagged', [node_imputation])
            node_final = SecondaryNode('ridge', [node_lagged])
        else:
            if has_categorical_features:
                node_encoder = SecondaryNode('one_hot_encoding', [node_imputation])
                node_preprocessing = SecondaryNode('scaling', [node_encoder])
            else:
                node_preprocessing = SecondaryNode('scaling', [node_imputation])

            if task.task_type == TaskTypesEnum.classification:
                node_final = SecondaryNode('xgboost', nodes_from=[node_preprocessing])
            elif task.task_type == TaskTypesEnum.regression:
                node_final = SecondaryNode('xgbreg', nodes_from=[node_preprocessing])
            else:
                raise NotImplementedError(f"Don't have initial pipeline for task type: {task.task_type}")

        return node_final

    def create_multidata_pipeline(self, task: Task, data: MultiModalData, has_categorical_features: bool) -> Node:
        if task.task_type == TaskTypesEnum.ts_forecasting:
            node_final = SecondaryNode('ridge', nodes_from=[])
            for data_source_name, values in data.items():
                if data_source_name.startswith('data_source_ts'):
                    node_primary = PrimaryNode(data_source_name)
                    node_imputation = SecondaryNode('simple_imputation', [node_primary])
                    node_lagged = SecondaryNode('lagged', [node_imputation])
                    node_last = SecondaryNode('ridge', [node_lagged])
                    node_final.nodes_from.append(node_last)
        elif task.task_type == TaskTypesEnum.classification:
            node_final = SecondaryNode('xgboost', nodes_from=[])
            node_final.nodes_from = self.create_first_multimodal_nodes(data, has_categorical_features)
        elif task.task_type == TaskTypesEnum.regression:
            node_final = SecondaryNode('xgbreg', nodes_from=[])
            node_final.nodes_from = self.create_first_multimodal_nodes(data, has_categorical_features)
        else:
            raise NotImplementedError(f"Don't have initial pipeline for task type: {task.task_type}")

        return node_final

    def create_first_multimodal_nodes(self, data: MultiModalData, has_categorical: bool) -> List[SecondaryNode]:
        nodes_from = []

        for data_source_name, values in data.items():
            node_primary = PrimaryNode(data_source_name)
            node_imputation = SecondaryNode('simple_imputation', [node_primary])
            if data_source_name.startswith('data_source_table') and has_categorical:
                node_encoder = SecondaryNode('one_hot_encoding', [node_imputation])
                node_preprocessing = SecondaryNode('scaling', [node_encoder])
            else:
                node_preprocessing = SecondaryNode('scaling', [node_imputation])
            node_last = SecondaryNode('ridge', [node_preprocessing])
            nodes_from.append(node_last)

        return nodes_from
