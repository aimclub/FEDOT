from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum

NOT_FITTED_ERR_MSG = 'Model not fitted yet'


class API_initial_assumptions_helper:
    def assumption_by_data(self,
                           data,
                           node_from_task) -> Pipeline:
        if isinstance(data, MultiModalData):
            node_final = SecondaryNode('ridge', nodes_from=[])
            for data_source_name in data.keys():
                last_node_for_sub_chain = \
                    SecondaryNode('ridge', [SecondaryNode('lagged', [PrimaryNode(data_source_name)])])
                node_final.nodes_from.append(last_node_for_sub_chain)
        else:
            node_final = node_from_task

        return node_final

    def assumption_by_task(self,
                           task: Task) -> SecondaryNode:
        node_lagged = PrimaryNode('scaling')
        initial_assumption_dict = {TaskTypesEnum.classification: SecondaryNode('xgboost', nodes_from=[node_lagged]),
                                   TaskTypesEnum.regression: SecondaryNode('ridge', nodes_from=[node_lagged]),
                                   TaskTypesEnum.ts_forecasting: SecondaryNode('ridge',
                                                                               nodes_from=[PrimaryNode('lagged')])}
        # init_pipeline = Pipeline(initial_assumption_dict[task.task_type])
        return initial_assumption_dict[task.task_type]
