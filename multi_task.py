from cases.credit_scoring.credit_scoring_problem import get_scoring_data
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum

train_file_path, _ = get_scoring_data()

train_regr = InputData.from_csv(train_file_path, task=Task(TaskTypesEnum.regression))
train_class = InputData.from_csv(train_file_path, task=Task(TaskTypesEnum.classification))

fit_data = MultiModalData({
    'data_source_table/regr': train_regr,
    'data_source_table/class': train_regr
})

ds_regr = PrimaryNode('data_source_table/regr')
ds_class = PrimaryNode('data_source_table/class')

regr_node = SecondaryNode('ridge', nodes_from=[ds_regr])
class_node = SecondaryNode('logit', nodes_from=[ds_class])

root = SecondaryNode('linear', nodes_from=[regr_node])

pipeline = Pipeline(root)

pipeline.fit(fit_data)
