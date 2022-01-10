from cases.credit_scoring.credit_scoring_problem import get_scoring_data, get_scoring_data_regr
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskParams, TaskTypesEnum

train_file_path, test_file_path = get_scoring_data()
train_file_path_regr, test_file_path_regr = get_scoring_data_regr()

train_regr = InputData.from_csv(train_file_path,
                                task=Task(TaskTypesEnum.regression))
train_class = InputData.from_csv(train_file_path,
                                 task=Task(TaskTypesEnum.classification,
                                           task_params=TaskParams(is_main_task=False)))

test_regr = InputData.from_csv(train_file_path,
                               task=Task(TaskTypesEnum.regression))
test_class = InputData.from_csv(train_file_path,
                                task=Task(TaskTypesEnum.classification,
                                          task_params=TaskParams(is_main_task=False)))

fit_data = MultiModalData({
    'data_source_table/regr': train_regr,
    'data_source_table/class': train_class
})

predict_data = MultiModalData({
    'data_source_table/regr': test_regr,
    'data_source_table/class': test_class
})

ds_regr = PrimaryNode('data_source_table/regr')
ds_class = PrimaryNode('data_source_table/class')

imp_regr = SecondaryNode('simple_imputation', nodes_from=[ds_regr])
imp_class = SecondaryNode('simple_imputation', nodes_from=[ds_class])

regr_node = SecondaryNode('ridge', nodes_from=[imp_regr])
class_node = SecondaryNode('logit', nodes_from=[imp_class])

root = SecondaryNode('linear', nodes_from=[regr_node, class_node])

pipeline = Pipeline(root)

pipeline.fit(fit_data)
pipeline.predict(predict_data)
