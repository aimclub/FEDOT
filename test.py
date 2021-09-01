from test.unit.tasks.test_classification import pipeline_simple, get_iris_data
from test.unit.validation.test_table_cv import get_data
from sklearn.metrics import accuracy_score
from fedot.core.repository.tasks import Task, TaskTypesEnum

folds = 2
task = Task(task_type=TaskTypesEnum.classification)
dataset = get_iris_data()

simple_pipeline = pipeline_simple()
tuned = simple_pipeline.fine_tune_all_nodes(loss_function=accuracy_score,
                                                loss_params=None,
                                                input_data=dataset,
                                                iterations=1, timeout=1,
                                                cv_folds=folds)