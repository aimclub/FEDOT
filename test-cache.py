from test.data.datasets import get_dataset
from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

# metric = ["rmse"]
# task_type = "regression"
# X, y = make_regression(n_samples=100, n_features=1, noise=1)

metric = ["f1"]
task_type = "classification"
# X, y = make_classification(n_samples=100, n_features=1)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.33, random_state=42)

train_data, test_data, _ = get_dataset(task_type)

"""Regression"""
# bug_pipeline = (
#     PipelineBuilder()
#     .add_node('resample',
#               params={'balance': 'expand_minority',
#                       'replace': False, 'balance_ratio': 1})
#     .add_node('pca',
#               params={'svd_solver': 'full',
#                       'n_components': 0.47911835000292824})
#     .add_node('scaling')
#     .add_node('ridge',
#               params={'alpha': 7.359303296600219})
#     .build()
# )

"""Classification"""
# bug_pipeline = (
#     PipelineBuilder()
#     .add_node("isolation_forest_class")
#     .add_node("bernb")
#     .add_node("knn",
#               params={'n_neighbors': 36, 'weights': 'uniform', 'p': 2})
#     .build()
# )

# bug_pipeline = (
#     PipelineBuilder()
#     .add_node("scaling")
#     .add_node("dt")
#     .add_node("rf", params={"n_jobs": 1})
#     .build()
# )

# bug_pipeline = (
#     PipelineBuilder()
#     .add_node("scaling")
#     .add_node("dt")
#     .build()
# )

bug_pipeline = (
    PipelineBuilder()
    # .add_node("resample", params={'balance': 'expand_minority', 'replace': False, 'balance_ratio': 1})
    .add_node("mlp")
    .add_node("qda")
    .add_node("qda")
    # .add_node("mlp")
    .build()
)

auto_model = Fedot(
    problem=task_type,
    metric=metric,
    preset="best_quality",
    with_tuning=False,
    timeout=0.1,
    cv_folds=5,
    seed=42,
    n_jobs=1,
    # logging_level=10,
    use_pipelines_cache=False,
    use_auto_preprocessing=False,
    # history_dir="./saved_history"
    initial_assumption=bug_pipeline
)

auto_model.fit(features=train_data)
# auto_model.fit(features=train_data, predefined_model=bug_pipeline)

prediction = auto_model.predict(features=test_data, save_predictions=False)

auto_model.current_pipeline.show()
print(auto_model.current_pipeline.descriptive_id)

# auto_model.current_pipeline.save(path="./saved_pipelines", create_subdir=True, is_datetime_in_path=True)
# auto_model.history.save("saved_history.json")
print(auto_model.get_metrics())
print(auto_model.return_report().head(10))
