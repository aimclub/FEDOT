from test.data.datasets import get_dataset
from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

# metric = ["rmse"]
# task_type = "regression"
# X, y = make_regression(n_samples=100, n_features=1, noise=1)

# metric = ["f1"]
# task_type = "classification"
# X, y = make_classification(n_samples=100, n_features=1)

metric = ["rmse"]
task_type = "ts_forecasting"

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.33, random_state=42)

train_data, test_data, _ = get_dataset(task_type,
                                       validation_blocks=1,
                                       n_samples=10000,
                                       n_features=100,
                                       forecast_length=1)

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

# bug_pipeline = (
#     PipelineBuilder()
#     .add_node("ransac_non_lin_reg", params={'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000}, branch_idx=0)
#     .add_node("pca", params={'svd_solver': 'full', 'n_components': 0.7})
#     .add_node("linear")
#     .add_node("sgdr")
#     .add_node("normalization")
#     .add_branch(("ransac_non_lin_reg", {'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000}), branch_idx=1)
#     .join_branches("linear")
#     .build()
# )

bug_pipeline_0 = (
    PipelineBuilder()
    .add_node(
        "ransac_non_lin_reg",
        params={'min_samples': 0.4, 'residual_threshold': 10, 'max_trials': 100, 'max_skips': 1000},
        branch_idx=0)
    .add_node("pca", params={'svd_solver': 'full', 'n_components': 0.7})
    .add_node(
        "ransac_non_lin_reg",
        params={'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000})
    .add_node("sgdr")
    .add_branch(
        ("ransac_non_lin_reg",
         {'min_samples': 0.4, 'residual_threshold': 10, 'max_trials': 100, 'max_skips': 1000}),
        branch_idx=1)
    .join_branches("linear")
    .build())

""""""

"""((((/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_pca_{'svd_solver': 'full', 'n_components': 0.7};)/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_sgdr;)/n_linear"""

"""
(
(
(
(
/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};
)
/n_pca_{'svd_solver': 'full', 'n_components': 0.7};
)
/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};
)
/n_sgdr;
)
/n_linear
"""

bug_pipeline_1 = (
    PipelineBuilder()
    .add_node("ransac_non_lin_reg", params={'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000})
    .add_node("pca", params={'svd_solver': 'full', 'n_components': 0.7})
    .add_node("ransac_non_lin_reg", params={'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000})
    .add_node("sgdr")
    .add_node("linear")
    .build()
)

""""""

"((((/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_normalization;)/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_sgdr;;/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_linear"


"""
(
(
(
(
/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};
)
/n_normalization;
)
/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};
)
/n_sgdr;
;
/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};
)
/n_linear
"""

bug_pipeline_2 = (
    PipelineBuilder()
    .add_node(
        "ransac_non_lin_reg",
        params={'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000},
        branch_idx=0)
    .add_node("normalization")
    .add_node(
        "ransac_non_lin_reg",
        params={'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000})
    .add_node("sgdr")
    .add_branch(
        ("ransac_non_lin_reg", {'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000}), branch_idx=1)
    .join_branches("linear")
    .build())

""""""

"((((/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_pca_{'svd_solver': 'full', 'n_components': 0.7};)/n_ransac_non_lin_reg_{'min_samples': 0.290191412541976, 'residual_threshold': 19.187633275042266, 'max_trials': 457.13281912005596, 'max_skips': 273915.78487223637};)/n_sgdr;;/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_linear"

"""
(
(
(
(
/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};
)
/n_pca_{'svd_solver': 'full', 'n_components': 0.7};
)
/n_ransac_non_lin_reg_{'min_samples': 0.290191412541976, 'residual_threshold': 19.187633275042266, 'max_trials': 457.13281912005596, 'max_skips': 273915.78487223637};
)
/n_sgdr;
;
/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};
)
/n_linear
"""

bug_pipeline_3 = (
    PipelineBuilder()
    .add_node(
        "ransac_non_lin_reg",
        params={'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000},
        branch_idx=0)
    .add_node("pca", params={'svd_solver': 'full', 'n_components': 0.7})
    .add_node(
        "ransac_non_lin_reg",
        params={'min_samples': 0.290191412541976, 'residual_threshold': 19.187633275042266, 'max_trials': 457.13281912005596, 'max_skips': 273915.78487223637})
    .add_node("sgdr")
    .add_branch(("ransac_non_lin_reg", {'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000}), branch_idx=1)
    .join_branches("linear")
    .build()
)

""""""

"""MetricsObjective - Objective evaluation error for graph {'depth': 3, 'length': 4, 'nodes': [linear, fast_ica, pca, ransac_non_lin_reg]} on metric rmse: Metric can not be evaluated because of: X has 11 features, but LinearRegression is expecting 13 features as input."""

"""((/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};)/n_pca_{'svd_solver': 'full', 'n_components': 0.7};;/n_fast_ica_{'whiten': 'unit-variance'};)/n_linear"""

"""
(
(
/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};
)
/n_pca_{'svd_solver': 'full', 'n_components': 0.7};
;
/n_fast_ica_{'whiten': 'unit-variance'};
)
/n_linear
"""

bug_pipeline_00 = (
    PipelineBuilder().add_node(
        "ransac_non_lin_reg",
        params={'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000},
        branch_idx=0).add_node("pca", params={'svd_solver': 'full', 'n_components': 0.7}).add_branch(
        ("fast_ica", {'whiten': 'unit-variance'}),
        branch_idx=1)
    .join_branches("linear").build())


""""""

"""MetricsObjective - Objective evaluation error for graph {'depth': 6, 'length': 6, 'nodes': [linear, scaling, isolation_forest_reg, ransac_non_lin_reg, sgdr, pca]} on metric rmse: Metric can not be evaluated because of: X has 3 features, but SGDRegressor is expecting 5 features as input."""

"""(((((/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};)/n_pca_{'svd_solver': 'full', 'n_components': 0.7};)/n_sgdr;;/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};)/n_isolation_forest_reg;;((/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};)/n_pca_{'svd_solver': 'full', 'n_components': 0.7};)/n_sgdr;)/n_scaling;;(((/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};)/n_pca_{'svd_solver': 'full', 'n_components': 0.7};)/n_sgdr;;/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};)/n_isolation_forest_reg;;((/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};)/n_pca_{'svd_solver': 'full', 'n_components': 0.7};)/n_sgdr;)/n_linear"""

"""
(
(
(
(
(
/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};
)
/n_pca_{'svd_solver': 'full', 'n_components': 0.7};
)
/n_sgdr;
;
/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};
)
/n_isolation_forest_reg;
;
(
(
/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};
)
/n_pca_{'svd_solver': 'full', 'n_components': 0.7};
)
/n_sgdr;
)
/n_scaling;
;
(
(
(
/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};
)
/n_pca_{'svd_solver': 'full', 'n_components': 0.7};
)
/n_sgdr;
;
/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};
)
/n_isolation_forest_reg;
;
(
(
/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};
)
/n_pca_{'svd_solver': 'full', 'n_components': 0.7};
)
/n_sgdr;`
)
/n_linear
"""

""""""

"""MetricsObjective - Objective evaluation error for graph {'depth': 6, 'length': 6, 'nodes': [linear, fast_ica, pca, ransac_non_lin_reg, ransac_lin_reg, resample]} on metric rmse: Metric can not be evaluated because of: X has 10 features, but FastICA is expecting 12 features as input."""

"""(((((/n_resample_{'balance': 'expand_minority', 'replace': False, 'balance_ratio': 1};)/n_ransac_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10, 'max_trials': 100, 'max_skips': 1000};)/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};)/n_pca_{'svd_solver': 'full', 'n_components': 0.5078567721808118};;/n_resample_{'balance': 'expand_minority', 'replace': False, 'balance_ratio': 1};)/n_fast_ica_{'whiten': 'unit-variance'};)/n_linear"""

"""(((((/n_resample_{'balance': 'expand_minority', 'replace': False, 'balance_ratio': 1};)/n_ransac_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10, 'max_trials': 100, 'max_skips': 1000};)/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};)/n_pca_{'svd_solver': 'full', 'n_components': 0.5078567721808118};;/n_resample_{'balance': 'expand_minority', 'replace': False, 'balance_ratio': 1};)/n_fast_ica_{'whiten': 'unit-variance'};)/n_linear"""


""""""

"""MetricsObjective - Objective evaluation error for graph {'depth': 6, 'length': 7, 'nodes': [linear, sgdr, ridge, scaling, pca, ransac_non_lin_reg, ransac_non_lin_reg]} on metric rmse: Metric can not be evaluated because of: index 3 is out of bounds for axis 1 with size 3"""

"""(((((/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};)/n_pca_{'svd_solver': 'full', 'n_components': 0.7};)/n_scaling;)/n_ridge_{'alpha': 0.49067007867022183};;/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};)/n_sgdr;;(/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};)/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10, 'max_trials': 100, 'max_skips': 1000};)/n_linear"""


""""""

"""MetricsObjective - Objective evaluation error for graph {'depth': 6, 'length': 6, 'nodes': [linear, fast_ica, pca, resample, catboostreg, ransac_non_lin_reg]} on metric rmse: Metric can not be evaluated because of: X has 19 features, but FastICA is expecting 17 features as input."""

"""(((((/n_resample_{'balance': 'expand_minority', 'replace': False, 'balance_ratio': 1};)/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};)/n_catboostreg_{'allow_writing_files': False, 'verbose': False, 'iterations': 1000, 'enable_categorical': True, 'use_eval_set': True, 'use_best_model': True, 'early_stopping_rounds': 30, 'n_jobs': 1};;(/n_resample_{'balance': 'expand_minority', 'replace': False, 'balance_ratio': 1};)/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};;/n_resample_{'balance': 'expand_minority', 'replace': False, 'balance_ratio': 1};)/n_pca_{'svd_solver': 'full', 'n_components': 0.7};;(/n_resample_{'balance': 'expand_minority', 'replace': False, 'balance_ratio': 1};)/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};;/n_resample_{'balance': 'expand_minority', 'replace': False, 'balance_ratio': 1};)/n_fast_ica_{'whiten': 'unit-variance'};)/n_linear"""

""""""

"""MetricsObjective - Objective evaluation error for graph {'depth': 6, 'length': 6, 'nodes': [linear, ransac_lin_reg, ransac_lin_reg, ransac_non_lin_reg, sgdr, pca]} on metric rmse: Metric can not be evaluated because of: X has 27 features, but SGDRegressor is expecting 26 features as input."""

"""(((((/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_ransac_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10, 'max_trials': 100, 'max_skips': 1000};)/n_ransac_lin_reg_{'min_samples': 0.5024646909061051, 'residual_threshold': 1.8119360430821913e+41, 'max_trials': 244.04573231429495, 'max_skips': 173075.51871026386};)/n_pca_{'svd_solver': 'full', 'n_components': 0.38958856945380116};;((/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_ransac_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10, 'max_trials': 100, 'max_skips': 1000};)/n_ransac_lin_reg_{'min_samples': 0.5024646909061051, 'residual_threshold': 1.8119360430821913e+41, 'max_trials': 244.04573231429495, 'max_skips': 173075.51871026386};;(/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_ransac_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10, 'max_trials': 100, 'max_skips': 1000};;/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_sgdr;;((/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_ransac_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10, 'max_trials': 100, 'max_skips': 1000};)/n_ransac_lin_reg_{'min_samples': 0.5024646909061051, 'residual_threshold': 1.8119360430821913e+41, 'max_trials': 244.04573231429495, 'max_skips': 173075.51871026386};;(/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_ransac_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10, 'max_trials': 100, 'max_skips': 1000};;/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_linear"""

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

# bug_pipeline = (
#     PipelineBuilder()
#     .add_node("qda")
#     .add_node("qda")
#     .build()
# )

auto_model = Fedot(
    problem=task_type,
    metric=metric,
    preset="best_quality",
    with_tuning=False,
    timeout=40,
    cv_folds=2,
    seed=42,
    n_jobs=1,
    # logging_level=10,
    use_operations_cache=False,
    use_auto_preprocessing=False,
    history_dir="./saved_history",
    # initial_assumption=bug_pipeline_00
)

auto_model.fit(features=train_data)
# auto_model.fit(features=train_data, predefined_model=bug_pipeline)

prediction = auto_model.predict(features=test_data, save_predictions=False)

auto_model.current_pipeline.show()
print()
print(auto_model.current_pipeline.descriptive_id)
# print(
#     auto_model.current_pipeline.descriptive_id ==
#     "(((((/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_pca_{'svd_solver': 'full', 'n_components': 0.7};)/n_linear;)/n_sgdr;)/n_normalization;;/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_linear")

# bug_pipeline_0
# print(
#     auto_model.current_pipeline.descriptive_id ==
#     "((((/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10, 'max_trials': 100, 'max_skips': 1000};)/n_pca_{'svd_solver': 'full', 'n_components': 0.7};)/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_sgdr;;/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10, 'max_trials': 100, 'max_skips': 1000};)/n_linear")

# bug_pipeline_1
# print(
#     auto_model.current_pipeline.descriptive_id ==
#     "((((/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_pca_{'svd_solver': 'full', 'n_components': 0.7};)/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_sgdr;)/n_linear")

# bug_pipeline_2
# print(
#     auto_model.current_pipeline.descriptive_id ==
#     "((((/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_normalization;)/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_sgdr;;/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_linear"
# )

# bug_pipeline_3
# print(
#     auto_model.current_pipeline.descriptive_id ==
#     "((((/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_pca_{'svd_solver': 'full', 'n_components': 0.7};)/n_ransac_non_lin_reg_{'min_samples': 0.290191412541976, 'residual_threshold': 19.187633275042266, 'max_trials': 457.13281912005596, 'max_skips': 273915.78487223637};)/n_sgdr;;/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 10240, 'max_trials': 100, 'max_skips': 1000};)/n_linear")

# bug_pipeline_00
# print(
#     auto_model.current_pipeline.descriptive_id ==
#     "((/n_ransac_non_lin_reg_{'min_samples': 0.4, 'residual_threshold': 40960, 'max_trials': 100, 'max_skips': 1000};)/n_pca_{'svd_solver': 'full', 'n_components': 0.7};;/n_fast_ica_{'whiten': 'unit-variance'};)/n_linear")


# auto_model.current_pipeline.save(path="./saved_pipelines", create_subdir=True, is_datetime_in_path=True)
# auto_model.history.save("saved_history.json")

# NOTE: doesn't work with ts_forecasting
# print(auto_model.get_metrics())
# print(auto_model.return_report().head(10))
