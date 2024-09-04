
Tuning of Hyperparameters
=========================
To tune pipeline hyperparameters you can use GOLEM. There are two ways:

1. Tuning of all models hyperparameters simultaneously. Implemented via ``SimultaneousTuner``, ``OptunaTuner`` and ``IOptTuner`` classes.

2. Tuning of models hyperparameters sequentially node by node optimizing metric value for the whole pipeline or tuning
   only one node hyperparametrs. Implemented via ``SequentialTuner`` class.

More information about these approaches can be found
`here <https://towardsdatascience.com/hyperparameters-tuning-for-machine-learning-model-ensembles-8051782b538b>`_.

If ``with_tuning`` flag is set to ``True`` when using :doc:`FEDOT API </api/api>`, simultaneous hyperparameters tuning
using ``SimultaneousTuner`` is applied for composed pipeline and ``metric`` value is used as a metric for tuning.

FEDOT uses tuners implementation from GOLEM, see `GOLEM documentation`_ for more information.

.. list-table:: Tuners comparison
   :widths: 10 30 30 30 30
   :header-rows: 1

   * -
     - ``SimultaneousTuner``
     - ``SequentialTuner``
     - ``IOptTuner``
     - ``OptunaTuner``
   * - Based on
     - Hyperopt
     - Hyperopt
     - iOpt
     - Optuna
   * - Type of tuning
     - Simultaneous
     - | Sequential or
       | for one node only
     - Simultaneous
     - Simultaneous
   * - | Optimized
       | parameters
     - | categorical
       | discrete
       | continuous
     - | categorical
       | discrete
       | continuous
     - | discrete
       | continuous
     - | categorical
       | discrete
       | continuous
   * - Algorithm type
     - stochastic
     - stochastic
     - deterministic
     - stochastic
   * - | Supported
       | constraints
     - | timeout
       | iterations
       | early_stopping_rounds
       | eval_time_constraint
     - | timeout
       | iterations
       | early_stopping_rounds
       | eval_time_constraint
     - | iterations
       | eval_time_constraint
     - | timeout
       | iterations
       | early_stopping_rounds
       | eval_time_constraint
   * - | Supports initial
       | point
     - Yes
     - No
     - No
     - Yes
   * - | Supports multi
       | objective tuning
     - No
     - No
     - No
     - Yes

Hyperopt based tuners usually take less time for one iteration, but ``IOptTuner`` is able to obtain much more stable results.


Simple example
~~~~~~~~~~~~~~

To initialize a tuner you can use ``TunerBuilder``.

.. code-block:: python

    from fedot.core.repository.tasks import TaskTypesEnum, Task
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder

    task = Task(TaskTypesEnum.classification)
    train_data = InputData.from_csv('train_file.csv')
    pipeline = PipelineBuilder().add_node('knn', branch_idx=0).add_branch('logit', branch_idx=1)\
        .grow_branches('logit', 'rf').join_branches('knn').build()

    pipeline_tuner = TunerBuilder(task).build(train_data)

    tuned_pipeline = pipeline_tuner.tune(pipeline)

``TunerBuilder`` methods
~~~~~~~~~~~~~~~~~~~~~~~~

* with_tuner_
* with_requirements_
* with_cv_folds_
* with_n_jobs_
* with_metric_
* with_iterations_
* with_early_stopping_rounds_
* with_timeout_
* with_eval_time_constraint_
* with_search_space_
* with_additional_params_

Tuner class
-----------

.. _with_tuner:

Use ``.with_tuner()`` to specify tuner class to use. ``PipelineTuner`` is used by default.

.. code-block:: python

    from golem.core.tuning.sequential import SequentialTuner

    tuner = SequentialTuner

    pipeline_tuner = TunerBuilder(Task(TaskTypesEnum.classification)) \
        .with_tuner(tuner) \
        .build(train_data)

    tuned_pipeline = pipeline_tuner.tune(pipeline)

Evaluation
----------

.. _with_requirements:

Use ``.with_requirements()`` to set number of cv_folds and n_jobs.

.. code-block:: python

    requirements = PipelineComposerRequirements(cv_folds=2, n_jobs=2)

    pipeline_tuner = TunerBuilder(Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=10))) \
        .with_requirements(requirements) \
        .build(train_data)

    tuned_pipeline = pipeline_tuner.tune(pipeline)

.. _with_cv_folds:


.. _with_n_jobs:

Or use methods ``.with_cv_folds()``, ``.with_n_jobs()`` to set corresponding values separately.

.. code-block:: python

    pipeline_tuner = TunerBuilder(Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=10))) \
        .with_cv_folds(3) \
        .with_n_jobs(-1) \
        .build(train_data)

    tuned_pipeline = pipeline_tuner.tune(pipeline)

Metric
------

.. _with_metric:

Specify metric to optimize using ``.with_metric()``.

1. Metric can be chosen from ``ClassificationMetricsEnum``, ``RegressionMetricsEnum``.

.. code-block:: python

    metric = ClassificationMetricsEnum.ROCAUC

    pipeline_tuner = TunerBuilder(Task(TaskTypesEnum.classification)) \
        .with_metric(metric) \
        .build(train_data)

    tuned_pipeline = pipeline_tuner.tune(pipeline)

2. You can pass custom metric. For that, implement abstract class ``QualityMetric`` and pass ``CustomMetric.get_value`` as metric. **Note** that tuner will minimize the metric.

.. code-block:: python

    import sys
    from copy import deepcopy
    from sklearn.metrics import mean_squared_error as mse
    from fedot.core.composer.metrics import QualityMetric
    from fedot.core.data.data import InputData, OutputData
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
    from fedot.core.repository.tasks import TaskTypesEnum, Task


    class CustomMetric(QualityMetric):
        default_value = sys.maxsize

        @staticmethod
        def metric(reference: InputData, predicted: OutputData) -> float:
            mse_value = mse(reference.target, predicted.predict, squared=False)
            return (mse_value + 2) * 0.5


    pipeline_tuner = TunerBuilder(Task(TaskTypesEnum.regression)) \
        .with_metric(CustomMetric.get_value) \
        .build(train_data)

    tuned_pipeline = pipeline_tuner.tune(pipeline)

3. Another way to pass custom metric is to implement a function with the following signature: ``Callable[[G], Real]``. **Note** that tuner will minimize the metric.

.. code-block:: python

    from sklearn.metrics import mean_squared_error as mse
    from golem.core.dag.graph import Graph
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
    from fedot.core.repository.tasks import Task, TaskTypesEnum


    def custom_metric(graph: Graph, reference_data: InputData, **kwargs):
        result = graph.predict(reference_data)
        mse_value = mse(reference_data.target, result.predict, squared=False)
        return (mse_value + 2) * 0.5


    pipeline_tuner = TunerBuilder(Task(TaskTypesEnum.regression)) \
        .with_metric(custom_metric) \
        .build(train_data)

    tuned_pipeline = pipeline_tuner.tune(pipeline)

Search Space
------------

.. _with_search_space:

To set search space use ``.with_search_space()``. By default, tuner uses search space specified in ``fedot/core/pipelines/tuning/search_space.py``
To customize search space use ``PipelineSearchSpace`` class.

.. code-block:: python

    custom_search_space = {
        'logit': {
            'C': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [1e-1, 5.0],
                'type': 'continuous'}
        },
        'pca': {
            'n_components': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.1, 0.5],
                'type': 'continuous'}
        },
        'knn': {
            'n_neighbors': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 20],
                'type': 'discrete'},
            'weights': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [["uniform", "distance"]],
                'type': 'categorical'},
            'p': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [[1, 2]],
                'type': 'categorical'}
        }
    }
    search_space = PipelineSearchSpace(custom_search_space=custom_search_space, replace_default_search_space=True)

    pipeline_tuner = TunerBuilder(Task(TaskTypesEnum.classification)) \
            .with_search_space(search_space) \
            .build(train_data)

    tuned_pipeline = pipeline_tuner.tune(pipeline)

Additional parameters
---------------------

.. _with_additional_params:

If there is no ``TunerBuilder`` function to set a specific parameter of a tuner use ``.with_additional_params()``.

Possible additional parameters you can see in the `GOLEM documentation`_.

For example, you can set algorithm for with signature similar to ``hyperopt.tse.suggest`` for ``SimultaneousTuner`` or
``SequentialTuner``.

By default, ``hyperopt.tse.suggest`` is used.

.. code-block:: python

    pipeline_tuner = TunerBuilder(Task(TaskTypesEnum.classification)) \
        .with_additional_params(algo = hyperopt.rand.suggest) \
        .build(train_data)

    tuned_pipeline = pipeline_tuner.tune(pipeline)

For ``IOptTuner`` such parameters as ``r``, ``evolvent_density``, ``eps_r`` and etc can be set.

.. code-block:: python

    pipeline_tuner = TunerBuilder(Task(TaskTypesEnum.classification)) \
        .with_tuner(IOptTuner) \
        .with_additional_params(r = 1, evolvent_density = 5) \
        .build(train_data)

    tuned_pipeline = pipeline_tuner.tune(pipeline)

Constraints
-----------

.. _with_timeout:

* Use ``.with_timeout()`` to set timeout for tuning.

.. _with_iterations:

* Use ``.with_iterations()`` to set maximal number of tuning iterations.

.. _with_early_stopping_rounds:

* Use ``.with_early_stopping_rounds()`` to specify after what number of iterations without metric improvement tuning will be stopped.

.. _with_eval_time_constraint:

* Use ``.with_eval_time_constraint()`` to set  time constraint for pipeline fitting while it's evaluation.

.. code-block:: python

    timeout = datetime.timedelta(minutes=1)

    iterations = 500

    early_stopping_rounds = 50

    eval_time_constraint = datetime.timedelta(seconds=30)

    pipeline_tuner = TunerBuilder(task) \
        .with_timeout(timeout) \
        .with_iterations(iterations) \
        .with_early_stopping_rounds(early_stopping_rounds) \
        .with_eval_time_constraint(eval_time_constraint) \
        .build(input_data)

    tuned_pipeline = pipeline_tuner.tune(pipeline)

Examples
~~~~~~~~

Tuning all hyperparameters simultaneously
-----------------------------------------
Example for ``SimultaneousTuner``:

.. code-block:: python

    import datetime
    import hyperopt
    from golem.core.tuning.simultaneous import SimultaneousTuner
    from hyperopt import hp
    from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
    from fedot.core.repository.metrics_repository import ClassificationMetricsEnum
    from fedot.core.repository.tasks import TaskTypesEnum, Task

    task = Task(TaskTypesEnum.classification)

    tuner = SimultaneousTuner

    requirements = PipelineComposerRequirements(cv_folds=2, n_jobs=2)

    metric = ClassificationMetricsEnum.ROCAUC

    iterations = 500

    early_stopping_rounds = 50

    timeout = datetime.timedelta(minutes=1)

    eval_time_constraint = datetime.timedelta(seconds=30)

    custom_search_space = {
        'logit': {
            'C': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.01, 5.0],
                'type': 'continuous'}
        },
        'knn': {
            'n_neighbors': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 20],
                'type': 'discrete'},
            'weights': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [["uniform", "distance"]],
                'type': 'categorical'},
            'p': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [[1, 2]],
                'type': 'categorical'}}
    }
    search_space = PipelineSearchSpace(custom_search_space=custom_search_space, replace_default_search_space=True)

    algo = hyperopt.rand.suggest

    train_data = InputData.from_csv('train_file.csv')

    pipeline = PipelineBuilder().add_node('knn', branch_idx=0).add_branch('logit', branch_idx=1) \
        .grow_branches('logit', 'rf').join_branches('knn').build()

    pipeline_tuner = TunerBuilder(task) \
        .with_tuner(tuner) \
        .with_requirements(requirements) \
        .with_metric(metric) \
        .with_iterations(iterations) \
        .with_early_stopping_rounds(early_stopping_rounds) \
        .with_timeout(timeout) \
        .with_search_space(search_space) \
        .with_additional_params(algo=algo) \
        .with_eval_time_constraint(eval_time_constraint) \
        .build(train_data)

    tuned_pipeline = pipeline_tuner.tune(pipeline)

    tuned_pipeline.print_structure()

Tuned pipeline structure:

.. code-block:: python

    Pipeline structure:
    {'depth': 3, 'length': 5, 'nodes': [knn, logit, knn, rf, logit]}
    knn - {'n_neighbors': 3, 'p': 2, 'weights': 'uniform'}
    logit - {'C': 4.564184562288343}
    knn - {'n_neighbors': 6, 'p': 2, 'weights': 'uniform'}
    rf - {'n_jobs': 1, 'bootstrap': True, 'criterion': 'entropy', 'max_features': 0.46348491415788157, 'min_samples_leaf': 11, 'min_samples_split': 2, 'n_estimators': 100}
    logit - {'C': 3.056080157518786}


Example for ``IOptTuner``:

.. code-block:: python

    import datetime
    from golem.core.tuning.iopt_tuner import IOptTuner
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
    from fedot.core.repository.metrics_repository import RegressionMetricsEnum
    from fedot.core.repository.tasks import TaskTypesEnum, Task

    task = Task(TaskTypesEnum.regression)

    tuner = IOptTuner

    requirements = PipelineComposerRequirements(cv_folds=2, n_jobs=2)

    metric = RegressionMetricsEnum.MSE

    iterations = 100

    eval_time_constraint = datetime.timedelta(seconds=30)

    train_data = InputData.from_csv('train_data.csv', task='regression')

    pipeline = PipelineBuilder().add_node('knnreg', branch_idx=0).add_branch('rfr', branch_idx=1) \
        .join_branches('knnreg').build()

    pipeline_tuner = TunerBuilder(task) \
        .with_tuner(tuner) \
        .with_requirements(requirements) \
        .with_metric(metric) \
        .with_iterations(iterations) \
        .with_additional_params(eps=0.02, r=1, refine_solution=True) \
        .with_eval_time_constraint(eval_time_constraint) \
        .build(train_data)

    tuned_pipeline = pipeline_tuner.tune(pipeline)

    tuned_pipeline.print_structure()

Tuned pipeline structure:

.. code-block:: python

    Pipeline structure:
    {'depth': 2, 'length': 3, 'nodes': [knnreg, knnreg, rfr]}
    knnreg - {'n_neighbors': 51}
    knnreg - {'n_neighbors': 40}
    rfr - {'n_jobs': 1, 'max_features': 0.05324, 'min_samples_split': 12, 'min_samples_leaf': 11}

Example for ``OptunaTuner``:

.. code-block:: python

    from golem.core.tuning.optuna_tuner import OptunaTuner
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
    from fedot.core.repository.metrics_repository import RegressionMetricsEnum
    from fedot.core.repository.tasks import TaskTypesEnum, Task

    task = Task(TaskTypesEnum.regression)

    tuner = OptunaTuner

    metric = RegressionMetricsEnum.MSE

    iterations = 100

    train_data = InputData.from_csv('train_data.csv', task='regression')

    pipeline = PipelineBuilder().add_node('knnreg', branch_idx=0).add_branch('rfr', branch_idx=1) \
        .join_branches('knnreg').build()

    pipeline_tuner = TunerBuilder(task) \
        .with_tuner(tuner) \
        .with_metric(metric) \
        .with_iterations(iterations) \
        .build(train_data)

    tuned_pipeline = pipeline_tuner.tune(pipeline)

    tuned_pipeline.print_structure()

Tuned pipeline structure:

.. code-block:: python

    Pipeline structure:
    {'depth': 2, 'length': 3, 'nodes': [knnreg, knnreg, rfr]}
    knnreg - {'n_neighbors': 51}
    knnreg - {'n_neighbors': 40}
    rfr - {'n_jobs': 1, 'max_features': 0.05, 'min_samples_split': 12, 'min_samples_leaf': 11}


Multi objective tuning
^^^^^^^^^^^^^^^^^^^^^^

Multi objective tuning is available only for ``OptunaTuner``. Pass a list of metrics to ``.with_metric()``
and obtain a list of tuned pipelines representing a pareto front after tuning.

.. code-block:: python

    from typing import Iterable
    from golem.core.tuning.optuna_tuner import OptunaTuner
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
    from fedot.core.repository.metrics_repository import RegressionMetricsEnum
    from fedot.core.repository.tasks import TaskTypesEnum, Task

    task = Task(TaskTypesEnum.regression)

    tuner = OptunaTuner

    metric = [RegressionMetricsEnum.MSE, RegressionMetricsEnum.MAE]

    iterations = 100

    train_data = InputData.from_csv('train_data.csv', task='regression')

    pipeline = PipelineBuilder().add_node('knnreg', branch_idx=0).add_branch('rfr', branch_idx=1) \
        .join_branches('knnreg').build()

    pipeline_tuner = TunerBuilder(task) \
        .with_tuner(tuner) \
        .with_metric(metric) \
        .with_iterations(iterations) \
        .build(train_data)

    pareto_front: Iterable[Pipeline] = pipeline_tuner.tune(pipeline)


Sequential tuning
-----------------

.. code-block:: python

    import datetime
    from golem.core.tuning.sequential import SequentialTuner
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
    from fedot.core.repository.metrics_repository import RegressionMetricsEnum
    from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams

    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=10))

    tuner = SequentialTuner

    cv_folds = 3

    metric = RegressionMetricsEnum.RMSE

    iterations = 1000

    early_stopping_rounds = 50

    timeout = datetime.timedelta(minutes=1)

    train_data = InputData.from_csv_time_series(file_path='train_file.csv',
                                                task=task,
                                                target_column='target_name')

    pipeline = PipelineBuilder() \
        .add_sequence('locf', branch_idx=0) \
        .add_sequence('lagged', branch_idx=1) \
        .join_branches('ridge') \
        .build()

    pipeline_tuner = TunerBuilder(task) \
        .with_tuner(tuner) \
        .with_cv_folds(cv_folds) \
        .with_metric(metric) \
        .with_iterations(iterations) \
        .with_early_stopping_rounds(early_stopping_rounds) \
        .with_timeout(timeout) \
        .build(train_data)

    tuned_pipeline = pipeline_tuner.tune(pipeline)

    tuned_pipeline.print_structure()

Tuned pipeline structure:

.. code-block:: python

    Pipeline structure:
    {'depth': 2, 'length': 3, 'nodes': [ridge, locf, lagged]}
    ridge - {'alpha': 9.335457825369645}
    locf - {'part_for_repeat': 0.34751615772622124}
    lagged - {'window_size': 127}

Tuning of a node
----------------

.. code-block:: python

    import datetime
    from golem.core.tuning.sequential import SequentialTuner
    from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
    from fedot.core.repository.metrics_repository import RegressionMetricsEnum
    from fedot.core.repository.tasks import TaskTypesEnum, Task
    from test.integration.quality.test_synthetic_tasks import get_regression_data

    task = Task(TaskTypesEnum.regression)

    tuner = SequentialTuner

    requirements = PipelineComposerRequirements(cv_folds=2, n_jobs=-1)

    metric = RegressionMetricsEnum.SMAPE

    timeout = datetime.timedelta(minutes=5)

    train_data = get_regression_data()

    pipeline = PipelineBuilder().add_node('dtreg').grow_branches('lasso').build()


    pipeline_tuner = TunerBuilder(task) \
        .with_tuner(tuner) \
        .with_requirements(requirements) \
        .with_metric(metric) \
        .with_timeout(timeout) \
        .build(train_data)

    pipeline_with_tuned_node = pipeline_tuner.tune_node(pipeline, node_index=1)

    print('Node name: ', pipeline_with_tuned_node.nodes[1].content['name'])
    print('Node parameters: ', pipeline_with_tuned_node.nodes[1].custom_params)

Output:

.. code-block:: python

    Node name:  dtreg
    Node parameters:  {'max_depth': 2, 'min_samples_leaf': 6, 'min_samples_split': 21}

Another examples can be found here:

**Regression**

* `Regression with tuning <https://github.com/aimclub/FEDOT/blob/master/examples/simple/regression/regression_with_tuning.py>`_
* `Regression refinement example <https://github.com/aimclub/FEDOT/blob/master/examples/advanced/decompose/regression_refinement_example.py>`_

**Classification**

* `Classification with tuning <https://github.com/aimclub/FEDOT/blob/master/examples/simple/classification/classification_with_tuning.py>`_
* `Classification refinement example <https://github.com/aimclub/FEDOT/blob/master/examples/advanced/decompose/classification_refinement_example.py>`_
* `Resample example <https://github.com/aimclub/FEDOT/blob/master/examples/simple/classification/resample_example.py>`_
* `Pipeline tuning for classification task <https://github.com/aimclub/FEDOT/blob/master/examples/simple/pipeline_tune.py>`_

**Forecasting**

* `Pipeline tuning for time series forecasting <https://github.com/aimclub/FEDOT/blob/master/examples/simple/time_series_forecasting/tuning_pipelines.py>`_
* `Tuning pipelines with sparse_lagged / lagged node  <https://github.com/aimclub/FEDOT/blob/master/examples/advanced/time_series_forecasting/sparse_lagged_tuning.py>`_
* `Topaz multi time series forecasting <https://github.com/aimclub/FEDOT/blob/master/examples/advanced/time_series_forecasting/multi_ts_arctic_forecasting.py>`_
* `Custom model tuning <https://github.com/aimclub/FEDOT/blob/master/examples/advanced/time_series_forecasting/custom_model_tuning.py>`_
* `Case: river level forecasting with composer <https://github.com/aimclub/FEDOT/blob/master/examples/real_cases/river_levels_prediction/river_level_case_composer.py>`_
* `Case: river level forecasting (manual) <https://github.com/aimclub/FEDOT/blob/master/examples/real_cases/river_levels_prediction/river_level_case_manual.py>`_

**Multitask**

* `Multitask pipeline: classification and regression <https://github.com/aimclub/FEDOT/blob/master/examples/advanced/multitask_classification_regression.py>`_

.. _GOLEM documentation: https://thegolem.readthedocs.io/en/latest/api/tuning.html
