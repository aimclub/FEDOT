Tuning of Hyperparameters
=========================
FEDOT provide you with two ways for tuning of pipeline hyperparameters:

1. Tuning of all models hyperparameters simultaneously. Implemented via ``PipelineTuner`` class.

2. Tuning of models hyperparameters sequentially node by node optimizing metric value for the whole pipeline. Implemented via ``SequentialTuner`` class.

More information about these approaches can be found
'here <https://habr.com/ru/post/672486/>'_.

Simple example
~~~~~~~~~~~~~~
To initialize a tuner you can use ``TunerBuilder``:

.. code-block:: python

    from fedot.core.repository.tasks import TaskTypesEnum, Task
    from fedot.core.data.data import InputData
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder

    task = Task(TaskTypesEnum.classification)
    train_data = InputData.from_csv('train_file.csv')
    pipeline = PipelineBuilder().add_node('knn', branch_idx=0).add_branch('logit', branch_idx=1)\
        .grow_branches('logit', 'rf').join_branches('knn').to_pipeline()

    pipeline_tuner = TunerBuilder(task).build(train_data)

    tuned_pipeline = pipeline_tuner.tune(pipeline)
Detailed explanation
~~~~~~~~~~~~~~~~~~~~

Use ``.with_tuner()`` to specify tuner class to use. ``PipelineTuner`` is default value.

.. code-block:: python

    tuner = SequentialTuner

    pipeline_tuner = TunerBuilder(Task(TaskTypesEnum.classification)) \
        .with_tuner(tuner) \
        .build(train_data)
    tuned_pipeline = pipeline_tuner.tune(pipeline)


Use ``.with_requirements()`` to set number of cv_folds, validation_blocks (only for timeseries forecasting) and n_jobs.

.. code-block:: python

    requirements = ComposerRequirements(cv_folds=2, validation_blocks=3, n_jobs=2)

    pipeline_tuner = TunerBuilder(Task(TaskTypesEnum.ts_forecasting)) \
        .with_requirements(requirements) \
        .build(train_data)
    tuned_pipeline = pipeline_tuner.tune(pipeline)
Or use methods ``.with_cv_folds()``, ``.with_validation_blocks()``, ``.with_n_jobs()`` to set corresponding values.

.. code-block:: python

    pipeline_tuner = TunerBuilder(Task(TaskTypesEnum.ts_forecasting)) \
        .with_cv_folds(3) \
        .with_validation_blocks(2) \
        .with_n_jobs(-1) \
        .build(train_data)
    tuned_pipeline = pipeline_tuner.tune(pipeline)

Specify metric to optimize using ``.with_metric()``.

1. Metric can be chosen from ClusteringMetricsEnum, ClassificationMetricsEnum, RegressionMetricsEnum.

.. code-block:: python

    metric = ClassificationMetricsEnum.ROCAUC

    pipeline_tuner = TunerBuilder(Task(TaskTypesEnum.classification)) \
        .with_metric(metric) \
        .build(train_data)
    tuned_pipeline = pipeline_tuner.tune(pipeline)
2. You can pass custom metric. For that implement abstract class QualityMetric and pass CustomMetric.get_value. Note that tuner will minimize the metric.

.. code-block:: python

    import sys
    from copy import deepcopy

    from sklearn.metrics import mean_squared_error

    from fedot.core.composer.metrics import QualityMetric
    from fedot.core.data.data import InputData, OutputData
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
    from fedot.core.repository.tasks import TaskTypesEnum, Task


    class CustomMetric(QualityMetric):
        default_value = sys.maxsize

        @staticmethod
        def metric(reference: InputData, predicted: OutputData) -> float:
            return mean_squared_error(y_true=reference.target,
                                      y_pred=predicted.predict, squared=True)


    pipeline_tuner = TunerBuilder(Task(TaskTypesEnum.regression)) \
        .with_metric(CustomMetric.get_value) \
        .build(train_data)
    tuned_pipeline = pipeline_tuner.tune(deepcopy(pipeline))
Extended example
~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    import datetime
    from copy import deepcopy

    import hyperopt
    from hyperopt import hp

    from examples.simple.classification.classification_pipelines import classification_complex_pipeline
    from fedot.core.data.data import InputData
    from fedot.core.optimisers.composer_requirements import ComposerRequirements
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder
    from fedot.core.pipelines.tuning.search_space import SearchSpace
    from fedot.core.pipelines.tuning.sequential import SequentialTuner
    from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
    from fedot.core.pipelines.tuning.unified import PipelineTuner
    from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
    from fedot.core.repository.tasks import TaskTypesEnum, Task
    from fedot.core.utils import fedot_project_root

    # use task to initialize TunerBuilder
    task = Task(TaskTypesEnum.classification)

    # specify tuner class to use. Default tuner - PipelineTuner
    tuner = PipelineTuner

    # use requirements to set number of cv_folds, validation blocks for ts forecasting and n_jobs
    requirements = ComposerRequirements(cv_folds=2, n_jobs=2, show_progress=True)
    # or use special methods: with_cv_folds(), with_validation_blocks(), with_n_jobs()

    # pass metric.
    # 1. Metric can be chosen from ClusteringMetricsEnum, ClassificationMetricsEnum, RegressionMetricsEnum
    # 2. You can pass your custom metric. For that implement abstract class QualityMetric and pass CustomMetric.get_value.
    # Note that tuner will minimize custom metric.
    metric = ClassificationMetricsEnum.ROCAUC

    # set number of tuning iterations
    iterations = 500

    # early_stopping_rounds specify after what number of iterations without metric improvement search will be stopped
    early_stopping_rounds = 50

    # timeout for tuning
    timeout = datetime.timedelta(minutes=1)

    # eval_time_constraint - time constraint for pipeline fit while it's evaluation
    eval_time_constraint = 5

    # set search_space. Use SeqrchSpace class to customize it.
    custom_search_space = {
        'logit': {
            'C': (hp.uniform, [0.01, 5.0])
        },
        'pca': {
            'n_components': (hp.uniform, [0.2, 0.8])
        },
        'knn': {
            'n_neighbors': (hp.uniformint, [1, 6]),
            'weights': (hp.choice, [["uniform", "distance"]]),
            'p': (hp.choice, [[1, 2]])}
    }
    search_space = SearchSpace(custom_search_space=custom_search_space, replace_default_search_space=True)

    # set algorithm for hyperparameters optimization with signature similar to :obj:`hyperopt.tse.suggest`
    # By default, `hyperopt.tse.suggest` is used
    algo = hyperopt.rand.suggest

    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join(str(fedot_project_root()), 'test/data/advanced_classification.csv')
    input_data = InputData.from_csv(os.path.join(test_file_path, file), task=task)
    # pipeline = classification_complex_pipeline()
    pipeline = PipelineBuilder().add_node('knn', branch_idx=0).add_branch('logit', branch_idx=1)\
        .grow_branches('logit', 'rf').join_branches('knn').to_pipeline()
    # pipeline1.show()


    pipeline_tuner = TunerBuilder(task) \
        .with_tuner(tuner) \
        .with_requirements(requirements) \
        .with_metric(metric) \
        .with_iterations(iterations) \
        .with_early_stopping_rounds(early_stopping_rounds) \
        .with_timeout(timeout) \
        .with_search_space(search_space) \
        .with_algo(algo) \
        .with_eval_time_constraint(eval_time_constraint) \
        .build(input_data)
    print('_________pipeline', pipeline_tuner.get_metric_value(pipeline))
    tuned_pipeline = pipeline_tuner.tune(deepcopy(pipeline))

    tuner = SequentialTuner
    seq_pipeline_tuner = TunerBuilder(task) \
        .with_tuner(tuner) \
        .with_requirements(requirements) \
        .with_metric(metric) \
        .with_iterations(iterations) \
        .with_early_stopping_rounds(early_stopping_rounds) \
        .with_timeout(timeout) \
        .with_search_space(search_space) \
        .with_algo(algo) \
        .with_eval_time_constraint(eval_time_constraint) \
        .build(input_data)
    seq_tuned_pipeline = seq_pipeline_tuner.tune(deepcopy(pipeline))
    pipeline_with_tuned_node = seq_pipeline_tuner.tune_node(deepcopy(pipeline), node_index=0)

    print('pipeline', pipeline_tuner.get_metric_value(pipeline))
    print('tuned_pipeline', pipeline_tuner.get_metric_value(tuned_pipeline))
    print('seq_tuned_pipeline', seq_pipeline_tuner.get_metric_value(seq_tuned_pipeline))
    print('pipeline_with_tuned_node', seq_pipeline_tuner.get_metric_value(pipeline_with_tuned_node))
