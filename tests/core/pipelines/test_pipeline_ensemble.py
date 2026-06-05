import numpy as np

from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.pipelines.ensembling.routing import SamplingRoutingContext
from fedot.core.pipelines.ensembling.pipeline_ensemble import PipelineEnsemble
from fedot.core.pipelines.ensembling.utils import calculate_validation_metrics
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


class _FakePipeline:
    use_input_preprocessing = False
    preprocessor = None

    def __init__(self, prediction, is_fitted=True, probabilities=None):
        self.prediction = np.asarray(prediction)
        self.probabilities = None if probabilities is None else np.asarray(probabilities)
        self.is_fitted = is_fitted
        self.predict_calls = 0

    def predict(self, input_data, output_mode='default', predictions_cache=None, fold_id=None):
        self.predict_calls += 1
        prediction = self.probabilities if output_mode in ('probs', 'full_probs') else self.prediction
        if len(prediction) != len(input_data.idx):
            prediction = prediction[np.asarray(input_data.idx, dtype=int)]
        return OutputData(
            idx=input_data.idx,
            features=input_data.features,
            task=input_data.task,
            data_type=input_data.data_type,
            target=input_data.target,
            predict=prediction,
        )


def _regression_data():
    return _make_regression_data(row_count=2)


def _make_regression_data(row_count):
    return InputData(
        idx=np.arange(row_count),
        features=np.arange(row_count * 2).reshape(row_count, 2),
        target=np.zeros(row_count),
        task=Task(TaskTypesEnum.regression),
        data_type=DataTypesEnum.table,
    )


def _classification_data():
    return InputData(
        idx=np.arange(2),
        features=np.arange(4).reshape(2, 2),
        target=np.array([0, 1]),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )


def test_pipeline_ensemble_predict_builds_output_without_template_prediction():
    input_data = _regression_data()
    first = _FakePipeline([1.0, 3.0])
    second = _FakePipeline([3.0, 5.0])
    ensemble = PipelineEnsemble(
        pipelines=[first, second],
        validation_metric='rmse',
        ensemble_method='voting',
    )

    output = ensemble.predict(input_data)

    assert np.array_equal(output.idx, input_data.idx)
    assert output.task is input_data.task
    assert output.data_type is input_data.data_type
    assert np.array_equal(output.predict, np.array([2.0, 4.0]))
    assert first.predict_calls == 1
    assert second.predict_calls == 1


def test_classification_ensemble_aggregates_pipeline_probabilities():
    input_data = _classification_data()
    first = _FakePipeline([0, 0], probabilities=[[0.8, 0.2], [0.7, 0.3]])
    second = _FakePipeline([1, 1], probabilities=[[0.4, 0.6], [0.1, 0.9]])
    ensemble = PipelineEnsemble(
        pipelines=[first, second],
        validation_metric='f1',
        ensemble_method='voting',
    )

    output = ensemble.predict(input_data)
    proba = ensemble.predict(input_data, output_mode='probs')

    assert np.array_equal(output.predict, np.array([0, 1]))
    assert np.allclose(proba.predict, np.array([[0.6, 0.4], [0.4, 0.6]]))


def test_ensemble_validation_metrics_use_fedot_minimization_values():
    metrics = calculate_validation_metrics(
        y_true=np.array([0, 1, 1, 0]),
        y_labels=np.array([0, 1, 1, 0]),
        y_proba=np.array([
            [0.9, 0.1],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.8, 0.2],
        ]),
        task_type=TaskTypesEnum.classification,
    )

    assert metrics['accuracy'] < 0
    assert metrics['f1'] < 0
    assert metrics['roc_auc'] < 0
    assert metrics['neg_log_loss'] > 0


def test_pipeline_ensemble_predict_uses_batches_for_large_input():
    input_data = _make_regression_data(row_count=5)
    first = _FakePipeline(np.arange(5))
    second = _FakePipeline(np.arange(5) + 2)
    ensemble = PipelineEnsemble(
        pipelines=[first, second],
        validation_metric='rmse',
        ensemble_method='voting',
        batch_size=2,
    )

    output = ensemble.predict(input_data)

    assert np.array_equal(output.predict, np.array([1., 2., 3., 4., 5.]))
    assert first.predict_calls == 3
    assert second.predict_calls == 3


def test_select_best_models_checks_all_pipeline_subsets_without_repredicting():
    input_data = _regression_data()
    first = _FakePipeline([-1.0, 1.0])
    second = _FakePipeline([1.0, -1.0])
    third = _FakePipeline([2.0, 2.0])
    ensemble = PipelineEnsemble(
        pipelines=[first, second, third],
        validation_metric='rmse',
        ensemble_method='voting',
        pipeline_infos=[
            {'pipeline': first, 'metrics': {'rmse': 1.0}, 'val_predictions': first.prediction},
            {'pipeline': second, 'metrics': {'rmse': 1.0}, 'val_predictions': second.prediction},
            {'pipeline': third, 'metrics': {'rmse': 2.0}, 'val_predictions': third.prediction},
        ],
    )

    selected, best_score = ensemble.select_best_models(input_data, validation_metric='rmse')

    assert selected == [0, 1]
    assert best_score == 0.0
    assert ensemble.pipelines == [first, second]
    assert first.predict_calls == 0
    assert second.predict_calls == 0
    assert third.predict_calls == 0


def test_pipeline_ensemble_finalize_drops_unfitted_candidates_and_selects_best_subset():
    input_data = _regression_data()
    first = _FakePipeline([-1.0, 1.0])
    second = _FakePipeline([1.0, -1.0])
    unfitted = _FakePipeline([2.0, 2.0], is_fitted=False)
    ensemble = PipelineEnsemble(
        pipelines=[first, second, unfitted],
        validation_metric='rmse',
        ensemble_method='voting',
        pipeline_infos=[
            {'pipeline': first, 'metrics': {'rmse': 1.0}, 'val_predictions': first.prediction},
            {'pipeline': second, 'metrics': {'rmse': 1.0}, 'val_predictions': second.prediction},
            {'pipeline': unfitted, 'metrics': {'rmse': 2.0}, 'val_predictions': unfitted.prediction},
        ],
    )

    ensemble.finalize(validation_data=input_data)

    assert ensemble.pipelines == [first, second]
    assert ensemble.is_fitted is True


def test_weighted_ensemble_uses_lower_fedot_metric_value_as_better():
    first = _FakePipeline([0.0, 0.0])
    second = _FakePipeline([10.0, 10.0])
    ensemble = PipelineEnsemble(
        pipelines=[first, second],
        validation_metric='f1',
        ensemble_method='weighted',
        pipeline_infos=[
            {'pipeline': first, 'metrics': {'f1': -0.9}},
            {'pipeline': second, 'metrics': {'f1': -0.7}},
        ],
    )

    weights = ensemble._compute_model_weights(ensemble.pipeline_infos)

    assert weights[0] > weights[1]


class _FakePartitionPredictor:
    partition_names_ = ('chunk_0', 'chunk_1')

    def predict_partition_proba(self, features):
        return np.asarray([[1.0, 0.0], [0.0, 1.0]])


def test_routed_weighted_ensemble_uses_sample_wise_partition_weights():
    input_data = _regression_data()
    first = _FakePipeline([1.0, 1.0])
    second = _FakePipeline([10.0, 10.0])
    routing_context = SamplingRoutingContext.from_predictor(
        predictor=_FakePartitionPredictor(),
        partition_names=('chunk_0', 'chunk_1'),
    )
    ensemble = PipelineEnsemble(
        pipelines=[first, second],
        validation_metric='rmse',
        ensemble_method='routed_weighted',
        routing_context=routing_context,
        pipeline_infos=[
            {'name': 'chunk_0', 'pipeline': first, 'metrics': {'rmse': 1.0}, 'val_predictions': first.prediction},
            {'name': 'chunk_1', 'pipeline': second, 'metrics': {'rmse': 1.0}, 'val_predictions': second.prediction},
        ],
    )

    output = ensemble.predict(input_data)

    assert np.array_equal(output.predict, np.array([1.0, 10.0]))
