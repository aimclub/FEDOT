"""H21 observation only: no extra fits, predictions, RNG changes or graph selection."""
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np


def save(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v))
        f.write('\n')


def graph_record(graph):
    from examples.advanced.diagnostics.calibrated_tuner_probe import graph_record as record
    return record(graph)


@contextmanager
def observe_pools(directory):
    from fedot.core.operations.evaluation.operation_implementations.models.boostings_implementations import FedotCatBoostImplementation
    directory.mkdir(parents=True, exist_ok=True)
    original = FedotCatBoostImplementation.fit

    @wraps(original)
    def fit(implementation, input_data):
        pools = []
        convert = implementation.convert_to_pool
        def pool(value, identify_cats):
            result = convert(value, identify_cats)
            pools.append(dict(rows=np.asarray(value.idx).tolist(), count=result.num_row(),
                              categories=result.get_cat_feature_indices()))
            return result
        implementation.convert_to_pool = pool
        try:
            result = original(implementation, input_data)
            save(directory/f'{os.getpid()}-{time.monotonic_ns()}.json', dict(pools=pools,
                params=result.get_params(), trees=int(result.tree_count_),
                callback_count=len(implementation.fit_callbacks)))
            return result
        finally:
            del implementation.convert_to_pool
    with patch.object(FedotCatBoostImplementation, 'fit', fit):
        yield


def install(model, directory):
    """Factory calls on BOTH policies with the same observer code."""
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluate
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    original_fit = model.fit
    original_tune = model.api_composer.tune_final_pipeline
    original_full = model._train_pipeline_on_full_dataset
    evaluations = []
    state = {}

    def tune(data, graph, history):
        save(directory/'tuning-entry.json', dict(graph=graph_record(graph), rows=len(data.idx),
            elapsed=time.monotonic()-model._fit_started_at,
            cv_folds=model.params.composer_requirements.cv_folds,
            per_fit_cap=model.params.composer_requirements.max_graph_fit_time.total_seconds()))
        with (directory/'train-target.npz').open('xb') as f:
            np.savez_compressed(f, target=np.asarray(data.target).ravel(), rows=np.asarray(data.idx))
        evaluate = PipelineObjectiveEvaluate.evaluate
        prepare = PipelineObjectiveEvaluate.prepare_graph
        predict = Pipeline.predict

        def observed_evaluate(evaluator, pipeline):
            number = len(evaluations)
            cell = directory/'cv'/f'{number:03d}'
            cell.mkdir(parents=True)
            state.update(cell=cell, trains={}, predictions=0)
            before = time.monotonic()
            record = dict(number=number, graph=graph_record(pipeline),
                cap=evaluator._evaluation_time_constraint.total_seconds() if evaluator._evaluation_time_constraint else None)
            try:
                result = evaluate(evaluator, pipeline)
                record.update(valid=bool(result.valid), fitness=float(result.value) if result.valid else None)
                return result
            finally:
                record['seconds'] = time.monotonic()-before
                evaluations.append(record)
                save(cell/'evaluation.json', record)
                state.clear()

        def observed_prepare(evaluator, pipeline, train, fold_id=None, n_jobs=-1, **kwargs):
            state['trains'][fold_id] = np.asarray(train.idx).copy()
            with observe_pools(state['cell']/f'pool-{fold_id}'):
                return prepare(evaluator, pipeline, train, fold_id, n_jobs, **kwargs)

        def observed_predict(pipeline, data, *args, **kwargs):
            output = predict(pipeline, data, *args, **kwargs)
            probs = np.asarray(output.predict)
            fold = kwargs.get('fold_id')
            if probs.ndim == 2 and probs.shape[1] > 1 and fold in state.get('trains', {}):
                filename = state['cell']/f'prediction-{state["predictions"]}.npz'
                state['predictions'] += 1
                fitted = pipeline.root_node.fitted_operation
                estimator = getattr(fitted, 'model', fitted)
                classes = np.asarray(getattr(estimator, 'classes_', np.unique(np.asarray(data.target))))
                with filename.open('xb') as f:
                    np.savez_compressed(f, probs=probs, truth=np.asarray(data.target).ravel(),
                        rows=np.asarray(data.idx), train_rows=state['trains'][fold], classes=classes,
                        fold=np.asarray(fold))
            return output

        before = time.monotonic()
        try:
            with patch.object(PipelineObjectiveEvaluate, 'evaluate', observed_evaluate), \
                 patch.object(PipelineObjectiveEvaluate, 'prepare_graph', observed_prepare), \
                 patch.object(Pipeline, 'predict', observed_predict):
                result = original_tune(data, graph, history)
            save(directory/'tuning-return.json', dict(graph=graph_record(result)))
            return result
        finally:
            policy = getattr(model.api_composer, '_experimental_paid_policy', None)
            save(directory/'tuning-exit.json', dict(seconds=time.monotonic()-before,
                policy=policy.record if policy else None, evaluations=evaluations))

    def full(*args, **kwargs):
        before = time.monotonic()
        try:
            with observe_pools(directory/'final-pools'):
                return original_full(*args, **kwargs)
        finally:
            save(directory/'final-fit.json', dict(seconds=time.monotonic()-before,
                graph=graph_record(model.current_pipeline)))

    @wraps(original_fit)
    def fit(*args, **kwargs):
        before = time.monotonic()
        error = None
        try:
            return original_fit(*args, **kwargs)
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            import fedot, golem
            if getattr(model,'history',None):
                history_path=directory/'history.json'
                assert not history_path.exists()
                model.history.save(str(history_path))
            save(directory/'fit.json', dict(seconds=time.monotonic()-before, error=error,
                policy_enabled=hasattr(model.api_composer, '_experimental_paid_policy'),
                api_seconds=model.params.timeout*60,
                final_graph=graph_record(model.current_pipeline) if model.current_pipeline else None,
                runtime=dict(fedot=fedot.__file__, golem=golem.__file__),
                environment={k:os.environ.get(k) for k in ['PYTHONPATH','FEDOT_OPERATION_DEFAULTS_OVERLAY']}))
    model.fit = fit
    model.api_composer.tune_final_pipeline = tune
    model._train_pipeline_on_full_dataset = full


def patch_adapter(source):
    old = '''        install(model)
    return model


def _fit_full_candidate('''
    new = '''        install(model)
    observation_dir = config.framework_params.get("_paid_policy_observation_dir")
    if observation_dir:
        from examples.advanced.diagnostics.full_policy_observer import install as observe
        observe(model, observation_dir)
    return model


def _fit_full_candidate('''
    assert source.count(old) == 1
    return source.replace(old, new)


def transport_probe(dataset, config):
    from examples.advanced.diagnostics.adapter_cv_probe import load_adapter
    from frameworks.shared.callee import result
    import fedot
    adapter=load_adapter(Path(config.observer_stage))
    model=adapter._make_fedot(config,'neg_log_loss',dict(n_jobs=8,with_tuning=True,
        use_operations_cache=False,use_preprocessing_cache=False,use_predictions_cache=False),54,5.4)
    directory=Path(config.framework_params['_paid_policy_observation_dir'])
    return result(models_count=0,training_duration=0,predict_duration=0,probe=dict(
        flag=hasattr(model.api_composer,'_experimental_paid_policy'),
        observed=directory.is_dir() and hasattr(model.fit,'__wrapped__'),
        fits_recorded=len(list(directory.glob('fit.json'))),fedot_file=fedot.__file__))


if __name__=='__main__':
    from frameworks.shared.callee import call_run
    call_run(transport_probe)
