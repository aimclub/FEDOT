"""Train-only 2x2 graph ablation: frozen parameters versus leaf Hessian.

Reads the final graphs from two completed logs. Uses only a prepared OpenML
outer-training cache; this is not a full AutoMLBenchmark score. No tuning.
"""
import argparse
import ast
import json
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def final_parameters(path):
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    start = max(i for i, line in enumerate(lines) if 'SimultaneousTuner - Final graph:' in line)
    parameters = {}
    for line in lines[start + 1:start + 6]:
        operation, values = line.split(' - ', 1)
        parameters[operation] = ast.literal_eval(values)
    if set(parameters) != {'scaling', 'catboost', 'xgboost', 'lgbm', 'logit'}:
        raise ValueError('Unexpected graph: this diagnostic requires the recorded five-node stack')
    return parameters


def parameter_variants(baseline, candidate, component_ablation=False, lgbm_ablation=False,
                       learning_rate_horizon=False, horizon_confirmation=False):
    """Frozen controls and one-factor swaps; never tune on the holdout."""
    if sum((component_ablation, lgbm_ablation, learning_rate_horizon, horizon_confirmation)) > 1:
        raise ValueError('Select only one ablation mode')
    if learning_rate_horizon or horizon_confirmation:
        variants = []
        rates = [('baseline', baseline['lgbm']['learning_rate']), ('010', 0.1)]
        if learning_rate_horizon:
            rates = [rates[0], ('006', 0.06), rates[1], ('015', 0.15)]
        for rounds in (100, 400):
            for label, rate in rates:
                parameters = deepcopy(baseline)
                parameters['lgbm'].update(learning_rate=rate, n_estimators=rounds)
                variants.append((f'lr_{label}_trees_{rounds}', parameters))
        return variants
    if lgbm_ablation:
        fields = ('learning_rate', 'num_leaves', 'min_data_in_leaf',
                  'colsample_bytree', 'reg_alpha', 'reg_lambda', 'subsample')
        changed = {key for key in baseline['lgbm'].keys() | candidate['lgbm'].keys()
                   if baseline['lgbm'].get(key) != candidate['lgbm'].get(key)}
        if changed != set(fields):
            raise ValueError(f'Unexpected LightGBM parameter differences: {changed}')
        full_swap = deepcopy(baseline)
        full_swap['lgbm'] = deepcopy(candidate['lgbm'])
        variants = [('baseline', deepcopy(baseline)), ('swap_lgbm', full_swap)]
        for field in fields:
            parameters = deepcopy(baseline)
            parameters['lgbm'][field] = candidate['lgbm'][field]
            variants.append(('lgbm_' + field, parameters))
        return variants
    variants = [('baseline', deepcopy(baseline)), ('candidate', deepcopy(candidate))]
    if component_ablation:
        for operation in ('catboost', 'xgboost', 'lgbm', 'logit'):
            parameters = deepcopy(baseline)
            parameters[operation] = deepcopy(candidate[operation])
            variants.append(('swap_' + operation, parameters))
    return variants


def lgbm_fit_diagnostics(pipeline):
    node = next(node for node in pipeline.nodes if node.operation.operation_type == 'lgbm')
    operation = node.fitted_operation
    model = getattr(operation, 'model', operation)
    if model is None:
        return {}
    return dict(requested_rounds=int(model.n_estimators),
                fitted_rounds=int(model.n_estimators_),
                best_iteration=int(model.best_iteration_),
                learning_rate=float(model.learning_rate))


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.data) as data:
        features, target = data['features'], data['target']
        task_id, fold = int(data['task_id']), int(data['fold'])
    rows_train, rows_valid = train_test_split(np.arange(len(target)), test_size=0.2,
                                             stratify=target, random_state=42)

    def input_data(rows):
        return InputData(idx=np.arange(len(rows)), features=features[rows], target=target[rows],
                         task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)

    train, valid = input_data(rows_train), input_data(rows_valid)
    results = []
    lgbm_ablation = getattr(args, 'lgbm_ablation', False)
    horizon = getattr(args, 'learning_rate_horizon', False)
    confirmation = getattr(args, 'horizon_confirmation', False)
    parameter_sets = parameter_variants(final_parameters(args.baseline_log),
                                       final_parameters(args.candidate_log),
                                       args.component_ablation, lgbm_ablation, horizon, confirmation)
    single_factor = args.component_ablation or lgbm_ablation or horizon or confirmation
    seeds = (42,) if single_factor else (42, 104732)
    weights = (0.001,) if single_factor else (0.001, 1.0)
    for seed in seeds:
        for label, frozen_parameters in parameter_sets:
            for weight in weights:
                random.seed(seed)
                np.random.seed(seed)
                parameters = deepcopy(frozen_parameters)
                parameters['lgbm']['min_child_weight'] = weight
                for model in ('lgbm', 'xgboost', 'logit'):
                    parameters[model]['random_state'] = seed
                parameters['catboost']['random_seed'] = seed
                scaling = PipelineNode('scaling')
                learners = [PipelineNode(name, nodes_from=[scaling])
                            for name in ('catboost', 'xgboost', 'lgbm')]
                pipeline = Pipeline(PipelineNode('logit', nodes_from=learners))
                for node in pipeline.nodes:
                    node.parameters = parameters[node.operation.operation_type]
                started = time.monotonic()
                pipeline.fit(train, n_jobs=8)
                probabilities = pipeline.predict(valid, output_mode='full_probs').predict
                assert np.isfinite(probabilities).all()
                assert (probabilities >= 0).all() and (probabilities <= 1).all()
                assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)
                result = dict(parameter_source=label, min_child_weight=weight, seed=seed,
                              seconds=time.monotonic() - started,
                              inner_logloss=log_loss(valid.target, probabilities),
                              parameters=parameters,
                              lgbm_fit=lgbm_fit_diagnostics(pipeline))
                if horizon or confirmation:
                    fitted = result['lgbm_fit']
                    assert fitted['requested_rounds'] == parameters['lgbm']['n_estimators']
                    assert 1 <= fitted['fitted_rounds'] <= fitted['requested_rounds']
                    assert fitted['learning_rate'] == parameters['lgbm']['learning_rate']
                if args.save_predictions:
                    prediction_path = args.output.with_name(f'{args.output.stem}-cell{len(results)}.npz')
                    with prediction_path.open('xb') as stream:
                        np.savez_compressed(stream, probabilities=probabilities,
                                            truth=np.asarray(valid.target).ravel(),
                                            classes=np.unique(train.target))
                    result['predictions'] = prediction_path.name
                results.append(result)
                print(json.dumps(result), flush=True)
                pipeline.unfit()
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(dict(task_id=task_id, fold=fold, split='outer_train_inner_holdout',
                       train_rows=len(rows_train), validation_rows=len(rows_valid),
                       results=results), stream, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'baseline-log', 'candidate-log', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--component-ablation', action='store_true')
    modes.add_argument('--lgbm-ablation', action='store_true')
    modes.add_argument('--learning-rate-horizon', action='store_true')
    modes.add_argument('--horizon-confirmation', action='store_true')
    parser.add_argument('--save-predictions', action='store_true')
    main(parser.parse_args())
