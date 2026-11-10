"""Train-only 2x2 graph ablation: frozen parameters versus leaf Hessian.

Reads the final graphs from two completed logs. Uses only a prepared OpenML
outer-training cache; this is not a full AutoMLBenchmark score. No tuning.
"""
import argparse
import ast
import json
import random
import time
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
    for seed in (42, 104732):
        for label, log in [('baseline', args.baseline_log), ('candidate', args.candidate_log)]:
            for weight in (0.001, 1.0):
                random.seed(seed)
                np.random.seed(seed)
                parameters = final_parameters(log)
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
                              parameters=parameters)
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
    main(parser.parse_args())
