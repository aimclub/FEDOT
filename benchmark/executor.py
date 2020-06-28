from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.metrics import f1_score, mean_squared_error, r2_score, roc_auc_score

from benchmark.H2O.b_h2o import run_h2o
from benchmark.autokeras.b_autokeras import run_autokeras
from benchmark.baseline.b_xgboost import run_xgboost
from benchmark.benchmark_model_types import BenchmarkModelTypesEnum
from benchmark.fedot.b_fedot import run_fedot
from benchmark.mlbox.b_mlbox import run_mlbox
from benchmark.tpot.b_tpot import run_tpot
from core.repository.tasks import TaskTypesEnum


def calculate_metrics(metric_list: list, target: list, predicted: list):
    metric_dict = {'roc_auc': roc_auc_score,
                   'f1': f1_score,
                   'mse': mean_squared_error,
                   'r2': r2_score
                   }
    target_only_metrics = ['f1', 'accuracy', 'precision']
    result_metric = []
    for metric_name in metric_list:
        if metric_name in target_only_metrics:
            bound = np.mean(predicted)
            predicted = [1 if x >= bound else 0 for x in predicted]

        result_metric.append(round(metric_dict[metric_name](target, predicted), 3))

    result_dict = dict(zip(metric_list, result_metric))

    return result_dict


@dataclass
class ExecutionParams:
    train_file: str
    test_file: str
    case_label: str
    target_name: str
    task: TaskTypesEnum


@dataclass
class CaseExecutor:
    models: List[BenchmarkModelTypesEnum]
    metric_list: List[str]
    params: ExecutionParams

    _strategy_by_type = {
        BenchmarkModelTypesEnum.tpot: run_tpot,
        BenchmarkModelTypesEnum.h2o: run_h2o,
        BenchmarkModelTypesEnum.autokeras: run_autokeras,
        BenchmarkModelTypesEnum.mlbox: run_mlbox,
        BenchmarkModelTypesEnum.fedot: run_fedot,
        BenchmarkModelTypesEnum.baseline: run_xgboost
    }

    def execute(self):
        print('START EXECUTION')

        result = {'task': self.params.task.value}

        strategies = {model_type: self._strategy_by_type[model_type] for
                      model_type in self.models}

        for model_type, strategy_func in strategies.items():
            print(f'---------\nRUN {model_type.name}\n---------')
            target, predicted = strategy_func(self.params)
            result[f'{model_type.name}_metric'] = calculate_metrics(self.metric_list,
                                                                    target=target,
                                                                    predicted=predicted)

        return result
