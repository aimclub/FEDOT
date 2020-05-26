from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.metrics import f1_score, mean_squared_error, r2_score, roc_auc_score

from benchmark.H2O.b_h2o import run_h2o
from benchmark.autokeras.b_autokeras import run_autokeras
from benchmark.baseline.b_xgboost import run_xgboost
from benchmark.benchmark_model_types import BenchmarkModelTypesEnum
from benchmark.fedot.fedot_classification import run_classification_problem
from benchmark.fedot.fedot_regression import run_regression_problem
from benchmark.mlbox.b_mlbox import run_mlbox
from benchmark.tpot.b_tpot import run_tpot
from core.repository.tasks import TaskTypesEnum, Task


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
class CaseExecutor:
    train_file: str
    test_file: str
    case_label: str
    target_name: str
    task: TaskTypesEnum
    models: List[BenchmarkModelTypesEnum]
    metric_list: List[str]

    def execute(self):
        print('START EXECUTION')

        result = {'task': self.task.value}

        if BenchmarkModelTypesEnum.tpot in self.models:
            print('---------\nRUN TPOT\n---------')
            tpot_result = run_tpot(train_file_path=self.train_file,
                                   test_file_path=self.test_file,
                                   case_name=self.case_label,
                                   task=Task(self.task))

            result['tpot_metric'] = calculate_metrics(self.metric_list,
                                                      target=tpot_result[0],
                                                      predicted=tpot_result[1])
        if BenchmarkModelTypesEnum.h2o in self.models:
            print('---------\nRUN H2O\n---------')
            h2o_result = run_h2o(train_file_path=self.train_file,
                                 test_file_path=self.test_file,
                                 case_name=self.case_label,
                                 task=self.task)
            result['h2o_metric'] = calculate_metrics(self.metric_list,
                                                     target=h2o_result[0],
                                                     predicted=h2o_result[1])
        if BenchmarkModelTypesEnum.autokeras in self.models:
            print('---------\nRUN AUTOKERAS\n---------')
            autokeras_result = run_autokeras(train_file_path=self.train_file,
                                             test_file_path=self.test_file,
                                             case_name=self.case_label,
                                             task=self.task)
            result['autokeras_metric'] = calculate_metrics(self.metric_list,
                                                           target=autokeras_result[0],
                                                           predicted=autokeras_result[1])
        if BenchmarkModelTypesEnum.fedot in self.models:
            print('---------\nRUN FEDOT\n---------')

            if self.task is TaskTypesEnum.classification:
                fedot_problem_func = run_classification_problem
            elif self.task is TaskTypesEnum.regression:
                fedot_problem_func = run_regression_problem
            else:
                raise NotImplementedError()
            single, static, evo_composed, target = fedot_problem_func(train_file_path=self.train_file,
                                                                      test_file_path=self.test_file)

            result['fedot_metric'] = {'composed': calculate_metrics(self.metric_list,
                                                                    target=target,
                                                                    predicted=evo_composed),
                                      'static': calculate_metrics(self.metric_list,
                                                                  target=target,
                                                                  predicted=static),
                                      'single': calculate_metrics(self.metric_list,
                                                                  target=target,
                                                                  predicted=single)}
        if BenchmarkModelTypesEnum.mlbox in self.models:
            print('---------\nRUN MLBOX\n---------')
            mlbox_result = run_mlbox(train_file_path=self.train_file,
                                     test_file_path=self.test_file,
                                     target_name=self.target_name,
                                     task=self.task)

            result['mlbox_metric'] = calculate_metrics(self.metric_list,
                                                       target=mlbox_result[0],
                                                       predicted=mlbox_result[1])

        if BenchmarkModelTypesEnum.baseline in self.models:
            print('---------\nRUN BASELINE\n---------')
            xgboost_result = run_xgboost(train_file_path=self.train_file, test_file_path=self.test_file,
                                         target_name=self.target_name, task=self.task)
            result['baseline_metric'] = calculate_metrics(self.metric_list,
                                                          target=xgboost_result[0],
                                                          predicted=xgboost_result[1])

        return result
