from dataclasses import dataclass
from typing import List

from benchmark.H2O.b_h2o import run_h2o
from benchmark.autokeras.b_autokeras import run_autokeras
from benchmark.baseline.b_xgboost import run_xgboost
from benchmark.benchmark_model_types import BenchmarkModelTypesEnum
from benchmark.mlbox.b_mlbox import run_mlbox
from benchmark.tpot.b_tpot import run_tpot
from cases.credit_scoring_problem import run_credit_scoring_problem
from core.repository.task_types import MachineLearningTasksEnum


@dataclass
class CaseExecutor:
    train_file: str
    test_file: str
    case_label: str
    target_name: str
    task: MachineLearningTasksEnum
    models: List[BenchmarkModelTypesEnum]

    def execute(self):
        print('START EXECUTION')

        saved_metric_results = {'task': self.task.value}

        if BenchmarkModelTypesEnum.tpot in self.models:
            print('---------\nRUN TPOT\n---------')
            tpot_result = run_tpot(train_file_path=self.train_file,
                                   test_file_path=self.test_file,
                                   case_name=self.case_label,
                                   task=self.task)
            saved_metric_results['tpot_metric'] = tpot_result
        if BenchmarkModelTypesEnum.h2o in self.models:
            print('---------\nRUN H2O\n---------')
            h2o_result = run_h2o(train_file_path=self.train_file,
                                 test_file_path=self.test_file,
                                 case_name=self.case_label,
                                 task=self.task)
            saved_metric_results['h2o_metric'] = h2o_result
        if BenchmarkModelTypesEnum.autokeras in self.models:
            print('---------\nRUN AUTOKERAS\n---------')
            autokeras_result = run_autokeras(train_file_path=self.train_file,
                                             test_file_path=self.test_file,
                                             case_name=self.case_label,
                                             task=self.task)
            saved_metric_results['autokeras_metric'] = autokeras_result
        if BenchmarkModelTypesEnum.fedot in self.models:
            print('---------\nRUN FEDOT\n---------')
            if self.task is MachineLearningTasksEnum.classification:
                fedot_result = run_credit_scoring_problem(train_file_path=self.train_file,
                                                          test_file_path=self.test_file)
                saved_metric_results['fedot_metric'] = {'composed_roc_auc': fedot_result[0],
                                                        'static_roc_auc': fedot_result[1],
                                                        'single_model_roc_auc': fedot_result[2]}
        if BenchmarkModelTypesEnum.mlbox in self.models:
            print('---------\nRUN MLBOX\n---------')
            mlbox_result = run_mlbox(train_file_path=self.train_file,
                                     test_file_path=self.test_file,
                                     target_name=self.target_name,
                                     task=self.task)

            saved_metric_results['mlbox_metric'] = mlbox_result

        if BenchmarkModelTypesEnum.baseline in self.models:
            print('---------\nRUN BASELINE\n---------')
            xgboost_result = run_xgboost(train_file_path=self.train_file, test_file_path=self.test_file,
                                         target_name=self.target_name)
            saved_metric_results['baseline_metric'] = xgboost_result

        return saved_metric_results
