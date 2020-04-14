from dataclasses import dataclass

from benchmark.H2O.b_h2o import run_h2o
from benchmark.autokeras.b_autokeras import run_autokeras
from benchmark.baseline.b_xgboost import run_xgboost
from benchmark.tpot.b_tpot import run_tpot
from cases.credit_scoring_problem import run_credit_scoring_problem


@dataclass
class CaseExecutor:
    train_file: str
    test_file: str
    case: str
    label: str
    hyperparameters: dict
    is_classification: bool
    tpot: bool = True
    h2o: bool = True
    autokeras: bool = True
    fedot: bool = True
    baseline: bool = True

    def execute(self):
        print('START EXECUTION')

        task = 'classification' if self.is_classification else 'regression'
        saved_metric_results = {'task': task}

        if self.tpot:
            print('---------')
            print('RUN TPOT')
            print('---------')
            tpot_result = run_tpot(train_file_path=self.train_file,
                                   test_file_path=self.test_file,
                                   case_name=self.case,
                                   config_data=self.hyperparameters['TPOT'],
                                   is_classification=True if self.is_classification else False)
            saved_metric_results['tpot_metric'] = tpot_result
        if self.h2o:
            print('---------')
            print('RUN H2O')
            print('---------')
            h2o_result = run_h2o(train_file_path=self.train_file,
                                 test_file_path=self.test_file,
                                 case_name=self.case,
                                 target_name=self.label,
                                 config_data=self.hyperparameters['H2O'],
                                 is_classification=True if self.is_classification else False)
            saved_metric_results['h2o_metric'] = h2o_result
        if self.autokeras:
            print('---------')
            print('RUN AUTOKERAS')
            print('---------')
            autokeras_result = run_autokeras(train_file_path=self.train_file,
                                             test_file_path=self.test_file,
                                             case_name=self.case,
                                             config_data=self.hyperparameters['AutoKeras'],
                                             is_classification=True if self.is_classification else False)
            saved_metric_results['autokeras_metric'] = autokeras_result
        if self.fedot:
            print('---------')
            print('RUN FEDOT')
            print('---------')
            if self.is_classification:
                fedot_result = run_credit_scoring_problem(train_file_path=self.train_file,
                                                          test_file_path=self.test_file)
                saved_metric_results['fedot_metric'] = {'composed_roc_auc': fedot_result[0],
                                                        'static_roc_auc': fedot_result[1],
                                                        'single_model_roc_auc': fedot_result[2]}
        if self.baseline:
            print('---------')
            print('RUN BASELINE')
            print('---------')
            xgboost_result = run_xgboost(train_file_path=self.train_file, test_file_path=self.test_file,
                                         target_name=self.label)
            saved_metric_results['baseline_metric'] = xgboost_result

        return saved_metric_results
