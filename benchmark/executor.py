from dataclasses import dataclass

from benchmark.H2O.b_h2o import run_h2o
from benchmark.baseline.b_xgboost import run_xgboost
from benchmark.tpot.b_tpot import run_tpot
from cases.credit_scoring_problem import run_credit_scoring_problem


@dataclass
class CaseExecutor:
    train_file: str
    test_file: str
    case: str
    tpot: bool = True
    h2o: bool = True
    fedot: bool = True
    baseline: bool = True

    def execute(self):
        if self.tpot:
            print('---------')
            print('RUN TPOT')
            print('---------')
            run_tpot(train_file_path=self.train_file, test_file_path=self.test_file, case_name=self.case)
        if self.h2o:
            print('---------')
            print('RUN H2O')
            print('---------')
            run_h2o(train_file_path=self.train_file, test_file_path=self.test_file, case_name=self.case)
        if self.fedot:
            print('---------')
            print('RUN FEDOT')
            print('---------')
            run_credit_scoring_problem(train_file_path=self.train_file, test_file_path=self.test_file)
        if self.baseline:
            print('---------')
            print('RUN BASELINE')
            print('---------')
            run_xgboost(train_file_path=self.train_file, test_file_path=self.test_file)