import logging
from datetime import datetime

from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.preprocessing import LabelEncoder

from fedot import Fedot
from fedot.core.constants import Consts
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import fedot_project_root
from fedot.core.utils import set_random_seed


def calculate_validation_metric(pipeline: Pipeline, dataset_to_validate: InputData) -> float:
    # the execution of the obtained composite models
    predicted = pipeline.predict(dataset_to_validate)
    # the quality assessment for the simulation results
    roc_auc_value = roc_auc(y_true=dataset_to_validate.target,
                            y_score=predicted.predict)
    return roc_auc_value


def run_problem(timeout: float = 5.0,
                visualization=False,
                target='target',
                model_type="auto",
                **composer_args):

    # file_path_train = 'cases/data/mfeat-pixel.csv'
    # full_path_train = fedot_project_root().joinpath(file_path_train)

    file_path_train = 'cases/data/cows/train.csv'
    full_path_train = fedot_project_root().joinpath(file_path_train)

    data = InputData.from_csv(full_path_train, task='regression', target_columns='milk_yield_10')
    # target = data.target

    # encoded = LabelEncoder().fit_transform(target)
    # data.target = encoded

    train, test = train_test_data_setup(data, shuffle=True)
    print('Model:', model_type, '-- Use Label Encoding:', Consts.USE_LABEL_ENC_AS_DEFAULT, end='\t')
    print('-- Before preprocessing', train.features.shape, end=' ')

    metric_name = 'rmse'
    automl = Fedot(problem='regression',
                   timeout=timeout,
                   logging_level=logging.FATAL,
                   metric=metric_name,
                   **composer_args)

    if model_type != "auto":
        start_time = datetime.now()
        automl.fit(train, predefined_model=model_type)
        end_time = datetime.now()
        print('-- Stated Time limit:', timeout, end=' ')
        print('- Run Time:', end_time - start_time, end='\t')
    else:
        automl.fit(train)

    automl.predict(test)
    metrics = automl.get_metrics()

    if automl.history and automl.history.generations:
        print(automl.history.get_leaderboard())
        automl.history.show()

    if visualization:
        automl.current_pipeline.show()

    print(f'{metric_name} = {round(metrics["f1"], 3)}')
    print('-' * 10)

    return metrics["f1"]


if __name__ == '__main__':
    set_random_seed(42)

    Consts.USE_LABEL_ENC_AS_DEFAULT = True
    print('\t\t -- Label Encoding --')
    # run_problem(timeout=1,
    #             visualization=False,
    #             with_tuning=False, model_type='logit')
    #
    # run_problem(timeout=1,
    #             visualization=False,
    #             with_tuning=False, model_type='dt')
    #
    # run_problem(timeout=1,
    #             visualization=False,
    #             with_tuning=False, model_type='rf')
    #
    # run_problem(timeout=1,
    #             visualization=False,
    #             with_tuning=False, model_type='xgboost')
    #
    # run_problem(timeout=1,
    #             visualization=False,
    #             with_tuning=False, model_type='lgbm')

    run_problem(timeout=10,
                visualization=False,
                with_tuning=True, model_type='auto')

    print('\t\t -- One Hot Encoding --')

    Consts.USE_LABEL_ENC_AS_DEFAULT = False

    # run_problem(timeout=1,
    #             visualization=False,
    #             with_tuning=True, model_type='logit')
    #
    # run_problem(timeout=1,
    #             visualization=False,
    #             with_tuning=False, model_type='dt')
    #
    # run_problem(timeout=1,
    #             visualization=False,
    #             with_tuning=False, model_type='rf')
    #
    # run_problem(timeout=1,
    #             visualization=False,
    #             with_tuning=False, model_type='xgboost')
    #
    # run_problem(timeout=1,
    #             visualization=False,
    #             with_tuning=False, model_type='lgbm')

    run_problem(timeout=10,
                visualization=False,
                with_tuning=True, model_type='auto')
