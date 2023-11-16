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
    file_path_train = 'cases/data/mfeat-pixel.csv'
    full_path_train = fedot_project_root().joinpath(file_path_train)

    data = InputData.from_csv(full_path_train, task='classification', target_columns='class')
    target = data.target
    encoded = LabelEncoder().fit_transform(target)
    data.target = encoded

    train, test = train_test_data_setup(data, shuffle=True)
    print(model_type, Consts.USE_LABEL_ENC_AS_DEFAULT)
    automl = Fedot(problem='classification',
                   timeout=timeout,
                   logging_level=logging.FATAL,
                   metric='f1',
                   **composer_args)
    if model_type != "auto":
        start_time = datetime.now()
        automl.fit(train, predefined_model=model_type)
        end_time = datetime.now()
        print(end_time - start_time)
        print(train.features.shape)
    else:
        automl.fit(train)


    automl.predict(test)
    metrics = automl.get_metrics()

    if automl.history and automl.history.generations:
        print(automl.history.get_leaderboard())
        automl.history.show()

    if visualization:
        automl.current_pipeline.show()

    print(f'f1 is {round(metrics["f1"], 3)}')

    return metrics["f1"]


if __name__ == '__main__':
    set_random_seed(42)

    Consts.USE_LABEL_ENC_AS_DEFAULT = True
    print('Labelenc')
    run_problem(timeout=1,
                visualization=False,
                with_tuning=False, model_type='logit')

    run_problem(timeout=1,
                visualization=False,
                with_tuning=False, model_type='xgboost')

    # run_problem(timeout=10,
    #             visualization=True,
    #             with_tuning=True, model_type='auto')

    print('OH etc')

    Consts.USE_LABEL_ENC_AS_DEFAULT = False

    run_problem(timeout=1,
                visualization=False,
                with_tuning=True, model_type='logit')

    run_problem(timeout=1,
                visualization=False,
                with_tuning=False, model_type='xgboost')

    # run_problem(timeout=10,
    #             visualization=True,
    #             with_tuning=True, model_type='auto')
