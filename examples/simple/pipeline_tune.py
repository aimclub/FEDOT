import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from cases.data.data_utils import get_scoring_case_data_paths
from examples.simple.classification.classification_pipelines import classification_complex_pipeline
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.unified import PipelineTuner


def get_case_train_test_data():
    """ Function for getting data for train and validation """
    train_file_path, test_file_path = get_scoring_case_data_paths()

    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)
    return train_data, test_data


def pipeline_tuning(pipeline: Pipeline, train_data: InputData,
                    test_data: InputData, local_iter: int,
                    tuner_iter_num: int = 30) -> (float, list):
    """ Function for tuning pipeline with PipelineTuner

    :param pipeline: pipeline to tune
    :param train_data: InputData for train
    :param test_data: InputData for validation
    :param local_iter: amount of tuner launches
    :param tuner_iter_num: amount of iterations, which tuner will perform

    :return mean_metric: mean value of ROC AUC metric
    :return several_iter_scores_test: list with metrics
    """
    several_iter_scores_test = []
    for iteration in range(local_iter):
        print(f'current local iteration {iteration}')

        # Pipeline tuning
        pipeline_tuner = PipelineTuner(pipeline=pipeline,
                                       task=train_data.task,
                                       iterations=tuner_iter_num)
        tuned_pipeline = pipeline_tuner.tune_pipeline(input_data=train_data,
                                                      loss_function=roc_auc)

        # After tuning prediction
        tuned_pipeline.fit(train_data)
        after_tuning_predicted = tuned_pipeline.predict(test_data)

        # Metrics
        aft_tun_roc_auc = roc_auc(y_true=test_data.target,
                                  y_score=after_tuning_predicted.predict)
        several_iter_scores_test.append(aft_tun_roc_auc)

    mean_metric = float(np.mean(several_iter_scores_test))
    return mean_metric, several_iter_scores_test


if __name__ == '__main__':
    train_data, test_data = get_case_train_test_data()

    # Pipeline composition
    pipeline = classification_complex_pipeline()

    # Before tuning prediction
    pipeline.fit(train_data, use_fitted=False)
    before_tuning_predicted = pipeline.predict(test_data)
    bfr_tun_roc_auc = roc_auc(y_true=test_data.target,
                              y_score=before_tuning_predicted.predict)

    local_iter = 5
    # Pipeline tuning
    after_tune_roc_auc, several_iter_scores_test = pipeline_tuning(pipeline=pipeline,
                                                                   train_data=train_data,
                                                                   test_data=test_data,
                                                                   local_iter=local_iter)

    print(f'Several test scores {several_iter_scores_test}')
    print(f'Mean test score over {local_iter} iterations: {after_tune_roc_auc}')
    print(round(bfr_tun_roc_auc, 3))
    print(round(after_tune_roc_auc, 3))
