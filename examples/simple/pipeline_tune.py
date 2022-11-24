from copy import deepcopy

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from cases.data.data_utils import get_scoring_case_data_paths
from examples.simple.classification.classification_pipelines import classification_complex_pipeline
from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum


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
    tuner = TunerBuilder(train_data.task) \
        .with_tuner(PipelineTuner) \
        .with_metric(ClassificationMetricsEnum.ROCAUC) \
        .with_iterations(tuner_iter_num) \
        .build(train_data)
    for iteration in range(local_iter):
        print(f'current local iteration {iteration}')

        # Pipeline tuning
        tuned_pipeline = tuner.tune(pipeline)

        # After tuning prediction
        tuned_pipeline.fit(train_data)
        after_tuning_predicted = tuned_pipeline.predict(test_data)

        # Metrics
        aft_tun_roc_auc = roc_auc(y_true=test_data.target,
                                  y_score=after_tuning_predicted.predict)
        several_iter_scores_test.append(aft_tun_roc_auc)

    max_metric = float(np.max(several_iter_scores_test))
    return max_metric, several_iter_scores_test


if __name__ == '__main__':
    train_data, test_data = get_case_train_test_data()

    # Pipeline composition
    pipeline = classification_complex_pipeline()

    # Before tuning prediction
    pipeline.fit(train_data)
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
    print(f'Maximal test score over {local_iter} iterations: {after_tune_roc_auc}')
    print(f'ROC-AUC before tuning {round(bfr_tun_roc_auc, 3)}')
    print(f'ROC-AUC after tuning {round(after_tune_roc_auc, 3)}')
