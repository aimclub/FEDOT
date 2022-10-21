from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.model_selection import train_test_split

from examples.simple.classification.classification_pipelines import classification_pipeline_without_balancing, \
    classification_pipeline_with_balancing
from examples.simple.classification.classification_with_tuning import get_classification_dataset
from fedot.core.data.data import InputData
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.utils import fedot_project_root


def run_resample_example(path_to_data=None, tune=False):
    if path_to_data is None:
        samples = 1000
        features = 10
        classes = 2
        weights = [0.45, 0.55]
        features_options = {'informative': 1, 'redundant': 1, 'repeated': 1, 'clusters_per_class': 1}

        x_train, y_train, x_test, y_test = get_classification_dataset(features_options,
                                                                      samples,
                                                                      features,
                                                                      classes,
                                                                      weights)
    else:
        data = pd.read_csv(path_to_data, header=0)

        features = data.drop(columns='Class')
        target = data['Class']

        x_train, x_test, y_train, y_test = train_test_split(np.array(features),
                                                            np.array(target), test_size=0.3)

    unique_class, counts_class = np.unique(y_train, return_counts=True)
    print(f'Two classes: {unique_class}')
    print(f'{unique_class[0]}: {counts_class[0]}')
    print(f'{unique_class[1]}: {counts_class[1]}')

    task = Task(TaskTypesEnum.classification)

    train_input = InputData(idx=np.arange(0, len(x_train)),
                            features=x_train,
                            target=y_train,
                            task=task,
                            data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_test)),
                              features=x_test,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.table)

    print('Begin fit Pipeline without balancing')
    # Pipeline without balancing
    pipeline = classification_pipeline_without_balancing()
    pipeline.fit_from_scratch(train_input)

    # Predict
    predict_labels = pipeline.predict(predict_input)
    preds = predict_labels.predict
    print('---')
    print(f'ROC-AUC of pipeline without balancing {roc_auc(y_test, preds):.4f}\n')

    # Pipeline with balancing
    pipeline = classification_pipeline_with_balancing()

    print('Begin fit Pipeline with balancing')
    # pipeline.fit(train_input)
    pipeline.fit(train_input)

    # Predict
    predict_labels = pipeline.predict(predict_input)
    preds = predict_labels.predict
    print('---')
    print(f'ROC-AUC of pipeline with balancing {roc_auc(y_test, preds):.4f}\n')

    if tune:
        print('Start tuning process ...')
        tuner = TunerBuilder(train_input.task)\
            .with_tuner(PipelineTuner)\
            .with_metric(RegressionMetricsEnum.MAE)\
            .with_iterations(50) \
            .with_timeout(timedelta(minutes=1))\
            .build(train_input)
        tuned_pipeline = tuner.tune(pipeline)
        # Fit
        pipeline.fit(train_input)
        # Predict
        predicted_values_tuned = tuned_pipeline.predict(predict_input)
        preds_tuned = predicted_values_tuned.predict

        print('Obtained metrics after tuning:')
        print(f'ROC-AUC of tuned pipeline with balancing - {roc_auc(y_test, preds_tuned):.4f}\n')


if __name__ == '__main__':
    run_resample_example()
    print('=' * 25)
    run_resample_example(f'{fedot_project_root()}/examples/data/credit_card_anomaly.csv', tune=True)
