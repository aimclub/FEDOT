import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.model_selection import train_test_split

from examples.simple.classification.classification_pipelines import classification_random_forest_pipeline
from fedot.core.data.data import InputData
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.synth_dataset_generator import classification_dataset

np.random.seed(2020)


def get_classification_dataset(features_options, samples_amount=250,
                               features_amount=5, classes_amount=2, weights=None):
    """
    Prepares four numpy arrays with different scale features and target
    :param samples_amount: Total amount of samples in the resulted dataset.
    :param features_amount: Total amount of features per sample.
    :param classes_amount: The amount of classes in the dataset.
    :param features_options: The dictionary containing features options in key-value
    :param weights: The proportions of samples assigned to each class. If None, then classes are balanced.
    format:
        - informative: the amount of informative features;
        - redundant: the amount of redundant features;
        - repeated: the amount of features that repeat the informative features;
        - clusters_per_class: the amount of clusters for each class;
    :return x_data_train: features to train
    :return y_data_train: target to train
    :return x_data_test: features to test
    :return y_data_test: target to test
    """

    x_data, y_data = classification_dataset(samples_amount=samples_amount,
                                            features_amount=features_amount,
                                            classes_amount=classes_amount,
                                            features_options=features_options,
                                            weights=weights)

    # Changing the scale of the data
    for i, coeff in zip(range(0, features_amount),
                        np.random.randint(1, 100, features_amount)):
        # Get column
        feature = np.array(x_data[:, i])

        # Change scale for this feature
        rescaled = feature * coeff
        x_data[:, i] = rescaled

    # Train and test split
    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(x_data, y_data,
                                                                            test_size=0.3)

    return x_data_train, y_data_train, x_data_test, y_data_test


def convert_to_labels(root_operation, prediction):
    if any(root_operation == acceptable_model for acceptable_model in
           ['logit', 'lda', 'qda', 'mlp', 'svc', 'rf', 'bernb']):
        preds = np.array(prediction.predict)
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
    else:
        preds = np.array(prediction.predict, dtype=int)

    return preds


def run_classification_tuning_experiment(pipeline, tuner=None):
    samples = [50, 550, 150]
    features = [1, 5, 10]
    classes = [2, 2, 2]
    options = [{'informative': 1, 'redundant': 0,
                'repeated': 0, 'clusters_per_class': 1},
               {'informative': 2, 'redundant': 1,
                'repeated': 1, 'clusters_per_class': 1},
               {'informative': 3, 'redundant': 1,
                'repeated': 2, 'clusters_per_class': 2}]

    for samples_amount, features_amount, \
        classes_amount, features_options in zip(samples, features, classes,
                                                options):
        print('=======================================')
        print(f'\nAmount of samples {samples_amount}, '
              f'amount of features {features_amount}, '
              f'amount of clsses {classes_amount}, '
              f'additional options {features_options}')

        x_train, y_train, x_test, y_test = get_classification_dataset(features_options,
                                                                      samples_amount,
                                                                      features_amount,
                                                                      classes_amount)
        task = Task(TaskTypesEnum.classification)

        # Prepare data to train the model
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

        # Fit it
        pipeline.fit(train_input)

        # Predict
        predicted_labels = pipeline.predict(predict_input)
        preds = predicted_labels.predict

        print(f"{roc_auc(y_test, preds):.4f}\n")

        if tuner is not None:
            print('Start tuning process ...')
            pipeline_tuner = TunerBuilder(task)\
                .with_tuner(tuner)\
                .with_metric(ClassificationMetricsEnum.ROCAUC)\
                .with_iterations(50)\
                .build(train_input)
            tuned_pipeline = pipeline_tuner.tune(pipeline)

            # Fit it
            tuned_pipeline.fit(train_input)

            # Predict
            print('predict')
            predicted_values_tuned = tuned_pipeline.predict(predict_input)
            preds_tuned = predicted_values_tuned.predict

            print('Obtained metrics after tuning:')
            print(f"{roc_auc(y_test, preds_tuned):.4f}\n")


# Script for testing is pipeline can process different datasets for classification
if __name__ == '__main__':
    run_classification_tuning_experiment(pipeline=classification_random_forest_pipeline(),
                                         tuner=PipelineTuner)
