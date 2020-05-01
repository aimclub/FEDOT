import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from core.models.data import InputData
from core.repository.model_types_repository import ModelTypesIdsEnum
from core.repository.task_types import MachineLearningTasksEnum
from experiments.chain_template import (chain_template_balanced_tree, fit_template,
                                        show_chain_template, real_chain)
from experiments.composer_benchmark import to_labels
from experiments.generate_data import synthetic_dataset


def models_to_use():
    models = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.knn,
              ModelTypesIdsEnum.dt]
    return models


def source_chain(model_types, samples, features, classes):
    template = chain_template_balanced_tree(model_types=model_types, depth=4, models_per_level=[8, 4, 2, 1],
                                            samples=samples, features=features)
    show_chain_template(template)
    fit_template(template, classes=classes, skip_fit=False)
    initialized_chain = real_chain(template)

    return initialized_chain, template


def data_generated_by(chain, samples, features_amount, classes):
    task_type = MachineLearningTasksEnum.classification
    features, target = synthetic_dataset(samples_amount=samples,
                                         features_amount=features_amount,
                                         classes_amount=classes)
    target = np.expand_dims(target, axis=1)
    train = InputData(idx=np.arange(0, samples),
                      features=features, target=target, task_type=task_type)
    synth_target = chain.predict(input_data=train).predict
    synth_labels = to_labels(synth_target)
    data_synth_train = InputData(idx=np.arange(0, samples),
                                 features=features, target=synth_labels, task_type=task_type)

    chain.fit_from_scratch(input_data=data_synth_train)

    features, target = synthetic_dataset(samples_amount=samples,
                                         features_amount=features_amount,
                                         classes_amount=classes)
    target = np.expand_dims(target, axis=1)
    test = InputData(idx=np.arange(0, samples),
                     features=features, target=target, task_type=task_type)
    synth_target = chain.predict(input_data=test).predict
    synth_labels = to_labels(synth_target)
    data_synth_test = InputData(idx=np.arange(0, samples),
                                features=features, target=synth_labels, task_type=task_type)
    return data_synth_test


def roc_score(chain, data_to_compose, data_to_validate):
    predicted_train = chain.predict(data_to_compose)
    predicted_test = chain.predict(data_to_validate)
    # the quality assessment for the simulation results
    roc_train = roc_auc(y_true=data_to_compose.target,
                        y_score=predicted_train.predict)

    roc_test = roc_auc(y_true=data_to_validate.target,
                       y_score=predicted_test.predict)
    print(f'Train ROC: {roc_train}')
    print(f'Test ROC: {roc_test}')

    return roc_train, roc_test


if __name__ == '__main__':
    samples, features_amount, classes = 10000, 10, 2
    chain, template = source_chain(model_types=models_to_use(),
                                   samples=samples, features=features_amount,
                                   classes=classes)
    data_synth_test = data_generated_by(chain, samples, features_amount, classes)
    roc_score(chain, data_synth_test, data_synth_test)
