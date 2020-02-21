import random

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from core.composer.composer import DummyChainTypeEnum
from core.composer.composer import DummyComposer
from core.models.data import Data
from core.repository.dataset_types import NumericalDataTypesEnum, CategoricalDataTypesEnum
from core.repository.model_types_repository import (
    ModelMetaInfoTemplate,
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from core.repository.task_types import MachineLearningTasksEnum

random.seed(1)
np.random.seed(1)


def log_function_dataset():
    samples = 1000
    x = 10.0 * np.random.rand(samples, ) - 5.0
    x = np.expand_dims(x, axis=1)
    y = 1.0 / (1.0 + np.exp(np.power(x, -1.0)))
    threshold = 0.5
    classes = np.array([0.0 if val <= threshold else 1.0 for val in y])
    classes = np.expand_dims(classes, axis=1)
    data = Data(features=x, target=classes, idx=np.arange(0, len(x)))

    return data


# file_path = '../test/data/test_dataset.csv'
# path = Path(__file__).parent / file_path
# dataset = Data.from_csv(path)

# a dataset that will be used as a train and test set during composition
dataset_to_compose = log_function_dataset()
# a dataset for a final validation of the composed model
dataset_to_validate = log_function_dataset()

# the search of the models provided by the framework that can be used as nodes in a chain for the selected task
models_repo = ModelTypesRepository()
available_model_names = models_repo.search_model_types_by_attributes(
    desired_metainfo=ModelMetaInfoTemplate(input_type=NumericalDataTypesEnum.table,
                                           output_type=CategoricalDataTypesEnum.vector,
                                           task_type=MachineLearningTasksEnum.classification))

models_impl = [models_repo.obtain_model_implementation(model_name) for model_name in available_model_names]

# the choice of the metric for the chain quality assessment during composition
metric_function = MetricsRepository().obtain_metric_implementation(ClassificationMetricsEnum.ROCAUC)

# the choice and initialisation of the composer
composer = DummyComposer(DummyChainTypeEnum.flat)

# the optimal chain generation by composition - the most time-consuming task
chain_seq = composer.compose_chain(data=dataset_to_compose,
                                   initial_chain=None,
                                   primary_requirements=[models_impl[1]],
                                   secondary_requirements=[models_impl[1]],
                                   metrics=metric_function)

# the second variant of optimal chain generation by composition with another requirements
chain_single = composer.compose_chain(data=dataset_to_compose,
                                      initial_chain=None,
                                      primary_requirements=[models_impl[1]],
                                      secondary_requirements=[],
                                      metrics=metric_function)

#
# the execution of the obtained composite models
predicted_seq = chain_seq.evaluate_with_specific_data(dataset_to_validate)
predicted_single = chain_single.evaluate_with_specific_data(dataset_to_validate)

# the quality assessment for the simulation results
roc_on_train_seq = roc_auc(y_true=dataset_to_validate.target,
                           y_score=predicted_seq)

roc_on_train_single = roc_auc(y_true=dataset_to_validate.target,
                              y_score=predicted_single)

print(f'Seq chain ROC AUC is {round(roc_on_train_seq, 3)}')
print(f'Single-model chain ROC AUC is {round(roc_on_train_single, 3)}')
