import random
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from core.composer.composer import DummyChainTypeEnum
from core.composer.composer import DummyComposer
from core.composer.visualisation import ChainStyles, ChainVisualiser
from core.models.data import InputData
from core.repository.dataset_types import NumericalDataTypesEnum, CategoricalDataTypesEnum
from core.repository.model_types_repository import (
    ModelMetaInfoTemplate,
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from core.repository.task_types import MachineLearningTasksEnum

random.seed(1)
np.random.seed(1)

file_path = '../test/data/test_dataset.csv'
path = Path(__file__).parent / file_path
dataset_from_file = InputData.from_csv(path)

# a dataset that will be used as a train and test set during composition
dataset_to_compose = dataset_from_file
# a dataset for a final validation of the composed model
dataset_to_validate = dataset_from_file

# the search of the models provided by the framework that can be used as nodes in a chain for the selected task
models_repo = ModelTypesRepository()
available_model_names = models_repo.search_model_types_by_attributes(
    desired_metainfo=ModelMetaInfoTemplate(input_type=NumericalDataTypesEnum.table,
                                           output_type=CategoricalDataTypesEnum.vector,
                                           task_type=MachineLearningTasksEnum.classification))

models_impl = [models_repo.model_by_id(model_name) for model_name in available_model_names]

# the choice of the metric for the chain quality assessment during composition
metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

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
predicted_seq = chain_seq.evaluate(dataset_to_validate)
predicted_single = chain_single.evaluate(dataset_to_validate)

# the quality assessment for the simulation results
roc_on_train_seq = roc_auc(y_true=dataset_to_validate.target,
                           y_score=predicted_seq.predict)

roc_on_train_single = roc_auc(y_true=dataset_to_validate.target,
                              y_score=predicted_single.predict)

print(f'Seq chain ROC AUC is {round(roc_on_train_seq, 3)}')
print(f'Single-model chain ROC AUC is {round(roc_on_train_single, 3)}')

visualiser = ChainVisualiser(ChainStyles.circular)
visualiser.visualise(chain_seq)
