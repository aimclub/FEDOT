import random
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from core.composer.composer import DummyChainTypeEnum
from core.composer.composer import DummyComposer
from core.composer.random_composer import RandomSearchComposer
from core.models.data import InputData
from core.models.model import XGBoost
from core.repository.dataset_types import NumericalDataTypesEnum, CategoricalDataTypesEnum
from core.repository.model_types_repository import (
    ModelMetaInfoTemplate,
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from core.repository.task_types import MachineLearningTasksEnum

random.seed(1)
np.random.seed(1)

# the dataset was obtained from https://www.kaggle.com/kashnitsky/a5-demo-logit-and-rf-for-credit-scoring

# a dataset that will be used as a train and test set during composition
dataset_to_compose = InputData.from_csv(Path(__file__).parent / 'data/scoring/scoring_train.csv')
# a dataset for a final validation of the composed model
dataset_to_validate = InputData.from_csv(Path(__file__).parent / 'data/scoring/scoring_test.csv')

# the search of the models provided by the framework that can be used as nodes in a chain for the selected task
models_repo = ModelTypesRepository()
available_model_names = models_repo.search_model_types_by_attributes(
    desired_metainfo=ModelMetaInfoTemplate(input_type=NumericalDataTypesEnum.table,
                                           output_type=CategoricalDataTypesEnum.vector,
                                           task_type=MachineLearningTasksEnum.classification,
                                           can_be_initial=True,
                                           can_be_secondary=True))

models_impl = [models_repo.model_by_id(model_name) for model_name in available_model_names]

# the choice of the metric for the chain quality assessment during composition
metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

# the choice and initialisation of the dummy_composer
dummy_composer = DummyComposer(DummyChainTypeEnum.hierarchical)
# the choice and initialisation of the random_search
random_composer = RandomSearchComposer(iter_num=10)

# the optimal chain generation by composition - the most time-consuming task
chain_random_composed = random_composer.compose_chain(data=dataset_to_compose,
                                                      initial_chain=None,
                                                      primary_requirements=models_impl,
                                                      secondary_requirements=models_impl,
                                                      metrics=metric_function)

chain_static = dummy_composer.compose_chain(data=dataset_to_compose,
                                            initial_chain=None,
                                            primary_requirements=models_impl,
                                            secondary_requirements=[XGBoost()],
                                            metrics=metric_function)

# the single-model variant of optimal chain
chain_single = DummyComposer(DummyChainTypeEnum.flat).compose_chain(data=dataset_to_compose,
                                                                    initial_chain=None,
                                                                    primary_requirements=[XGBoost()],
                                                                    secondary_requirements=[],
                                                                    metrics=metric_function)

print("Composition finished")

#
# the execution of the obtained composite models
predicted_seq = chain_static.predict(dataset_to_validate)
predicted_single = chain_single.predict(dataset_to_validate)
predicted_random_composed = chain_random_composed.predict(dataset_to_validate)

# the quality assessment for the simulation results
roc_on_valid_seq = roc_auc(y_true=dataset_to_validate.target,
                           y_score=predicted_seq.predict)

roc_on_valid_single = roc_auc(y_true=dataset_to_validate.target,
                              y_score=predicted_single.predict)

roc_on_valid_random_composed = roc_auc(y_true=dataset_to_validate.target,
                                       y_score=predicted_random_composed.predict)

print(f'Composed ROC AUC is {round(roc_on_valid_random_composed, 3)}')
print(f'Static ROC AUC is {round(roc_on_valid_seq, 3)}')
print(f'Single-model ROC AUC is {round(roc_on_valid_single, 3)}')
