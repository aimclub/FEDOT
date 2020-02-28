import os
import random

import numpy as np

from core.composer.composer import DummyChainTypeEnum
from core.composer.composer import DummyComposer
from core.composer.visualisation import ChainVisualiser
from core.models.data import InputData
from core.repository.dataset_types import NumericalDataTypesEnum, CategoricalDataTypesEnum
from core.repository.model_types_repository import (
    ModelMetaInfoTemplate,
    ModelTypesRepository
)
from core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from core.repository.task_types import MachineLearningTasksEnum
from core.utils import project_root

random.seed(1)
np.random.seed(1)


def dataset():
    file_path = 'test/data/test_dataset.csv'
    full_path = os.path.join(str(project_root()), file_path)
    dataset_from_file = InputData.from_csv(full_path)
    # a dataset that will be used as a train and test set during composition
    dataset_to_compose = dataset_from_file
    # a dataset for a final validation of the composed model
    dataset_to_validate = dataset_from_file

    return dataset_to_compose, dataset_to_validate


def used_models():
    models_repo = ModelTypesRepository()
    available_model_names = models_repo.search_model_types_by_attributes(
        desired_metainfo=ModelMetaInfoTemplate(input_type=NumericalDataTypesEnum.table,
                                               output_type=CategoricalDataTypesEnum.vector,
                                               task_type=MachineLearningTasksEnum.classification))

    models = [models_repo.model_by_id(model_name) for model_name in available_model_names]

    return models


if __name__ == '__main__':
    dataset_to_compose, _ = dataset()
    models = used_models()
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    composer = DummyComposer(DummyChainTypeEnum.hierarchical)

    chain = composer.compose_chain(data=dataset_to_compose,
                                   initial_chain=None,
                                   primary_requirements=[models[1], models[1], models[1]],
                                   secondary_requirements=[models[0]],
                                   metrics=metric_function)

    visualiser = ChainVisualiser().visualise(chain)
