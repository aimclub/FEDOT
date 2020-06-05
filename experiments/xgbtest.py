import csv
import itertools
import os
from collections import Counter
from copy import deepcopy
from random import seed

import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.metrics import roc_auc_score as roc_auc

from core.chain_validation import validate
from core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from core.composer.visualisation import ComposerVisualiser
from core.models.data import InputData
from core.models.data import InputData, train_test_data_setup
from core.models.data import OutputData
from core.models.data import train_test_data_setup
from core.models.model import Model
from core.repository.dataset_types import NumericalDataTypesEnum, CategoricalDataTypesEnum
from core.repository.model_types_repository import ModelTypesIdsEnum
from core.repository.model_types_repository import (
    ModelTypesIdsEnum, ModelTypesRepository, ModelMetaInfoTemplate)
from core.repository.quality_metrics_repository import (
    MetricsRepository, ClassificationMetricsEnum)
from core.repository.task_types import MachineLearningTasksEnum
from core.repository.task_types import MachineLearningTasksEnum
from experiments.chain_template import (chain_template_balanced_tree, fit_template,
                                        show_chain_template, real_chain, with_calculated_shapes)
from experiments.chain_template import (
    chain_template_balanced_tree, show_chain_template,
    real_chain, fit_template
)
from experiments.composer_benchmark import to_labels, predict_with_xgboost
from experiments.exp_generate_data import synthetic_dataset
from experiments.exp_generate_data import synthetic_dataset, gauss_quantiles
from experiments.viz import fitness_by_generations_boxplots, show_fitness_history_all

np.random.seed(42)
seed(42)


def output_dataset():
    task_type = MachineLearningTasksEnum.classification
    samples = 1000
    x = 10.0 * np.random.rand(samples, ) - 5.0
    x = np.expand_dims(x, axis=1)
    threshold = 0.5
    y = 1.0 / (1.0 + np.exp(np.power(x, -1.0)))
    classes = np.array([0.0 if val <= threshold else 1.0 for val in y])
    classes = np.expand_dims(classes, axis=1)
    data = InputData(idx=np.arange(0, 100), features=x, target=classes,
                     task_type=task_type)

    return data


data_synth_test = output_dataset()

test = InputData(features=data_synth_test.features[0:100], target=data_synth_test.target[0:100],
                 idx=data_synth_test.idx[0:100], task_type=data_synth_test.task_type)

model = Model(model_type=ModelTypesIdsEnum.dt)

fitted_model, predict_train = model.fit(data=test)

np.random.seed(42)
predict_full = model.predict(fitted_model=fitted_model, data=data_synth_test)
np.random.seed(42)
predict_test = model.predict(fitted_model=fitted_model, data=test)

for i in range(len(predict_test)):
    if predict_full[i] - predict_test[i] > 0.0001:
        print(i)
        raise ValueError("!1")
