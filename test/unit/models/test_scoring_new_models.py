import datetime
import numpy as np
import pytest

# Dataclass for wrapping arrays into it
from fedot.core.data.data import InputData

# Tasks to solve
from fedot.core.repository.tasks import Task, TaskTypesEnum

# Type of the input data
from fedot.core.repository.dataset_types import DataTypesEnum

# Repository with operations in the FEDOT
from fedot.core.repository.operation_types_repository import get_operations_for_task

# Chain of the FEDOT
from fedot.core.chains.chain import Chain

# Evolutionary algorithm classes
from fedot.core.composer.gp_composer.gp_composer import GPComposerBuilder, GPComposerRequirements
from fedot.core.composer.optimisers.gp_comp.gp_optimiser import GPChainOptimiserParameters, GeneticSchemeTypesEnum
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.utilities.synth_dataset_generator import classification_dataset
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score as roc_auc, mean_squared_error

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.data_operation import DataOperation
from fedot.core.operations.model import Model
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.log import default_log
from fedot.core.operations.model import Model
from fedot.core.data.data import InputData, OutputData


def get_roc_auc(valid_data: InputData, predicted_data: OutputData) -> float:
    n_classes = valid_data.num_classes
    if n_classes > 2:
        additional_params = {'multi_class': 'ovo', 'average': 'macro'}
    else:
        additional_params = {}

    try:
        roc_on_train = round(roc_auc(valid_data.target,
                                     predicted_data.predict,
                                     **additional_params), 3)
    except Exception as ex:
        print(ex)
        roc_on_train = 0.5

    return roc_on_train


def test_classification_models_fit_correct():
    # Generate numpy arrays with features and target
    features_options = {'informative': 6, 'redundant': 0,
                        'repeated': 0, 'clusters_per_class': 1}
    x_data, y_data = classification_dataset(samples_amount=2500,
                                            features_amount=6,
                                            classes_amount=2,
                                            features_options=features_options)

    print(f'Features table shape: {x_data.shape}, type: {type(x_data)}')
    print(f'Target vector: {y_data.shape}, type: {type(y_data)}')

    # Define classification task
    task = Task(TaskTypesEnum.classification)

    # Prepare data to train the model
    input_data = InputData(idx=np.arange(0, len(x_data)), features=x_data,
                           target=y_data, task=task,
                           data_type=DataTypesEnum.table)

    # The search of the models provided by the framework that can be used as nodes in a chain for the selected task
    available_model_types = get_operations_for_task(task=task, mode='models')
    print(available_model_types)

    # The choice of the metric for the chain quality assessment during composition
    metric_function = ClassificationMetricsEnum.ROCAUC_penalty

    # The choice and initialisation of the GP search
    max_lead_time = datetime.timedelta(minutes=3)
    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types,
        max_arity=3,
        max_depth=3, pop_size=10,
        num_of_generations=10,
        crossover_prob=0.8,
        mutation_prob=0.8,
        max_lead_time=max_lead_time)

    node_logit = PrimaryNode('logit')
    node_scaling = PrimaryNode('scaling')
    node_xg = SecondaryNode('xgboost', nodes_from=[node_logit, node_scaling])
    chain_xg = Chain(node_xg)

    node_ctb = SecondaryNode('catboost', nodes_from=[node_logit, node_scaling])
    chain_ctb = Chain(node_ctb)

    node_lgbm = SecondaryNode('lgbm', nodes_from=[node_logit, node_scaling])
    chain_lgbm = Chain(node_lgbm)

    xg_metrics = []
    ctb_metrics = []
    lgbm_metrics = []

    for chn, a in zip([chain_xg, chain_ctb, chain_lgbm], [xg_metrics, ctb_metrics, lgbm_metrics]):
        for i in range(3):
            # GP optimiser parameters choice
            scheme_type = GeneticSchemeTypesEnum.parameter_free
            optimiser_parameters = GPChainOptimiserParameters(genetic_scheme_type=scheme_type)

            # Create builder for composer and set composer params
            builder = GPComposerBuilder(task=task).with_requirements(composer_requirements).with_metrics(
                metric_function).with_optimiser_parameters(optimiser_parameters).with_initial_chain(chn)

            # Create GP-based composer
            composer = builder.build()

            # the optimal chain generation by composition - the most time-consuming task
            chain_evo_composed = composer.compose_chain(data=input_data,
                                                        is_visualise=True)
            chain_evo_composed.fine_tune_all_nodes(loss_function=roc_auc,
                                                   loss_params=None,
                                                   input_data=input_data)
            prediction = chain_evo_composed.predict(input_data)
            print(f'ROC AUC score on training sample: {roc_auc(y_data, prediction.predict):.3f}')
            a.append(roc_auc(y_data, prediction.predict))

    print(xg_metrics, ctb_metrics, lgbm_metrics)
