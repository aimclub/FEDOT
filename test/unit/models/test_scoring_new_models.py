import datetime
import numpy as np

from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.composer.gp_composer.gp_composer import GPComposerBuilder, GPComposerRequirements
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.utilities.synth_dataset_generator import classification_dataset
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.data.data import InputData
from sklearn.metrics import roc_auc_score as roc_auc


def test_classification_models_fit_correct():
    features_options = {'informative': 6, 'redundant': 0,
                        'repeated': 0, 'clusters_per_class': 1}
    x_data, y_data = classification_dataset(samples_amount=2500,
                                            features_amount=6,
                                            classes_amount=2,
                                            features_options=features_options)

    task = Task(TaskTypesEnum.classification)
    input_data = InputData(idx=np.arange(0, len(x_data)), features=x_data,
                           target=y_data, task=task,
                           data_type=DataTypesEnum.table)

    available_model_types = get_operations_for_task(task=task, mode='models')

    metric_function = ClassificationMetricsEnum.ROCAUC_penalty
    max_lead_time = datetime.timedelta(minutes=10)
    composer_requirements = GPComposerRequirements(
        primary=available_model_types,
        secondary=available_model_types,
        max_arity=3,
        max_depth=3, pop_size=10,
        num_of_generations=10,
        crossover_prob=0.8,
        mutation_prob=0.8,
        max_lead_time=max_lead_time)

    xg_metrics = []
    ctb_metrics = []
    lgbm_metrics = []

    for chain_type, a in zip(['catboost', 'xgboost', 'lgbm'], [ctb_metrics, xg_metrics, lgbm_metrics]):
        for i in range(10):
            node_logit = PrimaryNode('logit')
            node_scaling = PrimaryNode('scaling')
            node_custom = SecondaryNode(chain_type, nodes_from=[node_logit, node_scaling])
            chain = Chain(node_custom)

            # GP optimiser parameters choice
            scheme_type = GeneticSchemeTypesEnum.parameter_free
            optimiser_parameters = GPGraphOptimiserParameters(genetic_scheme_type=scheme_type)

            # Create builder for composer and set composer params
            builder = GPComposerBuilder(task=task).with_requirements(composer_requirements).with_metrics(
                metric_function).with_optimiser_parameters(optimiser_parameters).with_initial_chain(chain)

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

    # ALL EXPERIMENT COST 300 MINUTES OR 5 HOURS.
    # MORE TAKE TAKE MODELS TUNING THEN COMPOSING.
    # CATBOOST AND LGBM SHOW BETTER RESULT FOR THIS CASE
    # 2500 objects and 6 features in dataset, classification problem, ROC-AUC optimisation metric

    # XGBOOST: [0.9648406970603768, 0.9604855099134275, 0.9603379686674882, 0.985633006065936, 0.9669519455224277,
    # 0.976354562509481, 0.9805950211574883, 0.9836142277148695, 0.9849071675592433, 0.9913533047734444]
    # MEAN: 0.9755073410944182

    # LGBM: [0.9690518430908732, 0.9922380005171253, 0.9748829428013343, 0.9722555591612788, 0.986118818611421,
    # 0.9815135205023775, 0.9783022931760299, 0.9861245792344899, 0.9580716250269629, 0.9687066953161654]
    # MEAN: 0.9767265877438056

    # CATBOOST: [1.0, 0.9847768723147844, 0.9849608733746307, 0.9672662195143027, 0.9799325495045544,
    # 0.9688132668429417, 0.9752040380687576, 0.9636856722423097, 0.9717255818389318, 0.9736928346169922]
    # MEAN: 0.9770057908318204
