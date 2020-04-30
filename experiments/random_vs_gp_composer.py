import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc

from core.composer.composer import ComposerRequirements
from core.composer.gp_composer.gp_composer import GPComposerRequirements, GPComposer
from core.composer.random_composer import RandomSearchComposer, History
from core.models.data import InputData
from core.models.data import train_test_data_setup
from core.repository.model_types_repository import ModelTypesIdsEnum
from core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository
from core.repository.task_types import MachineLearningTasksEnum
from experiments.chain_template import (chain_template_balanced_tree, fit_template,
                                        show_chain_template, real_chain)
from experiments.composer_benchmark import to_labels
from experiments.generate_data import synthetic_dataset
from experiments.viz import show_history_optimization_comparison


def models_to_use():
    models = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.knn,
              ModelTypesIdsEnum.qda, ModelTypesIdsEnum.dt]
    return models


def source_chain(model_types, samples, features, classes):
    template = chain_template_balanced_tree(model_types=model_types, depth=4, models_per_level=[8, 4, 2, 1],
                                            samples=samples, features=features)
    show_chain_template(template)
    fit_template(template, classes=classes, skip_fit=False)
    initialized_chain = real_chain(template)

    return initialized_chain


def data_generated_by(chain, samples, features_amount, classes):
    task_type = MachineLearningTasksEnum.classification
    features, target = synthetic_dataset(samples_amount=samples,
                                         features_amount=features_amount,
                                         classes_amount=classes)
    target = np.expand_dims(target, axis=1)
    data_test = InputData(idx=np.arange(0, samples),
                          features=features, target=target, task_type=task_type)
    synth_target = chain.predict(input_data=data_test).predict
    synth_labels = to_labels(synth_target)
    data = InputData(idx=np.arange(0, samples),
                     features=features, target=synth_labels, task_type=task_type)

    return data


def _reduced_history_best(history, generations, pop_size):
    reduced = []
    for gen in range(generations):
        fitness_values = [abs(individ[1]) for individ in history[gen: gen + pop_size]]
        best = max(fitness_values)
        print(f'Min in generation #{gen}: {best}')
        reduced.append(best)

    return reduced


def print_roc_score(chain, data_to_compose, data_to_validate):
    predicted_train = chain.predict(data_to_compose)
    predicted_test = chain.predict(data_to_validate)
    # the quality assessment for the simulation results
    roc_train = roc_auc(y_true=data_to_compose.target,
                        y_score=predicted_train.predict)

    roc_test = roc_auc(y_true=data_to_validate.target,
                       y_score=predicted_test.predict)
    print(f'Train ROC: {roc_train}')
    print(f'Test ROC: {roc_test}')


def compare_composers():
    runs = 5
    iterations = 10
    pop_size = 5
    models_in_source_chain = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.knn]
    samples, features_amount, classes = 10000, 10, 2
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    history_random, history_gp = [], []
    for run in range(runs):
        source = source_chain(models_in_source_chain, samples=samples,
                              features=features_amount, classes=classes)
        data_full = data_generated_by(source, samples, features_amount, classes)
        data_to_compose, data_to_validate = train_test_data_setup(data_full)
        available_model_types = models_to_use()

        # Init and run RandomComposer
        print('Running RandomComposer:')
        random_composer = RandomSearchComposer(iter_num=iterations)
        random_reqs = ComposerRequirements(primary=available_model_types, secondary=available_model_types)
        history_best_random = History()
        random_composed = random_composer.compose_chain(data=data_to_compose,
                                                        initial_chain=None,
                                                        composer_requirements=random_reqs,
                                                        metrics=metric_function,
                                                        history_callback=history_best_random)
        history_random.append(history_best_random.values)
        random_composed.fit(input_data=data_to_compose, verbose=True)
        print_roc_score(random_composed, data_to_compose, data_to_validate)

        # Init and run GPComposer
        print('Running GPComposer:')
        gp_requirements = GPComposerRequirements(
            primary=available_model_types,
            secondary=available_model_types, max_arity=2,
            max_depth=4, pop_size=pop_size, num_of_generations=iterations,
            crossover_prob=0.8, mutation_prob=0.4)
        gp_composer = GPComposer()
        gp_composed = gp_composer.compose_chain(data=data_to_compose,
                                                initial_chain=None,
                                                composer_requirements=gp_requirements,
                                                metrics=metric_function, is_visualise=False)
        history_gp.append(gp_composer.history)
        gp_composed.fit(input_data=data_to_compose, verbose=True)
        print_roc_score(gp_composed, data_to_compose, data_to_validate)

    reduced_history_gp = [_reduced_history_best(history, iterations, pop_size) for history in history_gp]

    show_history_optimization_comparison(first=history_random, second=reduced_history_gp,
                                         iterations=iterations,
                                         label_first='Random', label_second='GP')


if __name__ == '__main__':
    compare_composers()
