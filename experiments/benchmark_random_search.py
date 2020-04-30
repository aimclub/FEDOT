from sklearn.metrics import roc_auc_score as roc_auc

from core.composer.composer import ComposerRequirements
from core.composer.random_composer import RandomSearchComposer, History
from core.models.data import train_test_data_setup
from core.repository.model_types_repository import ModelTypesIdsEnum
from core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from experiments.chain_template import (
    chain_template_balanced_tree, show_chain_template,
    real_chain, fit_template
)
from experiments.composer_benchmark import data_by_synthetic_chain
from experiments.viz import show_history_optimization_comparison


def models_to_use():
    models = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.knn,
              ModelTypesIdsEnum.qda, ModelTypesIdsEnum.dt]
    return models


if __name__ == '__main__':
    runs = 4
    iterations = 5
    history_all = []
    model_types = [ModelTypesIdsEnum.logit, ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.knn]
    for run in range(runs):
        chain = chain_template_balanced_tree(model_types=model_types, depth=4, models_per_level=[8, 4, 2, 1],
                                             samples=1000, features=10)
        show_chain_template(chain)
        fit_template(chain, classes=2, skip_fit=True)
        initial_chain = real_chain(chain)

        random_composer = RandomSearchComposer(iter_num=iterations)
        available_model_types = models_to_use()
        metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)
        req = ComposerRequirements(primary=available_model_types, secondary=available_model_types)

        dataset_to_compose, data_to_validate = train_test_data_setup(data_by_synthetic_chain(with_gaussian=True))

        history_best = History()
        composed_chain = random_composer.compose_chain(data=dataset_to_compose,
                                                       initial_chain=None,
                                                       composer_requirements=req,
                                                       metrics=metric_function,
                                                       history_callback=history_best)
        composed_chain.fit(input_data=dataset_to_compose, verbose=True)
        # ComposerVisualiser.visualise(composed_chain)
        history_all.append(history_best.values)
        predicted_train = composed_chain.predict(dataset_to_compose)
        predicted_test = composed_chain.predict(data_to_validate)
        # the quality assessment for the simulation results
        roc_train = roc_auc(y_true=dataset_to_compose.target,
                            y_score=predicted_train.predict)

        roc_test = roc_auc(y_true=data_to_validate.target,
                           y_score=predicted_test.predict)
        print(f'Train ROC: {roc_train}')
        print(f'Test ROC: {roc_test}')

    show_history_optimization_comparison(first=history_all[:2],
                                         second=history_all[2:],
                                         iterations=iterations,
                                         label_first='first', label_second='second')
