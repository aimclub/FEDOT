import numpy as np

from core.composer.chain import Chain
from core.composer.composer import DummyComposer, DummyChainTypeEnum, ComposerRequirements
from core.models.data import InputData, train_test_data_setup
from core.repository.model_types_repository import ModelTypesIdsEnum
from core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from test.test_split_train_test import get_synthetic_input_data, get_roc_auc_value


def compose_chain(data: InputData) -> Chain:
    dummy_composer = DummyComposer(DummyChainTypeEnum.hierarchical)
    composer_requirements = ComposerRequirements(primary=[ModelTypesIdsEnum.kmeans, ModelTypesIdsEnum.kmeans],
                                                 secondary=[ModelTypesIdsEnum.logit])

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    chain = dummy_composer.compose_chain(data=data,
                                         initial_chain=None,
                                         composer_requirements=composer_requirements,
                                         metrics=metric_function, is_visualise=False)
    return chain


def test_chain_with_clusters_fit_correct():
    mean_roc_on_test = 0
    for _ in range(15):
        # mean ROC AUC analysed because of stochastic clustering

        data = get_synthetic_input_data(n_samples=10000)

        chain = compose_chain(data=data)
        train_data, test_data = train_test_data_setup(data)

        chain.fit(input_data=train_data)
        _, roc_on_test = get_roc_auc_value(chain, train_data, test_data)
        mean_roc_on_test = np.mean([mean_roc_on_test, roc_on_test])

    roc_threshold = 0.6
    assert mean_roc_on_test > roc_threshold
