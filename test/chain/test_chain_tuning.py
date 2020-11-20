import os
from datetime import timedelta
from random import seed

import pytest
from sklearn.metrics import mean_squared_error as mse, roc_auc_score as roc

from fedot.core.composer.chain import Chain
from fedot.core.composer.chain_tune import Tune
from fedot.core.composer.node import PrimaryNode, SecondaryNode
from fedot.core.models.data import InputData, train_test_data_setup
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.test_chain_import_export import create_four_depth_chain

seed(1)


@pytest.fixture()
def regression_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join('../data', 'advanced_regression.csv')
    return InputData.from_csv(os.path.join(test_file_path, file), task=Task(TaskTypesEnum.regression))


@pytest.fixture()
def classification_dataset():
    test_file_path = str(os.path.dirname(__file__))
    file = os.path.join('../data', 'advanced_classification.csv')
    return InputData.from_csv(os.path.join(test_file_path, file), task=Task(TaskTypesEnum.classification))


def get_regr_chain():
    # Chain composition
    first = PrimaryNode(model_type='xgbreg')
    second = PrimaryNode(model_type='knnreg')
    final = SecondaryNode(model_type='linear',
                          nodes_from=[first, second])
    chain = Chain()
    chain.add_node(final)

    return chain


def get_class_chain():
    # Chain composition
    first = PrimaryNode(model_type='xgboost')
    second = PrimaryNode(model_type='knn')
    final = SecondaryNode(model_type='xgboost',
                          nodes_from=[first, second])

    chain = Chain()
    chain.add_node(final)

    return chain


@pytest.mark.parametrize('data_fixture', ['regression_dataset'])
def test_fine_tune_primary_nodes(data_fixture, request):
    # TODO still stochatic
    result_list = []
    for _ in range(3):
        data = request.getfixturevalue(data_fixture)
        train_data, test_data = train_test_data_setup(data=data)

        # Chain composition
        chain = get_regr_chain()

        # Before tuning prediction
        chain.fit(train_data, use_cache=False)
        before_tuning_predicted = chain.predict(test_data)

        # Chain tuning
        chain.fine_tune_primary_nodes(train_data, max_lead_time=timedelta(minutes=1), iterations=10)

        # After tuning prediction
        chain.fit(train_data)
        after_tuning_predicted = chain.predict(test_data)

        # Metrics
        bfr_tun_mse = mse(y_true=test_data.target, y_pred=before_tuning_predicted.predict)
        aft_tun_mse = mse(y_true=test_data.target, y_pred=after_tuning_predicted.predict)

        print(f'Before tune test {bfr_tun_mse}')
        print(f'After tune test {aft_tun_mse}', '\n')
        result_list.append(aft_tun_mse <= bfr_tun_mse)

    assert any(result_list)


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_fine_tune_all_nodes(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    # Chain composition
    chain = get_class_chain()

    # Before tuning prediction
    chain.fit(train_data, use_cache=False)
    before_tuning_predicted = chain.predict(test_data)

    # root node tuning
    chain.fine_tune_all_nodes(train_data, max_lead_time=timedelta(minutes=1), iterations=30)
    after_tun_root_node_predicted = chain.predict(test_data)

    bfr_tun_roc_auc = round(roc(y_true=test_data.target, y_score=before_tuning_predicted.predict), 2)
    aft_tun_roc_auc = round(roc(y_true=test_data.target, y_score=after_tun_root_node_predicted.predict), 2)

    print(f'Before tune test {bfr_tun_roc_auc}')
    print(f'After tune test {aft_tun_roc_auc}', '\n')

    assert aft_tun_roc_auc >= bfr_tun_roc_auc


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_custom_params_setter(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    chain = get_class_chain()

    custom_params = dict(C=10)

    chain.root_node.custom_params = custom_params
    chain.fit(data)
    params = chain.root_node.cache.actual_cached_state.model.get_params()

    assert params['C'] == 10


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_tune_primary_with_tune_class_correctly(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    chain = get_class_chain()

    chain.fit(train_data, use_cache=False)
    before_tuning_predicted = chain.predict(test_data)

    tuned_chain = Tune(chain=chain,
                       verbose=True).fine_tune_primary_nodes(input_data=train_data,
                                                             max_lead_time=timedelta(minutes=1),
                                                             iterations=30)
    tuned_chain.fit(train_data)
    after_tun_root_node_predicted = tuned_chain.predict(test_data)

    bfr_tun_roc_auc = round(roc(y_true=test_data.target, y_score=before_tuning_predicted.predict), 2)
    aft_tun_roc_auc = round(roc(y_true=test_data.target, y_score=after_tun_root_node_predicted.predict), 2)

    print(f'Before tune test {bfr_tun_roc_auc}')
    print(f'After tune test {aft_tun_roc_auc}', '\n')

    assert aft_tun_roc_auc >= bfr_tun_roc_auc


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_tune_all_with_tune_class_correctly(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    chain = get_class_chain()
    chain.fit(train_data, use_cache=False)
    before_tuning_predicted = chain.predict(test_data)

    tuned_chain = Tune(chain, verbose=True).fine_tune_all_nodes(input_data=train_data,
                                                                max_lead_time=timedelta(minutes=1),
                                                                iterations=30)

    tuned_chain.fit(train_data)
    after_tun_root_node_predicted = tuned_chain.predict(test_data)

    bfr_tun_roc_auc = round(roc(y_true=test_data.target, y_score=before_tuning_predicted.predict), 2)
    aft_tun_roc_auc = round(roc(y_true=test_data.target, y_score=after_tun_root_node_predicted.predict), 2)

    print(f'Before tune test {bfr_tun_roc_auc}')
    print(f'After tune test {aft_tun_roc_auc}', '\n')

    assert aft_tun_roc_auc >= bfr_tun_roc_auc


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_tune_root_with_tune_class_correctly(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    chain = get_class_chain()
    chain.fit(train_data, use_cache=False)
    before_tuning_predicted = chain.predict(test_data)

    tuned_chain = Tune(chain, verbose=True).fine_tune_root_node(input_data=train_data,
                                                                max_lead_time=timedelta(minutes=1),
                                                                iterations=30)

    tuned_chain.fit(train_data)
    after_tun_root_node_predicted = tuned_chain.predict(test_data)

    bfr_tun_roc_auc = round(roc(y_true=test_data.target, y_score=before_tuning_predicted.predict), 2)
    aft_tun_roc_auc = round(roc(y_true=test_data.target, y_score=after_tun_root_node_predicted.predict), 2)

    print(f'Before tune test {bfr_tun_roc_auc}')
    print(f'After tune test {aft_tun_roc_auc}', '\n')

    assert aft_tun_roc_auc >= bfr_tun_roc_auc


@pytest.mark.parametrize('data_fixture', ['classification_dataset'])
def test_tune_certain_node_with_tune_class_correctly(data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    train_data, test_data = train_test_data_setup(data=data)

    chain = create_four_depth_chain()
    chain.fit(train_data, use_cache=False)
    before_tuning_predicted = chain.predict(test_data)

    model_id_to_tune = 4

    tuned_chain = Tune(chain).fine_tune_certain_node(model_id=model_id_to_tune,
                                                     input_data=train_data,
                                                     max_lead_time=timedelta(minutes=1),
                                                     iterations=30)

    tuned_chain.fit(train_data)
    after_tun_root_node_predicted = tuned_chain.predict(test_data)

    bfr_tun_roc_auc = round(roc(y_true=test_data.target, y_score=before_tuning_predicted.predict), 1)
    aft_tun_roc_auc = round(roc(y_true=test_data.target, y_score=after_tun_root_node_predicted.predict), 1)

    print(f'Before tune test {bfr_tun_roc_auc}')
    print(f'After tune test {aft_tun_roc_auc}', '\n')

    assert aft_tun_roc_auc >= bfr_tun_roc_auc
